import torch
import numpy as np
from scipy.sparse.linalg import splu
import scipy.sparse as sp

def sum_chunk(A, B, device="cpu"):
    A = torch.tensor(A).cpu().squeeze().numpy()
    B = torch.tensor(B).cpu().squeeze().numpy()
    sum_result = np.zeros_like(A)
    chunk_size = 10000  
    size = A.size

    # Perform addition in chunks
    for i in range(0, size, chunk_size):
        chunk_A = A[i:i+chunk_size]
        chunk_B = B[i:i+chunk_size]
        sum_result[i:i+chunk_size] = chunk_A + chunk_B
    
    return torch.tensor(sum_result).to(device)

### To sample from N(mu, LtL)
def sparse_cholesky(A):
    n = A.shape[0]
    LU = splu(A.tocsc(), diag_pivot_thresh=0)
    if (LU.perm_r == np.arange(n)).all() and (LU.U.diagonal() > 0).all():
        return LU.L @ sp.diags(LU.U.diagonal()**0.5)
    else:
        raise ValueError('The matrix is not positive definite')

### Sample from N(mu,LtL)
def sample_from_sparse_gaussian(mu, cov_matrix, device="cpu"):
    dim = len(mu)
    z = np.random.normal(size=(dim,))
    L = sparse_cholesky(cov_matrix)
    derive = L.dot(z)
    x = mu + derive
    return torch.tensor(x).to(device)

### DEFINING CLASS OF OPERATORS

class Inpainting():

    def __init__(self,mask=0.5, imgshape=None, device="cpu"):

        self.device = device
        # If mask is a float, then it corresponds to the amount of known pixels for a random mask.
        if isinstance(mask,float):
            height, width = imgshape
            mask = self.build_random_mask(imgshape, N=int(height*width*mask))

        if isinstance(mask,tuple):
            center, square_size = mask
            mask = self.build_square_mask(imgshape=imgshape,square_size=square_size,center=center)

        self.set_mask(mask=mask)
    
    def set_mask(self,mask):
        self.mask = mask
        self.H = self.build_H(mask=self.mask)
        self.HtH = (self.H).dot(self.H.T)

    def build_random_mask(self,imgshape, N):
        """
        Create a binary matrix representing a subsampling pattern for image inpainting.

        Parameters:
            N (int): Total number of pixels in the original, full-resolution image.
            M (int): Number of observed or known pixel values in the subsampled image.

        Returns:
            torch.Tensor: Binary matrix representing the subsampling pattern.
        """

        height, width = imgshape
        vol = height*width

        # Initialize the mask as a flattened vector of zeros
        mask = torch.zeros(vol, dtype=torch.float32, device=self.device)
        
        # Randomly choose M indices to be observed (set to 1)
        observed_indices = torch.randperm(vol)[:N]
        mask[observed_indices] = 1
        
        # Reshape the mask to a 2D image if necessary  
        mask = mask.reshape(height, width)
        
        mask = mask.repeat(1, 3, 1, 1)

        return mask
    
    def build_square_mask(self,imgshape, square_size, center=None):

        height,width = imgshape

        mask = torch.ones((1, 3, height, width), dtype=torch.float32, device=self.device)
    
        if center is None:
            center_y, center_x = height // 2, width // 2
        else:
            center_y, center_x = center

        half_size = square_size // 2
        y1 = max(center_y - half_size, 0)
        y2 = min(center_y + half_size, height)
        x1 = max(center_x - half_size, 0)
        x2 = min(center_x + half_size, width)

        mask[:, :, y1:y2, x1:x2] = 0.0

        return mask

    def build_H(self,mask):
        # Get indices of non-zero elements from the mask matrix
        nonzero_indices = torch.nonzero(mask.flatten(), as_tuple=False)[:, 0].to(self.device)

        # Define the number of measurements 
        M = len(nonzero_indices)
        nonzero_indices = torch.stack([nonzero_indices, torch.arange(0, M, device=self.device)]).cpu()

        # Create a sparse tensor from the non-zero indices
        values = np.ones(M)  # All elements are ones for binary matrix
        H = sp.coo_matrix((values, (nonzero_indices[0], nonzero_indices[1])), shape=(len(mask.flatten()), M)).tocsc()

        return H
    
    ### MAIN FUNCTION (sampling ~ p(x|z,y))
    def sample_x_given_z_y(self, z, p2, y_flat, sigma2):
        z_flat = z.flatten()

        eye_sparse = sp.eye(self.HtH.shape[0])

        cov = p2 * (eye_sparse - p2 * self.HtH / (p2 + sigma2))
        A = y_flat / sigma2 
        B = z_flat / p2
        summ = sum_chunk(A, B).cpu()
        moy = cov.dot(summ)
    
        return sample_from_sparse_gaussian(moy, cov).view(z.shape).to(self.device)
    
    def linear_operator(self,x):
        x = x.to(self.device)
        return x*self.mask
    



class Deblurring():

    def __init__(self, kernel, imgshape, device="cpu"):
        self.device = device
        self.imgshape = imgshape
        H, W = imgshape

        kernel = kernel.squeeze().double()
        kH, kW = kernel.shape

        kernel_padded = torch.zeros(H, W, dtype=torch.float64)
        kernel_padded[:kH, :kW] = kernel
        kernel_padded = torch.roll(kernel_padded, shifts=(-kH // 2, -kW // 2), dims=(0, 1))

        H_fft_2d = torch.fft.rfft2(kernel_padded)
        self.H_fft   = H_fft_2d.unsqueeze(0).repeat(3, 1, 1)
        self.Hc_fft  = self.H_fft.conj()
        self.HtH_fft = (self.H_fft * self.Hc_fft).real

        # Pour compatibilité avec algo.py qui fait operator.HtH.dot(vecteur_numpy)
        self.HtH = self

    def dot(self, x_numpy):
        H, W = self.imgshape
        x = torch.tensor(x_numpy).reshape(1, 3, H, W).float().to(self.device)
        X_fft = torch.fft.rfft2(x.double())
        Y_fft = X_fft * self.HtH_fft.unsqueeze(0).to(self.device)
        result = torch.fft.irfft2(Y_fft, s=self.imgshape).float()
        return result.flatten().cpu().numpy()

    def linear_operator(self, x):
        x = x.to(self.device)
        squeeze = (x.dim() == 3)
        if squeeze:
            x = x.unsqueeze(0)
        X_fft = torch.fft.rfft2(x.double())
        Y_fft = X_fft * self.H_fft.unsqueeze(0).to(self.device)
        y = torch.fft.irfft2(Y_fft, s=self.imgshape).float()
        return y.squeeze(0) if squeeze else y

    def sample_x_given_z_y(self, z, p2, y_flat, sigma2):
        z   = z.to(self.device)
        y   = torch.tensor(y_flat).reshape(1, 3, *self.imgshape).float().to(self.device) \
              if not isinstance(y_flat, torch.Tensor) else y_flat.reshape(1, 3, *self.imgshape).to(self.device)

        Z_fft = torch.fft.rfft2(z.double().squeeze(0))
        Y_fft = torch.fft.rfft2(y.double().squeeze(0))

        HtH = self.HtH_fft.to(self.device)
        Hc  = self.Hc_fft.to(self.device)

        sigma_post_fft = (p2 * sigma2) / (sigma2 + p2 * HtH)
        HtY_fft        = Hc * Y_fft
        mu_post_fft    = sigma_post_fft * (HtY_fft / sigma2 + Z_fft / p2)

        std_fft = sigma_post_fft.sqrt()
        noise   = torch.randn_like(mu_post_fft) + 1j * torch.randn_like(mu_post_fft)
        x_fft   = mu_post_fft + std_fft * noise

        x = torch.fft.irfft2(x_fft, s=self.imgshape).float()
        return x.unsqueeze(0).to(self.device)



class SuperResolution():

    def __init__(self, scale_factor, imgshape, device="cpu"):
        """
        scale_factor : int — facteur de sous-échantillonnage (ex: 4 → image 4x plus petite)
        imgshape     : (H, W) — taille de l'image haute résolution
        """
        self.device = device
        self.imgshape = imgshape
        self.scale_factor = scale_factor
        H, W = imgshape
        self.lr_shape = (H // scale_factor, W // scale_factor)

        # Pour compatibilité avec algo.py qui fait operator.HtH.dot(vecteur_numpy)
        self.HtH = self

    def dot(self, x_numpy):
        H, W = self.imgshape
        x = torch.tensor(x_numpy).reshape(1, 3, H, W).float().to(self.device)
        # HtH = H^T H = upscale(downscale(x)) via avg_pool + upsample
        result = self._HtH(x)
        return result.flatten().cpu().numpy()

    def _downsample(self, x):
        return torch.nn.functional.avg_pool2d(x, self.scale_factor, self.scale_factor)

    def _upsample(self, x):
        return torch.nn.functional.interpolate(x, size=self.imgshape, mode='nearest')

    def _HtH(self, x):
        return self._upsample(self._downsample(x))

    def linear_operator(self, x):
        x = x.to(self.device)
        squeeze = (x.dim() == 3)
        if squeeze:
            x = x.unsqueeze(0)
        y = self._downsample(x)
        return y.squeeze(0) if squeeze else y

    def sample_x_given_z_y(self, z, p2, y_flat, sigma2):
        z = z.to(self.device)

        # Reconstruire y en haute résolution
        if not isinstance(y_flat, torch.Tensor):
            y_flat = torch.tensor(y_flat)
        y_flat = y_flat.reshape(1, 3, *self.imgshape).float().to(self.device)

        # Posterior : mean = (1/p2 * I + 1/sigma2 * HtH)^{-1} * (z/p2 + H^T y / sigma2)
        # On résout dans l'espace image via Fourier (HtH est diagonal en Fourier pour avg_pool)
        # Approximation : on utilise la formule de Woodbury
        # Sigma_post = p2 * I - p2^2 / (sigma2 + p2) * HtH  (car HtH est une projection : HtH^2 = HtH)
        
        s2 = torch.tensor(sigma2, dtype=torch.float32)
        p2t = torch.tensor(p2, dtype=torch.float32)

        HtH_z = self._HtH(z)
        Ht_y  = self._upsample(
            y_flat.reshape(1, 3, *self.imgshape) if y_flat.shape[-2:] == self.imgshape
            else torch.nn.functional.avg_pool2d(y_flat, self.scale_factor, self.scale_factor)
            # y_flat est déjà en LR ou HR ?
        ) if y_flat.shape[-2:] != self.imgshape else y_flat

        # Si y_flat est en HR (upsampled depuis algo.py via HtH.dot)
        # H^T y_lr = upsample(y_lr) / scale_factor^2
        # On recalcule proprement depuis y_flat HR
        y_lr = self._downsample(y_flat)
        Ht_y = self._upsample(y_lr) / (self.scale_factor ** 2)

        mu = (z / p2t + Ht_y / s2) 

        # Covariance : pixel dans le support de H → variance sigma2*p2/(sigma2+p2*s^2)
        # pixel hors support → variance p2
        alpha = p2t * s2 / (s2 + p2t * self.scale_factor**2)

        # Partie projetée (dans l'image de H^T H)
        mu_proj    = self._HtH(mu) * alpha
        mu_unproj  = (mu - self._HtH(mu)) * p2t  

        moy = mu_proj + mu_unproj

        # Bruit
        noise = torch.randn_like(z)
        noise_proj   = self._HtH(noise) * alpha.sqrt()
        noise_unproj = (noise - self._HtH(noise)) * p2t.sqrt()

        x = moy + noise_proj + noise_unproj
        return x.to(self.device)
