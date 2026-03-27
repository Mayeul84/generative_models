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

        mask = torch.ones((1, height, width), dtype=torch.float32, device=self.device)
    
        if center is None:
            center_y, center_x = height // 2, width // 2
        else:
            center_y, center_x = center

        half_size = square_size // 2
        y1 = max(center_y - half_size, 0)
        y2 = min(center_y + half_size, height)
        x1 = max(center_x - half_size, 0)
        x2 = min(center_x + half_size, width)

        mask[:, y1:y2, x1:x2] = 0.0

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
        A = self.H.T @ y_flat / sigma2 
        B = z_flat / p2
        summ = sum_chunk(A, B).cpu()
        moy = cov.dot(summ)
    
        return sample_from_sparse_gaussian(moy, cov).view(z.shape).to(self.device)
    
    def linear_operator(self,x):
        x = x.to(self.device)
        return x*self.mask