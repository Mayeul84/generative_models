import torch
from skimage.restoration import estimate_sigma
from diffusers import LDMPipeline
from tqdm import tqdm
import numpy as np

def inverse_variance_function(noise_level, model):
    closest_t_index = np.argmin(np.abs( (1-model.alphas_cumprod) - noise_level**2))
    return closest_t_index

def PNP_SGS(ro, MCMC_steps, x_true, y, Burn_in_steps, diffusing_model, operator, show_only_last=False):
    N = 256  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = torch.tensor(y).to(device)  # Observed measurements
    z = torch.randn(y.shape).to(device)  # initialize z
    ro = ro

    noise_level = estimate_sigma(y[0].cpu().numpy(), channel_axis=0, average_sigmas=True)
    x_flatten = x_true.flatten()  
    y_flatten = x_flatten + noise_level * torch.randn_like(x_flatten)
    y_flatten = operator.HtH.dot(y_flatten.cpu().numpy())

    N_burn_in = Burn_in_steps
    x_samples = []
    time = []

    show = not show_only_last

    for n in tqdm(range(MCMC_steps)):  # MCMC
        if n == MCMC_steps - 1:
            show = True

        if show:
            print(f"---------------- Iteration {n} ------------")
        
        # Step 1: sample from x given z and y  : equation 6
        x = operator.sample_x_given_z_y(z, ro**2, y_flatten, noise_level**2, device=device).float()

        # Step 2: estimating noise level
        noise_level = estimate_sigma(x[0].cpu().numpy(), channel_axis=0, average_sigmas=True)
        
        # Step 3: find t
        t_star = inverse_variance_function(noise_level,model=diffusing_model)
        time.append(t_star)

        if show:
            print(f"noise level estimated = {noise_level}")
            print(f"number of noising steps = {t_star}")
        
        # Step 3 : Sample z via reverse diffusion : equation 7
        z = diffusing_model.sampling_spliting_z(t_star, x, x_true, y, n, show_steps=show)

        if n > N_burn_in:
            x_samples.append(z)

    return x_samples, time

class LDM:
    def __init__(self, repo_id="CompVis/ldm-celebahq-256", guidance_scale=1.0,
                 imgshape=(1, 3, 256, 256)):
        pipe = LDMPipeline.from_pretrained(repo_id)
        self.vae = pipe.vqvae
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler
        self.vae.eval()
        self.unet.eval()

        self.guidance_scale = guidance_scale
        self.imgshape = imgshape
        self.device = "cpu"

        # Reproduction de la même interface que DDPM
        self.num_diffusion_timesteps = self.scheduler.config.num_train_timesteps
        self.reversed_time_steps = np.arange(self.num_diffusion_timesteps)[::-1]

        alphas_cumprod = self.scheduler.alphas_cumprod.numpy()
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.alphas = self.scheduler.alphas.numpy()
        self.betas = self.scheduler.betas.numpy()

        # shape de l'espace latent : (1, C_lat, H/f, W/f)
        # Pour LDM-256 CelebA-HQ : f=4, C_lat=3 → latent (1,3,64,64)
        b, c, h, w = imgshape
        ds = getattr(pipe.vqvae.config, "down_block_types", None)
        f = 2 ** (len(ds) - 1) if ds else 4
        c_lat = getattr(pipe.vqvae.config, "latent_channels",
                        getattr(pipe.vqvae.config, "vq_embed_dim", 3))
        self.latent_shape = (b, c_lat, h // f, w // f)

    def to(self, device):
        self.vae = self.vae.to(device)
        self.unet = self.unet.to(device)
        self.device = device
        return self

    # ------------------------------------------------------------------ #
    #  Helpers (même interface que DDPM)
    # ------------------------------------------------------------------ #
    def encode(self, x):
        """Pixel → latent  (sans gradient)"""
        with torch.no_grad():
            return self.vae.encode(x).latents

    def decode(self, l):
        """Latent → pixel  (sans gradient par défaut)"""
        with torch.no_grad():
            return self.vae.decode(l).sample

    def get_eps_from_model(self, l, t):
        t_tensor = torch.tensor([t], device=self.device)
        return self.unet(l, t_tensor).sample

    def predict_xstart_from_eps(self, l, eps, t):
        """Tweedie : estimation de ℓ_0 dans l'espace latent"""
        alpha_bar = self.alphas_cumprod[t]
        l0 = (l - np.sqrt(1.0 - alpha_bar) * eps) / np.sqrt(alpha_bar)
        return l0.clamp(-1., 1.)

    # ------------------------------------------------------------------ #
    #  Équivalent sampling_splitting_z  →  étape z du SGS  (eq. 7)
    # ------------------------------------------------------------------ #
    def sampling_splitting_z(self, t_start, u_0, x_true, y,
                             iteration, show_steps):
        """
        Débruitage stochastique dans l'espace latent.
        u_0 : image pixel (1,3,H,W) – observation bruitée courante x^(n)
        Retourne une image pixel débruitée.
        """
        with torch.no_grad():
            # Encodage dans l'espace latent
            lt = self.encode(u_0)

            # Curriculum burn-in : early-stopping pendant les 20 premières iter
            if iteration < 20:
                t_end = self.num_diffusion_timesteps - (t_start // 2)
            else:
                t_end = self.num_diffusion_timesteps

            diff_iter = self.reversed_time_steps[
                self.num_diffusion_timesteps - t_start : t_end
            ]

            for i, t in enumerate(diff_iter):
                z = (torch.randn_like(lt) if t > 1
                     else torch.zeros_like(lt))

                alpha_t      = self.alphas[t]
                alpha_bar_t  = self.alphas_cumprod[t]
                alpha_bar_tm1 = self.alphas_cumprod[t - 1] if t > 0 else 1.0
                sigma_t = np.sqrt(
                    ((1 - alpha_bar_tm1) / (1 - alpha_bar_t)) * self.betas[t]
                )

                eps = self.get_eps_from_model(lt, t)

                # Pas DDPM dans l'espace latent
                lt = ((1 / np.sqrt(alpha_t))
                      * (lt - ((1 - alpha_t) / np.sqrt(1 - alpha_bar_t)) * eps)
                      + sigma_t * z)

            # Décodage vers l'espace pixel
            xt = self.decode(lt)
            lhat = self.predict_xstart_from_eps(lt, eps, diff_iter[-1])
            xhat = self.decode(lhat)

            if show_steps:
                pilimg = display_as_pilimg(
                    torch.cat((y, x_true, xt, xhat), dim=3)
                )

        return xt

    # ------------------------------------------------------------------ #
    #  Équivalent posterior_sampling  →  étape x du SGS  (eq. 6, DPS)
    # ------------------------------------------------------------------ #
    def posterior_sampling(self, linear_operator, y,
                           x_true=None, show_steps=True, vis_y=None):
        """
        Echantillonnage guidé par gradient dans l'espace latent (DPS-LDM).
        linear_operator opère dans l'espace pixel.
        """
        if vis_y is None:
            vis_y = y

        # Initialisation : bruit pur dans l'espace latent
        l = torch.randn(self.latent_shape, device=self.device)
        l.requires_grad_(True)

        for t in tqdm(self.reversed_time_steps[1:]):
            alpha_t       = self.alphas[t]
            alpha_bar_t   = self.alphas_cumprod[t]
            alpha_bar_tm1 = self.alphas_cumprod_prev[t]
            beta_t        = self.betas[t]
            sigma_t       = np.sqrt(beta_t)

            z = torch.randn(self.latent_shape, device=self.device)

            # Prédiction du bruit et de ℓ_0
            eps  = self.get_eps_from_model(l, t)
            lhat = self.predict_xstart_from_eps(l, eps, t)

            # Pas DDPM dans l'espace latent (posterior mean + bruit)
            l_prime = (np.sqrt(alpha_t) * (1 - alpha_bar_tm1) / (1 - alpha_bar_t) * l
                       + np.sqrt(alpha_bar_tm1) * beta_t / (1 - alpha_bar_t) * lhat
                       + sigma_t * z)

            # Guidance : résidu dans l'espace pixel via décodage de ℓ̂_0
            xhat = self.vae.decode(lhat).sample          # grad activé sur lhat
            df_term = torch.sum((y - linear_operator(xhat)) ** 2)
            grad = torch.autograd.grad(df_term, l)[0]
            zeta = 1.0 / torch.sqrt(df_term)

            # Mise à jour corrigée
            with torch.no_grad():
                l = l_prime - zeta * grad
            l = l.detach().requires_grad_(True)

            if show_steps and t % 100 == 0:
                print(f"Iteration : {t}")
                x = self.decode(l)
                pilimg = display_as_pilimg(
                    torch.cat((y, x_true, x, xhat), dim=3)
                )

        return self.decode(l)


ldm = LDM()