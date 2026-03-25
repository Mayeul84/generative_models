import torch
import numpy as np

import tqdm
from utils import *

class DDPM:
  def __init__(self, model, num_diffusion_timesteps=1000, beta_start=0.0001, beta_end=0.002, imgshape=(1,3,256,256), device="cpu"):
    self.num_diffusion_timesteps = num_diffusion_timesteps
    self.reversed_time_steps = np.arange(self.num_diffusion_timesteps)[::-1]
    self.betas = np.linspace(beta_start, beta_end, self.num_diffusion_timesteps,
                              dtype=np.float64)
    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
    self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
    self.sigma = np.sqrt(1 - self.alphas_cumprod_prev )
    self.model = model
    self.imgshape = imgshape
    self.device = device


  def get_eps_from_model(self, x, t):
    model_output = self.model(x, torch.tensor(t, device=self.device).unsqueeze(0))
    model_output = model_output[:,:3,:,:]
    return(model_output)

  def predict_xstart_from_eps(self, x, eps, t):
    x_start = (
        np.sqrt(1.0 / self.alphas_cumprod[t])* x
        - np.sqrt(1.0 / self.alphas_cumprod[t] - 1) * eps
    )
    x_start = x_start.clamp(-1.,1.)
    return(x_start)

  def sampling_spliting_z(self, t_start, u_0, x_true, y, iteration, show_steps):
        with torch.no_grad():  # mode eval
            xt = u_0 
            xhat = torch.randn(self.imgshape, device=self.device)
            # t_start = min(t_start, 100)
            if iteration < 20 : 
                t_end = self.num_diffusion_timesteps - (t_start //2)
            else : 
                t_end = self.num_diffusion_timesteps

            diff_iter = self.reversed_time_steps[self.num_diffusion_timesteps - t_start:t_end]

            for i, t in enumerate(diff_iter):
                if t > 1:
                    z = torch.randn(self.imgshape, device=self.device)
                else:
                    z = torch.zeros(self.imgshape, device=self.device)

                alpha_t = self.alphas[t]
                alpha_bar_t = self.alphas_cumprod[t]
                sigma_t = np.sqrt(((1 - self.alphas_cumprod[t - 1]) / (1 - self.alphas_cumprod[t])) * self.betas[t])
                eps = self.get_eps_from_model(xt, t)

                xt = (1 / np.sqrt(alpha_t)) * (xt -  ((1 - alpha_t) / np.sqrt(1 - alpha_bar_t)) * eps ) + sigma_t * z
                xhat = self.predict_xstart_from_eps(xt, eps, t)

                
            if show_steps :
                    y = y.to(self.device)
                    x_true = x_true.to(self.device)
                    xt = xt.to(self.device)
                    xhat = xhat.to(self.device)
                    pilimg = display_as_pilimg(torch.cat(( y, x_true, xt, xhat), dim=3))

        return xt
    
  def posterior_sampling(self, linear_operator, y, x_true=None, show_steps=True, vis_y=None):

    # visualization image for the observation y:
    if vis_y==None:
      vis_y = y

    # initialize xt for t=T
    x = torch.randn(self.imgshape,device=self.device)
    x.requires_grad = True


    for t in tqdm(self.reversed_time_steps[1:]):
      alpha_t = self.alphas[t]
      alpha_bar_t = self.alphas_cumprod[t]
      alpha_bar_tm1 = self.alphas_cumprod_prev[t]

      beta_t = self.betas[t]
      sigma_t = np.sqrt(beta_t)

      z = torch.randn(self.imgshape, device=self.device)

      xhat = self.predict_xstart_from_eps(x, self.get_eps_from_model(x,t), t)

      x_prime = np.sqrt(alpha_t) * (1-alpha_bar_tm1) / (1-alpha_bar_t) * x
      x_prime += np.sqrt(alpha_bar_tm1)*beta_t / (1-alpha_bar_t)*xhat
      x_prime += sigma_t*z

      df_term = torch.sum((y-linear_operator(xhat))**2)
      grad = torch.autograd.grad(df_term, x)[0]
      zeta = 1 / torch.sqrt(df_term)

      x = x_prime - zeta * grad

      if show_steps and (t)%100==0:
        print('Iteration :', t)
        y = y.to(self.device)
        x_true = x_true.to(self.device)
        x = x.to(self.device)
        xhat = xhat.to(self.device)
        pilimg = display_as_pilimg(torch.cat(( y, x_true, x, xhat), dim=3))

    return(x)
  

from diffusers import LDMPipeline
import torch
import numpy as np
from tqdm import tqdm

class LDM:
    def __init__(self, repo_id="CompVis/ldm-celebahq-256", guidance_scale=1.0,
                 imgshape=(1, 3, 256, 256), device="cpu"):
        pipe = LDMPipeline.from_pretrained(repo_id)
        self.vae = pipe.vqvae
        self.unet = pipe.unet
        self.device = "cpu"
        self.to(device)

        self.scheduler = pipe.scheduler
        self.vae.eval()
        self.unet.eval()

        self.guidance_scale = guidance_scale
        self.imgshape = imgshape

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
                y = y.to(self.device)
                x_true = x_true.to(self.device)
                xt = xt.to(self.device)
                xhat = xhat.to(self.device)
                pilimg = display_as_pilimg(torch.cat(( y, x_true, xt, xhat), dim=3))
                
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
                y = y.to(self.device)
                x_true = x_true.to(self.device)
                x = x.to(self.device)
                xhat = xhat.to(self.device)
                pilimg = display_as_pilimg(torch.cat(( y, x_true, x, xhat), dim=3))

        return self.decode(l)