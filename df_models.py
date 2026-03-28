import torch
import numpy as np

import tqdm
from utils import *

# class DDPM:
#   def __init__(self, model, num_diffusion_timesteps=1000, beta_start=0.0001, beta_end=0.02, imgshape=(1,3,256,256), device="cpu"):
#     self.num_diffusion_timesteps = num_diffusion_timesteps
#     self.reversed_time_steps = np.arange(self.num_diffusion_timesteps)[::-1]
#     self.betas = np.linspace(beta_start, beta_end, self.num_diffusion_timesteps,
#                               dtype=np.float64)
#     self.alphas = 1.0 - self.betas
#     self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
#     self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
#     self.sigma = np.sqrt(1 - self.alphas_cumprod_prev )
#     self.model = model
#     self.imgshape = imgshape
#     self.device = device


#   def get_eps_from_model(self, x, t):
#     model_output = self.model(x, torch.tensor(t, device=self.device).unsqueeze(0))
#     model_output = model_output[:,:3,:,:]
#     return(model_output)

#   def predict_xstart_from_eps(self, x, eps, t):
#     x_start = (
#         np.sqrt(1.0 / self.alphas_cumprod[t])* x
#         - np.sqrt(1.0 / self.alphas_cumprod[t] - 1) * eps
#     )
#     x_start = x_start.clamp(-1.,1.)
#     return(x_start)

#   def sampling_splitting_z(self, t_start, u_0, x_true, y, iteration, show_steps, t_end=None):
#         with torch.no_grad():  # mode eval
#             xt = u_0 
#             xhat = torch.randn(self.imgshape, device=self.device)
#             # t_start = min(t_start, 100)

#             if t_end is None:
#                 if iteration < 20 : 
#                     t_end = self.num_diffusion_timesteps - (t_start //2)
#                 else : 
#                     t_end = self.num_diffusion_timesteps
#             else:
#                 t_end = self.num_diffusion_timesteps - t_end

#             diff_iter = self.reversed_time_steps[self.num_diffusion_timesteps - t_start:t_end]

#             for i, t in enumerate(diff_iter):
#                 if t > 1:
#                     z = torch.randn(self.imgshape, device=self.device)
#                 else:
#                     z = torch.zeros(self.imgshape, device=self.device)

#                 alpha_t = self.alphas[t]
#                 alpha_bar_t = self.alphas_cumprod[t]
#                 sigma_t = np.sqrt(((1 - self.alphas_cumprod[t - 1]) / (1 - self.alphas_cumprod[t])) * self.betas[t])
#                 eps = self.get_eps_from_model(xt, t)

#                 xt = (1 / np.sqrt(alpha_t)) * (xt -  ((1 - alpha_t) / np.sqrt(1 - alpha_bar_t)) * eps ) + sigma_t * z
#                 xhat = self.predict_xstart_from_eps(xt, eps, t)
                
#             if show_steps :
#                     y = y.to(self.device)
#                     x_true = x_true.to(self.device)
#                     xt = xt.to(self.device)
#                     xhat = xhat.to(self.device)
#                     pilimg = display_as_pilimg(torch.cat(( y, x_true, xt, xhat), dim=3))

#         return xt
    
#   def posterior_sampling(self, linear_operator, y, x_true=None, show_steps=True, vis_y=None):

#     # visualization image for the observation y:
#     if vis_y==None:
#       vis_y = y

#     # initialize xt for t=T
#     x = torch.randn(self.imgshape,device=self.device)
#     x.requires_grad = True


#     for t in tqdm(self.reversed_time_steps[1:]):
#       alpha_t = self.alphas[t]
#       alpha_bar_t = self.alphas_cumprod[t]
#       alpha_bar_tm1 = self.alphas_cumprod_prev[t]

#       beta_t = self.betas[t]
#       sigma_t = np.sqrt(beta_t)

#       z = torch.randn(self.imgshape, device=self.device)

#       xhat = self.predict_xstart_from_eps(x, self.get_eps_from_model(x,t), t)

#       x_prime = np.sqrt(alpha_t) * (1-alpha_bar_tm1) / (1-alpha_bar_t) * x
#       x_prime += np.sqrt(alpha_bar_tm1)*beta_t / (1-alpha_bar_t)*xhat
#       x_prime += sigma_t*z

#       df_term = torch.sum((y-linear_operator(xhat))**2)
#       grad = torch.autograd.grad(df_term, x)[0]
#       zeta = 1 / torch.sqrt(df_term)

#       x = x_prime - zeta * grad

#       if show_steps and (t)%100==0:
#         print('Iteration :', t)
#         y = y.to(self.device)
#         x_true = x_true.to(self.device)
#         x = x.to(self.device)
#         xhat = xhat.to(self.device)
#         pilimg = display_as_pilimg(torch.cat(( y, x_true, x, xhat), dim=3))

#     return(x)
  


class DDPM:
  
  def __init__(self, model, num_diffusion_timesteps=1000, beta_start=0.0001, beta_end=0.02, 
               imgshape=(1,3,256,256), device="cpu", base_timesteps=1000):
    self.num_diffusion_timesteps = num_diffusion_timesteps
    self.base_timesteps = base_timesteps
    self.reversed_time_steps = np.arange(self.num_diffusion_timesteps)[::-1]

    # Grille de base (celle sur laquelle le modèle a été entraîné)
    self.base_betas = np.linspace(beta_start, beta_end, base_timesteps, dtype=np.float64)
    self.base_alphas = 1.0 - self.base_betas
    self.base_alphas_cumprod = np.cumprod(self.base_alphas, axis=0)

    # Nouvelle grille interpolée
    # t normalisé entre 0 et 1 pour les deux grilles
    base_t = np.linspace(0, 1, base_timesteps)
    new_t  = np.linspace(0, 1, num_diffusion_timesteps)

    # Interpolation de alphas_cumprod sur la nouvelle grille
    alphas_cumprod_interp = np.interp(new_t, base_t, self.base_alphas_cumprod)

    self.alphas_cumprod = alphas_cumprod_interp
    self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

    # On en déduit betas et alphas sur la nouvelle grille
    self.alphas = self.alphas_cumprod / self.alphas_cumprod_prev
    self.betas  = 1.0 - self.alphas

    self.sigma = np.sqrt(1 - self.alphas_cumprod_prev)
    self.model = model
    self.imgshape = imgshape
    self.device = device

  def get_eps_from_model(self, x, t):
    # Remapping : t dans [0, num_diffusion_timesteps-1] -> t_base dans [0, base_timesteps-1]
    t_base = round(t / (self.num_diffusion_timesteps - 1) * (self.base_timesteps - 1))
    model_output = self.model(x, torch.tensor(t_base, device=self.device).unsqueeze(0))
    model_output = model_output[:,:3,:,:]
    return model_output
  
  def predict_xstart_from_eps(self, x, eps, t):
    x_start = (
        np.sqrt(1.0 / self.alphas_cumprod[t])* x
        - np.sqrt(1.0 / self.alphas_cumprod[t] - 1) * eps
    )
    x_start = x_start.clamp(-1.,1.)
    return(x_start)

  def sampling_splitting_z(self, t_start, u_0, x_true, y, iteration, show_steps, t_end=None, N_burn_in=20):
        with torch.no_grad():  # mode eval
            xt = u_0 
            xhat = torch.randn(self.imgshape, device=self.device)
            # t_start = min(t_start, 100)

            if t_end is None:
                if iteration < N_burn_in : 
                    t_end = self.num_diffusion_timesteps - 60
                else : 
                    t_end = self.num_diffusion_timesteps
            else:
                if iteration < N_burn_in : 
                    t_end = self.num_diffusion_timesteps - 60
                else:
                    t_end = self.num_diffusion_timesteps - t_end

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
    def __init__(self,num_diffusion_timesteps=1000, beta_start=0.0001, beta_end=0.002, guidance_scale=1.0,
                 imgshape=(1, 3, 256, 256), device="cpu", repo_id="CompVis/ldm-celebahq-256"):
        pipe = LDMPipeline.from_pretrained(repo_id)
        self.vae = pipe.vqvae
        self.unet = pipe.unet
        self.device = "cpu"
        self.to(device)

        self.scheduler = pipe.scheduler
        self.scheduler.set_timesteps(num_diffusion_timesteps)
        self.vae.eval()
        self.unet.eval()

        self.guidance_scale = guidance_scale
        self.imgshape = imgshape

        # Reproduction de la même interface que DDPM
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.reversed_time_steps = np.arange(self.num_diffusion_timesteps)[::-1]
        self.betas = np.linspace(beta_start, beta_end, self.num_diffusion_timesteps,
                                  dtype=np.float64)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sigma = np.sqrt(1 - self.alphas_cumprod_prev )
        
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

    def sampling_splitting_z(self, t_start, u_0, x_true, y, iteration, show_steps, t_end=None, N_burn_in=20):
        u_0 = u_0.to(self.device)

        with torch.no_grad():
            lt = self.encode(u_0)

            if t_end is None:
                if iteration <N_burn_in : 
                    t_end = self.num_diffusion_timesteps - 60
                else : 
                    t_end = self.num_diffusion_timesteps
            else:
                if iteration < N_burn_in : 
                    t_end = self.num_diffusion_timesteps - 60
                else:
                    t_end = self.num_diffusion_timesteps - t_end

            diff_iter = self.reversed_time_steps[
                self.num_diffusion_timesteps - t_start : t_end
            ]

            if len(diff_iter) == 0:
                return self.decode(lt)

            for t in diff_iter:
                t_tensor = torch.tensor([t], device=self.device)
                eps = self.unet(lt, t_tensor).sample
                # Utiliser le scheduler du pipeline au lieu de tes alphas
                lt = self.scheduler.step(eps, t, lt).prev_sample

            xt = self.decode(lt)

            if show_steps:
                y = y.to(self.device)
                x_true = x_true.to(self.device)
                pilimg = display_as_pilimg(torch.cat((y, x_true, xt, xt), dim=3))

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