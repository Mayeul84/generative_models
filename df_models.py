import torch
import numpy as np

import tqdm
from utils import *

class DDPM:
  def __init__(self, num_diffusion_timesteps=1000, beta_start=0.0001, beta_end=0.002, imgshape=(1,3,256,256), model=model):
    self.reversed_time_steps = np.arange(self.num_diffusion_timesteps)[::-1]
    self.betas = np.linspace(beta_start, beta_end, self.num_diffusion_timesteps,
                              dtype=np.float64)
    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
    self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
    self.sigma = np.sqrt(1 - self.alphas_cumprod_prev )
    self.model = model
    self.imgshape = imgshape


  def get_eps_from_model(self, x, t, device="cpu"):
    model_output = self.model(x, torch.tensor(t, device=device).unsqueeze(0))
    model_output = model_output[:,:3,:,:]
    return(model_output)

  def predict_xstart_from_eps(self, x, eps, t):
    x_start = (
        np.sqrt(1.0 / self.alphas_cumprod[t])* x
        - np.sqrt(1.0 / self.alphas_cumprod[t] - 1) * eps
    )
    x_start = x_start.clamp(-1.,1.)
    return(x_start)

  def sampling_spliting_z(self, t_start, u_0, x_true, y, iteration, show_steps, device="cpu"):
        with torch.no_grad():  # mode eval
            xt = u_0 
            xhat = torch.randn(self.imgshape, device=device)
            # t_start = min(t_start, 100)
            if iteration < 20 : 
                t_end = self.num_diffusion_timesteps - (t_start //2)
            else : 
                t_end = self.num_diffusion_timesteps

            diff_iter = self.reversed_time_steps[self.num_diffusion_timesteps - t_start:t_end]

            for i, t in enumerate(diff_iter):
                if t > 1:
                    z = torch.randn(self.imgshape, device=device)
                else:
                    z = torch.zeros(self.imgshape, device=device)

                alpha_t = self.alphas[t]
                alpha_bar_t = self.alphas_cumprod[t]
                sigma_t = np.sqrt(((1 - self.alphas_cumprod[t - 1]) / (1 - self.alphas_cumprod[t])) * self.betas[t])
                eps = self.get_eps_from_model(xt, t)

                xt = (1 / np.sqrt(alpha_t)) * (xt -  ((1 - alpha_t) / np.sqrt(1 - alpha_bar_t)) * eps ) + sigma_t * z
                xhat = self.predict_xstart_from_eps(xt, eps, t)

                
            if show_steps :
                    pilimg = display_as_pilimg(torch.cat((y, x_true, xt, xhat), dim=3))

        return xt
    
  def posterior_sampling(self, linear_operator, y, x_true=None, show_steps=True, vis_y=None, device="cpu"):

    # visualization image for the observation y:
    if vis_y==None:
      vis_y = y

    # initialize xt for t=T
    x = torch.randn(self.imgshape,device=device)
    x.requires_grad = True


    for t in tqdm(self.reversed_time_steps[1:]):
      alpha_t = self.alphas[t]
      alpha_bar_t = self.alphas_cumprod[t]
      alpha_bar_tm1 = self.alphas_cumprod_prev[t]

      beta_t = self.betas[t]
      sigma_t = np.sqrt(beta_t)

      z = torch.randn(self.imgshape, device=device)

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
        pilimg = display_as_pilimg(torch.cat(( y, x_true, x, xhat), dim=3))

    return(x)

ddpm = DDPM()