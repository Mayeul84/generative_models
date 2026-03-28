import torch
from skimage.restoration import estimate_sigma
from diffusers import LDMPipeline
from tqdm import tqdm
import numpy as np

from df_models import LDM

def inverse_variance_function(noise_level, model):
    alphas_cumprod = model.alphas_cumprod
    closest_t_index = np.argmin(np.abs((1 - alphas_cumprod) - noise_level**2))
    return closest_t_index

def PNP_SGS(rho, MCMC_steps, x_true, y, Burn_in_steps, diffusing_model, operator, show_only_last=False, method_t_star="rho", sigma_noise=0.01, diffusion_steps_burn_in=20):
    assert operator.device == diffusing_model.device
    device = operator.device 

    y = torch.tensor(y).to(device) 
    if isinstance(diffusing_model, LDM):
        with torch.no_grad():
            l_init = torch.randn(diffusing_model.latent_shape, device=device)
            z = diffusing_model.decode(l_init)
    else:
        z = torch.randn(x_true.shape, device=device)
    rho = rho

    x_flatten = x_true.flatten()  
    y_flatten = x_flatten + sigma_noise * torch.randn_like(x_flatten)
    y_flatten = operator.HtH.dot(y_flatten.cpu().numpy())

    N_burn_in = Burn_in_steps
    x_samples = []
    time = []

    show = not show_only_last

    pbar = tqdm(range(MCMC_steps))
    for n in pbar:  # MCMC
        if n == MCMC_steps - 1:
            show = True

        if show:
            print(f"---------------- Iteration {n} ------------")
        
        # sample from x given z and y  : equation 6
        x = operator.sample_x_given_z_y(z, rho**2, y_flatten, sigma_noise**2).float()

        if isinstance(diffusing_model,LDM):
            with torch.no_grad():
                z_noisy = diffusing_model.encode(x)

            eps_scale = 0.01 * torch.std(z_noisy).item()
            epsilon = torch.randn_like(z_noisy) * eps_scale
            x_perturbed = diffusing_model.decode(z_noisy + epsilon)
            x_original = diffusing_model.decode(z_noisy)

            dx = x_perturbed - x_original
            dz = epsilon

            # jacobian
            factor_j = torch.sqrt(torch.mean(dx**2) / torch.mean(dz**2)).item()

            # method from article:
            if method_t_star == "estimated":
                sigma_estimated = estimate_sigma(x[0].cpu().numpy(), channel_axis=0, average_sigmas=True)
                sigma_latent_estimated = sigma_estimated / factor_j
                t_star = inverse_variance_function(sigma_latent_estimated,model=diffusing_model)
                t_end = None
                
            # method from rho:
            elif method_t_star == "rho":
                rho_min = 0.04
                if rho < rho_min:
                    sigma_estimated_latent = rho_min/factor_j
                    t_star = inverse_variance_function(sigma_estimated_latent,model=diffusing_model)
                else:
                    sigma_estimated_latent = rho/factor_j
                    t_star = inverse_variance_function(sigma_estimated_latent,model=diffusing_model)

                t_end = None
                
            # our method:
            elif method_t_star == "estimated+rho":

                sigma_latent_estimated = estimate_sigma(z_noisy[0].cpu().numpy(),channel_axis=0,average_sigmas=True)#sigma_estimated / factor_j
                t_star = inverse_variance_function(sigma_latent_estimated,model=diffusing_model)

                if sigma_latent_estimated <= (rho / factor_j):
                    t_end = 0
                else:
                    sigma_latent_estimated_end = np.sqrt(sigma_latent_estimated**2 - (rho / factor_j)**2)
                    t_end = inverse_variance_function(sigma_latent_estimated_end,model=diffusing_model)

                if show:
                    print(f"\nt_star: {t_star} and t_end: {t_end}.   ")

            if t_end is None:
                deltat = t_star
            else:
                deltat = t_star - t_end
                
            pbar.set_postfix(t_star=t_star,t_end=t_end,deltat=deltat)
            if t_end == t_star:
                t_end = t_end - 2 if t_end>=2 else 0
            
            if show:
                print(f"number of noising steps = {deltat}")

        else:

            # method from article
            if method_t_star == "estimated":
                sigma_estimated = estimate_sigma(x[0].cpu().numpy(), channel_axis=0, average_sigmas=True)
                t_star = inverse_variance_function(sigma_estimated,model=diffusing_model)
                t_end = None

            # method from rho:
            elif method_t_star == "rho":
                rho_min = 0.04
                if rho < rho_min:
                    t_star = inverse_variance_function(rho_min,model=diffusing_model)
                else:
                    t_star = inverse_variance_function(rho_min,model=diffusing_model)

                t_end = None

            # method from another article
            elif method_t_star == "rho_chen":
                rho_min = 0.04
                if rho < rho_min:
                    t_star = inverse_variance_function(np.sqrt(1 - 1/(1+rho_min**2)),model=diffusing_model)
                else:
                    t_star = inverse_variance_function(np.sqrt(1 - 1/(1+rho**2)),model=diffusing_model)

                t_end = None

            # method from our group
            elif method_t_star == "estimated+rho":
                sigma_estimated = estimate_sigma(x[0].cpu().numpy(), channel_axis=0, average_sigmas=True)
                t_star = inverse_variance_function(sigma_estimated,model=diffusing_model)
                
                if sigma_estimated <= rho:
                    t_end = 0
                else:
                    t_end = inverse_variance_function(np.sqrt(abs(sigma_estimated**2 - rho**2)),model=diffusing_model)

                if show:
                    print(f"\nt_star: {t_star} and t_end: {t_end}.   ")

            if t_end is None:
                deltat = t_star
            else:
                deltat = t_star - t_end
                
            pbar.set_postfix(t_star=t_star,t_end=t_end,deltat=deltat)
            if t_end == t_star:
                t_end = t_end - 2 if t_end>=2 else 0

            if show:
                print(f"number of noising steps = {deltat}")

        time.append(t_star)


        # sample z via reverse diffusion : equation 7
        z = diffusing_model.sampling_splitting_z(t_star, x, x_true, torch.tensor(y_flatten).reshape(1,3,256,256), n, show_steps=show, t_end=t_end, N_burn_in=Burn_in_steps, diffusion_steps_burn_in=diffusion_steps_burn_in)

        if isinstance(diffusing_model,LDM) and show:
            mse = torch.mean((z - x)**2)
            rmse = torch.sqrt(mse)
            print(f"Observed noise level: {rmse}")
            
        if n > N_burn_in:
            x_samples.append(z)

    return x_samples, time
