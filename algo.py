# Je viens de modifier sigma_estimate, maintenant on prend directement ro comme sigma (c'est littéralement le but de l'article)
# Voir si ça marche encore sur le cas de base. Puis voir si ça marche sur le latent (Normalement c'est plus simple car on a plus
# à estimer ro)


import torch
from skimage.restoration import estimate_sigma
from diffusers import LDMPipeline
from tqdm import tqdm
import numpy as np

from df_models import LDM

def inverse_variance_function(noise_level, model):
    if isinstance(model, LDM):
        alphas_cumprod = model.scheduler.alphas_cumprod.numpy()
    else:
        alphas_cumprod = model.alphas_cumprod
    closest_t_index = np.argmin(np.abs((1 - alphas_cumprod) - noise_level**2))
    return closest_t_index

def PNP_SGS(ro, MCMC_steps, x_true, y, Burn_in_steps, diffusing_model, operator, show_only_last=False):
    assert operator.device == diffusing_model.device
    device = operator.device 

    y = torch.tensor(y).to(device)  # Observed measurements
    if isinstance(diffusing_model, LDM):
        with torch.no_grad():
            l_init = torch.randn(diffusing_model.latent_shape, device=device)
            z = diffusing_model.decode(l_init)
    else:
        z = torch.randn(y.shape, device=device)
    ro = ro

    sigma_noise = 0.01#estimate_sigma(y[0].cpu().numpy(), channel_axis=0, average_sigmas=True)
    x_flatten = x_true.flatten()  
    y_flatten = x_flatten + sigma_noise * torch.randn_like(x_flatten)
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

        ### L'erreur est peut être ici sur le sigma_noise ? Sur la fonction sampling ?
        x = operator.sample_x_given_z_y(z, ro**2, y_flatten, sigma_noise**2).float()

        # # Step 2: estimating noise level
        # if isinstance(diffusing_model, LDM):
        #     noise_level = diffusing_model.estimate_sigma_latent(x)
        # else:
        #     noise_level = estimate_sigma(x[0].cpu().numpy(), channel_axis=0, average_sigmas=True)
        noise_level=None
        # L'article suppose que sigma= rho c'est tout l'enjeu

        # Step 3: find t
        if isinstance(diffusing_model,LDM):
            with torch.no_grad():
                z_noisy = diffusing_model.encode(x)

            eps_scale = 0.01 * torch.std(z_noisy).item()
            epsilon = torch.randn_like(z_noisy) * eps_scale
            x_perturbed = diffusing_model.decode(z_noisy + epsilon)
            x_original = diffusing_model.decode(z_noisy)

            # Différence dans l'image
            dx = x_perturbed - x_original
            dz = epsilon

            # Norme moyenne pour approx le jacobien
            factor_j = torch.sqrt(torch.mean(dx**2) / torch.mean(dz**2))
            sigma_latent = (ro / factor_j).item()
            t_star = inverse_variance_function(sigma_latent,model=diffusing_model)
            if show:
                print(f"noise level estimated = {sigma_latent}")
                print(f"number of noising steps = {t_star}")
        else:
            t_star = inverse_variance_function(ro,model=diffusing_model)
            if show:
                #print(f"noise level estimated = {noise_level}")
                print(f"number of noising steps = {t_star}")

        time.append(t_star)

        
        # Step 3 : Sample z via reverse diffusion : equation 7
        z = diffusing_model.sampling_splitting_z(t_star, x, x_true, torch.tensor(y_flatten).reshape(1,3,256,256), n, show_steps=show)

        if isinstance(diffusing_model,LDM) and show:
            mse = torch.mean((z - x)**2)
            rmse = torch.sqrt(mse)
            print(f"Observed noise level: {rmse}")
            
        if n > N_burn_in:
            x_samples.append(z)

    return x_samples, time
