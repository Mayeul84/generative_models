import torch
from skimage.restoration import estimate_sigma
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
        t_star = inverse_variance_function(noise_level)
        time.append(t_star)

        if show:
            print(f"noise level estimated = {noise_level}")
            print(f"number of noising steps = {t_star}")
        
        # Step 3 : Sample z via reverse diffusion : equation 7
        z = diffusing_model.sampling_spliting_z(t_star, x, x_true, y, n, show_steps=show)

        if n > N_burn_in:
            x_samples.append(z)

    return x_samples, time
