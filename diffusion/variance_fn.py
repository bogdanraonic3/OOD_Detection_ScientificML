import torch
import numpy as np

'''
    Std of the conditional prob. distribution of $p_{0t}(x(t) | x(0))$ that
    corresponds to the SDE:

    dx = sigma^t dw

    In this case, f(x,t) = 0 and g(t) = sigma^t. 
    This choice of f and g gives rise to the condition prob of the form:

    p_{0t}(x(t) | x(0)) ~ N(0, (sigma**(2 * t) - 1.) / 2. / np.log(sigma) * I)
'''
def marginal_prob_std_1(t, sigma, device = "cuda"):
    """
    Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

    Returns:
    The standard deviation.
    """
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff_1(t, sigma, device = "cuda"):
    """Compute the diffusion coefficient of our SDE.

    Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

    Returns:
    The vector of diffusion coefficients.
    """
    return torch.tensor(sigma**t, device=device)

def marginal_prob_std_2(t, sigma_min, sigma_max, device = "cuda"):
    """
    Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

    Returns:
    The standard deviation.
    """
    t = torch.tensor(t, device=device)
    #print(torch.sqrt(sigma_min * torch.pow(sigma_max/sigma_min, t)))
    return sigma_min * torch.pow(sigma_max/sigma_min, t)

def diffusion_coeff_2(t, sigma_min, sigma_max, device = "cuda"):
    """Compute the diffusion coefficient of our SDE.

    Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

    Returns:
    The vector of diffusion coefficients.
    """
    return torch.tensor(sigma_min * torch.pow(sigma_max/sigma_min, t) * np.sqrt(2*np.log(sigma_max/sigma_min)), device=device)
