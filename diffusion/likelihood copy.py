
import torch
import functools
import numpy as np
from scipy import integrate

from torch.func import functional_call, vmap, jvp
import time

from torchdiffeq import odeint



def prior_likelihood(z, sigma):  
    """The likelihood of a Gaussian distribution with mean zero and
      standard deviation sigma."""
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * torch.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=(1,2,3)) / (2 * sigma**2)

def ode_likelihood(x,
                   condition,
                   model,
                   marginal_prob_std,
                   diffusion_coeff,
                   t_batch = None,
                   batch_size=64,
                   device='cuda',
                   eps = 1e-6,
                   rtol = 1e-6,
                   atol = 1e-6,
                   is_skip = False):
    """Compute the likelihood with probability flow ODE.

    Args:
        x: Input data.
        model: A PyTorch model representing the score-based model.
        marginal_prob_std: A function that gives the standard deviation of the
        perturbation kernel.
        diffusion_coeff: A function that gives the diffusion coefficient of the
        forward SDE.
        batch_size: The batch size of epsilon.
        device: 'cuda' for evaluation on GPUs, and 'cpu' for evaluation on CPUs.
        eps: A `float` number. The smallest time step for numerical stability.
`
    Returns:
        z: The latent code for `x`.
        bpd: The log-likelihoods per dim.
    """
        
    shape = x.shape
    shape_epsilon = (batch_size,)+shape[1:] 
    epsilon = torch.randn(shape_epsilon, device = device).reshape(batch_size,-1).type(torch.float32) #Shape (batchsize, other)
    #epsilon = np.sqrt(np.prod(shape[1:])) * epsilon / torch.norm(epsilon, dim=1, keepdim=True) # normalize magnitude (to decrease variance)
    epsilon = torch.sqrt(torch.prod(torch.tensor(shape[1:], device=device))) * epsilon / torch.norm(epsilon, dim=1, keepdim=True)

    def score_model(x, condition, time_steps, t_batch):
        if not is_skip:
            return model(x, condition, time_steps, t_batch)
        denoised = model(x, condition, time_steps, t_batch)
        std = marginal_prob_std(batch_time_step)
        return (denoised - x)/std[:, None, None, None]**2

    def divergence_eval(x, time_steps, epsilon, condition):
        """Compute the divergence of the score-based model with Skilling-Hutchinson."""
        
        with torch.enable_grad():
            x = torch.tensor(x, device = device).type(torch.float32)
            time_steps =  torch.tensor(time_steps, device = device).type(torch.float32)
            x.requires_grad_(True)
            B, C = shape[0], torch.prod(torch.tensor(shape[1:], device=device))
            
            res = torch.zeros(B, device = device).type(torch.float32)
            for i in range(batch_size):
                f_e = lambda x, eps: torch.sum(score_model(x.reshape(shape), condition, time_steps, t_batch).reshape(B,C) * eps.view(1,-1), dim=-1) # shape (B,)
                e_Jf_e = lambda eps: torch.autograd.functional.jvp(f_e, (x, eps), (eps.repeat(B,1).reshape(-1,), torch.zeros_like(eps)))[1]
                res += e_Jf_e(epsilon[i])
 
            return res/batch_size

            
    def score_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the score-based model for the black-box ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
        with torch.no_grad():
            return score_model(sample, condition, time_steps, t_batch)
        #return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for the black-box solver."""
        time_steps = torch.full((shape[0],), t, device=device)
        sample, logp = x[:-shape[0]], x[-shape[0]:]
        #g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        g = diffusion_coeff(torch.tensor(t, device=device))
        sample_grad = -0.5 * g**2 * score_eval_wrapper(sample, time_steps)
        logp_grad = -0.5 * g**2 * divergence_eval(sample, time_steps, epsilon,condition)
        #logp_grad = logp_grad.detach().cpu().numpy()

        #return np.concatenate([sample_grad, logp_grad], axis=0)
        print(sample_grad.shape, logp_grad.shape)
        return torch.cat([sample_grad, logp_grad], axis = 0)
    
    '''
    init = np.concatenate([x.cpu().numpy().reshape((-1,)), np.zeros((shape[0],))], axis=0)
    
    # Black-box ODE solver
    res = integrate.solve_ivp(ode_func, (eps, 1.), init, rtol=rtol, atol=atol, max_step = 32, method='RK45')
    
    zp = torch.tensor(res.y[:, -1], device=device)
    z = zp[:-shape[0]].reshape(shape)
    delta_logp = zp[-shape[0]:].reshape(shape[0])

    sigma_max = marginal_prob_std(1.)
    prior_logp = prior_likelihood(z, sigma_max)

    N = np.prod(shape[1:])
    bpd = (prior_logp + delta_logp)/N
    
    print(prior_logp.detach().cpu().numpy()/N)
    print(" ")
    print(delta_logp.detach().cpu().numpy()/N)
    '''

    init = torch.cat([x.view(-1), torch.zeros(shape[0], device=device)])
    res = odeint(ode_func, init, torch.tensor([eps, 1.], device=device), rtol=rtol, atol=atol, method='rk4')
    
    zp = res[-1]
    z, delta_logp = zp[:-shape[0]].view(shape), zp[-shape[0]:]
    sigma_max = marginal_prob_std(torch.tensor(1., device=device))
    prior_logp = prior_likelihood(z, sigma_max)
    N = torch.prod(torch.tensor(shape[1:], device=device))
    bpd = (prior_logp + delta_logp) / N
    return z, bpd

'''

import torch
import numpy as np
from torchdiffeq import odeint

def prior_likelihood(z, sigma):  
    shape = z.shape
    N = torch.prod(torch.tensor(shape[1:], device=z.device))
    return -N / 2. * torch.log(2 * np.pi * sigma**2) - torch.sum(z**2, dim=(1,2,3)) / (2 * sigma**2)

def ode_likelihood(x, condition, model, marginal_prob_std, diffusion_coeff,
                   batch_size=64, device='cuda', eps=1e-6, rtol=1e-6, atol=1e-6, is_skip=False):
    shape = x.shape
    epsilon = torch.randn((batch_size,) + shape[1:], device=device).reshape(batch_size, -1)
    epsilon = torch.sqrt(torch.prod(torch.tensor(shape[1:], device=device))) * epsilon / torch.norm(epsilon, dim=1, keepdim=True)
    
    def score_model(x, condition, time_steps):
        if not is_skip:
            return model(x, condition, time_steps)
        denoised = model(x, condition, time_steps)
        std = marginal_prob_std(time_steps).view(-1, 1, 1, 1)
        return (denoised - x) / (std ** 2)
    
    def divergence_eval(x, time_steps, epsilon, condition):
        with torch.enable_grad():
            x = x.requires_grad_(True)
            B, C = shape[0], torch.prod(torch.tensor(shape[1:], device=device))
            res = torch.zeros(B, device=device)
            for i in range(batch_size):
                def f_e(x):
                    return torch.sum(score_model(x.view(shape), condition, time_steps).view(B, C) * epsilon[i].view(1, -1), dim=-1)
                res += torch.autograd.functional.jvp(f_e, (x,), (epsilon[i].repeat(B, 1).reshape(-1,),))[1]
        return res / batch_size
    
    def ode_func(t, x):
        time_steps = torch.full((shape[0],), t, device=device)
        sample, logp = x[:-shape[0]], x[-shape[0]:]
        g = diffusion_coeff(torch.tensor(t, device=device))
        sample_grad = -0.5 * g**2 * score_model(sample.view(shape), condition, time_steps).view(-1)
        logp_grad = -0.5 * g**2 * divergence_eval(sample, time_steps, epsilon, condition)
        return torch.cat([sample_grad, logp_grad])
    
    init = torch.cat([x.view(-1), torch.zeros(shape[0], device=device)])
    res = odeint(ode_func, init, torch.tensor([eps, 1.], device=device), rtol=rtol, atol=atol, method='rk4')
    
    zp = res[-1]
    z, delta_logp = zp[:-shape[0]].view(shape), zp[-shape[0]:]
    sigma_max = marginal_prob_std(torch.tensor(1., device=device))
    prior_logp = prior_likelihood(z, sigma_max)
    N = torch.prod(torch.tensor(shape[1:], device=device))
    bpd = (prior_logp + delta_logp) / N
    
    return z, bpd
'''