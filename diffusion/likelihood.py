import torch

import functools
import numpy as np
from torch.func import vmap, jvp
from xitorch import integrate as integrate_torch

import time

def prior_likelihood_torch(z, sigma, device = "cuda"):  
    """The likelihood of a Gaussian distribution with mean zero and
      standard deviation sigma."""
    shape = z.shape
    N = torch.prod(torch.tensor(shape[1:], device=device))
    dim = [i for i in range(1, len(shape))]
    return -N / 2. * torch.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=dim) / (2 * sigma**2)


def ode_likelihood(model,
                    x,
                    condition,
                    marginal_prob_std,
                    diffusion_coeff,
                    t_batch = None,
                    batch_size=64,
                    device='cuda',
                    eps = 1e-6,
                    rtol = 1e-6,
                    atol = 1e-6,
                    epsilon = None,
                    ode_method = "rk38",
                    reduce_prior = False,
                    avg_grad = False,
                    list_of_indicies = None):

    shape = x.shape    

    B,C = shape[0], torch.prod(torch.tensor(shape[1:], device=device))

    grad_counter = 0
    grads = torch.zeros(tuple(x.shape),  device=device)
    grad_norms = torch.zeros((x.shape[0],),  device=device)
    all_grads = 0

    def divergence_eval(fn, x, epsilon):
        """Compute the divergence of the score-based model with Skilling-Hutchinson."""

        epsilon.requires_grad_(True)
        x.requires_grad_(True)

        
        f_e = lambda x, eps: torch.sum(fn(x).reshape(B,C) * eps.view(1,-1), dim=-1) # shape (B,)

        def e_Jf_e(eps):
            _, jvp_result = jvp(f_e, (x, eps), (eps.repeat((B,) + (len(shape)-1)*(1,)), torch.zeros_like(eps)))
            return jvp_result

        e_Jf_e_multisample = vmap(e_Jf_e, out_dims=1, randomness="same") # allow additional "batch-dimension" for eps (this is sample_size)
        res = e_Jf_e_multisample(epsilon)
        return res.mean(dim=1)

    def ode_func(t, x):
        nonlocal grad_counter, grads, grad_norms

        """The ODE function for the black-box solver."""
        time_steps = torch.ones((shape[0],), device = device) * t   
        sample = x[:B*C]
        logp = x[B*C:]
        g = diffusion_coeff(torch.tensor(t, device=device))
        
        sample = sample.reshape(shape)  # Convert to tensor
        time_steps = time_steps.reshape((sample.shape[0],))

        std = marginal_prob_std(time_steps)
        
        model.train()
        if t_batch is None:
            #print(x.shape, condition.shape, time_steps.reshape((B,)).shape, std.shape)
            fn = lambda x: (model(x, condition, time_steps.reshape((B,))) - x.reshape(shape)) / std[(slice(None),) + (None,) * (len(shape) - 1)]**2
        else:
            fn = lambda x: (model(x, condition, time_steps.reshape((B,)), t_batch) - x.reshape(shape)) / std[(slice(None),) + (None,) * (len(shape) - 1)]**2
        
        logp_grad = -0.5 * g**2 * divergence_eval(fn, sample, epsilon)
        sample_grad = -0.5 * g**2 * fn(sample)

        grads = grads + 2 * std.view(std.shape[0], 1, 1, 1) * sample_grad/g**2
        grad_norms =  grad_norms + torch.mean(torch.abs(grads), axis = (1,2,3))
        grad_counter+=1 

        return torch.cat([sample_grad.reshape(-1), logp_grad.reshape(-1)], axis = 0)

    '''
    init = np.concatenate([x.cpu().numpy().reshape((-1,)), np.zeros((shape[0],))], axis=0)
    # Black-box ODE solver
    res = integrate.solve_ivp(ode_func, (eps, 1.), init, rtol=rtol, atol=atol, method='RK45')
    zp = res.y[:,-1]
    '''

    init = torch.cat([x.view(-1), torch.zeros(shape[0], device=device)])
    
    res = integrate_torch.solve_ivp(fcn = ode_func, y0=init, ts = torch.tensor([eps, 1.0], device=device), rtol=rtol, atol=atol, method=ode_method)
    zp = res[-1]
    
    z, delta_logp = zp[:B*C], zp[B*C:]
    z = z.reshape((B,C))

    sigma_max = marginal_prob_std(torch.tensor(1.0, device=device))
    N = torch.prod(torch.tensor(shape[1:], device=device))
    
    if reduce_prior:
        prior_logp = prior_likelihood_torch(z, sigma_max,device=device)/N
    else:
        prior_logp = z
    
    dt = 1.0/(grad_counter - 1)
    time_grads = (grads[:, 1:] - grads[:, :-1])/((grad_counter - 1) * dt)
    grads = grads / grad_counter
    
    if avg_grad:
        return grads, time_grads, grad_norms/grad_counter
    else:
        return z, prior_logp, delta_logp/N
