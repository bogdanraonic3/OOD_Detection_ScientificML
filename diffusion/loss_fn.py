import torch
import numpy as np

'''
    Let score be defined as 
    
        score(x,t) = F_theta(x, t) / std(t), 
    
    as in our model (where F_theta is a NN).

    The loss function is defined as

        L = E_{t~U(0,1)} E_{y~p_data} E_{n ~ N(0,I)} || sigma(t) * score(y + sigma(t) n, t) + n ||_2^2

    This specic loss function can be transformed to the loss function in Karas sense in

         -- Elucidating the Design Space of Diffusion-Based Generative Models --
    
    Thus, we multiply the score by ONE sigma(t) -- It represents standard deviation defined in Song sense in

        -- SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS
'''

# Loss function defined in Song sese (SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS)

def log_uniform_sample(sigma_min, sigma_max, shape):
    # Convert sigma_min and sigma_max to tensors for proper operations
    sigma_min = torch.tensor(sigma_min, dtype=torch.float32)
    sigma_max = torch.tensor(sigma_max, dtype=torch.float32)
    
    log_sigma = torch.rand(shape) * (torch.log(sigma_max) - torch.log(sigma_min)) + torch.log(sigma_min)
    return (torch.exp(log_sigma)-sigma_min)/(sigma_max-sigma_min)

def loss_fn(model, 
            x, 
            x_cond,
            t = None,
            marginal_prob_std = None, 
            eps=1e-4, 
            is_log_uniform = False,
            log_uniform_frac = 3,
            is_train = True,
            mask = None):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    dim_cond: dimension of the conditions in the conditional diffusion
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """

  B, C = x.shape[0], x.shape[1]
      
  if mask is None:
    mask = torch.ones((B, C), device=out.device, dtype=out.dtype)
  M = torch.sum(mask, dim = [1])
  mask = mask.view(-1, C, 1, 1)

  if is_train:
    if not is_log_uniform:
      random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    else:
      random_t = log_uniform_sample(0.1, 0.3, x.shape[0]) * (1. - eps) + eps
      #samples = torch.rand(x.shape[0], device=x.device)
      #log_min, log_max = np.log(eps), np.log(1.0)
      #samples = (log_max - log_min) * samples + log_min
      #random_t = torch.exp(samples)
    
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)  
    #perturbed_x = x + z * std[:, None, None, None]
    #score = model(perturbed_x, x_cond, random_t, t)

    score = model(x + z * std[:, None, None, None], x_cond, random_t, t)

    squared_loss = (score * std[:, None, None, None] + z)**2
    squared_loss = mask * squared_loss
    loss = torch.mean(torch.mean(squared_loss, dim=(1,2,3))*(C/M))
    return loss
  else:
    err_val = 0
    for level in range(8):
      t_min = level/8.0
      t_max = (level + 1)/8.0
      random_t = (log_uniform_sample(0.1, 0.3, x.shape[0]) * (1. - eps) + eps)*(t_max-t_min)+t_min
      z = torch.randn_like(x)
      std = marginal_prob_std(random_t)  
      score = model(x + z * std[:, None, None, None], x_cond, random_t, t)
      squared_loss = (score * std[:, None, None, None] + z)**2
      loss = torch.mean(torch.mean(squared_loss, dim=(1,2,3))*(C/M))
      err_val = err_val + loss.item()
    return err_val/8.0

def edm_weight(sigma, 
              sigma_data = 0.5):        
        sigma_squared = sigma**2
        return (sigma_squared + sigma_data**2)/(sigma_squared * sigma_data**2)

def loss_fn_denoised(model, 
                    x, 
                    x_cond,
                    t = None,
                    marginal_prob_std = None, 
                    eps=1e-4, 
                    is_log_uniform = False,
                    log_uniform_frac = 3,
                    is_train = True,
                    weighting = "edm",
                    sigma_data = 0.5,
                    consistent_weight = 0.01,
                    channel_weight = None,
                    mask = None):

  if is_train:
    if not is_log_uniform:
      random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    else:
      random_t = log_uniform_sample(0.1, 0.3, x.shape[0]) * (1. - eps) + eps
    
    z = torch.randn_like(x)
    
    std = marginal_prob_std(random_t) 
    if weighting is None: 
      weight = torch.ones_like(std)
    elif weighting == "edm":
      weight = edm_weight(std, sigma_data=sigma_data)

    denoised = model(x + z * std[:, None, None, None], x_cond, random_t, t)

    x_squared = torch.square(x)
    denoise_squared = torch.square(denoised)
    rel_norm = torch.mean(x_squared / torch.mean(torch.square(x_squared)))

    if channel_weight == None:
      channel_weight = 1.0
    else:
      channel_weight = torch.tensor(channel_weight, device = weight.device).view(1, x.shape[1], 1, 1)

    loss = torch.mean(channel_weight * weight[:, None, None, None]* torch.square(denoised - x))
    loss = loss + consistent_weight * rel_norm * torch.mean(weight[:, None, None, None] * torch.square(denoise_squared - x_squared))

    return loss
  else:
    err_val = 0
    for level in range(8):
      t_min = level/8.0
      t_max = (level + 1)/8.0
      random_t = (log_uniform_sample(0.1, 0.3, x.shape[0]) * (1. - eps) + eps)*(t_max-t_min)+t_min
      
      z = torch.randn_like(x)
      std = marginal_prob_std(random_t) 
      if weighting is None: 
        weight = torch.ones_like(std)
      elif weighting == "edm":
        weight = edm_weight(std, sigma_data=sigma_data)

      denoised = model(x + z * std[:, None, None, None], x_cond, random_t, t)
      loss = torch.mean(weight[:, None, None, None] * torch.square(denoised - x))
      err_val = err_val + loss.item()
    return err_val/8.0
