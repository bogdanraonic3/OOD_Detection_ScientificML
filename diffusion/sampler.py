
import torch
import tqdm
from scipy import integrate
import numpy as np
import time

def Euler_Maruyama_sampler(model,
                           marginal_prob_std,
                           diffusion_coeff,
                           condition,
                           t_batch = None,
                           batch_size=64,
                           num_steps=128,
                           device='cuda',
                           dimension = (3,32,32),
                           eps=1e-3,
                           is_skip = False,
                           guidance_condition = None):
  """Generate samples from score-based models with the Euler-Maruyama solver.

  Args:
    model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps.
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

  Returns:
    Samples.
  """

  if condition is not None:
    device = condition.device
  elif t_batch is not None:
    devce = t_batch.device
  else:
    device = next(model.parameters()).device

  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, dimension[0], dimension[1], dimension[2], device=device) * marginal_prob_std(t).to(device)[:, None, None, None]
  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  with torch.no_grad():
    for time_step in tqdm.tqdm(time_steps):
        batch_time_step = torch.ones(batch_size, device=device) * time_step
        g = diffusion_coeff(batch_time_step).to(device)

        if guidance_condition is not None:
            guidance, dim = guidance_condition
            dim_in, dim_out = dim
            assert guidance.shape[0] == batch_size
            #x[:,dim_in:dim_out] = guidance

        if not is_skip:
          score = model(x, condition, batch_time_step, t_batch)
        else:
          denoised = model(x, condition, batch_time_step, t_batch)
          if guidance_condition is not None:
            denoised[:,dim_in:dim_out] = guidance
          std = marginal_prob_std(batch_time_step)
          score = (denoised - x)/std[:, None, None, None]**2
        mean_x = x + (g**2)[:, None, None, None] * score * step_size
        x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
        
  # Do not include any noise in the last sampling step.
  return mean_x



'''
## The error tolerance for the black-box ODE solver
def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                condition,
                t_bacth = None,
                batch_size=64,
                atol=1e-5,
                rtol=1e-5,
                device='cuda',
                z=None,
                dimension = (3,32,32),
                eps=1e-3):
  """Generate samples from score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
  """
  t = torch.ones(batch_size, device=device)
  # Create the latent code
  if z is None:
    init_x = torch.randn(batch_size, dimension[0], dimension[1], dimension[2], device=device) * marginal_prob_std(t)[:, None, None, None]
  else:
    init_x = z

  shape = init_x.shape

  #print(shape)
  #time.sleep(1000)

  def score_eval_wrapper(sample, time_steps):
    """A wrapper of the score-based model for use by the ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
    with torch.no_grad():
      score = score_model(sample, condition, time_steps, t_bacth)
    
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)

  def ode_func(t, x):
    """The ODE function for use by the ODE solver."""
    time_steps = np.ones((shape[0],)) * t
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)

  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
  print(f"Number of function evaluations: {res.nfev}")
  x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

  return x
'''