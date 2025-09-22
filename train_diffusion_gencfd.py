import torch
import os
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from dataloader.dataloader import CIFAR10_Dataset, Wave2d_Dataset
import tqdm
import argparse
import copy
import wandb
import matplotlib.pyplot as plt

from diffusion.loss_fn import loss_fn
from diffusion.lr_schedulers import CosineLinearWarmupCustomScheduler

from GenCFD import model
from diffusion.model import EMA

from diffusion.sampler import Euler_Maruyama_sampler, ode_sampler

from diffusion.likelihood import ode_likelihood
from diffusion.variance_fn import marginal_prob_std_1, diffusion_coeff_1, marginal_prob_std_2, diffusion_coeff_2
from visualization.plot import plot_prediction
from utils.utils_data import get_loader, save_data, load_data, read_cli_regression, save_errors, read_cli_diffusion_gencfd, select_variable_condition
import time 

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_DIR"] = "/cluster/work/math/braonic/TrainedModels/OOD_Generalization/ood_wandb_logs"

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="load parameters for training")
  params = read_cli_diffusion_gencfd(parser).parse_args()

  if params.config is None:
    config = params
  else:
    config = argparse.Namespace(**load_data(params.config))

  device = config.device
  which_data = config.which_data
  which_type = config.which_type
  tag = config.tag
  is_log_uniform = config.is_log_uniform
  log_uniform_frac = config.log_uniform_frac
  is_exploding = config.is_exploding

  sigma =  config.sigma
  n_epochs = config.epochs
  peak_lr=  config.peak_lr
  end_lr =  config.end_lr
  warmup_epochs =  config.warmup_epochs
  batch_size = config.batch_size
  N_train = config.N_train
  ood_share = config.ood_share
  in_dim = config.in_dim
  out_dim = config.out_dim
  channels = tuple(config.unet_param)
  
  if is_exploding:
    marginal_prob_std_fn = functools.partial(marginal_prob_std_2, sigma_min = 0.001, sigma_max=sigma, device = device)
    diffusion_coeff_fn = functools.partial(diffusion_coeff_2, sigma_min = 0.001, sigma_max=sigma, device = device)
  else:
    marginal_prob_std_fn = functools.partial(marginal_prob_std_1, sigma=sigma, device = device)
    diffusion_coeff_fn = functools.partial(diffusion_coeff_1, sigma=sigma, device = device)

  run = wandb.init(entity="bogdanraonic", 
                  project=config.wandb_project_name, 
                  name=tag + config.wandb_run_name, 
                  config=config)

  train_loader = get_loader(which_data = which_data,
                            which_type = "train",
                            N_samples = N_train,
                            batch_size = batch_size,
                            ood_share = ood_share)
  
  val_loader = get_loader(which_data = which_data,
                            which_type = "val",
                            N_samples = 0,
                            batch_size = batch_size,
                            ood_share = ood_share)

  if which_type == "xy":
    dim = in_dim
    dim_cond = out_dim 
  elif which_type == "yx":
    dim = out_dim
    dim_cond = in_dim
  elif which_type == "x":
    dim = in_dim
    dim_cond = 0
  elif which_type == "y":
    dim = out_dim
    dim_cond = 0
  elif which_type == "x&y":
    dim = out_dim + in_dim
    dim_cond = 0
  else:
    raise ValueError('which_type needs to be in [x, y, xy, yx]')

  score_model = model.PreconditionedDenoiser(
      in_channels=dim + dim_cond, # Conditioning thus stacked input and output
      out_channels=dim,
      spatial_resolution=(128,128),
      time_cond=False,
      num_channels= channels,
      downsample_ratio=(2,)*len(channels),
      marginal_prob_std = marginal_prob_std_fn,
      num_blocks=4,
      noise_embed_dim=128,
      output_proj_channels=128,
      input_proj_channels=128,
      padding_method='zeros',
      dropout_rate=0.0,
      use_attention=True,
      use_position_encoding=True,
      num_heads=8,
      normalize_qk=False,
      dtype=torch.float32,
      device= device,
      buffer_dict=dict(),
      sigma_data=0.5,
  )

  #score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn, dim = dim, dim_cond = dim_cond, channels=channels)
  
  total_params = sum(p.numel() for p in score_model.parameters())
  print(total_params)   
  #score_model = score_model.to(device)
  #score_model.print_size()
  score_model = torch.nn.DataParallel(score_model)

  dir_path = f"/cluster/work/math/braonic/TrainedModels/OOD_Generalization/{which_data}/{tag}_diffusion_gencfd"
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  save_data(vars(config), f"{dir_path}/param_diffusion_gencfd_{tag}.json")

  optimizer = Adam(score_model.parameters(), lr=peak_lr)
  scheduler = CosineLinearWarmupCustomScheduler(optimizer, 
                                                warmup_epochs = warmup_epochs, 
                                                total_epochs = n_epochs, 
                                                peak_lr = peak_lr, 
                                                end_lr = end_lr)
  tqdm_epoch = tqdm.trange(n_epochs)
  ema = EMA(score_model, decay=0.999)

  best_avg_loss = 1000.0

  best_model = copy.deepcopy(score_model)
  for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    avg_loss_val = 0.

    score_model.train()
    '''
    for param_group in optimizer.param_groups:
          tagparam_group['lr'])
          break  # Update optimizer LR
    '''

    for input_batch, output_batch in train_loader:
      input_batch = input_batch.to(device)
      output_batch = output_batch.to(device)

      variable, condition = select_variable_condition(input_batch, output_batch, which_type = which_type)
      loss = loss_fn(score_model, variable, condition, marginal_prob_std = marginal_prob_std_fn)
      '''
      if which_type == "xy":
        loss = loss_fn(score_model, input_batch, output_batch, marginal_prob_std_fn)
        condition = output_batch
      elif which_type == "yx":
        loss = loss_fn(score_model, output_batch, input_batch, marginal_prob_std_fn)
        condition = input_batch
      elif which_type == "x":
        loss = loss_fn(score_model, input_batch, None, marginal_prob_std_fn)
        condition = None
      elif which_type == "y":
        loss = loss_fn(score_model, output_batch, None, marginal_prob_std_fn)
        condition = None
      elif which_type == "x&y":
        loss = loss_fn(score_model, torch.cat((input_batch, output_batch), axis = 1), None, marginal_prob_std_fn, is_log_uniform = is_log_uniform, log_uniform_frac = log_uniform_frac)
        condition = None
      '''

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      ema.update()

      avg_loss += loss.item() * output_batch.shape[0]
      num_items += output_batch.shape[0]
    avg_loss /= num_items

    # Apply EMA for inference
    ema.apply_shadow()  # Temporarily sets model to EMA params
    score_model.eval()
    scheduler.step()

    num_items = 0
    with torch.no_grad():
        for input_batch, output_batch in val_loader:
          #x = torch.cat((input_batch, output_batch), dim = 1)
          input_batch = input_batch.to(device)
          output_batch = output_batch.to(device)

          '''
          if which_type == "xy":
            variable = input_batch
            condition = output_batch
          elif which_type == "yx":
            variable = output_batch
            condition = input_batch
          elif which_type == "x":
            variable = input_batch
            condition = None
          elif which_type == "y":
            variable = output_batch
            condition = None
          elif which_type == "x&y":
            variable = torch.cat((input_batch, output_batch), axis = 1)
            condition = None
          '''
          variable, condition = select_variable_condition(input_batch, output_batch, which_type = which_type)
          loss = loss_fn(score_model, variable, condition, marginal_prob_std = marginal_prob_std_fn, is_train = False)
          avg_loss_val += loss * output_batch.shape[0]
          num_items += output_batch.shape[0]
    avg_loss_val/= num_items
  
    if epoch%50 == 0 and which_data in ["shear_layer_rpb", "wave"]:
      
      if condition is not None:
        condition = condition[:4]
      '''
      samples = ode_sampler(best_model,
                            marginal_prob_std_fn,
                            diffusion_coeff_fn,
                            condition = condition,
                            batch_size = 2,
                            atol=5e-4,
                            rtol=5e-4,
                            device=device,
                            dimension = (dim, config.s, config.s))
      '''
      batch_size_plot = 4
      samples = Euler_Maruyama_sampler(best_model,
                                        marginal_prob_std_fn,
                                        diffusion_coeff_fn,
                                        condition,
                                        batch_size=batch_size_plot,
                                        num_steps=256,
                                        device=device,
                                        dimension = (dim, config.s, config.s),
                                        eps=1e-3)
      fig = plot_prediction(batch_size_plot, (1,1), input_batch[:batch_size_plot].reshape(-1,in_dim,config.s,config.s), output_batch[:batch_size_plot].reshape(-1,out_dim,config.s,config.s), samples, f"{dir_path}/train_plot_ep_{epoch}.png")
      wandb.log({f"fig_train/train_plot_ep_{epoch+1}": wandb.Image(fig)})
      plt.close()

    ema.restore()

    if avg_loss_val<best_avg_loss:
      best_avg_loss = avg_loss_val
      checkpoint = {
            'model_state_dict': score_model.state_dict(),
            'ema_state_dict': ema.shadow,
            'optimizer_state_dict': optimizer.state_dict(),
          }
      best_model = copy.deepcopy(score_model)
      torch.save(checkpoint, f"{dir_path}/ckpt_diffusion_gencfd_{tag}.pth")
    
    dict_wandb = dict()
    dict_wandb['train/loss'] = avg_loss
    dict_wandb['train/epoch'] = epoch + 1
    if epoch>=10 and avg_loss>0.5:
      model = best_model
      dict_wandb['train/reset'] =  1
    else:
      dict_wandb['train/reset'] =  0
    dict_wandb['train/val_loss'] =  avg_loss_val
    dict_wandb['train/best_val_loss'] =  best_avg_loss
    wandb.log(dict_wandb, step = epoch + 1)

    tqdm_epoch.set_description('Train: {:.5f} Val: {:.5f}'.format(avg_loss, avg_loss_val))
