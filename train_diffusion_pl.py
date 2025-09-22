import torch
import os
import functools
import tqdm
import argparse
import copy
import wandb
import matplotlib.pyplot as plt

from diffusion.loss_fn import loss_fn, loss_fn_denoised
from diffusion.variance_fn import marginal_prob_std_1, diffusion_coeff_1, marginal_prob_std_2, diffusion_coeff_2
import time 

from GenCFD.model.lightning_wrap.pl_conditional_denoiser import PreconditionedDenoiser_pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from utils.utils_data import get_loader, save_data, load_data, read_cli_regression, save_errors, read_cli_diffusion_gencfd, select_variable_condition

os.environ["WANDB__SERVICE_WAIT"] = "300"

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="load parameters for training")
  params = read_cli_diffusion_gencfd(parser).parse_args()

  if params.config is None:
    config = params
  else:
    config = argparse.Namespace(**load_data(params.config))

  device = config.device
  tag = config.tag
  is_log_uniform = config.is_log_uniform
  log_uniform_frac = config.log_uniform_frac
  is_exploding = config.is_exploding
  sigma =  config.sigma
  ema_param = config.ema_param

  if is_exploding:
    marginal_prob_std_fn = functools.partial(marginal_prob_std_2, sigma_min = 0.001, sigma_max=sigma, device = device)
    diffusion_coeff_fn = functools.partial(diffusion_coeff_2, sigma_min = 0.001, sigma_max=sigma, device = device)
  else:
    marginal_prob_std_fn = functools.partial(marginal_prob_std_1, sigma=sigma, device = device)
    diffusion_coeff_fn = functools.partial(diffusion_coeff_1, sigma=sigma, device = device)

  is_skip = config.skip

  which_data = config.which_data
  which_type = config.which_type
  weight = None

  if not is_skip:
    loss = functools.partial(loss_fn, is_log_uniform = is_log_uniform, log_uniform_frac = log_uniform_frac)
  else:
    loss = functools.partial(loss_fn_denoised, is_log_uniform = is_log_uniform, log_uniform_frac = log_uniform_frac, weighting = "edm", sigma_data = 0.5, consistent_weight = 0.1, channel_weight = weight)
    
  tag = config.tag # Just a random string to be added to the model name
  workdir = f"trained_models" #Defualt, change of needed

  config_arch = load_data(config.config_arch)
  config_train = vars(config)
  config_train["workdir"] = workdir

  
  if which_type == "yx":
    dim = config.out_dim
    dim_cond = config.in_dim
  if which_type == "xy":
    dim_cond = config.out_dim
    dim = config.in_dim
  elif which_type == "x":
    dim = config.in_dim
    dim_cond = 0
  elif which_type == "x&y":
    dim = config.out_dim + config.in_dim
    dim_cond = 0
  
  model = PreconditionedDenoiser_pl(dim = dim, 
                                    dim_cond = dim_cond,
                                    loss_fn = loss,
                                    marginal_prob_std_fn = marginal_prob_std_fn,
                                    diffusion_coeff_fn = diffusion_coeff_fn,
                                    config_train = config_train,
                                    config_arch = config_arch,
                                    )

  run = wandb.init(entity="your_wandb", 
                  project=config.wandb_project_name, 
                  name=tag + config.wandb_run_name, 
                  config=config)

  
  if not os.path.exists(workdir):
    os.makedirs(workdir)
  save_data(vars(config), f"{workdir}/param_diffusion_gencfd_{tag}.json")
  
  lr_monitor = LearningRateMonitor(logging_interval='step')
  checkpoint_callback = ModelCheckpoint(dirpath = workdir+"/model", monitor='val_loss_ema', save_top_k=-1)
  logger = TensorBoardLogger(save_dir=workdir, version=1, name="logs")


  if "mix" in which_data:
    check_interval = 0.2
  else:
    check_interval = 1.0

  trainer = Trainer(devices = -1,
                  max_epochs = config.epochs,
                  callbacks = [checkpoint_callback],
                  logger=logger,
                  val_check_interval = check_interval,
                  num_sanity_val_steps = 0)
  trainer.fit(model)
  trainer.validate(model)


  