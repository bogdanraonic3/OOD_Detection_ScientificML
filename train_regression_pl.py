import torch
import os
import functools
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import tqdm
import argparse

from regression.UNetModule import UNetModel_pl
from regression.FNOModule import FNOModel_pl
from regression.CNOModule_pl import CNOModel_pl
from regression.ViTModulev2 import MultiVit3_pl, MultiVit2_pl, Vit3_pl

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from utils.utils_data import get_loader, save_data, load_data, read_cli_regression, save_errors
import time 
import json
import torch.nn as nn

from regression.loss_fn import relative_lp_loss_fn

import wandb

os.environ["WANDB__SERVICE_WAIT"] = "300"

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="load parameters for training")
  params = read_cli_regression(parser).parse_args()

  if params.config is None:
    config = params
  else:
    config = argparse.Namespace(**load_data(params.config))

  device = config.device
  tag = config.tag
  p = int(config.loss)
  which_data = config.which_data
  
  workdir = f"trained_models" #Defualt, change of needed

  if not os.path.exists(workdir):
    os.makedirs(workdir)
  save_data(vars(config), f"{workdir}/param_regression_{tag}.json")

  config_arch = load_data(config.config_arch)
  config_train = vars(config)
  config_train["workdir"] = workdir

  loss = functools.partial(relative_lp_loss_fn, p=p)

  if config.which_model == "cno":
    model = CNOModel_pl(in_dim = config.in_dim, 
                        out_dim = config.out_dim,
                        loss_fn = loss,
                        config_train = config_train,
                        config_arch = config_arch)
  elif config.which_model == "fno":
    model = FNOModel_pl(in_dim = config.in_dim, 
                        out_dim = config.out_dim,
                        loss_fn = loss,
                        config_train = config_train,
                        config_arch = config_arch)
  elif config.which_model == "unet":
    model = UNetModel_pl(in_dim = config.in_dim, 
                        out_dim = config.out_dim,
                        loss_fn = loss,
                        config_train = config_train,
                        config_arch = config_arch)
  elif config.which_model == "basic_vit3":
    model = Vit3_pl(in_dim = config.in_dim, 
                    out_dim = config.out_dim,
                    loss_fn = loss,
                    config_train = config_train,
                    config_arch = config_arch)
  
  run = wandb.init(entity="your_wandb", 
                  project=config.wandb_project_name, 
                  name=tag + config.wandb_run_name, 
                  config=config)

  if "mix" in which_data or "merra" in which_data or "pdegym" in which_data:
    check_interval = 0.1
  else:
    check_interval = 1.0
  
  checkpoint_callback = ModelCheckpoint(dirpath = workdir+"/model", monitor='val_loss', save_top_k=3)
  logger = TensorBoardLogger(save_dir=workdir, version=1, name="logs")
  
  if config.which_model == "cno":
    trainer = Trainer(devices = -1,
                    max_epochs = config.epochs,
                    callbacks = [checkpoint_callback],
                    strategy="ddp_find_unused_parameters_true", #IMPORTANT!!!
                    logger=logger,
                    val_check_interval = check_interval,
                    num_sanity_val_steps = 0)
  else:
    trainer = Trainer(devices = -1,
                    max_epochs = config.epochs,
                    callbacks = [checkpoint_callback],
                    strategy=DDPStrategy(find_unused_parameters=False), #IMPORTANT!!!
                    logger=logger,
                    val_check_interval = check_interval,
                    num_sanity_val_steps = 0)
  trainer.fit(model)
  trainer.validate(model)