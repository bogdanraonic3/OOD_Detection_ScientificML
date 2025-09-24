import torch
import os
import functools
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from dataloader.dataloader import CIFAR10_Dataset, Wave2d_Dataset
import tqdm
import argparse
import numpy as np
from regression.CNOClassification import CNOClassificationModel_pl

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from utils.utils_data import get_loader, save_data, load_data, read_cli_regression, save_errors
import time 
import json
import torch.nn as nn
import torch.optim as optim
from regression.loss_fn import relative_lp_loss_fn

import wandb

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="load parameters for training")
  params = read_cli_regression(parser).parse_args()

  if params.config is None:
    config = params
  else:
    config = argparse.Namespace(**load_data(params.config))

  device = config.device
  which_exp = config.which_exp
  which_data = config.which_data
  tag = f"{config.tag}_{which_exp}_{config.N_train}_{config.only_x}"
  

  workdir = f"/path_to_workdir/"
  if not os.path.exists(workdir):
    os.makedirs(workdir)
  save_data(vars(config), f"{workdir}/param_regression_{tag}.json")

  config_arch = load_data(config.config_arch)
  config_train = vars(config)
  config_train["workdir"] = workdir

  loss = functools.partial(relative_lp_loss_fn, p=1)

  if config.only_x:
    dim = config.dim
  else:
    dim = 2*config.dim

  model = CNOClassificationModel_pl(in_dim = dim, 
                      out_dim = dim,
                      loss_fn = loss,
                      config_train = config_train,
                      config_arch = config_arch)
  model.model.print_size()

  model = model.to("cuda")

  n_epochs = config.epochs

  train_loader = get_loader(which_data = 'baseline_class',
                            which_type = "train",
                            N_samples = config.N_train,
                            batch_size = config.batch_size,
                            baseline_folder = config.baseline_folder,
                            baseline_only_x = config.only_x,
                            N_max = config.N_max,
                            num_workers=1)
  
  valid_loader = get_loader(which_data = "baseline_class",
                            which_type = "val",
                            N_samples = config.N_train,
                            batch_size = config.batch_size,
                            baseline_folder = config.baseline_folder,
                            baseline_only_x = config.only_x,
                            N_max = config.N_max,
                            num_workers = 1)

  # specify loss function
  criterion = nn.NLLLoss()

  # specify optimizer
  optimizer = optim.Adam(model.parameters())

  epochs_no_improve = 0
  max_epochs_stop = 50
  valid_loss_min = np.Inf

  for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    train_acc = 0
    valid_acc = 0

    ###################
    # train the model #
    ###################
    model.train()

    for ii, (data, target) in enumerate(train_loader):
            
      data, target = data.cuda(), target.cuda()
      
      if len(target.shape)>1:
        target = target[:,0]

      # clear the gradients of all optimized variables
      optimizer.zero_grad()
      # forward pass: compute predicted outputs by passing inputs to the model
      output = model(data)
      #####print(data.shape, target.shape, output.shape)
      #print(output.shape)
      # calculate the batch loss
      loss = criterion(output, target)
      # backward pass: compute gradient of the loss with respect to model parameters
      loss.backward()
      # perform a single optimization step (parameter update)
      optimizer.step()
      # update training loss
      train_loss += loss.item()

      # Calculate accuracy
      ps = torch.exp(output)
      topk, topclass = ps.topk(1, dim = 1)
      equals = topclass == target.view(*topclass.shape)
      accuracy = torch.mean(equals.type(torch.FloatTensor))
      train_acc += accuracy.item()

    print(f'Epoch: {epoch} \t {100 * ii / len(train_loader):.2f}% complete.', end = '\r')

    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, target in valid_loader:
      # move tensors to GPU if CUDA is available
      data, target = data.cuda(), target.cuda()
      if len(target.shape)>1:
        target = target[:,0]

      # forward pass: compute predicted outputs by passing inputs to the model
      output = model(data)
      # calculate the batch loss
      loss = criterion(output, target)
      # update average validation loss 
      valid_loss += loss.item()

      # Calculate accuracy
      ps = torch.exp(output)
      topk, topclass = ps.topk(1, dim = 1)
      equals = topclass == target.view(*topclass.shape)
      accuracy = torch.mean(equals.type(torch.FloatTensor))
      valid_acc += accuracy.item()

    # calculate average losses
    train_loss = train_loss/len(train_loader)
    valid_loss = valid_loss/len(valid_loader)

    train_acc = train_acc/len(train_loader)
    valid_acc = valid_acc/len(valid_loader)

    valid_loss = (valid_loss*0.2 +  (1 - valid_acc)*0.8)/2.
    # print training/validation statistics 
    print('\nEpoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    print(f'Training Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%')

    # save model if validation loss has decreased
    
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), f"{workdir}/model-cifar.pt")
        epochs_no_improve = 0
        valid_loss_min = valid_loss
    else:
        epochs_no_improve += 1
        print(f'{epochs_no_improve} epochs with no improvement.')
        if epochs_no_improve >= max_epochs_stop:
            print('Early Stopping')
            break
    print(valid_loss_min)