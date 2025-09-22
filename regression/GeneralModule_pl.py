import torch.nn as nn
import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from diffusion.lr_schedulers import CosineLinearWarmupCustomScheduler

import wandb
import time
import math

import matplotlib.pyplot as plt

from visualization.plot import plot_prediction
from utils.utils_data import get_loader
import torch.nn.functional as F

# Define GeneralModel_pl in pytorch_lightning framework.

class GeneralModel_pl(pl.LightningModule):
    def __init__(self,  
                in_dim, 
                out_dim,
                config_train: dict = dict()
                ):
        super(GeneralModel_pl, self).__init__()

        '''
            -- For example, in the child class, there must me smth like this:
            self.model = FNO2d(in_dim = in_dim, 
                             out_dim = out_dim,
                             n_layers = n_layers,
                             width = width,
                             modes = modes,
                             hidden_dim = hidden_dim,
                             use_conv = use_conv,
                             conv_filters = conv_filters,
                             padding = padding,
                             include_grid = include_grid,
                             is_time = is_time)
        '''

        self.model = None # TO BE DEFINED IN THE CHILD CLASS !!!
        self.loss_fn = None
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        #--------------------
        # Training parameters
        #--------------------
        
        self.peak_lr = config_train["peak_lr"]
        self.end_lr = config_train["end_lr"]
        self.warmup_epochs = config_train["warmup_epochs"]
        self.epochs = config_train["epochs"]
        self.batch_size = config_train["batch_size"]
        self.N_train = config_train["N_train"]

        self.is_time = config_train["is_time"]
        self.is_masked = config_train["is_masked"]
        self.max_num_time_steps = config_train["max_num_time_steps"]
        self.time_step_size = config_train["time_step_size"]
        self.fix_input_to_time_step = config_train["fix_input_to_time_step"]
        self.allowed_transitions = config_train["allowed_transitions"]

        self._curr_epoch = -1
        self._cur_step   = 0
        self._wandb_aggregation = 32
        self._plot_epoch = 1
        self.best_val_loss = 1000.0

        self._which_benchmark = config_train["which_data"]
        self._res = config_train["s"]
        self._workdir = config_train["workdir"]

        if "ns" in self._which_benchmark or "eul" in self._which_benchmark:
            self.interval = "step"
            _num_gpus = torch.cuda.device_count()//4+1
            self.lr_step_per_epoch = math.floor(self.N_train * self.max_num_time_steps / (self.batch_size * _num_gpus)) + 1
        else:
            self.interval = "epoch"
            self.lr_step_per_epoch = 1

        """ 
          - If we traing the model to predict different physical quantities (velocity + pressure + ...)
          - For example, if the variables are [rho, vx, vy, p], then "separate_dim" should be [1,2,1]
          - 2 means that vx and vy are grouped together!
        """
        # Are the physical quantities separated in the loss function?

        '''
        if  ("separate" in self.config) and self.config["separate"]:
            self._is_separate = True
            if "separate_dim" in self.config:
                self._separate_dim = self.config["separate_dim"]
            else:
                self._separate_dim = [out_dim]
        else:
            self._is_separate = False
        '''
         
        self._spatial_mask = False
        self.is_ft = False
        '''
        # Are we interested in all the channels or we want to predict just a few of them and ignore others?
        self._is_masked = "is_masked" in self.config and self.config["is_masked"] is not None

        # Is there a spatial mask, like in the airfoil benchmark?
        self._spatial_mask = "spatial_mask" in self.config and self.config["spatial_mask"] is not None and self.config["spatial_mask"]
        self._spatial_mask = self._spatial_mask or self._which_benchmark == "airfoil"
        '''
        
    def forward(self, x, time = None):        
        return self.model(x, time)

    def configure_optimizers(self):

        if not self.is_ft:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.end_lr)
            scheduler = CosineLinearWarmupCustomScheduler(optimizer, 
                                                        warmup_epochs = self.lr_step_per_epoch*self.warmup_epochs, 
                                                        total_epochs = self.lr_step_per_epoch*self.epochs, 
                                                        peak_lr = self.peak_lr, 
                                                        end_lr = self.end_lr)

            return [optimizer], [{"scheduler": scheduler, "interval": self.interval}]
        else:

            params_1 = [param for name, param in self.named_parameters() if (("project" in name) or ("lift" in name)) or ("film" in name) or ("adapt" in name)]
            params_2 = [param for name, param in self.named_parameters() if ("project" not in name) and ("lift" not in name) and ("film" not in name) and ("adapt" not in name)]

            """if self.reinit_ft:
                lr2 = self.end_lr/5.
            else:"""
            
            lr2 = self.end_lr
            
            print(" ")
            print("----------")  
            print("FINETUNNING - ", lr2, self.end_lr) 
            print("----------")  
            print(" ")
            
            optimizer = torch.optim.AdamW([{'params': params_1},
                                           {'params': params_2,
                                            'lr': lr2}],
                                           lr=self.end_lr)     

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    def training_step(self, batch, batch_idx):
        

        optimizer = self.trainer.optimizers[0]  # if you have one optimizer
        current_lr = optimizer.param_groups[0]['lr']
        #print(f"Step {batch_idx}, LR: {current_lr}")
        #time.sleep(1000)
        if self.is_masked is not None:
            t_batch, input_batch, output_batch, masked_dim = batch
        else:
            if self.is_time:
                t_batch, input_batch, output_batch = batch
            else:
                input_batch, output_batch = batch
                t_batch = None
            masked_dim = None
        
        t_batch = t_batch.type(torch.float32)
        #time.sleep(10)

        # Predict:
        output_pred_batch = self(input_batch, t_batch)
        
        # If spatial mask, as in airfoil, mask it
        if self._spatial_mask:
            output_pred_batch[input_batch==1] = 1.0
            output_batch[input_batch==1] = 1.0

        loss = self.loss_fn(output_batch, 
                            output_pred_batch,
                            reduction = True,
                            mask = masked_dim)

        '''
            wandb logs:
        '''
        if batch_idx == 0:
            self._curr_epoch+=1
            self.avg_loss = loss.detach().cpu().item() * output_batch.shape[0]
            self.num_train_items = output_batch.shape[0]
        else:
            self.avg_loss += loss.detach().cpu().item() * output_batch.shape[0]
            self.num_train_items += output_batch.shape[0]

        if self._cur_step % self._wandb_aggregation == 0:
            self.log('loss', loss, prog_bar=True)
            dict_log = {'train/loss': loss.detach().cpu().item(), 'train/step': self._cur_step, 'train/epoch': self._curr_epoch}
            
            if batch_idx >= 10:
                loss_log = self.avg_loss/self.num_train_items
                dict_log['train/loss_avg'] =  loss_log
            if self.global_rank ==0:
                wandb.log(dict_log, step=self._cur_step)

            self.avg_loss = 0.0
            self.num_train_items = 0
        self._cur_step+=1
        
        return loss
    
    def on_train_epoch_end(self):
        torch.cuda.empty_cache()
    
    def validation_step(self, batch, batch_idx):
        
        if self.is_masked is not None:
            t_batch, input_batch, output_batch, masked_dim = batch
        else:
            if self.is_time:
                t_batch, input_batch, output_batch = batch
            else:
                input_batch, output_batch = batch
                t_batch = None
            masked_dim = None
        
        t_batch = t_batch.type(torch.float32)

        # Predict:
        output_pred_batch = self(input_batch, t_batch)
        
        if batch_idx==0 and self._curr_epoch%self._plot_epoch == 0 and self.model is not None:
            batch_size_plot = min(20, input_batch.shape[0])
            #if "mix" in self._which_benchmark:
            #    batch_size_plot = 20

            fig = plot_prediction(batch_size_plot, (1,1), input_batch[:batch_size_plot], output_batch[:batch_size_plot], output_pred_batch[:batch_size_plot], f"{self._workdir}/train_plot_ep_{self._curr_epoch}.png")
            if self.global_rank == 0:
                wandb.log({f"fig_train/train_plot_ep_{self._curr_epoch+1}": wandb.Image(fig)})
            plt.close()
            #self.best_model_ema.to("cpu")
            torch.cuda.empty_cache()

        # If spatial mask, as in airfoil, mask it
        if self._spatial_mask:
            output_pred_batch[input_batch==1] = 1.0
            output_batch[input_batch==1] = 1.0

        ########output_pred_batch = input_batch
        loss = self.loss_fn(output_batch, 
                            output_pred_batch,
                            reduction = False,
                            mask = masked_dim)
        
        # Save validation errs:

        if batch_idx==0:
            self.validation_times = t_batch
            self.validation_errs = loss

            '''
            if self._curr_epoch%self._plot_epoch == 0 and self.model is not None:
                batch_size_plot = 8
                fig = plot_prediction(batch_size_plot, (1,1), input_batch[:batch_size_plot], output_batch[:batch_size_plot], output_pred_batch[:batch_size_plot], f"{self._workdir}/train_plot_ep_{self._curr_epoch}.png")
                wandb.log({f"fig_train/train_plot_ep_{self._curr_epoch+1}": wandb.Image(fig)})
                plt.close()
                #self.best_model_ema.to("cpu")
                torch.cuda.empty_cache()
            '''
        else:
            self.validation_times = torch.cat((self.validation_times, t_batch))
            self.validation_errs = torch.cat((self.validation_errs, loss))
        
        return loss

    def on_validation_epoch_end(self):

        _max_time = torch.max(self.validation_times)
        _validation_errs_last = self.validation_errs[self.validation_times==_max_time]

        _min_time = torch.min(self.validation_times)
        _validation_errs_first = self.validation_errs[self.validation_times==_min_time]

        median_loss = torch.median(self.validation_errs).item()
        mean_loss = torch.mean(self.validation_errs).item() 
        median_loss_last = torch.median(_validation_errs_last).item()
        mean_loss_last = torch.mean(_validation_errs_last).item() 
        median_loss_first = torch.median(_validation_errs_first).item()
        mean_loss_first = torch.mean(_validation_errs_first).item() 
        
        self.log("median_val_loss", median_loss, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        self.log("val_loss",  mean_loss, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        
        # Save the best loss
        if mean_loss < self.best_val_loss:
            self.best_val_loss = mean_loss
        
        self.log("best_val_loss",self.best_val_loss,on_step=False, on_epoch=True,sync_dist=True)
        
        if self.global_rank == 0:
            wandb.log({'val/best_val_loss': self.best_val_loss, 'val/mean_val_all': mean_loss, 'val/med_val_all': median_loss, 'val/med_val_last':median_loss_last, 'val/mean_val_last':mean_loss_last, 'val/mean_val_first':mean_loss_first, 'val/median_val_first':median_loss_first}, step=self._cur_step)
        
        return {"val_loss": mean_loss,} 

    def train_dataloader(self):
        
        _rel_time = True

        train_loader = get_loader(which_data = self._which_benchmark,
                                which_type = "train",
                                N_samples = self.N_train,
                                batch_size = self.batch_size,
                                masked_input = self.is_masked,
                                is_time = self.is_time,
                                max_num_time_steps = self.max_num_time_steps,
                                time_step_size = self.time_step_size,
                                fix_input_to_time_step = self.fix_input_to_time_step,
                                allowed_transitions = self.allowed_transitions,
                                rel_time = _rel_time)
        return train_loader

    def val_dataloader(self):

        _rel_time = True

        val_loader = get_loader(which_data = self._which_benchmark,
                            which_type = "val",
                            N_samples = 1,
                            batch_size = self.batch_size,
                            masked_input = self.is_masked,
                            is_time = self.is_time,
                            max_num_time_steps = self.max_num_time_steps,
                            time_step_size = self.time_step_size,
                            fix_input_to_time_step = self.fix_input_to_time_step,
                            allowed_transitions = self.allowed_transitions,
                            rel_time = _rel_time)

        return val_loader