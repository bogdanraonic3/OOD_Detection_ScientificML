import torch.nn as nn
import torch
import pytorch_lightning as pl
import numpy as np
import wandb
import time
import matplotlib.pyplot as plt
import copy
import math

from utils.utils_data import get_loader, select_variable_condition

from visualization.plot import plot_prediction
from diffusion.lr_schedulers import CosineLinearWarmupCustomScheduler
from torch.optim.swa_utils import AveragedModel
from diffusion.sampler import Euler_Maruyama_sampler
from torch.optim.swa_utils import update_bn

class GeneralModel_pl(pl.LightningModule):
    def __init__(self,  
                dim : int, 
                dim_cond : int,
                config_train: dict = dict()
                ):
        super(GeneralModel_pl, self).__init__()

        '''
            The following variables must be defined in the child class:
        '''
        self.model = None
        self.ema_model = None
        self.marginal_prob_std_fn = None
        self.diffusion_coeff_fn = None
        self.loss_fn = None
    
        self.dim_cond = dim_cond
        self.dim = dim
        self.peak_lr = config_train["peak_lr"]
        self.end_lr = config_train["end_lr"]
        self.warmup_epochs = config_train["warmup_epochs"]
        self.epochs = config_train["epochs"]
        self.batch_size = config_train["batch_size"]
        self.N_train = config_train["N_train"]
        self.ood_share = config_train["ood_share"]

        self.is_time = config_train["is_time"]
        self.is_masked = config_train["is_masked"]
        self.max_num_time_steps = config_train["max_num_time_steps"]
        self.time_step_size = config_train["time_step_size"]
        self.fix_input_to_time_step = config_train["fix_input_to_time_step"]
        self.allowed_transitions = config_train["allowed_transitions"]

        self._curr_epoch = -1
        self._cur_step   = 0
        self._wandb_aggregation = 32
        

        self._which_benchmark = config_train["which_data"]
        if "mix" in  self._which_benchmark or "cifar" in self._which_benchmark:
            self._plot_epoch = 1
        else:
            self._plot_epoch = 5
        

        self._which_type = config_train["which_type"]
        self._res = config_train["s"]
        self._workdir = config_train["workdir"]


        self.best_val_loss = 1000.0
        self.best_val_loss_ema = 1000.0
        self.best_model_ema = None

        self.ema_param = config_train["ema_param"]

        if "skip" in config_train:
            self.is_skip = config_train["skip"]
        else:
            self.is_skip = False

        if "ns" in self._which_benchmark or "eul" in self._which_benchmark:
            self.interval = "step"
            _num_gpus = torch.cuda.device_count()//4+1
            self.lr_step_per_epoch = math.floor(self.N_train * self.max_num_time_steps / (self.batch_size * _num_gpus)) + 1
        else:
            self.interval = "epoch"
            self.lr_step_per_epoch = 1

    def forward(self, x, x_cond, t_diffusion, t = None):        
        return self.model(x, x_cond, t_diffusion, t)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.end_lr)
        scheduler = CosineLinearWarmupCustomScheduler(optimizer, 
                                                    warmup_epochs = self.lr_step_per_epoch*self.warmup_epochs, 
                                                    total_epochs = self.lr_step_per_epoch*self.epochs, 
                                                    peak_lr = self.peak_lr, 
                                                    end_lr = self.end_lr)
        
        return [optimizer], [{"scheduler": scheduler, "interval": self.interval}]
    
    def ema_update(self, avg_model_param, model_param, num_averaged):
        # Exponential moving average update
        return self.ema_param * avg_model_param + (1 - self.ema_param) * model_param

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # Standard optimizer step
        optimizer.step(closure=optimizer_closure)

        # Update EMA weights
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data = self.ema_update(ema_param.data, param.data, None)

        wandb.log({"lr":optimizer.param_groups[-1]['lr']}, step=self._cur_step-1)

    def training_step(self, batch, batch_idx):
        
        '''
            unpack:
        '''
        if self.is_masked:
            if self.is_time:
                t_batch, input_batch, output_batch, mask = batch
            else:
                input_batch, output_batch, mask = batch
                t_batch = None
        else:
            if self.is_time:
                t_batch, input_batch, output_batch = batch
            else:
                input_batch, output_batch = batch
                t_batch = None
            
            mask = None
        
        '''
            compute loss:
        '''

        batch = output_batch.shape[0]

        variable, condition, mask = select_variable_condition(input_batch, output_batch, which_type = self._which_type, mask = mask)
        loss = self.loss_fn(self, variable, condition, t_batch, self.marginal_prob_std_fn, is_train = True, mask = mask)
        
        '''
            wandb logs:
        '''

        if batch_idx == 0:
            self._curr_epoch+=1
            self.avg_loss = loss.detach().cpu().item() * batch
            self.num_train_items = batch
        else:
            self.avg_loss += loss.detach().cpu().item() * batch
            self.num_train_items += batch

        if self._cur_step % self._wandb_aggregation == 0:
            self.log('loss', loss, prog_bar=True)
            dict_log = {'train/loss': loss.detach().cpu().item(), 'train/step': self._cur_step, 'train/epoch': self._curr_epoch}
            
            if batch_idx >= 10:
                loss_log = self.avg_loss/self.num_train_items
                dict_log['train/loss_avg'] =  loss_log
            wandb.log(dict_log, step=self._cur_step)

            self.avg_loss = 0.0
            self.num_train_items = 0
        self._cur_step+=1
        
        return loss

    def on_train_epoch_end(self):
        train_dataloader = self.trainer.train_dataloader
        update_bn(train_dataloader, self.ema_model)
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        
        '''
            unpack:
        '''
        if self.is_masked:
            if self.is_time:
                t_batch, input_batch, output_batch, mask = batch
            else:
                input_batch, output_batch, mask = batch
                t_batch = None
        else:
            if self.is_time:
                t_batch, input_batch, output_batch = batch
            else:
                input_batch, output_batch = batch
                t_batch = None
            mask = None
        '''
            compute loss:
        '''
        variable, condition, mask = select_variable_condition(input_batch, output_batch, which_type = self._which_type, mask = mask)
        loss = self.loss_fn(self, variable, condition, t_batch, self.marginal_prob_std_fn, is_train = False)
        loss_ema = self.loss_fn(self.ema_model, variable, condition, t_batch, self.marginal_prob_std_fn, is_train = False)

        '''
            wandb logs:
        '''

        if batch_idx == 0:
            self.loss_val = loss * output_batch.shape[0]
            self.loss_val_ema = loss_ema * output_batch.shape[0]
            self.num_val_items = output_batch.shape[0]

            if self._curr_epoch%self._plot_epoch == 0 and self.best_model_ema is not None:
                
                batch_size_plot = 16
                if condition is not None:
                    condition = condition[:batch_size_plot]
                    
                #self.best_model_ema.to(self.model.device)
                if self.is_skip:
                    steps = 128
                else:
                    steps = 256
                
                if "cifar" not in self._which_benchmark:
                    guidance_condition = input_batch[:batch_size_plot], (0, input_batch.shape[1])
                else:
                    guidance_condition = output_batch[:batch_size_plot], (input_batch.shape[1], -1)
                
                if "mnist" in self._which_benchmark:
                    guidance_condition = None
                

                samples = Euler_Maruyama_sampler(self.best_model_ema,
                                                self.marginal_prob_std_fn,
                                                self.diffusion_coeff_fn,
                                                condition,
                                                t_batch = t_batch,
                                                batch_size=batch_size_plot,
                                                num_steps=steps,
                                                device=self.model.device,
                                                dimension = (self.dim, self._res, self._res),
                                                eps=1e-3,
                                                is_skip = self.is_skip,
                                                guidance_condition = None)
                
                is_cifar = "cifar" in self._which_benchmark

                fig = plot_prediction(batch_size_plot, (1,1), input_batch[:batch_size_plot], output_batch[:batch_size_plot], samples, f"{self._workdir}/train_plot_ep_{self._curr_epoch}.png", is_cifar = is_cifar)
                wandb.log({f"fig_train/train_plot_ep_{self._curr_epoch+1}": wandb.Image(fig)})
                plt.close()
                #self.best_model_ema.to("cpu")
                del samples
                del condition
                torch.cuda.empty_cache()
        else:
            self.loss_val += loss * output_batch.shape[0]
            self.loss_val_ema += loss_ema * output_batch.shape[0]
            self.num_val_items += output_batch.shape[0]

        return loss
    
    def on_validation_epoch_end(self):

        val_loss = self.loss_val / self.num_val_items
        val_loss_ema = self.loss_val_ema / self.num_val_items
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        self.log("val_loss_ema", val_loss_ema, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        
        # Save the best loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        
        if val_loss_ema < self.best_val_loss_ema:
            self.best_val_loss_ema = val_loss_ema
            del self.best_model_ema
            torch.cuda.empty_cache()
            self.best_model_ema = copy.deepcopy(self.ema_model)#.to("cpu")
            for param in self.best_model_ema.parameters():
                param.requires_grad = False

        self.log("best_val_loss",self.best_val_loss,on_step=False, on_epoch=True, sync_dist=True)
        wandb.log({'val/best_val_loss': self.best_val_loss, 'val/val_loss':val_loss}, step=self._cur_step)
        wandb.log({'val/best_val_loss_ema': self.best_val_loss_ema, 'val/val_loss_ema':val_loss_ema}, step=self._cur_step)

        torch.cuda.empty_cache()

        return {"val_loss": val_loss, "val_loss_ema": val_loss_ema} 
    

    def train_dataloader(self):

        _rel_time = True
        if self._which_type == "x":
            _rel_time = False
        
        loader = get_loader(which_data = self._which_benchmark,
                            which_type = "train",
                            N_samples = self.N_train,
                            ood_share = self.ood_share,
                            batch_size = self.batch_size,
                            masked_input = self.is_masked,
                            is_time = self.is_time,
                            max_num_time_steps = self.max_num_time_steps,
                            time_step_size = self.time_step_size,
                            fix_input_to_time_step = self.fix_input_to_time_step,
                            allowed_transitions = self.allowed_transitions,
                            rel_time = _rel_time)
        
        

        return loader
    
    def val_dataloader(self):
        
        _rel_time = True
        if self._which_type == "x":
            _rel_time = False

        val_loader = get_loader(which_data = self._which_benchmark,
                            which_type = "val",
                            N_samples = 1,
                            ood_share = self.ood_share,
                            batch_size = self.batch_size,
                            masked_input = self.is_masked,
                            is_time = self.is_time,
                            max_num_time_steps = self.max_num_time_steps,
                            time_step_size = self.time_step_size,
                            fix_input_to_time_step = self.fix_input_to_time_step,
                            allowed_transitions = self.allowed_transitions,
                            rel_time = _rel_time)
        
        return val_loader