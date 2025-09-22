from GenCFD.model.lightning_wrap.pl_wrapper import GeneralModel_pl
from typing import Callable
from GenCFD import model
from diffusion.model import EMA
import torch
from torch.optim.swa_utils import AveragedModel
import time
import copy

class PreconditionedDenoiser_pl(GeneralModel_pl):
    def __init__(self,  
                dim : int, 
                dim_cond : int,
                loss_fn: Callable,
                marginal_prob_std_fn: Callable,
                diffusion_coeff_fn: Callable,
                config_train: dict = dict(),
                config_arch: dict = dict(),
                is_inference: bool = False
                ):
        super().__init__(dim, 
                        dim_cond, 
                        config_train)
        
        '''
            --- Must be defined ---

            self.model = None
            self.ema = None
            self.marginal_prob_std_fn = None
            self.diffusion_coeff_fn = None
            self.loss_fn = None

        '''

        self.marginal_prob_std_fn = marginal_prob_std_fn
        self.diffusion_coeff_fn = diffusion_coeff_fn
        self.loss_fn = loss_fn

        #if "cifar" in config_train["which_data"]:
        #    use_attention = False
        #else:
        #    use_attention = True

        use_attention = True
        if "skip" not in config_train or not config_train["skip"]:
                
            self.model = model.PreconditionedDenoiser(in_channels = dim + dim_cond, # Conditioning thus stacked input and output
                                                    out_channels = dim,
                                                    spatial_resolution = (config_train['s'],config_train['s']),
                                                    time_cond = config_train['is_time'],
                                                    num_channels= tuple(config_arch["channels"]),
                                                    downsample_ratio=(2,)*len(config_arch["channels"]),
                                                    marginal_prob_std = self.marginal_prob_std_fn,
                                                    num_blocks=config_arch["num_blocks"],
                                                    noise_embed_dim=config_arch["noise_embed_dim"],
                                                    output_proj_channels=config_arch["proj_channels"],
                                                    input_proj_channels=config_arch["proj_channels"],
                                                    padding_method='zeros',
                                                    dropout_rate=0.0,
                                                    use_attention= use_attention,
                                                    use_position_encoding=True,
                                                    num_heads=config_arch["num_heads"],
                                                    normalize_qk=False,
                                                    dtype=torch.float32,
                                                    device= config_train["device"],
                                                    buffer_dict=dict(),
                                                    sigma_data=0.5
                                                    )
        else:

            self.model = model.PreconditionedDenoiser_skip(in_channels = dim + dim_cond, # Conditioning thus stacked input and output
                                                            out_channels = dim,
                                                            spatial_resolution = (config_train['s'],config_train['s']),
                                                            time_cond = config_train['is_time'],
                                                            num_channels= tuple(config_arch["channels"]),
                                                            downsample_ratio=(2,)*len(config_arch["channels"]),
                                                            marginal_prob_std = self.marginal_prob_std_fn,
                                                            num_blocks=config_arch["num_blocks"],
                                                            noise_embed_dim=config_arch["noise_embed_dim"],
                                                            output_proj_channels=config_arch["proj_channels"],
                                                            input_proj_channels=config_arch["proj_channels"],
                                                            padding_method='zeros',
                                                            dropout_rate=0.0,
                                                            use_attention=use_attention,
                                                            use_position_encoding=True,
                                                            num_heads=config_arch["num_heads"],
                                                            normalize_qk=False,
                                                            dtype=torch.float32,
                                                            device= config_train["device"],
                                                            buffer_dict=dict(),
                                                            sigma_data=0.5
                                                            )

        self.ema_model = AveragedModel(self.model, avg_fn=self.ema_update)
        # Freeze EMA model parameters
        for param in self.ema_model.parameters():
            param.requires_grad = False  # Prevent updates from optimizer

        if is_inference:
            self.best_model_ema = copy.deepcopy(self.ema_model)