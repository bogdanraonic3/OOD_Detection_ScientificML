# Implementation of the filters is borrowed from paper "Alias-Free Generative Adversarial Networks (StyleGAN3)" https://nvlabs.github.io/stylegan3/
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited. 

from typing import Any, Sequence, Callable

import torch.nn as nn
import torch
import einops
from CNO2d_original_version.training.filtered_networks import LReLu, CNO_SiLu
from CNO2d_original_version.debug_tools import format_tensor_size
import pytorch_lightning as pl
import numpy as np

import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn as nn
import copy
import time
from regression.GeneralModule_pl import GeneralModel_pl
from regression.ViTModule import SimpleViT
from regression.EmbeddingModule import FourierEmbedding, AdaptiveScale

Tensor = torch.Tensor


#-------------------------------------------------
# CNO:
#-------------------------------------------------

#Depending on in_size, out_size, the CNOBlock can be:
#   -- (D) Block
#   -- (U) Block
#   -- (I) Block

class CNOBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                in_size,
                out_size,
                cutoff_den = 2.0001,
                conv_kernel = 3,
                filter_size = 6,
                lrelu_upsampling = 2,
                half_width_mult  = 0.8,
                activation = 'cno_lrelu',
                is_norm = False,
                is_time = True,
                emb_channels = 128,
                device = "cuda"
                ):
        super(CNOBlock, self).__init__()
        
        self.in_channels  = in_channels
        self.out_channels = out_channels #important for conditioning
        self.in_size      = in_size
        self.out_size     = out_size
        self.conv_kernel  = conv_kernel
        
        #---------- Filter properties -----------
        self.citically_sampled = False #We use w_c = s/2.0001 --> NOT critically sampled

        if cutoff_den == 2.0:
            self.citically_sampled = True
        
        self.in_cutoff  = self.in_size / cutoff_den
        self.out_cutoff = self.out_size / cutoff_den
        
        self.in_halfwidth  =  half_width_mult*self.in_size - self.in_size / cutoff_den
        self.out_halfwidth = half_width_mult*self.out_size - self.out_size / cutoff_den
        
        #-----------------------------------------

        # We apply Conv -> GN -> Activation
        # Up/Downsampling happens inside Activation
        
        pad = (self.conv_kernel-1)//2
        self.convolution = torch.nn.Conv2d(in_channels   = self.in_channels, 
                                            out_channels = self.out_channels, 
                                            kernel_size  = self.conv_kernel, 
                                            padding      = pad)

        if is_norm:
            self.norm  = nn.GroupNorm(min(max(self.out_channels // 4, 1), 32), out_channels)
        else:
            self.norm = nn.Identity()
        
        if activation == "cno_lrelu":
            self.act  = LReLu(in_channels          = self.out_channels,
                                    out_channels          = self.out_channels,                   
                                    in_size               = self.in_size,                       
                                    out_size              = self.out_size,                       
                                    in_sampling_rate      = self.in_size,               
                                    out_sampling_rate     = self.out_size,             
                                    in_cutoff             = self.in_cutoff,                     
                                    out_cutoff            = self.out_cutoff,                  
                                    in_half_width         = self.in_halfwidth,                
                                    out_half_width        = self.out_halfwidth,              
                                    filter_size           = filter_size,       
                                    lrelu_upsampling      = lrelu_upsampling,
                                    is_critically_sampled = self.citically_sampled,
                                    use_radial_filters    = False)
        
        elif activation == "silu":
            self.act  = CNO_SiLu(in_channels          = self.out_channels,
                                        out_channels          = self.out_channels,                   
                                        in_size               = self.in_size,                       
                                        out_size              = self.out_size,                       
                                        in_sampling_rate      = self.in_size,               
                                        out_sampling_rate     = self.out_size)
        else:
            raise ValueError("Please specify different activation function")
        
        # Time conditioning:
        self.is_time = is_time
    
        if is_time:
            '''
            self.film = FILM(out_channels,
                            dim = nl_dim,
                            s = self.in_size)
            '''
            self.film = AdaptiveScale(emb_channels = emb_channels,
                                    input_channels = out_channels,
                                    device = device)
        
            
    def forward(self, x, emb = None):
        x = self.convolution(x)
        x = self.norm(x)
        if emb is not None:
            x = self.film(x, emb)
        x = self.act(x)
        return x

# Contains CNOBlock -> Convolution -> BN
class LiftProjectBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                in_size,
                out_size,
                latent_dim = 128,
                cutoff_den = 2.0001,
                conv_kernel = 3,
                filter_size = 6,
                lrelu_upsampling = 2,
                half_width_mult  = 0.8,
                activation = 'cno_lrelu',
                is_norm = False,
                is_time = True,
                emb_channels = 128,
                device = "cuda"
                ):
        super(LiftProjectBlock, self).__init__()
    
        self.out_channels = out_channels #important for time conditioning
    
        self.inter_CNOBlock = CNOBlock(in_channels       = in_channels,
                                        out_channels     = latent_dim,
                                        in_size          = in_size,
                                        out_size         = out_size,
                                        cutoff_den       = cutoff_den,
                                        conv_kernel      = conv_kernel,
                                        filter_size      = filter_size,
                                        lrelu_upsampling = lrelu_upsampling,
                                        half_width_mult  = half_width_mult,
                                        activation       = activation,
                                        is_time = is_time,
                                        is_norm = is_norm,
                                        emb_channels= emb_channels,
                                        device = device)
        
        pad = (conv_kernel-1)//2
        self.convolution = torch.nn.Conv2d(in_channels   = latent_dim, 
                                            out_channels = out_channels, 
                                            kernel_size  = conv_kernel, 
                                            stride       = 1, 
                                            padding      = pad)

    def forward(self, x, emb = None):
        x = self.inter_CNOBlock(x, emb)
        x = self.convolution(x)
        return x
        

# Residual Block containts:
# Convolution -> BN -> Activation -> Convolution -> BN -> SKIP CONNECTION

class ResidualBlock(nn.Module):
    def __init__(self,
                channels,
                size,
                cutoff_den = 2.0001,
                conv_kernel = 3,
                filter_size = 6,
                lrelu_upsampling = 2,
                half_width_mult  = 0.8,
                activation = 'cno_lrelu',
                is_norm = True,
                is_time = True,
                emb_channels = 128,
                device = "cuda"
                ):
        super(ResidualBlock, self).__init__()

        self.channels = channels #important for time conditioning
        self.size  = size
        self.conv_kernel = conv_kernel
        
        #---------- Filter properties -----------
        self.citically_sampled = False #We use w_c = s/2.0001 --> NOT critically sampled

        if cutoff_den == 2.0:
            self.citically_sampled = True
        self.cutoff  = self.size / cutoff_den        
        self.halfwidth =  half_width_mult*self.size - self.size / cutoff_den
        
        #-----------------------------------------
        
        pad = (self.conv_kernel-1)//2
        self.convolution1 = torch.nn.Conv2d(in_channels  = self.channels, 
                                            out_channels = self.channels, 
                                            kernel_size  = self.conv_kernel, 
                                            stride       = 1, 
                                            padding      = pad)
        self.convolution2 = torch.nn.Conv2d(in_channels = self.channels, out_channels=self.channels, 
                                            kernel_size=self.conv_kernel, stride = 1, 
                                            padding = pad)
        
        if activation == "cno_lrelu":

            self.act  = LReLu(in_channels          = self.channels, #In _channels is not used in these settings
                                    out_channels          = self.channels,                   
                                    in_size               = self.size,                       
                                    out_size              = self.size,                       
                                    in_sampling_rate      = self.size,               
                                    out_sampling_rate     = self.size,             
                                    in_cutoff             = self.cutoff,                     
                                    out_cutoff            = self.cutoff,                  
                                    in_half_width         = self.halfwidth,                
                                    out_half_width        = self.halfwidth,              
                                    filter_size           = filter_size,       
                                    lrelu_upsampling      = lrelu_upsampling,
                                    is_critically_sampled = self.citically_sampled,
                                    use_radial_filters    = False)
        
        elif activation == "silu":

            self.act = CNO_SiLu(in_channels           = self.channels, #In _channels is not used in these settings
                                out_channels          = self.channels,                   
                                in_size               = self.size,                       
                                out_size              = self.size,                       
                                in_sampling_rate      = self.size,               
                                out_sampling_rate     = self.size)
        else:
            raise ValueError("Please specify different activation function")
        
        # Time conditioning:
        self.is_time = is_time
        if  self.is_time == True:
            self.film1 = AdaptiveScale(emb_channels = emb_channels,
                                    input_channels = channels,
                                    device = device)
            self.film2 = AdaptiveScale(emb_channels = emb_channels,
                                    input_channels = channels,
                                    device = device)

        if is_norm:
            self.norm1 = nn.GroupNorm(min(max(channels // 4, 1), 32), channels)
            self.norm2 = nn.GroupNorm(min(max(channels // 4, 1), 32), channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x, emb = None):
        out = self.convolution1(x)
        out = self.norm1(out)
        if emb is not None:
            out = self.film1(out, emb)
        
        out = self.act(out)
        out = self.convolution2(out)
        out = self.norm2(out)

        if emb is not None:
            out = self.film2(out, emb)

        x = x + out
        
        return x


# CNO NETWORK:
class CNO(nn.Module):
    def __init__(self,  
                in_dim,                    # Number of input channels.
                in_size,                   # Input spatial size
                N_layers,                  # Number of (D) or (U) blocks in the network
                N_res = 1,                 # Number of (R) blocks per level (except the neck)
                N_res_neck = 6,            # Number of (R) blocks in the neck
                channel_multiplier = 32,   # How the number of channels evolve?
                conv_kernel=3,             # Size of all the kernels
                cutoff_den = 2.0001,       # Filter property 1.
                filter_size=6,             # Filter property 2.
                lrelu_upsampling = 2,      # Filter property 3.
                half_width_mult  = 0.8,    # Filter property 4.
                out_dim = 1,               # Target dimension
                out_size = 1,              # If out_size is 1, Then out_size = in_size. Else must be int
                expand_input = False,      # Start with original in_size, or expand it (pad zeros in the spectrum)
                latent_lift_proj_dim = 128,# Intermediate latent dimension in the lifting/projection layer
                add_inv = True,            # Add invariant block (I) after the intermediate connections?
                activation = 'cno_lrelu',  # Activation function can be 'cno_lrelu' or 'lrelu'

                att_layers = None,
                att_blocks = 1,
                att_hidden_dim = 128, 
                att_heads = 4, 
                att_mlp_dim = 4 * 128, 
                att_dim_head = 128,

                is_time = True,
                emb_channels = 128,

                device = "cuda"
                ):
        
        super(CNO, self).__init__()

        ### Define the parameters & specifications ###        
        
        # Number od (D) & (U) Blocks
        self.N_layers = int(N_layers)
        
        # Input is lifted to the half on channel_multiplier dimension
        self.lift_dim = channel_multiplier//2         
        self.in_dim = in_dim
        self.out_dim = out_dim   
        
        #Should we add invariant layers in the decoder?
        self.add_inv = add_inv
        
        # The growth of the channels : d_e parametee
        self.channel_multiplier = channel_multiplier        
        
        ### Define evolution of the number features ###

        # How the features in Encoder evolve (number of features)
        self.encoder_features = [self.lift_dim]
        for i in range(self.N_layers):
            self.encoder_features.append(2 ** i *   self.channel_multiplier)

        #time.sleep(100)


        # How the features in Decoder evolve (number of features)
        self.decoder_features_in = self.encoder_features[1:]
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()

        for i in range(1, self.N_layers):
            self.decoder_features_in[i] = 2*self.decoder_features_in[i] #Pad the outputs of the resnets
        
        self.inv_features = self.decoder_features_in
        self.inv_features.append(self.encoder_features[0] + self.decoder_features_out[-1])
        
        ### Define evolution of sampling rates ###
        
        if not expand_input:
            latent_size = in_size # No change in in_size
        else:
            down_exponent = 2 ** N_layers
            latent_size = in_size - (in_size % down_exponent) + down_exponent # Jump from 64 to 72, for example
        
        #Are inputs and outputs of the same size? If not, how should the size of the decoder evolve?
        if out_size == 1:
            latent_size_out = latent_size
        else:
            if not expand_input:
                latent_size_out = out_size # No change in in_size
            else:
                down_exponent = 2 ** N_layers
                latent_size_out = out_size - (out_size % down_exponent) + down_exponent # Jump from 64 to 72, for example
        
        self.encoder_sizes = []
        self.decoder_sizes = []
        for i in range(self.N_layers + 1):
            self.encoder_sizes.append(latent_size // 2 ** i)
            self.decoder_sizes.append(latent_size_out // 2 ** (self.N_layers - i))
        
        '''
            Attention:
        '''
        if att_layers is not None:
            self.att_layers = [bool(x) for x in att_layers]
        else:
            self.att_layers = None

        if self.att_layers is not None:
            assert len(self.att_layers) == self.N_layers + 1
            self.attention_layers = []
            for i, a in enumerate(self.att_layers):
                if a:
                    _size = max(self.encoder_sizes[i]//16,1)

                    self.attention_layers.append(SimpleViT(image_size = self.encoder_sizes[i], 
                                                    patch_size = _size, 
                                                    dim = att_hidden_dim, 
                                                    depth = att_blocks, 
                                                    heads = att_heads, 
                                                    mlp_dim = min(4*att_hidden_dim, 512), 
                                                    channels = self.encoder_features[i], 
                                                    dim_head = att_dim_head,
                                                    is_time = False,
                                                    emb_channels = emb_channels,
                                                    device = device))
                else:
                    self.attention_layers.append(nn.Identity())
        else:
            self.attention_layers = [nn.Identity() for _ in range(self.N_layers + 1)]
        self.attention_layers = torch.nn.Sequential(*self.attention_layers)    

        ### Define Projection & Lift ###
    
        self.lift = LiftProjectBlock(in_channels  = in_dim,
                                    out_channels = self.encoder_features[0],
                                    in_size      = in_size,
                                    out_size     = self.encoder_sizes[0],
                                    latent_dim   = latent_lift_proj_dim,
                                    cutoff_den   = cutoff_den,
                                    conv_kernel  = conv_kernel,
                                    filter_size  = filter_size,
                                    lrelu_upsampling  = lrelu_upsampling,
                                    half_width_mult   = half_width_mult,
                                    activation = activation,
                                    is_time = False,
                                    emb_channels = emb_channels,
                                    device = device)
        
        _out_size = out_size
        if out_size == 1:
            _out_size = in_size
        
        self.project = LiftProjectBlock(in_channels  = self.encoder_features[0] + self.decoder_features_out[-1],
                                        out_channels = out_dim,
                                        in_size      = self.decoder_sizes[-1],
                                        out_size     = _out_size,
                                        latent_dim   = latent_lift_proj_dim,
                                        cutoff_den   = cutoff_den,
                                        conv_kernel  = conv_kernel,
                                        filter_size  = filter_size,
                                        lrelu_upsampling  = lrelu_upsampling,
                                        half_width_mult   = half_width_mult,
                                        activation = activation,
                                        is_time = is_time,
                                        is_norm = False,
                                        emb_channels = emb_channels,
                                        device = device)
        
        ### Define U & D blocks ###

        self.encoder         = nn.ModuleList([(CNOBlock(in_channels  = self.encoder_features[i],
                                                        out_channels = self.encoder_features[i+1],
                                                        in_size      = self.encoder_sizes[i],
                                                        out_size     = self.encoder_sizes[i+1],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        activation = activation,
                                                        is_time = is_time,
                                                        is_norm = True,
                                                        emb_channels = emb_channels,
                                                        device = device))                                  
                                                for i in range(self.N_layers)])
        
        # After the ResNets are executed, the sizes of encoder and decoder might not match (if out_size>1)
        # We must ensure that the sizes are the same, by aplying CNO Blocks
        self.ED_expansion     = nn.ModuleList([(CNOBlock(in_channels = self.encoder_features[i],
                                                        out_channels = self.encoder_features[i],
                                                        in_size      = self.encoder_sizes[i],
                                                        out_size     = self.decoder_sizes[self.N_layers - i],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        activation = activation,
                                                        is_time = is_time,
                                                        is_norm = True,
                                                        emb_channels = emb_channels,
                                                        device = device))                                  
                                                for i in range(self.N_layers + 1)])
        
        """        print(self.decoder_features_in[-1])
        print(self.decoder_features_out[-1])
        print(self.decoder_sizes[-1])
        print(self.decoder_sizes[-1])"""
        
        self.decoder         = nn.ModuleList([(CNOBlock(in_channels  = self.decoder_features_in[i],
                                                        out_channels = self.decoder_features_out[i],
                                                        in_size      = self.decoder_sizes[i],
                                                        out_size     = self.decoder_sizes[i+1],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        activation = activation,
                                                        is_time = is_time,
                                                        is_norm = True,
                                                        emb_channels = emb_channels,
                                                        device = device))                                  
                                                for i in range(self.N_layers)])
        
        self.decoder_inv    = nn.ModuleList([(CNOBlock(in_channels  =  self.inv_features[i],
                                                        out_channels = self.inv_features[i],
                                                        in_size      = self.decoder_sizes[i],
                                                        out_size     = self.decoder_sizes[i],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        activation = activation,
                                                        is_time=is_time,
                                                        is_norm = True,
                                                        emb_channels = emb_channels,
                                                        device = device))                               
                                                for i in range(self.N_layers + 1)])

        #### Define ResNets Blocks ###

        # Here, we define ResNet Blocks. 
        # We also define the BatchNorm layers applied BEFORE the ResNet blocks 
        
        # Operator UNet:
        # Outputs of the middle networks are patched (or padded) to corresponding sets of feature maps in the decoder 

        self.res_nets = []
        self.N_res = int(N_res)
        self.N_res_neck = int(N_res_neck)

        # Define the ResNet blocks & BatchNorm
        for l in range(self.N_layers):
            for i in range(self.N_res):
                self.res_nets.append(ResidualBlock(channels = self.encoder_features[l],
                                                    size     = self.encoder_sizes[l],
                                                    cutoff_den = cutoff_den,
                                                    conv_kernel = conv_kernel,
                                                    filter_size = filter_size,
                                                    lrelu_upsampling = lrelu_upsampling,
                                                    half_width_mult  = half_width_mult,
                                                    activation = activation,
                                                    is_time = is_time,
                                                    is_norm = True,
                                                    emb_channels = emb_channels,
                                                    device = device))
        for i in range(self.N_res_neck):
            self.res_nets.append(ResidualBlock(channels = self.encoder_features[self.N_layers],
                                                size     = self.encoder_sizes[self.N_layers],
                                                cutoff_den = cutoff_den,
                                                conv_kernel = conv_kernel,
                                                filter_size = filter_size,
                                                lrelu_upsampling = lrelu_upsampling,
                                                half_width_mult  = half_width_mult,
                                                activation = activation,
                                                is_time = is_time,
                                                is_norm = True,
                                                emb_channels = emb_channels,
                                                device = device))
        self.res_nets = torch.nn.Sequential(*self.res_nets)    
        
        if is_time:
            self.embedding = FourierEmbedding(dims = emb_channels, device = device)

    def forward(self, x, time = None):

        if time is not None:
            emb = self.embedding(time)
        else:
            emb = None
        ###time_emb = torch.exp(-time)[..., None, None]
        #Execute Lift ---------------------------------------------------------

        x = self.lift(x, None)
        skip = []
        # Execute Encoder -----------------------------------------------------
        for i in range(self.N_layers):
            
            if self.att_layers is not None and self.att_layers[i]:
                x = self.attention_layers[i](x, None)

            for j in range(self.N_res):
                x = self.res_nets[i*self.N_res + j](x, emb)
            
            skip.append(x)
            
            # Apply (D) block
            x = self.encoder[i](x, emb)   
        
        if self.att_layers is not None and self.att_layers[-1]:
            x = self.attention_layers[-1](x, None)
        
        # Apply the deepest ResNet (bottle neck)
        for j in range(self.N_res_neck):
            x = self.res_nets[-j-1](x, emb)
        
        # Execute Decoder -----------------------------------------------------
        for i in range(self.N_layers):
            
            # Apply (I) block (ED_expansion) & cat if needed
            if i == 0:
                x = self.ED_expansion[self.N_layers - i](x, emb) #BottleNeck : no cat
            else:
                x = torch.cat((x, self.ED_expansion[self.N_layers - i](skip[-i], emb)),1)
            if self.add_inv:
                x = self.decoder_inv[i](x, emb)
                
            # Apply (U) block
            x = self.decoder[i](x, emb)
        
        # Cat & Execute Projetion ---------------------------------------------
        
        x = torch.cat((x, self.ED_expansion[0](skip[0], emb)),1)
        x = self.project(x, emb)
        return x
    
    def get_n_params(self):
        pp = 0
        
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp
    
    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams


'''
    Wrapper:
'''

class CNOModel_pl(GeneralModel_pl):
    def __init__(self,  
                in_dim, 
                out_dim,
                loss_fn,
                config_train: dict = dict(),
                config_arch: dict = dict()
                ):
        super().__init__(in_dim, out_dim, config_train)

        self.loss_fn = loss_fn
        self.model = CNO(in_dim = in_dim,                
                        out_dim = out_dim,         
                        in_size = config_train["s"],                
                        N_layers = config_arch["cno_layers"],               
                        N_res = config_arch["cno_res"],   
                        N_res_neck = config_arch["cno_res_neck"],    
                        channel_multiplier = config_arch["cno_channels"],    
                        latent_lift_proj_dim = config_arch["cno_lift_dim"],                         
                        is_time = config_train["is_time"],
                        emb_channels = config_arch["cno_emb_dim"],
                        att_layers = config_arch["att_layers"],
                        att_blocks = config_arch["att_blocks"],
                        att_hidden_dim = config_arch["att_hidden_dim"], 
                        att_heads = config_arch["att_heads"], 
                        att_mlp_dim = config_arch["att_mlp_dim"], 
                        att_dim_head = config_arch["att_dim_head"],
                        device = config_train["device"],
                        activation = "silu"
                        )
