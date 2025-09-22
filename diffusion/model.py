# The code for the diffusion is adapted from a tutorial
# https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class MLP(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.linear = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.linear(x)[..., None, None]

class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, 
              marginal_prob_std, 
              dim,
              dim_cond=0,
              channels=[64, 64, 128, 256, 512], 
              embed_dim=256):
        
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))

    self.n_layers = len(channels)
    self.dim = dim
    self.dim_cond = dim_cond
    
    self.encoder_channels = [self.dim + self.dim_cond] + channels
    self.encoder_conv  = [nn.Conv2d(self.encoder_channels[0], self.encoder_channels[1], 3, stride=1, padding = 1, bias=False)] 
    self.encoder_conv  = self.encoder_conv + [nn.Conv2d(self.encoder_channels[i], self.encoder_channels[i+1], 3, stride=2, padding = 1, bias=False) for i in range(1, self.n_layers)]
    self.encoder_conv  = nn.ModuleList(self.encoder_conv)
    self.encoder_mlp   = nn.ModuleList([MLP(embed_dim, self.encoder_channels[i+1]) for i in range(self.n_layers)])
    ###self.encoder_group = nn.ModuleList([nn.BatchNorm2d(self.encoder_channels[i+1]) for i in range(self.n_layers)])

    self.encoder_conv_inv = nn.ModuleList([nn.Conv2d(self.encoder_channels[i+1], self.encoder_channels[i+1], 3, padding = 1) for i in range(self.n_layers)])
    self.encoder_group_inv = nn.ModuleList([nn.BatchNorm2d(self.encoder_channels[i+1]) for i in range(self.n_layers)])

    self.decoder_channels_in = self.encoder_channels[::-1][:-1]
    self.decoder_channels_out = self.encoder_channels[::-1][1:]
    self.decoder_channels_out[-1] = embed_dim
    for i in range(1, self.n_layers):
        self.decoder_channels_in[i]*=2
    
    self.decoder_deconv  = [nn.ConvTranspose2d(self.decoder_channels_in[0],self.decoder_channels_out[0], 3, stride=2, padding=1, output_padding=1, bias=False)]
    for i in range(1, self.n_layers - 1):
        self.decoder_deconv = self.decoder_deconv + [nn.ConvTranspose2d(self.decoder_channels_in[i],self.decoder_channels_out[i], 3, stride=2, padding=1, output_padding=1, bias=False)]
    self.decoder_deconv = self.decoder_deconv + [nn.ConvTranspose2d(self.decoder_channels_in[-1],self.decoder_channels_out[-1], 3, stride=1, padding = 1)]
    self.decoder_deconv = nn.ModuleList(self.decoder_deconv)
    self.decoder_mlp  = nn.ModuleList([MLP(embed_dim, self.decoder_channels_out[i]) for i in range(self.n_layers-1)])
    ###self.decoder_group = nn.ModuleList([nn.BatchNorm2d(self.decoder_channels_out[i]) for i in range(self.n_layers-1)])
    
    self.decoder_conv = [nn.Conv2d(self.decoder_channels_out[i], self.decoder_channels_out[i], 3, padding = 1, bias=False) for i in range(self.n_layers-1)]
    self.decoder_conv = self.decoder_conv + [nn.Conv2d(self.decoder_channels_out[-1], dim, 3, padding = 1, bias=False)]
    self.decoder_conv = nn.ModuleList(self.decoder_conv)
    self.decoder_conv_group = nn.ModuleList([nn.BatchNorm2d(self.decoder_channels_out[i]) for i in range(self.n_layers)])
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std

  def forward(self, x, x_cond, t):
    # Obtain the Gaussian random feature embedding for t
    embed = self.act(self.embed(t))

    skip = []

    if x_cond is not None:
      x = torch.cat((x, x_cond), dim=1)
    for i in range(self.n_layers):
      x = self.encoder_conv[i](x)
      x = x + self.encoder_mlp[i](embed)
      x = self.act(x)
      x = self.encoder_conv_inv[i](x)
      x = self.encoder_group_inv[i](x)
      x = self.act(x)

      if i<self.n_layers-1:
        skip.append(x)
    
    x = self.decoder_deconv[0](x)
    x = x + self.decoder_mlp[0](embed)
    x = self.act(x)      
    x = self.decoder_conv[0](x)
    x = self.decoder_conv_group[0](x)
    x = self.act(x)

    for i in range(1, self.n_layers-1):
      x = self.decoder_deconv[i](torch.cat([x, skip[-i]], dim=1))
      x = x + self.decoder_mlp[i](embed)
      x = self.act(x)      
      x = self.decoder_conv[i](x)
      x = self.decoder_conv_group[i](x)
      x = self.act(x)

    x = self.decoder_deconv[-1](torch.cat([x, skip[0]], dim=1))
    x = self.decoder_conv[-1](x)
    
    # Normalize output
    x = x / self.marginal_prob_std(t)[:, None, None, None]
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

    print(f'Total number of model parameters: {nparams}')

    return nparams


import torch

class EMA:
    def __init__(self, model, decay):
        """
        Initialize EMA class to manage exponential moving average of model parameters.
        
        Args:
            model (torch.nn.Module): The model for which EMA will track parameters.
            decay (float): Decay rate, typically a value close to 1, e.g., 0.999.
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Store initial parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        Update shadow parameters with exponential decay.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """
        Apply shadow (EMA) parameters to model.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """
        Restore original model parameters from backup.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]