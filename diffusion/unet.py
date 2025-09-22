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

class ScoreUNet(nn.Module):
    def __init__(self,  
                 in_dim,                    # Number of input channels.
                 N_layers,                  # Number of (D) or (U) blocks in the network
                 N_res = 1,                 # Number of (R) blocks per level (except the neck)
                 N_res_neck = 6,            # Number of (R) blocks in the neck
                 channel_multiplier = 32,   # How the number of channels evolve?
                ):
        
        super(ScoreUNet, self).__init__()
        
        self.N_layers = int(N_layers)        
        self.lift_dim = channel_multiplier//2         
        self.out_dim  = out_dim
        self.channel_multiplier = channel_multiplier        
        
        self.encoder_features = [self.lift_dim]
        for i in range(self.N_layers):
            self.encoder_features.append(2 ** i *   self.channel_multiplier)
        
        self.decoder_features_in = self.encoder_features[1:]
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()

        for i in range(1, self.N_layers):
            self.decoder_features_in[i] = 2*self.decoder_features_in[i] #Pad the outputs of the resnets
        
        self.inv_features = self.decoder_features_in
        self.inv_features.append(self.encoder_features[0] + self.decoder_features_out[-1])

        self.encoder_sizes = []
        self.decoder_sizes = []
        for i in range(self.N_layers + 1):
            self.encoder_sizes.append(latent_size // 2 ** i)
            self.decoder_sizes.append(latent_size_out // 2 ** (self.N_layers - i))
