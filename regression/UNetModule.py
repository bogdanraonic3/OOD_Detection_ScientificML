from diffusion.model import MLP
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from regression.GeneralModule_pl import GeneralModel_pl


class FILM(torch.nn.Module):
    def __init__(self, 
                channels,
                intermediate = 128):
        super(FILM, self).__init__()
        self.channels = channels
        
        self.inp2lat_sacale = nn.Linear(in_features=1, out_features=intermediate,bias=True)
        self.lat2scale = nn.Linear(in_features=intermediate, out_features=channels)

        self.inp2lat_bias = nn.Linear(in_features=1, out_features=intermediate,bias=True)
        self.lat2bias = nn.Linear(in_features=intermediate, out_features=channels)
        
        self.inp2lat_sacale.weight.data.fill_(0)
        self.inp2lat_sacale.weight.data.fill_(0)
        self.lat2scale.weight.data.fill_(0)
        self.lat2scale.weight.data.fill_(0)

        self.inp2lat_bias.weight.data.fill_(0)
        self.inp2lat_bias.bias.data.fill_(0)
        self.lat2bias.weight.data.fill_(0)
        self.lat2bias.bias.data.fill_(0)

        if channels is not None:
            self.norm = nn.BatchNorm2d(channels)
        else:
            self.norm = nn.Identity()
        
    def forward(self, x, time):

        x = self.norm(x)
        time = time.reshape(-1,1).type_as(x)
        scale     = self.lat2scale(self.inp2lat_sacale(time))
        bias      = self.lat2bias(self.inp2lat_bias(time))
        scale     = scale.unsqueeze(2).unsqueeze(3)
        scale     = scale.expand_as(x)
        bias  = bias.unsqueeze(2).unsqueeze(3).expand_as(x)
        
        return x * (scale + 1. ) + bias 

#----------------------------------

class ResidualBlock(nn.Module):
    def __init__(self,
                channels
                ):
        super(ResidualBlock, self).__init__()

        self.channels = channels #important for time conditioning
        
        self.convolution1 = torch.nn.Conv2d(in_channels  = self.channels, 
                                            out_channels = self.channels, 
                                            kernel_size  = 5, 
                                            stride       = 1, 
                                            padding      = 2)
        self.convolution2 = torch.nn.Conv2d(in_channels = self.channels, out_channels=self.channels, 
                                            kernel_size=3, stride = 1, 
                                            padding = 1)
        
        self.act  = nn.GELU()
        self.norm1 = nn.GroupNorm(min(max(channels // 4, 1), 32), channels)
        self.norm2 = nn.GroupNorm(min(max(channels // 4, 1), 32), channels)

    def forward(self, x, time):
        out = self.convolution1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.convolution2(out)
        out = self.norm2(out)
        x = x + out
        return x

class UNet(nn.Module):

  def __init__(self, 
                in_dim,
                out_dim,
                channels=[32, 64, 128, 256],
                n_res = 2,
                is_time = True):
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.n_layers = len(channels)
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.is_time = is_time

    self.encoder_channels = [self.in_dim] + channels
    self.encoder_conv  = [nn.Conv2d(self.encoder_channels[0], self.encoder_channels[1], 3, stride=1, padding = 1, bias=False)] 
    self.encoder_conv  = self.encoder_conv + [nn.Conv2d(self.encoder_channels[i], self.encoder_channels[i+1], 3, stride=2, padding = 1, bias=False) for i in range(1, self.n_layers)]
    self.encoder_conv  = nn.ModuleList(self.encoder_conv)
    
    if not self.is_time:
      self.encoder_group = nn.ModuleList([nn.BatchNorm2d(self.encoder_channels[i+1]) for i in range(self.n_layers)])
    else:
      self.encoder_group = nn.ModuleList([FILM(self.encoder_channels[i+1]) for i in range(self.n_layers)])

    self.decoder_channels_in = self.encoder_channels[::-1][:-1]
    self.decoder_channels_out = self.encoder_channels[::-1][1:]
    self.decoder_channels_out[-1] = self.out_dim

    for i in range(1, self.n_layers):
        self.decoder_channels_in[i]*=2
    self.decoder_conv  = [nn.ConvTranspose2d(self.decoder_channels_in[0],self.decoder_channels_out[0], 3, stride=2, padding=1, output_padding=1, bias=False)]
    for i in range(1, self.n_layers - 1):
        self.decoder_conv = self.decoder_conv + [nn.ConvTranspose2d(self.decoder_channels_in[i],self.decoder_channels_out[i], 3, stride=2, padding=1, output_padding=1, bias=False)]
    self.decoder_conv = self.decoder_conv + [nn.ConvTranspose2d(self.decoder_channels_in[-1],self.decoder_channels_in[-1], 3, stride=1, padding = 1)]
    self.decoder_conv = nn.ModuleList(self.decoder_conv)
    
    self.n_res = n_res
    self.res_nets = nn.ModuleList([ResidualBlock(channels = self.encoder_channels[-1]) for i in range(n_res)])

    if not self.is_time:
      self.decoder_group = nn.ModuleList([nn.BatchNorm2d(self.decoder_channels_out[i]) for i in range(self.n_layers-1)])
    else:
      self.decoder_group = nn.ModuleList([FILM(self.decoder_channels_out[i]) for i in range(self.n_layers-1)])


    self.conv_last = torch.nn.Conv2d(in_channels = self.decoder_channels_in[-1],
                                    out_channels= self.decoder_channels_out[-1],
                                    kernel_size = 3,
                                    padding     = 1)

    # The swish activation function
    self.act = nn.GELU()

  def forward(self, x, time):

    skip = []

    for i in range(self.n_layers):
      x = self.encoder_conv[i](x)
      if not self.is_time:
        x = self.encoder_group[i](x)
      else:
        x = self.encoder_group[i](x, time)
      x = self.act(x)
      if i<self.n_layers-1:
        skip.append(x)
    
    #print(x.shape)
    for i in range(self.n_res):
      x = self.res_nets[i](x, time)

    x = self.decoder_conv[0](x)
    if not self.is_time:
      x = self.decoder_group[0](x)
    else:
      x = self.decoder_group[0](x, time)
    x = self.act(x)

    for i in range(1, self.n_layers-1):
      x = self.decoder_conv[i](torch.cat([x, skip[-i]], dim=1))
      if not self.is_time:
        x = self.decoder_group[i](x)
      else:
        x = self.decoder_group[i](x, time)
      x = self.act(x)

    x = self.decoder_conv[-1](torch.cat([x, skip[0]], dim=1))
    x = self.conv_last(x)
    return x

class UNetModel_pl(GeneralModel_pl):
    def __init__(self,  
                in_dim, 
                out_dim,
                loss_fn,
                config_train: dict = dict(),
                config_arch: dict = dict()
                ):
        super().__init__(in_dim, out_dim, config_train)

        self.loss_fn = loss_fn
        self.model = UNet(in_dim = in_dim,
                          out_dim = out_dim,
                          channels = config_arch["channels"],
                          is_time = config_train["is_time"])