import torch.nn as nn
import torch
import numpy as np
import einops
from einops import rearrange
from einops.layers.torch import Rearrange
from regression.EmbeddingModule import FourierEmbedding, AdaptiveScale
from regression.GeneralModule_pl import GeneralModel_pl
import torch.nn.functional as F


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
            self.norm = nn.GroupNorm(4, channels)
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


#--------------------------------------
# Define 2d Spectral Convolution:
#--------------------------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


#--------------------------------------
# Define 2d Local Convolution:
#--------------------------------------
class LocalConv2d(nn.Module):
    def __init__(self,
                 dim,
                 conv_filters = [3, 5], #In what order should we apply convolutions?
                 
                 ):
        super(LocalConv2d, self).__init__()

        self.conv_filters = conv_filters
        self.conv_list   = nn.ModuleList([nn.Conv2d(dim, dim, k, padding=(k-1)//2) for k in conv_filters])
        
    def forward(self, x):

        for l, conv in enumerate(self.conv_list):
            x = conv(x)
            if l < len(self.conv_filters) - 1:
                x = nn.SiLU()(x)

        return x

#--------------------------------------
# Define Lift/Project Layers:
#--------------------------------------
class LiftProject(nn.Module):
    def __init__(self, 
                 in_dim,
                 out_dim,
                 hidden_dim = 128):
        super().__init__()
        
        """
           in_dim:        input dimension
           out_dim:       output dimension
           hidden_dim:    intermediate duimension
        """
        
        self.r = nn.Linear(in_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.r(x)
        x = x.permute(0, 3, 1, 2)

        x = nn.SiLU()(x)
        
        x = x.permute(0, 2, 3, 1)
        x = self.q(x)
        x = x.permute(0, 3, 1, 2)

        return x
        
#--------------------------------------
# Define 2d, time-dependent FNO:
#--------------------------------------
class FNO2d(nn.Module):
    def __init__(self,
                 in_dim, 
                 out_dim,
                 n_layers = 4,
                 width = 64,
                 modes = (16,16),
                 hidden_dim = 128,
                 padding = None,
                 include_grid = True,
                 is_time = True,
                 is_conv = True
                 ):
        super(FNO2d, self).__init__()

        self.modes1 , self.modes2 = modes
        self.width = width
        self.n_layers = n_layers
        self.padding = padding
        self.include_grid = include_grid
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.is_time = is_time

        
        if self.include_grid in [1, True]:
            
            self.r = LiftProject(in_dim = self.in_dim + 2,
                                 out_dim = self.width,
                                 hidden_dim = self.hidden_dim)
        else:
            self.r = LiftProject(in_dim = self.in_dim,
                                 out_dim = self.width,
                                 hidden_dim = self.hidden_dim)
        
        self.q = LiftProject(in_dim = self.width,
                             out_dim = self.out_dim,
                             hidden_dim = self.hidden_dim)
        

        self.linear_list   = nn.ModuleList([nn.Linear(self.width, self.width) for _ in range(self.n_layers)])
        self.spectral_list = nn.ModuleList([SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(self.n_layers)])
       
        if is_conv:
            self.conv_list = nn.ModuleList([LocalConv2d(self.width, conv_filters = [3,5]) for _ in range(self.n_layers)])
        else:
            self.conv_list = nn.ModuleList([nn.Identity() for _ in range(self.n_layers)])

        if self.is_time:
            self.normalization_list = nn.ModuleList([FILM(self.width) for _ in range(self.n_layers)])
        else:
            self.normalization_list = nn.ModuleList([nn.Identity() for _ in range(self.n_layers)])

    def get_grid(self, samples, res, device):
        size_x = size_y = res
        samples = samples
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([samples, size_y, 1, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, size_y, 1, 1).repeat([samples, 1, size_x, 1])
        grid = torch.cat((gridy, gridx), dim=-1)

        return grid.permute(0, 3, 1, 2).to(device)

    def forward(self, x, time): 
        '''
        x is of shape (B, C, X, Y)
        time is of shape (B, )
        '''

        if self.include_grid == 1:
            grid = self.get_grid(x.shape[0], x.shape[-1], device = x.device)
            x = torch.cat((grid, x), 1)

        x = self.r(x)
                
        if self.padding is not None and self.padding>0: 
            x1_padding =  self.padding
            x2_padding =  self.padding
            x = F.pad(x, [x1_padding, x1_padding, x2_padding, x2_padding])

        for k, (s, l) in enumerate(zip(self.spectral_list, self.linear_list)):
            
            x1 = s(x)
            x2 = l(x.permute(0, 2, 3, 1))
            x2 = x2.permute(0, 3, 1, 2)
            x = x1 + x2
            if k != self.n_layers - 1:
                x = nn.SiLU()(x)
            x = self.normalization_list[k](x, time)
            x = self.conv_list[k](x)
            

        del x1
        del x2
        
        if self.padding is not None and self.padding>0:
            x = x[..., x1_padding:-x1_padding, x2_padding:-x2_padding]            
        x = self.q(x)

        return x

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams}')

        return nparams

class FNOModel_pl(GeneralModel_pl):
    def __init__(self,  
                in_dim, 
                out_dim,
                loss_fn,
                config_train: dict = dict(),
                config_arch: dict = dict()
                ):
        super().__init__(in_dim, out_dim, config_train)

        self.loss_fn = loss_fn
        self.model = FNO2d(in_dim = in_dim, 
                        out_dim = out_dim,
                        n_layers = config_arch["n_layers"],
                        width = config_arch["width"],
                        modes = (config_arch["modes"], config_arch["modes"]),
                        hidden_dim = 128,
                        padding = config_arch["padding"],
                        include_grid = True,
                        is_time = config_train["is_time"],
                        is_conv = config_arch["is_conv"])
