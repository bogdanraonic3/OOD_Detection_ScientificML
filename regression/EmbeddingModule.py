import torch.nn as nn
import torch
import einops
from typing import Any, Sequence, Callable
import torch.nn.functional as F

Tensor = torch.Tensor

class FourierEmbedding(nn.Module):
    """Fourier embedding."""

    def __init__(
        self,
        dims: int = 64,
        max_freq: float = 2e4,
        projection: bool = True,
        act_fun: Callable[[Tensor], Tensor] = F.silu,
        max_val: float = 1e6,  # for numerical stability
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ):
        super(FourierEmbedding, self).__init__()

        self.dims = dims
        self.max_freq = max_freq
        self.projection = projection
        self.act_fun = act_fun
        self.max_val = max_val
        self.dtype = dtype
        self.device = device
        
        logfreqs = torch.linspace(
            0,
            torch.log(
                torch.tensor(self.max_freq, device=self.device)
            ),
            self.dims // 2,
            device=self.device,
        )

        # freqs are constant and scaled with pi!
        const_freqs = torch.pi * torch.exp(logfreqs)[None, :]  # Shape: (1, dims//2)

        # Store freqs as a non-trainable buffer also to ensure device and dtype transfers
        self.register_buffer("const_freqs", const_freqs)

        if self.projection:
            self.lin_layer1 = nn.Linear(
                self.dims, 2 * self.dims, device=self.device
            )
            self.lin_layer2 = nn.Linear(
                2 * self.dims, self.dims,device=self.device
            )

    def forward(self, x):
        assert len(x.shape) == 1, "Input tensor must be 1D"
        # Use the registered buffer const_freqs
        x_proj = self.const_freqs * x[:, None]

        # x_proj is now a 2D tensor
        x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        # clamping values to avoid running into numerical instability!
        x_proj = torch.clamp(x_proj, min=-self.max_val, max=self.max_val)

        if self.projection:
            x_proj = self.lin_layer1(x_proj)
            x_proj = self.act_fun(x_proj)
            x_proj = self.lin_layer2(x_proj)

        return x_proj

class AdaptiveScale(nn.Module):
    """Adaptively scale the input based on embedding.

    Conditional information is projected to two vectors of length c where c is
    the number of channels of x, then x is scaled channel-wise by first vector
    and offset channel-wise by the second vector.

    This method is now standard practice for conditioning with diffusion models,
    see e.g. https://arxiv.org/abs/2105.05233, and for the
    more general FiLM technique see https://arxiv.org/abs/1709.07871.
    """

    def __init__(
        self,
        emb_channels: int,
        input_channels: int,
        dim: int = 2,
        act_fun: Callable[[Tensor], Tensor] = F.silu,
        device: str = "cuda"
    ):
        super(AdaptiveScale, self).__init__()

        self.emb_channels = emb_channels
        self.input_channels = input_channels
        self.dim = dim
        self.act_fun = act_fun

        # self.affine = None
        self.affine = nn.Linear(
            in_features=emb_channels,
            out_features=input_channels * 2,
            device=device
        )
        #default_init(.0)(self.affine.weight)
        torch.nn.init.zeros_(self.affine.bias)
        torch.nn.init.zeros_(self.affine.weight)

    def forward(self, x, emb):
        """Adaptive scaling applied to the channel dimension.

        Args:
          x: Tensor to be rescaled.
          emb: Embedding values that drives the rescaling.

        Returns:
          Rescaled tensor plus bias
        """
        scale_params = self.affine(self.act_fun(emb))
        scale, bias = torch.chunk(scale_params, 2, dim=-1)
        
        if self.dim>1:
            scale = scale[(...,) + (None,) * self.dim]
            bias = bias[(...,) + (None,) * self.dim]
        else:
            scale = scale.unsqueeze(1)
            bias = bias.unsqueeze(1)
        return x * (scale + 1) + bias