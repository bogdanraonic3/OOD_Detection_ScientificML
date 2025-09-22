import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from regression.EmbeddingModule import FourierEmbedding
from regression.GeneralModule_pl import GeneralModel_pl

import einops
from typing import Any, Sequence, Callable
import torch.nn.functional as F
import time

Tensor = torch.Tensor

class AdaptiveScale(nn.Module):
    def __init__(
        self,
        emb_channels: int,
        input_channels: int,
        dim: int = 2,  # 2 = image, 1 = tokens
        act_fun: Callable[[Tensor], Tensor] = F.silu,
        device: str = "cuda"
    ):
        super(AdaptiveScale, self).__init__()

        self.emb_channels = emb_channels
        self.input_channels = input_channels
        self.dim = dim
        self.act_fun = act_fun

        self.affine = nn.Linear(
            in_features=emb_channels,
            out_features=input_channels * 2,
            device=device
        )

        torch.nn.init.zeros_(self.affine.bias)
        torch.nn.init.zeros_(self.affine.weight)

    def forward(self, x, emb):
        """
        Args:
            x: (B, N, C) for tokens OR (B, C, H, W) for images
            emb: (B, emb_channels)
        Returns:
            Rescaled tensor with same shape as x
        """
        scale_params = self.affine(self.act_fun(emb))
        scale, bias = torch.chunk(scale_params, 2, dim=-1)

        if self.dim == 1:  # Token-based tensor (B, N, C)
            scale = scale.unsqueeze(1)  # (B, 1, C)
            bias = bias.unsqueeze(1)    # (B, 1, C)
        else:  # Spatial tensor (e.g., B, C, H, W)
            scale = scale.view(scale.size(0), -1, *[1] * self.dim)  # (B, C, 1, 1, ...)
            bias = bias.view(bias.size(0), -1, *[1] * self.dim)

        #print(scale.shape, bias.shape, x.shape)
        x =  x * (scale + 1) + bias
        return x

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# === Multiscale Patch Embedding ===

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, dim, channels, pos_embedding = True):
        super().__init__()

        ih, iw = pair(image_size)
        ph, pw = pair(patch_size)
        assert ih % ph == 0 and iw % pw == 0, "Patch size must divide image size"
        num_patches = (ih // ph) * (iw // pw)
        patch_dim = channels * ph * pw

        self.rearrange = Rearrange("b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=ph, pw=pw)
        self.linear = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )
        self.num_patches = num_patches

        self.is_pos_embedding = pos_embedding
        if pos_embedding:
            self.pos_embedding = posemb_sincos_2d(
            h = ih // ph,
            w = iw // pw,
            dim = dim,
            ) 

    def forward(self, x):
        x = self.rearrange(x)
        x = self.linear(x)
        if self.is_pos_embedding:
            x = x + self.pos_embedding.to(x.device, dtype=x.dtype)
        return x

class Depatchify(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_channels):
        """
        Args:
            image_size: Tuple (H, W) of the full image size
            patch_size: Tuple (pH, pW) of the patch size
            in_dim:     Dimension of each patch embedding vector (i.e., ViT dim)
            out_channels: Number of channels in the reconstructed image
        """
        super().__init__()
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        self.in_dim = in_dim
        self.out_channels = out_channels

        ph, pw = self.patch_size
        patch_dim = ph * pw * out_channels

        # Project back from transformer dim to patch pixels
        self.project = nn.Linear(in_dim, patch_dim)

    def forward(self, x):
        """
        Args:
            x: (B, N, D), where N = num_patches, D = in_dim
        Returns:
            Reconstructed image: (B, C, H, W)
        """
        B, N, D = x.shape
        H, W = self.image_size
        ph, pw = self.patch_size
        gh, gw = H // ph, W // pw
        assert N == gh * gw, "Number of patches does not match image size"

        # Project patch embedding to flattened pixels
        x = self.project(x)  # (B, N, patch_dim)
        x = rearrange(x, "b (h w) (ph pw c) -> b c (h ph) (w pw)",
                      h=gh, w=gw, ph=ph, pw=pw, c=self.out_channels)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, emb_channels=None, device="cuda"):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.adapt = AdaptiveScale(emb_channels, dim, dim=1, device=device) if emb_channels else None

    def forward(self, x, emb=None):
        x = self.norm(x)
        x = self.net(x)
        if self.adapt and emb is not None:
            x = self.adapt(x, emb)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, emb_channels=None, device="cuda"):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.adapt = AdaptiveScale(emb_channels, dim, dim = 1, device=device) if emb_channels else None

    def forward(self, x, emb=None):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        if self.adapt and emb is not None:
            out = self.adapt(out, emb)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, emb_channels=None, device="cuda"):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads, dim_head, emb_channels, device),
                FeedForward(dim, mlp_dim, emb_channels, device)
            ]) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, emb=None):
        for attn, ff in self.layers:
            x = attn(x, emb) + x
            x = ff(x, emb) + x
        x = self.norm(x)
        return x

class ViT3(nn.Module):
    def __init__(self, *, 
                image_size=(128, 128), 
                in_channels=4,
                out_channels=4,
                latent_channels=128, 
                patch_sizes=8,
                dims=512, 
                depth=10, 
                heads=8, 
                dim_head=128, 
                mlp_dim=1024,
                emb_channels=128):
        super().__init__()

        self.fourier_embedding = FourierEmbedding(dims=emb_channels)
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1)
        )

        print(patch_sizes, dims)
        self.embedding =  PatchEmbedding(image_size, patch_sizes, dims, latent_channels, pos_embedding=True)
        self.transformers = TransformerBlock(dims, depth, heads, dim_head, mlp_dim, emb_channels=emb_channels)

        self.patch_to_image = Depatchify(image_size=image_size, patch_size=patch_sizes, in_dim=dims, out_channels=latent_channels)
        self.project = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(latent_channels, out_channels, kernel_size=1, bias=False)
        )
        self.project = nn.Sequential(
                nn.Conv2d(latent_channels, latent_channels, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
                nn.Conv2d(latent_channels, out_channels, kernel_size=1, bias=False)
)

    def forward(self, x, t):
        x = self.lift(x)
        emb = self.fourier_embedding(t)

        x = self.embedding(x)
        x = self.transformers(x, emb)

        x = self.patch_to_image(x)
        x = self.project(x)
        return x

class Vit3_pl(GeneralModel_pl):
    def __init__(self,  
                in_dim, 
                out_dim,
                loss_fn,
                config_train: dict = dict(),
                config_arch: dict = dict()):
        super().__init__(in_dim, out_dim, config_train)
        self.loss_fn = loss_fn
        self.model = ViT3(
            in_channels=in_dim,
            out_channels=out_dim,
            latent_channels=config_arch["latent_channels"],
            image_size=(config_train["s"], config_train["s"]),
            patch_sizes=config_arch["patch_sizes"],
            dims=config_arch["dims"],
            depth=config_arch["depth"],
            heads=config_arch["heads"],
            dim_head=config_arch["dim_head"],
            mlp_dim=config_arch["mlp_dim"],
            emb_channels= config_arch["emb_channels"]
        )