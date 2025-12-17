# src/model.py

"""U-Net architecture implementation for Diffusion Models.

This module defines the U-Net architecture tailored for 28x28 image generation.
It includes Sinusoidal Position Embeddings for time steps, Residual Blocks
for feature extraction, and Self-Attention blocks for global context.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Creates sinusoidal embeddings for time steps.

    Based on the "Attention Is All You Need" paper.
    Encodes the time step `t` into a vector of size `dim`.
    """

    def __init__(self, dim: int):
        """Initializes the embedding layer.

        Args:
            dim (int): The dimensionality of the embedding vector.
        """
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Computes the sinusoidal embeddings.

        Args:
            time (torch.Tensor): A 1-D tensor of time steps with shape (batch_size,).

        Returns:
            torch.Tensor: Embeddings of shape (batch_size, dim).
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """A Residual Block with integrated time embeddings.

    Standard ResNet block with GroupNorm, SiLU activation, and a projection
    layer to inject time information.
    Structure: Input -> Conv1 -> Add Time Emb -> Conv2 -> Add Residual -> Output
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
        """Initializes the ResidualBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            time_emb_dim (int): Dimension of the time embedding vector.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels), nn.SiLU(), nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.time_emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResidualBlock.

        Args:
            x (torch.Tensor): Input feature map (B, C, H, W).
            time_emb (torch.Tensor): Time embedding vector (B, time_emb_dim).

        Returns:
            torch.Tensor: Output feature map (B, out_channels, H, W).
        """
        h = self.conv1(x)
        # Broadcast time embedding to spatial dimensions
        h = h + self.time_emb_proj(time_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """Self-Attention block for capturing global dependencies.

    Applied at lower resolutions to allow the model to attend to distant parts
    of the image. Uses standard Multi-Head Attention.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        """Initializes the AttentionBlock.

        Args:
            channels (int): Number of channels in the feature map.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
        """
        super().__init__()
        assert channels % num_heads == 0, f"Channels {channels} must be divisible by num_heads {num_heads}."
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the AttentionBlock.

        Args:
            x (torch.Tensor): Input feature map (B, C, H, W).

        Returns:
            torch.Tensor: Output feature map with attention applied.
        """
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.view(B, self.num_heads, -1, H * W).transpose(-1, -2)
        k = k.view(B, self.num_heads, -1, H * W)
        v = v.view(B, self.num_heads, -1, H * W).transpose(-1, -2)

        attn = F.softmax(torch.matmul(q, k) * self.scale, dim=-1)
        out = torch.matmul(attn, v).transpose(-1, -2).reshape(B, C, H, W)

        return x + self.proj_out(out)


class DownBlock(nn.Module):
    """Downsampling block for the U-Net encoder.

    Consists of multiple ResidualBlocks followed by a strided convolution
    for downsampling.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, num_res_blocks: int, dropout: float):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(in_channels if i == 0 else out_channels, out_channels, time_emb_dim, dropout)
                for i in range(num_res_blocks)
            ]
        )
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for res_block in self.res_blocks:
            x = res_block(x, time_emb)
        skip_connection = x
        return self.downsample(x), skip_connection


class UpBlock(nn.Module):
    """Upsampling block for the U-Net decoder.

    Consists of an Upsample layer (Bilinear), concatenation with skip connection,
    and multiple ResidualBlocks.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, num_res_blocks: int, dropout: float):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(in_channels, out_channels, time_emb_dim, dropout)]
            + [ResidualBlock(out_channels, out_channels, time_emb_dim, dropout) for _ in range(num_res_blocks - 1)]
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Handle potential slight shape mismatch due to padding
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([skip, x], dim=1)
        for res_block in self.res_blocks:
            x = res_block(x, time_emb)
        return x


class UNet(nn.Module):
    """The main U-Net model for predicting noise in the diffusion process.

    Architecture:
    1. Time Embedding Projection.
    2. Initial Convolution.
    3. Downsampling Path (Encoder) with optional Attention.
    4. Middle Bottleneck (ResBlock -> Attn -> ResBlock).
    5. Upsampling Path (Decoder) with Skip Connections and optional Attention.
    6. Final Output Convolution.
    """

    def __init__(self, **kwargs):
        """Initializes the U-Net.

        Expected kwargs from Config:
            in_channels, out_channels, model_channels, channel_mult,
            num_res_blocks, attention_resolutions, num_heads, dropout,
            time_emb_dim, image_size
        """
        super().__init__()
        self.config = kwargs

        in_channels = kwargs.get("in_channels", 1)
        out_channels = kwargs.get("out_channels", 1)
        model_channels = kwargs.get("model_channels", 128)
        channel_mult = kwargs.get("channel_mult", (1, 2, 4))
        num_res_blocks = kwargs.get("num_res_blocks", 2)
        attention_resolutions = kwargs.get("attention_resolutions", (7,))
        num_heads = kwargs.get("num_heads", 8)
        dropout = kwargs.get("dropout", 0.1)
        time_emb_dim = kwargs.get("time_emb_dim", 512)
        image_size = kwargs.get("image_size", 28)

        # Time Embedding MLP
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.init_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        # --- Downsampling Path ---
        self.down_blocks = nn.ModuleList()
        self.down_attentions = nn.ModuleList()
        in_ch = model_channels
        current_res = image_size

        for _i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            self.down_blocks.append(DownBlock(in_ch, out_ch, time_emb_dim, num_res_blocks, dropout))
            self.down_attentions.append(
                AttentionBlock(out_ch, num_heads) if current_res in attention_resolutions else nn.Identity()
            )
            in_ch = out_ch
            current_res //= 2

        # --- Bottleneck ---
        self.mid_block1 = ResidualBlock(in_ch, in_ch, time_emb_dim, dropout)
        self.mid_attention = AttentionBlock(in_ch, num_heads)
        self.mid_block2 = ResidualBlock(in_ch, in_ch, time_emb_dim, dropout)

        # --- Upsampling Path ---
        self.up_blocks = nn.ModuleList()
        self.up_attentions = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            skip_ch = out_ch
            # Calculate input channels for UpBlock: output of prev up + skip connection
            prev_up_ch = channel_mult[i + 1] * model_channels if i + 1 < len(channel_mult) else in_ch
            up_in_ch = prev_up_ch + skip_ch

            self.up_blocks.append(UpBlock(up_in_ch, out_ch, time_emb_dim, num_res_blocks, dropout))
            current_res *= 2
            self.up_attentions.append(
                AttentionBlock(out_ch, num_heads) if current_res in attention_resolutions else nn.Identity()
            )

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Predicts noise given input image x_t and time step t.

        Args:
            x (torch.Tensor): Noisy input image (B, C, H, W).
            time (torch.Tensor): Time step indices (B,).

        Returns:
            torch.Tensor: Predicted noise (B, C, H, W).
        """
        time_emb = self.time_embed(time)
        x = self.init_conv(x)

        skips = []
        # Encoder
        for block, attn in zip(self.down_blocks, self.down_attentions, strict=False):
            x, skip = block(x, time_emb)
            x = attn(x)
            skips.append(skip)

        # Bottleneck
        x = self.mid_block1(x, time_emb)
        x = self.mid_attention(x)
        x = self.mid_block2(x, time_emb)

        # Decoder
        skips = skips[::-1]
        for i, (block, attn) in enumerate(zip(self.up_blocks, self.up_attentions, strict=False)):
            x = block(x, skips[i], time_emb)
            x = attn(x)

        return self.final_conv(x)


def count_parameters(model: nn.Module) -> int:
    """Counts the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Simple test to verify architecture integrity
    print("🔬 Testing UNet architecture...")
    model = UNet(image_size=28)
    x_dummy = torch.randn(4, 1, 28, 28)
    t_dummy = torch.randint(0, 1000, (4,))
    out = model(x_dummy, t_dummy)
    print(f"✅ Forward pass successful. Output shape: {out.shape}")
