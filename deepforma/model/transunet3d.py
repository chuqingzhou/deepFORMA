from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for global context modeling."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        return out


class TransformerBlock(nn.Module):
    """Transformer encoder block with residual attention and MLP."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbedding3D(nn.Module):
    """3D patch embedding to convert CNN feature maps to a token sequence."""

    def __init__(self, patch_size: int = 2, in_channels: int = 256, embed_dim: int = 768):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        # x: (B, C, D, H, W)
        x = self.proj(x)  # (B, E, D', H', W')
        b, c, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        x = self.norm(x)
        return x, (d, h, w)


class CNNEncoder(nn.Module):
    """3D CNN encoder for hierarchical feature extraction."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.enc1 = self._block(in_channels, 32, stride=1)
        self.enc2 = self._block(32, 64, stride=2)
        self.enc3 = self._block(64, 128, stride=2)
        self.enc4 = self._block(128, 256, stride=2)

    @staticmethod
    def _block(in_ch: int, out_ch: int, stride: int):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        skip1 = self.enc1(x)
        skip2 = self.enc2(skip1)
        skip3 = self.enc3(skip2)
        feat = self.enc4(skip3)
        return feat, [skip1, skip2, skip3]


class CNNDecoder(nn.Module):
    """3D CNN decoder with skip connections."""

    def __init__(self, transformer_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(transformer_dim, 256)
        self.up1 = self._block(256 + 128, 128)
        self.up2 = self._block(128 + 64, 64)
        self.up3 = self._block(64 + 32, 32)
        self.final = nn.Conv3d(32, 1, 1)

    @staticmethod
    def _block(in_ch: int, out_ch: int):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, transformer_features: torch.Tensor, spatial_shape, skip_connections):
        b, n, _c = transformer_features.shape
        d, h, w = spatial_shape
        x = self.proj(transformer_features).transpose(1, 2).reshape(b, 256, d, h, w)

        # Upsample to match skip feature sizes exactly
        x = F.interpolate(x, size=skip_connections[2].shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip_connections[2]], dim=1)
        x = self.up1(x)

        x = F.interpolate(x, size=skip_connections[1].shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip_connections[1]], dim=1)
        x = self.up2(x)

        x = F.interpolate(x, size=skip_connections[0].shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip_connections[0]], dim=1)
        x = self.up3(x)

        return self.final(x)


class TransUNet3D(nn.Module):
    """Hybrid CNN-Transformer 3D segmentation model."""

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        patch_size: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cnn_encoder = CNNEncoder(in_channels)
        self.patch_embed = PatchEmbedding3D(patch_size, 256, embed_dim)
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)]
        )
        self.cnn_decoder = CNNDecoder(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat, skips = self.cnn_encoder(x)
        tokens, spatial_shape = self.patch_embed(cnn_feat)
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        return self.cnn_decoder(tokens, spatial_shape, skips)


def build_model(dropout: float = 0.1) -> TransUNet3D:
    """Factory for the default TransUNet3D configuration used in this project."""
    return TransUNet3D(
        in_channels=1,
        embed_dim=768,
        num_heads=12,
        num_layers=6,
        patch_size=2,
        mlp_ratio=4.0,
        dropout=float(dropout),
    )

