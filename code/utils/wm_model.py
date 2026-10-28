"""
utils/wm_model.py — PyTorch Encoder / Decoder / NoiseLayer for watermarking.

Architecture (sized for RTX 2050 4 GB, 32×32 images):
  Encoder: ConvBlock stack on cover + spatially-tiled message → residual δ
           output x_w = clip(x + alpha * tanh(δ), 0, 1)  (bounded per-pixel change)
  Decoder: small CNN → L logits
  NoiseLayer: differentiable approximations sampled per batch
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Building blocks ──────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=k // 2, bias=False,
                      padding_mode="reflect"),  # no zero-padding edge artefacts
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Encoder ──────────────────────────────────────────────────────────────────

class EncoderCNN(nn.Module):
    """
    cover (B, 3, H, W) + message (B, L) → watermarked (B, 3, H, W)
    message is spatially tiled to (B, L, H, W) and concatenated with cover.
    """

    def __init__(self, message_bits: int = 48, strength_alpha: float = 1.0,
                 lum_only: bool = False) -> None:
        super().__init__()
        self.L = message_bits
        self.alpha = strength_alpha
        self.lum_only = lum_only
        # Learned spatial embedding of the message (each bit gets its own 32x32 pattern).
        # Tiling bits as constant maps gives a conv net no positional information, so with
        # reflect padding + zero-mean residual it could only produce a uniform (removed) shift.
        self.msg_ch, self.size = 4, 32
        self.msg_proj = nn.Linear(message_bits, self.msg_ch * self.size * self.size)
        in_ch = 3 + self.msg_ch

        self.encoder = nn.Sequential(
            ConvBnRelu(in_ch, 64),
            ConvBnRelu(64, 64),
            ConvBnRelu(64, 64),
            ConvBnRelu(64, 32),
            nn.Conv2d(32, 1 if lum_only else 3, 1, bias=False),  # residual δ
        )

    def forward(self, cover: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        B, _, H, W = cover.shape
        # Tile message spatially: (B, L) → (B, L, H, W)
        msg_pm = message * 2.0 - 1.0  # {0,1} -> {-1,+1}, zero-mean input
        msg_map = self.msg_proj(msg_pm).view(B, self.msg_ch, self.size, self.size)
        if (H, W) != (self.size, self.size):
            msg_map = F.interpolate(msg_map, size=(H, W), mode="bilinear", align_corners=False)
        x = torch.cat([cover, msg_map], dim=1)
        # tanh bounds each pixel change to ±alpha, so image quality is set by alpha
        # (PSNR >= 20*log10(1/alpha)) instead of by balancing loss weights.
        delta = torch.tanh(self.encoder(x))
        # Zero-mean per channel: the message cannot hide in a global colour cast / tint (which
        # survives down-scaling and is very visible on full-size photos); it must be a spatial
        # pattern. lum_only additionally forces a grey (brightness-only) pattern.
        delta = delta - delta.mean(dim=(2, 3), keepdim=True)
        return torch.clamp(cover + self.alpha * delta.expand(-1, 3, -1, -1), 0.0, 1.0)


# ─── Decoder ──────────────────────────────────────────────────────────────────

class DecoderCNN(nn.Module):
    """
    image (B, 3, H, W) → logits (B, L)
    """

    def __init__(self, message_bits: int = 48) -> None:
        super().__init__()
        self.L = message_bits

        # Keep spatial layout (stride-2 convs, then flatten): global pooling would discard
        # *where* the encoder put its changes and cap how many bits can be read back.
        self.net = nn.Sequential(
            ConvBnRelu(3, 64),
            ConvBnRelu(64, 64),
            ConvBnRelu(64, 64, stride=2),    # 16x16
            ConvBnRelu(64, 64, stride=2),    # 8x8
            ConvBnRelu(64, 32, stride=2),    # 4x4
            nn.Flatten(),                    # 32*4*4 = 512
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, message_bits),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # raw logits


# ─── Noise layer (differentiable, for training) ───────────────────────────────

class DifferentiableJPEG(nn.Module):
    """Very rough JPEG simulation: quantise → dequantise + Gaussian blur."""

    def __init__(self, levels: int = 64) -> None:
        super().__init__()
        self.levels = levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantise with a straight-through estimator: forward = rounded, backward = identity
        # (plain torch.round has zero gradient, so the encoder would learn nothing from it).
        # Step 1/levels must stay comparable to the watermark budget alpha, otherwise only
        # a global colour shift survives and the encoder learns a visible tint.
        q = torch.round(x * self.levels) / self.levels
        quantised = x + (q - x).detach()
        # Slight smoothing
        k = torch.ones(1, 1, 3, 3, device=x.device) / 9.0
        k = k.expand(x.shape[1], 1, 3, 3)
        blurred = F.conv2d(quantised, k, padding=1, groups=x.shape[1])
        return torch.clamp(blurred, 0, 1)


class NoiseLayer(nn.Module):
    """
    Randomly applies one of: identity, gaussian noise, blur, resize, crop, jpeg-approx.
    Probabilities set in config.
    """

    def __init__(self, probs: Optional[dict] = None, noise_sigma: float = 0.01,
                 jpeg_levels: int = 64) -> None:
        super().__init__()
        self.noise_sigma = noise_sigma  # keep below the watermark budget alpha
        default_probs = {
            "identity": 0.3,
            "gaussian": 0.15,
            "blur": 0.15,
            "resize": 0.15,
            "crop": 0.125,
            "jpeg_approx": 0.125,
        }
        self.probs = probs or default_probs
        self._jpeg = DifferentiableJPEG(jpeg_levels)

    def _gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + self.noise_sigma * torch.randn_like(x), 0, 1)

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.ones(1, 1, 3, 3, device=x.device) / 9.0
        k = k.expand(x.shape[1], 1, 3, 3)
        return torch.clamp(F.conv2d(x, k, padding=1, groups=x.shape[1]), 0, 1)

    def _resize(self, x: torch.Tensor, factor: float = 0.5) -> torch.Tensor:
        _, _, H, W = x.shape
        small = F.interpolate(x, scale_factor=factor, mode="bilinear", align_corners=False)
        return F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)

    def _crop(self, x: torch.Tensor, frac: float = 0.8) -> torch.Tensor:
        _, _, H, W = x.shape
        ch, cw = int(H * frac), int(W * frac)
        top = (H - ch) // 2
        left = (W - cw) // 2
        cropped = x[:, :, top:top+ch, left:left+cw]
        return F.pad(cropped, (left, W - cw - left, top, H - ch - top), mode="replicate")

    def forward(self, x: torch.Tensor, epoch: int = 999) -> torch.Tensor:
        names = list(self.probs.keys())
        cum = 0.0
        boundaries = []
        for n in names:
            cum += self.probs[n]
            boundaries.append(cum)

        r = torch.rand(1).item()
        if r < boundaries[0]:
            return x
        elif r < boundaries[1]:
            return self._gaussian_noise(x)
        elif r < boundaries[2]:
            return self._blur(x)
        elif r < boundaries[3]:
            return self._resize(x)
        elif r < boundaries[4]:
            return self._crop(x)
        else:
            return self._jpeg(x)


# ─── Combined model (convenience) ─────────────────────────────────────────────

class HybridMarkerWM(nn.Module):
    def __init__(self, message_bits: int = 48, alpha: float = 1.0,
                 noise_probs: Optional[dict] = None, lum_only: bool = False) -> None:
        super().__init__()
        self.encoder = EncoderCNN(message_bits, alpha, lum_only)
        self.decoder = DecoderCNN(message_bits)
        self.noise = NoiseLayer(noise_probs)

    def forward(self, cover: torch.Tensor, message: torch.Tensor,
                epoch: int = 999) -> tuple:
        marked = self.encoder(cover, message)
        noised = self.noise(marked, epoch)
        logits = self.decoder(noised)
        return marked, noised, logits
