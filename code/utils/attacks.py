"""
utils/attacks.py — real (non-differentiable) image attacks for eval-time robustness.
All functions: input float32 (H,W,3) [0,1] → output float32 (H,W,3) [0,1].
"""
from __future__ import annotations

import io
from typing import List

import numpy as np
from PIL import Image, ImageFilter


def _to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))


def _from_pil(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0


def jpeg_compress(img: np.ndarray, quality: int = 70) -> np.ndarray:
    buf = io.BytesIO()
    _to_pil(img).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return _from_pil(Image.open(buf))


def resize_attack(img: np.ndarray, factor: float = 0.5) -> np.ndarray:
    pil = _to_pil(img)
    H, W = pil.height, pil.width
    small = pil.resize((max(1, int(W * factor)), max(1, int(H * factor))), Image.BILINEAR)
    return _from_pil(small.resize((W, H), Image.BILINEAR))


def crop_attack(img: np.ndarray, crop_fraction: float = 0.8) -> np.ndarray:
    H, W = img.shape[:2]
    ch, cw = int(H * crop_fraction), int(W * crop_fraction)
    top = (H - ch) // 2
    left = (W - cw) // 2
    cropped = img[top:top+ch, left:left+cw]
    # Pad back to original size
    pad_top = top
    pad_bot = H - ch - top
    pad_left = left
    pad_right = W - cw - left
    return np.pad(cropped, ((pad_top, pad_bot), (pad_left, pad_right), (0, 0)),
                  mode="edge").astype(np.float32)


def gaussian_noise(img: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    return np.clip(img + noise, 0, 1)


def gaussian_blur(img: np.ndarray, radius: float = 1) -> np.ndarray:
    pil = _to_pil(img).filter(ImageFilter.GaussianBlur(radius=radius))
    return _from_pil(pil)


# ─── Batch apply ──────────────────────────────────────────────────────────────

ATTACK_REGISTRY = {
    "clean": lambda img, cfg: img,
    "jpeg_90": lambda img, cfg: jpeg_compress(img, quality=90),
    "jpeg_70": lambda img, cfg: jpeg_compress(img, quality=70),
    "jpeg_50": lambda img, cfg: jpeg_compress(img, quality=50),
    "resize_0.5": lambda img, cfg: resize_attack(img, factor=cfg.get("resize_factor", 0.5)),
    "crop_80": lambda img, cfg: crop_attack(img, crop_fraction=cfg.get("crop_fraction", 0.8)),
    "gauss_noise": lambda img, cfg: gaussian_noise(img, sigma=cfg.get("gaussian_sigma", 0.1)),
    "gauss_blur": lambda img, cfg: gaussian_blur(img, radius=cfg.get("blur_radius", 1)),
}


def apply_attack(imgs: np.ndarray, attack_name: str, cfg: dict = None) -> np.ndarray:
    """Apply named attack to all images in (N, H, W, 3)."""
    cfg = cfg or {}
    fn = ATTACK_REGISTRY[attack_name]
    return np.stack([fn(imgs[i], cfg) for i in range(len(imgs))], axis=0)


def all_attack_names() -> List[str]:
    return list(ATTACK_REGISTRY.keys())
