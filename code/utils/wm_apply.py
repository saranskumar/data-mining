"""
utils/wm_apply.py — apply the 32x32-trained watermark to images of ANY size.

The encoder/decoder work at 32x32. To keep the user's image at its original resolution:
  embed : full -> down to 32x32 -> encoder -> residual r (32x32, |r| <= alpha)
          -> upsample r to full size (bicubic, smooth) -> full + strength * r_up
  decode: full -> down to 32x32 -> decoder
Down-sampling the up-sampled residual approximately recovers r, so the decoder sees the
same signal it was trained on. The residual is low-frequency at full size, which makes it
far less visible than the same change on a 32x32 image shown enlarged.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

SIZE = (32, 32)


def _resize(arr: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """float32 HxWx3 in any range -> resized float32 (per-channel, bicubic, no clipping)."""
    out = [np.asarray(Image.fromarray(arr[..., c].astype(np.float32), mode="F").resize(size_wh, Image.BICUBIC))
           for c in range(arr.shape[2])]
    return np.stack(out, axis=-1).astype(np.float32)


def to_small(img_full: np.ndarray) -> np.ndarray:
    """float32 HxWx3 [0,1] -> 32x32 [0,1] (the same reduction used by the detector)."""
    pil = Image.fromarray((np.clip(img_full, 0, 1) * 255).round().astype(np.uint8))
    return np.asarray(pil.resize(SIZE, Image.BICUBIC), dtype=np.float32) / 255.0


def texture_mask(img_full: np.ndarray, floor: float = 0.25, tau: float = 0.05) -> np.ndarray:
    """Perceptual (JND-style) mask in [floor, 1]: ~1 in textured regions where a small change is
    invisible, ~floor in flat regions (sky, white paper) where it would show."""
    from scipy.ndimage import gaussian_filter
    y = img_full @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    s = max(1.5, min(img_full.shape[:2]) / 250)
    mu = gaussian_filter(y, s)
    std = np.sqrt(np.maximum(gaussian_filter(y * y, s) - mu * mu, 0))
    m = np.clip(std / tau, floor, 1.0)
    return gaussian_filter(m, 3 * s).astype(np.float32)  # smooth: no visible mask edges


def embed_fullres(img_full: np.ndarray, bits: np.ndarray, enc, device, strength: float = 1.0,
                  adaptive: bool = False):
    """Return (marked_full float32 [0,1] same HxW as input, residual_full).
    adaptive=True scales the watermark by texture_mask (less change in flat areas)."""
    import torch
    small = to_small(img_full)
    cover = torch.from_numpy(small.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    msg = torch.as_tensor(bits, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        marked_small = enc(cover, msg).cpu().squeeze(0).numpy().transpose(1, 2, 0)
    residual = marked_small - small                      # |r| <= alpha
    h, w = img_full.shape[:2]
    r_up = _resize(residual, (w, h)) * strength
    if adaptive:
        r_up *= texture_mask(img_full)[..., None]
    marked = np.clip(img_full + r_up, 0.0, 1.0).astype(np.float32)
    return marked, marked - img_full


def decode_fullres(img_full: np.ndarray, dec, device) -> np.ndarray:
    """Any-size image -> decoded bits (int32)."""
    import torch
    small = to_small(img_full)
    t = torch.from_numpy(np.ascontiguousarray(small.transpose(2, 0, 1))).unsqueeze(0).float().to(device)
    with torch.no_grad():
        return (torch.sigmoid(dec(t)) > 0.5).cpu().numpy().astype(np.int32)[0]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100.0 if mse < 1e-12 else 10 * np.log10(1.0 / mse)


def _jpeg_roundtrip(img: np.ndarray, quality: int = 80) -> np.ndarray:
    import io
    buf = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8)).save(buf, format="JPEG", quality=quality)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0


def embed_verified(img_full: np.ndarray, bits: np.ndarray, enc, dec, device, strength: float = 1.0,
                   adaptive: bool = True, max_boost: float = 2.5, steps: int = 7, trace: list | None = None):
    """Embed, decode, and raise the strength only as far as needed for the message to read back
    exactly — clean and after a JPEG-80 round-trip. Ordinary photos stay at `strength` (least
    visible); flat graphics / screenshots get boosted. Returns (marked, residual, used_strength, ok).
    If `trace` is a list, one dict per attempt is appended (strength, clean/jpeg read-back, psnr)."""
    bits = np.asarray(bits).astype(np.int32)
    marked = residual = None
    for s in np.linspace(strength, strength * max_boost, steps):
        marked, residual = embed_fullres(img_full, bits, enc, device, strength=float(s), adaptive=adaptive)
        # JPEG check only for real-size images: JPEG on a ~32 px image destroys any fine pattern
        jpeg_ok = (min(img_full.shape[:2]) < 128 or
                   np.array_equal(decode_fullres(_jpeg_roundtrip(marked), dec, device), bits))
        clean_ok = np.array_equal(decode_fullres(marked, dec, device), bits)
        if trace is not None:
            trace.append(dict(strength=float(s), clean_ok=bool(clean_ok),
                              jpeg_ok=None if min(img_full.shape[:2]) < 128 else bool(jpeg_ok),
                              psnr=psnr(img_full, marked)))
        if clean_ok and jpeg_ok:
            return marked, residual, float(s), True
    return marked, residual, float(s), False
