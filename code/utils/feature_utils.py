"""
utils/feature_utils.py — per-image feature extraction.

Feature groups (all from 03-feature-spec.md §4):
  1. noise_features    (~20 dims)  — Gaussian/median residue stats
  2. freq_features     (~22 dims)  — FFT radial spectrum + DCT bands
  3. color_features    (~114 dims) — RGB/HSV histograms + moments
  4. glcm_features     (~20 dims)  — GLCM texture (64-level, 2 dist × 4 angles)

Public API:
  extract_all(img_float32) -> (vector, names)
  batch_extract(imgs_float32, n_workers) -> (X, names)
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage, stats
from scipy.fft import fft2, fftshift

logger = logging.getLogger(__name__)

# ─── helpers ──────────────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    """float32 (H,W,3) → float32 (H,W) Rec-601 luma."""
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def _safe_stats(arr: np.ndarray) -> Tuple[float, float, float, float]:
    """mean, var, skew, kurtosis — robust on flat arrays."""
    flat = arr.flatten()
    if np.std(flat) < 1e-8:
        return float(np.mean(flat)), 0.0, 0.0, 0.0
    return (
        float(np.mean(flat)),
        float(np.var(flat)),
        float(stats.skew(flat)),
        float(stats.kurtosis(flat)),
    )


# ─── Group 1: Noise residue (~20 dims) ───────────────────────────────────────

def noise_features(img: np.ndarray, gaussian_sigma: float = 1.0, median_ksize: int = 3) -> Tuple[np.ndarray, List[str]]:
    """
    img: float32 (H,W,3) in [0,1].
    Returns (vector, names).
    """
    feats: List[float] = []
    names: List[str] = []

    # Gaussian residue per channel: mean, var, skew, kurt, energy (5 × 3 = 15)
    gauss_smooth = np.stack(
        [ndimage.gaussian_filter(img[:, :, c], sigma=gaussian_sigma) for c in range(3)],
        axis=2,
    )
    gauss_res = img - gauss_smooth  # (H,W,3)

    for ci, ch in enumerate(["R", "G", "B"]):
        r = gauss_res[:, :, ci]
        m, v, sk, ku = _safe_stats(r)
        energy = float(np.sum(r ** 2) / r.size)
        feats += [m, v, sk, ku, energy]
        names += [f"noise_gauss_{ch}_mean", f"noise_gauss_{ch}_var",
                  f"noise_gauss_{ch}_skew", f"noise_gauss_{ch}_kurt",
                  f"noise_gauss_{ch}_energy"]

    # Median residue (grayscale): mean_abs, var, energy (3)
    gray = _to_gray(img)
    med_smooth = ndimage.median_filter(gray, size=median_ksize)
    med_res = gray - med_smooth
    feats += [float(np.mean(np.abs(med_res))), float(np.var(med_res)),
              float(np.sum(med_res ** 2) / med_res.size)]
    names += ["noise_med_gray_abs_mean", "noise_med_gray_var", "noise_med_gray_energy"]

    # Autocorrelation of grayscale Gaussian residue at lag-1 (horiz, vert) (2)
    gr = gauss_res[:, :, 0] * 0.299 + gauss_res[:, :, 1] * 0.587 + gauss_res[:, :, 2] * 0.114
    def lag1_corr(arr, axis):
        a = arr.flatten() if axis == 0 else arr.T.flatten()
        n = len(a)
        if n < 2 or np.std(a) < 1e-8:
            return 0.0
        a0, a1 = a[:-1], a[1:]
        return float(np.corrcoef(a0, a1)[0, 1])

    feats += [lag1_corr(gr, 0), lag1_corr(gr, 1)]
    names += ["noise_autocorr_horiz", "noise_autocorr_vert"]

    return np.array(feats, dtype=np.float32), names


# ─── Group 2: Frequency domain FFT + DCT (~22 dims) ─────────────────────────

def freq_features(img: np.ndarray, radial_bins: int = 8) -> Tuple[np.ndarray, List[str]]:
    gray = _to_gray(img)
    H, W = gray.shape
    feats: List[float] = []
    names: List[str] = []

    # Hann window
    hann = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    spectrum = np.abs(fftshift(fft2(gray * hann))) ** 2
    spectrum_log = np.log1p(spectrum)

    # Radial coordinates
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    max_r = min(cy, cx)

    # Azimuthal average in bins
    bin_edges = np.linspace(0, max_r, radial_bins + 1)
    radial_power = []
    for i in range(radial_bins):
        mask = (R >= bin_edges[i]) & (R < bin_edges[i + 1])
        val = float(np.mean(spectrum_log[mask])) if mask.sum() > 0 else 0.0
        radial_power.append(val)
    feats += radial_power
    names += [f"fft_radial_bin{i}" for i in range(radial_bins)]

    # High-freq ratio (outer half of max_r)
    outer_mask = R >= max_r / 2
    inner_mask = R < max_r / 2
    total = spectrum_log.sum() + 1e-8
    hf_ratio = float(spectrum_log[outer_mask].sum() / total)
    feats.append(hf_ratio)
    names.append("fft_hf_ratio")

    # Spectral slope (log-log fit)
    freqs = np.array([(bin_edges[i] + bin_edges[i+1])/2 for i in range(radial_bins)])
    log_f = np.log1p(freqs)
    log_p = np.array(radial_power)
    if np.std(log_f) > 1e-8 and np.std(log_p) > 1e-8:
        slope = float(np.polyfit(log_f, log_p, 1)[0])
    else:
        slope = 0.0
    feats.append(slope)
    names.append("fft_spectral_slope")

    # Spectral centroid
    all_r = R.flatten()
    all_s = spectrum_log.flatten()
    centroid = float((all_r * all_s).sum() / (all_s.sum() + 1e-8))
    feats.append(centroid)
    names.append("fft_spectral_centroid")

    # Spectral flatness (geometric mean / arithmetic mean)
    pos = spectrum_log[spectrum_log > 0]
    if len(pos) > 0:
        flat = float(np.exp(np.mean(np.log(pos + 1e-8))) / (np.mean(pos) + 1e-8))
    else:
        flat = 0.0
    feats.append(flat)
    names.append("fft_spectral_flatness")

    # Peak-to-mean ratio (non-DC)
    non_dc = spectrum_log.copy()
    non_dc[cy, cx] = 0.0
    ptm = float(non_dc.max() / (non_dc.mean() + 1e-8))
    feats.append(ptm)
    names.append("fft_peak_to_mean")

    # DCT on grayscale — 8×8 block DCT
    from scipy.fft import dctn
    bsize = 8
    pad_h = (bsize - H % bsize) % bsize
    pad_w = (bsize - W % bsize) % bsize
    g_pad = np.pad(gray, ((0, pad_h), (0, pad_w)))
    Hb, Wb = g_pad.shape
    nb_h, nb_w = Hb // bsize, Wb // bsize

    block_energies_low, block_energies_mid, block_energies_high = [], [], []
    all_ac = []
    dc_vals = []

    for bi in range(nb_h):
        for bj in range(nb_w):
            blk = g_pad[bi*bsize:(bi+1)*bsize, bj*bsize:(bj+1)*bsize]
            d = dctn(blk, norm="ortho")
            # Zig-zag energy bands (approximate by index sum)
            low, mid, high = 0.0, 0.0, 0.0
            for r in range(bsize):
                for c in range(bsize):
                    e = d[r, c] ** 2
                    s = r + c
                    if s < 4:
                        low += e
                    elif s < 8:
                        mid += e
                    else:
                        high += e
            block_energies_low.append(low)
            block_energies_mid.append(mid)
            block_energies_high.append(high)
            dc_vals.append(d[0, 0])
            all_ac.extend(np.abs(d.flatten()[1:]).tolist())

    tot = np.mean(block_energies_low) + np.mean(block_energies_mid) + np.mean(block_energies_high) + 1e-8
    feats += [
        float(np.mean(block_energies_low)),
        float(np.mean(block_energies_mid)),
        float(np.mean(block_energies_high)),
        float(np.mean(block_energies_low) / tot),
        float(np.mean(block_energies_high) / tot),
    ]
    names += ["dct_energy_low", "dct_energy_mid", "dct_energy_high",
              "dct_low_ratio", "dct_high_ratio"]

    feats += [
        float(np.mean(all_ac)),
        float(np.mean(dc_vals)),
        float(np.mean(np.array(all_ac) < 0.1)),
    ]
    names += ["dct_mean_ac", "dct_mean_dc", "dct_near_zero_frac"]

    # Block variance of high-band energy
    feats.append(float(np.var(block_energies_high)))
    names.append("dct_highband_var")

    return np.array(feats, dtype=np.float32), names


# ─── Group 3: Colour statistics (~114 dims) ──────────────────────────────────

def color_features(img: np.ndarray, hist_bins: int = 16) -> Tuple[np.ndarray, List[str]]:
    feats: List[float] = []
    names: List[str] = []

    # RGB histograms (16 bins × 3 = 48)
    for ci, ch in enumerate(["R", "G", "B"]):
        h, _ = np.histogram(img[:, :, ci], bins=hist_bins, range=(0, 1))
        h = h / (h.sum() + 1e-8)
        feats += h.tolist()
        names += [f"rgb_hist_{ch}_b{i}" for i in range(hist_bins)]

    # HSV histograms (16 bins × 3 = 48)
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(img_uint8)
    hsv_arr = np.array(pil_img.convert("HSV"), dtype=np.float32) / 255.0
    for ci, ch in enumerate(["H", "S", "V"]):
        h, _ = np.histogram(hsv_arr[:, :, ci], bins=hist_bins, range=(0, 1))
        h = h / (h.sum() + 1e-8)
        feats += h.tolist()
        names += [f"hsv_hist_{ch}_b{i}" for i in range(hist_bins)]

    # Per-channel mean/std/skewness (3 × 3 = 9)
    for ci, ch in enumerate(["R", "G", "B"]):
        c = img[:, :, ci].flatten()
        feats += [float(np.mean(c)), float(np.std(c)), float(stats.skew(c))]
        names += [f"rgb_{ch}_mean", f"rgb_{ch}_std", f"rgb_{ch}_skew"]

    # Mean saturation, mean value (2)
    feats += [float(hsv_arr[:, :, 1].mean()), float(hsv_arr[:, :, 2].mean())]
    names += ["hsv_mean_sat", "hsv_mean_val"]

    # Fraction near-saturated per channel (0 or 255) (3)
    for ci, ch in enumerate(["R", "G", "B"]):
        c_u8 = img_uint8[:, :, ci]
        frac = float(((c_u8 <= 2) | (c_u8 >= 253)).sum() / c_u8.size)
        feats.append(frac)
        names.append(f"rgb_{ch}_near_sat_frac")

    # Inter-channel correlations RG, GB, RB (3)
    r = img[:, :, 0].flatten()
    g = img[:, :, 1].flatten()
    b = img[:, :, 2].flatten()
    for a, b_, tag in [(r, g, "RG"), (g, b, "GB"), (r, b, "RB")]:
        if np.std(a) < 1e-8 or np.std(b_) < 1e-8:
            feats.append(0.0)
        else:
            feats.append(float(np.corrcoef(a, b_)[0, 1]))
        names.append(f"rgb_corr_{tag}")

    # Hue histogram entropy (1)
    h_hist, _ = np.histogram(hsv_arr[:, :, 0], bins=hist_bins, range=(0, 1))
    h_prob = h_hist / (h_hist.sum() + 1e-8)
    entropy = float(-np.sum(h_prob * np.log(h_prob + 1e-8)))
    feats.append(entropy)
    names.append("hsv_hue_entropy")

    return np.array(feats, dtype=np.float32), names


# ─── Group 4: GLCM texture (~20 dims) ────────────────────────────────────────

def glcm_features(
    img: np.ndarray,
    levels: int = 64,
    distances: List[int] = [1, 2],
    angles: List[float] = [0, 0.7854, 1.5708, 2.3562],
) -> Tuple[np.ndarray, List[str]]:
    from skimage.feature import graycomatrix, graycoprops

    gray = _to_gray(img)
    # Quantise to [0, levels-1]
    g_q = (gray * (levels - 1)).clip(0, levels - 1).astype(np.uint8)

    feats: List[float] = []
    names: List[str] = []

    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    for d in distances:
        glcm = graycomatrix(g_q, distances=[d], angles=angles,
                            levels=levels, symmetric=True, normed=True)
        for prop in props:
            vals = graycoprops(glcm, prop)[0]  # shape (n_angles,)
            # Handle NaN in correlation for flat images
            vals = np.nan_to_num(vals, nan=0.0)
            feats.append(float(vals.mean()))
            names.append(f"glcm_d{d}_{prop}_mean")
            feats.append(float(vals.std()))
            names.append(f"glcm_d{d}_{prop}_std")

    return np.array(feats, dtype=np.float32), names


# ─── Combined extractor ───────────────────────────────────────────────────────

def extract_all(
    img: np.ndarray,
    cfg: Optional[dict] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    img: float32 (H,W,3) in [0,1].
    Returns: (feature_vector, feature_names)
    """
    if cfg is None:
        cfg = {}
    noise_cfg = cfg.get("noise", {})
    fft_cfg = cfg.get("fft", {})
    color_cfg = cfg.get("color", {})
    glcm_cfg = cfg.get("glcm", {})

    vn, nn = noise_features(img, **{k: v for k, v in noise_cfg.items()
                                    if k in ["gaussian_sigma", "median_ksize"]})
    vf, nf = freq_features(img, **{k: v for k, v in fft_cfg.items()
                                   if k in ["radial_bins"]})
    vc, nc = color_features(img, **{k: v for k, v in color_cfg.items()
                                    if k in ["hist_bins"]})
    vg, ng = glcm_features(
        img,
        levels=glcm_cfg.get("levels", 64),
        distances=glcm_cfg.get("distances", [1, 2]),
        angles=glcm_cfg.get("angles", [0, 0.7854, 1.5708, 2.3562]),
    )

    vec = np.concatenate([vn, vf, vc, vg])
    names = nn + nf + nc + ng

    # Sanity: no NaN/Inf
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec, names


def _worker_extract(args):
    idx, img, cfg = args
    vec, names = extract_all(img, cfg)
    return idx, vec, names


def batch_extract(
    imgs: np.ndarray,
    cfg: Optional[dict] = None,
    n_workers: int = 4,
) -> Tuple[np.ndarray, List[str]]:
    """
    imgs: float32 (N, H, W, 3)
    Returns: X (N, D), feature_names (D,)
    """
    from tqdm import tqdm

    N = len(imgs)
    # accept uint8 (0-255) or float (0-1); convert per image to keep memory low
    prep = (lambda a: a.astype(np.float32) / 255.0) if imgs.dtype == np.uint8 else (lambda a: a)
    jobs = ((i, prep(imgs[i]), cfg) for i in range(N))

    # Run in process pool
    results = [None] * N
    with mp.Pool(n_workers) as pool:
        for idx, vec, names in tqdm(
            pool.imap_unordered(_worker_extract, jobs, chunksize=32),
            total=N, desc="Extracting features"
        ):
            results[idx] = (vec, names)

    all_vecs = np.stack([r[0] for r in results], axis=0)
    feature_names = results[0][1]
    return all_vecs.astype(np.float32), feature_names


def save_feature_names(names: List[str], path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(names, f, indent=2)


def load_feature_names(path: str | Path) -> List[str]:
    with open(path) as f:
        return json.load(f)
