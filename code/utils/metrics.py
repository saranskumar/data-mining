"""
utils/metrics.py — evaluation metrics used across pipeline stages.
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, adjusted_rand_score,
    normalized_mutual_info_score,
)


# ─── Classification ───────────────────────────────────────────────────────────

def classification_report_dict(y_true, y_pred, y_prob=None) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    d = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
         "confusion_matrix": cm.tolist()}
    if y_prob is not None:
        try:
            d["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            d["roc_auc"] = float("nan")
    return d


# ─── Clustering ───────────────────────────────────────────────────────────────

def cluster_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Purity: fraction of samples assigned to the majority class in their cluster."""
    n = len(labels_true)
    if n == 0:
        return 0.0
    total = 0
    for k in np.unique(labels_pred):
        mask = labels_pred == k
        total += np.bincount(labels_true[mask]).max()
    return total / n


def clustering_metrics(labels_true: np.ndarray, labels_pred: np.ndarray) -> dict:
    return {
        "purity": cluster_purity(labels_true, labels_pred),
        "ARI": adjusted_rand_score(labels_true, labels_pred),
        "NMI": normalized_mutual_info_score(labels_true, labels_pred),
    }


# ─── Watermark ────────────────────────────────────────────────────────────────

def bit_accuracy(bits_true: np.ndarray, bits_pred: np.ndarray) -> float:
    """Mean fraction of correctly decoded bits."""
    return float(np.mean(bits_true == bits_pred))


def exact_match_rate(bits_true: np.ndarray, bits_pred: np.ndarray) -> float:
    """Fraction of messages decoded perfectly."""
    return float(np.all(bits_true == bits_pred, axis=1).mean())


def psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (dB). inputs float32 [0,1]."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(max_val ** 2 / mse))


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Structural Similarity Index (mean over images). inputs float32 [0,1]."""
    from skimage.metrics import structural_similarity
    if img1.ndim == 4:
        scores = [structural_similarity(a, b, channel_axis=2, data_range=1.0)
                  for a, b in zip(img1, img2)]
        return float(np.mean(scores))
    return float(structural_similarity(img1, img2, channel_axis=2, data_range=1.0))


def binomial_threshold(n_bits: int, fpr_target: float = 1e-3) -> float:
    """
    Minimum bit-accuracy threshold t = k/n_bits such that
    P(bit_acc >= t | unmarked image, bits ~ Bernoulli(0.5)) <= fpr_target.
    Exact Binomial(n_bits, 0.5) tail (the normal approximation is loose for small n_bits).
    """
    from scipy.stats import binom
    for k in range(n_bits + 1):
        if binom.sf(k - 1, n_bits, 0.5) <= fpr_target:   # P(X >= k)
            return k / n_bits
    return 1.0
