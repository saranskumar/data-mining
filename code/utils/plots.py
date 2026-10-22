"""
utils/plots.py — consistent matplotlib/seaborn figure helpers.
All figures saved to outputs/figures/ with tight_layout.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

FIG_DPI = 150
PALETTE = {"REAL": "#4C9BE8", "FAKE": "#E8644C"}
STYLE = "seaborn-v0_8-whitegrid"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def _save(fig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ─── EDA ──────────────────────────────────────────────────────────────────────

def plot_sample_grid(
    imgs: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    n: int = 16,
    title: str = "Sample images",
) -> None:
    with plt.style.context(STYLE):
        n = min(n, len(imgs))
        cols = 8
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        axes = axes.flatten()
        for i in range(n):
            axes[i].imshow(imgs[i])
            axes[i].set_title("REAL" if labels[i] == 0 else "FAKE", fontsize=7,
                              color=PALETTE["REAL"] if labels[i] == 0 else PALETTE["FAKE"])
            axes[i].axis("off")
        for j in range(n, len(axes)):
            axes[j].axis("off")
        fig.suptitle(title, fontsize=13, fontweight="bold")
        _save(fig, save_path)


def plot_mean_image(
    mean_real: np.ndarray, mean_fake: np.ndarray, save_path: str
) -> None:
    with plt.style.context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.5))
        ax1.imshow(mean_real.clip(0, 1))
        ax1.set_title("Mean REAL", color=PALETTE["REAL"])
        ax2.imshow(mean_fake.clip(0, 1))
        ax2.set_title("Mean FAKE", color=PALETTE["FAKE"])
        for ax in (ax1, ax2):
            ax.axis("off")
        fig.suptitle("Mean images per class", fontweight="bold")
        _save(fig, save_path)


def plot_pixel_histograms(
    real_imgs: np.ndarray, fake_imgs: np.ndarray, save_path: str
) -> None:
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        ch_names = ["R", "G", "B"]
        for ci, ax in enumerate(axes):
            ax.hist(real_imgs[:, :, :, ci].flatten(), bins=64, density=True,
                    alpha=0.6, label="REAL", color=PALETTE["REAL"])
            ax.hist(fake_imgs[:, :, :, ci].flatten(), bins=64, density=True,
                    alpha=0.6, label="FAKE", color=PALETTE["FAKE"])
            ax.set_title(f"Channel {ch_names[ci]}")
            ax.legend(fontsize=8)
        fig.suptitle("Pixel value distributions per channel", fontweight="bold")
        _save(fig, save_path)


def plot_mean_fft(
    real_imgs: np.ndarray, fake_imgs: np.ndarray, save_path: str
) -> None:
    from scipy.fft import fft2, fftshift
    def mean_log_spectrum(imgs):
        acc = None
        for img in imgs:
            g = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            s = np.log1p(np.abs(fftshift(fft2(g))) ** 2)
            acc = s if acc is None else acc + s
        return acc / len(imgs)

    real_spec = mean_log_spectrum(real_imgs.astype(np.float32) / 255)
    fake_spec = mean_log_spectrum(fake_imgs.astype(np.float32) / 255)

    with plt.style.context(STYLE):
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12, 3.5))
        vmin = min(real_spec.min(), fake_spec.min())
        vmax = max(real_spec.max(), fake_spec.max())
        a1.imshow(real_spec, cmap="inferno", vmin=vmin, vmax=vmax)
        a1.set_title("Mean FFT — REAL")
        a2.imshow(fake_spec, cmap="inferno", vmin=vmin, vmax=vmax)
        a2.set_title("Mean FFT — FAKE")
        diff = fake_spec - real_spec
        a3.imshow(diff, cmap="RdBu_r", vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
        a3.set_title("Diff (FAKE - REAL)")
        for ax in (a1, a2, a3):
            ax.axis("off")
        fig.suptitle("Mean log-power FFT spectra", fontweight="bold")
        _save(fig, save_path)


# ─── Feature analysis ─────────────────────────────────────────────────────────

def plot_feature_separation(
    ks_stats: np.ndarray,
    feature_names: List[str],
    save_path: str,
    top_n: int = 30,
) -> None:
    with plt.style.context(STYLE):
        idx = np.argsort(ks_stats)[::-1][:top_n]
        fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.25)))
        ax.barh([feature_names[i] for i in idx[::-1]], ks_stats[idx[::-1]],
                color="#5A9FD4")
        ax.set_xlabel("KS statistic (real vs fake)")
        ax.set_title(f"Top-{top_n} most discriminative features", fontweight="bold")
        _save(fig, save_path)


# ─── PCA ──────────────────────────────────────────────────────────────────────

def plot_scree(explained_var_ratio: np.ndarray, save_path: str) -> None:
    with plt.style.context(STYLE):
        cumvar = np.cumsum(explained_var_ratio)
        k95 = int(np.searchsorted(cumvar, 0.95)) + 1
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(range(1, len(explained_var_ratio) + 1), explained_var_ratio, color="#5A9FD4")
        ax1.set_xlabel("Component")
        ax1.set_ylabel("Explained variance ratio")
        ax1.set_title("Scree plot")
        ax2.plot(range(1, len(cumvar) + 1), cumvar, color="#E8644C", linewidth=2)
        ax2.axhline(0.95, linestyle="--", color="grey", linewidth=1)
        ax2.axvline(k95, linestyle="--", color="grey", linewidth=1)
        ax2.set_xlabel("Number of components")
        ax2.set_ylabel("Cumulative variance")
        ax2.set_title(f"95 % variance → k={k95}")
        fig.suptitle("PCA — Variance explained", fontweight="bold")
        _save(fig, save_path)


def plot_pca_scatter(
    X_2d: np.ndarray, labels: np.ndarray, save_path: str, title: str = "PCA 2-D"
) -> None:
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        for lbl, name, col in [(0, "REAL", PALETTE["REAL"]), (1, "FAKE", PALETTE["FAKE"])]:
            mask = labels == lbl
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=5, alpha=0.4,
                       color=col, label=name, rasterized=True)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        ax.set_title(title, fontweight="bold")
        _save(fig, save_path)


def plot_pca_loadings(
    components: np.ndarray,
    feature_names: List[str],
    save_path: str,
    n_pcs: int = 3,
    top_n: int = 10,
) -> None:
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, n_pcs, figsize=(5 * n_pcs, 4))
        if n_pcs == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            if i >= len(components):
                break
            loading = np.abs(components[i])
            idx = np.argsort(loading)[::-1][:top_n]
            ax.barh([feature_names[j] for j in idx[::-1]], loading[idx[::-1]], color="#7B9FE8")
            ax.set_title(f"PC{i+1} top loadings")
        fig.suptitle("PCA loadings", fontweight="bold")
        _save(fig, save_path)


# ─── Classification ───────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: list, label_names: List[str], save_path: str, title: str = "") -> None:
    cm_arr = np.array(cm)
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title or "Confusion Matrix", fontweight="bold")
        _save(fig, save_path)


def plot_roc_curves(results: dict, save_path: str) -> None:
    from sklearn.metrics import roc_curve
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = plt.cm.tab10.colors
        for ci, (name, (y_true, y_prob)) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = np.trapezoid(tpr, fpr)
            ax.plot(fpr, tpr, color=colors[ci % 10], linewidth=2,
                    label=f"{name} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=9)
        ax.set_title("ROC Curves", fontweight="bold")
        _save(fig, save_path)


def plot_mlp_loss(loss_curve: List[float], save_path: str) -> None:
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(loss_curve, color="#E8644C", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("MLP training loss curve", fontweight="bold")
        _save(fig, save_path)


def plot_error_grid(imgs: np.ndarray, labels: np.ndarray, preds: np.ndarray,
                    kind: str, save_path: str, n: int = 16) -> None:
    mask = (preds != labels) & (labels == (0 if kind == "FN" else 1))
    idx = np.where(mask)[0][:n]
    if len(idx) == 0:
        return
    cols = 8
    rows = (len(idx) + cols - 1) // cols
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for i, ai in enumerate(idx):
            axes[i].imshow(imgs[ai])
            axes[i].set_title(f"T:{labels[ai]} P:{preds[ai]}", fontsize=7)
            axes[i].axis("off")
        for j in range(len(idx), len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"Error analysis — {kind}", fontweight="bold")
        _save(fig, save_path)


def plot_ablation(ablation: dict, save_path: str) -> None:
    names = list(ablation.keys())
    accs = [ablation[n] for n in names]
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(names, accs, color=["#5A9FD4" if n != "full" else "#E8644C" for n in names])
        ax.set_ylabel("Test accuracy")
        ax.set_title("Feature-group ablation study", fontweight="bold")
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.002, f"{acc:.3f}",
                    ha="center", va="bottom", fontsize=8)
        plt.xticks(rotation=15, ha="right")
        _save(fig, save_path)


# ─── Clustering ───────────────────────────────────────────────────────────────

def plot_elbow(inertias: List[float], k_range: range, save_path: str) -> None:
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(list(k_range), inertias, marker="o", color="#5A9FD4", linewidth=2)
        ax.set_xlabel("k")
        ax.set_ylabel("Inertia (WCSS)")
        ax.set_title("K-Means elbow curve", fontweight="bold")
        _save(fig, save_path)


def plot_silhouette(scores: List[float], k_range: range, save_path: str) -> None:
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(list(k_range), scores, marker="s", color="#E8644C", linewidth=2)
        ax.set_xlabel("k")
        ax.set_ylabel("Silhouette score")
        ax.set_title("Silhouette scores", fontweight="bold")
        _save(fig, save_path)


def plot_cluster_scatter(
    X_2d: np.ndarray,
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    save_path: str,
) -> None:
    with plt.style.context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        k = len(np.unique(cluster_labels))
        cmap = matplotlib.colormaps["tab10"].resampled(k)
        for ki in range(k):
            m = cluster_labels == ki
            ax1.scatter(X_2d[m, 0], X_2d[m, 1], s=5, alpha=0.4,
                        color=cmap(ki), label=f"C{ki}", rasterized=True)
        ax1.set_title("Clusters (K-Means)")
        ax1.legend(markerscale=3, fontsize=7)

        for lbl, name, col in [(0, "REAL", PALETTE["REAL"]), (1, "FAKE", PALETTE["FAKE"])]:
            m = true_labels == lbl
            ax2.scatter(X_2d[m, 0], X_2d[m, 1], s=5, alpha=0.4,
                        color=col, label=name, rasterized=True)
        ax2.set_title("True labels (REAL / FAKE)")
        ax2.legend(markerscale=3, fontsize=7)
        fig.suptitle("PCA 2-D — cluster vs label", fontweight="bold")
        _save(fig, save_path)


# ─── Watermark ────────────────────────────────────────────────────────────────

def plot_watermark_grid(
    covers: np.ndarray,
    marked: np.ndarray,
    save_path: str,
    n: int = 8,
    amplify: float = 10.0,
) -> None:
    """cover / marked / residual×10 grid."""
    n = min(n, len(covers))
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(3, n, figsize=(n * 1.5, 5))
        for i in range(n):
            residual = np.clip((marked[i] - covers[i]) * amplify + 0.5, 0, 1)
            axes[0, i].imshow(covers[i].clip(0, 1))
            axes[1, i].imshow(marked[i].clip(0, 1))
            axes[2, i].imshow(residual)
            for r in range(3):
                axes[r, i].axis("off")
        axes[0, 0].set_ylabel("Cover", fontsize=9)
        axes[1, 0].set_ylabel("Marked", fontsize=9)
        axes[2, 0].set_ylabel(f"Residual ×{amplify:.0f}", fontsize=9)
        fig.suptitle("Watermark visual quality", fontweight="bold")
        _save(fig, save_path)


def plot_training_curves(
    train_losses: List[float],
    bit_accs: List[float],
    psnrs: List[float],
    save_path: str,
) -> None:
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(train_losses, color="#E8644C", linewidth=2)
        axes[0].set_title("Training loss")
        axes[1].plot(bit_accs, color="#5A9FD4", linewidth=2)
        axes[1].axhline(0.9, color="grey", linestyle="--")
        axes[1].set_title("Bit accuracy")
        axes[2].plot(psnrs, color="#7BBF8E", linewidth=2)
        axes[2].axhline(30, color="grey", linestyle="--")
        axes[2].set_title("PSNR (dB)")
        for ax in axes:
            ax.set_xlabel("Epoch")
        fig.suptitle("Watermark training curves", fontweight="bold")
        _save(fig, save_path)
