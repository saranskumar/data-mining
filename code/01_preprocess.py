"""
01_preprocess.py — Phase 1: Load CIFAKE, validate, deduplicate, split, EDA.

Usage:
    python 01_preprocess.py [--config configs/config.yaml] [--subset N]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Allow running from code/ directory
sys.path.insert(0, str(Path(__file__).parent))

from utils.image_utils import (
    find_duplicates, img_md5, load_images, normalize,
    set_seed, setup_logging, stratified_split,
)
from utils.plots import (
    plot_mean_fft, plot_mean_image, plot_pixel_histograms, plot_sample_grid,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="HybridMarker — Preprocessing & EDA")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--subset", type=int, default=None, help="Per-class subset size for dev")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    paths = cfg["paths"]
    preproc = cfg["preprocessing"]

    subset = args.subset or preproc.get("subset") or None
    out_feat = Path(paths["features"])
    out_fig = Path(paths["figures"])
    out_tables = Path(paths["tables"])
    out_logs = Path(paths["logs"])
    for d in [out_feat, out_fig, out_tables, out_logs]:
        d.mkdir(parents=True, exist_ok=True)

    setup_logging(out_logs / "01_preprocess.log")
    log = logging.getLogger("preprocess")
    set_seed(seed)

    # ── Load images ────────────────────────────────────────────────────────────
    t0 = time.time()
    per_class_subset = subset  # per-class limit
    log.info("Loading TRAIN/REAL …")
    tr_real, y_tr_real, paths_tr_real = load_images(
        Path(paths["train_real"]), label=0, subset=per_class_subset)
    log.info("Loading TRAIN/FAKE …")
    tr_fake, y_tr_fake, paths_tr_fake = load_images(
        Path(paths["train_fake"]), label=1, subset=per_class_subset)
    log.info("Loading TEST/REAL …")
    te_real, y_te_real, _ = load_images(
        Path(paths["test_real"]), label=0, subset=per_class_subset)
    log.info("Loading TEST/FAKE …")
    te_fake, y_te_fake, _ = load_images(
        Path(paths["test_fake"]), label=1, subset=per_class_subset)

    X_train_raw = np.concatenate([tr_real, tr_fake], axis=0)
    y_train_raw = np.concatenate([y_tr_real, y_tr_fake], axis=0)
    X_test_raw  = np.concatenate([te_real, te_fake],  axis=0)
    y_test_raw  = np.concatenate([y_te_real, y_te_fake], axis=0)

    log.info("Train: %d  Test: %d", len(X_train_raw), len(X_test_raw))

    # ── Duplicate detection ────────────────────────────────────────────────────
    log.info("Computing MD5 hashes …")
    train_hashes = [img_md5(img) for img in X_train_raw]
    test_hashes  = [img_md5(img) for img in X_test_raw]
    intra_train_dups = find_duplicates(train_hashes)
    intra_test_dups  = find_duplicates(test_hashes)
    cross_set = set(train_hashes) & set(test_hashes)

    with open(out_tables / "data_quality.txt", "w") as f:
        f.write(f"Train size: {len(X_train_raw)}\n")
        f.write(f"Test size:  {len(X_test_raw)}\n")
        f.write(f"Intra-train duplicate groups: {len(intra_train_dups)}\n")
        f.write(f"Intra-test  duplicate groups: {len(intra_test_dups)}\n")
        f.write(f"Train-test overlap hashes:   {len(cross_set)}\n")
    log.info("Duplicate report written → data_quality.txt")
    log.info("Train-test overlap: %d hashes", len(cross_set))

    # Remove test images that also appear in train
    if cross_set:
        keep = [i for i, h in enumerate(test_hashes) if h not in cross_set]
        X_test_raw = X_test_raw[keep]
        y_test_raw = y_test_raw[keep]
        log.info("Removed %d test duplicates; test size now %d",
                 len(test_hashes) - len(keep), len(X_test_raw))

    # ── Shuffle train ──────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_train_raw))
    X_train_raw = X_train_raw[perm]
    y_train_raw = y_train_raw[perm]

    # ── Train / Val split ──────────────────────────────────────────────────────
    val_frac = preproc.get("val_split", 0.10)
    X_tr, X_val, y_tr, y_val = stratified_split(
        X_train_raw, y_train_raw, val_frac=val_frac, seed=seed)
    log.info("Split → train: %d  val: %d  test: %d", len(X_tr), len(X_val), len(X_test_raw))

    # uint8 arrays only: float32 [0,1] copies are made on demand (batch_extract, WM batches)
    # so a full 120k run fits in a few GB of RAM.

    # ── Save arrays ───────────────────────────────────────────────────────────
    np.save(out_feat / "X_img_train.npy",   X_tr)
    np.save(out_feat / "X_img_val.npy",     X_val)
    np.save(out_feat / "X_img_test.npy",    X_test_raw)
    np.save(out_feat / "y_train.npy",        y_tr)
    np.save(out_feat / "y_val.npy",          y_val)
    np.save(out_feat / "y_test.npy",         y_test_raw)
    log.info("Arrays saved to %s", out_feat)

    # ── EDA figures ───────────────────────────────────────────────────────────
    log.info("Generating EDA figures …")
    # Sample grid
    idx = rng.choice(len(X_tr), size=min(32, len(X_tr)), replace=False)
    plot_sample_grid(X_tr[idx], y_tr[idx], str(out_fig / "eda_sample_grid.png"))

    # Mean images
    real_mask = y_tr == 0
    fake_mask = y_tr == 1
    mean_real = X_tr[real_mask].mean(axis=0, dtype=np.float64) / 255.0
    mean_fake = X_tr[fake_mask].mean(axis=0, dtype=np.float64) / 255.0
    plot_mean_image(mean_real, mean_fake, str(out_fig / "eda_mean_images.png"))

    # Pixel histograms (sample 2000 images max for speed)
    n_eda = min(2000, real_mask.sum(), fake_mask.sum())
    plot_pixel_histograms(
        X_tr[real_mask][:n_eda], X_tr[fake_mask][:n_eda],
        str(out_fig / "eda_pixel_hist.png"),
    )

    # Mean FFT
    plot_mean_fft(
        X_tr[real_mask][:n_eda], X_tr[fake_mask][:n_eda],
        str(out_fig / "eda_mean_fft.png"),
    )

    # Class balance
    total = len(y_tr)
    log.info("Class balance — REAL: %d (%.1f%%)  FAKE: %d (%.1f%%)",
             real_mask.sum(), 100 * real_mask.sum() / total,
             fake_mask.sum(), 100 * fake_mask.sum() / total)

    log.info("Phase 1 done in %.1f s", time.time() - t0)
    print("\n[01_preprocess] Done. Arrays + EDA figures saved.")
    print(f"  train={len(X_tr)}  val={len(X_val)}  test={len(X_test_raw)}")


if __name__ == "__main__":
    main()
