"""
02_feature_extraction.py — Phase 2: Extract feature matrix from image arrays.

Usage:
    python 02_feature_extraction.py [--config configs/config.yaml] [--subset N] [--workers 4]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent))

from utils.feature_utils import batch_extract, extract_all, save_feature_names
from utils.image_utils import set_seed, setup_logging
from utils.plots import plot_feature_separation


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="HybridMarker — Feature Extraction")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    paths = cfg["paths"]
    feat_cfg = cfg.get("features", {})

    out_feat = Path(paths["features"])
    out_fig  = Path(paths["figures"])
    out_logs = Path(paths["logs"])
    for d in [out_feat, out_fig, out_logs]:
        d.mkdir(parents=True, exist_ok=True)

    setup_logging(out_logs / "02_feature_extraction.log")
    log = logging.getLogger("feature_extraction")
    set_seed(seed)

    # ── Load preprocessed image arrays ────────────────────────────────────────
    log.info("Loading image arrays …")
    X_tr_f   = np.load(out_feat / "X_img_train.npy")
    X_val_f  = np.load(out_feat / "X_img_val.npy")
    X_test_f = np.load(out_feat / "X_img_test.npy")
    y_tr     = np.load(out_feat / "y_train.npy")
    y_val    = np.load(out_feat / "y_val.npy")
    y_test   = np.load(out_feat / "y_test.npy")

    # Optional subset
    if args.subset:
        n = min(args.subset, len(X_tr_f))
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_tr_f), size=n, replace=False)
        X_tr_f = X_tr_f[idx]; y_tr = y_tr[idx]
        np.save(out_feat / "y_train.npy", y_tr)  # keep labels aligned with the subsetted features
        log.info("Subsetting train to %d samples (y_train.npy overwritten to match)", n)

    log.info("Shapes — train: %s  val: %s  test: %s",
             X_tr_f.shape, X_val_f.shape, X_test_f.shape)

    # ── Feature extraction ────────────────────────────────────────────────────
    n_workers = args.workers

    log.info("Extracting train features with %d workers …", n_workers)
    t0 = time.time()
    X_feat_tr, feature_names = batch_extract(X_tr_f, cfg=feat_cfg, n_workers=n_workers)
    log.info("Train features extracted: %s  (%.1f s)", X_feat_tr.shape, time.time() - t0)

    log.info("Extracting val features …")
    X_feat_val, _ = batch_extract(X_val_f, cfg=feat_cfg, n_workers=n_workers)

    log.info("Extracting test features …")
    X_feat_test, _ = batch_extract(X_test_f, cfg=feat_cfg, n_workers=n_workers)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    D = len(feature_names)
    assert X_feat_tr.shape[1] == D, f"Feature dim mismatch: {X_feat_tr.shape[1]} vs {D}"
    assert not np.any(np.isnan(X_feat_tr)), "NaN in train features"
    assert not np.any(np.isinf(X_feat_tr)), "Inf in train features"
    log.info("Sanity checks passed. D = %d features", D)

    # Check for constant columns
    const_cols = np.where(X_feat_tr.std(axis=0) < 1e-8)[0]
    if len(const_cols) > 0:
        log.warning("%d constant feature columns: %s", len(const_cols),
                    [feature_names[i] for i in const_cols[:10]])

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(out_feat / "X_feat_train.npy", X_feat_tr)
    np.save(out_feat / "X_feat_val.npy",   X_feat_val)
    np.save(out_feat / "X_feat_test.npy",  X_feat_test)
    save_feature_names(feature_names, out_feat / "feature_names.json")
    log.info("Feature arrays + names saved to %s", out_feat)

    # ── Per-feature class separation (KS statistic) ───────────────────────────
    log.info("Computing KS statistics for feature separation …")
    real_mask = y_tr == 0
    fake_mask = y_tr == 1
    ks_stats = np.array([
        scipy_stats.ks_2samp(
            X_feat_tr[real_mask, i], X_feat_tr[fake_mask, i]
        ).statistic
        for i in range(D)
    ])
    np.save(out_feat / "ks_stats.npy", ks_stats)
    plot_feature_separation(ks_stats, feature_names,
                            str(out_fig / "feature_separation.png"), top_n=30)
    log.info("Top-5 most discriminative features: %s",
             [feature_names[i] for i in np.argsort(ks_stats)[::-1][:5]])

    log.info("Phase 2 done in %.1f s total", time.time() - t0)
    print(f"\n[02_feature_extraction] Done. D={D} features saved.")


if __name__ == "__main__":
    main()
