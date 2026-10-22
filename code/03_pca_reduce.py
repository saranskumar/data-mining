"""
03_pca_reduce.py — Phase 3: Scale + PCA dimensionality reduction.

Usage:
    python 03_pca_reduce.py [--config configs/config.yaml]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from utils.feature_utils import load_feature_names
from utils.image_utils import set_seed, setup_logging
from utils.plots import plot_pca_loadings, plot_pca_scatter, plot_scree


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="HybridMarker — PCA Reduction")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    paths = cfg["paths"]
    pca_cfg = cfg.get("pca", {})

    out_feat   = Path(paths["features"])
    out_models = Path(paths["models"])
    out_fig    = Path(paths["figures"])
    out_logs   = Path(paths["logs"])
    for d in [out_feat, out_models, out_fig, out_logs]:
        d.mkdir(parents=True, exist_ok=True)

    setup_logging(out_logs / "03_pca_reduce.log")
    log = logging.getLogger("pca_reduce")
    set_seed(seed)

    # ── Load ──────────────────────────────────────────────────────────────────
    log.info("Loading feature arrays …")
    X_tr    = np.load(out_feat / "X_feat_train.npy")
    X_val   = np.load(out_feat / "X_feat_val.npy")
    X_test  = np.load(out_feat / "X_feat_test.npy")
    y_tr    = np.load(out_feat / "y_train.npy")
    y_val   = np.load(out_feat / "y_val.npy")
    y_test  = np.load(out_feat / "y_test.npy")
    feature_names = load_feature_names(out_feat / "feature_names.json")
    D = X_tr.shape[1]
    log.info("D=%d  train=%d  val=%d  test=%d", D, len(X_tr), len(X_val), len(X_test))

    # ── Scale (fit on TRAIN ONLY) ─────────────────────────────────────────────
    scaler = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_val_s  = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    log.info("StandardScaler fit on train only. Mean ≈ %.4f  Std ≈ %.4f",
             X_tr_s.mean(), X_tr_s.std())

    # Verify: scaler was fit only on train rows
    assert scaler.n_samples_seen_ == len(X_tr), "Scaler fitted on wrong data!"

    # ── PCA (fit on TRAIN ONLY) ───────────────────────────────────────────────
    var_ret = pca_cfg.get("variance_retained", 0.95)
    solver  = pca_cfg.get("svd_solver", "full")
    pca = PCA(n_components=var_ret, svd_solver=solver, random_state=seed)
    X_pca_tr   = pca.fit_transform(X_tr_s)
    X_pca_val  = pca.transform(X_val_s)
    X_pca_test = pca.transform(X_test_s)
    k = pca.n_components_
    log.info("PCA: D=%d → k=%d (retaining %.3f variance)", D, k, var_ret)

    # ── Diagnostics figures ───────────────────────────────────────────────────
    plot_scree(pca.explained_variance_ratio_, str(out_fig / "pca_scree.png"))
    log.info("Scree plot saved")

    # 2-D scatter using first 2 PCs
    X_2d = np.concatenate([X_pca_tr[:, :2], X_pca_val[:, :2]], axis=0)
    y_all = np.concatenate([y_tr, y_val], axis=0)
    plot_pca_scatter(X_2d, y_all, str(out_fig / "pca_scatter.png"),
                     title=f"PCA 2-D scatter (k={k})")
    log.info("PCA scatter saved")

    plot_pca_loadings(pca.components_, feature_names,
                      str(out_fig / "pca_loadings.png"), n_pcs=3, top_n=10)
    log.info("PCA loadings saved")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(out_feat / "X_pca_train.npy",  X_pca_tr)
    np.save(out_feat / "X_pca_val.npy",    X_pca_val)
    np.save(out_feat / "X_pca_test.npy",   X_pca_test)
    np.save(out_feat / "X_scaled_train.npy", X_tr_s)   # for pre-PCA tree
    np.save(out_feat / "X_scaled_val.npy",   X_val_s)
    np.save(out_feat / "X_scaled_test.npy",  X_test_s)
    joblib.dump(scaler, out_models / "scaler.joblib")
    joblib.dump(pca,    out_models / "pca.joblib")

    # Save reduction info
    info = {"D": int(D), "k": int(k), "variance_retained": float(var_ret),
            "explained_variance": float(sum(pca.explained_variance_ratio_))}
    with open(out_feat / "pca_info.json", "w") as f:
        json.dump(info, f, indent=2)
    log.info("PCA artifacts saved. D=%d → k=%d", D, k)

    print(f"\n[03_pca_reduce] Done. D={D} → k={k} components.")


if __name__ == "__main__":
    main()
