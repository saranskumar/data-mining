"""
05_cluster.py — Phase 5: K-Means clustering on PCA features.

Usage:
    python 05_cluster.py [--config configs/config.yaml]
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent))

from utils.image_utils import set_seed, setup_logging
from utils.metrics import clustering_metrics
from utils.plots import (
    plot_cluster_scatter, plot_elbow, plot_silhouette,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="HybridMarker — Clustering")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    paths = cfg["paths"]
    cl_cfg = cfg.get("clustering", {})

    out_feat   = Path(paths["features"])
    out_fig    = Path(paths["figures"])
    out_tables = Path(paths["tables"])
    out_logs   = Path(paths["logs"])
    for d in [out_fig, out_tables, out_logs]:
        d.mkdir(parents=True, exist_ok=True)

    setup_logging(out_logs / "05_cluster.log")
    log = logging.getLogger("cluster")
    set_seed(seed)

    # ── Load PCA arrays (combine train + test for clustering) ─────────────────
    X_tr    = np.load(out_feat / "X_pca_train.npy")
    X_test  = np.load(out_feat / "X_pca_test.npy")
    y_tr    = np.load(out_feat / "y_train.npy")
    y_test  = np.load(out_feat / "y_test.npy")

    # Use train only for sweep (faster), combined for final scatter
    X_all = np.concatenate([X_tr, X_test], axis=0)
    y_all = np.concatenate([y_tr, y_test], axis=0)
    log.info("Clustering on %d samples (train + test), PCA shape: %s", len(X_all), X_all.shape)

    k_min = cl_cfg.get("k_min", 2)
    k_max = cl_cfg.get("k_max", 12)
    n_init = cl_cfg.get("n_init", 10)
    k_range = range(k_min, k_max + 1)

    # ── K sweep ───────────────────────────────────────────────────────────────
    inertias, silhouette_scores = [], []
    results_rows = []

    t0 = time.time()
    for k in k_range:
        log.info("Fitting K-Means k=%d …", k)
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
        cluster_labels = km.fit_predict(X_all)
        inertia = km.inertia_
        sil = silhouette_score(X_all, cluster_labels, sample_size=min(5000, len(X_all)),
                               random_state=seed)
        metrics = clustering_metrics(y_all, cluster_labels)
        inertias.append(inertia)
        silhouette_scores.append(sil)
        row = {"k": k, "inertia": inertia, "silhouette": sil,
               **{f"rf_{k2}": v for k2, v in metrics.items()}}
        results_rows.append(row)
        log.info("k=%d  inertia=%.1f  sil=%.4f  purity=%.4f  ARI=%.4f  NMI=%.4f",
                 k, inertia, sil, metrics["purity"], metrics["ARI"], metrics["NMI"])

    log.info("K sweep done in %.1f s", time.time() - t0)

    # ── Plots: elbow + silhouette ──────────────────────────────────────────────
    plot_elbow(inertias, k_range, str(out_fig / "cluster_elbow.png"))
    plot_silhouette(silhouette_scores, k_range, str(out_fig / "cluster_silhouette.png"))

    # ── Chosen k (best silhouette) ────────────────────────────────────────────
    best_k = k_min + int(np.argmax(silhouette_scores))
    log.info("Best k by silhouette: %d (score=%.4f)", best_k, max(silhouette_scores))

    km_final = KMeans(n_clusters=best_k, n_init=n_init, random_state=seed)
    final_labels = km_final.fit_predict(X_all)

    # ── 2-D scatter ───────────────────────────────────────────────────────────
    X_2d = X_all[:, :2]
    plot_cluster_scatter(X_2d, final_labels, y_all, str(out_fig / "cluster_scatter.png"))

    # ── Save results ─────────────────────────────────────────────────────────
    if results_rows:
        fieldnames = list(results_rows[0].keys())
        with open(out_tables / "clustering.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_rows)

    np.save(out_feat / "cluster_labels.npy", final_labels)
    log.info("Clustering results saved.")

    # ── Narrative ─────────────────────────────────────────────────────────────
    best_metrics = results_rows[best_k - k_min]
    narrative = (
        f"K-Means sweep k={k_min}–{k_max}. "
        f"Best k={best_k} by silhouette ({max(silhouette_scores):.4f}). "
        f"Purity vs real/fake: {best_metrics['rf_purity']:.4f}, "
        f"ARI: {best_metrics['rf_ARI']:.4f}, NMI: {best_metrics['rf_NMI']:.4f}. "
        f"{'Clusters weakly track real/fake.' if best_metrics['rf_ARI'] < 0.1 else 'Clusters show some alignment with real/fake label.'}"
    )
    with open(out_tables / "clustering_narrative.txt", "w") as f:
        f.write(narrative)
    log.info("Narrative: %s", narrative)
    print(f"\n[05_cluster] Done. Best k={best_k}. Tables + figures saved.")


if __name__ == "__main__":
    main()
