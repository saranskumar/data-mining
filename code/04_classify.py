"""
04_classify.py — Phase 4: Classification (DT-Gini, DT-IG, MLP) + evaluation.

Usage:
    python 04_classify.py [--config configs/config.yaml]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).parent))

from utils.feature_utils import load_feature_names
from utils.image_utils import set_seed, setup_logging
from utils.metrics import classification_report_dict
from utils.plots import (
    plot_ablation, plot_confusion_matrix, plot_error_grid,
    plot_mlp_loss, plot_roc_curves,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_model(name, model, X_test, y_test, out_fig, results_rows, roc_data):
    t0 = time.time()
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None

    metrics = classification_report_dict(y_test, y_pred, y_prob)
    metrics["model"] = name
    metrics["time_s"] = round(time.time() - t0, 3)
    results_rows.append(metrics)

    plot_confusion_matrix(
        metrics["confusion_matrix"], ["REAL", "FAKE"],
        str(out_fig / f"cm_{name}.png"), title=f"Confusion Matrix — {name}"
    )

    if y_prob is not None:
        roc_data[name] = (y_test, y_prob)

    return metrics


def ablation_study(X_tr_pca, y_tr, X_val_pca, y_val, X_test_pca, y_test,
                   feature_names, X_tr_feat, X_val_feat, X_test_feat,
                   best_dt_params, best_mlp_params, seed, log):
    """Drop each feature group, retrain best DT and MLP, record val accuracy."""
    groups = {
        "noise": [i for i, n in enumerate(feature_names) if n.startswith("noise_")],
        "freq":  [i for i, n in enumerate(feature_names) if n.startswith(("fft_", "dct_"))],
        "color": [i for i, n in enumerate(feature_names) if n.startswith(("rgb_", "hsv_"))],
        "glcm":  [i for i, n in enumerate(feature_names) if n.startswith("glcm_")],
    }

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    ablation = {}
    all_indices = list(range(len(feature_names)))

    for dropped_group, drop_idx in [("full", [])] + [(g, gi) for g, gi in groups.items()]:
        keep = [i for i in all_indices if i not in drop_idx]
        if not keep:
            continue
        X_tr_sub = X_tr_feat[:, keep]
        X_val_sub = X_val_feat[:, keep]

        sc = StandardScaler()
        X_tr_sub_s = sc.fit_transform(X_tr_sub)
        X_val_sub_s = sc.transform(X_val_sub)
        pca_sub = PCA(n_components=min(0.95, X_tr_sub_s.shape[1] - 1), random_state=seed)
        X_tr_sub_pca = pca_sub.fit_transform(X_tr_sub_s)
        X_val_sub_pca = pca_sub.transform(X_val_sub_s)

        dt = DecisionTreeClassifier(random_state=seed, **best_dt_params)
        dt.fit(X_tr_sub_pca, y_tr)
        acc = dt.score(X_val_sub_pca, y_val)
        ablation[f"no_{dropped_group}" if dropped_group != "full" else "full"] = acc
        log.info("Ablation drop=%s  val_acc=%.4f", dropped_group, acc)

    return ablation


def main() -> None:
    parser = argparse.ArgumentParser(description="HybridMarker — Classification")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="reuse saved dt_*.joblib instead of re-running their grid search")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    paths = cfg["paths"]
    cls_cfg = cfg.get("classification", {})

    out_feat   = Path(paths["features"])
    out_models = Path(paths["models"])
    out_fig    = Path(paths["figures"])
    out_tables = Path(paths["tables"])
    out_logs   = Path(paths["logs"])
    for d in [out_models, out_fig, out_tables, out_logs]:
        d.mkdir(parents=True, exist_ok=True)

    setup_logging(out_logs / "04_classify.log")
    log = logging.getLogger("classify")
    set_seed(seed)

    # ── Load PCA arrays ───────────────────────────────────────────────────────
    log.info("Loading arrays …")
    X_tr   = np.load(out_feat / "X_pca_train.npy")
    X_val  = np.load(out_feat / "X_pca_val.npy")
    X_test = np.load(out_feat / "X_pca_test.npy")
    y_tr   = np.load(out_feat / "y_train.npy")
    y_val  = np.load(out_feat / "y_val.npy")
    y_test = np.load(out_feat / "y_test.npy")

    # For pre-PCA tree importance
    X_tr_feat   = np.load(out_feat / "X_scaled_train.npy")
    X_val_feat  = np.load(out_feat / "X_scaled_val.npy")
    X_test_feat = np.load(out_feat / "X_scaled_test.npy")
    feature_names = load_feature_names(out_feat / "feature_names.json")

    # Combined train+val for final model training
    X_trainval     = np.concatenate([X_tr, X_val], axis=0)
    y_trainval     = np.concatenate([y_tr, y_val], axis=0)
    X_trainval_raw = np.concatenate([X_tr_feat, X_val_feat], axis=0)

    log.info("PCA shapes — train: %s  val: %s  test: %s",
             X_tr.shape, X_val.shape, X_test.shape)

    results_rows = []
    roc_data = {}
    cv_folds = cls_cfg.get("cv_folds", 5)

    # ── Baseline: Dummy ───────────────────────────────────────────────────────
    log.info("Training Dummy classifier …")
    dummy = DummyClassifier(strategy="most_frequent", random_state=seed)
    dummy.fit(X_trainval, y_trainval)
    evaluate_model("dummy", dummy, X_test, y_test, out_fig, results_rows, roc_data)

    # ── Baseline: Logistic Regression ─────────────────────────────────────────
    log.info("Training LogisticRegression …")
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X_trainval, y_trainval)
    evaluate_model("logreg", lr, X_test, y_test, out_fig, results_rows, roc_data)
    joblib.dump(lr, out_models / "logreg.joblib")

    # ── Decision Tree — grid on val ───────────────────────────────────────────
    dt_grid_cfg = cls_cfg.get("dt_param_grid", {})
    dt_param_grid = {
        "max_depth":        dt_grid_cfg.get("max_depth", [5, 10, 15, None]),
        "min_samples_leaf": dt_grid_cfg.get("min_samples_leaf", [1, 5, 10]),
        "ccp_alpha":        dt_grid_cfg.get("ccp_alpha", [0.0, 0.0001, 0.001]),
    }

    for criterion, tag in [("gini", "dt_gini"), ("entropy", "dt_entropy")]:
        saved = out_models / f"{tag}.joblib"
        if args.resume and saved.exists():
            prev = joblib.load(saved)
            best_params = {k: prev.get_params()[k] for k in ("max_depth", "min_samples_leaf", "ccp_alpha")}
            log.info("%s: --resume, reusing saved params %s", tag, best_params)
            grid = []
        else:
            log.info("Grid search for %s …", tag)
            grid = [(d, m, c) for d in dt_param_grid["max_depth"]
                    for m in dt_param_grid["min_samples_leaf"] for c in dt_param_grid["ccp_alpha"]]
        best_acc = -1
        for max_d, min_s, ccp in grid:
            dt = DecisionTreeClassifier(
                criterion=criterion, max_depth=max_d,
                min_samples_leaf=min_s, ccp_alpha=ccp,
                random_state=seed
            )
            dt.fit(X_tr, y_tr)
            acc = dt.score(X_val, y_val)
            if acc > best_acc:
                best_acc = acc
                best_params = {"max_depth": max_d, "min_samples_leaf": min_s, "ccp_alpha": ccp}

        if grid:
            log.info("%s best val acc=%.4f params=%s", tag, best_acc, best_params)
        # Refit on train+val
        dt_final = DecisionTreeClassifier(criterion=criterion, random_state=seed, **best_params)
        dt_final.fit(X_trainval, y_trainval)
        m = evaluate_model(tag, dt_final, X_test, y_test, out_fig, results_rows, roc_data)

        # CV on trainval
        cv_scores = cross_val_score(dt_final, X_trainval, y_trainval, cv=cv_folds, scoring="accuracy")
        m["cv_mean"] = float(cv_scores.mean())
        m["cv_std"]  = float(cv_scores.std())
        log.info("%s test acc=%.4f  CV=%.4f±%.4f", tag, m["accuracy"], m["cv_mean"], m["cv_std"])

        joblib.dump(dt_final, out_models / f"{tag}.joblib")

        # Export tree text (top 3 levels)
        tree_txt = export_text(dt_final, max_depth=3)
        with open(out_tables / f"{tag}_top3levels.txt", "w") as f:
            f.write(tree_txt)

        if tag == "dt_gini":
            best_dt_params_for_ablation = best_params

    # ── Pre-PCA tree for named feature importances ─────────────────────────────
    log.info("Training pre-PCA tree for feature importances …")
    dt_prepca = DecisionTreeClassifier(
        criterion="gini", max_depth=10, min_samples_leaf=5, random_state=seed)
    dt_prepca.fit(
        np.concatenate([X_tr_feat, X_val_feat], axis=0),
        y_trainval
    )
    importances = dt_prepca.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    top_importance = {feature_names[i]: float(importances[i]) for i in top_idx}
    with open(out_tables / "feature_importances_prepca.json", "w") as f:
        json.dump(top_importance, f, indent=2)
    log.info("Top feature importances: %s", list(top_importance.items())[:5])

    # ── MLP ───────────────────────────────────────────────────────────────────
    log.info("Grid search for MLP …")
    mlp_grid_cfg = cls_cfg.get("mlp_param_grid", {})
    mlp_param_grid = {
        "hidden_layer_sizes": [tuple(h) for h in mlp_grid_cfg.get(
            "hidden_layer_sizes", [[64], [128, 64], [256, 128, 64]])],
        "alpha":              mlp_grid_cfg.get("alpha", [0.0001, 0.001, 0.01]),
    }
    max_iter = cls_cfg.get("mlp_max_iter", 500)

    best_acc, best_mlp_params = -1, {}
    for hls in mlp_param_grid["hidden_layer_sizes"]:
        for alpha in mlp_param_grid["alpha"]:
            mlp = MLPClassifier(hidden_layer_sizes=hls, alpha=alpha,
                                solver="adam", early_stopping=True,
                                max_iter=max_iter, random_state=seed)
            mlp.fit(X_tr, y_tr)
            acc = mlp.score(X_val, y_val)
            if acc > best_acc:
                best_acc = acc
                best_mlp_params = {"hidden_layer_sizes": hls, "alpha": alpha}

    log.info("MLP best val acc=%.4f params=%s", best_acc, best_mlp_params)
    mlp_final = MLPClassifier(
        solver="adam", early_stopping=True, max_iter=max_iter,
        random_state=seed, **best_mlp_params
    )
    mlp_final.fit(X_trainval, y_trainval)
    m = evaluate_model("mlp", mlp_final, X_test, y_test, out_fig, results_rows, roc_data)

    cv_scores = cross_val_score(mlp_final, X_trainval, y_trainval, cv=cv_folds, scoring="accuracy")
    m["cv_mean"] = float(cv_scores.mean())
    m["cv_std"]  = float(cv_scores.std())
    log.info("MLP test acc=%.4f  CV=%.4f±%.4f", m["accuracy"], m["cv_mean"], m["cv_std"])

    plot_mlp_loss(list(mlp_final.loss_curve_), str(out_fig / "mlp_loss_curve.png"))
    joblib.dump(mlp_final, out_models / "mlp.joblib")

    # ── McNemar test (DT-Gini vs MLP) ─────────────────────────────────────────
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        dt_g = joblib.load(out_models / "dt_gini.joblib")
        dt_preds  = dt_g.predict(X_test)
        mlp_preds = mlp_final.predict(X_test)
        b = int(np.sum((dt_preds == y_test) & (mlp_preds != y_test)))
        c = int(np.sum((dt_preds != y_test) & (mlp_preds == y_test)))
        table = [[0, b], [c, 0]]
        result = mcnemar(table, exact=True)
        log.info("McNemar DT-Gini vs MLP: stat=%.4f  p=%.4f", result.statistic, result.pvalue)
        with open(out_tables / "mcnemar.json", "w") as f:
            json.dump({"statistic": result.statistic, "pvalue": result.pvalue,
                       "b": b, "c": c}, f, indent=2)
    except Exception as e:
        log.warning("McNemar test failed: %s", e)

    # ── Ablation ──────────────────────────────────────────────────────────────
    log.info("Running feature-group ablation …")
    ablation = ablation_study(
        X_tr, y_tr, X_val, y_val, X_test, y_test,
        feature_names, X_tr_feat, X_val_feat, X_test_feat,
        best_dt_params_for_ablation, best_mlp_params, seed, log
    )
    plot_ablation(ablation, str(out_fig / "ablation.png"))
    with open(out_tables / "ablation.json", "w") as f:
        json.dump(ablation, f, indent=2)

    # ── Error analysis ─────────────────────────────────────────────────────────
    X_img_test = np.load(out_feat / "X_img_test.npy")
    dt_g = joblib.load(out_models / "dt_gini.joblib")
    dt_preds_all = dt_g.predict(X_test)
    plot_error_grid(X_img_test, y_test, dt_preds_all, "FP",
                    str(out_fig / "error_false_positives.png"))
    plot_error_grid(X_img_test, y_test, dt_preds_all, "FN",
                    str(out_fig / "error_false_negatives.png"))

    # ── ROC curves ────────────────────────────────────────────────────────────
    if roc_data:
        plot_roc_curves(roc_data, str(out_fig / "roc_curves.png"))

    # ── Save results table ────────────────────────────────────────────────────
    if results_rows:
        fieldnames = list({k for row in results_rows for k in row.keys() if k != "confusion_matrix"})
        with open(out_tables / "classification.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results_rows)
    log.info("Results saved to classification.csv")

    log.info("Phase 4 complete.")
    print("\n[04_classify] Done. Models + tables saved.")


if __name__ == "__main__":
    main()
