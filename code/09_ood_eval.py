"""
09_ood_eval.py — Evaluate the saved forensic models on the modern-generator OOD set.

Needs: data_ood/out/{X_img_ood,y_ood}.npy + meta_ood.csv (python data_ood/build_modern.py)
       outputs/models/{scaler,pca,<model>}.joblib (04_classify.py)

Reports accuracy / recall-on-fakes (detection rate) / false-positive rate per generator and per
source, plus the drop versus the CIFAKE test accuracy. Test-only: nothing here is trained or tuned.
Output: outputs/tables/ood.csv, outputs/tables/ood_by_generator.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils.feature_utils import batch_extract
from utils.image_utils import set_seed, setup_logging

MODELS = ["logreg", "dt_gini", "dt_entropy", "mlp"]


def main() -> None:
    ap = argparse.ArgumentParser(description="HybridMarker — OOD evaluation")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ood-dir", default="data_ood/out")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    paths = cfg["paths"]
    out_models, out_tables, out_feat = Path(paths["models"]), Path(paths["tables"]), Path(paths["features"])
    out_tables.mkdir(parents=True, exist_ok=True)
    setup_logging(Path(paths["logs"]) / "09_ood_eval.log")
    log = logging.getLogger("ood_eval")
    set_seed(cfg["seed"])

    d = Path(args.ood_dir)
    X = np.load(d / "X_img_ood.npy").astype(np.float32) / 255.0
    y = np.load(d / "y_ood.npy")
    meta = pd.read_csv(d / "meta_ood.csv")
    log.info("OOD set: %s  real=%d fake=%d", X.shape, int((y == 0).sum()), int((y == 1).sum()))

    scaler = joblib.load(out_models / "scaler.joblib")
    pca = joblib.load(out_models / "pca.joblib")
    feats, _ = batch_extract(X, cfg=cfg.get("features", {}), n_workers=args.workers)
    Z = pca.transform(scaler.transform(feats))

    # In-distribution reference (CIFAKE test accuracy) from 04_classify, if present.
    ref = {}
    cls_csv = out_tables / "classification.csv"
    if cls_csv.exists():
        ref = pd.read_csv(cls_csv).set_index("model")["accuracy"].to_dict()

    summary, by_gen = [], []
    for name in MODELS:
        p = out_models / f"{name}.joblib"
        if not p.exists():
            log.warning("missing model %s — skipped", name)
            continue
        pred = joblib.load(p).predict(Z)
        real, fake = y == 0, y == 1
        # balanced accuracy: the OOD set has many more fakes than reals
        tnr = float((pred[real] == 0).mean())
        tpr = float((pred[fake] == 1).mean())
        summary.append(dict(model=name, ood_balanced_acc=(tnr + tpr) / 2, detection_rate_tpr=tpr,
                            false_positive_rate=1 - tnr, cifake_test_acc=ref.get(name),
                            drop=(ref.get(name) - (tnr + tpr) / 2) if name in ref else None))
        # group by label too: a source can hold real and fake rows under one generator name
        for (src, gen, lab), idx in meta.groupby(["source", "generator", "label"]).groups.items():
            m = np.zeros(len(y), bool)
            m[list(idx)] = True
            is_fake = bool(lab)
            by_gen.append(dict(model=name, source=src, generator=gen if is_fake else f"{gen} (real)", n=int(m.sum()),
                               rate=float((pred[m] == (1 if is_fake else 0)).mean()),
                               metric="detection_rate" if is_fake else "specificity"))
        log.info("%-10s bal-acc=%.3f TPR=%.3f FPR=%.3f", name, (tnr + tpr) / 2, tpr, 1 - tnr)

    pd.DataFrame(summary).to_csv(out_tables / "ood.csv", index=False)
    pd.DataFrame(by_gen).to_csv(out_tables / "ood_by_generator.csv", index=False)
    log.info("saved ood.csv, ood_by_generator.csv")
    print("\n[09_ood_eval] Done.")
    print(pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
