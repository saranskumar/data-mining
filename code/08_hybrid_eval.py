"""
08_hybrid_eval.py — Phase 8: Evaluate hybrid forensic + watermark fusion rule.

Scores the rule-based fusion on 4 groups (real/fake × marked/unmarked) × all attacks.

Usage:
    python 08_hybrid_eval.py [--config configs/config.yaml] [--subset N]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from utils.attacks import all_attack_names, apply_attack
from utils.feature_utils import batch_extract, load_feature_names
from utils.image_utils import set_seed, setup_logging
from utils.metrics import binomial_threshold, bit_accuracy


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def fusion_verdict(p_fake: float, mark_detected: bool) -> str:
    """
    Rule-based fusion (from 01-architecture.md):
      watermark decoded → Marked / provenance verified
      no mark + p_fake >= 0.5 → Likely AI-generated
      no mark + p_fake < 0.5  → Likely real / unverified
    """
    if mark_detected:
        return "marked"
    elif p_fake >= 0.5:
        return "likely_ai"
    else:
        return "likely_real"


def main() -> None:
    parser = argparse.ArgumentParser(description="HybridMarker — Hybrid Evaluation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--subset", type=int, default=300)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    paths = cfg["paths"]
    wm_cfg = cfg.get("watermark", {})
    atk_cfg = cfg.get("attacks", {})
    det_cfg = cfg.get("detection", {})
    feat_cfg = cfg.get("features", {})

    out_feat   = Path(paths["features"])
    out_models = Path(paths["models"])
    out_tables = Path(paths["tables"])
    out_logs   = Path(paths["logs"])
    for d in [out_tables, out_logs]:
        d.mkdir(parents=True, exist_ok=True)

    setup_logging(out_logs / "08_hybrid_eval.log")
    log = logging.getLogger("hybrid_eval")
    set_seed(seed)

    try:
        import torch
        from utils.wm_model import DecoderCNN, EncoderCNN
    except ImportError:
        log.error("PyTorch required.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load models ───────────────────────────────────────────────────────────
    log.info("Loading forensic models …")
    mlp    = joblib.load(out_models / "mlp.joblib")
    scaler = joblib.load(out_models / "scaler.joblib")
    pca_m  = joblib.load(out_models / "pca.joblib")
    feature_names = load_feature_names(out_feat / "feature_names.json")

    with open(out_models / "wm_train_info.json") as f:
        wm_info = json.load(f)
    L = wm_info["message_bits"]
    alpha = wm_info.get("strength_alpha", wm_cfg.get("strength_alpha", 1.0))  # what the encoder was trained with
    fpr_target = det_cfg.get("false_positive_rate_target", 1e-3)
    threshold = binomial_threshold(L, fpr_target)

    enc = EncoderCNN(message_bits=L, strength_alpha=alpha).to(device)
    dec = DecoderCNN(message_bits=L).to(device)
    enc.load_state_dict(torch.load(out_models / "wm_enc.pt", map_location=device))
    dec.load_state_dict(torch.load(out_models / "wm_dec.pt", map_location=device))
    enc.eval(); dec.eval()

    # ── Load test images ──────────────────────────────────────────────────────
    X_test_f = np.load(out_feat / "X_img_test.npy")
    y_test   = np.load(out_feat / "y_test.npy")
    rng = np.random.default_rng(seed)
    n = min(args.subset, len(X_test_f))
    idx = rng.choice(len(X_test_f), size=n, replace=False)
    X_eval = X_test_f[idx].astype(np.float32) / 255.0
    y_eval = y_test[idx]
    log.info("Evaluating on %d images", n)

    msgs_np = rng.integers(0, 2, size=(n, L)).astype(np.float32)

    def encode(imgs_np, msgs_np):
        covers = torch.from_numpy(imgs_np.transpose(0, 3, 1, 2)).float().to(device)
        msgs   = torch.from_numpy(msgs_np).float().to(device)
        with torch.no_grad():
            marked = enc(covers, msgs)
        return marked.cpu().numpy().transpose(0, 2, 3, 1)

    def decode(imgs_np, msgs_int):
        t = torch.from_numpy(imgs_np.transpose(0, 3, 1, 2)).float().to(device)
        with torch.no_grad():
            logits = dec(t)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.int32)
        per_img_ba = np.mean(msgs_int == preds, axis=1)
        return per_img_ba >= threshold  # bool array: mark detected?

    def forensic_score(imgs_np):
        feats, _ = batch_extract(imgs_np, cfg=feat_cfg, n_workers=2)
        feats_s = scaler.transform(feats)
        feats_pca = pca_m.transform(feats_s)
        proba = mlp.predict_proba(feats_pca)[:, 1]
        return proba

    # ── Encode (create marked versions) ───────────────────────────────────────
    log.info("Encoding all eval images …")
    X_marked = encode(X_eval, msgs_np)
    msgs_int = msgs_np.astype(np.int32)

    # ── 4 groups × all attacks ────────────────────────────────────────────────
    groups = {
        "real_unmarked":  (X_eval[y_eval == 0],   False, y_eval[y_eval == 0],
                           msgs_int[y_eval == 0]),
        "fake_unmarked":  (X_eval[y_eval == 1],   False, y_eval[y_eval == 1],
                           msgs_int[y_eval == 1]),
        "real_marked":    (X_marked[y_eval == 0],  True,  y_eval[y_eval == 0],
                           msgs_int[y_eval == 0]),
        "fake_marked":    (X_marked[y_eval == 1],  True,  y_eval[y_eval == 1],
                           msgs_int[y_eval == 1]),
    }

    results_rows = []
    attack_names = all_attack_names()

    for atk_name in attack_names:
        for grp_name, (imgs, was_marked, y_grp, msgs_grp) in groups.items():
            if len(imgs) == 0:
                continue
            attacked = apply_attack(imgs, atk_name, atk_cfg)

            p_fakes = forensic_score(attacked)
            mark_dets = decode(attacked, msgs_grp)
            verdicts = [fusion_verdict(float(pf), bool(md))
                        for pf, md in zip(p_fakes, mark_dets)]

            # Ground truth: real_* should be "likely_real"; fake_* → "likely_ai"
            # marked_* → "marked"
            if was_marked:
                correct_verdict = "marked"
            elif grp_name.startswith("real"):
                correct_verdict = "likely_real"
            else:
                correct_verdict = "likely_ai"

            verdict_acc = float(np.mean(np.array(verdicts) == correct_verdict))
            row = {
                "attack": atk_name, "group": grp_name,
                "n": len(imgs), "correct_verdict": correct_verdict,
                "verdict_accuracy": verdict_acc,
                "mean_p_fake": float(p_fakes.mean()),
                "mark_detection_rate": float(mark_dets.mean()),
            }
            results_rows.append(row)
            log.info("atk=%-15s  grp=%-18s  verdict_acc=%.4f", atk_name, grp_name, verdict_acc)

    # ── Save ──────────────────────────────────────────────────────────────────
    fieldnames = list(results_rows[0].keys())
    with open(out_tables / "hybrid.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    log.info("hybrid.csv saved (%d rows)", len(results_rows))
    print("\n[08_hybrid_eval] Done. hybrid.csv saved.")


if __name__ == "__main__":
    main()
