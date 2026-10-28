"""
07_watermark_eval.py — Phase 7: Evaluate watermark robustness under real attacks.

Usage:
    python 07_watermark_eval.py [--config configs/config.yaml] [--subset N]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from utils.attacks import all_attack_names, apply_attack
from utils.image_utils import set_seed, setup_logging
from utils.metrics import (
    bit_accuracy, binomial_threshold, exact_match_rate, psnr as compute_psnr, ssim as compute_ssim,
)
from utils.plots import plot_watermark_grid
from utils.wm_model import DecoderCNN, EncoderCNN


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="HybridMarker — Watermark Evaluation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--subset", type=int, default=2000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    paths = cfg["paths"]
    wm_cfg = cfg.get("watermark", {})
    atk_cfg = cfg.get("attacks", {})
    det_cfg = cfg.get("detection", {})

    out_feat   = Path(paths["features"])
    out_models = Path(paths["models"])
    out_fig    = Path(paths["figures"])
    out_tables = Path(paths["tables"])
    out_logs   = Path(paths["logs"])
    for d in [out_fig, out_tables, out_logs]:
        d.mkdir(parents=True, exist_ok=True)

    setup_logging(out_logs / "07_watermark_eval.log")
    log = logging.getLogger("wm_eval")
    set_seed(seed)

    try:
        import torch
    except ImportError:
        log.error("PyTorch required.")
        sys.exit(1)

    # Load watermark info
    wm_info_path = out_models / "wm_train_info.json"
    if not wm_info_path.exists():
        log.error("No wm_train_info.json found. Run 06_watermark_train.py first.")
        sys.exit(1)
    with open(wm_info_path) as f:
        wm_info = json.load(f)
    L = wm_info.get("message_bits", wm_cfg.get("message_bits", 48))
    log.info("Evaluating watermark: L=%d bits", L)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha = wm_info.get("strength_alpha", wm_cfg.get("strength_alpha", 1.0))  # what the encoder was trained with

    enc = EncoderCNN(message_bits=L, strength_alpha=alpha, lum_only=wm_info.get("lum_only", False)).to(device)
    dec = DecoderCNN(message_bits=L).to(device)
    enc.load_state_dict(torch.load(out_models / "wm_enc.pt", map_location=device))
    dec.load_state_dict(torch.load(out_models / "wm_dec.pt", map_location=device))
    enc.eval(); dec.eval()
    log.info("Models loaded from %s", out_models)

    # ── Load test images ──────────────────────────────────────────────────────
    X_test_f = np.load(out_feat / "X_img_test.npy")
    rng = np.random.default_rng(seed)
    n = min(args.subset, len(X_test_f))
    idx = rng.choice(len(X_test_f), size=n, replace=False)
    X_eval = X_test_f[idx].astype(np.float32) / 255.0
    log.info("Evaluating on %d test images", n)

    def encode_decode_batch(imgs_np, msgs_np):
        covers = torch.from_numpy(imgs_np.transpose(0, 3, 1, 2)).float().to(device)
        msgs   = torch.from_numpy(msgs_np).float().to(device)
        with torch.no_grad():
            marked = enc(covers, msgs)
        marked_np = marked.cpu().numpy().transpose(0, 2, 3, 1)
        return marked_np

    def decode_batch(imgs_np):
        t = torch.from_numpy(imgs_np.transpose(0, 3, 1, 2)).float().to(device)
        with torch.no_grad():
            logits = dec(t)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.int32)
        return preds

    # Generate messages
    msgs_np = rng.integers(0, 2, size=(n, L)).astype(np.float32)

    # ── Encode ────────────────────────────────────────────────────────────────
    log.info("Encoding messages …")
    marked_np = encode_decode_batch(X_eval, msgs_np)

    # Visual grid (before attacks)
    plot_watermark_grid(X_eval, marked_np, str(out_fig / "wm_visual_grid.png"), n=8)
    log.info("Visual grid saved")

    # ── Eval across attacks ───────────────────────────────────────────────────
    fpr_target = det_cfg.get("false_positive_rate_target", 1e-3)
    threshold = binomial_threshold(L, fpr_target)
    log.info("Detection threshold bit-acc >= %.4f (FPR target=%.4f)", threshold, fpr_target)

    results_rows = []
    attack_names = all_attack_names()

    for atk_name in attack_names:
        attacked = apply_attack(marked_np, atk_name, atk_cfg)
        preds = decode_batch(attacked)
        msgs_int = msgs_np.astype(np.int32)
        ba  = bit_accuracy(msgs_int, preds)
        emr = exact_match_rate(msgs_int, preds)
        p   = compute_psnr(X_eval, attacked)
        s   = compute_ssim(X_eval, attacked)
        # TPR: fraction where bit_acc >= threshold
        per_img_ba = np.mean(msgs_int == preds, axis=1)
        tpr = float(np.mean(per_img_ba >= threshold))
        row = {"attack": atk_name, "bit_acc": ba, "exact_match": emr,
               "psnr": p, "ssim": s, "tpr_at_threshold": tpr}
        results_rows.append(row)
        log.info("Attack=%-15s  bit_acc=%.4f  emr=%.4f  psnr=%.2f  ssim=%.4f  tpr=%.4f",
                 atk_name, ba, emr, p, s, tpr)

    # ── Binomial threshold calibration on unmarked images ─────────────────────
    log.info("Calibrating threshold on unmarked images …")
    unmarked_np = X_eval.copy()  # no watermark
    unmarked_preds = decode_batch(unmarked_np)
    unmarked_ba = np.mean(msgs_int == unmarked_preds, axis=1)
    empirical_fpr = float(np.mean(unmarked_ba >= threshold))
    log.info("Empirical FPR on unmarked: %.4f (target=%.4f)", empirical_fpr, fpr_target)

    with open(out_tables / "wm_threshold_info.json", "w") as f:
        json.dump({"threshold": threshold, "fpr_target": fpr_target,
                   "empirical_fpr": empirical_fpr}, f, indent=2)

    # ── Interaction: does watermark shift forensic classifier verdict? ─────────
    try:
        import joblib
        mlp = joblib.load(out_models / "mlp.joblib")
        scaler = joblib.load(out_models / "scaler.joblib")
        pca_model = joblib.load(out_models / "pca.joblib")
        from utils.feature_utils import batch_extract, load_feature_names
        feat_cfg = cfg.get("features", {})
        fn = load_feature_names(out_feat / "feature_names.json")
        X_orig_feat, _ = batch_extract(X_eval, cfg=feat_cfg, n_workers=2)
        X_mark_feat, _ = batch_extract(marked_np, cfg=feat_cfg, n_workers=2)
        X_orig_pca = pca_model.transform(scaler.transform(X_orig_feat))
        X_mark_pca = pca_model.transform(scaler.transform(X_mark_feat))
        orig_preds = mlp.predict(X_orig_pca)
        mark_preds = mlp.predict(X_mark_pca)
        shifted = float(np.mean(orig_preds != mark_preds))
        log.info("Interaction study: %.2f%% of verdicts shifted after watermarking", shifted * 100)
        with open(out_tables / "wm_interaction.json", "w") as f:
            json.dump({"verdict_shift_rate": shifted}, f, indent=2)
    except Exception as e:
        log.warning("Interaction study failed: %s", e)

    # ── Save table ────────────────────────────────────────────────────────────
    fieldnames = list(results_rows[0].keys())
    with open(out_tables / "watermark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)
    log.info("watermark.csv saved")

    print("\n[07_watermark_eval] Done.")
    print(f"  Threshold: {threshold:.4f} | Empirical FPR: {empirical_fpr:.4f}")


if __name__ == "__main__":
    main()
