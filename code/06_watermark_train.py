"""
06_watermark_train.py — Phase 6: Train Encoder/Decoder CNN watermark model.

Usage:
    python 06_watermark_train.py [--config configs/config.yaml] [--subset N]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from utils.image_utils import set_seed, setup_logging
from utils.metrics import bit_accuracy, psnr as compute_psnr
from utils.plots import plot_training_curves
from utils.wm_model import HybridMarkerWM


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_dataloader(imgs: np.ndarray, batch_size: int, seed: int, shuffle: bool = True):
    """Minimal numpy-based dataloader (no torch DataLoader dependency)."""
    N = len(imgs)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(N) if shuffle else np.arange(N)
    for start in range(0, N, batch_size):
        batch_idx = indices[start:start + batch_size]
        b = imgs[batch_idx]
        yield b.astype(np.float32) / 255.0 if b.dtype == np.uint8 else b


def main() -> None:
    parser = argparse.ArgumentParser(description="HybridMarker — Watermark Training")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--tag", default="", help="suffix for checkpoint/log names (sweeps)")
    parser.add_argument("--bits", type=int, default=None, help="override watermark.message_bits")
    parser.add_argument("--alpha", type=float, default=None, help="override watermark.strength_alpha")
    parser.add_argument("--lam-img", type=float, default=None, help="override watermark.lambda_img")
    parser.add_argument("--epochs", type=int, default=None, help="override watermark.epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    paths = cfg["paths"]
    wm_cfg = cfg.get("watermark", {})

    out_feat   = Path(paths["features"])
    out_models = Path(paths["models"])
    out_fig    = Path(paths["figures"])
    out_logs   = Path(paths["logs"])
    for d in [out_models, out_fig, out_logs]:
        d.mkdir(parents=True, exist_ok=True)

    sfx = f"_{args.tag}" if args.tag else ""
    setup_logging(out_logs / f"06_watermark_train{sfx}.log")
    log = logging.getLogger("wm_train")
    set_seed(seed)

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        log.error("PyTorch not installed. Skipping watermark training.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Hyperparameters ───────────────────────────────────────────────────────
    L       = args.bits if args.bits is not None else wm_cfg.get("message_bits", 48)
    alpha   = args.alpha if args.alpha is not None else wm_cfg.get("strength_alpha", 1.0)
    lam_img = args.lam_img if args.lam_img is not None else wm_cfg.get("lambda_img", 1.0)
    lam_msg = wm_cfg.get("lambda_msg", 1.0)
    bs      = wm_cfg.get("batch_size", 128)
    epochs  = args.epochs if args.epochs is not None else wm_cfg.get("epochs", 30)
    lr_val  = wm_cfg.get("lr", 1e-4)
    warmup  = wm_cfg.get("curriculum_warmup_epochs", 5)
    ckpt_every = wm_cfg.get("checkpoint_every", 1)
    noise_probs = wm_cfg.get("noise_probs", None)
    log.info("Config: bits=%d  alpha=%.2f  lam_img=%.2f  lam_msg=%.2f  bs=%d  epochs=%d",
             L, alpha, lam_img, lam_msg, bs, epochs)

    # ── Load images ───────────────────────────────────────────────────────────
    log.info("Loading image arrays …")
    X_tr   = np.load(out_feat / "X_img_train.npy")   # uint8; batches converted to [0,1]
    X_val  = np.load(out_feat / "X_img_val.npy")

    if args.subset:
        n = min(args.subset, len(X_tr))
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_tr), size=n, replace=False)
        X_tr = X_tr[idx]
        log.info("Using train subset: %d images", n)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = HybridMarkerWM(message_bits=L, alpha=alpha,
                           noise_probs=noise_probs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_val)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    # ── Overfit-tiny-batch sanity test ────────────────────────────────────────
    log.info("Sanity: overfit tiny batch …")
    tiny = torch.from_numpy(X_tr[:8].astype(np.float32).transpose(0, 3, 1, 2) / 255.0).to(device)  # (8,3,32,32)
    msg = torch.randint(0, 2, (8, L), dtype=torch.float32).to(device)
    for _ in range(20):
        optimizer.zero_grad()
        marked, noised, logits = model(tiny, msg, epoch=0)
        loss = lam_img * mse_loss(marked, tiny) + lam_msg * bce_loss(logits, msg)
        loss.backward()
        optimizer.step()
    preds = (torch.sigmoid(logits) > 0.5).float()
    ba = (preds == msg).float().mean().item()
    log.info("Sanity bit-acc after 20 steps: %.4f", ba)

    # ── Training loop ──────────────────────────────────────────────────────────
    train_losses, val_bit_accs, val_psnrs = [], [], []
    best_val_ba = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        in_warmup = epoch < warmup

        for batch_imgs in make_dataloader(X_tr, bs, seed + epoch, shuffle=True):
            covers = torch.from_numpy(batch_imgs.transpose(0, 3, 1, 2)).to(device)
            msgs   = torch.randint(0, 2, (len(covers), L), dtype=torch.float32).to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                if in_warmup:
                    # Identity noise only during warmup
                    marked = model.encoder(covers, msgs)
                    logits = model.decoder(marked)
                    noised = marked
                else:
                    marked, noised, logits = model(covers, msgs, epoch=epoch)
                loss = lam_img * mse_loss(marked, covers) + lam_msg * bce_loss(logits, msgs)

            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        val_ba_list, val_psnr_list = [], []
        with torch.no_grad():
            for batch_imgs in make_dataloader(X_val, bs * 2, seed, shuffle=False):
                covers = torch.from_numpy(batch_imgs.transpose(0, 3, 1, 2)).to(device)
                msgs   = torch.randint(0, 2, (len(covers), L), dtype=torch.float32).to(device)
                marked = model.encoder(covers, msgs)
                logits = model.decoder(marked)
                preds  = (torch.sigmoid(logits) > 0.5).float()
                val_ba_list.append((preds == msgs).float().mean().item())
                m_np = marked.cpu().numpy().transpose(0, 2, 3, 1)
                c_np = covers.cpu().numpy().transpose(0, 2, 3, 1)
                val_psnr_list.append(compute_psnr(c_np, m_np))

        val_ba   = float(np.mean(val_ba_list))
        val_psnr = float(np.mean(val_psnr_list))
        val_bit_accs.append(val_ba)
        val_psnrs.append(val_psnr)

        log.info("Epoch %3d/%d  loss=%.4f  val_bit_acc=%.4f  val_psnr=%.2f dB",
                 epoch + 1, epochs, avg_loss, val_ba, val_psnr)

        # Checkpoint
        if epoch % ckpt_every == 0 or epoch == epochs - 1:
            torch.save(model.encoder.state_dict(), out_models / f"wm_enc_last{sfx}.pt")
            torch.save(model.decoder.state_dict(), out_models / f"wm_dec_last{sfx}.pt")

        if val_ba > best_val_ba:
            best_val_ba = val_ba
            best_epoch = epoch
            torch.save(model.encoder.state_dict(), out_models / f"wm_enc{sfx}.pt")
            torch.save(model.decoder.state_dict(), out_models / f"wm_dec{sfx}.pt")
            log.info("  → Best checkpoint saved (bit_acc=%.4f)", val_ba)

        # Fallback gate check
        if epoch == epochs - 1 and val_ba < 0.85 and L > 16:
            log.warning("Val bit-acc %.4f < 0.85 with L=%d — consider reducing message_bits.", val_ba, L)

    # ── Save curves + info ────────────────────────────────────────────────────
    plot_training_curves(train_losses, val_bit_accs, val_psnrs,
                         str(out_fig / f"wm_training_curves{sfx}.png"))

    wm_train_info = {
        "message_bits": L,
        "best_epoch": best_epoch,
        "best_val_bit_acc": best_val_ba,
        "best_val_psnr": val_psnrs[best_epoch] if val_psnrs else None,
        "strength_alpha": alpha,
        "lambda_img": lam_img,
        "train_images": int(len(X_tr)),
        "final_val_psnr": val_psnrs[-1] if val_psnrs else None,
    }
    with open(out_models / f"wm_train_info{sfx}.json", "w") as f:
        json.dump(wm_train_info, f, indent=2)

    log.info("Training complete. Best epoch=%d  val_bit_acc=%.4f", best_epoch, best_val_ba)
    print(f"\n[06_watermark_train] Done. Best val bit-acc={best_val_ba:.4f} at epoch {best_epoch}.")


if __name__ == "__main__":
    main()
