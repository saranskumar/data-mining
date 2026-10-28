"""
07b_watermark_fullres_eval.py — watermark quality/robustness on FULL-RESOLUTION photos.

The model is trained at 32x32, but users mark real photos. This embeds via utils/wm_apply.py
(residual upsampled to the original size) and measures, per strength:
  PSNR / SSIM at full size, bit accuracy + detection rate clean and after JPEG / resize / crop,
  and the false-positive rate on unmarked photos.
Images: real photos from dataset/modern (long side capped at --max-side for speed).
Output: outputs/tables/watermark_fullres.csv, outputs/figures/wm_fullres_example.png
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from utils.image_utils import set_seed, setup_logging
from utils.metrics import binomial_threshold
from utils.wm_apply import decode_fullres, embed_fullres, embed_verified, psnr

DATA = Path(__file__).resolve().parents[1] / "dataset" / "modern" / "mllm_gptimage2_nb2" / "images" / "Real"


def jpeg(img, q):
    buf = io.BytesIO()
    Image.fromarray((img * 255).round().astype(np.uint8)).save(buf, format="JPEG", quality=q)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0


def resize_half(img):
    h, w = img.shape[:2]
    pil = Image.fromarray((img * 255).round().astype(np.uint8))
    return np.asarray(pil.resize((max(1, w // 2), max(1, h // 2)), Image.BICUBIC), dtype=np.float32) / 255.0


def crop80(img):
    h, w = img.shape[:2]
    ch, cw = int(h * 0.8), int(w * 0.8)
    t, l = (h - ch) // 2, (w - cw) // 2
    return img[t:t + ch, l:l + cw]


ATTACKS = {"clean": lambda x: x, "jpeg_90": lambda x: jpeg(x, 90), "jpeg_70": lambda x: jpeg(x, 70),
           "jpeg_50": lambda x: jpeg(x, 50), "resize_0.5": resize_half, "crop_80": crop80}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--strengths", default="1.0,0.75,0.5")
    ap.add_argument("--bits", type=int, default=None, help="override (e.g. while training is still running)")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--tag", default="", help="evaluate a sweep checkpoint (wm_enc_<tag>.pt)")
    ap.add_argument("--modes", default="uniform,adaptive,verified",
                    help="uniform | adaptive (texture-masked) | verified (adaptive + embed-and-verify boost)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    paths = cfg["paths"]
    out_models, out_tables, out_fig = Path(paths["models"]), Path(paths["tables"]), Path(paths["figures"])
    setup_logging(Path(paths["logs"]) / "07b_watermark_fullres_eval.log")
    log = logging.getLogger("wm_fullres")
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    import torch
    from skimage.metrics import structural_similarity
    from utils.wm_model import DecoderCNN, EncoderCNN

    sfx = f"_{args.tag}" if args.tag else ""
    info = json.load(open(out_models / f"wm_train_info{sfx}.json"))
    L = args.bits or info["message_bits"]
    alpha = args.alpha or info.get("strength_alpha", 1.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = EncoderCNN(L, alpha, info.get("lum_only", False)).to(device); dec = DecoderCNN(L).to(device)
    enc.load_state_dict(torch.load(out_models / f"wm_enc{sfx}.pt", map_location=device))
    dec.load_state_dict(torch.load(out_models / f"wm_dec{sfx}.pt", map_location=device))
    enc.eval(); dec.eval()
    thr = binomial_threshold(L, cfg.get("detection", {}).get("false_positive_rate_target", 1e-3))

    files = sorted(f for c in DATA.iterdir() if c.is_dir() for f in c.iterdir())
    files = [files[i] for i in rng.choice(len(files), size=min(args.n, len(files)), replace=False)]
    imgs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        im.thumbnail((args.max_side, args.max_side))
        imgs.append(np.asarray(im, dtype=np.float32) / 255.0)
    log.info("%d real photos, sizes e.g. %s; L=%d alpha=%.3f thr=%.4f", len(imgs), imgs[0].shape, L, alpha, thr)

    rows = []
    example = None
    combos = [(m, float(x)) for m in args.modes.split(",") for x in args.strengths.split(",")]
    for mode, s in combos:
        acc = {a: [] for a in ATTACKS}
        ps, ss, boosts = [], [], []
        for i, img in enumerate(imgs):
            bits = rng.integers(0, 2, L)
            if mode == "verified":
                marked, _, used, _ = embed_verified(img, bits, enc, dec, device, strength=s, adaptive=True)
                boosts.append(used / s)
            else:
                marked, _ = embed_fullres(img, bits, enc, device, strength=s, adaptive=(mode == "adaptive"))
            ps.append(psnr(img, marked))
            ss.append(structural_similarity(img, marked, channel_axis=2, data_range=1.0))
            for a, fn in ATTACKS.items():
                acc[a].append(float(np.mean(decode_fullres(fn(marked), dec, device) == bits)))
            if example is None and s == 1.0 and mode == "verified":
                example = (img, marked)
        for a in ATTACKS:
            ba = np.array(acc[a])
            rows.append(dict(mode=mode, strength=s, attack=a, psnr_db=float(np.mean(ps)), ssim=float(np.mean(ss)),
                             bit_acc=float(ba.mean()), detection_rate=float(np.mean(ba >= thr)),
                             boosted_share=float(np.mean(np.array(boosts) > 1.0001)) if boosts else 0.0))
        log.info("%s strength %.2f: PSNR %.2f dB  SSIM %.4f  clean bit-acc %.3f  jpeg70 %.3f",
                 mode, s, np.mean(ps), np.mean(ss), np.mean(acc["clean"]), np.mean(acc["jpeg_70"]))

    # false positives: unmarked photos vs random keys
    fp = [float(np.mean(decode_fullres(img, dec, device) == rng.integers(0, 2, L)) >= thr) for img in imgs]
    df = pd.DataFrame(rows)
    df["unmarked_fpr"] = float(np.mean(fp))
    df.to_csv(out_tables / f"watermark_fullres{sfx}.csv", index=False)
    log.info("unmarked FPR %.4f (target %.0e)", np.mean(fp), 1e-3)

    if example is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        img, marked = example
        diff = np.clip((marked - img) * 10 + 0.5, 0, 1)
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
        for a, im, t in zip(ax, [img, marked, diff], ["Original", f"Watermarked (PSNR {psnr(img, marked):.1f} dB)", "Difference ×10"]):
            a.imshow(im); a.set_title(t); a.axis("off")
        fig.tight_layout(); fig.savefig(out_fig / f"wm_fullres_example{sfx}.png", dpi=110); plt.close(fig)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
