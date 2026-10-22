"""Build the modern-generator out-of-distribution (OOD) set from the LOCAL downloads in dataset/modern.

Outputs (default code/data_ood/out/):
  X_img_ood.npy  uint8 (N,32,32,3)    y_ood.npy  int64 (0=real, 1=fake)    meta_ood.csv (source, generator, category, label)

Sources
  julienlucas_nbpro  parquet; label 0=fake, 1=real (inverted vs. project convention, handled here);
                     no per-generator column, so generator = "mj_dalle_sd_nbpro_mix".
  mllm_gptimage2_nb2 image folders: images/<Generator>/<Category>/*.{png,jpg}; folder "Real" = real.

All images are resized to 32x32 with antialiased bicubic (assumed equivalent of CIFAKE's downsizing).
Run:  python code/data_ood/build_modern.py [--per-source 1000] [--seed 42]
"""
import argparse
import io
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "dataset" / "modern"
RESAMPLE = Image.BICUBIC


def to32(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB").resize((32, 32), RESAMPLE), dtype=np.uint8)


def load_julienlucas(per_class, rng, split="test"):
    out = []
    files = sorted((DATA / "julienlucas_nbpro" / "data").glob(f"{split}-*.parquet"))
    rows = {0: [], 1: []}  # project label: 0 real, 1 fake
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg).to_pylist()
            for r in t:
                y = 0 if r["label"] == 1 else 1  # source: 0=fake,1=real
                rows[y].append(r["image"]["bytes"])
    for y, items in rows.items():
        rng.shuffle(items)
        for b in items[:per_class]:
            out.append((to32(Image.open(io.BytesIO(b))), y, "julienlucas_nbpro", "mj_dalle_sd_nbpro_mix", "mixed"))
    return out


def load_mllm(per_class, rng):
    root = DATA / "mllm_gptimage2_nb2" / "images"
    out = []
    for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        y = 0 if gen_dir.name.lower() == "real" else 1
        files = [(f, c.name) for c in sorted(gen_dir.iterdir()) if c.is_dir()
                 for f in sorted(c.iterdir()) if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
        rng.shuffle(files)
        for f, cat in files[:per_class]:
            try:
                out.append((to32(Image.open(f)), y, "mllm_gptimage2_nb2", gen_dir.name, cat))
            except Exception as e:
                print("skip", f, e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=1000,
                    help="cap per class (julienlucas) / per generator folder (mllm)")
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = load_julienlucas(a.per_source, rng) + load_mllm(a.per_source, rng)
    if not rows:
        raise SystemExit("no data found under " + str(DATA))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    X = np.stack([r[0] for r in rows])
    y = np.array([r[1] for r in rows], dtype=np.int64)
    meta = pd.DataFrame([dict(source=r[2], generator=r[3], category=r[4], label=r[1]) for r in rows])
    np.save(out / "X_img_ood.npy", X)
    np.save(out / "y_ood.npy", y)
    meta.to_csv(out / "meta_ood.csv", index=False)
    print("saved", X.shape, "->", out)
    print(meta.groupby(["source", "generator", "label"]).size())


if __name__ == "__main__":
    main()
