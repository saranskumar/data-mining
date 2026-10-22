"""Download and organise all datasets into this folder (idempotent; re-run to resume).

Layout produced:
  dataset/cifake/{train,test}/{REAL,FAKE}/*.png|jpg   <- CIFAKE (HF mirror of the Kaggle set)
  dataset/modern/julienlucas_nbpro/data/*.parquet      <- Midjourney/DALL-E/SD/Nano Banana Pro, 12k imgs
  dataset/modern/mllm_gptimage2_nb2/images/<Gen>/<Category>/*.png   <- sampled subset, see N_PER_FOLDER
  dataset/DATASETS.md                                  <- sources, licences, label conventions

Not downloaded (needs your own login / terms acceptance):
  Shanmuk4622/ai-image-detection-dataset (HF gated, ~24 GB), Kaggle 'Midjourney CIFAKE-Inspired'.
"""
import io
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent
N_PER_FOLDER = int(os.environ.get("N_PER_FOLDER", 200))  # per generator x category in zr-zhang set


def cifake():
    out = ROOT / "cifake"
    if (out / "test" / "FAKE").exists() and len(list((out / "test" / "FAKE").iterdir())) >= 10000:
        print("cifake: already done")
        return
    import pyarrow.parquet as pq
    from PIL import Image

    src = ROOT / "_hf_cache" / "cifake"
    snapshot_download("dragonintelligence/CIFAKE-image-dataset", repo_type="dataset", local_dir=src)
    # NOTE: this mirror uses label 0=FAKE, 1=REAL (opposite of our project convention 0=real, 1=fake).
    names = {0: "FAKE", 1: "REAL"}
    for split in ("train", "test"):
        for f in sorted((src / "data").glob(f"{split}-*.parquet")):
            tbl = pq.read_table(f).to_pylist()
            for i, row in enumerate(tbl):
                d = out / split / names[row["label"]]
                d.mkdir(parents=True, exist_ok=True)
                b = row["image"]["bytes"]
                ext = "png" if b[:4] == b"\x89PNG" else "jpg"
                (d / f"{split}_{names[row['label']].lower()}_{i:06d}.{ext}").write_bytes(b)
        print("cifake", split, "written")


def julienlucas():
    snapshot_download("julienlucas/midjourney-dalle-sd-nanobananapro-dataset", repo_type="dataset",
                      local_dir=ROOT / "modern" / "julienlucas_nbpro", allow_patterns=["data/*", "README.md"])
    print("julienlucas_nbpro done")


def mllm():
    repo = "zr-zhang/MLLM-Generated-Image-Detection-Dataset"
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    by_folder = {}
    for f in files:
        if f.startswith("images/"):
            by_folder.setdefault(f.rsplit("/", 1)[0], []).append(f)
    pick = ["README.md"]
    for folder, fs in sorted(by_folder.items()):
        pick += sorted(fs)[:N_PER_FOLDER]  # deterministic first-N
    print(f"mllm: {len(by_folder)} folders, {len(pick)} files (N_PER_FOLDER={N_PER_FOLDER})")
    snapshot_download(repo, repo_type="dataset", local_dir=ROOT / "modern" / "mllm_gptimage2_nb2",
                      allow_patterns=pick, max_workers=8)
    print("mllm done")


NOTES = """# Datasets

| Folder | Source | Licence | Notes |
|---|---|---|---|
| cifake/ | HF mirror `dragonintelligence/CIFAKE-image-dataset` of Kaggle CIFAKE (Bird & Lotfi) | MIT (as CIFAR-10) | 100k train / 20k test, 32x32. Mirror labels were 0=FAKE,1=REAL; folders here are named REAL/FAKE so the project's 0=real,1=fake mapping applies. |
| modern/julienlucas_nbpro/ | `julienlucas/midjourney-dalle-sd-nanobananapro-dataset` | MIT | parquet; 10k train + 2k test; Midjourney/DALL-E/SD/Nano Banana Pro. Check columns with `code/data_ood/build_modern.py --inspect`. |
| modern/mllm_gptimage2_nb2/ | `zr-zhang/MLLM-Generated-Image-Detection-Dataset` (arXiv 2608.01258) | Apache-2.0 (card tag) | Generators: FLUX.2, GPT Image 2, Nano Banana 2, Qwen Image 2.1, plus Real. Categories: Texture/Structure/Hybrid. Only the first N files per folder downloaded. Label = folder name (Real vs others). |

Not downloaded: `Shanmuk4622/ai-image-detection-dataset` (HF gated, ~24.5 GB, accept terms with your HF login), Kaggle 'Midjourney CIFAKE-Inspired' (needs Kaggle token).
"""

if __name__ == "__main__":
    which = sys.argv[1:] or ["cifake", "julienlucas", "mllm"]
    for w in which:
        globals()[w]()
    (ROOT / "DATASETS.md").write_text(NOTES, encoding="utf-8")
