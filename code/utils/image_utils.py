"""
utils/image_utils.py — image I/O, normalisation, seeding helpers.
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Fix numpy + random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except (ImportError, OSError):  # torch absent or broken: forensic stages don't need it
        pass


def load_images(
    directory: str | Path,
    label: int,
    target_size: Tuple[int, int] = (32, 32),
    subset: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all JPEG/PNG images from *directory*, force RGB & target_size.
    Returns:
        imgs_uint8   – (N, H, W, 3) uint8
        labels       – (N,) int32
        paths        – list of file paths (for deduplication)
    Corrupt files are logged and skipped.
    """
    directory = Path(directory)
    paths = sorted(directory.glob("*.jpg")) + sorted(directory.glob("*.png"))
    if subset is not None:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(paths), size=min(subset, len(paths)), replace=False)
        paths = [paths[i] for i in sorted(idx)]

    imgs, labels_list, str_paths = [], [], []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            if img.size != target_size:
                img = img.resize(target_size, Image.BILINEAR)
            imgs.append(np.array(img, dtype=np.uint8))
            labels_list.append(label)
            str_paths.append(str(p))
        except Exception as exc:
            logger.warning("Skipping %s — %s", p, exc)

    imgs_arr = np.stack(imgs, axis=0) if imgs else np.zeros((0, *target_size, 3), dtype=np.uint8)
    labels_arr = np.array(labels_list, dtype=np.int32)
    return imgs_arr, labels_arr, str_paths


def img_md5(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()


def find_duplicates(hashes: List[str]) -> dict:
    """Return {hash: [indices]} for duplicate hashes."""
    from collections import defaultdict
    d: dict = defaultdict(list)
    for i, h in enumerate(hashes):
        d[h].append(i)
    return {h: idxs for h, idxs in d.items() if len(idxs) > 1}


def normalize(imgs_uint8: np.ndarray) -> np.ndarray:
    """Convert uint8 (N, H, W, 3) → float32 [0, 1]."""
    return imgs_uint8.astype(np.float32) / 255.0


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split X, y into train / val preserving class ratio."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_frac, stratify=y, random_state=seed
    )
    return X_tr, X_val, y_tr, y_val


def setup_logging(log_path: str | Path, level: int = logging.INFO) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        handlers=handlers,
        force=True,
    )
