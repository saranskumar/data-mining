"""Build the demo's offline sample gallery (app/samples/) from the local datasets.

Picks a few CIFAKE test images (real + fake) and one image per modern generator, saved as PNG
(full resolution for the modern ones, so the demo shows the 32x32 reduction honestly).
Run: python code/app/make_samples.py
"""
import json
import random
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "dataset"
OUT = HERE / "samples"


def main(seed: int = 7, n_cifake: int = 4):
    rng = random.Random(seed)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    items = []
    for cls in ("REAL", "FAKE"):
        files = sorted((DATA / "cifake" / "test" / cls).iterdir())
        for f in rng.sample(files, n_cifake):
            dst = f"cifake_{cls.lower()}_{f.stem}{f.suffix}"
            shutil.copy(f, OUT / dst)
            items.append(dict(file=dst, name=f.stem, label="real" if cls == "REAL" else "AI",
                              source="CIFAKE test (SD 1.4)"))
    mllm = DATA / "modern" / "mllm_gptimage2_nb2" / "images"
    if mllm.exists():
        from PIL import Image
        for gen in sorted(p for p in mllm.iterdir() if p.is_dir()):
            files = sorted(f for c in gen.iterdir() if c.is_dir() for f in c.iterdir())
            f = rng.choice(files)
            im = Image.open(f).convert("RGB")
            im.thumbnail((512, 512))  # keep the gallery small
            dst = f"modern_{gen.name.replace(' ', '_').replace('.', '')}_{f.stem}.png"
            im.save(OUT / dst)
            items.append(dict(file=dst, name=f"{gen.name} {f.stem}",
                              label="real" if gen.name.lower() == "real" else "AI",
                              source=f"modern set — {gen.name}"))
    json.dump(items, open(OUT / "samples.json", "w"), indent=2)
    print(f"{len(items)} samples written to {OUT}")




def add_watermarked(message: str = "HybridMarker"):
    """Add watermarked copies of one real photo (PNG + JPEG-75 re-save) so the Decode tab can be
    shown offline, and register the message so Decode can name it."""
    import hashlib
    import io
    import sys
    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0, str(HERE.parent))
    from utils.wm_apply import embed_verified
    from utils.wm_model import DecoderCNN, EncoderCNN
    models = HERE.parent / "outputs" / "models"
    info = json.load(open(models / "wm_train_info.json"))
    L = info["message_bits"]
    enc = EncoderCNN(L, info["strength_alpha"], info.get("lum_only", False))
    dec = DecoderCNN(L)
    enc.load_state_dict(torch.load(models / "wm_enc.pt", map_location="cpu"))
    dec.load_state_dict(torch.load(models / "wm_dec.pt", map_location="cpu"))
    enc.eval(); dec.eval()
    h = hashlib.sha256(message.encode()).digest()
    bits = np.array([(h[i // 8] >> (7 - i % 8)) & 1 for i in range(L)])
    items = json.load(open(OUT / "samples.json"))
    src = next(i for i in items if i["file"].startswith("modern_GPT"))  # textured photo-like image
    x = np.asarray(Image.open(OUT / src["file"]).convert("RGB"), dtype=np.float32) / 255
    marked, _, used, ok = embed_verified(x, bits, enc, dec, "cpu", strength=0.75, adaptive=True)
    pil = Image.fromarray((marked * 255).round().clip(0, 255).astype(np.uint8))
    pil.save(OUT / "watermarked_png.png")
    pil.save(OUT / "watermarked_jpeg75.jpg", quality=75)
    items = [i for i in items if not i["file"].startswith("watermarked_")]
    items += [dict(file="watermarked_png.png", name=f"“{message}” (PNG)", label="watermarked", source="Watermark tab output"),
              dict(file="watermarked_jpeg75.jpg", name=f"“{message}” (re-saved JPEG 75)", label="watermarked",
                   source="Watermark tab output, re-compressed")]
    json.dump(items, open(OUT / "samples.json", "w"), indent=2)
    reg_p = HERE.parent / "outputs" / "wm_registry.json"
    reg = json.load(open(reg_p)) if reg_p.exists() else {}
    reg[message] = "".join(map(str, bits))
    json.dump(reg, open(reg_p, "w"), indent=2)
    print(f"watermarked samples added (strength used {used:.2f}, verified={ok})")


if __name__ == "__main__":
    main()
    add_watermarked()
