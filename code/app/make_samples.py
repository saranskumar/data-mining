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


if __name__ == "__main__":
    main()
