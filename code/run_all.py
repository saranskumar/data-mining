"""
run_all.py — Execute all pipeline stages in order.

Usage:
    python run_all.py [--config configs/config.yaml] [--subset N] [--skip-watermark]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(script: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, script] + extra_args
    print(f"\n{'='*60}\nRunning: {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        print(f"[FAILED] {script}")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="HybridMarker — Run full pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--subset", type=int, default=None,
                        help="Per-script subset size for dev runs")
    parser.add_argument("--skip-watermark", action="store_true",
                        help="Skip watermark training + eval (Track B)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    base_args = ["--config", args.config]
    subset_args = ["--subset", str(args.subset)] if args.subset else []

    stages = [
        ("01_preprocess.py",       base_args + subset_args),
        ("02_feature_extraction.py", base_args + ["--workers", str(args.workers)]),  # 01 already subsets
        ("03_pca_reduce.py",       base_args),
        ("04_classify.py",         base_args),
        ("05_cluster.py",          base_args),
    ]

    if not args.skip_watermark:
        stages += [
            ("06_watermark_train.py", base_args + subset_args),
            ("07_watermark_eval.py",  base_args + subset_args),
            ("08_hybrid_eval.py",     base_args + subset_args),
        ]

    code_dir = Path(__file__).parent
    for script, extra_args in stages:
        run(str(code_dir / script), extra_args)

    print("\n[run_all] All stages complete!")


if __name__ == "__main__":
    main()
