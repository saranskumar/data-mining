# Datasets

| Folder | Source | Licence | Notes |
|---|---|---|---|
| cifake/ | HF mirror `dragonintelligence/CIFAKE-image-dataset` of Kaggle CIFAKE (Bird & Lotfi) | MIT (as CIFAR-10) | 100k train / 20k test, 32x32. Mirror labels were 0=FAKE,1=REAL; folders here are named REAL/FAKE so the project's 0=real,1=fake mapping applies. |
| modern/julienlucas_nbpro/ | `julienlucas/midjourney-dalle-sd-nanobananapro-dataset` | MIT | parquet; 10k train + 2k test; Midjourney/DALL-E/SD/Nano Banana Pro. Check columns with `code/data_ood/build_modern.py --inspect`. |
| modern/mllm_gptimage2_nb2/ | `zr-zhang/MLLM-Generated-Image-Detection-Dataset` (arXiv 2608.01258) | Apache-2.0 (card tag) | Generators: FLUX.2, GPT Image 2, Nano Banana 2, Qwen Image 2.1, plus Real. Categories: Texture/Structure/Hybrid. Only the first N files per folder downloaded. Label = folder name (Real vs others). |

Not downloaded: `Shanmuk4622/ai-image-detection-dataset` (HF gated, ~24.5 GB, accept terms with your HF login), Kaggle 'Midjourney CIFAKE-Inspired' (needs Kaggle token).
