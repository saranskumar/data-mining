# Feature Map — HybridMarker

Full specs: [03-feature-spec.md](03-feature-spec.md) · Checklist: [04-checklist.md](04-checklist.md) · Plan: [02-phases.md](02-phases.md)

## Code modules
| File | Phase | Status | Purpose |
|------|-------|--------|---------|
| 01_preprocess.py | 1 | TODO | Load CIFAKE, validate, dedupe, split, save arrays, EDA |
| 02_feature_extraction.py | 2 | TODO | Noise, FFT/DCT, color, GLCM → X_feat |
| 03_pca_reduce.py | 3 | TODO | StandardScaler + PCA(95% variance), diagnostics |
| 04_classify.py | 4 | TODO | Baselines, Decision Tree (Gini/IG), MLP, ablation |
| 05_cluster.py | 5 | TODO | K-Means sweep, purity/ARI/NMI, scatter |
| 06_watermark_train.py | 6 | TODO | Encoder/Decoder CNN training with noise layer |
| 07_watermark_eval.py | 7 | TODO | Bit recovery, PSNR/SSIM, real-attack robustness, threshold |
| 08_hybrid_eval.py | 7 | TODO | Fusion rule scoring |
| app/demo.py | 8 | TODO | Streamlit/Gradio demo |
| run_all.py | 9 | TODO | Run stages 1→8 |
| utils/image_utils.py | 1 | TODO | load_images(), normalize(), splits, seeding |
| utils/feature_utils.py | 2 | TODO | extract_features() + per-group functions |
| utils/wm_model.py | 6 | TODO | EncoderCNN, DecoderCNN, NoiseLayer |
| utils/attacks.py | 7 | TODO | Real JPEG/resize/crop/noise/blur attacks |
| utils/metrics.py | 4 | TODO | Classification, clustering, PSNR/SSIM, bit-acc, binomial threshold |
| utils/plots.py | 1 | TODO | Shared figure helpers |
| configs/config.yaml | 0 | TODO | Seeds, paths, hyperparameters |
| tests/ | 2+ | TODO | Shape/NaN/round-trip tests |

## Data artifacts (gitignored)
| Path | Description |
|------|-------------|
| data/cifake/ | Raw CIFAKE images |
| outputs/features/X_img_*.npy | uint8 image arrays |
| outputs/features/X_feat_*.npy | Full feature matrix |
| outputs/features/X_pca_*.npy | PCA-reduced matrix |
| outputs/features/y_*.npy | Labels |
| outputs/models/ | sklearn (.joblib) + torch (.pt) models |
| outputs/figures/ | All plots |
| outputs/tables/ | Result CSVs (committed) |
