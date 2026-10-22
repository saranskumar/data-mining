# Architecture — HybridMarker

## 1. Two layers, one decision
```
                    ┌───────────────── Layer 1: Forensic (no watermark needed) ─────────────────┐
 image (32×32) ───► │ preprocess → features (noise|FFT/DCT|color|GLCM) → scale → PCA → DT / MLP │──► p_fake
        │           └────────────────────────────────────────────────────────────────────────────┘
        │           ┌──────────────── Layer 2: Provenance (watermark present?) ────────────────┐
        └─────────► │ Decoder CNN → bit string → bit-accuracy vs expected key → binomial test   │──► mark_detected, confidence
                    └────────────────────────────────────────────────────────────────────────────┘
                                                   │
                                      Fusion rule (08_hybrid_eval / demo)
```

### Fusion rule (simple, explainable — rule-based, not learned)
| Watermark decoded? | Forensic p_fake | Verdict |
|--------------------|-----------------|---------|
| Yes (p-value < α) | any | **Marked / provenance verified** (high confidence) |
| No | ≥ 0.5 | **Likely AI-generated** (forensic, lower confidence) |
| No | < 0.5 | **Likely real / unverified** |

Always show: "no watermark ≠ real", and "fake ≠ Stable Diffusion" (no attribution).

## 2. Pipeline stages and data contracts
| Stage | Script | Input | Output | Contract |
|-------|--------|-------|--------|----------|
| 1 | `01_preprocess.py` | `data/cifake/{train,test}/{REAL,FAKE}/*.jpg` | `outputs/features/{X_img_train,X_img_test}.npy` (uint8 N×32×32×3), `y_*.npy`, `cls_*.npy` (CIFAR class if recoverable) | label 0=real, 1=fake; fixed seed; corrupt/duplicate report |
| 2 | `02_feature_extraction.py` | image arrays | `X_feat_{train,test}.npy` (N×D float32), `feature_names.json` | D fixed, names ↔ columns 1:1, no NaN/inf |
| 3 | `03_pca_reduce.py` | `X_feat_*` | `X_pca_{train,test}.npy`, `scaler.joblib`, `pca.joblib` | **fit on train only**; keep 95 % variance |
| 4 | `04_classify.py` | `X_pca_*`, `y_*` | `models/{dt_gini,dt_entropy,mlp}.joblib`, `tables/classification.csv`, confusion matrices | val split from train for tuning; test touched once |
| 5 | `05_cluster.py` | `X_pca_*` | `tables/clustering.csv`, elbow/silhouette/scatter figs | k sweep; purity/ARI/NMI vs real-fake and CIFAR class |
| 6 | `06_watermark_train.py` | images | `models/wm_{enc,dec}.pt`, training log | message length from config; noise layer in loop |
| 7 | `07_watermark_eval.py` | models + test images | `tables/watermark.csv`, example grids | bit-acc, PSNR, SSIM, per-attack |
| 8 | `08_hybrid_eval.py` | all models | `tables/hybrid.csv` | fusion rule scored on marked/unmarked × real/fake |
| – | `app/demo.py` | models | UI | loads saved artifacts only (no retraining) |

**CIFAKE note:** the Kaggle folders only give REAL/FAKE. CIFAR-10 class labels are *not* shipped. For the CIFAR-class clustering analysis, either (a) skip it for FAKE images and use the real CIFAR-10 test set class labels via `torchvision.datasets.CIFAR10` mapping (real images only), or (b) pseudo-label with a tiny pretrained classifier. Decision to be taken in Phase 5; default (a): class-alignment analysis on REAL images only. Tracked in checklist.

## 3. Module responsibilities
- `utils/image_utils.py` — `load_images(dir)`, `normalize(arr)`, `train_val_split`, seeding.
- `utils/feature_utils.py` — one function per feature group + `extract_features(img) -> (vec, names)`; vectorised/multiprocessing wrapper.
- `utils/wm_model.py` — `EncoderCNN`, `DecoderCNN`, `NoiseLayer` (differentiable JPEG approx, resize, crop, gaussian noise/blur, identity).
- `utils/attacks.py` — *real* (non-differentiable) PIL/OpenCV attacks used only at eval time.
- `utils/metrics.py` — accuracy/P/R/F1, confusion matrix, purity/ARI/NMI, PSNR/SSIM, bit-acc, binomial threshold.
- `utils/plots.py` — consistent matplotlib/seaborn figures saved to `outputs/figures/`.
- `configs/config.yaml` — single source for seeds, paths, hyperparameters, message length.

## 4. Watermark model sketch (sized for RTX 2050 4 GB)
- **Encoder:** conv blocks on cover image concatenated with message bits spatially replicated → residual `δ`; output `x_w = clip(x + α·δ)`. Strength α is a config knob.
- **Noise layer:** randomly pick per batch {identity, gaussian noise, blur, resize-down-up, random crop+pad, differentiable-JPEG-approx}.
- **Decoder:** small CNN → L logits (L = message bits).
- **Loss:** `λ_img · MSE(x, x_w) + λ_msg · BCEWithLogits(decoded, bits)`; optional curriculum (start with identity noise, add attacks later).
- **Training budget:** AMP on, batch 128–256, ≤ ~30 epochs; subsample if needed. Checkpoint every epoch.
- Inspired by WAM / HiDDeN-style systems; **not** claimed identical to WAM (paper in `papers/`).

## 5. Reproducibility rules
- One global seed (e.g. 42) set for numpy / sklearn / torch; stored in `config.yaml`.
- Every table in the report is produced by a script, never hand-typed.
- Pin `requirements.txt` (include CUDA-matched `torch` install note for Windows).
- Features and models cached under `outputs/` (gitignored); small result tables committed.
