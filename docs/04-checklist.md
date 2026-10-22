# Master Checklist — HybridMarker

Tick as you go. `[M]` must · `[S]` should · `[C]` could. Mark dropped items as `~~...~~ (dropped: reason)`.

## Phase 0 — Setup & data
- [ ] [M] Git repo healthy; branches/ownership agreed
- [ ] [M] `venv` created; `requirements.txt` pinned
- [ ] [M] CUDA PyTorch installed; `torch.cuda.is_available()` is True on RTX 2050
- [ ] [M] Kaggle CIFAKE downloaded to `code/data/cifake/`; 100k/20k counts verified
- [ ] [M] Folder skeleton + `configs/config.yaml` + `.gitignore` (data, outputs, venv, secrets)
- [ ] [M] Root `README.md` (setup + run order)
- [ ] [S] `pytest` runs (even with one trivial test)

## Phase 1 — Preprocessing & EDA (`01_preprocess.py`)
- [ ] [M] Loader with labels REAL=0/FAKE=1
- [ ] [M] RGB / 32×32 validation; corrupt files logged
- [ ] [M] Duplicate check incl. train↔test overlap
- [ ] [M] float32 [0,1] + uint8 arrays saved
- [ ] [M] Stratified train/val split + official test; seed in config
- [ ] [M] `--subset N` dev mode
- [ ] [M] EDA figures: balance, sample grid, mean images, pixel hist
- [ ] [S] Mean FFT spectrum real vs fake figure
- [ ] [M] `data_quality.txt`

## Phase 2 — Features (`02_feature_extraction.py`)
- [ ] [M] Noise residue group (Gaussian + median residues, moments, energy, autocorr)
- [ ] [M] FFT group (radial bins, HF ratio, slope, centroid/flatness, peak ratio)
- [ ] [M] DCT group (band energies, ratios, AC stats)
- [ ] [M] Colour group (RGB hist, HSV hist, moments, saturation stats, channel corr)
- [ ] [M] GLCM group (levels=64, d∈{1,2}, 4 angles, 5 props)
- [ ] [M] `extract_features()` returns vector + names; D fixed
- [ ] [M] Multiprocessing + progress + caching to `.npy`
- [ ] [M] `feature_names.json`; no NaN/inf; constant-column check
- [ ] [M] Unit test per group
- [ ] [S] Per-feature separation chart (Cohen's d / KS)
- [ ] [C] LBP / wavelet extras

## Phase 3 — PCA (`03_pca_reduce.py`)
- [ ] [M] StandardScaler fit on train only
- [ ] [M] PCA 95 % variance fit on train only; D→k recorded
- [ ] [M] Scree + cumulative variance plot
- [ ] [M] 2-D PCA scatter by label
- [ ] [S] Top loadings of PC1–PC3
- [ ] [M] `scaler.joblib`, `pca.joblib`, `X_pca_*.npy` saved & reload test
- [ ] [M] Assertion/test proving no test-set leakage

## Phase 4 — Classification (`04_classify.py`)
- [ ] [M] Baselines: majority + logistic regression
- [ ] [M] DT Gini tuned on validation
- [ ] [M] DT Entropy (IG) tuned on validation
- [ ] [M] MLP tuned (early stopping, alpha, hidden sizes); loss curve
- [ ] [M] Test evaluation once: accuracy, P, R, F1, ROC-AUC, confusion matrix
- [ ] [M] 5-fold CV mean ± std
- [ ] [M] `classification.csv` produced by script
- [ ] [M] Tree depth/complexity curve; tree top levels exported
- [ ] [M] Named feature importances (pre-PCA tree)
- [ ] [M] Feature-group ablation table
- [ ] [S] Misclassified-example grid
- [ ] [S] McNemar DT vs MLP
- [ ] [S] 3-seed repeat for headline numbers
- [ ] [C] With vs without PCA comparison
- [ ] [M] Models saved to `outputs/models/`

## Phase 5 — Clustering (`05_cluster.py`)
- [ ] [M] K-Means sweep k=2…12
- [ ] [M] Elbow + silhouette plots; k justified
- [ ] [M] Purity / ARI / NMI vs real-fake
- [ ] [M] Decide + document CIFAR-class labelling approach (real images via CIFAR-10 mapping, or skip)
- [ ] [S] Purity / ARI / NMI vs CIFAR class
- [ ] [M] PCA-2D scatter by cluster and by label
- [ ] [S] Cluster composition bars + nearest-to-centroid sample images
- [ ] [M] Written 5–10 line interpretation (forgery vs content vs neither)

## Phase 6 — Watermark training (`06_watermark_train.py`)
- [ ] [M] `EncoderCNN`, `DecoderCNN`, `NoiseLayer` implemented; shape tests pass
- [ ] [M] Differentiable JPEG approximation (or documented substitute)
- [ ] [M] Overfit-tiny-batch sanity test passes
- [ ] [M] Training loop: joint loss, AMP, checkpointing, per-epoch log
- [ ] [M] Curriculum (identity → attacks)
- [ ] [M] 48-bit run attempted; gate result recorded
- [ ] [S] Fallback runs (32 / 24 / 16 bits; no-attack) if gate failed
- [ ] [M] Best checkpoint + training curves saved
- [ ] [M] Chosen message length + α + λ recorded in `decisions.md`

## Phase 7 — Watermark eval & hybrid (`07`, `08`)
- [ ] [M] Clean bit-acc, exact-match, PSNR, SSIM on test
- [ ] [M] Cover / marked / ×10 residual figure
- [ ] [M] `utils/attacks.py` real attacks: JPEG 90/70/50, resize ½, crop, noise, blur
- [ ] [M] Robustness table + plot
- [ ] [M] Binomial threshold + empirical FPR on unmarked images
- [ ] [S] TPR@FPR / ROC for watermark detection
- [ ] [S] Interaction study: forensic verdict on watermarked real images
- [ ] [M] Fusion rule implemented
- [ ] [M] `hybrid.csv` for 4 groups × clean/attacked
- [ ] [M] SynthID qualitative comparison table with citations

## Phase 8 — Demo UI (`app/demo.py`)
- [ ] [M] Upload → RGB → 32×32 preprocessing with downscale warning
- [ ] [M] Detect tab: p_fake + verdict
- [ ] [S] Detect tab: feature-group explanation
- [ ] [M] Watermark tab: embed, PSNR, side-by-side, download
- [ ] [S] Attack slider + decoded bit-accuracy gauge
- [ ] [M] Hybrid tab: fused verdict + explanation + limitations footer
- [ ] [S] Sample gallery for offline demo
- [ ] [M] Error handling (corrupt, RGBA/greyscale, large, missing artefacts)
- [ ] [M] Launch instructions tested from clean clone
- [ ] [S] 1-minute demo video backup

## Phase 9 — Report, deck, polish
- [ ] [M] `run_all` reproduces results in a clean venv
- [ ] [M] Tests pass
- [ ] [M] All result tables/figs generated by scripts
- [ ] [M] Report written (outline in doc 05)
- [ ] [M] Abstract updated to mention watermark layer + CIFAKE-only
- [ ] [M] Deck: slide 8 real results; demo slide; key finding/limits slide
- [ ] [M] README + docs updated (`feature-map.md` statuses → DONE)
- [ ] [M] Viva Q&A rehearsed (doc 05 §6)
- [ ] [M] Team contribution section written
- [ ] [M] Release tag `v1.0`

## Stretch (only after Phase 8)
- [ ] [C] Extra-generator OOD test set (32×32)
- [ ] [C] Higher-resolution forensic mini-experiment
- [ ] [C] Permutation importance; RandomForest/SVM baselines

## Progress dashboard (update weekly)
| Phase | Status | Owner | Notes |
|-------|--------|-------|-------|
| 0 | ☐ | Saran | |
| 1 | ☐ | Saran | |
| 2 | ☐ | Saran/Advyaith | |
| 3 | ☐ | Advyaith | |
| 4 | ☐ | Advyaith | |
| 5 | ☐ | Advyaith | |
| 6 | ☐ | Nadheem | |
| 7 | ☐ | Nadheem | |
| 8 | ☐ | Nadheem/Saran | |
| 9 | ☐ | All | |
