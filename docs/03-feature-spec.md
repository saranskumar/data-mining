# Feature Specification — every feature, in detail

Legend: **MUST** = required for submission · **SHOULD** = planned · **COULD** = stretch.

## 1. Data ingestion & preprocessing (MUST) — `01_preprocess.py`
| ID | Feature | Detail | Acceptance |
|----|---------|--------|------------|
| P1 | Loader | Walk `train/{REAL,FAKE}`, `test/{REAL,FAKE}`; label REAL=0, FAKE=1 | 100 000 / 20 000 counts match CIFAKE facts |
| P2 | Validation | Force RGB, assert 32×32, drop + log unreadable files | `data_quality.txt` written |
| P3 | Duplicate check | MD5/aHash of pixels; report intra-split and train↔test duplicates | Overlap count reported; overlapping test rows removed or flagged |
| P4 | Normalisation | float32 [0,1] for features; uint8 retained for watermark | Both arrays saved |
| P5 | Splits | Stratified train/val (90/10, seed) + official test | Sizes logged; seed in config |
| P6 | Subset mode | `--subset N` stratified | Dev runs < 1 min |
| P7 | EDA | class balance, sample grid, mean images, pixel histogram, mean FFT spectrum real vs fake | Figures in `outputs/figures/eda_*` |

## 2. Config, logging, utilities (MUST)
- `config.yaml`: seed, paths, split ratio, feature params, PCA variance, model grids, watermark L/α/λ, attack params.
- Logging to console + `outputs/logs/<stage>.log`; each script prints its output shapes and runtime.
- Each script is runnable standalone with `python 0X_*.py [--subset N] [--config path]`.

## 3. Quality features (SHOULD)
- `pytest` smoke tests: feature vector length == `len(feature_names)`, no NaN, PCA reload round-trip, encoder output shape/range, decoder output length.
- `run_all.py` (or `Makefile`/`.ps1`) executes stages 1→8 in order.

## 4. Feature groups
Features computed per image on 32×32 RGB; grayscale `g = 0.299R+0.587G+0.114B` where stated. Dimensions are targets; final D recorded in `feature_names.json`.

### 4.1 Noise residue (MUST) — ≈ 20 dims
Idea: generators leave different high-frequency residue than camera sensor + downsampling pipelines.
- Denoise with Gaussian (σ=1) **and** median 3×3 (two residues: `r = x − denoise(x)`).
- Per channel (R,G,B) of the Gaussian residue: mean, variance, skewness, kurtosis, energy (Σr²/N) → 15.
- Grayscale residue (median): mean abs, variance, energy → 3.
- Residue autocorrelation at lag 1 (horizontal, vertical) → 2.
- Acceptance: unit test on synthetic noisy vs smooth image shows energy ordering.

### 4.2 Frequency domain — FFT + DCT (MUST) — ≈ 22 dims
- **FFT** on grayscale (Hann window, `fftshift`, log-power):
  - Azimuthally-averaged radial power in 8 bins → 8
  - High-frequency energy ratio (outer ring / total) → 1
  - Spectral slope (log-log linear fit of radial power vs frequency) → 1
  - Spectral centroid, spectral flatness → 2
  - Peak-to-mean ratio of non-DC spectrum (grid/periodic artefacts) → 1
- **DCT** (2-D, full 32×32 or 8×8 blocks averaged):
  - Energy in low / mid / high zig-zag bands (and ratios) → 5
  - Mean |AC coefficient|, DC value, count of near-zero coefficients → 3
  - (if 8×8 blocks) variance across blocks of high-band energy → 1
- Acceptance: mean FFT spectra of real vs fake plotted and a measurable difference reported (even if small).

### 4.3 Colour statistics (MUST) — ≈ 114 dims
- RGB histograms: 16 bins × 3 channels (normalised) → 48
- HSV histograms: 16 bins × 3 channels → 48
- Per-channel mean / std / skewness (RGB) → 9
- Mean saturation, mean value, fraction of near-saturated (≤2 or ≥253) pixels per channel → 5
- Inter-channel correlation (RG, GB, RB) → 3
- (SHOULD) Hue-histogram entropy → 1
- If D too large for fast tree training, bins can drop to 8 (config).

### 4.4 GLCM texture (MUST) — ≈ 20 dims
- `skimage.feature.graycomatrix` on grayscale **quantised to 64 levels** (decision 2026-10-07), distances {1, 2}, angles {0°, 45°, 90°, 135°}, symmetric + normed.
- Properties: contrast, dissimilarity, homogeneity, energy, correlation → mean over angles per distance → 2×5 = 10; std over angles → 10 (SHOULD).
- Acceptance: values finite on flat images (handle zero-variance correlation → 0).

### 4.5 Optional features (COULD)
- LBP histogram (uniform, P=8,R=1) → 10; wavelet-subband energies (db2, 2 levels) → 7; JPEG-grid blockiness score; local-patch variance map statistics.
- Add only if ablation shows headroom.

**Expected total D ≈ 175 (MUST+SHOULD groups).** Scaling is per-feature standardisation before PCA (needed: histogram bins vs energies differ in scale by orders of magnitude).

## 5. Dimensionality reduction (MUST) — `03_pca_reduce.py`
| ID | Feature | Detail |
|----|---------|--------|
| R1 | Scaler | `StandardScaler` fit on train, applied to val/test |
| R2 | PCA | `PCA(n_components=0.95, svd_solver="full", random_state=seed)` fit on train |
| R3 | Diagnostics | scree + cumulative variance plot, top-5 loading bar charts per first 3 PCs, report D→k |
| R4 | Persistence | `scaler.joblib`, `pca.joblib`, `X_pca_*.npy` |
| R5 | Variant (COULD) | Compare classifier with/without PCA to quantify PCA's benefit |

## 6. Classification (MUST) — `04_classify.py`
| ID | Feature | Detail |
|----|---------|--------|
| C1 | Baselines | DummyClassifier (most frequent), LogisticRegression |
| C2 | DT-Gini | `DecisionTreeClassifier(criterion="gini")` + grid on `max_depth`, `min_samples_leaf`, `ccp_alpha` |
| C3 | DT-IG | same with `criterion="entropy"` (information gain) |
| C4 | MLP | `MLPClassifier(adam, early_stopping=True)`, grid on hidden layers & `alpha`; loss curve plotted |
| C5 | Evaluation | accuracy, precision, recall, F1 (fake = positive), confusion matrix, ROC-AUC, 5-fold CV mean±std |
| C6 | Tree interpretability | Top-3 levels exported; named-feature importances from pre-PCA tree |
| C7 | Ablation | Drop each feature group → retrain best DT & MLP → Δ-accuracy table |
| C8 | Error analysis | 16 false positives + 16 false negatives grid |
| C9 | Statistical test (SHOULD) | McNemar test DT vs MLP on test predictions |
| C10 | Persistence | models saved, `classification.csv` |

## 7. Clustering (MUST) — `05_cluster.py`
| ID | Feature | Detail |
|----|---------|--------|
| K1 | K-Means sweep | k=2…12, `n_init=10`, seeded |
| K2 | Elbow + silhouette | plots + chosen k justified |
| K3 | External validity | purity, ARI, NMI vs real/fake |
| K4 | Content check | alignment with CIFAR-10 class (real images; see architecture note) |
| K5 | Visuals | PCA-2D scatter by cluster / by label; composition bars; sample images per cluster centroid-nearest |
| K6 | Narrative | forgery vs content vs neither |

## 8. Learned watermarking (MUST, phased last) — `06/07`
| ID | Feature | Detail |
|----|---------|--------|
| W1 | Message generator | random L-bit messages per sample; fixed secret key message for the demo |
| W2 | Encoder CNN | cover + message → residual; strength α; output clipped to [0,1] |
| W3 | Decoder CNN | image → L logits |
| W4 | Noise layer | identity, Gaussian noise, blur, resize, crop, differentiable JPEG approximation; sampled per batch |
| W5 | Training loop | joint loss (image fidelity + message BCE), AMP, curriculum, checkpointing, curves |
| W6 | Metrics | bit-accuracy, exact-match, PSNR, SSIM |
| W7 | Real-attack robustness | JPEG 90/70/50, resize ½, crop 80 %, noise, blur via `utils/attacks.py` |
| W8 | Detection test | bit-acc threshold from binomial/unmarked calibration; TPR/FPR |
| W9 | Visuals | cover / marked / ×10 residual grid |
| W10 | Fallback ladder | 48→32→24→16 bits → drop attacks (documented in 02-phases) |
| W11 | Interaction study | does watermarking shift forensic classifier's verdicts? |

## 9. Hybrid layer (MUST) — `08_hybrid_eval.py`
- H1 Rule-based fusion (table in [01-architecture](01-architecture.md#fusion-rule-simple-explainable--rule-based-not-learned)).
- H2 Scored on 4 groups × clean + attacked. Metrics: verdict accuracy per group, confusion table.
- H3 SynthID qualitative comparison table (coverage, requires cooperating generator, robustness claims, access, false-positive behaviour) with citations to `papers/`.

## 10. Demo app (SHOULD→MUST for viva) — `app/demo.py`
| ID | Feature | Detail |
|----|---------|--------|
| A1 | Upload & preprocess | any image → RGB → 32×32 (show downscale warning for large inputs) |
| A2 | Detect tab | p_fake, verdict, which feature groups pushed the decision (tree-path or per-group z-scores) |
| A3 | Watermark tab | text/bits input → embed → side-by-side + PSNR; download marked PNG |
| A4 | Attack slider | JPEG quality / resize / noise → decode → bit-accuracy gauge |
| A5 | Hybrid tab | fused verdict + plain-language explanation + limitations footer |
| A6 | Sample gallery | 10 stored examples (real/fake/marked) for no-internet demo |
| A7 | Error handling | corrupt file, RGBA/greyscale, huge file, missing model artefacts message |
| A8 | Launch | `streamlit run code/app/demo.py` documented in README |

## 11. Documentation & deliverables (MUST)
- README (setup, run order, hardware notes), docs kept current, final report (PDF/DOCX), updated slide deck with real results, demo video backup, clean requirements, tagged release.
