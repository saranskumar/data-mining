# Phase-wise Execution Plan

Assumed budget: **~3 weeks (21 days)**. Each phase has: goal, tasks, deliverables, **exit criteria** (don't move on until met). Suggested owners are a starting point — swap freely.

| Phase | Name | Days | Suggested owner | Critical path |
|-------|------|------|-----------------|---------------|
| 0 | Setup & data | 1–2 | Saran | ✔ |
| 1 | Preprocessing & EDA | 2–3 | Saran | ✔ |
| 2 | Feature engineering | 3–6 | Saran + Advyaith | ✔ |
| 3 | Scaling + PCA | 6–7 | Advyaith | ✔ |
| 4 | Classification (DT + MLP) | 7–10 | Advyaith | ✔ |
| 5 | Clustering (K-Means) | 9–11 | Advyaith | |
| 6 | Watermark training | 7–14 (parallel) | Nadheem | |
| 7 | Watermark evaluation + hybrid fusion | 14–16 | Nadheem | |
| 8 | Demo UI | 16–18 | Nadheem + Saran | |
| 9 | Report, deck, polish | 18–21 | All | ✔ |
| (stretch) | Extra generators / ablations | only if ahead | – | |

Phase 6 starts early (day 7) in parallel because it is independent of Phases 3–5 and the GPU work is slow.

---

## Phase 0 — Setup & data (Day 1–2)
**Goal:** reproducible environment, dataset on disk, repo skeleton.
- Create `venv`, `requirements.txt` (numpy, scipy, pandas, pillow, opencv-python, scikit-image, scikit-learn, matplotlib, seaborn, joblib, pyyaml, tqdm, torch+CUDA, torchvision, streamlit *or* gradio, pytest).
- Install CUDA-enabled PyTorch for RTX 2050; verify `torch.cuda.is_available()`.
- Download CIFAKE from Kaggle into `code/data/cifake/` (needs Kaggle account/API token — **never commit the token**).
- Create folder skeleton, `config.yaml`, `.gitignore` (data/, outputs/ except tables/, venv, `__pycache__`, `.env`).
- Root `README.md` with setup + run order.
**Exit:** `python -c "import torch,sklearn,skimage"` works; CUDA true; 120 000 files counted; skeleton committed.

## Phase 1 — Preprocessing & EDA (Day 2–3) · `01_preprocess.py`
**Goal:** clean, consistent arrays + first look at the data.
- Load all images; verify mode RGB, size 32×32; report/drop corrupt files; hash-based duplicate check (esp. train/test overlap).
- Normalise to float [0,1] copy for features; keep uint8 copy for watermark.
- Fixed stratified **train / val / test** split (test = official 20 000; val = 10 % of train, seeded).
- EDA: class balance, mean/std image per class, mean pixel histograms, a sample grid, mean FFT spectrum real vs fake (motivates features).
- Subset config (`--subset N`) for fast development runs (e.g. 5 000 images).
**Deliverables:** arrays + EDA figures in `outputs/figures/eda_*.png`, data-quality note.
**Exit:** arrays load with expected shapes/dtypes; split sizes logged; EDA figures saved.

## Phase 2 — Feature engineering (Day 3–6) · `02_feature_extraction.py`, `utils/feature_utils.py`
**Goal:** structured feature matrix X (N×D) with named columns. Detailed definitions in [03-feature-spec](03-feature-spec.md#4-feature-groups).
- Implement four groups independently, each unit-tested on a handful of images: **Noise residue**, **FFT/DCT**, **Color (RGB/HSV hist + moments)**, **GLCM texture (levels=64)**.
- Multiprocessing extraction with progress bar; cache to `.npy`; write `feature_names.json`.
- Sanity: no NaN/inf, constant-column check, per-feature class-separation (e.g. KS-test / Cohen's d) bar chart.
**Deliverables:** `X_feat_*.npy`, `feature_names.json`, per-group separation figure.
**Exit:** D documented; extraction on full data < ~1 h on CPU (otherwise vectorise); features ↔ names verified.

## Phase 3 — Scaling + PCA (Day 6–7) · `03_pca_reduce.py`
- `StandardScaler` **fit on train only** → PCA retaining 95 % variance, `random_state` fixed.
- Figures: scree plot, cumulative variance, 2-D PCA scatter by label, top loadings per PC.
- Record D → k reduction; save `scaler`/`pca` with joblib.
**Exit:** k chosen & justified; no test leakage (assert fit only on train rows); artifacts reloadable.

## Phase 4 — Classification (Day 7–10) · `04_classify.py`
- Baselines: majority class, logistic regression (sanity floor).
- **Decision Tree** ×2: `criterion="gini"` and `criterion="entropy"` (= information gain). Tune `max_depth`, `min_samples_leaf`, `ccp_alpha` on **validation** set (or CV on train). Export tree top-levels + depth-vs-accuracy curve.
- **Interpretability caveat:** PCA components aren't human-readable. Also train a tree on the **pre-PCA** scaled features to get named feature importances; report both.
- **MLP** (`MLPClassifier`): hidden sizes sweep (e.g. (64,), (128,64)), `early_stopping=True`, `alpha` L2, adam, max_iter; plot loss curve.
- Final: refit best config on train(+val), evaluate on test **once**. Metrics: accuracy, precision, recall, F1, confusion matrix, ROC-AUC; 5-fold CV mean±std on train for stability.
- Error analysis: show misclassified examples; which feature groups matter (ablation: drop one group at a time, retrain DT/MLP).
**Deliverables:** `tables/classification.csv`, confusion-matrix figs, ablation table, importance chart.
**Exit:** comparison table DT-Gini vs DT-IG vs MLP complete; ablation done; results saved by script.

## Phase 5 — Clustering (Day 9–11) · `05_cluster.py`
- K-Means on PCA space (train+test combined or train only — pick one, state it): k sweep 2–12, elbow (inertia), silhouette.
- Evaluate vs labels: **purity, ARI, NMI** against real/fake; against CIFAR class (see data-contract note in [01-architecture §2](01-architecture.md)).
- PCA 2-D scatter coloured by cluster and by label; cluster composition bar chart.
- Written finding: do clusters track forgery, object content, or neither? Be honest if "neither".
**Exit:** table + 3 figures + 5–10 line interpretation.

## Phase 6 — Watermark training (Day 7–14, parallel) · `06_watermark_train.py`, `utils/wm_model.py`
Fallback ladder — move down only if the previous rung fails its exit gate:
1. 48 bits + noise layer (full) → gate: val bit-acc ≥ ~0.9 clean, PSNR ≥ ~30 dB.
2. 32 bits.
3. 24 → 16 bits.
4. Drop attack layer (clean-channel only) and report as limitation.
- Implement Encoder/Decoder/NoiseLayer; unit test shapes; overfit-a-tiny-batch sanity test first.
- Curriculum: identity noise → add attacks. Log loss, bit-acc, PSNR every epoch; checkpoint best.
- AMP + batch 128–256; subsample training set if epoch time is too long on 4 GB.
**Exit:** best checkpoint saved; training curves saved; chosen message length recorded in `decisions.md`.

## Phase 7 — Watermark evaluation + hybrid fusion (Day 14–16) · `07_watermark_eval.py`, `08_hybrid_eval.py`
- On held-out test images: bit accuracy, exact-match rate, PSNR, SSIM; visual cover/marked/residual (×10 amplified) grid.
- Robustness table with **real** attacks (`utils/attacks.py`): JPEG q∈{90,70,50}, resize 0.5×/up-down, crop 80 %, gaussian noise σ, blur.
- Detection threshold: calibrate bit-accuracy threshold from **unmarked** images (false-positive rate target ~1e-3 via binomial); report TPR/FPR.
- Interaction experiment: run Layer-1 classifier on watermarked real images — does the watermark change its verdict? (report FP-rate shift).
- `08_hybrid_eval.py`: score fusion rule on 4 groups (real/fake × marked/unmarked), after each attack.
**Exit:** `watermark.csv`, `hybrid.csv`, figures; threshold + FPR documented.

## Phase 8 — Demo UI (Day 16–18) · `app/demo.py`
- Streamlit/Gradio: **Tab 1 Detect** (upload → resize to 32×32 → verdict + p_fake + top feature contributions); **Tab 2 Watermark** (embed message → show marked image + PSNR; apply attack slider → decode → bit-acc); **Tab 3 Hybrid** (fusion verdict with explanation text).
- Loads saved artifacts only; clear "academic prototype / 32×32" banner; handles non-RGB/huge files.
- Record a 1-minute screen capture as demo backup.
**Exit:** fresh clone + README steps launches the app; demo backup video saved.

## Phase 9 — Report, deck, polish (Day 18–21)
- Replace the deck's "template only / no scores yet" slides with real numbers (slide 8 + conclusion).
- Write report per [05-evaluation-and-report](05-evaluation-and-report.md); add the watermark paragraph to the abstract.
- Final cleanup: README, docs status, tests green, requirements verified in a clean venv, tag release `v1.0`.
- Rehearse presentation; prepare answers for the likely viva questions (see doc 05 §6).
**Exit:** every checklist item in [04-checklist](04-checklist.md) ticked or explicitly marked "dropped (reason)".

## Stretch (only if ahead of schedule)
- Add one extra generator's images (resized to 32×32) as an out-of-distribution test set → report generalisation drop.
- Higher-res (64×64+) mini-experiment for the forensic layer.
- Feature-importance via permutation; SVM / RandomForest as extra baselines.

## Risk register
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Classical features give mediocre accuracy at 32×32 | High | Med | Report honestly + analyse; ablation shows which groups help; add DCT/FFT detail features |
| Watermark won't converge with 48 bits + attacks | High | Med | Fallback ladder; clean-channel result is still reportable |
| 4 GB VRAM OOM | Med | Low | Small model, AMP, batch 128, gradient checkpoint not needed at 32×32 |
| Feature extraction too slow on 120 k images | Med | Med | Multiprocessing, vectorise noise/FFT/hist, `--subset` for dev, cache |
| Data leakage (PCA/scaler fit on test, duplicates across split) | Med | High | Fit-on-train asserts, duplicate hash check |
| Kaggle download/auth friction | Low | Med | Do on Day 1; keep a copy of zip on external drive |
| Scope creep (generators, web app) | Med | High | Stretch list only after Phase 8 exit |
| Team merge conflicts on shared utils | Low | Low | Owners per file; small PRs |
