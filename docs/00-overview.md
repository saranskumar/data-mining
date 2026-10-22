# HybridMarker — Project Overview

> Course: Data Mining (PECST525), S5 · Team: Saran S Kumar (453), S J Advyaith (460), Nadheem K Sultan (440)
> Docs index: this file → [01-architecture](01-architecture.md) → [02-phases](02-phases.md) → [03-feature-spec](03-feature-spec.md) → [04-checklist](04-checklist.md) → [05-evaluation-and-report](05-evaluation-and-report.md). Existing: [decisions](decisions.md), [feature-map](feature-map.md), [tech-stack](tech-stack.md).

## 1. One-line pitch
Detect AI-generated images with a **classical data-mining pipeline** (feature engineering → PCA → Decision Tree / MLP → K-Means), and pair it with a **learned invisible watermark** layer for provenance. Forensics detects what is already unmarked; watermarking verifies what we marked. They are complementary, not replacements.

## 2. Goals
1. Demonstrate the full data-mining lifecycle end-to-end: collection → preprocessing → feature engineering → dimensionality reduction → classification → evaluation → clustering.
2. Compare Decision Tree (Gini vs Information Gain) against an MLP (backprop) on the same reduced features, with accuracy / precision / recall / F1 / confusion matrix.
3. Use K-Means to explore whether the feature space separates real vs fake and whether it captures *content* (CIFAR class) rather than *forgery*.
4. Build a learned watermark encoder/decoder (PyTorch), evaluate bit recovery, PSNR/SSIM and robustness to JPEG / resize / crop / noise.
5. Show the two layers working together in a small demo UI, and discuss SynthID qualitatively.

## 3. Non-goals
- Not a state-of-the-art detector; no end-to-end CNN classifier as the main model.
- No generator attribution (future scope), no adversarial-robustness claims, no C2PA integration.
- No quantitative SynthID comparison (no API access) — qualitative only.
- No production web/API deployment.

## 4. Scope decisions made in the planning session (2026-10-07)
| Question | Decision | Consequence |
|----------|----------|-------------|
| What does K-Means answer? | Cluster CIFAKE only; test alignment with **real/fake** and **CIFAR-10 class**. | Abstract's "different generators form clusters" is reworded as future work (CIFAKE has a single generator, SD 1.4). |
| What is "the app"? | Pipeline scripts + **small Streamlit/Gradio demo**. | No backend/API. Demo = upload → forensic verdict, + watermark embed/decode tab. |
| Watermark priority | **Core, phased last.** Forensic core (Phases 1–5) is finished first. | Watermark has a fallback ladder (fewer bits → no attack layer) so a weak result never blocks submission. |
| Hardware | **RTX 2050, 4 GB VRAM** (Windows 11). | Watermark CNN must stay small (batch 128–256 at 32×32 fits easily). Classical ML runs on CPU. |
| Timeline | **Assumed ~3 weeks** (deadline not given — adjust [02-phases](02-phases.md) if wrong). | Phases sized in days; Phase 8 (report) is protected. |

## 5. Abstract-vs-plan reconciliation (things the abstract/deck say that need care)
| Abstract / deck claim | Reality | How we handle it |
|-----------------------|---------|------------------|
| "Supplemented with Midjourney / DALL-E samples" | Not in scope per decisions.md (label noise, no time to validate). | Optional stretch (Phase 9). Report states CIFAKE-only. |
| "Generators form separable clusters" | Only one generator in CIFAKE. | Reframed per §4. Say so explicitly in report. |
| "FFT/DCT irregularities" | Both listed. | Implement both (FFT radial spectrum + DCT band energies). |
| "Neural Network trained via backpropagation" | sklearn MLPClassifier is backprop (SGD/Adam). | Documented in report; no PyTorch for classifier (see decisions.md). |
| "SynthID comparison" | No API. | Qualitative table + discuss 3 papers in `papers/`. |
| Deck shows no watermark in abstract | Watermark is a second layer in deck/feature-map. | Treated as extension; add one paragraph to abstract/report so it is not a surprise to the evaluator. |

## 6. Honest expectations (do not over-promise in the report)
- CIFAKE is 32×32; fake images were generated at higher res and downscaled, so classical features capture mostly downscaling/texture statistics. Expect a **moderate** accuracy for classical features, not near-perfect. Whatever number we get is the result — the report must analyse *why*.
- A 48-bit message at 32×32 under attacks is hard. The 48→24→16-bit fallback in decisions.md is expected to be used.
- Do not fill the deck's "Template only / No scores yet" slides with anything but measured numbers.

## 7. Repository layout (target)
```
Data mining/
├─ code/
│  ├─ 01_preprocess.py … 07_watermark_eval.py   (pipeline stages, see feature-map)
│  ├─ 08_hybrid_eval.py                         (new: both layers together)
│  ├─ app/demo.py                               (new: Streamlit/Gradio demo)
│  ├─ utils/{image_utils,feature_utils,wm_model,metrics,plots,attacks}.py
│  ├─ configs/config.yaml                       (seeds, paths, hyperparameters)
│  ├─ tests/                                    (smoke + shape tests)
│  ├─ data/cifake/            (gitignored)
│  └─ outputs/{features,models,figures,tables}/ (gitignored except tables)
├─ docs/        (this folder)
├─ papers/  presentation/
├─ requirements.txt  .gitignore  README.md
```
