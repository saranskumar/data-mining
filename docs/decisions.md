# Decisions — HybridMarker

## 2026-10-07

### Dataset
**Decision**: Use CIFAKE as primary. Supplement with Midjourney/DALL-E samples only if time permits.
**Rationale**: CIFAKE is balanced, labeled, publicly available. Additional generators risk label noise without enough time to validate.

### Watermark message length
**Decision**: 48 bits default; fall back to 16–24 bits if recovery on 32x32 images is poor.
**Rationale**: 32x32 has very limited spatial capacity.

### Neural network choice
**Decision**: Use sklearn MLPClassifier (not PyTorch) for forensic detection layer.
**Rationale**: Keeps forensic pipeline in pure scikit-learn; PyTorch reserved for watermark CNN only. Simpler, course-aligned.

### GLCM quantization
**Decision**: Use levels=64 for GLCM to reduce noise on small 32x32 images.
**Rationale**: 256-level GLCM on a 32-pixel image is very sparse and noisy.

### SynthID comparison
**Decision**: Qualitative discussion only — no API access. Reference papers in docs/papers.

## 2026-10-07 (planning session)

### K-Means purpose
**Decision**: Cluster CIFAKE only; evaluate alignment with real/fake and CIFAR-10 class. "Generator clusters" claim moves to future work.
**Rationale**: CIFAKE has a single fake generator (SD 1.4), so generator separability is untestable.

### App scope
**Decision**: Pipeline scripts + small Streamlit/Gradio demo (detect / watermark / hybrid tabs). No API/backend.
**Rationale**: Fits viva demo; deck lists web/API as future scope.

### Watermark priority
**Decision**: Core but phased last; forensic pipeline (Phases 1–5) completes first. Fallback ladder 48→32→24→16 bits → no attack layer.
**Rationale**: Abstract only promises the forensic pipeline; watermark must not endanger the graded core.

### Hardware & timeline
**Decision**: RTX 2050 4 GB; plan assumes ~3 weeks (deadline unconfirmed).
**Rationale**: Small 32x32 CNN fits; classical ML on CPU.

### Fusion rule
**Decision**: Rule-based (watermark decoded → provenance verified; else forensic p_fake ≥ 0.5 → likely AI; else unverified).
**Rationale**: Explainable, no extra training data needed.

### Evaluation hygiene
**Decision**: Scaler/PCA fit on train only; tuning on validation split; test evaluated once; DT importance also reported on pre-PCA features.
**Rationale**: Avoid leakage; PCA components are not interpretable.
