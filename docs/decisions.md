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

## 2026-10-27 (implementation)

### Watermark design and message length
**Decision**: Encoder residual is bounded with `tanh` and scaled by α (max per-pixel change α); α = 0.03, **16-bit** message (fallback ladder rung 4 of 48→32→24→16).
**Rationale**: Balancing image quality with a loss weight was unstable — λ_img ≤ 3 gave PSNR 18–24 dB, λ_img = 20 collapsed to "hide nothing" (bit-acc 0.50). Bounding the residual fixes quality by construction (PSNR ≥ ~30 dB). Sweep on 20k images, 12 epochs: 16 bits/α 0.03 → bit-acc 0.967 @ 32.3 dB; 32 bits/α 0.03 → 0.740; 32 bits/α 0.05 → 0.794 — 32 bits misses the 0.9 gate. Decoder also changed from global average pooling to stride-2 convs + flatten (pooling discarded where the bits were placed).

### Modern-generator evaluation
**Decision**: Added an out-of-distribution test (GPT Image 2, Nano Banana 2, FLUX.2, Qwen Image 2.1, MJ/DALL-E/SD/Nano Banana Pro mix), test-only, models never retrained on it.
**Rationale**: CIFAKE (SD 1.4) is old; the question "does it still work on current generators?" is answered with measured numbers (balanced accuracy ≈ 0.50 for every model).

### Watermark — keep original size, make it invisible (supersedes parts of the entry above)
**Decision**: (1) Embed at full resolution: residual computed at 32×32, upsampled (bicubic) and added to the original-size photo; decode by downscaling to 32×32 (`utils/wm_apply.py`). (2) Zero-mean residual + reflect padding + a learned spatial message embedding (`Linear(L → 4×32×32)`) instead of tiled constant bit maps. (3) Noise layer scaled to the watermark budget (σ 0.01, straight-through quantisation at 1/64). (4) Perceptual texture mask + embed-and-verify strength boost at embed time. Final model: 16 bits, α 0.03, 3-channel residual, trained on 90k images, last-epoch checkpoint.
**Rationale**: User feedback — the 32×32 output was tiny and the change visible. Diagnosis: the encoder had learned to hide bits in a global colour tint and in zero-padded borders (a conv net with tiled bits has no other positional signal); tint survives downscaling but is very visible at full size. With zero-mean alone bit-acc fell to ~0.63; the learned spatial embedding restored it (1.000 @ 37.5 dB on 20k). On 150 real full-size photos (strength 0.75, verified mode): PSNR 43.2 dB, SSIM 0.998, detection 100 % clean / 99.3 % after JPEG 50 / 100 % after resize ½, 0 false positives; crop still breaks it (limitation).
