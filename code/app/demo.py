"""
app/demo.py — HybridMarker Streamlit demo.

Tabs:
  1. Detect     — upload / sample / batch → forensic verdict per model + feature-group evidence
  2. Watermark  — embed message → marked image + PSNR; attack → decoded bit accuracy
  3. Hybrid     — rule-based fusion verdict with explanation
  4. Results    — tables + figures produced by the pipeline (for presenting)

Launch (from the repo root or code/):
    streamlit run code/app/demo.py
Loads saved artefacts only; never trains.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

import pandas as pd
from PIL import Image, UnidentifiedImageError

st.set_page_config(page_title="HybridMarker", page_icon="🔍", layout="wide")

MODELS_DIR  = CODE_DIR / "outputs" / "models"
FEAT_DIR    = CODE_DIR / "outputs" / "features"
TABLES_DIR  = CODE_DIR / "outputs" / "tables"
FIG_DIR     = CODE_DIR / "outputs" / "figures"
SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
TARGET_SIZE = (32, 32)
MAX_UPLOAD_PIXELS = 40_000_000  # refuse decompression-bomb sized inputs

FORENSIC_MODELS = {"MLP (main)": "mlp", "Decision tree (Gini)": "dt_gini",
                   "Decision tree (entropy)": "dt_entropy", "Logistic regression": "logreg"}
GROUPS = {
    "Noise residue": ("noise_",),
    "FFT / DCT":     ("fft_", "dct_"),
    "Colour":        ("rgb_", "hsv_", "color_", "colour_"),
    "GLCM texture":  ("glcm_",),
}


# ── Model loading (cached) ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading forensic models …")
def load_forensic_models():
    import joblib
    try:
        scaler = joblib.load(MODELS_DIR / "scaler.joblib")
        pca    = joblib.load(MODELS_DIR / "pca.joblib")
        fnames = json.load(open(FEAT_DIR / "feature_names.json"))
    except Exception as e:  # pipeline not run yet
        return None, None, {}, [], str(e)
    models = {}
    for label, name in FORENSIC_MODELS.items():
        p = MODELS_DIR / f"{name}.joblib"
        if p.exists():
            models[label] = joblib.load(p)
    return scaler, pca, models, fnames, None


@st.cache_resource(show_spinner="Loading watermark models …")
def load_wm_models():
    try:
        import torch
        from utils.wm_model import DecoderCNN, EncoderCNN
        info = json.load(open(MODELS_DIR / "wm_train_info.json"))
        L = info["message_bits"]
        alpha = info.get("strength_alpha", 1.0)  # must match training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        enc = EncoderCNN(message_bits=L, strength_alpha=alpha).to(device)
        dec = DecoderCNN(message_bits=L).to(device)
        enc.load_state_dict(torch.load(MODELS_DIR / "wm_enc.pt", map_location=device))
        dec.load_state_dict(torch.load(MODELS_DIR / "wm_dec.pt", map_location=device))
        enc.eval(); dec.eval()
        return enc, dec, L, device, info, None
    except Exception as e:
        return None, None, 0, "cpu", {}, str(e)


@st.cache_data
def load_feat_cfg():
    import yaml
    try:
        return yaml.safe_load(open(CODE_DIR / "configs" / "config.yaml")).get("features", {})
    except Exception:
        return {}


@st.cache_data
def load_threshold(L: int) -> float:
    p = TABLES_DIR / "wm_threshold_info.json"
    if p.exists():
        return float(json.load(open(p))["threshold"])
    from utils.metrics import binomial_threshold
    return float(binomial_threshold(L, 1e-3))


def list_samples() -> list[dict]:
    p = SAMPLES_DIR / "samples.json"
    return json.load(open(p)) if p.exists() else []


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_image(src) -> tuple[np.ndarray | None, str]:
    """File-like or path → float32 (32,32,3) in [0,1] + a note about what was done."""
    try:
        img = Image.open(src)
        if img.width * img.height > MAX_UPLOAD_PIXELS:
            return None, f"Image too large ({img.width}×{img.height})."
        note = []
        if img.mode != "RGB":
            note.append(f"converted {img.mode}→RGB")
            img = img.convert("RGB")
        if img.size != TARGET_SIZE:
            note.append(f"resized {img.width}×{img.height}→32×32 (bicubic)")
            img = img.resize(TARGET_SIZE, Image.BICUBIC)
        return np.asarray(img, dtype=np.float32) / 255.0, "; ".join(note)
    except (UnidentifiedImageError, OSError) as e:
        return None, f"Could not read image: {e}"


def forensic(img_f, scaler, pca, models, fnames):
    from utils.feature_utils import extract_all
    vec, _ = extract_all(img_f, cfg=load_feat_cfg())
    z = scaler.transform(vec.reshape(1, -1))
    zp = pca.transform(z)
    probs = {}
    for label, m in models.items():
        probs[label] = float(m.predict_proba(zp)[0, 1]) if hasattr(m, "predict_proba") else float(m.predict(zp)[0])
    # Evidence per group: mean |z| = how unusual that group's features are vs. training data.
    z = z[0]
    evidence = {}
    for g, prefixes in GROUPS.items():
        idx = [i for i, n in enumerate(fnames) if n.startswith(prefixes)]
        if idx:
            evidence[g] = float(np.mean(np.abs(z[idx])))
    return probs, evidence


def text_to_bits(msg: str, L: int) -> np.ndarray:
    s = msg.strip()
    if len(s) == L and set(s) <= {"0", "1"}:
        return np.array([int(c) for c in s], dtype=np.float32)
    h = hashlib.sha256(msg.encode()).digest()
    return np.array([(h[i // 8] >> (7 - i % 8)) & 1 for i in range(L)], dtype=np.float32)


def wm_encode(img_f, bits, enc, device):
    import torch
    msg = torch.tensor(bits, dtype=torch.float32).unsqueeze(0).to(device)
    cover = torch.from_numpy(img_f.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        return enc(cover, msg).cpu().squeeze(0).numpy().transpose(1, 2, 0)


def wm_decode(img_f, dec, device):
    import torch
    t = torch.from_numpy(np.ascontiguousarray(img_f.transpose(2, 0, 1))).unsqueeze(0).float().to(device)
    with torch.no_grad():
        return (torch.sigmoid(dec(t)) > 0.5).cpu().numpy().astype(np.int32)[0]


def psnr_val(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse < 1e-10 else float(10 * np.log10(1.0 / mse))


def to_pil(arr: np.ndarray, scale: int = 1) -> Image.Image:
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8))
    return im.resize((32 * scale, 32 * scale), Image.NEAREST) if scale > 1 else im


def pick_image(key: str, allow_samples: bool = True):
    """Upload widget + optional sample picker. Returns (img_f, caption) or (None, None)."""
    src = None
    samples = list_samples() if allow_samples else []
    cols = st.columns([3, 2]) if samples else [st.container()]
    with cols[0]:
        up = st.file_uploader("Upload an image (JPG / PNG / WEBP)", type=["jpg", "jpeg", "png", "webp"], key=f"{key}_up")
    if samples:
        with cols[1]:
            names = ["—"] + [f"{s['label']}: {s['name']}" for s in samples]
            choice = st.selectbox("…or pick a sample", names, key=f"{key}_sample")
    if up is not None:
        src, cap = up, up.name
    elif samples and choice != "—":
        s = samples[names.index(choice) - 1]
        src, cap = SAMPLES_DIR / s["file"], f"sample — {s['label']} ({s['source']})"
    if src is None:
        return None, None
    img_f, note = load_image(src)
    if img_f is None:
        st.error(note)
        return None, None
    if note:
        st.caption("ℹ️ " + note)
    return img_f, cap


# ── Header ────────────────────────────────────────────────────────────────────

st.title("🔍 HybridMarker")
st.markdown("**AI-image forensics + learned invisible watermarking** · Data Mining (PECST525) · "
            "Saran S Kumar, S J Advyaith, Nadheem K Sultan")
st.warning("Academic prototype. Every image is reduced to **32×32** before analysis. The forensic "
           "detector was trained on CIFAKE (Stable Diffusion 1.4); on modern generators (GPT Image 2, "
           "Nano Banana 2, FLUX.2 …) it performs at roughly chance — see the Results tab.")

scaler, pca, models, fnames, f_err = load_forensic_models()
enc, dec, L, device, wm_info, wm_err = load_wm_models()
models_ok = scaler is not None and bool(models)
wm_ok = enc is not None

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Detect", "💧 Watermark", "🔗 Hybrid", "📊 Results"])

# ═══════════════ Tab 1: Detect ════════════════════════════════════════════════
with tab1:
    st.header("Forensic detection")
    if not models_ok:
        st.error(f"Forensic models not found — run the pipeline (01→04) first. ({f_err})")
    else:
        mode = st.radio("Mode", ["Single image", "Batch"], horizontal=True, key="det_mode")
        if mode == "Single image":
            img_f, cap = pick_image("det")
            if img_f is not None:
                probs, evidence = forensic(img_f, scaler, pca, models, fnames)
                main = probs.get("MLP (main)", next(iter(probs.values())))
                c1, c2, c3 = st.columns([1, 1.2, 1.4])
                with c1:
                    st.image(to_pil(img_f, 6), caption=f"{cap} (shown ×6)")
                with c2:
                    st.metric("P(AI-generated) — MLP", f"{main:.3f}")
                    if main >= 0.5:
                        st.error("🤖 Likely AI-generated")
                    else:
                        st.success("📷 Likely real")
                    st.caption("Confidence is low near 0.5. 'Real' is not proof of authenticity.")
                    st.dataframe(pd.DataFrame({"model": list(probs), "P(fake)": [round(v, 3) for v in probs.values()],
                                               "verdict": ["AI" if v >= 0.5 else "real" for v in probs.values()]}),
                                 hide_index=True, use_container_width=True)
                with c3:
                    st.markdown("**Feature-group evidence** — mean |z-score| vs. training data "
                                "(higher = more unusual)")
                    ev = pd.DataFrame({"group": list(evidence), "mean |z|": list(evidence.values())}).set_index("group")
                    st.bar_chart(ev)
        else:
            files = st.file_uploader("Upload several images", type=["jpg", "jpeg", "png", "webp"],
                                     accept_multiple_files=True, key="det_batch")
            if files:
                rows, thumbs = [], []
                prog = st.progress(0.0)
                for i, f in enumerate(files):
                    img_f, note = load_image(f)
                    if img_f is None:
                        rows.append({"file": f.name, "error": note})
                    else:
                        probs, _ = forensic(img_f, scaler, pca, models, fnames)
                        p = probs.get("MLP (main)", next(iter(probs.values())))
                        rows.append({"file": f.name, "P(fake) MLP": round(p, 3),
                                     "verdict": "AI-generated" if p >= 0.5 else "real",
                                     **{k: round(v, 3) for k, v in probs.items() if k != "MLP (main)"}})
                        thumbs.append((to_pil(img_f, 3), f"{f.name}: {p:.2f}"))
                    prog.progress((i + 1) / len(files))
                df = pd.DataFrame(rows)
                st.dataframe(df, hide_index=True, use_container_width=True)
                if "verdict" in df:
                    st.write(df["verdict"].value_counts())
                st.download_button("Download results (CSV)", df.to_csv(index=False), "detections.csv", "text/csv")
                if thumbs:
                    st.image([t[0] for t in thumbs[:24]], caption=[t[1] for t in thumbs[:24]], width=96)

# ═══════════════ Tab 2: Watermark ════════════════════════════════════════════
with tab2:
    st.header("Invisible watermarking")
    if not wm_ok:
        st.error(f"Watermark models not found — run 06_watermark_train.py. ({wm_err})")
    else:
        st.caption(f"Model: {L}-bit message, encoder strength α={wm_info.get('strength_alpha', 1.0)}, "
                   f"val bit-accuracy {wm_info.get('best_val_bit_acc', float('nan')):.3f}")
        img_f, cap = pick_image("wm")
        msg_input = st.text_input("Secret message (any text — hashed to bits — or an exact bit string)",
                                  value="HybridMarker", key="wm_msg")
        if img_f is not None:
            from utils.attacks import ATTACK_REGISTRY
            bits = text_to_bits(msg_input, L)
            marked = wm_encode(img_f, bits, enc, device)
            c1, c2, c3 = st.columns(3)
            c1.image(to_pil(img_f, 6), caption="Original")
            c2.image(to_pil(marked, 6), caption=f"Watermarked · PSNR {psnr_val(img_f, marked):.1f} dB")
            c3.image(to_pil(np.clip((marked - img_f) * 10 + 0.5, 0, 1), 6), caption="Difference ×10")
            buf = io.BytesIO()
            to_pil(marked).save(buf, format="PNG")
            st.download_button("Download watermarked PNG", buf.getvalue(), "marked.png", "image/png")

            st.subheader("Robustness check")
            atk = st.selectbox("Attack", list(ATTACK_REGISTRY), key="wm_atk")
            attacked = ATTACK_REGISTRY[atk](marked, {})
            ba = float(np.mean(bits.astype(np.int32) == wm_decode(attacked, dec, device)))
            thr = load_threshold(L)
            a1, a2 = st.columns([1, 3])
            a1.image(to_pil(attacked, 4), caption=f"after {atk}")
            with a2:
                st.metric("Bit accuracy", f"{ba:.3f}", help=f"Detection threshold {thr:.3f}")
                (st.success if ba >= thr else st.error)(
                    "Watermark detected" if ba >= thr else "Watermark NOT detected")
            with st.expander("Bit accuracy under every attack"):
                st.dataframe(pd.DataFrame([{"attack": a, "bit accuracy": round(float(np.mean(
                    bits.astype(np.int32) == wm_decode(fn(marked, {}), dec, device))), 3)}
                    for a, fn in ATTACK_REGISTRY.items()]), hide_index=True)

# ═══════════════ Tab 3: Hybrid ═══════════════════════════════════════════════
with tab3:
    st.header("Hybrid verdict (forensics + provenance)")
    st.markdown("""
| Watermark matches key? | Forensic P(fake) | Verdict |
|---|---|---|
| ✅ yes | any | **Marked — provenance verified** |
| ❌ no | ≥ 0.5 | **Likely AI-generated** |
| ❌ no | < 0.5 | **Likely real / unverified** |

*No watermark ≠ real.* Tip: download a watermarked PNG from the Watermark tab and upload it here with the same message.
""")
    if not models_ok:
        st.error("Forensic models required.")
    else:
        img_f, cap = pick_image("hy")
        key_msg = st.text_input("Watermark key (the message used when embedding; blank = unknown)",
                                value="", key="hy_msg")
        if img_f is not None:
            probs, _ = forensic(img_f, scaler, pca, models, fnames)
            p_fake = probs.get("MLP (main)", next(iter(probs.values())))
            ba, thr, detected = None, None, False
            if wm_ok and key_msg.strip():
                ba = float(np.mean(text_to_bits(key_msg, L).astype(np.int32) == wm_decode(img_f, dec, device)))
                thr = load_threshold(L)
                detected = ba >= thr
            c1, c2 = st.columns([1, 2])
            c1.image(to_pil(img_f, 6), caption=cap)
            with c2:
                if detected:
                    st.success("✅ Marked — provenance verified")
                    why = f"Decoded bits match the key ({ba:.3f} ≥ threshold {thr:.3f})."
                elif p_fake >= 0.5:
                    st.warning("🤖 Likely AI-generated")
                    why = f"No matching watermark; forensic P(fake) = {p_fake:.3f} ≥ 0.5."
                else:
                    st.info("📷 Likely real / unverified")
                    why = f"No matching watermark; forensic P(fake) = {p_fake:.3f} < 0.5. This does not prove the image is real."
                st.markdown(f"**Why:** {why}")
                if ba is not None and not detected:
                    st.caption(f"Watermark bit accuracy vs key: {ba:.3f} (threshold {thr:.3f}).")
                st.caption("No generator attribution is claimed; 32×32 resolution limits accuracy.")

# ═══════════════ Tab 4: Results ══════════════════════════════════════════════
with tab4:
    st.header("Pipeline results")
    st.caption("Everything here is read from outputs/ — produced by the scripts, not typed by hand.")
    tables = [("Classification on CIFAKE test set", "classification.csv"),
              ("Modern generators (out-of-distribution)", "ood.csv"),
              ("Modern generators — per generator", "ood_by_generator.csv"),
              ("K-Means clustering", "clustering.csv"),
              ("Watermark robustness", "watermark.csv"),
              ("Hybrid fusion", "hybrid.csv")]
    for title, fname in tables:
        p = TABLES_DIR / fname
        if p.exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(p).round(4), hide_index=True, use_container_width=True)
    figs = sorted(FIG_DIR.glob("*.png")) if FIG_DIR.exists() else []
    if figs:
        st.subheader("Figures")
        sel = st.multiselect("Show figures", [f.name for f in figs],
                             default=[n for n in ("roc_curves.png", "cm_mlp.png", "eda_mean_fft.png",
                                                  "pca_scatter.png", "wm_examples.png") if (FIG_DIR / n).exists()])
        for n in sel:
            st.image(str(FIG_DIR / n), caption=n)
