"""
app/demo.py — HybridMarker Streamlit demo.

Tabs:
  1. Detect     — upload / sample / batch → forensic verdict per model + feature-group evidence
  2. Watermark  — embed message → marked image + PSNR; attack → decoded bit accuracy
  3. Decode     — read the hidden bits from any image and name the matching message
  4. Results    — tables + figures produced by the pipeline (for presenting)

Launch (from the repo root or code/):
    streamlit run code/app/demo.py
Loads saved artefacts only; never trains.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import sys
import time
from datetime import datetime
from html import escape
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

import jobs

st.set_page_config(page_title="HybridMarker", layout="wide", initial_sidebar_state="expanded")

MODELS_DIR  = CODE_DIR / "outputs" / "models"
FEAT_DIR    = CODE_DIR / "outputs" / "features"
TABLES_DIR  = CODE_DIR / "outputs" / "tables"
FIG_DIR     = CODE_DIR / "outputs" / "figures"
SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
TARGET_SIZE = (32, 32)
MAX_UPLOAD_PIXELS = 40_000_000  # refuse decompression-bomb sized inputs
MAX_WM_SIDE = 2048              # watermark large photos on a copy capped at this long side

FORENSIC_MODELS = {"MLP (main)": "mlp", "Decision tree (Gini)": "dt_gini",
                   "Decision tree (entropy)": "dt_entropy", "Logistic regression": "logreg"}
GROUPS = {
    "Noise residue": ("noise_",),
    "FFT / DCT":     ("fft_", "dct_"),
    "Colour":        ("rgb_", "hsv_", "color_", "colour_"),
    "GLCM texture":  ("glcm_",),
}


# ── Process log (what the app did, step by step, with timings) ────────────────

ACTIVITY_LOG = CODE_DIR / "outputs" / "logs" / "demo_activity.log"
ICON = {"ok": "OK", "warn": "WARN", "fail": "FAIL", "info": "INFO"}


class ProcessLog:
    """Collects the steps of one operation (detect / embed / decode) with timings and details."""

    def __init__(self, kind: str, subject: str) -> None:
        self.kind, self.subject = kind, subject
        self.started = datetime.now()
        self.t0 = self.t = time.perf_counter()
        self.rows: list[dict] = []

    def step(self, name: str, detail: str = "", status: str = "ok") -> None:
        """Log a finished step; 'took' is the time since the previous step."""
        now = time.perf_counter()
        self.rows.append({"#": len(self.rows) + 1, "step": name, "detail": detail,
                          "took (ms)": round((now - self.t) * 1000), "at (ms)": round((now - self.t0) * 1000),
                          "status": ICON.get(status, status)})
        self.t = now

    def sub(self, name: str, detail: str = "", status: str = "info") -> None:
        """A detail row that belongs to the previous step (no timing of its own)."""
        self.rows.append({"#": len(self.rows) + 1, "step": "    ↳ " + name, "detail": detail,
                          "took (ms)": None, "at (ms)": None, "status": ICON.get(status, status)})

    @property
    def total_ms(self) -> int:
        return round((self.t - self.t0) * 1000)

    @property
    def n_steps(self) -> int:
        return sum(r["took (ms)"] is not None for r in self.rows)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows).astype(object).fillna("")

    def text(self) -> str:
        head = f"[{self.started:%Y-%m-%d %H:%M:%S}] {self.kind} - {self.subject} ({self.total_ms} ms)"
        lines = [f"  {r['#']:>2}. {r['step']:<30} {'' if r['took (ms)'] is None else str(r['took (ms)']) + ' ms':>8}  "
                 f"{r['status']} {r['detail']}" for r in self.rows]
        return chr(10).join([head] + lines)

    def show(self, key: str, sig: str, expanded: bool = True) -> None:
        """Render the log, keep it in the session history and append it to outputs/logs once."""
        with st.expander(f"Process log — {self.kind}: {self.n_steps} steps, {self.total_ms} ms",
                         expanded=expanded):
            st.dataframe(self.frame(), hide_index=True, use_container_width=True)
            st.download_button("Download this log (.txt)", self.text(), f"{key}_log.txt", key=f"{key}_logdl")
        hist = st.session_state.setdefault("history", {})
        if sig not in hist:
            hist[sig] = self
            try:
                ACTIVITY_LOG.parent.mkdir(parents=True, exist_ok=True)
                with open(ACTIVITY_LOG, "a", encoding="utf-8") as fh:
                    fh.write(self.text() + chr(10))
            except OSError:
                pass


def img_note(key: str) -> str:
    """The note load_image produced for the image picked under `key` (resize / conversion info)."""
    return st.session_state.get(f"{key}_note", "")


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
        enc = EncoderCNN(message_bits=L, strength_alpha=alpha, lum_only=info.get("lum_only", False)).to(device)
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

def load_image(src, keep_full: bool = False):
    """File-like or path → (small 32×32 float32 [0,1], full-size float32 [0,1] or None, note).
    The detector always works on the 32×32 copy; the watermark works on the full-size image."""
    try:
        img = Image.open(src)
        if img.width * img.height > MAX_UPLOAD_PIXELS:
            return None, None, f"Image too large ({img.width}×{img.height})."
        note = []
        if img.mode != "RGB":
            note.append(f"converted {img.mode}→RGB")
            img = img.convert("RGB")
        full = None
        if keep_full:
            fimg = img.copy()
            if max(fimg.size) > MAX_WM_SIDE:
                fimg.thumbnail((MAX_WM_SIDE, MAX_WM_SIDE), Image.LANCZOS)
                note.append(f"watermark works on a {fimg.width}×{fimg.height} copy")
            full = np.asarray(fimg, dtype=np.float32) / 255.0
        if img.size != TARGET_SIZE:
            note.append(f"detector input: {img.width}×{img.height}→32×32 (bicubic)")
            img = img.resize(TARGET_SIZE, Image.BICUBIC)
        return np.asarray(img, dtype=np.float32) / 255.0, full, "; ".join(note)
    except (UnidentifiedImageError, OSError) as e:
        return None, None, f"Could not read image: {e}"


def forensic(img_f, scaler, pca, models, fnames, log: ProcessLog | None = None):
    from utils.feature_utils import extract_all
    vec, _ = extract_all(img_f, cfg=load_feat_cfg())
    if log:
        log.step("Extract features", f"{vec.size} handcrafted features (noise residue, FFT/DCT, colour, GLCM texture)")
    z = scaler.transform(vec.reshape(1, -1))
    if log:
        log.step("Standardise", "StandardScaler: zero mean / unit variance using training-set statistics")
    zp = pca.transform(z)
    if log:
        kept = float(np.sum(pca.explained_variance_ratio_)) * 100
        log.step("PCA projection", f"{vec.size} → {zp.shape[1]} components ({kept:.1f}% of training variance kept)")
    probs = {}
    for label, m in models.items():
        probs[label] = float(m.predict_proba(zp)[0, 1]) if hasattr(m, "predict_proba") else float(m.predict(zp)[0])
        if log:
            log.step(f"Classify: {label}", f"P(AI-generated) = {probs[label]:.3f} → "
                     f"{'AI-generated' if probs[label] >= 0.5 else 'real'}")
    # Evidence per group: mean |z| = how unusual that group's features are vs. training data.
    z = z[0]
    evidence = {}
    for g, prefixes in GROUPS.items():
        idx = [i for i, n in enumerate(fnames) if n.startswith(prefixes)]
        if idx:
            evidence[g] = float(np.mean(np.abs(z[idx])))
    if log and evidence:
        top = max(evidence, key=evidence.get)
        log.step("Feature-group evidence", f"{len(evidence)} groups; most unusual: {top} (mean |z| {evidence[top]:.2f})")
    return probs, evidence


def text_to_bits(msg: str, L: int) -> np.ndarray:
    s = msg.strip()
    if len(s) == L and set(s) <= {"0", "1"}:
        return np.array([int(c) for c in s], dtype=np.float32)
    h = hashlib.sha256(msg.encode()).digest()
    return np.array([(h[i // 8] >> (7 - i % 8)) & 1 for i in range(L)], dtype=np.float32)


REGISTRY = CODE_DIR / "outputs" / "wm_registry.json"


def load_registry() -> dict:
    """Messages embedded with this demo → their bit strings (lets Decode name the message)."""
    try:
        return json.load(open(REGISTRY))
    except Exception:
        return {}


def register_message(msg: str, bits: np.ndarray) -> None:
    reg = load_registry()
    reg[msg] = "".join(str(int(b)) for b in bits)
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    json.dump(reg, open(REGISTRY, "w"), indent=2)


def psnr_val(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse < 1e-10 else float(10 * np.log10(1.0 / mse))


def to_pil(arr: np.ndarray, scale: int = 1) -> Image.Image:
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8))
    return im.resize((32 * scale, 32 * scale), Image.NEAREST) if scale > 1 else im


def pick_image(key: str, allow_samples: bool = True, keep_full: bool = False):
    """Upload widget + optional sample picker. Returns (small, full, caption) or (None, None, None)."""
    src = None
    samples = list_samples() if allow_samples else []
    cols = st.columns([3, 2]) if samples else [st.container()]
    with cols[0]:
        up = st.file_uploader("Upload an image (JPG / PNG / WEBP)", type=["jpg", "jpeg", "png", "webp"], key=f"{key}_up")
    if samples:
        with cols[1]:
            names = ["—"] + [f"{s['label']}: {s['name']}" for s in samples]
            choice = st.selectbox("Or choose a sample image", names, key=f"{key}_sample")
    if up is not None:
        src, cap = up, up.name
    elif samples and choice != "—":
        s = samples[names.index(choice) - 1]
        src, cap = SAMPLES_DIR / s["file"], f"sample — {s['label']} ({s['source']})"
    if src is None:
        return None, None, None
    img_f, full, note = load_image(src, keep_full=keep_full)
    st.session_state[f"{key}_note"] = note if img_f is not None else ""
    if img_f is None:
        st.error(note)
        return None, None, None
    if note:
        st.caption(note)
    return img_f, full, cap


# ── Look and feel ─────────────────────────────────────────────────────────────

CSS = """
<style>
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }
h1 { font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.1rem; }
h2, h3 { letter-spacing: -0.01em; }
button[role="tab"] { font-weight: 600; padding: 0.55rem 1.1rem; }
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
.hm-sub { opacity: .7; margin-bottom: 1.2rem; }
.hm-verdict { border-radius: 12px; padding: 1rem 1.25rem; margin: 0.25rem 0 0.75rem 0; border: 1px solid transparent; }
.hm-vt { font-size: 1.3rem; font-weight: 700; line-height: 1.3; }
.hm-vs { opacity: .8; font-size: .95rem; margin-top: .2rem; }
.hm-ok   { background: rgba(46,160,67,.14);  border-color: rgba(46,160,67,.55); }
.hm-bad  { background: rgba(218,54,51,.14);  border-color: rgba(218,54,51,.55); }
.hm-warn { background: rgba(210,153,34,.14); border-color: rgba(210,153,34,.55); }
.hm-neutral { background: rgba(128,128,128,.12); border-color: rgba(128,128,128,.4); }
.hm-chip { display: inline-block; padding: .1rem .6rem; border-radius: 999px; font-size: .78rem; font-weight: 600;
           border: 1px solid transparent; white-space: nowrap; }
.hm-empty { border: 1px dashed rgba(128,128,128,.5); border-radius: 12px; padding: 2rem 1rem; text-align: center;
            opacity: .75; margin: .5rem 0 1rem 0; }
.hm-step { font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; opacity: .65;
           margin-bottom: .25rem; }
.hm-row { display: flex; justify-content: space-between; align-items: center; margin: .35rem 0; font-size: .92rem; }
[data-testid="stMetric"] { background: rgba(128,128,128,.08); border-radius: 10px; padding: .6rem .9rem; }
[data-testid="stMetricLabel"] { opacity: .75; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def verdict(kind: str, title: str, sub: str = "") -> None:
    """Coloured result card. kind: ok | bad | warn | neutral. All text is escaped."""
    st.markdown(f'<div class="hm-verdict hm-{kind}"><div class="hm-vt">{escape(title)}</div>'
                f'<div class="hm-vs">{escape(sub)}</div></div>', unsafe_allow_html=True)


def chip(text: str, kind: str = "neutral") -> str:
    return f'<span class="hm-chip hm-{kind}">{escape(text)}</span>'


def empty_state(text: str) -> None:
    st.markdown(f'<div class="hm-empty">{escape(text)}</div>', unsafe_allow_html=True)


def step_label(n: int, text: str) -> None:
    st.markdown(f'<div class="hm-step">Step {n} · {escape(text)}</div>', unsafe_allow_html=True)


# ── Header and sidebar ────────────────────────────────────────────────────────

scaler, pca, models, fnames, f_err = load_forensic_models()
enc, dec, L, device, wm_info, wm_err = load_wm_models()
models_ok = scaler is not None and bool(models)
wm_ok = enc is not None

st.title("HybridMarker")
st.markdown('<div class="hm-sub">AI-image forensics and learned invisible watermarking</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### HybridMarker")
    st.caption("Data Mining (PECST525)")
    st.markdown("**System status**")
    job_now = jobs.current()
    job_chip = (chip("running", "warn") if job_now and job_now["status"] == "running" else chip("idle", "neutral"))
    st.markdown(
        f'<div class="hm-row"><span>Forensic models</span>'
        f'{chip(f"{len(models)} loaded", "ok") if models_ok else chip("not built", "bad")}</div>'
        f'<div class="hm-row"><span>Watermark model</span>'
        f'{chip(f"{L}-bit, {str(device).upper()}", "ok") if wm_ok else chip("not built", "bad")}</div>'
        f'<div class="hm-row"><span>Training job</span>{job_chip}</div>', unsafe_allow_html=True)
    if not (models_ok and wm_ok):
        st.caption("Open the Train tab to build the missing models.")
    st.divider()
    st.markdown("**How to use**")
    st.markdown("1. **Detect**: check if an image looks AI-generated.\n"
                "2. **Watermark**: hide a message in your photo.\n"
                "3. **Decode**: read the message back.\n"
                "4. **Train**: build or rebuild the models.\n"
                "5. **Results**: tables and figures.")
    st.divider()
    st.caption("Saran S Kumar · S J Advyaith · Nadheem K Sultan")
    if st.button("Reload models", help="Clear the model cache (use after training finishes elsewhere)"):
        st.cache_resource.clear()
        st.rerun()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Detect", "Watermark", "Decode", "Train", "Results", "Logs"])

# ═══════════════ Tab 1: Detect ════════════════════════════════════════════════
with tab1:
    st.subheader("Forensic detection")
    st.caption("Estimates whether an image was generated by AI. Every image is reduced to 32×32 first. "
               "The detector was trained on Stable Diffusion 1.4 images and is much less reliable on newer "
               "generators (see Results).")
    if not models_ok:
        st.error(f"Forensic models not found. Build them in the Train tab (stages 01 to 04). ({f_err})")
    else:
        mode = st.radio("Mode", ["Single image", "Batch"], horizontal=True, key="det_mode", label_visibility="collapsed")
        if mode == "Single image":
            with st.container(border=True):
                step_label(1, "Choose an image")
                img_f, _, cap = pick_image("det")
            if img_f is None:
                empty_state("Pick a sample or upload an image to see the analysis.")
            else:
                dlog = ProcessLog("Detect", cap)
                dlog.step("Load image", f"{cap}; " + (img_note("det") or "already 32×32"))
                probs, evidence = forensic(img_f, scaler, pca, models, fnames, log=dlog)
                main = probs.get("MLP (main)", next(iter(probs.values())))
                dlog.step("Verdict", f"MLP P(AI) = {main:.3f} → {'AI-generated' if main >= 0.5 else 'real'}; "
                          f"{sum(v >= 0.5 for v in probs.values())}/{len(probs)} models say AI",
                          "warn" if abs(main - 0.5) < 0.1 else "ok")
                c1, c2 = st.columns([1, 1.6], gap="large")
                with c1:
                    st.image(to_pil(img_f, 6), caption=f"{cap} (shown ×6)", use_container_width=True)
                with c2:
                    unsure = abs(main - 0.5) < 0.15
                    sub = f"MLP probability of AI: {main:.0%}" + (" · low confidence" if unsure else "")
                    verdict("warn" if unsure else ("bad" if main >= 0.5 else "ok"),
                            "Likely AI-generated" if main >= 0.5 else "Likely real", sub)
                    st.markdown("**All models**")
                    for label, p in probs.items():
                        st.progress(min(max(p, 0.0), 1.0), text=f"{label}: {p:.0%} AI")
                    st.caption("A 'real' verdict is not proof of authenticity.")
                with st.expander("Why? Feature-group evidence", expanded=False):
                    st.caption("Mean absolute z-score per feature group against the training data (higher = more unusual).")
                    ev = pd.DataFrame({"group": list(evidence), "mean |z|": list(evidence.values())}).set_index("group")
                    st.bar_chart(ev)
                dlog.show("detect", f"detect|{cap}|{main:.4f}")
        else:
            with st.container(border=True):
                step_label(1, "Upload images")
                files = st.file_uploader("Upload several images", type=["jpg", "jpeg", "png", "webp"],
                                         accept_multiple_files=True, key="det_batch", label_visibility="collapsed")
            if not files:
                empty_state("Upload one or more images to analyse them together.")
            else:
                rows, thumbs = [], []
                blog = ProcessLog("Batch detect", f"{len(files)} files")
                prog = st.progress(0.0, text=f"Analysing 0/{len(files)} …")
                for i, f in enumerate(files):
                    img_f, _, note = load_image(f)
                    if img_f is None:
                        rows.append({"file": f.name, "error": note})
                        blog.step(f.name, note, "fail")
                    else:
                        probs, _ = forensic(img_f, scaler, pca, models, fnames)
                        p = probs.get("MLP (main)", next(iter(probs.values())))
                        blog.step(f.name, f"P(AI) MLP = {p:.3f} → {'AI-generated' if p >= 0.5 else 'real'}")
                        rows.append({"file": f.name, "P(fake) MLP": round(p, 3),
                                     "verdict": "AI-generated" if p >= 0.5 else "real",
                                     **{k: round(v, 3) for k, v in probs.items() if k != "MLP (main)"}})
                        thumbs.append((to_pil(img_f, 3), f"{f.name}: {p:.2f}"))
                    prog.progress((i + 1) / len(files), text=f"Analysing {i + 1}/{len(files)} …")
                prog.empty()
                df = pd.DataFrame(rows)
                if "verdict" in df:
                    n_ai = int((df["verdict"] == "AI-generated").sum())
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Images", len(files))
                    m2.metric("Flagged as AI", n_ai)
                    m3.metric("Look real", int((df["verdict"] == "real").sum()))
                st.dataframe(df, hide_index=True, use_container_width=True)
                st.download_button("Download results (CSV)", df.to_csv(index=False), "detections.csv", "text/csv")
                if thumbs:
                    st.image([t[0] for t in thumbs[:24]], caption=[t[1] for t in thumbs[:24]], width=96)
                blog.step("Done", f"{len(files)} files, {len(thumbs)} analysed, {len(files) - len(thumbs)} failed")
                blog.show("batch", "batch|" + ",".join(f.name for f in files), expanded=False)

# ═══════════════ Tab 2: Watermark ════════════════════════════════════════════
with tab2:
    st.subheader("Invisible watermarking")
    if not wm_ok:
        st.error(f"Watermark model not found. Train it in the Train tab (stage 06). ({wm_err})")
    else:
        st.caption(f"Hide a {L}-bit message in a photo at its original size. "
                   f"Model accuracy on validation data: {wm_info.get('best_val_bit_acc', float('nan')):.1%}.")
        with st.container(border=True):
            step_label(1, "Choose a photo")
            img_f, full, cap = pick_image("wm", keep_full=True)
        with st.container(border=True):
            step_label(2, "Message and strength")
            msg_input = st.text_input("Secret message (any text; it is hashed to bits)", value="HybridMarker", key="wm_msg")
            cs1, cs2 = st.columns([2, 1])
            strength = cs1.slider("Strength (lower = less visible, less robust)", 0.25, 1.0, 0.75, 0.05, key="wm_strength")
            adaptive = cs2.checkbox("Perceptual masking", value=True, key="wm_adaptive",
                                    help="Put less watermark in flat areas (sky, paper) where it could be seen")
        if img_f is None:
            empty_state("Choose a photo in step 1 to embed the message.")
        else:
            from utils.wm_apply import decode_fullres, embed_verified
            h, w = full.shape[:2]
            wlog = ProcessLog("Watermark", cap)
            wlog.step("Load image", f"{cap}; working size {w}×{h}; {img_note('wm') or 'no resize needed'}")
            bits = text_to_bits(msg_input, L)
            as_given = set(msg_input.strip()) <= {"0", "1"} and len(msg_input.strip()) == L
            wlog.step("Message → bits", f"“{msg_input}” → {''.join(str(int(b)) for b in bits)} "
                      f"({'bit string used as given' if as_given else f'SHA-256 hash, first {L} bits'})")
            trace: list = []
            with st.spinner("Embedding and verifying the watermark …"):
                marked, diff, used, ok = embed_verified(full, bits, enc, dec, device, strength=strength,
                                                        adaptive=adaptive, trace=trace)
            wlog.step("Embed + verify", f"image → 32×32 → encoder residual (≤ ±{wm_info.get('strength_alpha', 1.0) * 255:.1f}/255) "
                      f"→ upsample to {w}×{h}{' → perceptual mask' if adaptive else ''}; {len(trace)} attempt(s)")
            for t in trace:
                good = t["clean_ok"] and t["jpeg_ok"] is not False
                jpeg_txt = "skipped (image < 128 px)" if t["jpeg_ok"] is None else ("OK" if t["jpeg_ok"] else "FAIL")
                wlog.sub(f"strength {t['strength']:.2f}",
                         f"clean read-back {'OK' if t['clean_ok'] else 'FAIL'}; JPEG-80 read-back {jpeg_txt}; "
                         f"PSNR {t['psnr']:.1f} dB", "ok" if good else "warn")
            wlog.step("Strength decision", f"requested {strength:.2f}, used {used:.2f}"
                      f"{' (auto-raised: flat image)' if used > strength + 1e-6 else ''}; "
                      f"verified = {'yes' if ok else 'no'}", "ok" if ok else "warn")
            register_message(msg_input, bits)
            wlog.step("Register message", "saved to outputs/wm_registry.json so the Decode tab can name it")
            q = psnr_val(full, marked)
            wlog.step("Output", f"{w}×{h} (original size kept); PSNR {q:.1f} dB; max change {np.abs(diff).max() * 255:.1f}/255")

            st.markdown("#### Result")
            if ok:
                verdict("ok", "Watermark embedded and verified",
                        "The message reads back exactly, including after a JPEG-80 save.")
            elif np.array_equal(decode_fullres(marked, dec, device), bits.astype(np.int32)):
                verdict("warn", "Embedded, but may not survive JPEG",
                        "The message reads back from the PNG; compression may remove it on this image.")
            else:
                verdict("bad", "Message does not read back on this image",
                        "Even at maximum strength the decoder could not recover it.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Size (kept)", f"{w}×{h}")
            m2.metric("Image quality", f"{q:.1f} dB", help="PSNR; above 40 dB is visually transparent")
            m3.metric("Strength used", f"{used:.2f}", delta=f"{used - strength:+.2f}" if used > strength + 1e-6 else None,
                      help="Raised automatically when the image needs more signal")
            m4.metric("Max change", f"±{np.abs(diff).max() * 255:.1f}/255")
            c1, c2, c3 = st.columns(3)
            c1.image(to_pil(full), caption="Original", use_container_width=True)
            c2.image(to_pil(marked), caption="Watermarked", use_container_width=True)
            c3.image(to_pil(np.clip(diff * 10 + 0.5, 0, 1)), caption="Difference ×10 (illustration)", use_container_width=True)
            buf = io.BytesIO()
            to_pil(marked).save(buf, format="PNG")
            st.download_button(f"Download watermarked PNG ({w}×{h})", buf.getvalue(), "marked.png", "image/png",
                               type="primary")

            st.markdown("#### Robustness check")
            from utils.attacks import ATTACK_REGISTRY
            atk = st.selectbox("Attack applied to the full-size image", list(ATTACK_REGISTRY), key="wm_atk")
            attacked = ATTACK_REGISTRY[atk](marked, {})
            ba = float(np.mean(bits.astype(np.int32) == decode_fullres(attacked, dec, device)))
            thr = load_threshold(L)
            wlog.step(f"Robustness: {atk}", f"bit accuracy {ba:.3f} (threshold {thr:.3f}) → "
                      f"{'detected' if ba >= thr else 'NOT detected'}", "ok" if ba >= thr else "warn")
            a1, a2 = st.columns([1, 2], gap="large")
            a1.image(to_pil(attacked), caption=f"after {atk}", use_container_width=True)
            with a2:
                verdict("ok" if ba >= thr else "bad", "Watermark detected" if ba >= thr else "Watermark not detected",
                        f"Bit accuracy {ba:.3f} (detection threshold {thr:.3f})")
                with st.expander("Bit accuracy under every attack"):
                    st.table(pd.DataFrame([{"attack": a, "bit accuracy": round(float(np.mean(
                        bits.astype(np.int32) == decode_fullres(fn(marked, {}), dec, device))), 3)}
                        for a, fn in ATTACK_REGISTRY.items()]))
            wlog.show("watermark", f"wm|{cap}|{msg_input}|{strength}|{adaptive}|{atk}")

# ═══════════════ Tab 3: Decode ═══════════════════════════════════════════════
with tab3:
    st.subheader("Decode a watermark")
    st.caption("Upload an image (for example the PNG from the Watermark tab, even if it was re-saved as JPEG or "
               "resized). The decoder reads the hidden bits and matches them against messages embedded with this demo.")
    if not wm_ok:
        st.error(f"Watermark model not found. Train it in the Train tab (stage 06). ({wm_err})")
    else:
        from utils.wm_apply import decode_fullres
        with st.container(border=True):
            step_label(1, "Choose an image")
            img_f, full, cap = pick_image("dec", keep_full=True)
            key_msg = st.text_input("Optional: check against one specific message", value="", key="dec_msg")
        if img_f is None:
            empty_state("Choose an image to look for a hidden message.")
        else:
            thr = load_threshold(L)
            dlog = ProcessLog("Decode", cap)
            dlog.step("Load image", f"{cap}; {full.shape[1]}×{full.shape[0]}; {img_note('dec') or 'no resize needed'}")
            decoded = decode_fullres(full, dec, device)
            dlog.step("Decode", f"image → 32×32 (bicubic) → decoder CNN → {L} bits {''.join(map(str, decoded))}")
            candidates = dict(load_registry())
            if key_msg.strip():
                candidates = {key_msg: "".join(str(int(b)) for b in text_to_bits(key_msg, L))}
            scores = sorted(((float(np.mean(decoded == np.array([int(c) for c in b]))), m)
                             for m, b in candidates.items()), reverse=True)
            dlog.step("Match candidates", f"{len(candidates)} candidate message(s) "
                      f"({'typed key' if key_msg.strip() else 'registry from the Watermark tab'}); "
                      f"need ≥ {int(np.ceil(thr * L))}/{L} bits")
            for sc, m in scores[:5]:
                dlog.sub(f"“{m}”", f"{round(sc * L)}/{L} bits match ({sc:.3f})", "ok" if sc >= thr else "info")
            found = bool(scores) and scores[0][0] >= thr
            dlog.step("Verdict", f"watermark found: “{scores[0][1]}”" if found else "no watermark found",
                      "ok" if found else "warn")
            st.markdown("#### Result")
            c1, c2 = st.columns([1, 1.6], gap="large")
            c1.image(to_pil(full), caption=cap, use_container_width=True)
            with c2:
                if found:
                    acc, msg = scores[0]
                    verdict("ok", f"Watermark found: “{msg}”", f"{round(acc * L)} of {L} bits match")
                    chance = 2 ** -L * sum(math.comb(L, k) for k in range(round(acc * L), L + 1))
                    m1, m2 = st.columns(2)
                    m1.metric("Matching bits", f"{round(acc * L)}/{L}", help=f"Detection threshold {thr:.3f}")
                    m2.metric("Chance of a false match", f"{chance:.1e}", help="Probability that an unmarked image matches this well")
                else:
                    verdict("bad", "No watermark found" + (f" for “{key_msg}”" if key_msg.strip() else ""),
                            (f"Best match {round(scores[0][0] * L)}/{L} bits; needs at least {int(np.ceil(thr * L))}. "
                             f"Unmarked images match about {L // 2}/{L} by chance.") if scores else
                            "No messages have been embedded yet. Use the Watermark tab first, or type a message above.")
                with st.expander("Decoded bits"):
                    st.code("".join(map(str, decoded)))
            dlog.show("decode", f"dec|{cap}|{key_msg}|{len(candidates)}")

# ═══════════════ Tab 4: Train ════════════════════════════════════════════════
with tab4:
    st.subheader("Train and run the pipeline")
    st.caption("Run any stage from here. One job runs at a time, in the background; you can leave this page and "
               "come back. Each stage needs the previous ones to be built first.")
    running = jobs.is_running()
    if running:
        st.info("A job is running. Stage buttons are disabled until it finishes or you stop it.")

    @st.fragment(run_every=2 if running else None)
    def live_job_panel():
        job = jobs.current()
        st.markdown("#### Current job")
        if job is None:
            empty_state("No job has been run yet. Start one below.")
            return
        is_run = job["status"] == "running"
        kind = {"running": "warn", "done": "ok", "failed": "bad", "stopped": "neutral"}[job["status"]]
        elapsed = jobs.fmt_duration(job["ended"] - job["started"])
        verdict(kind, f"{job['title']}: {job['status']}",
                f"{elapsed} · python {job['script']} {' '.join(job['args'])}".strip())
        if is_run:
            if st.button("Stop job", key="job_stop"):
                jobs.stop()
                st.rerun()
        st.code(jobs.tail(job["log"], 150) or "(no output yet)", language=None, height=320)
        st.download_button("Download full log", Path(job["log"]).read_bytes() if Path(job["log"]).exists() else b"",
                           Path(job["log"]).name, key="job_log_dl")
        was_running = st.session_state.get("_job_was_running", False)
        st.session_state["_job_was_running"] = is_run
        if not is_run:
            fin = jobs.record_if_finished()
            if fin is not None and fin["status"] == "done":
                st.cache_resource.clear()   # new models are picked up on the next run
            if was_running or fin is not None:
                st.rerun()

    live_job_panel()

    def launch(stage: jobs.Stage, values: dict) -> None:
        try:
            args = jobs.build_args(stage, values)
            jobs.start(stage.title, stage.script, args)
        except (ValueError, RuntimeError) as e:
            st.error(str(e))
            return
        st.session_state["_job_was_running"] = True
        st.rerun()

    def param_widgets(stage: jobs.Stage) -> dict:
        vals: dict = {}
        cols = st.columns(min(3, max(1, len(stage.params))))
        for i, p in enumerate(stage.params):
            k = f"tr_{stage.id}_{p.key}"
            with cols[i % len(cols)]:
                if p.kind == "switch":
                    vals[p.key] = st.checkbox(p.label, value=bool(p.default), key=k, help=p.help or None)
                elif p.kind == "int":
                    vals[p.key] = st.number_input(p.label, value=p.default, min_value=1, step=1, key=k, help=p.help or None)
                elif p.kind == "float":
                    vals[p.key] = st.number_input(p.label, value=p.default, min_value=0.0, step=0.01, format="%.3f",
                                                  key=k, help=p.help or None)
                else:
                    vals[p.key] = st.text_input(p.label, value=p.default or "", key=k, help=p.help or None)
        return vals

    st.markdown("#### Run everything")
    with st.container(border=True):
        st.markdown(f"**{jobs.RUN_ALL.title}**  \n{jobs.RUN_ALL.summary} Typical time: {jobs.RUN_ALL.long}.")
        with st.expander("Options"):
            all_vals = param_widgets(jobs.RUN_ALL)
        if st.button("Run all stages", key="run_all", type="primary", disabled=running):
            launch(jobs.RUN_ALL, all_vals)

    st.markdown("#### Individual stages")
    for stage in jobs.STAGES:
        built, when = jobs.stage_status(stage)
        missing = [jobs.BY_ID[n] for n in stage.needs if not jobs.stage_status(jobs.BY_ID[n])[0]]
        with st.container(border=True):
            head, state = st.columns([4, 1])
            head.markdown(f"**{stage.id} · {stage.title}**" + ("  \nUses the GPU." if stage.gpu else ""))
            state.markdown(chip(when, "ok" if built else "neutral"), unsafe_allow_html=True)
            st.caption(f"{stage.summary} Typical time: {stage.long}.")
            if missing:
                st.caption("Build first: " + ", ".join(f"{m.id} {m.title}" for m in missing))
            vals = {}
            if stage.params:
                with st.expander("Options"):
                    vals = param_widgets(stage)
            if st.button(f"Run {stage.id}", key=f"run_{stage.id}", disabled=running or bool(missing),
                         help="Disabled while another job is running or a previous stage is missing"):
                launch(stage, vals)

    with st.expander("Previous runs"):
        hist_rows = jobs.history(15)
        if hist_rows:
            st.dataframe(pd.DataFrame([{"stage": h["title"], "status": h["status"],
                                        "started": datetime.fromtimestamp(h["started"]).strftime("%d %b %H:%M:%S"),
                                        "duration": jobs.fmt_duration(h["ended"] - h["started"]),
                                        "command": h["cmd"]} for h in hist_rows]),
                         hide_index=True, use_container_width=True)
        else:
            st.caption("Nothing yet.")

# ═══════════════ Tab 5: Results ══════════════════════════════════════════════
with tab5:
    st.subheader("Pipeline results")
    st.caption("Read from outputs/, produced by the scripts and not typed by hand.")
    rlog = ProcessLog("Results", "outputs/ tables and figures")

    def table(name: str):
        p = TABLES_DIR / name
        return pd.read_csv(p) if p.exists() else None

    kpis = []
    try:
        c = table("classification.csv").set_index("model")
        kpis += [("MLP accuracy", f"{c.loc['mlp', 'accuracy']:.1%}"), ("MLP ROC-AUC", f"{c.loc['mlp', 'roc_auc']:.3f}")]
    except Exception:
        pass
    try:
        o = table("ood.csv").set_index("model")
        kpis.append(("MLP on new generators", f"{o.loc['mlp', 'ood_balanced_acc']:.1%}"))
    except Exception:
        pass
    try:
        w = table("watermark_fullres.csv")
        r = w[(w["mode"] == "verified") & (w["strength"] == 0.75) & (w["attack"] == "jpeg_70")].iloc[0]
        kpis.append(("Mark after JPEG 70", f"{r['detection_rate']:.0%}"))
    except Exception:
        pass
    if kpis:
        for col, (lab, val) in zip(st.columns(len(kpis)), kpis):
            col.metric(lab, val)

    tables = [("Classification on CIFAKE test set", "classification.csv"),
              ("Modern generators (out-of-distribution)", "ood.csv"),
              ("Modern generators, per generator", "ood_by_generator.csv"),
              ("K-Means clustering", "clustering.csv"),
              ("Watermark on full-size photos", "watermark_fullres.csv"),
              ("Watermark robustness (32×32 benchmark)", "watermark.csv")]
    for title, fname in tables:
        df = table(fname)
        if df is not None:
            rlog.step(f"Load {fname}", f"{len(df)} rows × {len(df.columns)} columns")
            with st.expander(title, expanded=(fname == "classification.csv")):
                first = [c for c in ("model", "source", "generator", "attack", "group", "k") if c in df]
                st.dataframe(df[first + [c for c in df if c not in first]].round(4), hide_index=True,
                             use_container_width=True)
        else:
            rlog.step(f"Load {fname}", "file not found; run the stage that produces it in the Train tab", "warn")
    figs = sorted(FIG_DIR.glob("*.png")) if FIG_DIR.exists() else []
    if figs:
        st.markdown("#### Figures")
        sel = st.multiselect("Show figures", [f.name for f in figs],
                             default=[n for n in ("roc_curves.png", "cm_mlp.png", "eda_mean_fft.png",
                                                  "pca_scatter.png", "wm_examples.png") if (FIG_DIR / n).exists()])
        for n in sel:
            st.image(str(FIG_DIR / n), caption=n)
        rlog.step("Load figures", f"{len(figs)} available, {len(sel)} shown")
    else:
        rlog.step("Load figures", "no figures found in outputs/figures", "warn")
    rlog.show("results", "results|" + str(len(rlog.rows)), expanded=False)

# ═══════════════ Tab 6: Logs ═════════════════════════════════════════════════
with tab6:
    st.subheader("Activity log")
    st.caption("Every detect, watermark and decode run in this session, newest first, with step timings. "
               "Also appended to code/outputs/logs/demo_activity.log.")
    hist = list(st.session_state.get("history", {}).values())
    if not hist:
        empty_state("Nothing yet. Analyse, watermark or decode an image first.")
    else:
        b1, b2 = st.columns([1, 1])
        b1.download_button("Download full session log (.txt)", (chr(10) * 2).join(h.text() for h in reversed(hist)),
                           "session_log.txt", key="sess_dl")
        if b2.button("Clear session log"):
            st.session_state["history"] = {}
            st.rerun()
        for i, h in enumerate(reversed(hist)):
            with st.expander(f"{h.started:%H:%M:%S} · {h.kind} · {h.subject} · {h.total_ms} ms", expanded=(i == 0)):
                st.dataframe(h.frame(), hide_index=True, use_container_width=True)
