"""Background job runner for the demo's Train tab (no Streamlit imports, so it can be tested alone).

One pipeline stage runs at a time as a detached process whose output goes to a log file, so a job
keeps running if the page is refreshed or the demo restarts. State lives in outputs/logs/jobs/.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import psutil

CODE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
JOBS_DIR = CODE_DIR / "outputs" / "logs" / "jobs"
STATE = JOBS_DIR / "current.json"
HISTORY = JOBS_DIR / "history.jsonl"


# ── Stage catalogue ───────────────────────────────────────────────────────────

@dataclass
class Param:
    key: str
    label: str
    flag: str | None          # CLI flag; None = positional-less switch handled by `kind`
    kind: str                 # int | float | text | switch | choice
    default: object = None
    help: str = ""
    options: list | None = None
    pattern: str | None = None  # validation regex for text params


@dataclass
class Stage:
    id: str
    title: str
    script: str               # relative to code/
    summary: str
    params: list[Param] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)   # relative to code/; used for the "built" status
    needs: list[str] = field(default_factory=list)     # stage ids that should be built first
    gpu: bool = False
    long: str = ""            # rough duration hint


SUBSET = Param("subset", "Images per class (empty = all)", "--subset", "int", None,
               "Use a small number (e.g. 5000) for a quick test run")
WORKERS = Param("workers", "Worker processes", "--workers", "int", 4, "Parallel processes for CPU-heavy work")

STAGES: list[Stage] = [
    Stage("01", "Preprocess", "01_preprocess.py",
          "Load CIFAKE, resize to 32x32, split train/val/test and save as arrays.",
          [SUBSET], ["outputs/features/X_img_train.npy"], long="1-3 min"),
    Stage("02", "Feature extraction", "02_feature_extraction.py",
          "Compute 176 handcrafted features per image (noise residue, FFT/DCT, colour, GLCM texture).",
          [SUBSET, WORKERS], ["outputs/features/X_feat_train.npy"], ["01"], long="10-40 min on all data"),
    Stage("03", "Scale + PCA", "03_pca_reduce.py",
          "Standardise the features and reduce them with PCA (keeps 95% of the variance).",
          [], ["outputs/models/pca.joblib"], ["02"], long="under 1 min"),
    Stage("04", "Train classifiers", "04_classify.py",
          "Train Decision Tree (Gini and entropy), MLP and baselines; evaluate, run McNemar and ablation.",
          [Param("resume", "Reuse saved decision trees", "--resume", "switch", False,
                 "Skip the slow decision-tree grid search and reuse the saved parameters")],
          ["outputs/models/mlp.joblib", "outputs/tables/classification.csv"], ["03"], long="up to hours without Reuse"),
    Stage("05", "K-Means clustering", "05_cluster.py",
          "Cluster the PCA features and score purity / ARI / NMI.",
          [], ["outputs/tables/clustering.csv"], ["03"], long="1-5 min"),
    Stage("06", "Train watermark model", "06_watermark_train.py",
          "Train the encoder/decoder pair that embeds and reads the hidden message.",
          [Param("bits", "Message bits", "--bits", "int", None, "Empty = value from config (16)"),
           Param("alpha", "Max pixel change (alpha)", "--alpha", "float", None, "Empty = value from config (0.03)"),
           Param("epochs", "Epochs", "--epochs", "int", None, "Empty = value from config (20)"),
           Param("lam_img", "Image-quality loss weight", "--lam-img", "float", None, ""),
           Param("tag", "Checkpoint tag (optional)", "--tag", "text", "",
                 "Saves wm_enc_<tag>.pt instead of overwriting the main model", pattern=r"[A-Za-z0-9_]{0,24}"),
           SUBSET],
          ["outputs/models/wm_enc.pt", "outputs/models/wm_dec.pt"], ["01"], gpu=True, long="1-3 h on a laptop GPU"),
    Stage("07", "Watermark benchmark (32x32)", "07_watermark_eval.py",
          "Bit accuracy, image quality and detection under JPEG, resize, crop, blur and noise.",
          [Param("subset", "Test images", "--subset", "int", 2000)],
          ["outputs/tables/watermark.csv"], ["06"], long="2-5 min"),
    Stage("07b", "Watermark on full-size photos", "07b_watermark_fullres_eval.py",
          "Embed in real full-size photos and measure PSNR, SSIM and detection after attacks.",
          [Param("n", "Photos", "--n", "int", 150),
           Param("max_side", "Longest side (px)", "--max-side", "int", 768),
           Param("strengths", "Strengths", "--strengths", "text", "1.0,0.75,0.5", "Comma separated",
                 pattern=r"[0-9.]+(,[0-9.]+)*")],
          ["outputs/tables/watermark_fullres.csv"], ["06"], long="5-15 min"),
    Stage("08", "Combined (hybrid) evaluation", "08_hybrid_eval.py",
          "Check how watermarking interacts with the forensic verdicts.",
          [Param("subset", "Test images", "--subset", "int", 1000)],
          ["outputs/tables/hybrid.csv"], ["04", "06"], long="2-5 min"),
    Stage("ood-build", "Build modern-generator test set", "data_ood/build_modern.py",
          "Resize the downloaded GPT Image / Nano Banana / FLUX images to 32x32 arrays.",
          [Param("per_source", "Images per source", "--per-source", "int", 1000)],
          ["data_ood/out/X_img_ood.npy"], long="1-3 min"),
    Stage("09", "Modern-generator evaluation", "09_ood_eval.py",
          "Run the trained detectors on the modern-generator images (per-generator detection rates).",
          [WORKERS], ["outputs/tables/ood.csv"], ["04", "ood-build"], long="2-10 min"),
    Stage("samples", "Rebuild demo samples", "app/make_samples.py",
          "Rebuild the sample gallery, including the pre-watermarked images (needs the watermark model).",
          [], ["app/samples/samples.json"], ["06"], long="under 1 min"),
]
BY_ID = {s.id: s for s in STAGES}

RUN_ALL = Stage("all", "Run the whole pipeline", "run_all.py",
                "Stages 01 to 08 in order. Stops at the first failure.",
                [SUBSET, WORKERS,
                 Param("skip_wm", "Skip watermark stages (no GPU)", "--skip-watermark", "switch", False)],
                long="hours on all data; far less with a small subset")


def build_args(stage: Stage, values: dict) -> list[str]:
    """Turn widget values into a validated argv list (no shell is involved)."""
    args: list[str] = []
    for p in stage.params:
        v = values.get(p.key, p.default)
        if p.kind == "switch":
            if v:
                args.append(p.flag)
            continue
        if v is None or v == "":
            continue
        if p.kind == "text" and not re.fullmatch(p.pattern or r"[\w.,-]*", str(v)):
            raise ValueError(f"Invalid value for {p.label}")
        args += [p.flag, str(v)]
    return args


def stage_status(stage: Stage) -> tuple[bool, str]:
    """(built, 'updated <time>' or 'not built') judged by whether the output files exist."""
    paths = [CODE_DIR / o for o in stage.outputs]
    if paths and all(p.exists() for p in paths):
        t = max(p.stat().st_mtime for p in paths)
        return True, "built " + datetime.fromtimestamp(t).strftime("%d %b %H:%M")
    return False, "not built"


# ── Job control ───────────────────────────────────────────────────────────────

def _read_state() -> dict | None:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _alive(st: dict) -> bool:
    try:
        p = psutil.Process(st["pid"])
        return abs(p.create_time() - st["create_time"]) < 2 and p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except (psutil.Error, KeyError):
        return False


def current() -> dict | None:
    """The latest job with a computed 'status': running | done | failed | stopped."""
    st = _read_state()
    if not st:
        return None
    exit_file = Path(st["exit_file"])
    if exit_file.exists():
        try:
            rc = int(exit_file.read_text(encoding="utf-8").strip())
        except ValueError:
            rc = -1
        st["status"] = "stopped" if st.get("stopped") else ("done" if rc == 0 else "failed")
        st["returncode"] = rc
        st["ended"] = exit_file.stat().st_mtime
    elif _alive(st):
        st["status"] = "running"
        st["ended"] = time.time()
    else:
        st["status"] = "stopped"
        st["ended"] = st.get("ended") or time.time()
    return st


def is_running() -> bool:
    st = current()
    return bool(st and st["status"] == "running")


def start(title: str, script: str, args: list[str]) -> dict:
    """Launch `python -u <script> <args>` detached, logging to outputs/logs/jobs/. One job at a time."""
    if is_running():
        raise RuntimeError("Another job is already running.")
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_").lower()
    log = JOBS_DIR / f"{stamp}_{slug}.log"
    exit_file = JOBS_DIR / f"{stamp}_{slug}.exit"
    cmd = [sys.executable, "-u", str(APP_DIR / "_runner.py"), str(exit_file),
           sys.executable, "-u", str(CODE_DIR / script), *args]
    kw: dict = {}
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    with open(log, "ab") as fh:
        fh.write(f"$ python {script} {' '.join(args)}\n".encode())
        proc = subprocess.Popen(cmd, cwd=CODE_DIR, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"}, **kw)
    state = dict(title=title, script=script, args=args, pid=proc.pid,
                 create_time=psutil.Process(proc.pid).create_time(), started=time.time(),
                 log=str(log), exit_file=str(exit_file), recorded=False, stopped=False)
    STATE.write_text(json.dumps(state), encoding="utf-8")
    return state


def stop() -> None:
    """Kill the running job and everything it started."""
    st = _read_state()
    if not st or not _alive(st):
        return
    st["stopped"] = True
    STATE.write_text(json.dumps(st), encoding="utf-8")
    try:
        root = psutil.Process(st["pid"])
        procs = root.children(recursive=True) + [root]
    except psutil.Error:
        return
    for p in procs:
        try:
            p.kill()
        except psutil.Error:
            pass
    Path(st["exit_file"]).write_text("-9", encoding="utf-8")


def record_if_finished() -> dict | None:
    """Append a finished job to the history once. Returns the job if it was just recorded."""
    st = current()
    if not st or st["status"] == "running" or st.get("recorded"):
        return None
    st["recorded"] = True
    raw = _read_state() or {}
    raw["recorded"] = True
    STATE.write_text(json.dumps(raw), encoding="utf-8")
    entry = dict(title=st["title"], status=st["status"], started=st["started"], ended=st["ended"],
                 log=st["log"], cmd=f"{st['script']} {' '.join(st['args'])}".strip())
    with open(HISTORY, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return st


def history(n: int = 10) -> list[dict]:
    try:
        lines = HISTORY.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [json.loads(l) for l in lines[-n:]][::-1]


def tail(log: str | Path, lines: int = 200, max_bytes: int = 65536) -> str:
    """Last lines of a log; carriage-return progress bars collapse to their latest state."""
    try:
        with open(log, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            data = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    out = []
    for line in data.replace("\r\n", "\n").split("\n"):
        out.append(line.split("\r")[-1] if "\r" in line else line)
    return "\n".join(out[-lines:])


def fmt_duration(sec: float) -> str:
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else (f"{m}m {s:02d}s" if m else f"{s}s")
