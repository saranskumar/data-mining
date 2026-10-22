"""
tests/test_features.py — smoke tests for feature extraction, PCA, and model shapes.

Run: pytest tests/ -v
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CODE_DIR))

from utils.feature_utils import (
    color_features, extract_all, freq_features, glcm_features, noise_features,
    load_feature_names, save_feature_names,
)
from utils.metrics import (
    bit_accuracy, binomial_threshold, cluster_purity, psnr as compute_psnr,
)


@pytest.fixture
def dummy_img():
    """Random float32 (32,32,3) image."""
    rng = np.random.default_rng(42)
    return rng.random((32, 32, 3)).astype(np.float32)


@pytest.fixture
def smooth_img():
    """Near-flat image (tests edge cases in GLCM, autocorrelation)."""
    return np.full((32, 32, 3), 0.5, dtype=np.float32)


# ─── Feature group tests ──────────────────────────────────────────────────────

def test_noise_features_shape(dummy_img):
    vec, names = noise_features(dummy_img)
    assert len(vec) == len(names), "Noise: vector length != names length"
    assert not np.any(np.isnan(vec)), "Noise: NaN in output"
    assert not np.any(np.isinf(vec)), "Noise: Inf in output"


def test_noise_features_smooth(smooth_img):
    vec, names = noise_features(smooth_img)
    assert not np.any(np.isnan(vec)), "Noise (smooth): NaN"


def test_freq_features_shape(dummy_img):
    vec, names = freq_features(dummy_img)
    assert len(vec) == len(names), "Freq: vector length != names length"
    assert not np.any(np.isnan(vec))


def test_color_features_shape(dummy_img):
    vec, names = color_features(dummy_img)
    assert len(vec) == len(names), "Color: vector length != names length"
    assert not np.any(np.isnan(vec))


def test_glcm_features_shape(dummy_img):
    vec, names = glcm_features(dummy_img)
    assert len(vec) == len(names), "GLCM: vector length != names length"
    assert not np.any(np.isnan(vec))


def test_glcm_flat_image(smooth_img):
    """GLCM must not produce NaN on near-constant images."""
    vec, _ = glcm_features(smooth_img)
    assert not np.any(np.isnan(vec)), "GLCM: NaN on flat image"


def test_extract_all_consistent(dummy_img):
    vec1, names1 = extract_all(dummy_img)
    vec2, names2 = extract_all(dummy_img)
    assert names1 == names2, "extract_all: names not deterministic"
    np.testing.assert_array_almost_equal(vec1, vec2, err_msg="extract_all: not deterministic")
    assert len(vec1) == len(names1)


def test_no_nan_in_extract_all(dummy_img, smooth_img):
    for img in [dummy_img, smooth_img]:
        vec, _ = extract_all(img)
        assert not np.any(np.isnan(vec)), "extract_all: NaN found"
        assert not np.any(np.isinf(vec)), "extract_all: Inf found"


# ─── Feature name I/O ─────────────────────────────────────────────────────────

def test_feature_names_roundtrip(tmp_path, dummy_img):
    vec, names = extract_all(dummy_img)
    p = tmp_path / "feature_names.json"
    save_feature_names(names, p)
    loaded = load_feature_names(p)
    assert loaded == names, "Feature names round-trip failed"


# ─── Noise vs smooth discriminability ─────────────────────────────────────────

def test_noise_energy_ordering(smooth_img):
    rng = np.random.default_rng(7)
    noisy_img = np.clip(smooth_img + rng.normal(0, 0.1, smooth_img.shape).astype(np.float32), 0, 1)
    vec_smooth, names = noise_features(smooth_img)
    vec_noisy, _      = noise_features(noisy_img)
    # Gaussian noise energy (R channel) should be higher in noisy image
    energy_idx = names.index("noise_gauss_R_energy")
    assert vec_noisy[energy_idx] > vec_smooth[energy_idx], \
        "Noisy image should have higher noise energy than smooth image"


# ─── Metrics ──────────────────────────────────────────────────────────────────

def test_bit_accuracy():
    a = np.array([[1, 0, 1, 0]])
    b = np.array([[1, 0, 1, 0]])
    assert bit_accuracy(a, b) == 1.0
    c = np.array([[0, 1, 0, 1]])
    assert bit_accuracy(a, c) == 0.0


def test_psnr_identical():
    img = np.random.rand(32, 32, 3).astype(np.float32)
    assert compute_psnr(img, img) >= 99.0


def test_psnr_different():
    img1 = np.zeros((32, 32, 3), dtype=np.float32)
    img2 = np.ones((32, 32, 3), dtype=np.float32)
    assert compute_psnr(img1, img2) < 1.0


def test_cluster_purity():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert cluster_purity(y_true, y_pred) == 1.0
    y_pred2 = np.array([0, 1, 0, 1])
    assert cluster_purity(y_true, y_pred2) == 0.5


def test_binomial_threshold():
    t = binomial_threshold(48, 1e-3)
    assert 0.5 < t < 1.0, "Threshold should be above random chance"


# ─── Watermark model shape tests ──────────────────────────────────────────────

def test_encoder_decoder_shapes():
    pytest.importorskip("torch")
    import torch
    from utils.wm_model import EncoderCNN, DecoderCNN

    L = 48
    enc = EncoderCNN(message_bits=L)
    dec = DecoderCNN(message_bits=L)

    cover = torch.rand(2, 3, 32, 32)
    msg   = torch.randint(0, 2, (2, L)).float()

    marked = enc(cover, msg)
    assert marked.shape == (2, 3, 32, 32), f"Encoder output shape wrong: {marked.shape}"
    assert marked.min().item() >= -0.01, "Encoder output below 0"
    assert marked.max().item() <= 1.01, "Encoder output above 1"

    logits = dec(marked)
    assert logits.shape == (2, L), f"Decoder output shape wrong: {logits.shape}"


def test_noise_layer_passthrough():
    pytest.importorskip("torch")
    import torch
    from utils.wm_model import NoiseLayer

    nl = NoiseLayer(probs={"identity": 1.0, "gaussian": 0.0, "blur": 0.0,
                           "resize": 0.0, "crop": 0.0, "jpeg_approx": 0.0})
    x = torch.rand(4, 3, 32, 32)
    out = nl(x)
    assert out.shape == x.shape
