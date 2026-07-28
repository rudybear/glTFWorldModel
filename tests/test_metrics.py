"""Cross-validation of ``gltfworld.eval.metrics`` (our canonical PSNR/SSIM/MSE
implementations) against independent references: ``scikit-image`` (primary
anchor, per DESIGN.md/PRETRAINING_GATE.md) and ``torchmetrics``
(supplementary; see the module-level note below on a known, benign SSIM
border-handling difference between the two reference libraries themselves).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from gltfworld.eval.metrics import DATA_RANGE_UINT8, mse, psnr, ssim

_uint8_image = st.integers(min_value=16, max_value=64)


def _random_image(rng: np.random.Generator, h: int, w: int, channels: int | None) -> np.ndarray:
    shape = (h, w, channels) if channels else (h, w)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


# --- PSNR: must match skimage EXACTLY (same data_range convention) -----------


@given(seed=st.integers(0, 10_000), h=_uint8_image, w=_uint8_image, channels=st.sampled_from([None, 3]))
@settings(max_examples=40, deadline=None)
def test_psnr_matches_skimage_exactly_random(seed: int, h: int, w: int, channels: int | None):
    rng = np.random.default_rng(seed)
    a = _random_image(rng, h, w, channels)
    b = _random_image(rng, h, w, channels)

    ours = psnr(a, b, data_range=DATA_RANGE_UINT8)
    theirs = peak_signal_noise_ratio(a, b, data_range=255)
    assert ours == pytest.approx(theirs, abs=1e-9)


def test_psnr_matches_skimage_exactly_identical_images():
    rng = np.random.default_rng(42)
    a = _random_image(rng, 32, 32, 3)
    assert psnr(a, a) == float("inf")
    assert peak_signal_noise_ratio(a, a, data_range=255) == float("inf")


@given(seed=st.integers(0, 10_000), sigma=st.floats(0.5, 30.0))
@settings(max_examples=20, deadline=None)
def test_psnr_matches_skimage_exactly_structured(seed: int, sigma: float):
    """Structured pairs: a base image plus small Gaussian noise -- the
    regime PSNR is actually used for (near-duplicate frames), not just
    uniform random noise."""
    rng = np.random.default_rng(seed)
    a = _random_image(rng, 48, 48, 3)
    noise = rng.normal(0, sigma, size=a.shape)
    b = np.clip(a.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    ours = psnr(a, b)
    theirs = peak_signal_noise_ratio(a, b, data_range=255)
    assert ours == pytest.approx(theirs, abs=1e-9)


# --- SSIM: must match skimage within 1e-6 (Wang et al. 2004 parameters) ------

_SKIMAGE_SSIM_KWARGS = dict(gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)


@given(seed=st.integers(0, 10_000), h=_uint8_image, w=_uint8_image, channels=st.sampled_from([None, 3]))
@settings(max_examples=40, deadline=None)
def test_ssim_matches_skimage_within_1e6_random(seed: int, h: int, w: int, channels: int | None):
    rng = np.random.default_rng(seed)
    a = _random_image(rng, h, w, channels)
    b = _random_image(rng, h, w, channels)

    ours = ssim(a, b)
    channel_axis = -1 if channels else None
    theirs = structural_similarity(a, b, channel_axis=channel_axis, **_SKIMAGE_SSIM_KWARGS)
    assert abs(ours - theirs) <= 1e-6


@given(seed=st.integers(0, 10_000), sigma=st.floats(0.5, 30.0))
@settings(max_examples=20, deadline=None)
def test_ssim_matches_skimage_within_1e6_structured(seed: int, sigma: float):
    rng = np.random.default_rng(seed)
    a = _random_image(rng, 48, 48, 3)
    noise = rng.normal(0, sigma, size=a.shape)
    b = np.clip(a.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    ours = ssim(a, b)
    theirs = structural_similarity(a, b, channel_axis=-1, **_SKIMAGE_SSIM_KWARGS)
    assert abs(ours - theirs) <= 1e-6


def test_ssim_identical_images_is_one():
    rng = np.random.default_rng(7)
    a = _random_image(rng, 32, 32, 3)
    assert ssim(a, a) == pytest.approx(1.0, abs=1e-9)


def test_ssim_grayscale_matches_skimage():
    rng = np.random.default_rng(123)
    a = _random_image(rng, 40, 40, None)
    b = _random_image(rng, 40, 40, None)
    ours = ssim(a, b)
    theirs = structural_similarity(a, b, **_SKIMAGE_SSIM_KWARGS)
    assert abs(ours - theirs) <= 1e-6


def test_ssim_rejects_too_small_images():
    a = np.zeros((5, 5, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="at least"):
        ssim(a, a)


# --- MSE sanity ---------------------------------------------------------------


def test_mse_zero_for_identical():
    a = np.zeros((10, 10), dtype=np.uint8)
    assert mse(a, a) == 0.0


def test_mse_matches_manual_computation():
    a = np.array([0, 10, 20], dtype=np.uint8)
    b = np.array([2, 8, 25], dtype=np.uint8)
    expected = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    assert mse(a, b) == pytest.approx(expected)


# --- torchmetrics: supplementary cross-check ---------------------------------
#
# torchmetrics is a dev-dep per this milestone's spec, used here as a SECOND
# independent PSNR reference (matches ours/skimage to float32 precision).
# torchmetrics' SSIM is NOT cross-validated bit-for-bit here: it computes the
# gaussian-filtered SSIM map with edge padding (full-image output) rather
# than skimage's "filter then crop a (win_size-1)//2 border" convention (see
# gltfworld.eval.metrics module docstring) -- a real, documented difference
# between the two reference implementations' border handling, not a bug in
# either. Since skimage is this project's designated primary SSIM anchor
# (module docstring; DESIGN.md), only PSNR is cross-checked against
# torchmetrics.


def test_psnr_matches_torchmetrics_supplementary():
    torch = pytest.importorskip("torch")
    from torchmetrics.image import PeakSignalNoiseRatio

    rng = np.random.default_rng(99)
    a = _random_image(rng, 64, 64, 3)
    b = _random_image(rng, 64, 64, 3)

    ours = psnr(a, b)
    at = torch.from_numpy(a.astype(np.float32)).permute(2, 0, 1)[None]
    bt = torch.from_numpy(b.astype(np.float32)).permute(2, 0, 1)[None]
    metric = PeakSignalNoiseRatio(data_range=255.0)
    theirs = float(metric(at, bt))
    assert ours == pytest.approx(theirs, abs=1e-3)
