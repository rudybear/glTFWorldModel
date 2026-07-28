"""Canonical eval metrics for uint8 images: MSE, PSNR, SSIM.

These are gltfworld's own from-scratch implementations (not simply
re-exports of ``skimage``/``torchmetrics``) -- ``tests/test_metrics.py``
cross-validates them against those two independent references so a bug here
can't silently poison every downstream eval number.

Parameter choices (documented, not left implicit)
--------------------------------------------------

- **``data_range``**: always ``255.0`` for the uint8 images this module is
  built for (matches ``skimage.metrics.peak_signal_noise_ratio``'s default
  for a ``uint8`` array, and is passed explicitly to
  ``skimage.metrics.structural_similarity``/``torchmetrics`` in the
  cross-validation tests since both APIs otherwise try to *infer* it from
  dtype, which is exactly the kind of implicit convention this module
  refuses to leave to chance).
- **SSIM ``gaussian_weights=True``, ``sigma=1.5``, ``truncate=3.5`` (-> an
  11-tap window), ``use_sample_covariance=False``, ``K1=0.01``,
  ``K2=0.03``**: this is deliberately the *exact* Wang et al. 2004
  configuration -- ``skimage``'s own docstring says as much ("To match the
  implementation of Wang et al. [1]_, set `gaussian_weights` to True,
  `sigma` to 1.5, `use_sample_covariance` to False"). Border effects are
  handled the same way ``skimage`` does: the local SSIM map is computed
  densely (via ``scipy.ndimage.gaussian_filter``, ``mode="reflect"``, same
  as ``skimage.filters.gaussian``'s wrapper around it) and then a
  ``(win_size - 1) // 2`` pixel border is cropped before averaging, rather
  than only computing the map on valid (non-padded) windows -- numerically
  equivalent to skimage's ``crop(S, pad).mean()`` step.
- **Multi-channel (RGB) images**: SSIM is computed independently per
  channel (no cross-channel blur) and the three channel scores are
  averaged with equal weight -- matches
  ``structural_similarity(..., channel_axis=-1)``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

DATA_RANGE_UINT8 = 255.0

_SSIM_K1 = 0.01
_SSIM_K2 = 0.03
_SSIM_SIGMA = 1.5
_SSIM_TRUNCATE = 3.5  # -> win_size = 2 * int(truncate * sigma + 0.5) + 1 = 11


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between ``a`` and ``b`` (any shape, same shape),
    computed in float64 regardless of input dtype."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = DATA_RANGE_UINT8) -> float:
    """Peak signal-to-noise ratio in dB, ``10 * log10(data_range**2 / mse)``.

    Returns ``float("inf")`` for identical images (``mse == 0``), matching
    ``skimage.metrics.peak_signal_noise_ratio``'s convention.
    """
    m = mse(a, b)
    if m == 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / m))


def _ssim_single_channel(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    filter_args = {"sigma": _SSIM_SIGMA, "truncate": _SSIM_TRUNCATE, "mode": "reflect"}
    radius = int(_SSIM_TRUNCATE * _SSIM_SIGMA + 0.5)
    win_size = 2 * radius + 1

    ux = gaussian_filter(a, **filter_args)
    uy = gaussian_filter(b, **filter_args)
    uxx = gaussian_filter(a * a, **filter_args)
    uyy = gaussian_filter(b * b, **filter_args)
    uxy = gaussian_filter(a * b, **filter_args)

    # Population covariance (not sample), per Wang et al. 2004 / skimage's
    # `use_sample_covariance=False` -- see module docstring.
    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy

    c1 = (_SSIM_K1 * data_range) ** 2
    c2 = (_SSIM_K2 * data_range) ** 2

    numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
    denominator = (ux**2 + uy**2 + c1) * (vx + vy + c2)
    s = numerator / denominator

    pad = (win_size - 1) // 2
    if pad > 0:
        s = s[pad:-pad, pad:-pad]
    return float(np.mean(s, dtype=np.float64))


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = DATA_RANGE_UINT8) -> float:
    """Structural similarity index between ``a`` and ``b``.

    Grayscale: ``(H, W)`` arrays. RGB: ``(H, W, 3)`` arrays -- SSIM is
    computed per channel and averaged (see module docstring for exact
    parameter choices, matched to Wang et al. 2004 / skimage's
    ``gaussian_weights=True`` mode).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    radius = int(_SSIM_TRUNCATE * _SSIM_SIGMA + 0.5)
    win_size = 2 * radius + 1
    if a.shape[0] < win_size or a.shape[1] < win_size:
        raise ValueError(
            f"images must be at least {win_size}x{win_size} for the {win_size}-tap SSIM window, "
            f"got {a.shape[:2]}"
        )

    if a.ndim == 2:
        return _ssim_single_channel(a, b, data_range)
    if a.ndim == 3:
        channel_scores = [
            _ssim_single_channel(a[..., c], b[..., c], data_range) for c in range(a.shape[-1])
        ]
        return float(np.mean(channel_scores))
    raise ValueError(f"expected a 2D (grayscale) or 3D (H, W, C) image, got shape {a.shape}")
