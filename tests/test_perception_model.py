"""Unit tests for ``gltfworld.models.perception.PerceptionDETR``: output
shapes, finiteness, unit-quaternion outputs, workspace bounds, and parameter
count sanity. CPU-fast, no packed dataset needed."""

from __future__ import annotations

import numpy as np
import torch

from gltfworld.models.perception import (
    N_MAX,
    NUM_CLASSES,
    NUM_SHAPES,
    POS_MAX,
    POS_MIN,
    SIZE_MAX,
    SIZE_MIN,
    PerceptionDETR,
    count_params,
)

torch.manual_seed(0)


def test_forward_output_shapes_and_finite():
    model = PerceptionDETR()
    model.eval()
    rgb = torch.rand(4, 256, 256, 3)
    with torch.no_grad():
        out = model(rgb)

    assert out["existence_logit"].shape == (4, N_MAX)
    assert out["position"].shape == (4, N_MAX, 3)
    assert out["quat"].shape == (4, N_MAX, 4)
    assert out["size"].shape == (4, N_MAX, 3)
    assert out["shape_logits"].shape == (4, N_MAX, NUM_SHAPES)
    assert out["class_logits"].shape == (4, N_MAX, NUM_CLASSES)

    for v in out.values():
        assert torch.isfinite(v).all()


def test_quat_output_is_unit_and_hemisphere_normalized():
    model = PerceptionDETR()
    model.eval()
    rgb = torch.rand(3, 256, 256, 3)
    with torch.no_grad():
        out = model(rgb)
    quat = out["quat"]
    norms = torch.linalg.norm(quat, dim=-1)
    np.testing.assert_allclose(norms.numpy(), np.ones((3, N_MAX)), atol=1e-5)
    assert (quat[..., 3] >= -1e-6).all()


def test_position_within_workspace_bounds():
    model = PerceptionDETR()
    model.eval()
    rgb = torch.rand(5, 256, 256, 3)
    with torch.no_grad():
        out = model(rgb)
    pos = out["position"].numpy()
    for i, (lo, hi) in enumerate(zip(POS_MIN, POS_MAX)):
        assert (pos[..., i] >= lo - 1e-4).all()
        assert (pos[..., i] <= hi + 1e-4).all()


def test_size_within_bounds_and_positive():
    model = PerceptionDETR()
    model.eval()
    rgb = torch.rand(2, 256, 256, 3)
    with torch.no_grad():
        out = model(rgb)
    size = out["size"].numpy()
    assert (size >= SIZE_MIN - 1e-6).all()
    assert (size <= SIZE_MAX + 1e-6).all()


def test_batch_independence():
    """Each sample's output must not depend on other samples in the batch
    (no cross-batch leakage through, e.g., an accidental reduction over dim
    0 somewhere in the head)."""
    model = PerceptionDETR()
    model.eval()
    torch.manual_seed(42)
    rgb = torch.rand(3, 256, 256, 3)
    with torch.no_grad():
        out_batched = model(rgb)
        out_single = model(rgb[1:2])

    np.testing.assert_allclose(
        out_batched["position"][1].numpy(), out_single["position"][0].numpy(), atol=1e-5
    )
    np.testing.assert_allclose(
        out_batched["existence_logit"][1].numpy(), out_single["existence_logit"][0].numpy(), atol=1e-5
    )


def test_rejects_wrong_image_size():
    model = PerceptionDETR()
    rgb = torch.rand(1, 128, 128, 3)
    try:
        model(rgb)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_param_count_reported_and_reasonable():
    model = PerceptionDETR()
    n = count_params(model)
    # Not hard-asserting the milestone spec's approximate "~12-16M" band
    # (see the module docstring's documented deviation, same precedent as
    # DESIGN.md V5's NoInteractionMLP) -- just a sanity floor/ceiling so a
    # gross architecture regression (e.g. an accidentally-untied huge layer)
    # would still fail loudly.
    assert 3_000_000 <= n <= 30_000_000, f"param count {n:,} looks implausible for this architecture"
