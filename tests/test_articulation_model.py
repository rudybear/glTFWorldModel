"""Unit tests for ``gltfworld.models.articulation`` (V9 joint-state
estimator): forward shapes/dtypes, parameter-count sanity, loss correctness
on synthetic perfect/corrupted predictions, and -- the design decision this
milestone documents at length -- that the axis loss is *directed* (a
sign-flipped axis prediction is scored as wrong, not free). CPU-only, no
dataset/GPU needed.
"""

from __future__ import annotations

import pytest
import torch

from gltfworld.models.articulation import (
    ArticulationEstimator,
    LossWeights,
    compute_articulation_losses,
    count_params,
    denormalize_joint_pos,
)


def test_forward_shapes_and_finiteness():
    torch.manual_seed(0)
    model = ArticulationEstimator(d_model=64)
    rgb = torch.rand(4, 256, 256, 3)
    out = model(rgb)

    assert out["joint_pos_norm"].shape == (4,)
    assert out["type_logits"].shape == (4, 2)
    assert out["axis"].shape == (4, 3)
    for v in out.values():
        assert torch.isfinite(v).all()

    # axis must be unit-norm (L2-normalized in forward()).
    norms = torch.linalg.norm(out["axis"], dim=-1)
    torch.testing.assert_close(norms, torch.ones(4), atol=1e-5, rtol=0)


def test_param_count_reasonable():
    model = ArticulationEstimator()
    n = count_params(model)
    # "small model" per the milestone spec -- sanity bound, not a tuned target.
    assert 1_000_000 < n < 20_000_000


def test_denormalize_joint_pos_roundtrip():
    joint_pos_raw = torch.tensor([0.0, 0.5, 1.3, -0.1])
    limit_min = torch.tensor([0.0, 0.0, 0.0, -0.2])
    limit_max = torch.tensor([1.9, 1.9, 1.9, 0.3])
    norm = (joint_pos_raw - limit_min) / (limit_max - limit_min)
    recovered = denormalize_joint_pos(norm, limit_min, limit_max)
    torch.testing.assert_close(recovered, joint_pos_raw, atol=1e-6, rtol=0)


def _perfect_pred(axis_gt: torch.Tensor, joint_pos_norm_gt: torch.Tensor, type_gt: torch.Tensor) -> dict:
    b = axis_gt.shape[0]
    type_logits = torch.full((b, 2), -10.0)
    type_logits[torch.arange(b), type_gt] = 10.0
    return {
        "joint_pos_norm": joint_pos_norm_gt.clone(),
        "type_logits": type_logits,
        "axis": axis_gt.clone(),
    }


def test_loss_zero_for_perfect_prediction():
    axis_gt = torch.eye(3)[torch.tensor([0, 1, 2])]
    joint_pos_norm_gt = torch.tensor([0.2, 0.5, 0.8])
    type_gt = torch.tensor([0, 1, 0])

    pred = _perfect_pred(axis_gt, joint_pos_norm_gt, type_gt)
    total, comp = compute_articulation_losses(pred, joint_pos_norm_gt, type_gt, axis_gt, LossWeights())

    assert comp["joint_pos"] == pytest.approx(0.0, abs=1e-6)
    assert comp["axis"] == pytest.approx(0.0, abs=1e-6)
    assert total.item() < 1e-3  # cross-entropy on a 20-logit-margin one-hot is tiny but not bit-exact 0


def test_axis_loss_is_directed_not_sign_invariant():
    """A sign-flipped axis prediction (pointing exactly opposite the true
    axis) must be scored at the *worst* possible loss (cos = -1 -> loss =
    2), not zero -- see gltfworld.models.articulation's module docstring
    "Axis regression: directed, not sign-invariant" for why a
    sign-invariant loss would be wrong for this dataset's convention."""
    axis_gt = torch.tensor([[1.0, 0.0, 0.0]])
    joint_pos_norm_gt = torch.tensor([0.3])
    type_gt = torch.tensor([0])

    pred = _perfect_pred(axis_gt, joint_pos_norm_gt, type_gt)
    pred["axis"] = -axis_gt.clone()  # exact sign flip

    _total, comp = compute_articulation_losses(pred, joint_pos_norm_gt, type_gt, axis_gt, LossWeights())
    assert comp["axis"] == pytest.approx(2.0, abs=1e-6)


def test_loss_known_joint_pos_corruption():
    axis_gt = torch.eye(3)[torch.tensor([0, 1, 2])]
    joint_pos_norm_gt = torch.tensor([0.2, 0.5, 0.8])
    type_gt = torch.tensor([0, 1, 0])

    pred = _perfect_pred(axis_gt, joint_pos_norm_gt, type_gt)
    pred["joint_pos_norm"] = joint_pos_norm_gt + 0.1  # known constant offset

    _total, comp = compute_articulation_losses(pred, joint_pos_norm_gt, type_gt, axis_gt, LossWeights())
    assert comp["joint_pos"] == pytest.approx(0.01, abs=1e-6)  # MSE of a constant 0.1 offset


def test_type_misclassification_increases_loss():
    axis_gt = torch.eye(3)[torch.tensor([0, 1, 2])]
    joint_pos_norm_gt = torch.tensor([0.2, 0.5, 0.8])
    type_gt = torch.tensor([0, 1, 0])

    pred = _perfect_pred(axis_gt, joint_pos_norm_gt, type_gt)
    _total_ok, comp_ok = compute_articulation_losses(pred, joint_pos_norm_gt, type_gt, axis_gt, LossWeights())

    pred_wrong = dict(pred)
    wrong_type_logits = pred["type_logits"].clone()
    wrong_type_logits[0] = torch.tensor([-10.0, 10.0])  # flip sample 0's confident prediction
    pred_wrong["type_logits"] = wrong_type_logits
    _total_bad, comp_bad = compute_articulation_losses(pred_wrong, joint_pos_norm_gt, type_gt, axis_gt, LossWeights())

    assert comp_bad["type"] > comp_ok["type"]
