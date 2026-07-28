"""Unit tests for ``gltfworld.models.matching``: the cube's 24-element
rotational symmetry group, symmetry-aware rotation loss invariances (box/
cylinder/sphere), Hungarian matching correctness (incl. distractor queries),
and the full set-prediction loss (shape/finiteness/backward)."""

from __future__ import annotations

import numpy as np
import torch

from gltfworld.models.matching import (
    BOX_IDX,
    CYLINDER_IDX,
    SPHERE_IDX,
    CUBE_SYMMETRY_QUATS,
    MatchCostWeights,
    axis_alignment_angle,
    box_min_symmetry_angle,
    compute_perception_losses,
    hungarian_match,
    quat_to_local_y_axis,
    symmetry_rotation_angle,
    symmetry_rotation_loss,
)
from gltfworld.models.perception import PerceptionDETR
from gltfworld.models.rotations import axis_angle_to_quat, quat_hemisphere, quat_multiply, quat_normalize, quat_to_matrix

torch.manual_seed(0)


def _random_unit_quat(seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(4,)).astype(np.float32)
    q = q / np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return torch.from_numpy(q)[None]  # (1, 4)


# --- cube symmetry group -------------------------------------------------------


def test_cube_symmetry_group_has_24_proper_rotations():
    assert CUBE_SYMMETRY_QUATS.shape == (24, 4)
    matrices = quat_to_matrix(CUBE_SYMMETRY_QUATS)
    # every matrix must be orthogonal with det +1 (a proper rotation).
    should_be_identity = torch.matmul(matrices, matrices.transpose(-1, -2))
    identity = torch.eye(3).expand(24, 3, 3)
    np.testing.assert_allclose(should_be_identity.numpy(), identity.numpy(), atol=1e-5)
    dets = torch.linalg.det(matrices)
    np.testing.assert_allclose(dets.numpy(), np.ones(24), atol=1e-5)


def test_cube_symmetry_group_all_distinct():
    # no two of the 24 quaternions represent the same rotation (up to the
    # double-cover sign, already hemisphere-normalized by matrix_to_quat).
    q = CUBE_SYMMETRY_QUATS
    diffs = torch.cdist(q, q)
    diffs.fill_diagonal_(float("inf"))
    assert diffs.min() > 1e-3


# --- box: min geodesic angle over the 24 symmetries is ~0 for any symmetry ----


def test_box_rotation_loss_zero_for_every_cube_symmetry():
    gt_quat = _random_unit_quat(seed=1)
    for i in range(24):
        symmetry = CUBE_SYMMETRY_QUATS[i : i + 1]
        pred_quat = quat_multiply(gt_quat, symmetry)  # same physical box orientation
        angle = box_min_symmetry_angle(pred_quat, gt_quat)
        assert angle.item() < 1e-4, f"symmetry {i}: angle={angle.item()}"


def test_box_rotation_loss_nonzero_for_a_genuinely_different_orientation():
    gt_quat = _random_unit_quat(seed=2)
    # a 45-degree rotation about a generic axis is not any of the cube's
    # 24 symmetries (all of which are multiples of 90 degrees about a
    # coordinate axis or a body diagonal).
    pred_quat = quat_multiply(gt_quat, axis_angle_to_quat(torch.tensor([[0.3, 0.5, 0.1]])))
    angle = box_min_symmetry_angle(pred_quat, gt_quat)
    assert angle.item() > 0.1


def test_symmetry_rotation_loss_dispatches_box():
    gt_quat = _random_unit_quat(seed=3)
    symmetry = CUBE_SYMMETRY_QUATS[5:6]
    pred_quat = quat_multiply(gt_quat, symmetry)
    shape_idx = torch.tensor([BOX_IDX])
    loss = symmetry_rotation_loss(pred_quat, gt_quat, shape_idx)
    assert loss.item() < 1e-6


# --- cylinder: spin- and flip-invariant axis alignment ------------------------


def test_cylinder_loss_zero_for_spin_about_local_axis():
    gt_quat = _random_unit_quat(seed=4)
    spin = axis_angle_to_quat(torch.tensor([[0.0, 1.234, 0.0]]))  # local Y spin
    pred_quat = quat_multiply(gt_quat, spin)
    shape_idx = torch.tensor([CYLINDER_IDX])
    loss = symmetry_rotation_loss(pred_quat, gt_quat, shape_idx)
    assert loss.item() < 1e-6


def test_cylinder_loss_zero_for_180_degree_flip():
    gt_quat = _random_unit_quat(seed=5)
    flip = axis_angle_to_quat(torch.tensor([[np.pi, 0.0, 0.0]]))  # 180 about local X
    pred_quat = quat_multiply(gt_quat, flip)
    shape_idx = torch.tensor([CYLINDER_IDX])
    loss = symmetry_rotation_loss(pred_quat, gt_quat, shape_idx)
    assert loss.item() < 1e-6


def test_cylinder_loss_nonzero_for_a_genuinely_tilted_axis():
    gt_quat = _random_unit_quat(seed=6)
    tilt = axis_angle_to_quat(torch.tensor([[0.0, 0.0, 0.6]]))  # tilts the Y axis
    pred_quat = quat_multiply(gt_quat, tilt)
    shape_idx = torch.tensor([CYLINDER_IDX])
    loss = symmetry_rotation_loss(pred_quat, gt_quat, shape_idx)
    assert loss.item() > 0.01


def test_axis_alignment_angle_antiparallel_is_zero():
    axis = torch.tensor([[0.0, 1.0, 0.0]])
    angle = axis_alignment_angle(axis, -axis)
    assert angle.item() < 1e-6


def test_quat_to_local_y_axis_identity():
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    axis = quat_to_local_y_axis(identity)
    np.testing.assert_allclose(axis.numpy(), [[0.0, 1.0, 0.0]], atol=1e-6)


# --- sphere: rotation is entirely ignored -------------------------------------


def test_sphere_rotation_loss_always_zero_regardless_of_orientation():
    gt_quat = _random_unit_quat(seed=7)
    shape_idx = torch.tensor([SPHERE_IDX])
    for seed in range(8, 13):
        pred_quat = _random_unit_quat(seed)
        loss = symmetry_rotation_loss(pred_quat, gt_quat, shape_idx)
        assert loss.item() == 0.0


def test_symmetry_rotation_angle_batched_mixed_shapes():
    gt_quat = torch.cat([_random_unit_quat(s) for s in range(20, 23)], dim=0)
    pred_quat = torch.cat([_random_unit_quat(s) for s in range(30, 33)], dim=0)
    shape_idx = torch.tensor([SPHERE_IDX, BOX_IDX, CYLINDER_IDX])
    angles = symmetry_rotation_angle(pred_quat, gt_quat, shape_idx)
    assert angles.shape == (3,)
    assert angles[0].item() == 0.0  # sphere
    assert torch.isfinite(angles).all()


# --- Hungarian matching --------------------------------------------------------


def test_hungarian_match_hand_built_with_distractors():
    """3 real GT objects, 2 clearly-correct predictions each, plus 2
    distractor queries far from everything -- the distractors must stay
    unmatched."""
    pred_position = torch.tensor(
        [[[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 10.0, 10.0]]]
    )
    pred_class_logits = torch.zeros(1, 5, 3)
    pred_size = torch.zeros(1, 5, 3)

    gt_position = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    gt_class = torch.tensor([[0, 1, 2, 0, 0]])
    gt_size = torch.zeros(1, 5, 3)
    gt_mask = torch.tensor([[True, True, True, False, False]])

    matches = hungarian_match(
        pred_position, pred_class_logits, pred_size, gt_position, gt_class, gt_size, gt_mask, MatchCostWeights()
    )
    query_idx, gt_idx = matches[0]
    assert set(query_idx.tolist()) == {0, 2, 3}
    assert set(gt_idx.tolist()) == {0, 1, 2}
    # the pairing itself must be the obviously-correct one (query i <-> its
    # nearest GT), not just any valid assignment.
    pairing = dict(zip(query_idx.tolist(), gt_idx.tolist()))
    assert pairing[0] == 0 and pairing[2] == 1 and pairing[3] == 2


def test_hungarian_match_no_real_objects_returns_empty():
    pred_position = torch.zeros(1, 5, 3)
    pred_class_logits = torch.zeros(1, 5, 3)
    pred_size = torch.zeros(1, 5, 3)
    gt_position = torch.zeros(1, 5, 3)
    gt_class = torch.zeros(1, 5, dtype=torch.long)
    gt_size = torch.zeros(1, 5, 3)
    gt_mask = torch.zeros(1, 5, dtype=torch.bool)

    matches = hungarian_match(
        pred_position, pred_class_logits, pred_size, gt_position, gt_class, gt_size, gt_mask, MatchCostWeights()
    )
    query_idx, gt_idx = matches[0]
    assert query_idx.size == 0 and gt_idx.size == 0


# --- full set-prediction loss ---------------------------------------------------


def test_compute_perception_losses_shape_finite_and_backward():
    model = PerceptionDETR()
    rgb = torch.rand(3, 256, 256, 3)
    out = model(rgb)

    states = torch.zeros(3, 5, 22)
    states[..., 6] = 1.0  # identity quat w component
    mask = torch.zeros(3, 5, dtype=torch.bool)
    mask[:, :2] = True
    class_ids = torch.zeros(3, 5, dtype=torch.long)

    total, comps, matches = compute_perception_losses(out, states, mask, class_ids)
    assert torch.isfinite(total)
    for v in comps.values():
        assert np.isfinite(v)
    assert len(matches) == 3
    total.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(np.isfinite(g) for g in grad_norms)


def test_compute_perception_losses_zero_for_perfect_prediction():
    """If pred exactly matches GT (existence 1 for real objects via a huge
    logit, exact pos/size/shape/class/quat), every loss term must be ~0."""
    n_max = 5
    n_real = 3
    states = torch.zeros(1, n_max, 22)
    states[0, :n_real, 0:3] = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.5, 0.5]])
    states[0, :n_real, 6] = 1.0  # identity quat
    states[0, :n_real, 16:19] = 0.1
    mask = torch.zeros(1, n_max, dtype=torch.bool)
    mask[0, :n_real] = True
    class_ids = torch.zeros(1, n_max, dtype=torch.long)
    class_ids[0, :n_real] = torch.tensor([0, 1, 2])

    pred = {
        "existence_logit": torch.tensor([[10.0, 10.0, 10.0, -10.0, -10.0]]),
        "position": states[..., 0:3].clone(),
        "quat": states[..., 3:7].clone(),
        "size": states[..., 16:19].clone(),
        "shape_logits": torch.zeros(1, n_max, 3),
        "class_logits": torch.zeros(1, n_max, 3),
    }
    # states never set shape_onehot (13:16), so it defaults to all-zero ->
    # argmax -> shape 0 ("sphere") for every real GT row; match that with a
    # confident shape-0 prediction so the shape-CE term is ~0 too.
    pred["shape_logits"][0, :n_real, 0] = 10.0
    # class 0 for query0, class 1 for query1, class 2 for query2 -- high logits.
    pred["class_logits"][0, 0, 0] = 10.0
    pred["class_logits"][0, 1, 1] = 10.0
    pred["class_logits"][0, 2, 2] = 10.0

    total, comps, matches = compute_perception_losses(pred, states, mask, class_ids)
    assert comps["position"] < 1e-6
    assert comps["size"] < 1e-6
    assert comps["rotation"] < 1e-6
    assert comps["shape"] < 1e-3  # sphere is shape 0 -> uniform logits -> some CE, but small vs a wrong prediction
    assert comps["cls"] < 1e-3
    assert comps["existence"] < 1e-3
