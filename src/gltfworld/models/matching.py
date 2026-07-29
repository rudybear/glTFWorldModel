"""Hungarian matching + DETR-style set-prediction loss for
``gltfworld.models.perception.PerceptionDETR``.

Out-of-box GT filtering (:func:`filter_out_of_box_gt`)
-------------------------------------------------------

Applied first, inside :func:`compute_perception_losses` and by
``gltfworld.eval.perception_eval`` directly, before any matching happens: a
GT object whose position lies outside the model's representable
``[POS_MIN, POS_MAX]`` workspace box (``gltfworld.models.perception`` --
some `wm-scenes-v1` objects escape the finite ground plate over an episode)
is treated as *absent* for that frame, not merely "unsupervised on pose" --
existence supervision/eval skips it too. See DESIGN.md's V6.2 postmortem for
the full rationale and measured fractions.

Matching (:func:`hungarian_match`)
-----------------------------------

Per sample in a batch, a cost matrix ``(N_MAX queries, n_real GT objects)`` is
built from ``w_pos * L2(position) + w_cls * CE(class) + w_size * L2(size)``,
restricted to the sample's *existence-eligible* GT rows only (``mask`` -- a
padded/nonexistent GT slot never enters the cost matrix, so it can never be
"matched" against). ``scipy.optimize.linear_sum_assignment`` (exact, not an
approximation) finds the minimum-cost one-to-one assignment; queries beyond
the number of real GT objects are left unmatched (existence target 0, no
pose/size/shape/class loss).

Symmetry-aware rotation loss (:func:`symmetry_rotation_loss`)
----------------------------------------------------------------

Per-object, keyed off the GT shape (``gltfworld.scene.contract.SHAPE_ORDER``
index), computed via ``torch.where`` branches (not Python control flow) so it
batches and backprops cleanly:

- **sphere**: always ``0`` -- a sphere has no distinguishable orientation at
  all (continuous rotational symmetry about every axis), so there is no
  rotation signal to supervise, full stop.
- **box**: the minimum squared geodesic angle (``gltfworld.models.rotations
  .quat_geodesic_angle``) between the predicted quaternion and the GT
  quaternion composed with each of the cube's **24** rotational symmetries
  (:data:`CUBE_SYMMETRY_QUATS`, precomputed once at import time -- not
  data-fit): ``q_equivalent = q_gt (x) S`` for each symmetry ``S`` represents
  the exact same physical box orientation (the box's local axes get
  relabeled, but the box occupies the same world-space region), so the
  "distance" between a predicted orientation and "the" GT orientation must be
  measured to the *nearest* of these 24 equivalent representations.
- **cylinder**: a cylinder has one true symmetry axis (local Y, per
  ``ObjectSpec``/``KHR_implicit_shapes`` convention -- see DESIGN.md's
  "Cylinder axis convention"), invariant to (a) spinning around that axis and
  (b) flipping 180 degrees (swapping which end points which way -- the two
  circular end-caps are indistinguishable). :func:`axis_alignment_angle`
  compares only the predicted-vs-GT *axis direction* (``R(q) @ (0, 1, 0)``),
  aligned to whichever antiparallel sign is closer before measuring the angle
  between them -- both (a) and (b) fall out for free: spinning about the axis
  doesn't change the axis direction at all, and the 180-degree flip just
  negates it, which sign-alignment cancels.

Both symmetry angles are computed via the same ``atan2``-of-norms form
``gltfworld.models.rotations.quat_geodesic_angle`` uses (rather than
``arccos``), for the identical well-behaved-gradient-near-zero reason
documented there.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from gltfworld.models.perception import POS_MAX, POS_MIN
from gltfworld.models.rotations import matrix_to_quat, quat_geodesic_angle, quat_multiply, quat_to_matrix
from gltfworld.scene.contract import SHAPE_ORDER

SPHERE_IDX = SHAPE_ORDER.index("sphere")
BOX_IDX = SHAPE_ORDER.index("box")
CYLINDER_IDX = SHAPE_ORDER.index("cylinder")

_EPS = 1e-8


# --- the cube's 24-element rotational symmetry group, as quaternions --------


def _cube_symmetry_quats() -> torch.Tensor:
    """The 24 proper (``det == +1``) signed-permutation matrices of R^3 --
    the rotational symmetry group of a cube/box -- as unit quaternions xyzw.

    Every combination of an axis permutation (``3! = 6``) and a per-axis sign
    flip (``2**3 = 8``) gives one of 48 signed permutation matrices; exactly
    half (24) are proper rotations (``det = +1``), the other 24 are
    reflections (``det = -1``, discarded -- a box's symmetry group under
    *rotation* only, not reflection).
    """
    matrices = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3), dtype=np.float64)
            for row, (col, sign) in enumerate(zip(perm, signs)):
                m[row, col] = sign
            if abs(np.linalg.det(m) - 1.0) < 1e-6:
                matrices.append(m)
    assert len(matrices) == 24, f"expected 24 proper cube symmetries, got {len(matrices)}"
    matrices_t = torch.tensor(np.stack(matrices), dtype=torch.float32)
    return matrix_to_quat(matrices_t)  # (24, 4) xyzw


CUBE_SYMMETRY_QUATS = _cube_symmetry_quats()


def box_min_symmetry_angle(pred_quat: torch.Tensor, gt_quat: torch.Tensor) -> torch.Tensor:
    """``pred_quat``/``gt_quat`` (..., 4) xyzw -> the minimum geodesic angle
    (radians, (...,)) between ``pred_quat`` and ``gt_quat`` composed with any
    of the cube's 24 rotational symmetries."""
    cube = CUBE_SYMMETRY_QUATS.to(dtype=gt_quat.dtype, device=gt_quat.device)  # (24, 4)
    lead_shape = gt_quat.shape[:-1]
    gt_expanded = gt_quat.unsqueeze(-2).expand(*lead_shape, 24, 4)
    cube_expanded = cube.reshape((1,) * len(lead_shape) + (24, 4)).expand(*lead_shape, 24, 4)
    gt_equivalents = quat_multiply(gt_expanded, cube_expanded)  # (..., 24, 4)

    pred_expanded = pred_quat.unsqueeze(-2).expand_as(gt_equivalents)
    angles = quat_geodesic_angle(pred_expanded, gt_equivalents)  # (..., 24)
    return angles.min(dim=-1).values


# --- cylinder axis alignment (spin- and 180-flip-invariant) ------------------


def quat_to_local_y_axis(quat: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (..., 4) xyzw -> the local +Y axis direction in world
    frame (..., 3) -- a cylinder's true symmetry axis, per ``ObjectSpec``'s
    documented convention (see ``gltfworld.scene.primitives``)."""
    m = quat_to_matrix(quat)  # (..., 3, 3); m @ e_y = column 1
    return m[..., :, 1]


def axis_alignment_angle(axis_a: torch.Tensor, axis_b: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Angle (radians, in ``[0, pi/2]``) between two *axes* (undirected lines,
    not vectors) -- invariant to which of the two antiparallel directions
    each side happens to point along (a cylinder's local-Y symmetry axis has
    no intrinsic "positive" end, so spinning or 180-degree-flipping it must
    not change this).

    Uses the same ``atan2``-of-norms form as
    ``gltfworld.models.rotations.quat_geodesic_angle`` (well-behaved gradient
    near 0, unlike ``arccos``): for unit vectors ``a``, ``b`` (aligned to the
    same hemisphere first), ``||a - b|| = 2*sin(theta/2)``,
    ``||a + b|| = 2*cos(theta/2)``, so ``2*atan2(||a-b||, ||a+b||) = theta``.
    """
    a = F.normalize(axis_a, dim=-1, eps=eps)
    b = F.normalize(axis_b, dim=-1, eps=eps)
    dot = (a * b).sum(dim=-1, keepdim=True)
    b_aligned = torch.where(dot < 0, -b, b)
    diff_norm = torch.linalg.norm(a - b_aligned, dim=-1)
    sum_norm = torch.linalg.norm(a + b_aligned, dim=-1)
    return 2.0 * torch.atan2(diff_norm, sum_norm)


# --- symmetry-aware rotation loss/metric, dispatched by GT shape ------------


def symmetry_rotation_angle(pred_quat: torch.Tensor, gt_quat: torch.Tensor, shape_idx: torch.Tensor) -> torch.Tensor:
    """Per-object symmetry-aware rotation *angle* (radians, (...,)), zero for
    sphere GT (no rotation is meaningful for a sphere at all). Shared by both
    the training loss (:func:`symmetry_rotation_loss`, which squares this)
    and eval (which reports it directly, in degrees, per shape)."""
    box_angle = box_min_symmetry_angle(pred_quat, gt_quat)
    pred_axis = quat_to_local_y_axis(pred_quat)
    gt_axis = quat_to_local_y_axis(gt_quat)
    cyl_angle = axis_alignment_angle(pred_axis, gt_axis)

    is_box = shape_idx == BOX_IDX
    is_cylinder = shape_idx == CYLINDER_IDX
    zeros = torch.zeros_like(box_angle)
    return torch.where(is_box, box_angle, torch.where(is_cylinder, cyl_angle, zeros))


def symmetry_rotation_loss(pred_quat: torch.Tensor, gt_quat: torch.Tensor, shape_idx: torch.Tensor) -> torch.Tensor:
    """Squared symmetry-aware rotation angle -- the training-loss form
    (matches ``gltfworld.train.train_dynamics``' own squared-geodesic-angle
    rotation term convention)."""
    return symmetry_rotation_angle(pred_quat, gt_quat, shape_idx) ** 2


# --- out-of-box GT filtering (V6.2 data-defect fix) --------------------------
#
# See DESIGN.md's V6.2 postmortem: a small but nonzero fraction of
# `wm-scenes-v1` objects escape the finite ground plate over the course of an
# episode (observed extremes: y=-30.4m, z=+17.4m) -- physically real
# trajectories that end up both (a) outside the fixed camera's frustum (never
# actually visible in the rendered RGB frame the model is asked to predict
# from) and (b) outside the position head's representable
# `[POS_MIN, POS_MAX]` workspace box (`gltfworld.models.perception` -- the
# sigmoid-into-box parameterization makes it *structurally impossible* for
# any query to ever emit a position in that region). Treating such an object
# as "exists, supervise its pose" -- as every code path did before this fix --
# poisons both training (an unreachable regression target that can only ever
# contribute noise/bias to the position loss) and eval (a match that can
# never score well for a reason that has nothing to do with model quality).
# Measured fractions of existence-eligible GT rows affected, on the
# 4,000-episode `perception-v1` pack: train 4.51%, val 4.85%, test 3.60%.
#
# Implemented once here and shared by both the training loss
# (:func:`compute_perception_losses`) and eval matching
# (`gltfworld.eval.perception_eval.run_inference`/`run_inference_baseline`)
# so the two code paths can never quietly disagree about which GT objects are
# "real" for a frame -- an out-of-box GT object is treated as *absent* (not
# just "unsupervised on pose"): existence supervision/eval skips it too,
# exactly as if it had never existed for that frame.


def in_workspace_box(position: torch.Tensor) -> torch.Tensor:
    """``position`` (..., 3) -> bool (...,), True wherever every coordinate
    lies within the model's representable workspace box
    ``[POS_MIN, POS_MAX]`` (``gltfworld.models.perception``)."""
    pos_min = position.new_tensor(POS_MIN)
    pos_max = position.new_tensor(POS_MAX)
    return ((position >= pos_min) & (position <= pos_max)).all(dim=-1)


@dataclass
class WorkspaceFilterStats:
    """``n_eligible``/``n_out_of_box`` counts (of GT rows that were
    ``gt_mask``-eligible *before* this filter) -- callers print/log
    :attr:`fraction_out_of_box` so the defect's impact stays visible (see
    ``gltfworld.data.pack``/``gltfworld.eval.perception_eval``)."""

    n_eligible: int
    n_out_of_box: int

    @property
    def fraction_out_of_box(self) -> float:
        return (self.n_out_of_box / self.n_eligible) if self.n_eligible > 0 else 0.0


def filter_out_of_box_gt(
    gt_position: torch.Tensor, gt_mask: torch.Tensor
) -> tuple[torch.Tensor, WorkspaceFilterStats]:
    """``gt_position`` (..., N_max, 3), ``gt_mask`` (..., N_max) bool ->
    ``(filtered_mask, stats)`` -- ``filtered_mask`` is ``gt_mask`` with any
    row whose position lies outside ``[POS_MIN, POS_MAX]`` additionally
    cleared. See the module-level comment above for why."""
    in_box = in_workspace_box(gt_position)
    filtered_mask = gt_mask & in_box
    n_eligible = int(gt_mask.sum().item())
    n_out_of_box = int((gt_mask & ~in_box).sum().item())
    return filtered_mask, WorkspaceFilterStats(n_eligible=n_eligible, n_out_of_box=n_out_of_box)


# --- Hungarian matching -------------------------------------------------------


@dataclass
class MatchCostWeights:
    w_pos: float = 1.0
    w_cls: float = 1.0
    w_size: float = 1.0


def hungarian_match(
    pred_position: torch.Tensor,
    pred_class_logits: torch.Tensor,
    pred_size: torch.Tensor,
    gt_position: torch.Tensor,
    gt_class: torch.Tensor,
    gt_size: torch.Tensor,
    gt_mask: torch.Tensor,
    cost_weights: MatchCostWeights = MatchCostWeights(),
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-sample Hungarian assignment, restricted to existence-eligible
    (``gt_mask``) GT rows.

    All tensors are batched: ``pred_position``/``pred_size`` ``(B, Q, 3)``,
    ``pred_class_logits`` ``(B, Q, C)``, ``gt_position``/``gt_size``
    ``(B, N_max, 3)``, ``gt_class`` ``(B, N_max)`` int64, ``gt_mask``
    ``(B, N_max)`` bool.

    Returns a list of length ``B``, each element ``(query_idx, gt_idx)`` --
    parallel int arrays of matched query/GT indices (``gt_idx`` indexes into
    the *original* ``N_max`` axis, not the masked-down subset); length 0 for
    a sample with no real GT objects.
    """
    batch_size = pred_position.shape[0]
    matches: list[tuple[np.ndarray, np.ndarray]] = []

    for b in range(batch_size):
        real_idx = torch.nonzero(gt_mask[b], as_tuple=True)[0]
        n_real = real_idx.shape[0]
        if n_real == 0:
            matches.append((np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
            continue

        pos_pred_b = pred_position[b]  # (Q, 3)
        pos_gt_b = gt_position[b, real_idx]  # (n_real, 3)
        size_pred_b = pred_size[b]
        size_gt_b = gt_size[b, real_idx]
        class_gt_b = gt_class[b, real_idx]  # (n_real,)

        pos_cost = torch.cdist(pos_pred_b, pos_gt_b, p=2)  # (Q, n_real)
        size_cost = torch.cdist(size_pred_b, size_gt_b, p=2)
        log_probs = F.log_softmax(pred_class_logits[b], dim=-1)  # (Q, C)
        cls_cost = -log_probs[:, class_gt_b]  # (Q, n_real)

        cost = cost_weights.w_pos * pos_cost + cost_weights.w_cls * cls_cost + cost_weights.w_size * size_cost
        cost_np = cost.detach().to(torch.float64).cpu().numpy()
        row_idx, col_idx_local = linear_sum_assignment(cost_np)
        gt_idx = real_idx[col_idx_local].cpu().numpy()
        matches.append((row_idx, gt_idx))

    return matches


# --- full set-prediction loss -------------------------------------------------


@dataclass
class LossWeights:
    existence: float = 1.0
    position: float = 1.0
    size: float = 1.0
    shape: float = 1.0
    cls: float = 1.0
    rotation: float = 1.0


def compute_perception_losses(
    pred: dict[str, torch.Tensor],
    states: torch.Tensor,
    mask: torch.Tensor,
    class_ids: torch.Tensor,
    cost_weights: MatchCostWeights = MatchCostWeights(),
    loss_weights: LossWeights = LossWeights(),
    pos_scale: float = 1.0,
    size_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float], list[tuple[np.ndarray, np.ndarray]]]:
    """Hungarian-match ``pred`` (a :class:`~gltfworld.models.perception.
    PerceptionDETR` forward output) against the tensor-contract GT
    (``states (B, N_max, 22)``, ``mask (B, N_max)``, ``class_ids
    (B, N_max)``), then compute the full weighted set-prediction loss.

    Returns ``(total_loss, components, matches)`` -- ``components`` are
    detached floats for logging (including ``workspace_filtered_fraction``,
    see :func:`filter_out_of_box_gt`), ``matches`` is :func:`hungarian_match`'s
    per-sample ``(query_idx, gt_idx)`` output (reused by eval so matching
    logic never has two implementations).
    """
    device = states.device
    dtype = pred["position"].dtype
    batch_size, n_queries = pred["existence_logit"].shape

    gt_position = states[..., 0:3]
    gt_quat = states[..., 3:7]
    gt_shape_idx = states[..., 13:16].argmax(dim=-1)
    gt_size = states[..., 16:19]

    # V6.2 fix: exclude GT objects that escaped the representable workspace
    # box before matching -- see filter_out_of_box_gt's module-level comment.
    # An out-of-box GT row is now treated as absent for that frame: no query
    # can be Hungarian-matched to it, so its existence target stays 0 too.
    mask, workspace_stats = filter_out_of_box_gt(gt_position, mask)

    matches = hungarian_match(
        pred["position"], pred["class_logits"], pred["size"], gt_position, class_ids, gt_size, mask, cost_weights
    )

    existence_target = torch.zeros(batch_size, n_queries, dtype=dtype, device=device)

    pos_terms, size_terms, shape_terms, cls_terms, rot_terms = [], [], [], [], []

    for b, (query_idx, gt_idx) in enumerate(matches):
        if query_idx.size == 0:
            continue
        query_idx_t = torch.as_tensor(query_idx, dtype=torch.long, device=device)
        gt_idx_t = torch.as_tensor(gt_idx, dtype=torch.long, device=device)
        existence_target[b, query_idx_t] = 1.0

        pred_pos = pred["position"][b, query_idx_t]
        gt_pos = gt_position[b, gt_idx_t]
        pos_terms.append((((pred_pos - gt_pos) / pos_scale) ** 2).sum(-1))

        pred_size = pred["size"][b, query_idx_t]
        gt_sz = gt_size[b, gt_idx_t]
        size_terms.append((((pred_size - gt_sz) / size_scale) ** 2).sum(-1))

        pred_shape_logits = pred["shape_logits"][b, query_idx_t]
        gt_shape = gt_shape_idx[b, gt_idx_t]
        shape_terms.append(F.cross_entropy(pred_shape_logits, gt_shape, reduction="none"))

        pred_class_logits = pred["class_logits"][b, query_idx_t]
        gt_cls = class_ids[b, gt_idx_t]
        cls_terms.append(F.cross_entropy(pred_class_logits, gt_cls, reduction="none"))

        pred_quat = pred["quat"][b, query_idx_t]
        gt_q = gt_quat[b, gt_idx_t]
        rot_terms.append(symmetry_rotation_loss(pred_quat, gt_q, gt_shape))

    def _mean_or_zero(terms: list[torch.Tensor]) -> torch.Tensor:
        if not terms:
            return torch.zeros((), dtype=dtype, device=device)
        return torch.cat(terms).mean()

    pos_loss = _mean_or_zero(pos_terms)
    size_loss = _mean_or_zero(size_terms)
    shape_loss = _mean_or_zero(shape_terms)
    cls_loss = _mean_or_zero(cls_terms)
    rot_loss = _mean_or_zero(rot_terms)
    existence_loss = F.binary_cross_entropy_with_logits(pred["existence_logit"], existence_target)

    total = (
        loss_weights.existence * existence_loss
        + loss_weights.position * pos_loss
        + loss_weights.size * size_loss
        + loss_weights.shape * shape_loss
        + loss_weights.cls * cls_loss
        + loss_weights.rotation * rot_loss
    )

    components = {
        "existence": float(existence_loss.detach()),
        "position": float(pos_loss.detach()),
        "size": float(size_loss.detach()),
        "shape": float(shape_loss.detach()),
        "cls": float(cls_loss.detach()),
        "rotation": float(rot_loss.detach()),
        "workspace_filtered_fraction": workspace_stats.fraction_out_of_box,
    }
    return total, components, matches
