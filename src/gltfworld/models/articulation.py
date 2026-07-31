"""``ArticulationEstimator``: single RGB frame -> the moving joint's state
(V9, "articulation stage" -- joint-*state* estimation, not full object
detection; see module docstring's scope note below and DESIGN.md's V9
section).

Scope, stated plainly: this milestone's perception task is **joint-state
estimation** -- given a single rendered frame of a ``wm-articulated-v1``
door/drawer scene (``gltfworld.datagen.articulated``), estimate:

- the joint's generalized position (``joint_pos``, normalized by the
  joint's own limit range -- see below),
- the joint *type* (revolute/hinge vs. prismatic/slider), and
- the joint *axis* (a 3D unit vector, one of the world X/Y/Z basis vectors
  in this dataset's convention).

This is deliberately **not** ``PerceptionDETR``-style set-of-objects
detection (no Hungarian matching, no object queries, no existence head) --
every ``wm-articulated-v1`` episode has exactly one articulated joint, so
the whole per-frame task collapses to one small regression/classification
head on top of an image encoder, not a set-prediction problem. Articulated
*dynamics* (predicting how the joint state evolves over time, the V9
counterpart of ``gltfworld.models.dynamics.InteractionTransformer``) is
explicitly **out of scope** for this milestone -- future work (see
DESIGN.md's V9 section).

Architecture
------------

**Encoder: reused, not duplicated.** The image trunk is
``gltfworld.models.perception._CNNEncoder`` (the V6.3 small-data-regime CNN
encoder -- stride-1 stem + 4 stride-2 stages -> ``16x16xD_MODEL`` feature
map -> a ``(B, 256, d_model)`` token sequence + learned positional
embedding), imported and instantiated directly, not reimplemented. Same
rationale as V6.3's own (see ``gltfworld.models.perception``'s module
docstring and DESIGN.md's V6.3 section): a from-scratch ViT-style
transformer encoder has no spatial inductive bias and needs far more data
than this project's dataset scale (V9's ``articulated-v1`` is a 1,500-episode
dataset, smaller than ``perception-v1``'s already-data-hungry regime) to
learn one; a CNN's built-in locality/translation-equivariance bias is the
standard fix, and this task's per-frame signal (a single joint's swing
angle/slide distance, plus which of 2 shapes and which of 3 fixed axes) is
simpler than full multi-object pose regression, so the *same* proven trunk
is reused wholesale rather than inventing a second encoder.

**Pooling + heads**: unlike ``PerceptionDETR`` (which decodes a *set* of
``N_MAX`` object queries via cross-attention), this is a single-target
per-frame task, so the encoder's 256 image tokens are simply mean-pooled to
one ``(B, d_model)`` vector, passed through a small shared 2-layer MLP
trunk, then split into three small heads:

- ``joint_pos_head``: ``Linear(d_model, 1)`` -- see "Joint position
  normalization" below.
- ``type_head``: ``Linear(d_model, 2)`` -- revolute(0)/prismatic(1) logits.
- ``axis_head``: ``Linear(d_model, 3)`` -- raw 3-vector, L2-normalized in
  :meth:`ArticulationEstimator.forward` to a unit vector (see "Axis
  regression" below).

**Joint position normalization.** The raw regression target is
``(joint_pos - limit_min) / (limit_max - limit_min)`` -- not raw radians/
meters -- for the same reason ``InteractionTransformer``'s ``object_features``
picks fixed unit-ish scale constants (DESIGN.md's V5 section): a revolute
joint's raw range is up to ~1.9 rad while a prismatic joint's is up to
~0.35 m, an order-of-magnitude unit mismatch that would otherwise make one
term dominate a shared regression loss purely from unit choice, not
difficulty. Normalizing by each episode's own ``[limit_min, limit_max]``
range (``ArticulatedSpec.min``/``.max``, known dataset metadata -- not
predicted by the model, see the scope note above: the model is never asked
to *guess* the cabinet's own travel limits from a single frame, only the
current joint position within them) puts both joint types' targets on a
common ``~[0, 1]`` scale (transient overshoot past a soft limit stop, per
DESIGN.md's V9-prep report, can push this slightly outside ``[0, 1]`` in
either direction -- the head is a plain linear layer, not sigmoid-squashed,
so it can represent that). :func:`denormalize_joint_pos` converts a
normalized prediction back to raw units for reporting (see
``gltfworld.eval.articulation_eval``).

**Axis regression: directed, not sign-invariant.** A hinge/slide axis line
has two opposite unit-vector representations that describe the *same*
physical DOF in isolation (rotating by ``+theta`` about ``+axis`` is the
same rotation as ``-theta`` about ``-axis``). A naive "the axis is just a
line, so score it sign-invariantly" reading would suggest a symmetric loss
like ``1 - |cos(angle)|``. That would be **wrong for this dataset**:
``wm-articulated-v1``'s sampler (``gltfworld.datagen.articulated
.sample_articulated_scene``) always emits ``axis`` as one of the *positive*
world basis vectors (``_unit_axis(axis)``, never a mirrored negative one),
and pairs it with a ``joint_pos`` that is always non-negative (``min=0`` in
every sampled joint) increasing *specifically* in that ``+axis`` direction.
The axis and the sign convention of ``joint_pos`` are not independent
choices here -- together they fix one specific physical opening direction.
If the model predicted ``-axis`` while still predicting a positive
``joint_pos_norm``, reconstructing the joint's pose (see
``gltfworld.eval.articulation_eval``'s re-render check, which does exactly
this reconstruction) would swing/slide the part in the *opposite*, wrong
direction -- a real error, not a benign relabeling. So the axis loss here is
a plain **directed** cosine loss, ``1 - cos(angle) = 1 - pred . gt`` (both
unit vectors), which is minimized only when the predicted direction matches
the training convention's actual positive axis, and penalizes a
sign-flipped prediction at its worst value (loss 2, ``cos = -1``) rather
than scoring it as free. A sign-invariant loss would only be the right
choice if the dataset itself mixed both sign conventions for the same
physical joint (it doesn't).

Total parameter count is printed by ``python -m gltfworld.models.articulation``
(no fixed target band asserted here, unlike V5/V6's -- this milestone's spec
text gives no approximate parameter count to reconcile against, just "small
head(s)" on top of the reused trunk).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gltfworld.models.perception import D_MODEL, _CNNEncoder

JOINT_TYPE_NAMES = ("revolute", "prismatic")


def denormalize_joint_pos(joint_pos_norm: torch.Tensor, limit_min: torch.Tensor, limit_max: torch.Tensor) -> torch.Tensor:
    """Inverse of ``ArticulationDataset``'s ``(joint_pos - limit_min) /
    (limit_max - limit_min)`` normalization -- back to raw radians/meters."""
    return joint_pos_norm * (limit_max - limit_min) + limit_min


class ArticulationEstimator(nn.Module):
    """Single RGB frame -> ``(joint_pos_norm, type_logits, axis)``. See
    module docstring for the full architecture writeup."""

    def __init__(self, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.d_model = d_model
        # Reused wholesale, not duplicated -- see module docstring.
        self.encoder = _CNNEncoder(d_model)

        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.joint_pos_head = nn.Linear(d_model, 1)
        self.type_head = nn.Linear(d_model, len(JOINT_TYPE_NAMES))
        self.axis_head = nn.Linear(d_model, 3)

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        """``rgb``: ``(B, H, W, 3)`` float32 in ``[0, 1]`` (the layout
        ``gltfworld.data.dataset.ArticulationDataset`` returns)."""
        if rgb.shape[-1] != 3:
            raise ValueError(f"expected rgb (..., H, W, 3), got {tuple(rgb.shape)}")

        x = rgb.permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)
        x = x * 2.0 - 1.0  # [0, 1] -> [-1, 1], same convention as PerceptionDETR

        tokens = self.encoder(x)  # (B, num_patches, d_model)
        pooled = tokens.mean(dim=1)  # (B, d_model)
        h = self.trunk(pooled)

        joint_pos_norm = self.joint_pos_head(h).squeeze(-1)  # (B,)
        type_logits = self.type_head(h)  # (B, 2)
        axis_raw = self.axis_head(h)  # (B, 3)
        axis = F.normalize(axis_raw, dim=-1, eps=1e-8)

        return {
            "joint_pos_norm": joint_pos_norm,
            "type_logits": type_logits,
            "axis": axis,
        }


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LossWeights:
    """Plain container (not a dataclass -- kept trivially JSON-round-trippable
    by ``train_articulation.Config``) for the three loss terms' weights."""

    def __init__(self, joint_pos: float = 1.0, type_: float = 1.0, axis: float = 1.0) -> None:
        self.joint_pos = joint_pos
        self.type = type_
        self.axis = axis


def compute_articulation_losses(
    pred: dict[str, torch.Tensor],
    joint_pos_norm_gt: torch.Tensor,
    joint_type_id_gt: torch.Tensor,
    axis_gt: torch.Tensor,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    """``pred`` is :meth:`ArticulationEstimator.forward`'s output dict.
    Returns ``(total_loss, component_dict)`` -- ``component_dict`` values are
    plain Python floats (already detached), for logging.

    - ``joint_pos``: plain MSE on the normalized scalar.
    - ``type``: cross-entropy over the 2 joint-type classes.
    - ``axis``: directed cosine loss ``mean(1 - pred_axis . gt_axis)`` -- see
      module docstring's "Axis regression: directed, not sign-invariant" for
      why this is *not* symmetrized.
    """
    loss_joint_pos = F.mse_loss(pred["joint_pos_norm"], joint_pos_norm_gt)
    loss_type = F.cross_entropy(pred["type_logits"], joint_type_id_gt)
    cos_sim = (pred["axis"] * axis_gt).sum(dim=-1)
    loss_axis = (1.0 - cos_sim).mean()

    total = weights.joint_pos * loss_joint_pos + weights.type * loss_type + weights.axis * loss_axis

    components = {
        "joint_pos": float(loss_joint_pos.detach()),
        "type": float(loss_type.detach()),
        "axis": float(loss_axis.detach()),
    }
    return total, components


if __name__ == "__main__":
    model = ArticulationEstimator()
    n_params = count_params(model)
    print(f"ArticulationEstimator parameters: {n_params:,}")
