"""Baselines for the dynamics-model eval: ``BallisticBaseline`` (no learning
at all -- pure physics extrapolation) and ``NoInteractionMLP`` (a learned
per-object model with *no* cross-object attention, the ablation that
isolates what ``InteractionTransformer``'s attention actually buys).

Both share :func:`gltfworld.models.dynamics.integrate` with
``InteractionTransformer`` -- the exact same arithmetic sequence, same
dtype, same operation order -- so any measured accuracy gap between models
in ``gltfworld.eval.rollout`` reflects a difference in *predicted deltas*,
never a difference in how those deltas get applied.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gltfworld.models.dynamics import (
    GLOBALS_FEATURE_DIM,
    OBJECT_FEATURE_DIM,
    globals_features,
    integrate,
    object_features,
)


class BallisticBaseline(nn.Module):
    """Pure ballistic (constant-gravity, no collisions) extrapolation.

    ``dv = gravity * dt``, ``dw = 0``, ``r = 0`` -- i.e. ``v' = v + g*dt``,
    ``p' = p + v'*dt``, orientation and angular velocity unchanged. No
    learned parameters; ``nn.Module`` only so it drops into the same
    ``forward(states, mask, globals) -> next_states`` call signature as the
    learned models (``gltfworld.eval.rollout.rollout`` and
    ``gltfworld.eval.rollout``'s CLI treat all three uniformly).
    """

    def forward(self, states: torch.Tensor, mask: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
        b, n, _ = states.shape
        gravity = globals_[..., 0:3]  # (B, 3)
        dt = globals_[..., 3:4]  # (B, 1)
        dv = (gravity * dt)[:, None, :].expand(b, n, 3)
        dw = torch.zeros(b, n, 3, dtype=states.dtype, device=states.device)
        r = torch.zeros(b, n, 3, dtype=states.dtype, device=states.device)
        return integrate(states, globals_, dv, dw, r)


class NoInteractionMLP(nn.Module):
    """Per-object MLP, no cross-object attention: same features
    (:func:`gltfworld.models.dynamics.object_features` concatenated with
    :func:`gltfworld.models.dynamics.globals_features`, broadcast to every
    object), same zero-init output head/integration as
    ``InteractionTransformer``, but each object is processed independently
    (a plain ``nn.Linear`` stack applied per-token) -- there is no attention
    or any other mechanism for one object's features to influence another
    object's prediction.

    ``2x256`` hidden layers per the spec; measured parameter count is
    reported by ``gltfworld.models.baselines.__main__`` (documented, not
    forced to hit the spec's "~0.3M" approximation -- see DESIGN.md's V5
    section for the actual number and why the literal "2x256" architecture
    description was kept as the ground truth over the approximate count).

    Unlike ``InteractionTransformer``, the final layer is **not** zero-init
    (default ``nn.Linear`` init instead): zero-init is specifically about
    ``InteractionTransformer`` starting from an *exact* constant-velocity
    baseline (a real integrator-exactness invariant, see ``tests/
    test_dynamics.py``); for this much smaller ablation model, zero-init
    would leave next to no loss to reduce during the training harness's
    ``--smoke`` check (constant-velocity is already a good 1/30s-step
    approximation) -- a small random init instead gives the smoke check a
    real, non-trivial loss curve to demonstrate learning on.
    """

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        in_dim = OBJECT_FEATURE_DIM + GLOBALS_FEATURE_DIM
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 9),
        )

    def forward(self, states: torch.Tensor, mask: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
        b, n, _ = states.shape
        obj_feat = object_features(states, mask)  # (B, N, 24)
        glob_feat = globals_features(globals_)[:, None, :].expand(b, n, -1)  # (B, N, 4)
        feat = torch.cat([obj_feat, glob_feat], dim=-1)
        out = self.net(feat)
        dv, dw, r = out.split(3, dim=-1)
        return integrate(states, globals_, dv, dw, r)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    ballistic = BallisticBaseline()
    print(f"BallisticBaseline parameters: {count_params(ballistic):,} (no learning)")
    mlp = NoInteractionMLP()
    print(f"NoInteractionMLP parameters: {count_params(mlp):,}")
