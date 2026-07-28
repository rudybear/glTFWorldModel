"""``InteractionTransformer``: the state[t] -> state[t+1] dynamics model.

Consumes/produces the ``gltfworld.scene.contract`` tensor layout directly
(``states (B, N, 22)``, ``mask (B, N)``, ``globals (B, 12)``) so it drops
straight into ``gltfworld.data.dataset.DynamicsDataset`` and
``gltfworld.eval.rollout`` with no extra glue.

Architecture (see DESIGN.md's V5 section for the full writeup)
----------------------------------------------------------------

- One token per object, embedding a hand-picked, roughly unit-scaled feature
  vector (:func:`object_features`) -- not the raw 22-dim state directly,
  since raw position/velocity/mass are on wildly different physical scales
  and the quaternion's double-cover discontinuity is a poor learning target
  as an *input* (it's fine as an *output*, via the axis-angle exponential
  map, see below).
- One globals token (:func:`globals_features`: normalized gravity + dt;
  camera deliberately excluded -- irrelevant to physics).
- One learned "ground" token (a plain ``nn.Parameter``, not derived from any
  input feature): ``wm-scenes-v1``'s ground plate is geometrically identical
  across every episode (DESIGN.md), so there is no *per-episode* ground
  signal to encode -- this token instead gives every object token a fixed
  attention partner to learn ground-relative dynamics (support, contact,
  friction) against, the same way a `[CLS]`-style token works.
- 6 pre-norm ``nn.TransformerEncoder`` layers, ``d_model=256``, 8 heads,
  MLP ratio 4, key-padding mask built from ``mask`` (padded object slots are
  masked out of attention as *keys*, so they can never influence a real
  object's output -- see ``tests/test_dynamics.py::test_masking_invariance``).
- No positional encoding of any kind is added across the object axis, so the
  whole stack is permutation-equivariant in the object token order (only the
  globals/ground tokens, which are structurally distinguished by always
  occupying the last two token slots, are asymmetric) -- see
  ``tests/test_dynamics.py::test_permutation_equivariance``.
- Output head (shared, zero-init final layer): per real-object token,
  predicts ``(dv, dw, r)`` -- linear velocity delta, angular velocity delta,
  and a rotation-update rotation-vector (axis-angle), 3 each. Zero-init means
  a freshly constructed model outputs all-zero deltas, so :func:`integrate`
  reduces to *exact* constant-velocity extrapolation until the model has
  learned anything -- see ``tests/test_dynamics.py::test_integrator_exactness``.

Integration (:func:`integrate`) is a single, shared, semi-implicit Euler
step used by *every* model in this milestone (``InteractionTransformer``,
and ``gltfworld.models.baselines.BallisticBaseline``/``NoInteractionMLP``,
which import it directly) so a baseline-vs-model comparison is never
comparing different arithmetic, only different ``(dv, dw, r)`` predictions:

    v' = v + dv
    p' = p + v' * dt          (uses the *updated* velocity -- semi-implicit)
    w' = w + dw
    q' = normalize(hemisphere(exp(r) (x) q))

Static per-object features (shape one-hot, size, log-mass, friction,
restitution) are copied through unchanged (they don't evolve in time).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gltfworld.scene.contract import STATE_DIM
from gltfworld.models.rotations import (
    axis_angle_to_quat,
    quat_hemisphere,
    quat_multiply,
    quat_normalize,
    quat_to_6d,
)

D_MODEL = 256
N_LAYERS = 6
N_HEADS = 8
MLP_RATIO = 4

# --- feature normalization constants (fixed, not data-fit -- see module note) -
# Chosen to roughly range-normalize wm-scenes-v1's sampled distributions
# (DESIGN.md): pos in a few meters, |v|<=1.5 m/s but falls build up more,
# |w|<=3 rad/s (contacts spike higher), size in [0.05, 0.25] m radius/half-
# extent, log(mass) roughly in [-2, 5.3] for density in [300, 3000] kg/m^3
# and size in [0.05, 0.25] m.
POS_SCALE = 2.0
VEL_SCALE = 3.0
ANGVEL_SCALE = 6.0
SIZE_SCALE = 0.25
LOG_MASS_SCALE = 3.0
GRAVITY_SCALE = 9.81
DT_SCALE = 30.0  # dt * DT_SCALE, not dt / DT_SCALE -- dt ~ 1/30s

OBJECT_FEATURE_DIM = 3 + 3 + 3 + 6 + 3 + 3 + 1 + 1 + 1  # 24
GLOBALS_FEATURE_DIM = 3 + 1  # 4

_IDENTITY_QUAT = torch.tensor([0.0, 0.0, 0.0, 1.0])


def object_features(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``states (..., N, 22)``, ``mask (..., N)`` bool -> ``(..., N, 24)``.

    Masked-out (padded) rows get their quaternion swapped for the identity
    before the 6D conversion (a raw all-zero padding quaternion has zero
    norm, which would divide-by-zero in ``quat_to_6d``); the resulting
    feature row is nonsense but harmless, since padded rows are excluded
    from attention (key-padding mask) and never contribute to a real
    object's output.
    """
    pos = states[..., 0:3]
    quat = states[..., 3:7]
    vel = states[..., 7:10]
    ang_vel = states[..., 10:13]
    shape_onehot = states[..., 13:16]
    size = states[..., 16:19]
    log_mass = states[..., 19:20]
    friction = states[..., 20:21]
    restitution = states[..., 21:22]

    identity = _IDENTITY_QUAT.to(dtype=quat.dtype, device=quat.device)
    quat_safe = torch.where(mask[..., None], quat, identity.expand_as(quat))
    rot6d = quat_to_6d(quat_safe)

    return torch.cat(
        [
            pos / POS_SCALE,
            vel / VEL_SCALE,
            ang_vel / ANGVEL_SCALE,
            rot6d,
            shape_onehot,
            size / SIZE_SCALE,
            log_mass / LOG_MASS_SCALE,
            friction,
            restitution,
        ],
        dim=-1,
    )


def globals_features(globals_: torch.Tensor) -> torch.Tensor:
    """``globals_ (..., 12)`` -> ``(..., 4)``: normalized gravity + dt.

    Camera position/rotation/yfov (``globals_[..., 4:12]``) are deliberately
    dropped -- irrelevant to physics, per DESIGN.md.
    """
    gravity = globals_[..., 0:3]
    dt = globals_[..., 3:4]
    return torch.cat([gravity / GRAVITY_SCALE, dt * DT_SCALE], dim=-1)


def integrate(
    states: torch.Tensor,
    globals_: torch.Tensor,
    dv: torch.Tensor,
    dw: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    """Shared semi-implicit-Euler integrator: ``states (B, N, 22)`` + deltas
    ``(B, N, 3)`` each -> ``next_states (B, N, 22)``.

    ``globals_`` supplies ``dt`` (index 3); every other field of ``states``
    that isn't pos/quat/vel/ang_vel (shape/size/mass/friction/restitution)
    is copied through unchanged. See the module docstring for the exact
    update equations.
    """
    pos = states[..., 0:3]
    quat = states[..., 3:7]
    vel = states[..., 7:10]
    ang_vel = states[..., 10:13]
    static = states[..., 13:22]

    dt = globals_[..., 3:4]  # (B, 1)
    dt = dt[..., None, :]  # (B, 1, 1) to broadcast over the object axis

    vel_new = vel + dv
    pos_new = pos + vel_new * dt
    ang_vel_new = ang_vel + dw

    dq = axis_angle_to_quat(r)
    quat_new = quat_hemisphere(quat_normalize(quat_multiply(dq, quat)))

    return torch.cat([pos_new, quat_new, vel_new, ang_vel_new, static], dim=-1)


class _OutputHead(nn.Module):
    """Shared per-token head: ``d_model -> (dv, dw, r)`` (9), zero-init final
    layer so a fresh model predicts all-zero deltas (exact constant velocity
    through :func:`integrate`)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 9),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(x)
        dv, dw, r = out.split(3, dim=-1)
        return dv, dw, r


class InteractionTransformer(nn.Module):
    """State[t] -> state[t+1], with cross-object attention (see module docstring)."""

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_layers: int = N_LAYERS,
        n_heads: int = N_HEADS,
        mlp_ratio: int = MLP_RATIO,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.object_embed = nn.Linear(OBJECT_FEATURE_DIM, d_model)
        self.globals_embed = nn.Linear(GLOBALS_FEATURE_DIM, d_model)
        self.ground_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.ground_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )

        self.head = _OutputHead(d_model)

    def forward(self, states: torch.Tensor, mask: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
        b, n, d = states.shape
        assert d == STATE_DIM, f"expected states last dim {STATE_DIM}, got {d}"

        obj_tok = self.object_embed(object_features(states, mask))  # (B, N, D)
        glob_tok = self.globals_embed(globals_features(globals_))[:, None, :]  # (B, 1, D)
        ground_tok = self.ground_token.expand(b, 1, self.d_model)  # (B, 1, D)

        tokens = torch.cat([obj_tok, glob_tok, ground_tok], dim=1)  # (B, N+2, D)
        extra_valid = torch.ones(b, 2, dtype=torch.bool, device=mask.device)
        valid = torch.cat([mask, extra_valid], dim=1)  # (B, N+2)
        key_padding_mask = ~valid  # True = ignored, per nn.TransformerEncoderLayer's convention

        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        obj_encoded = encoded[:, :n, :]

        dv, dw, r = self.head(obj_encoded)
        return integrate(states, globals_, dv, dw, r)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = InteractionTransformer()
    n_params = count_params(model)
    print(f"InteractionTransformer parameters: {n_params:,}")
    assert 4_000_000 <= n_params <= 7_000_000, f"param count {n_params:,} outside the 4-7M target band"
    print("OK: within 4-7M target band")
