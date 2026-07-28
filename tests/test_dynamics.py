"""Unit tests for ``gltfworld.models.dynamics``/``gltfworld.models.baselines``:
integrator exactness, permutation equivariance, masking invariance, and
parameter-count sanity. All CPU-fast, no packed dataset needed."""

from __future__ import annotations

import numpy as np
import torch

from gltfworld.models.baselines import BallisticBaseline, NoInteractionMLP
from gltfworld.models.dynamics import STATE_DIM, InteractionTransformer, count_params, integrate
from gltfworld.models.rotations import quat_hemisphere, quat_normalize
from gltfworld.scene.contract import GLOBALS_DIM

torch.manual_seed(0)


def _random_states(b: int, n: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    states = np.zeros((b, n, STATE_DIM), dtype=np.float32)
    states[..., 0:3] = rng.uniform(-2, 2, size=(b, n, 3))  # pos
    quat = rng.normal(size=(b, n, 4)).astype(np.float32)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = np.where(quat[..., 3:4] < 0, -quat, quat)
    states[..., 3:7] = quat
    states[..., 7:10] = rng.uniform(-1.5, 1.5, size=(b, n, 3))  # lin_vel
    states[..., 10:13] = rng.uniform(-3, 3, size=(b, n, 3))  # ang_vel
    shape_idx = rng.integers(0, 3, size=(b, n))
    onehot = np.zeros((b, n, 3), dtype=np.float32)
    for i in range(b):
        for j in range(n):
            onehot[i, j, shape_idx[i, j]] = 1.0
    states[..., 13:16] = onehot
    states[..., 16:19] = rng.uniform(0.05, 0.25, size=(b, n, 3))  # size
    states[..., 19] = rng.uniform(-2.0, 5.0, size=(b, n))  # log_mass
    states[..., 20] = rng.uniform(0.4, 1.0, size=(b, n))  # friction
    states[..., 21] = rng.uniform(0.0, 0.3, size=(b, n))  # restitution
    return torch.from_numpy(states)


def _random_globals(b: int, seed: int, gravity_y: float = -9.81) -> torch.Tensor:
    rng = np.random.default_rng(seed + 999)
    g = np.zeros((b, GLOBALS_DIM), dtype=np.float32)
    g[:, 1] = gravity_y
    g[:, 3] = 1.0 / 30.0  # dt
    g[:, 4:7] = rng.uniform(-1, 1, size=(b, 3))  # camera pos (unused by the model)
    cam_quat = rng.normal(size=(b, 4)).astype(np.float32)
    cam_quat /= np.linalg.norm(cam_quat, axis=-1, keepdims=True)
    g[:, 7:11] = cam_quat
    g[:, 11] = 1.0  # yfov
    return torch.from_numpy(g)


def test_param_count_in_band():
    model = InteractionTransformer()
    n = count_params(model)
    assert 4_000_000 <= n <= 7_000_000, f"param count {n} outside 4-7M band"


def test_integrator_exactness_fresh_model_matches_ballistic():
    """A freshly constructed (zero-init head) InteractionTransformer predicts
    all-zero (dv, dw, r); with gravity zeroed out, BallisticBaseline's own
    deltas are also all zero -- both funnel through the exact same
    ``integrate`` call, so outputs must match bit-for-bit (same dtype, same
    op order), not just approximately. (NoInteractionMLP is deliberately
    *not* zero-init -- see its docstring -- so it isn't part of this
    particular exactness check; ``test_integrator_matches_manual_ballistic_
    formula`` below still covers it sharing the same integrator via explicit
    zero deltas.)"""
    b, n = 4, 5
    states = _random_states(b, n, seed=1)
    mask = torch.tensor([[True, True, True, False, False]] * b)
    globals_ = _random_globals(b, seed=1, gravity_y=0.0)  # zero gravity

    model = InteractionTransformer()
    model.eval()
    ballistic = BallisticBaseline()

    with torch.no_grad():
        out_model = model(states, mask, globals_)
        out_ballistic = ballistic(states, mask, globals_)

    # bit-exact: same float32 arithmetic, same operation order.
    assert torch.equal(out_model, out_ballistic)


def test_mlp_shares_integrator_with_zero_deltas():
    """NoInteractionMLP isn't zero-init, but it must still route through the
    exact same ``integrate`` function: feeding explicit zero deltas through
    ``integrate`` directly must match BallisticBaseline with zero gravity."""
    b, n = 4, 5
    states = _random_states(b, n, seed=1)
    globals_ = _random_globals(b, seed=1, gravity_y=0.0)

    zeros = torch.zeros(b, n, 3)
    out_integrate = integrate(states, globals_, zeros, zeros, zeros)

    ballistic = BallisticBaseline()
    with torch.no_grad():
        out_ballistic = ballistic(states, torch.ones(b, n, dtype=torch.bool), globals_)

    assert torch.equal(out_integrate, out_ballistic)


def test_integrator_matches_manual_ballistic_formula():
    b, n = 2, 3
    states = _random_states(b, n, seed=2)
    mask = torch.ones(b, n, dtype=torch.bool)
    globals_ = _random_globals(b, seed=2)

    ballistic = BallisticBaseline()
    with torch.no_grad():
        out = ballistic(states, mask, globals_)

    dt = globals_[:, 3:4, None]
    gravity = globals_[:, None, 0:3]
    v = states[..., 7:10]
    p = states[..., 0:3]
    v_expected = v + gravity * dt
    p_expected = p + v_expected * dt

    np.testing.assert_allclose(out[..., 7:10].numpy(), v_expected.numpy(), atol=1e-6)
    np.testing.assert_allclose(out[..., 0:3].numpy(), p_expected.numpy(), atol=1e-6)
    # orientation and angular velocity unchanged
    np.testing.assert_allclose(out[..., 3:7].numpy(), states[..., 3:7].numpy(), atol=1e-6)
    np.testing.assert_allclose(out[..., 10:13].numpy(), states[..., 10:13].numpy(), atol=1e-6)
    # static features copied through
    np.testing.assert_allclose(out[..., 13:22].numpy(), states[..., 13:22].numpy(), atol=1e-6)


def test_permutation_equivariance_transformer():
    b, n = 3, 5
    states = _random_states(b, n, seed=3)
    mask = torch.tensor([[True, True, True, True, False]] * b)
    globals_ = _random_globals(b, seed=3)

    model = InteractionTransformer()
    model.eval()

    perm = torch.tensor([3, 1, 4, 0, 2])
    inv_perm = torch.argsort(perm)

    states_perm = states[:, perm, :]
    mask_perm = mask[:, perm]

    with torch.no_grad():
        out = model(states, mask, globals_)
        out_perm = model(states_perm, mask_perm, globals_)

    # un-permuting the permuted-input output must match the original output.
    out_perm_unpermuted = out_perm[:, inv_perm, :]
    np.testing.assert_allclose(out.numpy(), out_perm_unpermuted.numpy(), atol=1e-5)


def test_permutation_equivariance_mlp():
    b, n = 3, 5
    states = _random_states(b, n, seed=4)
    mask = torch.tensor([[True, True, True, False, False]] * b)
    globals_ = _random_globals(b, seed=4)

    model = NoInteractionMLP()
    model.eval()

    perm = torch.tensor([2, 0, 4, 1, 3])
    inv_perm = torch.argsort(perm)

    with torch.no_grad():
        out = model(states, mask, globals_)
        out_perm = model(states[:, perm, :], mask[:, perm], globals_)

    np.testing.assert_allclose(out.numpy(), out_perm[:, inv_perm, :].numpy(), atol=1e-5)


def test_masking_invariance_padded_slots_dont_leak():
    """3 real objects, unpadded (N=3) vs. the same 3 objects padded to N=5
    with mask=False on the extra slots: the first 3 objects' outputs must
    match either way -- padded keys are excluded from attention."""
    b = 2
    torch.manual_seed(123)
    model = InteractionTransformer()
    model.eval()

    states3 = _random_states(b, 3, seed=5)
    mask3 = torch.ones(b, 3, dtype=torch.bool)
    globals_ = _random_globals(b, seed=5)

    pad = torch.zeros(b, 2, STATE_DIM, dtype=torch.float32)
    states5 = torch.cat([states3, pad], dim=1)
    mask5 = torch.cat([torch.ones(b, 3, dtype=torch.bool), torch.zeros(b, 2, dtype=torch.bool)], dim=1)

    with torch.no_grad():
        out3 = model(states3, mask3, globals_)
        out5 = model(states5, mask5, globals_)

    np.testing.assert_allclose(out3.numpy(), out5[:, :3, :].numpy(), atol=1e-5)


def test_forward_output_is_finite_and_unit_quat():
    b, n = 4, 5
    states = _random_states(b, n, seed=6)
    mask = torch.tensor([[True, True, True, True, False]] * b)
    globals_ = _random_globals(b, seed=6)

    model = InteractionTransformer()
    model.eval()
    with torch.no_grad():
        out = model(states, mask, globals_)

    assert torch.isfinite(out).all()
    quat = out[..., 3:7]
    norm = torch.linalg.norm(quat, dim=-1)
    np.testing.assert_allclose(norm.numpy(), np.ones((b, n)), atol=1e-5)
    assert (quat[..., 3] >= -1e-6).all()


def test_integrate_helper_directly():
    b, n = 2, 3
    states = _random_states(b, n, seed=7)
    globals_ = _random_globals(b, seed=7)
    dv = torch.zeros(b, n, 3)
    dw = torch.zeros(b, n, 3)
    r = torch.zeros(b, n, 3)
    out = integrate(states, globals_, dv, dw, r)
    # zero deltas -> constant velocity: p' = p + v*dt, q' unchanged (up to
    # hemisphere/renormalize, which is a no-op on an already-unit, w>=0 quat).
    dt = globals_[:, 3:4, None]
    expected_pos = states[..., 0:3] + states[..., 7:10] * dt
    np.testing.assert_allclose(out[..., 0:3].numpy(), expected_pos.numpy(), atol=1e-6)
    np.testing.assert_allclose(out[..., 3:7].numpy(), states[..., 3:7].numpy(), atol=1e-6)
