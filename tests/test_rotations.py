"""Cross-validate ``gltfworld.models.rotations`` against
``scipy.spatial.transform.Rotation`` (independent reference implementation,
same xyzw / axis-angle conventions -- no translation needed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from gltfworld.models.rotations import (
    axis_angle_to_quat,
    matrix_to_quat,
    quat_geodesic_angle,
    quat_hemisphere,
    quat_multiply,
    quat_normalize,
    quat_to_6d,
    quat_to_axis_angle,
    quat_to_matrix,
    rotation_6d_to_matrix,
    sixd_to_quat,
)

# Every tensor below is built via torch.from_numpy(<float64 numpy array>),
# which already yields a float64 torch tensor regardless of torch's global
# default dtype -- so this module deliberately does *not* call
# torch.set_default_dtype(...) (that's process-global, mutating state for
# every other test module that imports/runs in the same pytest session).


def _random_quats(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return Rotation.random(n, random_state=rng).as_quat().astype(np.float64)


def _hemisphere(q: np.ndarray) -> np.ndarray:
    sign = np.where(q[..., 3:4] < 0, -1.0, 1.0)
    return q * sign


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_quat_normalize_hemisphere(seed):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(50, 4))
    t = torch.from_numpy(raw)
    out = quat_hemisphere(quat_normalize(t))
    norm = torch.linalg.norm(out, dim=-1)
    np.testing.assert_allclose(norm.numpy(), np.ones(50), atol=1e-10)
    assert (out[..., 3] >= 0).all()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_quat_multiply_matches_scipy(seed):
    q1 = _random_quats(200, seed)
    q2 = _random_quats(200, seed + 100)
    expected = (Rotation.from_quat(q1) * Rotation.from_quat(q2)).as_quat()
    expected = _hemisphere(expected)

    got = quat_multiply(torch.from_numpy(q1), torch.from_numpy(q2))
    got = quat_hemisphere(got).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-8)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_axis_angle_to_quat_matches_scipy(seed):
    rng = np.random.default_rng(seed)
    axes = rng.normal(size=(200, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    angles = rng.uniform(-2.5 * np.pi, 2.5 * np.pi, size=(200, 1))
    rotvec = axes * angles

    expected = _hemisphere(Rotation.from_rotvec(rotvec).as_quat())
    got = quat_hemisphere(axis_angle_to_quat(torch.from_numpy(rotvec))).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-8)


def test_axis_angle_to_quat_small_angle_stable():
    # theta -> 0 must not produce NaN/Inf and must approach identity.
    r = torch.tensor([[0.0, 0.0, 0.0], [1e-9, 0.0, 0.0], [1e-6, -1e-6, 2e-6]])
    q = axis_angle_to_quat(r)
    assert torch.isfinite(q).all()
    np.testing.assert_allclose(q[0].numpy(), [0.0, 0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(torch.linalg.norm(q, dim=-1).numpy(), np.ones(3), atol=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_quat_to_matrix_matches_scipy(seed):
    q = _random_quats(100, seed)
    expected = Rotation.from_quat(q).as_matrix()
    got = quat_to_matrix(torch.from_numpy(q)).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-8)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matrix_to_quat_matches_scipy(seed):
    q = _random_quats(300, seed)
    mats = Rotation.from_quat(q).as_matrix()
    expected = _hemisphere(Rotation.from_matrix(mats).as_quat())
    got = matrix_to_quat(torch.from_numpy(mats)).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_quat_matrix_round_trip(seed):
    q = _random_quats(300, seed)
    q = _hemisphere(q)
    t = torch.from_numpy(q)
    back = matrix_to_quat(quat_to_matrix(t)).numpy()
    np.testing.assert_allclose(back, q, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_6d_round_trip(seed):
    q = _hemisphere(_random_quats(300, seed))
    t = torch.from_numpy(q)
    d6 = quat_to_6d(t)
    assert d6.shape == (300, 6)
    back = sixd_to_quat(d6).numpy()
    np.testing.assert_allclose(back, q, atol=1e-6)


def test_6d_gram_schmidt_orthonormal():
    rng = np.random.default_rng(0)
    d6 = torch.from_numpy(rng.normal(size=(50, 6)))
    m = rotation_6d_to_matrix(d6)
    # columns must be orthonormal, matrix must be a proper rotation (det=1).
    gram = torch.matmul(m.transpose(-1, -2), m)
    eye = torch.eye(3, dtype=gram.dtype).expand_as(gram)
    np.testing.assert_allclose(gram.numpy(), eye.numpy(), atol=1e-8)
    dets = torch.linalg.det(m).numpy()
    np.testing.assert_allclose(dets, np.ones(50), atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_geodesic_angle_matches_scipy(seed):
    q1 = _random_quats(200, seed)
    q2 = _random_quats(200, seed + 1000)
    relative = Rotation.from_quat(q1).inv() * Rotation.from_quat(q2)
    expected = relative.magnitude()  # angle in [0, pi]

    got = quat_geodesic_angle(torch.from_numpy(q1), torch.from_numpy(q2)).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_geodesic_angle_self_is_zero():
    q = torch.from_numpy(_hemisphere(_random_quats(50, 42)))
    angle = quat_geodesic_angle(q, q)
    np.testing.assert_allclose(angle.numpy(), np.zeros(50), atol=1e-10)


def test_geodesic_angle_double_cover_invariant():
    # q and -q represent the same rotation; angle must be identical either way.
    q1 = torch.from_numpy(_hemisphere(_random_quats(50, 7)))
    q2 = torch.from_numpy(_hemisphere(_random_quats(50, 8)))
    angle = quat_geodesic_angle(q1, q2)
    angle_flipped = quat_geodesic_angle(q1, -q2)
    np.testing.assert_allclose(angle.numpy(), angle_flipped.numpy(), atol=1e-8)


def test_geodesic_angle_gradient_finite_near_zero():
    q1 = torch.from_numpy(_hemisphere(_random_quats(20, 3))).requires_grad_(True)
    q2 = q1.detach().clone()
    angle = quat_geodesic_angle(q1, q2)
    angle.sum().backward()
    assert torch.isfinite(q1.grad).all()


# --- quat_to_axis_angle (log map): round trip + scipy cross-check -------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_quat_to_axis_angle_matches_scipy(seed):
    rng = np.random.default_rng(seed)
    axes = rng.normal(size=(200, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    # keep angles within [0, pi] (the principal range quat_to_axis_angle
    # returns after hemisphere normalization) so scipy's own rotvec -- which
    # has no such canonicalization -- is directly comparable.
    angles = rng.uniform(0.0, np.pi, size=(200, 1))
    rotvec = axes * angles

    q = Rotation.from_rotvec(rotvec).as_quat()
    got = quat_to_axis_angle(torch.from_numpy(q)).numpy()
    np.testing.assert_allclose(got, rotvec, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_quat_to_axis_angle_round_trips_with_exp_map(seed):
    rng = np.random.default_rng(seed)
    axes = rng.normal(size=(200, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    angles = rng.uniform(0.0, np.pi, size=(200, 1))  # principal range only
    rotvec = torch.from_numpy(axes * angles)

    q = axis_angle_to_quat(rotvec)
    back = quat_to_axis_angle(q)
    np.testing.assert_allclose(back.numpy(), rotvec.numpy(), atol=1e-6)

    # and the other composition: quat -> rotvec -> quat reproduces the
    # original (hemisphere-normalized) quaternion.
    q2 = _hemisphere(_random_quats(200, seed + 500))
    r2 = quat_to_axis_angle(torch.from_numpy(q2))
    back_q = quat_hemisphere(axis_angle_to_quat(r2)).numpy()
    np.testing.assert_allclose(back_q, q2, atol=1e-6)


def test_quat_to_axis_angle_small_angle_stable():
    q = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [1e-9, 0.0, 0.0, 1.0], [1e-6, -1e-6, 2e-6, 1.0]]
    )
    q = quat_normalize(q)
    r = quat_to_axis_angle(q)
    assert torch.isfinite(r).all()
    np.testing.assert_allclose(r[0].numpy(), [0.0, 0.0, 0.0], atol=1e-9)


def test_quat_to_axis_angle_identity_is_zero():
    q = torch.tensor([0.0, 0.0, 0.0, 1.0])
    r = quat_to_axis_angle(q)
    np.testing.assert_allclose(r.numpy(), [0.0, 0.0, 0.0], atol=1e-12)
