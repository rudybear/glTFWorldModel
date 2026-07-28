"""Unit tests for ``gltfworld.datagen.mj_convert``, THE MuJoCo<->contract
conversion module (see its module docstring for the exact fixed
change-of-basis matrix and velocity-frame reasoning).

None of these need MuJoCo itself or a GPU/EGL context -- pure numpy -- so
none are ``@pytest.mark.gpu``. The "did we get the free-joint velocity
frame right" exposé against a *real* MuJoCo simulation lives in
``tests/test_velocity_consistency.py`` and ``tests/test_freefall.py``.
"""

from __future__ import annotations

import math

import numpy as np

from gltfworld.datagen import mj_convert as mc

_IDENTITY_XYZW = np.array([0.0, 0.0, 0.0, 1.0])


def _rand_unit_quats(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q


def _quat_to_matrix_xyzw(q: np.ndarray) -> np.ndarray:
    """Reference (independent) xyzw-quaternion -> 3x3 rotation matrix, used
    only to build ground truth for the "rotation self-consistency" test."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat_to_matrix_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return _quat_to_matrix_xyzw(np.array([x, y, z, w]))


# --- known cases: position/vector axis mapping -------------------------------


def test_known_case_mj_plus_z_maps_to_gltf_plus_y():
    """MuJoCo's up axis (+Z) is glTF's up axis (+Y)."""
    assert np.allclose(mc.mj_pos_to_gltf(np.array([0.0, 0.0, 1.0])), [0.0, 1.0, 0.0])


def test_known_case_gltf_plus_y_maps_to_mj_plus_z():
    assert np.allclose(mc.gltf_pos_to_mj(np.array([0.0, 1.0, 0.0])), [0.0, 0.0, 1.0])


def test_known_case_shared_x_axis_unchanged():
    assert np.allclose(mc.gltf_pos_to_mj(np.array([1.0, 0.0, 0.0])), [1.0, 0.0, 0.0])
    assert np.allclose(mc.mj_pos_to_gltf(np.array([1.0, 0.0, 0.0])), [1.0, 0.0, 0.0])


def test_known_case_gltf_plus_z_maps_to_mj_minus_y():
    assert np.allclose(mc.gltf_pos_to_mj(np.array([0.0, 0.0, 1.0])), [0.0, -1.0, 0.0])


def test_known_case_identity_orientation_quaternion():
    """An identity-oriented gltf object is *not* identity-oriented in
    mj-frame: it picks up the fixed +90 degree axis-change rotation (about
    the shared X axis) itself, since the two "up" axes differ. This is the
    behavior the V2 crosscheck's IoU was empirically validated against and
    must not regress (see tests/test_crosscheck.py)."""
    quat_wxyz = mc.gltf_quat_to_mj(_IDENTITY_XYZW)
    assert quat_wxyz.shape == (4,)
    assert math.isclose(float(np.linalg.norm(quat_wxyz)), 1.0, rel_tol=1e-9)
    assert np.allclose(quat_wxyz, [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0], atol=1e-9)


# --- proper rotation (determinant / orthonormality) --------------------------


def test_gltf_to_mj_position_map_is_a_proper_rotation():
    basis = np.eye(3)
    mapped = np.array([mc.gltf_pos_to_mj(basis[i]) for i in range(3)])
    R = mapped.T  # mapped rows are R @ e_i, so R = mapped.T
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert math.isclose(float(np.linalg.det(R)), 1.0, abs_tol=1e-9)


def test_mj_to_gltf_position_map_is_a_proper_rotation():
    basis = np.eye(3)
    mapped = np.array([mc.mj_pos_to_gltf(basis[i]) for i in range(3)])
    R = mapped.T
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert math.isclose(float(np.linalg.det(R)), 1.0, abs_tol=1e-9)


# --- rotation self-consistency: orientation conversion agrees with the ------
# --- vector conversion applied to a rotated local reference point -----------


def test_orientation_conversion_agrees_with_vector_conversion():
    """For any local reference point p and gltf orientation q, rotating p by
    q (in gltf space) and then converting the resulting *vector* to mj axes
    must equal converting q to mj first and rotating p by the result
    directly. This is the actual defining correctness property of
    ``gltf_quat_to_mj`` (and its inverse): the two conversions must commute
    with "apply this rotation to a local vector"."""
    rng = np.random.default_rng(42)
    quats = _rand_unit_quats(rng, 20)
    points = rng.normal(size=(20, 3))

    for q_gltf, p_local in zip(quats, points):
        world_gltf = _quat_to_matrix_xyzw(q_gltf) @ p_local
        via_vector_map = mc.gltf_vec_to_mj(world_gltf)

        q_mj = mc.gltf_quat_to_mj(q_gltf)
        via_orientation_map = _quat_to_matrix_wxyz(q_mj) @ p_local

        assert np.allclose(via_vector_map, via_orientation_map, atol=1e-9)


def test_orientation_conversion_agrees_with_vector_conversion_inverse():
    rng = np.random.default_rng(43)
    quats = _rand_unit_quats(rng, 20)
    points = rng.normal(size=(20, 3))

    for q_wxyz, p_local in zip(quats, points):
        # reinterpret as a wxyz mj quat for this direction
        world_mj = _quat_to_matrix_wxyz(q_wxyz) @ p_local
        via_vector_map = mc.mj_vec_to_gltf(world_mj)

        q_gltf = mc.mj_quat_to_gltf(q_wxyz)
        via_orientation_map = _quat_to_matrix_xyzw(q_gltf) @ p_local

        assert np.allclose(via_vector_map, via_orientation_map, atol=1e-9)


# --- round trips --------------------------------------------------------------


def _assert_quats_equal_up_to_sign(a: np.ndarray, b: np.ndarray, atol=1e-6):
    same = np.abs(a - b).max(axis=-1)
    flipped = np.abs(a + b).max(axis=-1)
    assert np.all(np.minimum(same, flipped) < atol)


def test_pose_round_trip_gltf_to_mj_to_gltf():
    rng = np.random.default_rng(7)
    pos = rng.normal(size=(50, 3)).astype(np.float32)
    quat = _rand_unit_quats(rng, 50).astype(np.float32)

    pos_mj, quat_mj = mc.gltf_pose_to_mj(pos, quat)
    pos2, quat2 = mc.mj_pose_to_gltf(pos_mj, quat_mj)

    assert np.allclose(pos2, pos, atol=1e-5)
    _assert_quats_equal_up_to_sign(quat2, quat, atol=1e-5)


def test_pose_round_trip_mj_to_gltf_to_mj():
    rng = np.random.default_rng(8)
    pos = rng.normal(size=(50, 3)).astype(np.float32)
    quat = _rand_unit_quats(rng, 50).astype(np.float32)

    pos_gltf, quat_gltf = mc.mj_pose_to_gltf(pos, quat)
    pos2, quat2 = mc.gltf_pose_to_mj(pos_gltf, quat_gltf)

    assert np.allclose(pos2, pos, atol=1e-5)
    _assert_quats_equal_up_to_sign(quat2, quat, atol=1e-5)


def test_velocity_round_trip():
    rng = np.random.default_rng(9)
    quat_wxyz = _rand_unit_quats(rng, 30)
    lin = rng.normal(size=(30, 3)) * 2.0
    ang = rng.normal(size=(30, 3)) * 3.0

    lin_gltf, ang_gltf = mc.mj_freejoint_vel_to_gltf_world(quat_wxyz, lin, ang)
    lin2, ang2 = mc.gltf_world_vel_to_mj_freejoint(quat_wxyz, lin_gltf, ang_gltf)

    assert np.allclose(lin2, lin, atol=1e-8)
    assert np.allclose(ang2, ang, atol=1e-8)


# --- batch == scalar paths -----------------------------------------------------


def test_batch_equals_scalar_pos_and_quat():
    rng = np.random.default_rng(11)
    pos = rng.normal(size=(10, 3))
    quat = _rand_unit_quats(rng, 10)

    pos_mj_batch, quat_mj_batch = mc.gltf_pose_to_mj(pos, quat)
    for i in range(10):
        pos_mj_scalar, quat_mj_scalar = mc.gltf_pose_to_mj(pos[i], quat[i])
        assert np.allclose(pos_mj_scalar, pos_mj_batch[i])
        assert np.allclose(quat_mj_scalar, quat_mj_batch[i])


def test_batch_equals_scalar_poses_helper():
    rng = np.random.default_rng(12)
    poses = np.concatenate([rng.normal(size=(5, 3, 3)), _rand_unit_quats(rng, 15).reshape(5, 3, 4)], axis=-1)

    pos_mj_batch, quat_mj_batch = mc.gltf_poses_to_mj(poses)
    for t in range(5):
        for n in range(3):
            pos_s, quat_s = mc.gltf_pose_to_mj(poses[t, n, 0:3], poses[t, n, 3:7])
            assert np.allclose(pos_s, pos_mj_batch[t, n])
            assert np.allclose(quat_s, quat_mj_batch[t, n])


def test_batch_equals_scalar_velocity():
    rng = np.random.default_rng(13)
    quat_wxyz = _rand_unit_quats(rng, 8)
    lin = rng.normal(size=(8, 3))
    ang = rng.normal(size=(8, 3))

    lin_batch, ang_batch = mc.mj_freejoint_vel_to_gltf_world(quat_wxyz, lin, ang)
    for i in range(8):
        lin_s, ang_s = mc.mj_freejoint_vel_to_gltf_world(quat_wxyz[i], lin[i], ang[i])
        assert np.allclose(lin_s, lin_batch[i])
        assert np.allclose(ang_s, ang_batch[i])


def test_mj_poses_to_gltf_matches_pose_helper():
    rng = np.random.default_rng(14)
    pos_mj = rng.normal(size=(4, 3))
    quat_mj = _rand_unit_quats(rng, 4)

    combined = mc.mj_poses_to_gltf(pos_mj, quat_mj)
    pos_gltf, quat_gltf = mc.mj_pose_to_gltf(pos_mj, quat_mj)
    assert np.allclose(combined[:, 0:3], pos_gltf)
    assert np.allclose(combined[:, 3:7], quat_gltf)
