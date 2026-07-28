"""MuJoCo cross-render oracle tests (see `gltfworld.render.crosscheck`).

The coordinate-conversion unit test is pure numpy and runs everywhere; the
actual cross-render comparison needs both a GPU/EGL context (gltfworld's own
renderer) and MuJoCo's renderer, so it's `@pytest.mark.gpu`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import make_sample_episode

from gltfworld.render.crosscheck import gltf_pose_to_mujoco


def test_gltf_pose_to_mujoco_up_axis_and_position_mapping():
    """glTF Y-up -> MuJoCo Z-up: (x, y, z) -> (x, -z, y)."""
    identity_quat = np.array([0.0, 0.0, 0.0, 1.0])

    pos, _ = gltf_pose_to_mujoco([0.0, 1.0, 0.0], identity_quat)
    assert np.allclose(pos, [0.0, 0.0, 1.0]), "gltf +Y (up) must map to mujoco +Z (up)"

    pos, _ = gltf_pose_to_mujoco([1.0, 0.0, 0.0], identity_quat)
    assert np.allclose(pos, [1.0, 0.0, 0.0]), "the shared X axis must be unchanged"

    pos, _ = gltf_pose_to_mujoco([0.0, 0.0, 1.0], identity_quat)
    assert np.allclose(pos, [0.0, -1.0, 0.0]), "gltf +Z must map to mujoco -Y"


def test_gltf_pose_to_mujoco_quaternion_order_and_normalization():
    identity_quat = np.array([0.0, 0.0, 0.0, 1.0])
    _, quat_wxyz = gltf_pose_to_mujoco([0.0, 0.0, 0.0], identity_quat)

    assert quat_wxyz.shape == (4,)
    assert math.isclose(float(np.linalg.norm(quat_wxyz)), 1.0, rel_tol=1e-9)
    # w, x, y, z of a +90 degree rotation about the shared X axis.
    assert np.allclose(quat_wxyz, [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0], atol=1e-9)


def test_gltf_pose_to_mujoco_is_a_proper_rotation():
    """The axis-change map must be a proper rotation (det +1, orthonormal):
    not a reflection, so handedness/chirality of geometry is preserved."""
    identity_quat = np.array([0.0, 0.0, 0.0, 1.0])
    basis = np.eye(3)
    mapped = np.array([gltf_pose_to_mujoco(basis[i], identity_quat)[0] for i in range(3)])
    # `mapped` rows are R @ e_i, so `mapped.T` is R itself.
    R = mapped.T
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), "must be orthonormal"
    assert math.isclose(float(np.linalg.det(R)), 1.0, abs_tol=1e-9), "must be a proper rotation (det +1)"


@pytest.mark.gpu
def test_crosscheck_binary_silhouette_iou(episode_renderer, tmp_path):
    pytest.importorskip("mujoco")
    from gltfworld.render.crosscheck import crosscheck_frame0, write_side_by_side_png

    episode = make_sample_episode()
    result = crosscheck_frame0(episode, episode_renderer, width=256, height=256)

    write_side_by_side_png(result, "/tmp/gltfworld_crosscheck/sample_episode.png")

    assert result.iou >= 0.98, f"binary silhouette IoU too low: {result.iou:.4f}"

    for object_id, iou in result.per_object_iou.items():
        assert iou >= 0.0  # sanity: always computed when present; printed for the report
        print(f"object_id={object_id} per-object IoU={iou:.4f}")
    print(f"binary silhouette IoU={result.iou:.4f}")
