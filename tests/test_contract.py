"""Tensor-contract round trip: ``episode_to_tensors`` -> ``tensors_to_state``
reproduces the dynamic part of an Episode to <= 1e-6 relative error.
"""

from __future__ import annotations

import numpy as np
import pytest

from gltfworld.scene.contract import (
    CATEGORY_TO_CLASS_ID,
    GLOBALS_DIM,
    STATE_DIM,
    episode_to_tensors,
    tensors_to_state,
)

from conftest import make_sample_episode


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom < 1e-8, 1.0, denom)
    return float(np.max(np.abs(a - b) / denom))


def test_shapes_and_dtypes():
    ep = make_sample_episode(n_objects=4, T=20)
    tensors = episode_to_tensors(ep)

    n_dynamic = sum(1 for obj in ep.scene.objects if not obj.is_static)
    assert tensors["states"].shape == (20, n_dynamic, STATE_DIM)
    assert tensors["states"].dtype == np.float32
    assert tensors["mask"].shape == (n_dynamic,)
    assert tensors["mask"].dtype == np.bool_
    assert tensors["mask"].all()
    assert tensors["class_ids"].shape == (n_dynamic,)
    assert tensors["class_ids"].dtype == np.int64
    assert tensors["globals"].shape == (GLOBALS_DIM,)
    assert tensors["globals"].dtype == np.float32
    assert len(tensors["static"]) == 1  # ground only


def test_quaternion_hemisphere_normalized():
    ep = make_sample_episode(n_objects=3, T=10)
    tensors = episode_to_tensors(ep)
    w = tensors["states"][..., 6]
    assert np.all(w >= 0.0)
    w_cam = tensors["globals"][10]
    assert w_cam >= 0.0


def test_class_ids_match_category_map():
    ep = make_sample_episode(n_objects=5, T=5)
    tensors = episode_to_tensors(ep)
    dynamic = [obj for obj in ep.scene.objects if not obj.is_static]
    for obj, class_id in zip(dynamic, tensors["class_ids"]):
        assert class_id == CATEGORY_TO_CLASS_ID[obj.category]


@pytest.mark.parametrize("n_objects,T", [(1, 1), (3, 30), (5, 50)])
def test_round_trip_dynamic_part(n_objects: int, T: int):
    ep = make_sample_episode(n_objects=n_objects, T=T)
    tensors = episode_to_tensors(ep)
    recon = tensors_to_state(
        tensors["states"], tensors["mask"], tensors["class_ids"], tensors["globals"], template=ep.scene
    )

    dynamic_indices = [i for i, obj in enumerate(ep.scene.objects) if not obj.is_static]
    orig_poses = ep.series.poses[:, dynamic_indices, :]
    orig_lin_vel = ep.series.lin_vel[:, dynamic_indices, :]
    orig_ang_vel = ep.series.ang_vel[:, dynamic_indices, :]

    # Hemisphere-normalize the original quats the same way encode does, so
    # comparison is well defined regardless of input sign (see contract.py
    # module docstring).
    orig_quat = orig_poses[..., 3:7].copy()
    sign = np.where(orig_quat[..., 3:4] < 0, -1.0, 1.0)
    orig_quat = (orig_quat * sign).astype(np.float32)

    assert _rel_err(recon["poses"][..., 0:3], orig_poses[..., 0:3]) <= 1e-6
    assert _rel_err(recon["poses"][..., 3:7], orig_quat) <= 1e-6
    assert _rel_err(recon["lin_vel"], orig_lin_vel) <= 1e-6
    assert _rel_err(recon["ang_vel"], orig_ang_vel) <= 1e-6
    assert _rel_err(recon["gravity"], ep.scene.gravity) <= 1e-6
    assert abs(recon["dt"] - ep.scene.dt) / max(abs(ep.scene.dt), 1e-8) <= 1e-6

    for recon_obj, obj_index in zip(recon["objects"], dynamic_indices):
        orig_obj = ep.scene.objects[obj_index]
        assert recon_obj.shape == orig_obj.shape
        assert _rel_err(recon_obj.size, orig_obj.size) <= 1e-6
        assert abs(recon_obj.mass - orig_obj.mass) / max(abs(orig_obj.mass), 1e-8) <= 1e-6
        assert abs(recon_obj.friction - orig_obj.friction) / max(abs(orig_obj.friction), 1e-8) <= 1e-6
        assert (
            abs(recon_obj.restitution - orig_obj.restitution) / max(abs(orig_obj.restitution), 1e-8)
            <= 1e-6
        )


def test_static_object_excluded_from_states():
    ep = make_sample_episode(n_objects=2, T=5)
    tensors = episode_to_tensors(ep)
    ground = ep.scene.objects[0]
    assert ground.is_static
    static_entry = tensors["static"][str(ground.object_id)]
    assert static_entry["shape"] == ground.shape
    assert static_entry["category"] == "ground"
    np.testing.assert_allclose(static_entry["size"], ground.size)
