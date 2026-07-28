"""``wm-scenes-v1`` distribution sanity checks: determinism, constraint
compliance, camera framing. Pure numpy -- no MuJoCo, no GPU/EGL.
"""

from __future__ import annotations

import math

import numpy as np

from gltfworld.datagen.sample import (
    _DENSITY_MAX,
    _DENSITY_MIN,
    _DROP_HEIGHT_MIN,
    _FRICTION_MAX,
    _FRICTION_MIN,
    _N_OBJECTS_MAX,
    _N_OBJECTS_MIN,
    _SIZE_MAX,
    _SIZE_MIN,
    _WORKSPACE_HALF_XZ,
    object_bounding_radius,
    point_in_frustum,
    sample_scene,
    shape_volume,
)

_N_SCENES = 50


def _sampled_scenes(n=_N_SCENES):
    return [sample_scene(seed) for seed in range(n)]


def test_determinism_same_seed_gives_identical_scene():
    a = sample_scene(12345)
    b = sample_scene(12345)

    assert len(a.scene.objects) == len(b.scene.objects)
    for obj_a, obj_b in zip(a.scene.objects, b.scene.objects):
        assert obj_a.object_id == obj_b.object_id
        assert obj_a.shape == obj_b.shape
        assert np.array_equal(obj_a.size, obj_b.size)
        assert np.array_equal(obj_a.color, obj_b.color)
        assert obj_a.mass == obj_b.mass
        assert obj_a.friction == obj_b.friction

    assert np.array_equal(a.initial_poses, b.initial_poses)
    assert np.array_equal(a.initial_lin_vel, b.initial_lin_vel)
    assert np.array_equal(a.initial_ang_vel, b.initial_ang_vel)


def test_different_seeds_give_different_scenes():
    a = sample_scene(1)
    b = sample_scene(2)
    # overwhelmingly likely to differ somewhere (position, size, ...)
    same_shapes = [o.shape for o in a.scene.objects] == [o.shape for o in b.scene.objects]
    same_poses = a.initial_poses.shape == b.initial_poses.shape and np.array_equal(
        a.initial_poses, b.initial_poses
    )
    assert not (same_shapes and same_poses)


def test_n_objects_in_range():
    for sampled in _sampled_scenes():
        n_dynamic = sum(1 for obj in sampled.scene.objects if not obj.is_static)
        assert _N_OBJECTS_MIN <= n_dynamic <= _N_OBJECTS_MAX


def test_exactly_one_ground_object():
    for sampled in _sampled_scenes():
        grounds = [obj for obj in sampled.scene.objects if obj.category == "ground"]
        assert len(grounds) == 1
        assert grounds[0].is_static


def test_shapes_and_categories_valid():
    for sampled in _sampled_scenes():
        for obj in sampled.scene.objects:
            if obj.category == "ground":
                continue
            assert obj.shape in ("sphere", "box", "cylinder")
            assert obj.category in ("ball", "crate", "cylinder")


def test_characteristic_size_in_range():
    for sampled in _sampled_scenes():
        for obj in sampled.scene.objects:
            if obj.category == "ground":
                continue
            if obj.shape == "cylinder":
                # size = [radius, half_height, radius]; half_height carries
                # the sampled characteristic size directly (see _sample_size)
                characteristic = obj.size[1]
            else:
                characteristic = obj.size[0]
            assert _SIZE_MIN - 1e-6 <= characteristic <= _SIZE_MAX + 1e-6


def test_mass_matches_density_times_volume_in_range():
    for sampled in _sampled_scenes():
        for obj in sampled.scene.objects:
            if obj.category == "ground":
                continue
            volume = shape_volume(obj.shape, obj.size)
            implied_density = obj.mass / volume
            assert _DENSITY_MIN - 1e-3 <= implied_density <= _DENSITY_MAX + 1e-3


def test_friction_and_restitution_in_range():
    for sampled in _sampled_scenes():
        for obj in sampled.scene.objects:
            if obj.category == "ground":
                continue
            assert _FRICTION_MIN - 1e-6 <= obj.friction <= _FRICTION_MAX + 1e-6
            assert math.isclose(obj.restitution, 0.1, abs_tol=1e-6)


def test_velocity_bounds():
    for sampled in _sampled_scenes():
        speeds = np.linalg.norm(sampled.initial_lin_vel, axis=-1)
        ang_speeds = np.linalg.norm(sampled.initial_ang_vel, axis=-1)
        assert (speeds <= 1.5 + 1e-5).all()
        assert (ang_speeds <= 3.0 + 1e-5).all()


def test_orientation_quaternions_are_unit():
    for sampled in _sampled_scenes():
        norms = np.linalg.norm(sampled.initial_poses[:, 3:7], axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-4)


def test_no_initial_overlaps():
    for sampled in _sampled_scenes():
        n = len(sampled.scene.objects)
        for i in range(1, n):
            for j in range(i + 1, n):
                pi = sampled.initial_poses[i, 0:3]
                pj = sampled.initial_poses[j, 0:3]
                ri = object_bounding_radius(sampled.scene.objects[i])
                rj = object_bounding_radius(sampled.scene.objects[j])
                dist = float(np.linalg.norm(pi - pj))
                assert dist >= ri + rj - 1e-4, (
                    f"objects {i},{j} overlap: dist={dist:.4f} < r_i+r_j={ri + rj:.4f}"
                )


def test_objects_within_workspace_footprint():
    margin = 1e-4
    for sampled in _sampled_scenes():
        for obj, pose in zip(sampled.scene.objects, sampled.initial_poses):
            if obj.category == "ground":
                continue
            x, _, z = pose[0:3]
            assert -_WORKSPACE_HALF_XZ - margin <= x <= _WORKSPACE_HALF_XZ + margin
            assert -_WORKSPACE_HALF_XZ - margin <= z <= _WORKSPACE_HALF_XZ + margin


def test_objects_dropped_above_ground():
    for sampled in _sampled_scenes():
        for obj, pose in zip(sampled.scene.objects, sampled.initial_poses):
            if obj.category == "ground":
                continue
            radius = object_bounding_radius(obj)
            bottom = pose[1] - radius
            assert bottom >= -1e-4, f"object starts below ground: bottom={bottom:.4f}"
            clearance = bottom
            assert clearance <= _DROP_HEIGHT_MIN + 5.0  # sanity upper bound, not a tight physical one


def test_all_objects_inside_camera_frustum_at_t0():
    for sampled in _sampled_scenes():
        camera = sampled.scene.camera
        for obj, pose in zip(sampled.scene.objects, sampled.initial_poses):
            if obj.category == "ground":
                # the ground plane is intentionally huge (6m x 6m) and not
                # meant to fit entirely inside frame -- only the dynamic
                # workspace objects need to.
                continue
            radius = object_bounding_radius(obj)
            assert point_in_frustum(camera, pose[0:3], radius), (
                f"object {obj.object_id} ({obj.shape}) at {pose[0:3]} (radius {radius:.3f}) "
                f"is not fully inside the fixed camera's frustum"
            )
