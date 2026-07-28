"""Free-fall physics sanity check: a single sphere dropped from a known
height with zero initial velocity, high enough above the ground that it
never contacts anything in the recorded window, should match closed-form
projectile motion. Pure MuJoCo + numpy -- no GPU/EGL needed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from gltfworld.datagen.mujoco_env import simulate
from gltfworld.scene.scene import CameraSpec, ObjectSpec, SceneState

_G = 9.81


def _make_freefall_scene(h0: float, sphere_radius: float = 0.1) -> tuple[SceneState, np.ndarray]:
    ground = ObjectSpec(
        object_id=0,
        shape="box",
        size=np.array([5.0, 0.1, 5.0], dtype=np.float32),
        color=np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32),
        roughness=0.9,
        metallic=0.0,
        mass=1000.0,
        friction=0.6,
        restitution=0.1,
        is_static=True,
        category="ground",
    )
    ball = ObjectSpec(
        object_id=1,
        shape="sphere",
        size=np.array([sphere_radius, sphere_radius, sphere_radius], dtype=np.float32),
        color=np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        roughness=0.4,
        metallic=0.0,
        mass=1.0,
        friction=0.6,
        restitution=0.1,
        is_static=False,
        category="ball",
    )
    camera = CameraSpec(
        position=np.array([0.0, 2.0, 4.0], dtype=np.float32),
        rotation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        yfov=0.8,
        znear=0.05,
        zfar=100.0,
        aspect=1.0,
    )
    scene = SceneState(
        objects=[ground, ball],
        camera=camera,
        lights=[],
        gravity=np.array([0.0, -_G, 0.0], dtype=np.float32),
        dt=1.0 / 30.0,
        seed=0,
    )
    initial_poses = np.zeros((2, 7), dtype=np.float32)
    initial_poses[:, 6] = 1.0  # identity quat
    initial_poses[0, 1] = -0.1  # ground center (top at y=0)
    initial_poses[1, 1] = h0
    return scene, initial_poses


def test_freefall_position_matches_closed_form():
    h0 = 5.0  # well above ground; sphere never contacts anything in this window
    scene, initial_poses = _make_freefall_scene(h0)

    series = simulate(scene, initial_poses, T=20, record_hz=30)

    y = series.poses[:, 1, 1]
    y_expected = h0 - 0.5 * _G * series.times**2

    # never contacts ground in this window
    assert (y > 1.0).all()

    rel_err = np.abs(y - y_expected) / h0
    assert rel_err.max() < 0.01, f"position deviates from closed form by {rel_err.max():.4%} (want < 1%)"


def test_freefall_velocity_matches_closed_form():
    h0 = 5.0
    scene, initial_poses = _make_freefall_scene(h0)

    series = simulate(scene, initial_poses, T=20, record_hz=30)

    vy = series.lin_vel[:, 1, 1]
    vy_expected = -_G * series.times

    # denominator: max expected speed in the window, to avoid dividing by ~0 near t=0
    max_speed = float(np.abs(vy_expected).max())
    rel_err = np.abs(vy - vy_expected) / max_speed
    assert rel_err.max() < 0.01, f"lin_vel deviates from closed form by {rel_err.max():.4%} (want < 1%)"


def test_freefall_x_z_and_angular_velocity_stay_zero():
    """No horizontal drift, no rotation, for a sphere dropped with zero
    initial velocity/spin and no contacts -- gravity is purely -Y."""
    h0 = 5.0
    scene, initial_poses = _make_freefall_scene(h0)
    series = simulate(scene, initial_poses, T=20, record_hz=30)

    assert np.allclose(series.poses[:, 1, 0], 0.0, atol=1e-5)
    assert np.allclose(series.poses[:, 1, 2], 0.0, atol=1e-5)
    assert np.allclose(series.ang_vel[:, 1, :], 0.0, atol=1e-6)
