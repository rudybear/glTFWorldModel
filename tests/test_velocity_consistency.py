"""The frame-convention exposé: for a generated multi-object episode with
rotation and contacts, finite-differencing the recorded poses must agree
with the recorded velocities. This is the test that would fail loudly if
``gltfworld.datagen.mj_convert`` mixed up body-local vs. world-frame angular
velocity (MuJoCo free-joint qvel's angular part is body-local; getting that
wrong silently produces a *plausible-looking* but wrong ``ang_vel`` series
that still "looks like a rotation", just the wrong one).

Pure MuJoCo + numpy -- no GPU/EGL needed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from gltfworld.datagen.mujoco_env import simulate
from gltfworld.datagen.sample import sample_scene

_RECORD_HZ = 60.0  # finer sampling makes the finite-difference approximation tighter


def _quat_log_angle_axis(q_xyzw: np.ndarray) -> np.ndarray:
    """xyzw unit quaternion -> rotation vector (axis * angle), shortest path."""
    x, y, z, w = q_xyzw
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    vec_norm = math_sqrt(x * x + y * y + z * z)
    angle = 2.0 * np.arctan2(vec_norm, w)
    if vec_norm < 1e-12:
        return np.zeros(3)
    return np.array([x, y, z]) / vec_norm * angle


def math_sqrt(v: float) -> float:
    return float(np.sqrt(max(v, 0.0)))


def _relative_rotation_vector_world(q_from_xyzw: np.ndarray, q_to_xyzw: np.ndarray) -> np.ndarray:
    """World-frame rotation vector carrying orientation ``q_from`` to
    ``q_to`` in one step: ``dq = q_to * q_from^-1`` (Hamilton product,
    xyzw), consistent with gltfworld's contract world-frame ``ang_vel``
    (see ``gltfworld.datagen.mj_convert`` module docstring)."""
    x1, y1, z1, w1 = q_from_xyzw
    conj_from = np.array([-x1, -y1, -z1, w1])
    x1, y1, z1, w1 = q_to_xyzw
    x2, y2, z2, w2 = conj_from
    # Hamilton product q_to * conj(q_from)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return _quat_log_angle_axis(np.array([x, y, z, w]))


def _generate_episode_with_contacts_and_rotation():
    """A multi-object scene with nonzero initial angular velocity and enough
    fall height to actually contact the ground within the recorded window
    (so both free flight and post-contact tumbling are exercised)."""
    sampled = sample_scene(seed=2024)
    # Give every dynamic object some initial spin so ang_vel is never
    # trivially zero (sample_scene already samples bounded angular
    # velocity, but make sure at least one object clearly spins).
    sampled.initial_ang_vel[1:] = np.array([1.5, -2.0, 1.0], dtype=np.float32)

    series = simulate(
        sampled.scene,
        sampled.initial_poses,
        T=90,
        record_hz=_RECORD_HZ,
        initial_lin_vel=sampled.initial_lin_vel,
        initial_ang_vel=sampled.initial_ang_vel,
    )
    return sampled, series


def test_linear_velocity_matches_finite_difference_of_position():
    """Midpoint finite-difference of position should agree with recorded
    ``lin_vel`` almost everywhere. "Almost": a genuine ground contact
    (bounce) is an instantaneous velocity discontinuity that no
    finite-difference window can smoothly track -- confirmed by inspection,
    this scene's contacts produce a handful of such outlier frames (a
    sphere's downward velocity jumping from -3.5 to +0.4 m/s within one
    recorded interval, i.e. a real bounce, not a bug). A wrong frame
    convention would instead show up as a *systematic* mismatch at nearly
    every frame (wrong sign, wrong axis, or a rotation of the whole
    vector), so this checks the 85th percentile of the per-sample error
    (tolerating a handful of contact-frame outliers) rather than the max."""
    sampled, series = _generate_episode_with_contacts_and_rotation()
    dt = float(series.times[1] - series.times[0])
    n_objects = series.num_objects

    max_speed = float(np.abs(series.lin_vel).max())
    assert max_speed > 0.05, "sanity: episode should have some real motion"
    tol = 0.02 * max_speed  # 2% of max speed, per spec

    for i in range(n_objects):
        if sampled.scene.objects[i].is_static:
            continue
        pos = series.poses[:, i, 0:3]
        # midpoint finite difference: (p[t+1]-p[t-1]) / (2*dt) approximates
        # the velocity AT t, matching the recorded lin_vel[t] (not the
        # velocity of the t->t+1 interval).
        fd = (pos[2:] - pos[:-2]) / (2.0 * dt)
        recorded = series.lin_vel[1:-1, i, :]
        err = np.abs(fd - recorded)
        p85 = float(np.percentile(err, 85))
        assert p85 < tol, (
            f"object {sampled.scene.objects[i].object_id}: lin_vel finite-difference mismatch "
            f"(85th percentile) {p85:.4f} (tol {tol:.4f}); max was {err.max():.4f} "
            f"(a few large-error contact/bounce frames are expected and excluded by the percentile)"
        )


def test_angular_velocity_matches_finite_difference_of_quaternion_log():
    """Same reasoning as the linear-velocity check above: a genuine contact
    can impart an instantaneous angular-velocity change (friction-induced
    spin change on impact) that a finite-difference window can't track, so
    this checks the 85th-percentile error rather than the max. A wrong
    body-local/world-frame choice for MuJoCo's free-joint qvel would show
    up as a large, systematic (nearly-every-frame) mismatch here, not a
    handful of contact-frame outliers."""
    sampled, series = _generate_episode_with_contacts_and_rotation()
    dt = float(series.times[1] - series.times[0])
    n_objects = series.num_objects

    max_ang_speed = float(np.linalg.norm(series.ang_vel, axis=-1).max())
    assert max_ang_speed > 0.05, "sanity: episode should have some real rotation"
    tol = 0.02 * max_ang_speed

    for i in range(n_objects):
        if sampled.scene.objects[i].is_static:
            continue
        quat = series.poses[:, i, 3:7]
        T = quat.shape[0]
        errs = []
        for t in range(1, T - 1):
            # midpoint finite difference via quaternion log: rotation
            # vector carrying q[t-1] -> q[t+1], divided by the 2*dt window,
            # approximates the WORLD-frame angular velocity at t.
            rotvec = _relative_rotation_vector_world(quat[t - 1], quat[t + 1])
            fd_ang_vel = rotvec / (2.0 * dt)
            recorded = series.ang_vel[t, i, :]
            errs.append(np.abs(fd_ang_vel - recorded))
        errs = np.array(errs)
        p85 = float(np.percentile(errs, 85))
        assert p85 < tol, (
            f"object {sampled.scene.objects[i].object_id}: ang_vel finite-difference mismatch "
            f"(85th percentile) {p85:.4f} (tol {tol:.4f}); max was {errs.max():.4f} -- this is "
            f"exactly the check that fails if body-local vs world-frame angular velocity was "
            f"mixed up (a systematic bug shows up at nearly every frame, not just a few outliers)"
        )
