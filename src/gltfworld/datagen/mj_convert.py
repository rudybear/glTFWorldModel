"""THE MuJoCo <-> gltfworld contract conversion module.

All MuJoCo<->contract coordinate/quaternion/velocity conversion lives here
and ONLY here (``gltfworld.render.crosscheck`` imports its
``gltf_pose_to_mujoco`` from this module rather than defining its own -- see
DESIGN.md "conversion consolidation").

Frame conventions
------------------

- **gltfworld contract** (``gltfworld.scene``): Y-up, right-handed world;
  quaternions ``(x, y, z, w)``.
- **MuJoCo**: Z-up, right-handed world; quaternions ``(w, x, y, z)``.

The two world frames are related by a single fixed +90 degree rotation about
the shared X axis:

```
R_gltf_to_mj = [[1,  0,  0],
                [0,  0, -1],
                [0,  1,  0]]
```

i.e. ``(x, y, z)_gltf -> (x, -z, y)_mj``. This is a proper rotation
(determinant +1, orthonormal -- not a reflection), so it preserves
handedness/chirality; only which axis is "up" changes. Since it is
orthonormal, the inverse is just the transpose:

```
R_mj_to_gltf = R_gltf_to_mj.T = [[1,  0,  0],
                                 [0,  0,  1],
                                 [0, -1,  0]]
```

i.e. ``(x, y, z)_mj -> (x, z, -y)_gltf``.

Positions and any other world-frame *vectors* (linear/angular velocity)
transform by this same linear map -- there is no translation between the
two world origins, just a change of "which way is up". Orientations
transform by *composing* the fixed axis-change rotation with the object's
own rotation (``q_mj = q_axis_change (x) q_gltf``, applied in world space,
Hamilton product), since an object's local axes must map into the *new*
world frame the same way the world frame itself was remapped.

All functions below accept arrays whose last axis is the vector/quaternion
(shape ``(3,)``/``(4,)``) with arbitrary leading batch dimensions (including
none) -- there is exactly one implementation per conversion, used
identically for a single pose or a whole ``(T, N, ...)`` batch (see
"batch == scalar paths" in ``tests/test_mj_convert.py``).

Velocities
----------

MuJoCo's free-joint ``qvel`` is 6 numbers: 3 linear velocity components
**in the world frame**, followed by 3 angular velocity components **in the
body-local frame** -- verified empirically against MuJoCo 3.11 (see the
scratch scripts referenced in the V3 report; MuJoCo's own docs are not
fully explicit about this, so this module trusts the empirical check:
integrating ``qpos``'s quaternion one step and comparing the finite-difference
angular velocity, expressed in both the world frame and the body-local
frame, against the ``qvel[3:6]`` value that was set, shows an exact match
only for the body-local-frame interpretation).

gltfworld's contract stores **world-frame** linear and angular velocity
(``StateSeries.lin_vel``/``ang_vel``), in glTF axes. So converting a
free-joint's recorded ``qvel`` into the contract requires: linear part ->
straight axis remap (already world-frame); angular part -> first rotate
into MuJoCo's world frame using the body's *current* orientation
(``omega_world_mj = R(q_mj) @ omega_body_mj``), *then* axis-remap into glTF
axes. The inverse (used to seed initial ``qvel`` from a sampled/contract
world-frame angular velocity) un-rotates by the same orientation.
"""

from __future__ import annotations

import numpy as np

# --- fixed change-of-basis (see module docstring) ----------------------------

_R_GLTF_TO_MJ = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]
)
_R_MJ_TO_GLTF = _R_GLTF_TO_MJ.T

# The above rotation as an xyzw quaternion (glTF/world convention): +90
# degrees about the shared X axis.
_AXIS_QUAT_XYZW = np.array([np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0)])
_AXIS_QUAT_XYZW_CONJ = np.array([-np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0)])


# --- quaternion helpers (all operate elementwise over a trailing (4,) axis) --


def _quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two xyzw quaternions: ``q1 (x) q2`` means "first
    rotate by q2, then by q1", broadcasting over any leading batch shape."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.stack([x, y, z, w], axis=-1)


def _xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return np.stack([q_xyzw[..., 3], q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2]], axis=-1)


def _wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return np.stack([q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3], q_wxyz[..., 0]], axis=-1)


def _rotate_vector_wxyz(quat_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) ``v`` (..., 3) by unit quaternion(s) ``quat_wxyz``
    (..., 4), broadcasting over any leading batch shape. Standard
    branchless quaternion-vector rotation (Krasjet/Fabian identity):
    ``v' = v + w*t + cross(q_xyz, t)`` where ``t = 2 * cross(q_xyz, v)``.
    """
    w = quat_wxyz[..., 0]
    q_xyz = quat_wxyz[..., 1:4]
    t = 2.0 * np.cross(q_xyz, v)
    return v + w[..., None] * t + np.cross(q_xyz, t)


def _quat_conj_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion (wxyz): negate the vector part."""
    w = quat_wxyz[..., 0:1]
    xyz = -quat_wxyz[..., 1:4]
    return np.concatenate([w, xyz], axis=-1)


# --- position / vector conversion --------------------------------------------


def gltf_pos_to_mj(pos_gltf: np.ndarray) -> np.ndarray:
    """``(..., 3)`` gltf-axes position/vector -> mujoco-axes position/vector."""
    pos_gltf = np.asarray(pos_gltf, dtype=np.float64)
    return np.einsum("ij,...j->...i", _R_GLTF_TO_MJ, pos_gltf)


def mj_pos_to_gltf(pos_mj: np.ndarray) -> np.ndarray:
    """``(..., 3)`` mujoco-axes position/vector -> gltf-axes position/vector."""
    pos_mj = np.asarray(pos_mj, dtype=np.float64)
    return np.einsum("ij,...j->...i", _R_MJ_TO_GLTF, pos_mj)


# Vectors (velocities, angular velocities, gravity, ...) transform by the
# exact same linear map as positions -- there is no translation between the
# two world origins. Separate names are kept for readability at call sites.
gltf_vec_to_mj = gltf_pos_to_mj
mj_vec_to_gltf = mj_pos_to_gltf


# --- quaternion / pose conversion ---------------------------------------------


def gltf_quat_to_mj(quat_xyzw_gltf: np.ndarray) -> np.ndarray:
    """``(..., 4)`` xyzw gltf-frame orientation -> ``(..., 4)`` wxyz mujoco-frame orientation."""
    quat_xyzw_gltf = np.asarray(quat_xyzw_gltf, dtype=np.float64)
    q_mj_xyzw = _quat_mul_xyzw(_AXIS_QUAT_XYZW, quat_xyzw_gltf)
    return _xyzw_to_wxyz(q_mj_xyzw)


def mj_quat_to_gltf(quat_wxyz_mj: np.ndarray) -> np.ndarray:
    """``(..., 4)`` wxyz mujoco-frame orientation -> ``(..., 4)`` xyzw gltf-frame orientation."""
    quat_wxyz_mj = np.asarray(quat_wxyz_mj, dtype=np.float64)
    q_mj_xyzw = _wxyz_to_xyzw(quat_wxyz_mj)
    q_gltf_xyzw = _quat_mul_xyzw(_AXIS_QUAT_XYZW_CONJ, q_mj_xyzw)
    return q_gltf_xyzw


def gltf_pose_to_mj(position_gltf: np.ndarray, quat_xyzw_gltf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert one (or a batch of) (position, orientation) pose(s) from the
    gltfworld contract convention to MuJoCo's.

    Returns
    -------
    position_mj : (..., 3) float64, mujoco (x, y, z)
    quat_wxyz_mj : (..., 4) float64, mujoco (w, x, y, z)
    """
    return gltf_pos_to_mj(position_gltf), gltf_quat_to_mj(quat_xyzw_gltf)


def mj_pose_to_gltf(position_mj: np.ndarray, quat_wxyz_mj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`gltf_pose_to_mj`.

    Returns
    -------
    position_gltf : (..., 3) float64, gltf (x, y, z)
    quat_xyzw_gltf : (..., 4) float64, gltf (x, y, z, w)
    """
    return mj_pos_to_gltf(position_mj), mj_quat_to_gltf(quat_wxyz_mj)


# --- pose-array batch versions (StateSeries.poses is (T, N, 7): xyz + quat xyzw) --


def gltf_poses_to_mj(poses_gltf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(..., 7)`` gltf poses (xyz + quat xyzw) -> ``((..., 3), (..., 4))``
    mujoco (position, quat wxyz). Works for any leading batch shape,
    e.g. gltfworld's ``(T, N, 7)`` ``StateSeries.poses``, since every
    conversion in this module is already elementwise over trailing axes."""
    poses_gltf = np.asarray(poses_gltf, dtype=np.float64)
    return gltf_pose_to_mj(poses_gltf[..., 0:3], poses_gltf[..., 3:7])


def mj_poses_to_gltf(position_mj: np.ndarray, quat_wxyz_mj: np.ndarray) -> np.ndarray:
    """Inverse of :func:`gltf_poses_to_mj`: mujoco (position, quat wxyz) ->
    a single ``(..., 7)`` gltf pose array (xyz + quat xyzw)."""
    position_gltf, quat_xyzw_gltf = mj_pose_to_gltf(position_mj, quat_wxyz_mj)
    return np.concatenate([position_gltf, quat_xyzw_gltf], axis=-1)


# --- free-joint velocity conversion (see module docstring) -------------------


def mj_freejoint_vel_to_gltf_world(
    quat_wxyz_mj: np.ndarray, lin_vel_mj: np.ndarray, ang_vel_body_mj: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one free joint's recorded MuJoCo ``qvel`` (3 linear,
    world-frame + 3 angular, body-local-frame -- MuJoCo's convention, see
    module docstring) into the gltfworld contract's WORLD-frame linear and
    angular velocity, expressed in gltf axes.

    ``quat_wxyz_mj`` is the body's *current* orientation (needed to rotate
    the body-local angular velocity into MuJoCo's world frame before the
    axis remap). All arguments broadcast over any leading batch shape.

    Returns
    -------
    lin_vel_gltf_world : (..., 3) float64
    ang_vel_gltf_world : (..., 3) float64
    """
    quat_wxyz_mj = np.asarray(quat_wxyz_mj, dtype=np.float64)
    lin_vel_mj = np.asarray(lin_vel_mj, dtype=np.float64)
    ang_vel_body_mj = np.asarray(ang_vel_body_mj, dtype=np.float64)

    ang_vel_world_mj = _rotate_vector_wxyz(quat_wxyz_mj, ang_vel_body_mj)
    lin_vel_gltf = mj_vec_to_gltf(lin_vel_mj)
    ang_vel_gltf = mj_vec_to_gltf(ang_vel_world_mj)
    return lin_vel_gltf, ang_vel_gltf


def gltf_world_vel_to_mj_freejoint(
    quat_wxyz_mj: np.ndarray, lin_vel_gltf_world: np.ndarray, ang_vel_gltf_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`mj_freejoint_vel_to_gltf_world`: convert
    contract-convention WORLD-frame linear/angular velocity (gltf axes)
    into the ``(linear, angular)`` pair MuJoCo's free-joint ``qvel`` expects
    (linear world-frame, angular body-local-frame), given the body's
    current orientation ``quat_wxyz_mj``.

    Returns
    -------
    lin_vel_mj : (..., 3) float64
    ang_vel_body_mj : (..., 3) float64
    """
    quat_wxyz_mj = np.asarray(quat_wxyz_mj, dtype=np.float64)
    lin_vel_mj = gltf_vec_to_mj(lin_vel_gltf_world)
    ang_vel_world_mj = gltf_vec_to_mj(ang_vel_gltf_world)
    ang_vel_body_mj = _rotate_vector_wxyz(_quat_conj_wxyz(quat_wxyz_mj), ang_vel_world_mj)
    return lin_vel_mj, ang_vel_body_mj


# --- back-compat alias --------------------------------------------------------

# gltfworld.render.crosscheck's original name for gltf_pose_to_mj (V2); kept
# as an alias so that module and any external callers using the old name
# keep working after the V3 consolidation (see DESIGN.md).
gltf_pose_to_mujoco = gltf_pose_to_mj
