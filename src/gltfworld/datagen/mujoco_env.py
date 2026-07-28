"""MuJoCo MJCF construction + simulation for gltfworld episodes.

Two entry points:

- :func:`scene_to_mjcf` -- build an MJCF XML string mirroring a
  ``SceneState``'s objects (free-joint bodies for dynamic objects, static
  bodies welded to the world for static ones), given each object's initial
  pose.
- :func:`simulate` -- run that MJCF in MuJoCo and record a
  :class:`~gltfworld.scene.episode.StateSeries` (poses + world-frame
  linear/angular velocity, converted via :mod:`gltfworld.datagen.mj_convert`).

``SceneState``/``ObjectSpec`` (``gltfworld.scene.scene``) deliberately carry
no pose -- pose is a per-frame, per-episode quantity that lives in
``StateSeries`` (see DESIGN.md's architecture flow: one GLB = one
``SceneState`` + one ``StateSeries``). So both functions here take initial
poses/velocities as an explicit ``(N, 7)`` / ``(N, 3)`` array argument
alongside ``scene``, rather than reading them off ``scene`` itself -- a
small, documented deviation from the milestone spec text's one-argument
``simulate(scene, T, record_hz)`` shorthand (see DESIGN.md "wm-scenes-v1").
``gltfworld.datagen.sample.sample_scene`` is what actually produces this
initial state (a ``SampledScene`` bundling ``scene`` with initial
poses/velocities).

Known lossy/approximate mappings (documented per DESIGN.md):

- **Density, not mass**: dynamic object geoms are given a ``density``
  attribute (``mass / shape_volume``), not an explicit ``mass`` override,
  so MuJoCo derives mass *and* inertia tensor from the geometry the normal
  MJCF way. Round-trips back to (very nearly) the original ``ObjectSpec.mass``
  since :func:`_shape_volume` uses the exact analytic volume for every
  shape gltfworld supports.
- **Restitution -> solref**: MuJoCo's soft-contact model has no direct
  restitution coefficient. :func:`_restitution_to_solref` maps
  ``restitution`` to solref's standard ``(timeconst, dampratio)`` form via
  ``dampratio = 1 - restitution`` (critically damped, no bounce, at
  restitution 0; increasingly underdamped/"bouncier" as restitution rises).
  This is a documented, lossy approximation -- wm-scenes-v1 fixes
  restitution at a low 0.1, where it barely matters.
- **Cylinder axis convention (fixed in V3.1)**: MuJoCo's native
  ``type="cylinder"`` geom is symmetric about the geom's *local* Z axis and
  that can't be changed -- it's a builtin primitive type. gltfworld's
  contract convention (``ObjectSpec``, the ``KHR_implicit_shapes`` cylinder
  schema's own text "centered along the Y axis", and -- since V3.1 -- the
  actual trimesh-generated visual mesh) is symmetric about local **Y**.
  V3 shipped with the visual mesh left *wrong* (also Z-symmetric) so it
  would coincidentally match MuJoCo's native geom; that was a real,
  externally-visible interop bug (a spec-conformant ``KHR_implicit_shapes``
  reader reconstructs the collider centered on Y, so it would show the
  cylinder rotated 90 degrees relative to gltfworld's own renderer, even
  though gltfworld's *own* renderer and physics agreed with each other).
  V3.1 fixes the mesh (see ``gltfworld.scene.primitives``) and instead
  corrects the physics side here: every cylinder geom gets a fixed,
  body-orientation-independent local ``quat`` (see
  ``_CYLINDER_LOCAL_FIX_QUAT_WXYZ``, a -90 degree rotation about local X)
  that re-centers MuJoCo's native Z-symmetric shape onto local Y *within
  the body frame*, before the body's own orientation (composed via
  :mod:`gltfworld.datagen.mj_convert`, unchanged) is applied. Verified
  empirically and in ``tests/test_cylinder_axis.py``: with this local fix,
  a cylinder given an identity gltf orientation stands upright along
  mj's +Z, matching an upright Y-symmetric mesh under the same identity
  orientation; for random orientations, the geom's resulting world-space
  symmetry axis (mapped back through ``mj_vec_to_gltf``) matches
  ``R(q_gltf) @ (0, 1, 0)`` to float64 precision -- i.e. exactly the axis
  the corrected mesh sweeps out. ``gltfworld.render.crosscheck`` applies
  the identical local fix to its own (separate, static-frame-only) MJCF
  builder.
"""

from __future__ import annotations

import numpy as np

from gltfworld.datagen import mj_convert
from gltfworld.scene.episode import StateSeries
from gltfworld.scene.scene import ObjectSpec, SceneState

INTERNAL_HZ = 500.0
TIMESTEP = 1.0 / INTERNAL_HZ  # 0.002s, per the milestone spec


# --- shape volume (shared with gltfworld.datagen.sample for mass<->density) --


def shape_volume(shape: str, size) -> float:
    """Analytic volume of an ``ObjectSpec``-convention shape, used both to
    derive MJCF ``density`` from ``ObjectSpec.mass`` here and (inversely) to
    derive ``mass`` from a sampled density in ``gltfworld.datagen.sample``."""
    size = np.asarray(size, dtype=np.float64)
    if shape == "sphere":
        r = float(size[0])
        return (4.0 / 3.0) * np.pi * r**3
    if shape == "box":
        hx, hy, hz = (float(v) for v in size)
        return 8.0 * hx * hy * hz
    if shape == "cylinder":
        r = float(size[0])
        half_height = float(size[1])
        return np.pi * r * r * (2.0 * half_height)
    raise ValueError(f"unknown shape {shape!r}")


# MuJoCo's native `type="cylinder"` geom is symmetric about the geom's
# *local* Z axis and that can't be changed (it's a builtin primitive type,
# not something MJCF lets you parameterize). gltfworld's contract convention
# (ObjectSpec, the visual mesh, the KHR_implicit_shapes schema) is symmetric
# about local **Y** instead. Rather than leaving the mesh wrong to match
# MuJoCo (the old, since-removed V3 workaround), every cylinder geom gets a
# fixed, body-orientation-independent local `quat` -- a -90 degree rotation
# about local X -- that re-centers MuJoCo's native Z-symmetric shape onto
# local Y *within the body frame*, before the body's own (correctly
# *composed*, see mj_convert's module docstring) world orientation is
# applied on top. Verified empirically (see the V3.1 report): with this
# geom-local quat, a cylinder given `gltf_pose_to_mj`'s IDENTITY orientation
# stands upright along mj's +Z (matching an upright, Y-symmetric gltf mesh
# under the same identity orientation), and for a batch of random
# orientations the geom's resulting world-space symmetry axis (mapped back
# through `mj_vec_to_gltf`) matches `R(q_gltf) @ (0, 1, 0)` -- i.e. exactly
# the axis the *actual* (post-fix) trimesh-generated mesh sweeps out --
# to float64 precision.
_CYLINDER_LOCAL_FIX_QUAT_WXYZ = (
    f"{np.cos(-np.pi / 4.0):.17g} {np.sin(-np.pi / 4.0):.17g} 0 0"
)


def _geom_type_and_size(obj: ObjectSpec) -> tuple[str, str]:
    if obj.shape == "sphere":
        return "sphere", f"{obj.size[0]:.9g}"
    if obj.shape == "box":
        return "box", " ".join(f"{v:.9g}" for v in obj.size)
    if obj.shape == "cylinder":
        # size = [radius_bottom, half_height, radius_top]; MuJoCo's native
        # cylinder geom takes "radius half-height" along its LOCAL Z axis --
        # corrected to local Y (gltfworld's contract convention) by the
        # fixed geom-local quat added in `_body_xml`, see
        # `_CYLINDER_LOCAL_FIX_QUAT_WXYZ` above.
        return "cylinder", f"{obj.size[0]:.9g} {obj.size[1]:.9g}"
    raise ValueError(f"unsupported shape for MJCF: {obj.shape!r}")


def _geom_local_quat_attr(obj: ObjectSpec) -> str:
    """Extra ``quat="..."`` XML attribute text (with a leading space) for
    ``obj``'s geom, or ``""`` for shapes that need no local-frame
    correction (sphere/box are symmetric the same way in both engines
    already -- see module docstring)."""
    if obj.shape == "cylinder":
        return f' quat="{_CYLINDER_LOCAL_FIX_QUAT_WXYZ}"'
    return ""


def _restitution_to_solref(restitution: float, timestep: float) -> str:
    """LOSSY, documented approximation (see module docstring): map
    ``restitution`` (0 = inelastic, 1 = perfectly elastic) onto solref's
    standard positive ``(timeconst, dampratio)`` form. ``timeconst`` is
    kept at MuJoCo's recommended minimum (2x the physics timestep) rather
    than a larger fixed floor -- stiffer settles faster and with less
    steady-state penetration, at the cost of using more of the solver's
    stability margin (still comfortably stable at wm-scenes-v1's masses/
    speeds -- see the V3 report)."""
    dampratio = float(np.clip(1.0 - restitution, 0.1, 1.0))
    timeconst = 2.0 * timestep
    return f"{timeconst:.6g} {dampratio:.6g}"


# Tighter-than-default contact impedance (MuJoCo default solimp is
# "0.9 0.95 0.001 0.5 2"): raises the low/high impedance bounds so contacts
# approach non-penetration more closely at equilibrium. Still finite
# stiffness -- MuJoCo's soft-contact model always allows *some* penetration,
# see module docstring and the V3 report re: transient impact penetration.
_SOLIMP = "0.97 0.999 0.0001 0.5 2"


def _body_xml(obj: ObjectSpec, pos_mj: np.ndarray, quat_wxyz_mj: np.ndarray, timestep: float) -> str:
    geom_type, size_str = _geom_type_and_size(obj)
    pos_str = " ".join(f"{v:.9g}" for v in pos_mj)
    quat_str = " ".join(f"{v:.9g}" for v in quat_wxyz_mj)
    rgba_str = " ".join(f"{v:.9g}" for v in obj.color)
    geom_local_quat = _geom_local_quat_attr(obj)
    name = f"obj_{obj.object_id}"

    if obj.is_static:
        return (
            f'<body name="{name}" pos="{pos_str}" quat="{quat_str}">'
            f'<geom name="{name}" type="{geom_type}" size="{size_str}"{geom_local_quat} rgba="{rgba_str}"/>'
            f"</body>"
        )

    volume = shape_volume(obj.shape, obj.size)
    density = obj.mass / volume if volume > 0 else 1000.0
    friction = float(max(obj.friction, 1e-4))
    solref = _restitution_to_solref(obj.restitution, timestep)
    return (
        f'<body name="{name}" pos="{pos_str}" quat="{quat_str}">'
        f'<freejoint name="free_{obj.object_id}"/>'
        f'<geom name="{name}" type="{geom_type}" size="{size_str}" density="{density:.9g}" '
        f'friction="{friction:.9g} 0.005 0.0001" solref="{solref}" solimp="{_SOLIMP}"{geom_local_quat} rgba="{rgba_str}"/>'
        f"</body>"
    )


def scene_to_mjcf(scene: SceneState, initial_poses: np.ndarray, *, timestep: float = TIMESTEP) -> str:
    """Build an MJCF XML string for ``scene``, one body per
    ``scene.objects[i]`` placed at ``initial_poses[i]`` (``(N, 7)`` xyz +
    quat xyzw, gltfworld contract convention -- converted to MuJoCo
    convention via :mod:`gltfworld.datagen.mj_convert`).

    Dynamic (``not is_static``) objects get a ``<freejoint>`` (named
    ``free_{object_id}``) and a density-based geom (mass implied by
    ``ObjectSpec.mass`` and the shape's analytic volume, friction from
    ``ObjectSpec.friction``, restitution lossily mapped to ``solref`` --
    see module docstring). Static objects (including the scene's ground
    plane, modeled as an ordinary static ``ObjectSpec`` with
    ``category="ground"``) are welded directly to the world at their given
    pose -- no joint, matching ``KHR_physics_rigid_bodies``' own "no
    ``motion`` = immovable" semantics.
    """
    initial_poses = np.asarray(initial_poses, dtype=np.float64)
    if initial_poses.shape != (len(scene.objects), 7):
        raise ValueError(
            f"initial_poses must have shape ({len(scene.objects)}, 7), got {initial_poses.shape}"
        )

    gravity_mj = mj_convert.gltf_vec_to_mj(scene.gravity)

    body_xmls = []
    for i, obj in enumerate(scene.objects):
        pos_mj, quat_wxyz_mj = mj_convert.gltf_pose_to_mj(initial_poses[i, 0:3], initial_poses[i, 3:7])
        body_xmls.append(_body_xml(obj, pos_mj, quat_wxyz_mj, timestep))

    gravity_str = " ".join(f"{v:.9g}" for v in gravity_mj)

    return f"""<mujoco>
  <option timestep="{timestep:.9g}" gravity="{gravity_str}"/>
  <worldbody>
    {"".join(body_xmls)}
  </worldbody>
</mujoco>
"""


def simulate(
    scene: SceneState,
    initial_poses: np.ndarray,
    T: int,
    record_hz: float = 30.0,
    *,
    initial_lin_vel: np.ndarray | None = None,
    initial_ang_vel: np.ndarray | None = None,
    internal_hz: float = INTERNAL_HZ,
) -> StateSeries:
    """Simulate ``scene`` (placed at ``initial_poses``/``initial_lin_vel``/
    ``initial_ang_vel``) in MuJoCo at ``internal_hz`` (500 Hz, per the
    milestone spec) and record ``T`` frames at (approximately) ``record_hz``
    into a :class:`~gltfworld.scene.episode.StateSeries`.

    Frame 0 of the returned series is the *initial* state (before any
    stepping); frame ``t`` is the state after ``t * steps_per_record``
    internal steps, where ``steps_per_record = round(internal_hz /
    record_hz)``. Note ``internal_hz`` isn't always an exact multiple of
    ``record_hz`` (e.g. 500/30 = 16.67): the actual recorded frame period is
    ``steps_per_record / internal_hz`` seconds, which the returned series'
    ``times``/``scene.dt`` reflect exactly -- not the nominal ``1 /
    record_hz`` -- so downstream consumers never see a ``dt`` that doesn't
    match what was actually simulated. For record_hz values that *do*
    divide 500 evenly (e.g. 25, 50, 100), this rounding is exact.

    Every per-object array is converted between MuJoCo's free-joint qvel
    convention (3 linear world-frame + 3 angular body-local-frame) and the
    contract's world-frame linear/angular velocity via
    :mod:`gltfworld.datagen.mj_convert` -- see its module docstring for the
    empirically-verified frame convention.
    """
    import mujoco

    n = len(scene.objects)
    initial_poses = np.asarray(initial_poses, dtype=np.float64)
    if initial_poses.shape != (n, 7):
        raise ValueError(f"initial_poses must have shape ({n}, 7), got {initial_poses.shape}")
    if initial_lin_vel is None:
        initial_lin_vel = np.zeros((n, 3))
    if initial_ang_vel is None:
        initial_ang_vel = np.zeros((n, 3))
    initial_lin_vel = np.asarray(initial_lin_vel, dtype=np.float64)
    initial_ang_vel = np.asarray(initial_ang_vel, dtype=np.float64)

    timestep = 1.0 / internal_hz
    mjcf = scene_to_mjcf(scene, initial_poses, timestep=timestep)
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)

    dynamic_indices = [i for i, obj in enumerate(scene.objects) if not obj.is_static]
    joint_id = {
        i: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"free_{scene.objects[i].object_id}")
        for i in dynamic_indices
    }

    for i in dynamic_indices:
        jid = joint_id[i]
        qpos_adr = model.jnt_qposadr[jid]
        qvel_adr = model.jnt_dofadr[jid]

        pos_mj, quat_wxyz_mj = mj_convert.gltf_pose_to_mj(initial_poses[i, 0:3], initial_poses[i, 3:7])
        data.qpos[qpos_adr : qpos_adr + 3] = pos_mj
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat_wxyz_mj

        lin_mj, ang_body_mj = mj_convert.gltf_world_vel_to_mj_freejoint(
            quat_wxyz_mj, initial_lin_vel[i], initial_ang_vel[i]
        )
        data.qvel[qvel_adr : qvel_adr + 3] = lin_mj
        data.qvel[qvel_adr + 3 : qvel_adr + 6] = ang_body_mj

    mujoco.mj_forward(model, data)

    steps_per_record = max(1, round(internal_hz / record_hz))
    dt_actual = steps_per_record / internal_hz

    times = (np.arange(T, dtype=np.float64) * dt_actual).astype(np.float32)
    poses = np.zeros((T, n, 7), dtype=np.float32)
    lin_vel = np.zeros((T, n, 3), dtype=np.float32)
    ang_vel = np.zeros((T, n, 3), dtype=np.float32)

    def record(t_index: int) -> None:
        for i, obj in enumerate(scene.objects):
            if obj.is_static:
                poses[t_index, i] = initial_poses[i]
                continue
            jid = joint_id[i]
            qpos_adr = model.jnt_qposadr[jid]
            qvel_adr = model.jnt_dofadr[jid]

            pos_mj = data.qpos[qpos_adr : qpos_adr + 3]
            quat_wxyz_mj = data.qpos[qpos_adr + 3 : qpos_adr + 7]
            lin_mj = data.qvel[qvel_adr : qvel_adr + 3]
            ang_body_mj = data.qvel[qvel_adr + 3 : qvel_adr + 6]

            poses[t_index, i] = mj_convert.mj_poses_to_gltf(pos_mj, quat_wxyz_mj)
            lin_g, ang_g = mj_convert.mj_freejoint_vel_to_gltf_world(quat_wxyz_mj, lin_mj, ang_body_mj)
            lin_vel[t_index, i] = lin_g
            ang_vel[t_index, i] = ang_g

    record(0)
    for t_index in range(1, T):
        for _ in range(steps_per_record):
            mujoco.mj_step(model, data)
        record(t_index)

    scene.dt = float(dt_actual)

    return StateSeries(times=times, poses=poses, lin_vel=lin_vel, ang_vel=ang_vel)
