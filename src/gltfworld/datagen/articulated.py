"""``wm-articulated-v1`` (V9-prep): parametric MJCF generators for articulated
furniture -- a cabinet with a hinged door, or a chest/table with a sliding
drawer -- each with a cosmetic handle, exercising the
``KHR_physics_rigid_bodies`` joint transport added in
``gltfworld.ext.khr_physics`` (see its module docstring, "V9-prep: joints").

Scene layout (object ids, matching ``ArticulatedSpec``'s ``base``/``part``/
``handle`` roles):

- ``0`` ground (static box, category ``"ground"``)
- ``1`` base (static box, category ``"cabinet"`` -- the taxonomy in
  DESIGN.md's V9-prep section doesn't distinguish "chest" from "cabinet";
  both articulated kinds use the same base category)
- ``2`` part (dynamic box, category ``"door"`` or ``"drawer"``) -- the one
  moving, jointed body
- ``3`` handle (dynamic box, category ``"handle"``) -- rigidly welded to
  ``part`` (see below), not independently simulated

Only ``part`` gets a real MuJoCo joint (hinge or slide, 1 DOF); ``base`` and
``ground`` are ordinary static bodies (no joint, welded to the world, same
convention as ``gltfworld.datagen.mujoco_env``). ``base`` is **not** a MJCF
parent of ``part`` -- since ``base`` never moves, "part's motion relative to
base" and "part's motion relative to world" are kinematically identical, so
``part`` is placed directly under MuJoCo's ``worldbody`` (flat, matching
every other gltfworld body) with its hinge/slide joint's ``pos``/``axis``
given in ``part``'s own body-local frame (MJCF's standard convention) --
this sidesteps needing any new MJCF body-nesting machinery.

``handle`` is not simulated at all: it's rigidly welded to ``part`` at a
fixed local offset, so its per-frame world pose is *derived* directly from
``part``'s recorded trajectory (``handle_pose(t) = part_pose(t) ∘
(handle_local_offset, identity)``) rather than being an independent MuJoCo
body -- a documented simplification (see DESIGN.md's V9-prep gap notes:
this also means the KHR encoding doesn't describe the handle's attachment
via a joint/weld at all, just via its always-consistent animated pose).

**Axis coverage over realism**: ``axis`` (which world-aligned X/Y/Z axis the
joint acts about/along) is drawn uniformly from ``{0, 1, 2}`` by default,
even though a "flap" door hinged on a horizontal axis or a "drawer" that
lifts vertically are less common real-world furniture archetypes than a
vertical-hinge door / horizontal-slide drawer. This is deliberate: the
sampler's job for this milestone is exercising the KHR joint encoding's
``angularAxes``/``linearAxes`` across all three axes, not modeling
photorealistic furniture. Physics-sanity tests that need a reliably
monotonic, gravity-torque-free trajectory pin ``axis`` explicitly (a
vertical hinge axis has zero gravity torque about it; gravity is exactly
along -Y) rather than relying on the general sampler's random axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gltfworld.datagen import mj_convert
from gltfworld.datagen.mujoco_env import shape_volume
from gltfworld.scene.episode import StateSeries
from gltfworld.scene.scene import ArticulatedSpec, CameraSpec, LightSpec, ObjectSpec, SceneState

SCENE_VERSION = "wm-articulated-v1"

INTERNAL_HZ = 500.0

_GROUND_HALF_EXTENTS = np.array([1.5, 0.1, 1.5], dtype=np.float32)
_GROUND_TOP_Y = 0.0
_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

_KINDS = ("door", "drawer")

# Damping is derived per-scene from the part's own inertia/mass (not a fixed
# constant -- see sample_articulated_scene), targeting a residual-bounce
# decay time constant of about `_DAMPING_TAU` seconds: `damping = I / tau`
# (revolute) or `damping = mass / tau` (prismatic), so that a fixed *decay
# rate* (not a fixed damping coefficient) applies regardless of how the
# sampler's mass/size randomization landed. Found the hard way (see
# DESIGN.md's V9-prep report): a fixed damping *coefficient* is only "light"
# relative to some particular inertia -- applied to a much smaller or larger
# door, the same number is either so heavy the door never reaches its limit
# under a same-formula push, or so light the post-limit-bounce settling tail
# stretches over several seconds, well past a short recorded episode.
_DAMPING_TAU = 0.3  # seconds

# A brief, bounded-duration push (not held for the whole episode -- see
# module docstring) starting shortly after t=0; magnitude derived per-scene
# from mass/geometry (see sample_articulated_scene), not fixed.
_PUSH_START_TIME = 0.05
_PUSH_DURATION = 0.25
_PUSH_SAFETY_FACTOR = 1.3


def _conj_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([w, -x, -y, -z])


def _rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w = q[0]
    qxyz = q[1:4]
    t = 2.0 * np.cross(qxyz, v)
    return v + w * t + np.cross(qxyz, t)


def _unit_axis(axis: int) -> np.ndarray:
    v = np.zeros(3)
    v[axis] = 1.0
    return v


@dataclass
class SampledArticulatedScene:
    """A ``wm-articulated-v1`` sample: a static ``SceneState`` (one
    ``ArticulatedSpec``) plus everything ``simulate_articulated`` needs to
    seed and drive MuJoCo -- mirrors ``gltfworld.datagen.sample.SampledScene``'s
    role for the flat-object distribution."""

    scene: SceneState
    initial_poses: np.ndarray  # (N, 7) float32, gltf convention; row order matches scene.objects
    initial_joint_pos: float  # radians (revolute) or meters (prismatic)
    handle_local_offset: np.ndarray  # (3,) float32, part-local frame
    push_force: float  # generalized force (N, prismatic) or torque (N*m, revolute); positive = toward `max`
    push_start_time: float  # seconds
    push_end_time: float  # seconds
    damping: float  # MuJoCo joint damping, units per joint_type (see module docstring)


def sample_articulated_scene(
    seed: int,
    *,
    kind: str | None = None,
    axis: int | None = None,
) -> SampledArticulatedScene:
    """Sample one ``wm-articulated-v1`` scene, deterministic in ``seed``.

    ``kind`` (``"door"``/``"drawer"``) and ``axis`` (0/1/2) default to
    random (see module docstring on axis coverage); pass either explicitly
    to pin it (used by the physics-sanity tests, which need a reliably
    monotonic trajectory rather than the full random-axis distribution).
    """
    rng = np.random.default_rng(seed)
    kind = kind if kind is not None else str(rng.choice(_KINDS))
    if kind not in _KINDS:
        raise ValueError(f"unknown kind {kind!r}, expected one of {_KINDS}")
    axis = int(axis) if axis is not None else int(rng.integers(0, 3))
    sweep_axis = (axis + 1) % 3  # the axis the part extends along at rest, see module docstring
    thickness_axis = (axis + 2) % 3

    ground = ObjectSpec(
        object_id=0,
        shape="box",
        size=_GROUND_HALF_EXTENTS.copy(),
        color=np.array([0.55, 0.55, 0.58, 1.0], dtype=np.float32),
        roughness=0.9,
        metallic=0.0,
        mass=1000.0,
        friction=0.8,
        restitution=0.1,
        is_static=True,
        category="ground",
    )

    base_half = np.array(
        [rng.uniform(0.25, 0.4), rng.uniform(0.3, 0.45), rng.uniform(0.25, 0.4)], dtype=np.float32
    )
    base_pos = np.array([0.0, _GROUND_TOP_Y + base_half[1], 0.0], dtype=np.float32)
    base = ObjectSpec(
        object_id=1,
        shape="box",
        size=base_half,
        color=np.array([0.5, 0.35, 0.2, 1.0], dtype=np.float32),
        roughness=0.6,
        metallic=0.0,
        mass=float(rng.uniform(20.0, 60.0)),
        friction=float(rng.uniform(0.4, 0.9)),
        restitution=0.05,
        is_static=True,
        category="cabinet",
    )

    # Anchor: a point on base's own bounding surface along `sweep_axis`
    # (the edge/face the part hinges/slides from), at base's own center
    # along the other two axes.
    anchor = base_pos.copy()
    anchor[sweep_axis] += float(base_half[sweep_axis])

    part_extent = float(rng.uniform(0.15, 0.35))  # size along sweep_axis
    part_span = float(rng.uniform(0.9, 1.0) * base_half[axis] if kind == "door" else rng.uniform(0.7, 0.95) * base_half[axis])
    part_thickness = float(rng.uniform(0.015, 0.03))

    part_size = np.zeros(3, dtype=np.float32)
    part_size[sweep_axis] = part_extent
    part_size[axis] = max(part_span, 0.05)
    part_size[thickness_axis] = part_thickness

    if kind == "door":
        min_v, max_v = 0.0, float(rng.uniform(1.0, 1.9))  # ~57-109 degrees
        initial_joint_pos = float(rng.uniform(min_v, 0.35 * max_v))
        mass = float(rng.uniform(2.0, 8.0))
        category = "door"
    else:
        min_v, max_v = 0.0, float(rng.uniform(0.15, 0.35))
        initial_joint_pos = float(rng.uniform(min_v, 0.3 * max_v))
        mass = float(rng.uniform(1.5, 5.0))
        category = "drawer"

    # Rest pose (qpos == 0 reference, see gltfworld.ext.khr_physics /
    # ArticulatedSpec docstrings): part sits just outside the anchor,
    # extending away from it along +sweep_axis, identity orientation.
    part_pos = anchor.copy()
    part_pos[sweep_axis] += part_size[sweep_axis]
    part = ObjectSpec(
        object_id=2,
        shape="box",
        size=part_size,
        color=np.array([0.75, 0.6, 0.4, 1.0], dtype=np.float32),
        roughness=0.5,
        metallic=0.0,
        mass=mass,
        friction=float(rng.uniform(0.3, 0.7)),
        restitution=0.05,
        is_static=False,
        category=category,
    )

    # Push force/torque, sized from the sampled mass/geometry (not a fixed
    # constant) so a brief, bounded-duration push (see PUSH_DURATION) reliably
    # drives the joint to (or past, briefly -- see the physics-sanity report
    # in DESIGN.md) its limit regardless of how the rest of the sampler's
    # randomization landed, rather than sometimes under- or over-shooting by
    # an arbitrary amount. Rough kinematic estimate assuming the push is the
    # only force (ballpark, not exact -- damping/limit-constraint effects
    # during the push itself are deliberately not modeled here, just
    # compensated for with `_PUSH_SAFETY_FACTOR`):
    #   revolute: treat the part as a rod pivoting about one end,
    #     I ~= (1/3) * mass * (2 * part_extent)^2, and solve
    #     max_v ~= 0.5 * (torque / I) * PUSH_DURATION^2 for torque.
    #   prismatic: F = mass * a, solve max_v ~= 0.5 * (F / mass) * PUSH_DURATION^2.
    if kind == "door":
        lever = 2.0 * part_extent
        moment_of_inertia = (1.0 / 3.0) * mass * lever * lever
        push_force = _PUSH_SAFETY_FACTOR * 2.0 * moment_of_inertia * max_v / (_PUSH_DURATION**2)
        damping = moment_of_inertia / _DAMPING_TAU
    else:
        push_force = _PUSH_SAFETY_FACTOR * 2.0 * mass * max_v / (_PUSH_DURATION**2)
        damping = mass / _DAMPING_TAU
    push_force = float(push_force * rng.uniform(0.9, 1.15))

    handle_size = np.array(
        [rng.uniform(0.015, 0.03)] * 3, dtype=np.float32
    )
    # Handle sits near the part's outer edge (far from the anchor along
    # sweep_axis) and pokes out a little along the thickness axis.
    handle_local_offset = np.zeros(3, dtype=np.float32)
    handle_local_offset[sweep_axis] = 0.75 * part_size[sweep_axis]
    handle_local_offset[thickness_axis] = part_size[thickness_axis] + handle_size[thickness_axis]
    handle_pos0 = part_pos + handle_local_offset
    handle = ObjectSpec(
        object_id=3,
        shape="box",
        size=handle_size,
        color=np.array([0.15, 0.15, 0.15, 1.0], dtype=np.float32),
        roughness=0.3,
        metallic=0.6,
        mass=float(rng.uniform(0.05, 0.2)),
        friction=0.5,
        restitution=0.05,
        is_static=False,
        category="handle",
    )

    articulation = ArticulatedSpec(
        joint_index=0,
        base_object_id=1,
        part_object_id=2,
        joint_type="revolute" if kind == "door" else "prismatic",
        axis=axis,
        min=min_v,
        max=max_v,
        anchor=anchor,
        part_labels=(category,),
        affordances=("openable",),
        handle_object_id=3,
        handle_labels=("handle",),
        handle_affordances=("pullable",),
        base_labels=("cabinet",),
    )

    camera = CameraSpec(
        position=np.array([1.2, 1.4, 2.2], dtype=np.float32),
        rotation=_IDENTITY_QUAT.copy(),
        yfov=float(np.radians(55.0)),
        znear=0.05,
        zfar=50.0,
        aspect=1.0,
    )
    lights = [
        LightSpec(
            type="directional",
            color=np.array([1.0, 1.0, 0.97], dtype=np.float32),
            intensity=3.0,
            rotation=_IDENTITY_QUAT.copy(),
        ),
    ]

    scene = SceneState(
        objects=[ground, base, part, handle],
        camera=camera,
        lights=lights,
        gravity=np.array([0.0, -9.81, 0.0], dtype=np.float32),
        dt=1.0 / 30.0,
        seed=seed,
        scene_version=SCENE_VERSION,
        articulations=[articulation],
    )

    initial_poses = np.zeros((4, 7), dtype=np.float32)
    ground_pose = np.zeros(7, dtype=np.float32)
    ground_pose[1] = _GROUND_TOP_Y - float(_GROUND_HALF_EXTENTS[1])
    ground_pose[3:7] = _IDENTITY_QUAT
    initial_poses[0] = ground_pose
    initial_poses[1, 0:3] = base_pos
    initial_poses[1, 3:7] = _IDENTITY_QUAT
    initial_poses[2, 0:3] = part_pos
    initial_poses[2, 3:7] = _IDENTITY_QUAT
    initial_poses[3, 0:3] = handle_pos0
    initial_poses[3, 3:7] = _IDENTITY_QUAT

    return SampledArticulatedScene(
        scene=scene,
        initial_poses=initial_poses,
        initial_joint_pos=initial_joint_pos,
        handle_local_offset=handle_local_offset,
        push_force=push_force,
        push_start_time=_PUSH_START_TIME,
        push_end_time=_PUSH_START_TIME + _PUSH_DURATION,
        damping=damping,
    )


def _articulated_mjcf(sampled: SampledArticulatedScene, timestep: float) -> str:
    scene = sampled.scene
    art = scene.articulations[0]
    initial_poses = sampled.initial_poses

    gravity_mj = mj_convert.gltf_vec_to_mj(scene.gravity)
    gravity_str = " ".join(f"{v:.9g}" for v in gravity_mj)

    body_xmls = []
    for i, obj in enumerate(scene.objects):
        if obj.object_id == art.part_object_id:
            continue  # handled below, needs the joint tag
        pos_mj, quat_wxyz_mj = mj_convert.gltf_pose_to_mj(initial_poses[i, 0:3], initial_poses[i, 3:7])
        pos_str = " ".join(f"{v:.9g}" for v in pos_mj)
        quat_str = " ".join(f"{v:.9g}" for v in quat_wxyz_mj)
        rgba_str = " ".join(f"{v:.9g}" for v in obj.color)
        name = f"obj_{obj.object_id}"
        body_xmls.append(
            f'<body name="{name}" pos="{pos_str}" quat="{quat_str}">'
            f'<geom name="{name}" type="box" size="{obj.size[0]:.9g} {obj.size[1]:.9g} {obj.size[2]:.9g}" '
            f'rgba="{rgba_str}"/>'
            f"</body>"
        )

    part_index = next(i for i, obj in enumerate(scene.objects) if obj.object_id == art.part_object_id)
    part_obj = scene.objects[part_index]
    part_pos_mj, part_quat_wxyz_mj = mj_convert.gltf_pose_to_mj(
        initial_poses[part_index, 0:3], initial_poses[part_index, 3:7]
    )
    anchor_mj = mj_convert.gltf_pos_to_mj(art.anchor)
    # MJCF's <joint pos>/<joint axis> are given in the BODY's own local
    # frame (i.e. after undoing the body's world orientation), not the
    # world frame -- unlike `anchor_mj`/`axis_mj` above, which are still
    # world-frame quantities at this point. Un-rotate both by the
    # conjugate of the part body's own MJ-world orientation.
    anchor_local_mj = _rotate_wxyz(_conj_wxyz(part_quat_wxyz_mj), anchor_mj - part_pos_mj)
    axis_world_mj = mj_convert.gltf_vec_to_mj(_unit_axis(art.axis))
    axis_mj = _rotate_wxyz(_conj_wxyz(part_quat_wxyz_mj), axis_world_mj)

    joint_type = "hinge" if art.joint_type == "revolute" else "slide"
    volume = shape_volume("box", part_obj.size)
    density = part_obj.mass / volume if volume > 0 else 1000.0

    pos_str = " ".join(f"{v:.9g}" for v in part_pos_mj)
    quat_str = " ".join(f"{v:.9g}" for v in part_quat_wxyz_mj)
    joint_pos_str = " ".join(f"{v:.9g}" for v in anchor_local_mj)
    joint_axis_str = " ".join(f"{v:.9g}" for v in axis_mj)
    rgba_str = " ".join(f"{v:.9g}" for v in part_obj.color)
    name = f"obj_{part_obj.object_id}"

    body_xmls.append(
        f'<body name="{name}" pos="{pos_str}" quat="{quat_str}">'
        f'<joint name="art_joint" type="{joint_type}" pos="{joint_pos_str}" axis="{joint_axis_str}" '
        f'limited="true" range="{art.min:.9g} {art.max:.9g}" damping="{sampled.damping:.9g}"/>'
        # contype/conaffinity=0: the part deliberately does not participate
        # in contact collision (see module docstring -- its rest pose
        # overlaps the base's static geom by construction, and this
        # milestone's physics-sanity checks are about joint-constrained
        # motion, not contact response).
        f'<geom name="{name}" type="box" size="{part_obj.size[0]:.9g} {part_obj.size[1]:.9g} {part_obj.size[2]:.9g}" '
        f'density="{density:.9g}" contype="0" conaffinity="0" rgba="{rgba_str}"/>'
        f"</body>"
    )

    # compiler angle="radian": MJCF's *default* is degrees for angle-valued
    # attributes (e.g. a hinge joint's "range") -- found the hard way (see
    # DESIGN.md's V9-prep report): without this, `range="{min} {max}"`
    # (already radians, matching KHR_physics_rigid_bodies.joint.limit's own
    # unit convention) was silently reinterpreted as *degrees*, compiling a
    # door meant to swing ~1.5 radians into a joint limited to ~1.5
    # *degrees* (0.0262 rad) -- it hit that tiny limit almost immediately
    # and the "monotonic open, settle within limits" physics-sanity check
    # was actually observing limit-constraint equilibrium at a wildly wrong
    # limit, not real swinging. `gltfworld.datagen.mujoco_env`'s existing
    # MJCF never hit this because it only ever uses `<freejoint>` (no
    # angle-valued attributes at all).
    return f"""<mujoco>
  <compiler angle="radian"/>
  <option timestep="{timestep:.9g}" gravity="{gravity_str}"/>
  <worldbody>
    {"".join(body_xmls)}
  </worldbody>
</mujoco>
"""


def simulate_articulated(
    sampled: SampledArticulatedScene,
    T: int,
    record_hz: float = 30.0,
    *,
    internal_hz: float = INTERNAL_HZ,
) -> StateSeries:
    """Simulate ``sampled`` in MuJoCo and record ``T`` frames at
    (approximately) ``record_hz`` into a
    :class:`~gltfworld.scene.episode.StateSeries` with ``poses`` (4 objects:
    ground/base/part/handle) and ``joint_pos`` ``(T, 1)``.

    A generalized force/torque of ``sampled.push_force`` is applied to the
    joint's single DOF for ``[push_start_time, push_end_time)``, then
    released (0 applied force) for the remainder -- "a scripted push
    actuation" (see DESIGN.md's V9-prep MJCF<->KHR mapping section), letting
    the joint continue coasting/settling under its own damping and limit
    stop rather than staying externally driven for the whole episode.
    """
    import mujoco

    scene = sampled.scene
    art = scene.articulations[0]
    n = len(scene.objects)

    timestep = 1.0 / internal_hz
    mjcf = _articulated_mjcf(sampled, timestep=timestep)
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "art_joint")
    qpos_adr = model.jnt_qposadr[joint_id]
    qvel_adr = model.jnt_dofadr[joint_id]
    part_index = next(i for i, obj in enumerate(scene.objects) if obj.object_id == art.part_object_id)
    part_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obj_{art.part_object_id}")

    data.qpos[qpos_adr] = sampled.initial_joint_pos
    mujoco.mj_forward(model, data)

    steps_per_record = max(1, round(internal_hz / record_hz))
    dt_actual = steps_per_record / internal_hz

    times = (np.arange(T, dtype=np.float64) * dt_actual).astype(np.float32)
    poses = np.zeros((T, n, 7), dtype=np.float32)
    joint_pos = np.zeros((T, 1), dtype=np.float32)

    def record(t_index: int, sim_time: float) -> None:
        for i, obj in enumerate(scene.objects):
            if obj.object_id == art.part_object_id:
                pos_mj = data.xpos[part_body_id]
                quat_wxyz_mj = data.xquat[part_body_id]
                poses[t_index, i] = mj_convert.mj_poses_to_gltf(pos_mj, quat_wxyz_mj)
            elif obj.object_id == art.handle_object_id:
                # Derived, not simulated (see module docstring): rigidly
                # welded to `part` at a fixed local offset.
                part_pos_gltf = poses[t_index, part_index, 0:3]
                part_rot_gltf = poses[t_index, part_index, 3:7]
                poses[t_index, i, 0:3] = part_pos_gltf + _quat_rotate_xyzw_local(
                    part_rot_gltf, sampled.handle_local_offset
                )
                poses[t_index, i, 3:7] = part_rot_gltf
            else:
                poses[t_index, i] = sampled.initial_poses[i]
        joint_pos[t_index, 0] = data.qpos[qpos_adr]

    record(0, 0.0)
    sim_time = 0.0
    for t_index in range(1, T):
        for _ in range(steps_per_record):
            if sampled.push_start_time <= sim_time < sampled.push_end_time:
                data.qfrc_applied[qvel_adr] = sampled.push_force
            else:
                data.qfrc_applied[qvel_adr] = 0.0
            mujoco.mj_step(model, data)
            sim_time += timestep
        record(t_index, sim_time)

    scene.dt = float(dt_actual)

    return StateSeries(times=times, poses=poses, joint_pos=joint_pos)


def _quat_rotate_xyzw_local(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` (3,) by unit quaternion ``q_xyzw`` (4,) xyzw
    (gltf convention) -- local copy to keep this module dependency-free of
    ``gltfworld.scene.convert`` (which itself has an identical helper for
    the same reason: this operation is a tiny, self-contained numeric
    primitive duplicated rather than imported across module boundaries that
    otherwise have no reason to depend on each other)."""
    q = np.asarray(q_xyzw, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    qxyz = q[0:3]
    w = q[3]
    t = 2.0 * np.cross(qxyz, v)
    return v + w * t + np.cross(qxyz, t)
