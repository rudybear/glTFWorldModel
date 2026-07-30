"""HDF5 -> glTF transport conversion for one Physion trial (V8's "state-based
track", option (b) from ``docs/PHYSION.md``).

This is deliberately a *separate* encoder from ``gltfworld.scene.convert``
(THE wm-scenes-v1 transport codec), not a variant call into it, because a
Physion trial's geometry doesn't fit that codec's core assumption: every
``ObjectSpec`` there is one of exactly three procedural primitives
(sphere/box/cylinder, ``gltfworld.scene.primitives.mesh_for``). Physion's
HDF5 tier carries *real* per-object mesh geometry (arbitrary vertex/face
data, see ``docs/PHYSION.md``'s schema section) -- exactly the "their
meshes -> real glTF meshes" half of this milestone's gap experiment. So:

- **Visual mesh** (POSITION/NORMAL/indices accessors): the *real* HDF5
  vertices/faces, scaled by the object's own ``static/scale``. HDF5 carries
  no normals at all; :func:`compute_vertex_normals` derives them
  (area-weighted face-normal average per vertex), a real, lossy
  approximation relative to whatever the original TDW asset's authored
  normals were (unrecoverable from convex/triangulated face data alone).
- **Physics** (``KHR_physics_rigid_bodies``/``KHR_implicit_shapes``): still
  goes through the *existing*, unmodified ``gltfworld.ext.khr_physics``
  codec, via a genuine ``ObjectSpec`` whose ``shape``/``size`` are a
  *approximation* of the real mesh's AABB (:func:`classify_bounding_shape`
  -- sphere if roughly isotropic, box otherwise; our pinned
  ``KHR_implicit_shapes`` subset has no generic/convex mesh collider type at
  all, so *some* shape-type approximation is unavoidable here regardless of
  transport). Mass/friction/restitution reuse real HDF5 static metadata.
- **Poses/velocities** (animation + ``RWM_state_series``): real per-frame
  HDF5 positions/rotations/velocities/angular_velocities -- no
  finite-differencing needed (unlike a `wm-scenes-v1` episode lacking a
  velocity series), since TDW's own physics engine writes these directly.
- **Semantics** (``extras.rwm``): ``category`` = the real TDW model name;
  an added ``role`` field (agent/patient/probe/distractor/occluder/object)
  -- an extension of ``extras.rwm``'s documented shape, in the same spirit
  as this project's existing mass/is_static/gravity deviations (see
  DESIGN.md's "Documented deviations from the milestone spec text").

Every impedance mismatch actually hit while writing this (units, axis/
handedness conventions, mesh-pivot vs. geometric-center, the two-friction-
coefficients-into-one collapse, ...) is recorded in docs/PHYSION.md's
"Physion conversion findings" section -- this is the primary gap-report
evidence this milestone produces, per DESIGN.md's V9 scope.

Because the resulting ``Episode`` uses genuine ``ObjectSpec``/``SceneState``/
``StateSeries`` containers (just with an approximated shape/size and real
mesh data bolted on for the visual geometry only), the *existing*,
unmodified ``gltfworld.scene.convert.episode_from_gltf`` decodes a converted
GLB right back into a full ``Episode`` -- poses/velocities/physics metadata
round-trip through the real codec paths this project already tests
elsewhere. It just can't recover the *original* real mesh (``ObjectSpec``
has no slot for arbitrary vertex data -- by design, see
``gltfworld.scene.scene``'s docstring) -- :func:`read_mesh_accessors` is the
extra, physion-specific check for that half.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pygltflib

if TYPE_CHECKING:
    import h5py

from gltfworld.ext import khr_physics, rwm
from gltfworld.gltf.accessors import BufferAccumulator, read_accessor
from gltfworld.scene.convert import GENERATOR
from gltfworld.scene.episode import Episode, StateSeries
from gltfworld.scene.scene import CameraSpec, ObjectSpec, SceneState

# --- constants, see docs/PHYSION.md "Physion HDF5 schema" for the derivation ---

DEFAULT_DT = 1.0 / 30.0
DEFAULT_GRAVITY = np.array([0.0, -9.81, 0.0], dtype=np.float32)
SCENE_VERSION = "physion-collide-v1"

# AABB isotropy tolerance for the sphere/box bounding-shape classifier: every
# half-extent within this fraction of the mean half-extent -> "sphere".
_SPHERE_ISOTROPY_TOL = 0.2

ROLE_AGENT = "agent"  # HDF5 "target": red in mp4s-redyellow, the OCP subject
ROLE_PATIENT = "patient"  # HDF5 "zone": yellow, the OCP object
ROLE_PROBE = "probe"  # the launched impactor; not specially colored
ROLE_DISTRACTOR = "distractor"
ROLE_OCCLUDER = "occluder"
ROLE_OBJECT = "object"  # fallback for anything unclassified


# --- HDF5 -> in-memory trial ----------------------------------------------------


@dataclass
class PhysionObject:
    object_id: int
    model_name: str
    role: str
    is_static: bool
    mass: float
    friction: float
    restitution: float
    color: np.ndarray  # (4,) float32 rgba, [0,1]
    vertices: np.ndarray  # (V,3) float32, LOCAL, already scale-applied, real HDF5 geometry
    faces: np.ndarray  # (F,3) uint32
    positions: np.ndarray  # (T,3) float32, world, real per-frame pivot position
    rotations: np.ndarray  # (T,4) float32 xyzw, real per-frame
    lin_vel: np.ndarray  # (T,3) float32, real per-frame (not finite-differenced)
    ang_vel: np.ndarray  # (T,3) float32, real per-frame
    bounding_shape: str  # "sphere" | "box"
    bounding_size: np.ndarray  # (3,) float32, ObjectSpec.size convention
    bounding_center: np.ndarray  # (3,) float32, LOCAL AABB center (see findings: pivot != center)


@dataclass
class PhysionTrial:
    trial_id: str
    scenario: str
    dt: float
    times: np.ndarray  # (T,)
    objects: list[PhysionObject]
    target_id: int
    zone_id: int
    probe_id: int
    contact_per_frame: np.ndarray  # (T,) bool, labels/target_contacting_zone
    label: bool  # any(contact_per_frame) -- the trial-level OCP outcome
    camera_position: np.ndarray  # (3,) float32
    camera_rotation: np.ndarray  # (4,) float32 xyzw
    camera_yfov: float


def _decode_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals from triangle face data.

    HDF5 carries no normals at all (see module docstring) -- this is the
    same "average face normals per vertex" idea as
    ``gltfworld.scene.primitives._vertex_normals``, just area-weighted
    (via the un-normalized cross product's magnitude) rather than uniformly
    averaged, since Physion meshes (unlike gltfworld's own generated
    primitives) have widely varying face sizes.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0).astype(np.float64)  # magnitude ~ 2x triangle area

    vertex_normals = np.zeros((vertices.shape[0], 3), dtype=np.float64)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)

    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vertex_normals / norms).astype(np.float32)


def classify_bounding_shape(vertices: np.ndarray) -> tuple[str, np.ndarray, np.ndarray]:
    """Sphere/box AABB classifier: the "approximate each object by its
    bounding primitive" scheme both stage 2 (KHR physics collider) and
    stage 3 (dynamics-model D=22 shape_onehot) share, so this one
    approximation is documented once and reused everywhere rather than
    inventing a second scheme -- a real, honestly-labeled approximation, not
    a shape detector: Physion's real meshes (cones, toruses, chairs, ...)
    are not actually spheres/boxes/cylinders, gltfworld's own vocabulary
    (``gltfworld.scene.scene.SHAPES``) just doesn't have anything else.

    Returns ``(shape, size, center)``: ``shape`` in ``{"sphere", "box"}``
    (cylinder is not attempted -- distinguishing a cylinder from a box by
    AABB alone isn't possible, see docs/PHYSION.md findings), ``size`` per
    ``ObjectSpec.size``'s convention, ``center`` the *local* AABB center
    (since TDW meshes are base-pivoted, not centroid-pivoted -- see
    findings, this is generally nonzero).
    """
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    half_extents = (maxs - mins) / 2.0
    center = (maxs + mins) / 2.0
    mean_he = float(half_extents.mean())
    if mean_he > 0 and np.all(np.abs(half_extents - mean_he) <= _SPHERE_ISOTROPY_TOL * mean_he):
        radius = float(mean_he)
        return "sphere", np.array([radius, radius, radius], dtype=np.float32), center.astype(np.float32)
    return "box", half_extents.astype(np.float32), center.astype(np.float32)


def _role_for_object(object_id: int, model_name: str, static: h5py.Group) -> str:
    if object_id == int(static["target_id"][()]):
        return ROLE_AGENT
    if object_id == int(static["zone_id"][()]):
        return ROLE_PATIENT
    if object_id == int(static["probe_id"][()]):
        return ROLE_PROBE
    distractors = {_decode_str(v) for v in static["distractors"][()]}
    occluders = {_decode_str(v) for v in static["occluders"][()]}
    if model_name in distractors:
        return ROLE_DISTRACTOR
    if model_name in occluders:
        return ROLE_OCCLUDER
    return ROLE_OBJECT


def quat_rotate_points(quat: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Rotate ``points (V,3)`` by a single quaternion ``quat (4,)`` xyzw, the
    standard right-handed quaternion-rotate formula -- verified (see
    docs/PHYSION.md's schema section) to reproduce HDF5's own ``forwards``
    dataset exactly when applied to the raw, unmodified per-frame data, so
    this is the correct formula for *this* (left-handed, but internally
    self-consistent, see conversion findings) coordinate convention too."""
    q = np.asarray(quat, dtype=np.float64)
    v = np.asarray(points, dtype=np.float64)
    qv = q[:3]
    t = 2.0 * np.cross(qv, v)
    return v + q[3] * t + np.cross(qv, t)


def _look_at_quat(eye: np.ndarray, target: np.ndarray, world_up: np.ndarray = np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """Right-handed look-at quaternion (xyzw), glTF/OpenGL camera convention
    (looks down local -Z, up is local +Y). See docs/PHYSION.md findings for
    why this is a *synthesized* orientation (aimed at the scene) rather than
    a decode of HDF5's own camera_matrix: that matrix's rotation block has
    determinant -1 under a naive right-handed read (consistent with TDW/
    Unity's left-handed world convention), and gltfworld deliberately does
    not apply a chirality-correcting flip anywhere in this conversion (see
    findings) -- so decoding *this specific* matrix into a proper glTF
    camera rotation isn't attempted; only the real HDF5 camera *position*
    (handedness-independent) is used.
    """
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        forward = np.array([0.0, 0.0, -1.0])
    else:
        forward = forward / norm
    z_axis = -forward
    x_axis = np.cross(world_up, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-9:
        world_up = np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(world_up, z_axis)
        x_norm = np.linalg.norm(x_axis)
    x_axis = x_axis / x_norm
    y_axis = np.cross(z_axis, x_axis)

    m = np.stack([x_axis, y_axis, z_axis], axis=1)  # columns = local axes in world space
    return _matrix_to_quat_xyzw(m)


def _matrix_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
    """Shepperd's method, numpy, 3x3 proper-rotation matrix -> quaternion xyzw."""
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float32)
    return q / np.linalg.norm(q)


def _decode_camera(f: h5py.File) -> tuple[np.ndarray, np.ndarray, float]:
    cm = np.asarray(f["frames"]["0000"]["camera_matrices"]["camera_matrix"][()], dtype=np.float64).reshape(4, 4)
    proj = np.asarray(f["frames"]["0000"]["camera_matrices"]["projection_matrix"][()], dtype=np.float64).reshape(4, 4)
    r = cm[:3, :3]
    t = cm[:3, 3]
    camera_position = (-r.T @ t).astype(np.float32)

    # scene centroid at frame 0 (mean of every tracked object's position) as
    # the look-at target -- see _look_at_quat's docstring for why the real
    # matrix's rotation isn't decoded directly.
    positions0 = f["frames"]["0000"]["objects"]["positions"][()].astype(np.float64)
    target = positions0.mean(axis=0) if positions0.shape[0] else np.zeros(3)
    camera_rotation = _look_at_quat(camera_position.astype(np.float64), target)

    fovy_scale = proj[1, 1]
    yfov = float(2.0 * np.arctan(1.0 / fovy_scale)) if fovy_scale > 0 else float(np.radians(60.0))
    return camera_position, camera_rotation, yfov


def load_trial(hdf5_path: str | Path) -> PhysionTrial:
    """Load one Physion Collide HDF5 trial (real per-frame object/camera
    state + real per-object mesh geometry). Does **not** read any of
    ``frames/*/images/*`` (state-based track, no pixels needed -- also the
    overwhelming majority of the file's bytes, see docs/PHYSION.md).
    """
    import h5py  # local import: keeps `import gltfworld.physion.convert` light,
    # consistent with this project's existing pattern for optional/heavy deps
    # (e.g. gltfworld.datagen.mujoco_env's local `import mujoco`)

    hdf5_path = Path(hdf5_path)
    trial_id = hdf5_path.stem

    with h5py.File(hdf5_path, "r") as f:
        static = f["static"]
        object_ids = [int(v) for v in static["object_ids"][()]]
        model_names = [_decode_str(v) for v in static["model_names"][()]]
        mass = static["mass"][()]
        static_friction = static["static_friction"][()]
        dynamic_friction = static["dynamic_friction"][()]
        bounciness = static["bounciness"][()]
        scale = np.asarray(static["scale"][()], dtype=np.float32)
        color = np.asarray(static["color"][()], dtype=np.float32)
        target_id = int(static["target_id"][()])
        zone_id = int(static["zone_id"][()])
        probe_id = int(static["probe_id"][()])

        frame_keys = sorted(f["frames"].keys())
        t = len(frame_keys)
        dt = DEFAULT_DT
        times = np.arange(t, dtype=np.float32) * dt

        all_positions = np.stack([f["frames"][fk]["objects"]["positions"][()] for fk in frame_keys])
        all_rotations = np.stack([f["frames"][fk]["objects"]["rotations"][()] for fk in frame_keys])
        all_lin_vel = np.stack([f["frames"][fk]["objects"]["velocities"][()] for fk in frame_keys])
        all_ang_vel = np.stack([f["frames"][fk]["objects"]["angular_velocities"][()] for fk in frame_keys])
        contact_per_frame = np.array(
            [bool(f["frames"][fk]["labels"]["target_contacting_zone"][()]) for fk in frame_keys], dtype=bool
        )

        camera_position, camera_rotation, camera_yfov = _decode_camera(f)

        objects: list[PhysionObject] = []
        for i, object_id in enumerate(object_ids):
            raw_vertices = np.asarray(static["mesh"][f"vertices_{i}"][()], dtype=np.float32)
            raw_faces = np.asarray(static["mesh"][f"faces_{i}"][()], dtype=np.uint32)
            if raw_vertices.shape[0] == 0:
                # Some decorative occluder/distractor assets (real-world
                # furniture/animal models, e.g. "amphora_jar_vase",
                # "648972_chair_poliform_harmony", "shar_pei") have an
                # *empty* mesh in this HDF5 tier -- TDW's mesh-export step
                # evidently can't (or doesn't) dump geometry for every asset
                # kind, even though the object is fully tracked otherwise
                # (position/rotation/scale all present). Never true for
                # target/zone/probe in any trial checked (always a simple
                # primitive: cube/cone/sphere/torus/dumbbell/...), so this
                # never touches the OCP-relevant geometry -- a placeholder
                # unit box (scaled like a real object would be) stands in,
                # purely so the trial still converts to a complete, valid
                # GLB. See docs/PHYSION.md conversion findings.
                from gltfworld.scene.primitives import mesh_for

                placeholder_positions, _, placeholder_indices = mesh_for(
                    "box", np.array([0.5, 0.5, 0.5], dtype=np.float32)
                )
                vertices_local = placeholder_positions * scale[i]
                faces = placeholder_indices.reshape(-1, 3).astype(np.uint32)
            else:
                vertices_local = raw_vertices * scale[i]
                faces = raw_faces
                if faces.shape[1] != 3:
                    raise ValueError(f"{hdf5_path}: object {object_id} faces not triangulated: {faces.shape}")

            positions_i = np.ascontiguousarray(all_positions[:, i, :])
            rotations_i = np.ascontiguousarray(all_rotations[:, i, :])
            is_static = bool(
                np.max(np.abs(positions_i - positions_i[0])) < 1e-6
                and np.max(np.abs(rotations_i - rotations_i[0])) < 1e-6
            )

            bounding_shape, bounding_size, bounding_center = classify_bounding_shape(vertices_local)
            model_name = model_names[i]
            role = _role_for_object(object_id, model_name, static)

            # Distractors/occluders are only ever visual decoration, never a
            # real interactive rigid body: verified empirically (see
            # docs/PHYSION.md findings) that every such object is static
            # (zero pose change across the whole trial) *and* that TDW's own
            # static/mass|static_friction|dynamic_friction|bounciness arrays
            # are truncated to just the physically-relevant objects
            # (target/zone/probe) -- shorter than object_ids/model_names/
            # scale/color/mesh, which cover every object. Any index beyond
            # that truncated length gets a placeholder (mass/friction/
            # restitution are physically inert for a static body anyway --
            # KHR_physics_rigid_bodies omits "motion" for static bodies, see
            # gltfworld.ext.khr_physics), not a real measured value.
            has_physics_metadata = i < len(mass)
            objects.append(
                PhysionObject(
                    object_id=object_id,
                    model_name=model_name,
                    role=role,
                    is_static=is_static,
                    mass=float(mass[i]) if has_physics_metadata else 1.0,
                    friction=(
                        float((static_friction[i] + dynamic_friction[i]) / 2.0) if has_physics_metadata else 0.5
                    ),
                    restitution=float(bounciness[i]) if has_physics_metadata else 0.0,
                    color=np.array([color[i, 0], color[i, 1], color[i, 2], 1.0], dtype=np.float32),
                    vertices=vertices_local,
                    faces=faces,
                    positions=positions_i,
                    rotations=rotations_i,
                    lin_vel=np.ascontiguousarray(all_lin_vel[:, i, :]),
                    ang_vel=np.ascontiguousarray(all_ang_vel[:, i, :]),
                    bounding_shape=bounding_shape,
                    bounding_size=bounding_size,
                    bounding_center=bounding_center,
                )
            )

    label = bool(np.any(contact_per_frame))
    return PhysionTrial(
        trial_id=trial_id,
        scenario="Collide",
        dt=dt,
        times=times,
        objects=objects,
        target_id=target_id,
        zone_id=zone_id,
        probe_id=probe_id,
        contact_per_frame=contact_per_frame,
        label=label,
        camera_position=camera_position,
        camera_rotation=camera_rotation,
        camera_yfov=camera_yfov,
    )


# --- trial -> Episode (genuine ObjectSpec/SceneState/StateSeries) --------------


def trial_to_episode(trial: PhysionTrial) -> Episode:
    """Build a genuine ``Episode`` (real ``ObjectSpec``/``SceneState``/
    ``StateSeries``) from a loaded trial -- everything except the real
    visual mesh (``ObjectSpec`` has no slot for arbitrary geometry, see
    module docstring) round-trips through the *existing*,
    ``gltfworld.scene.convert.episode_from_gltf``-compatible codec.
    """
    objects = [
        ObjectSpec(
            object_id=obj.object_id,
            shape=obj.bounding_shape,
            size=obj.bounding_size,
            color=obj.color,
            roughness=0.8,
            metallic=0.0,
            mass=obj.mass,
            friction=obj.friction,
            restitution=obj.restitution,
            is_static=obj.is_static,
            category=obj.model_name,
            parts={"physion_role": obj.role, "physion_object_id": int(obj.object_id)},
        )
        for obj in trial.objects
    ]

    camera = CameraSpec(
        position=trial.camera_position,
        rotation=trial.camera_rotation,
        yfov=trial.camera_yfov,
        znear=0.1,
        zfar=100.0,
        aspect=1.0,
    )

    scene = SceneState(
        objects=objects,
        camera=camera,
        lights=[],
        gravity=DEFAULT_GRAVITY,
        dt=trial.dt,
        seed=hash(trial.trial_id) & 0x7FFFFFFF,
        scene_version=SCENE_VERSION,
    )

    poses = np.zeros((len(trial.times), len(objects), 7), dtype=np.float32)
    lin_vel = np.zeros((len(trial.times), len(objects), 3), dtype=np.float32)
    ang_vel = np.zeros((len(trial.times), len(objects), 3), dtype=np.float32)
    for i, obj in enumerate(trial.objects):
        poses[:, i, 0:3] = obj.positions
        poses[:, i, 3:7] = obj.rotations
        lin_vel[:, i, :] = obj.lin_vel
        ang_vel[:, i, :] = obj.ang_vel

    series = StateSeries(times=trial.times, poses=poses, lin_vel=lin_vel, ang_vel=ang_vel)
    return Episode(scene=scene, series=series)


# --- Episode + real meshes -> pygltflib.GLTF2 -----------------------------------


def _add_real_geometry_node(
    gltf: pygltflib.GLTF2,
    accumulator: BufferAccumulator,
    obj: ObjectSpec,
    physion_obj: PhysionObject,
    pose0: np.ndarray,
) -> int:
    """Adapted from ``gltfworld.scene.convert._add_object_node``/
    ``_add_geometry``: same node/mesh/material wiring, but the geometry
    accessors come from ``physion_obj``'s *real* HDF5 vertices/faces
    instead of ``gltfworld.scene.primitives.mesh_for``'s procedural
    primitives -- the one deliberate difference from that codec (see
    module docstring)."""
    positions = physion_obj.vertices
    normals = compute_vertex_normals(positions, physion_obj.faces)
    indices = physion_obj.faces.reshape(-1).astype(np.uint32)

    pos_acc = accumulator.add_accessor(
        gltf, positions, type_=pygltflib.VEC3, target=pygltflib.ARRAY_BUFFER, compute_minmax=True
    )
    normal_acc = accumulator.add_accessor(gltf, normals, type_=pygltflib.VEC3, target=pygltflib.ARRAY_BUFFER)
    indices_acc = accumulator.add_accessor(gltf, indices, type_=pygltflib.SCALAR, target=pygltflib.ELEMENT_ARRAY_BUFFER)

    material = pygltflib.Material(
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=[float(v) for v in obj.color],
            roughnessFactor=obj.roughness,
            metallicFactor=obj.metallic,
        ),
        name=f"mat_{obj.category}_{obj.object_id}",
    )
    gltf.materials.append(material)
    material_index = len(gltf.materials) - 1

    mesh = pygltflib.Mesh(
        primitives=[
            pygltflib.Primitive(
                attributes=pygltflib.Attributes(POSITION=pos_acc, NORMAL=normal_acc),
                indices=indices_acc,
                material=material_index,
                mode=pygltflib.TRIANGLES,
            )
        ]
    )
    gltf.meshes.append(mesh)
    mesh_index = len(gltf.meshes) - 1

    node = pygltflib.Node(
        name=f"obj_{obj.object_id}",
        mesh=mesh_index,
        translation=[float(v) for v in pose0[0:3]],
        rotation=[float(v) for v in pose0[3:7]],
    )
    gltf.nodes.append(node)
    return len(gltf.nodes) - 1


def _add_camera_node(gltf: pygltflib.GLTF2, camera: CameraSpec) -> int:
    gltf.cameras.append(
        pygltflib.Camera(
            type="perspective",
            perspective=pygltflib.Perspective(
                aspectRatio=camera.aspect, yfov=camera.yfov, znear=camera.znear, zfar=camera.zfar
            ),
        )
    )
    camera_index = len(gltf.cameras) - 1
    node = pygltflib.Node(
        name="camera",
        camera=camera_index,
        translation=[float(v) for v in camera.position],
        rotation=[float(v) for v in camera.rotation],
    )
    gltf.nodes.append(node)
    return len(gltf.nodes) - 1


def trial_to_gltf(trial: PhysionTrial) -> pygltflib.GLTF2:
    """Encode ``trial`` into a fresh ``pygltflib.GLTF2`` document: real
    per-object meshes, STEP pose animation, ``KHR_physics_rigid_bodies``/
    ``KHR_implicit_shapes`` (box/sphere approximation), ``RWM_state_series``
    (real velocities), ``extras.rwm`` (category=model name, +role)."""
    episode = trial_to_episode(trial)
    scene = episode.scene
    series = episode.series

    gltf = pygltflib.GLTF2()
    gltf.asset = pygltflib.Asset(generator=GENERATOR, version="2.0")
    accumulator = BufferAccumulator()

    node_index_by_object_id: dict[int, int] = {}
    object_ids_in_order = [obj.object_id for obj in scene.objects]

    for i, (obj, physion_obj) in enumerate(zip(scene.objects, trial.objects)):
        pose0 = series.poses[0, i]
        node_index = _add_real_geometry_node(gltf, accumulator, obj, physion_obj, pose0)
        node_index_by_object_id[obj.object_id] = node_index

    _add_camera_node(gltf, scene.camera)

    gltf.scenes = [pygltflib.Scene(nodes=list(range(len(gltf.nodes))))]
    gltf.scene = 0

    times_accessor = accumulator.add_accessor(gltf, series.times, type_=pygltflib.SCALAR, compute_minmax=True)

    animation = pygltflib.Animation()
    for i, obj in enumerate(scene.objects):
        node_index = node_index_by_object_id[obj.object_id]
        translations = np.ascontiguousarray(series.poses[:, i, 0:3])
        rotations = np.ascontiguousarray(series.poses[:, i, 3:7])

        trans_acc = accumulator.add_accessor(gltf, translations, type_=pygltflib.VEC3)
        rot_acc = accumulator.add_accessor(gltf, rotations, type_=pygltflib.VEC4)

        animation.samplers.append(
            pygltflib.AnimationSampler(input=times_accessor, output=trans_acc, interpolation="STEP")
        )
        trans_sampler_index = len(animation.samplers) - 1
        animation.samplers.append(
            pygltflib.AnimationSampler(input=times_accessor, output=rot_acc, interpolation="STEP")
        )
        rot_sampler_index = len(animation.samplers) - 1

        animation.channels.append(
            pygltflib.AnimationChannel(
                sampler=trans_sampler_index,
                target=pygltflib.AnimationChannelTarget(node=node_index, path="translation"),
            )
        )
        animation.channels.append(
            pygltflib.AnimationChannel(
                sampler=rot_sampler_index,
                target=pygltflib.AnimationChannelTarget(node=node_index, path="rotation"),
            )
        )
    gltf.animations = [animation]

    initial_lin_vel = {obj.object_id: series.lin_vel[0, i] for i, obj in enumerate(scene.objects)}
    initial_ang_vel = {obj.object_id: series.ang_vel[0, i] for i, obj in enumerate(scene.objects)}
    physics_doc = khr_physics.build_khr_physics(scene.objects, node_index_by_object_id, initial_lin_vel, initial_ang_vel)
    khr_physics.write_khr_physics(gltf, physics_doc)

    rwm_doc = rwm.build_state_series(
        gltf, accumulator, times_accessor, node_index_by_object_id, object_ids_in_order, series
    )
    rwm.write_state_series(gltf, rwm_doc)

    for obj, physion_obj in zip(scene.objects, trial.objects):
        node = gltf.nodes[node_index_by_object_id[obj.object_id]]
        extras = rwm.node_extras(obj.object_id, obj.category, obj.parts, obj.mass, obj.is_static)
        extras["role"] = physion_obj.role
        node.extras["rwm"] = extras

    gltf.scenes[0].extras["rwm"] = rwm.scene_extras(scene.seed, scene.scene_version, scene.dt, scene.gravity)
    gltf.scenes[0].extras["physion"] = {
        "trial_id": trial.trial_id,
        "scenario": trial.scenario,
        "label_contact": trial.label,
        "target_id": trial.target_id,
        "zone_id": trial.zone_id,
        "probe_id": trial.probe_id,
    }

    accumulator.finalize(gltf)

    extensions_used = [rwm.EXT_STATE_SERIES]
    if physics_doc.shapes:
        extensions_used.append(khr_physics.EXT_IMPLICIT_SHAPES)
    if physics_doc.physics_materials or physics_doc.node_physics:
        extensions_used.append(khr_physics.EXT_RIGID_BODIES)
    gltf.extensionsUsed = sorted(set(extensions_used))
    gltf.extensionsRequired = []

    return gltf


def save_trial_gltf(trial: PhysionTrial, path: str | Path) -> None:
    path = Path(path)
    gltf = trial_to_gltf(trial)
    if path.suffix.lower() == ".glb":
        gltf.save_binary(str(path))
    else:
        gltf.save(str(path), asset=gltf.asset)


def convert_trial(hdf5_path: str | Path, out_path: str | Path) -> PhysionTrial:
    """Convenience one-shot: load + convert + save. Returns the loaded
    ``PhysionTrial`` (useful for tests that want to compare against what was
    written without re-parsing the HDF5)."""
    trial = load_trial(hdf5_path)
    save_trial_gltf(trial, out_path)
    return trial


# --- read-back helpers for the physion-specific round-trip check ---------------


def read_mesh_accessors(gltf: pygltflib.GLTF2, node_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read back the (POSITION, NORMAL, indices) accessors for the mesh at
    ``node_index`` -- the direct "real mesh round-trips" check
    ``gltfworld.scene.convert.episode_from_gltf`` can't do on its own
    (``ObjectSpec`` has no arbitrary-mesh slot, see module docstring)."""
    node = gltf.nodes[node_index]
    mesh = gltf.meshes[node.mesh]
    prim = mesh.primitives[0]
    positions = read_accessor(gltf, prim.attributes.POSITION)
    normals = read_accessor(gltf, prim.attributes.NORMAL)
    indices = read_accessor(gltf, prim.indices)
    return positions, normals, indices.reshape(-1, 3)


def read_object_node_indices(gltf: pygltflib.GLTF2) -> list[int]:
    indices = []
    for i, node in enumerate(gltf.nodes):
        extras = node.extras or {}
        if "rwm" in extras and "object_id" in extras["rwm"]:
            indices.append(i)
    return indices
