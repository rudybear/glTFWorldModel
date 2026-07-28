"""``wm-scenes-v1``: the seeded scene distribution gltfworld's MuJoCo data
generation draws from (see DESIGN.md "wm-scenes-v1" for the full spec
table). Everything here is pure numpy -- no MuJoCo import -- so it has no
``sim``-extra dependency and no gpu marker.

``sample_scene(seed)`` returns a :class:`SampledScene`: a
``gltfworld.scene.scene.SceneState`` (no pose -- see
``gltfworld.datagen.mujoco_env`` module docstring for why) bundled with the
initial per-object pose/velocity ``gltfworld.datagen.mujoco_env.simulate``
needs to seed MuJoCo.
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass

import numpy as np

from gltfworld.datagen.mujoco_env import shape_volume
from gltfworld.scene.scene import CameraSpec, LightSpec, ObjectSpec, SceneState

SCENE_VERSION = "wm-scenes-v1"

# --- distribution constants (see DESIGN.md "wm-scenes-v1" table) ------------

_N_OBJECTS_MIN, _N_OBJECTS_MAX = 1, 5
_SHAPE_CHOICES = ("sphere", "box", "cylinder")
_SHAPE_WEIGHTS = (0.45, 0.45, 0.10)  # cylinder at 10%, per spec
_CATEGORY_BY_SHAPE = {"sphere": "ball", "box": "crate", "cylinder": "cylinder"}

_SIZE_MIN, _SIZE_MAX = 0.05, 0.25  # characteristic size, meters
_WORKSPACE_HALF_XZ = 0.75  # "1.5 m box" -> x, z in [-0.75, 0.75]
_DROP_HEIGHT_MIN, _DROP_HEIGHT_MAX = 0.2, 1.2  # clearance ABOVE ground, meters (see sample_scene docstring)
_GROUND_CLEARANCE_EPS = 0.005

_DENSITY_MIN, _DENSITY_MAX = 300.0, 3000.0  # kg/m^3
_FRICTION_MIN, _FRICTION_MAX = 0.4, 1.0
_RESTITUTION = 0.1  # fixed, per spec

_MAX_LIN_SPEED = 1.5
_MAX_ANG_SPEED = 3.0

_GRAVITY = np.array([0.0, -9.81, 0.0], dtype=np.float32)
_DEFAULT_DT = 1.0 / 30.0  # placeholder; gltfworld.datagen.mujoco_env.simulate overwrites with the actual dt

_GROUND_OBJECT_ID = 0
_GROUND_HALF_EXTENTS = np.array([3.0, 0.1, 3.0], dtype=np.float32)
_GROUND_TOP_Y = 0.0  # ground surface height; ground body center sits at _GROUND_TOP_Y - half_extent_y

_NON_OVERLAP_BUFFER = 0.02
_MAX_PLACEMENT_ATTEMPTS = 2000

_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


# --- fixed camera (see DESIGN.md: verified to keep the full sampled ----------
# --- workspace inside the frustum at render aspect 1:1) ----------------------


def _look_at_quat_xyzw(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """Build the glTF-convention (camera looks down local -Z, local +Y up)
    orientation quaternion (xyzw) for a camera at ``eye`` looking at
    ``target``."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    # Columns of the camera-local-to-world rotation matrix: local +X ->
    # right, local +Y -> true_up, local +Z -> -forward (camera looks down -Z).
    r = np.stack([right, true_up, -forward], axis=1)

    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2.0
    if w > 1e-8:
        x = (r[2, 1] - r[1, 2]) / (4.0 * w)
        y = (r[0, 2] - r[2, 0]) / (4.0 * w)
        z = (r[1, 0] - r[0, 1]) / (4.0 * w)
    else:  # pragma: no cover - not hit by this module's fixed look-at
        x, y, z = 0.0, 0.0, 0.0
    return np.array([x, y, z, w], dtype=np.float32)


_CAMERA_EYE = np.array([0.0, 1.7, 4.6], dtype=np.float64)
_CAMERA_TARGET = np.array([0.0, 1.0, 0.0], dtype=np.float64)
_CAMERA_YFOV = math.radians(62.0)

_FIXED_CAMERA = CameraSpec(
    position=_CAMERA_EYE.astype(np.float32),
    rotation=_look_at_quat_xyzw(_CAMERA_EYE, _CAMERA_TARGET),
    yfov=_CAMERA_YFOV,
    znear=0.05,
    zfar=100.0,
    aspect=1.0,
)

_FIXED_LIGHTS = [
    LightSpec(
        type="directional",
        color=np.array([1.0, 1.0, 0.97], dtype=np.float32),
        intensity=3.0,
        rotation=_look_at_quat_xyzw(np.array([1.0, 2.0, 1.0]), np.array([0.0, 0.0, 0.0])),
    ),
    LightSpec(
        type="point",
        color=np.array([1.0, 0.95, 0.9], dtype=np.float32),
        intensity=8.0,
        position=np.array([-1.5, 2.2, 1.5], dtype=np.float32),
    ),
]


def object_bounding_radius(obj: ObjectSpec) -> float:
    """A conservative (always-an-over-approximation) bounding-sphere radius
    for ``obj``, orientation-independent -- used for both non-overlap and
    ground-clearance checks so sampling never needs to know which way an
    object is (randomly, uniform-SO(3)) oriented.

    ``norm(size)`` over-approximates the true circumscribed radius for
    every shape gltfworld supports: exact for a box's half-extents
    (``sqrt(hx^2+hy^2+hz^2)`` is precisely a box's half-diagonal), and a
    safe over-estimate for sphere/cylinder (whose true bounding radius is
    ``size[0]`` / ``sqrt(size[0]^2+size[1]^2)`` respectively -- both <=
    ``norm(size)`` since it sums one or two extra nonnegative squared terms).
    """
    return float(np.linalg.norm(obj.size))


def point_in_frustum(camera: CameraSpec, position: np.ndarray, radius: float = 0.0) -> bool:
    """True if the sphere ``(position, radius)`` lies entirely inside
    ``camera``'s view frustum at its own authored ``aspect`` (wm-scenes-v1's
    fixed camera is authored at ``aspect=1.0``, matching the render size
    this milestone always uses -- see ``EpisodeRenderer``'s aspect-ratio
    note in DESIGN.md).

    Conservative sphere-vs-frustum test: projects ``position`` into camera
    space (camera looks down local -Z, local +Y up, matching glTF/OpenGL),
    then requires the whole sphere to clear each of the near/far/top/
    bottom/left/right planes with ``radius`` to spare.
    """
    position = np.asarray(position, dtype=np.float64)
    cam_pos = np.asarray(camera.position, dtype=np.float64)
    x, y, z, w = (float(v) for v in camera.rotation)
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    local = rot.T @ (position - cam_pos)
    depth = -local[2]  # camera looks down local -Z

    if depth - radius <= camera.znear or depth + radius >= camera.zfar:
        return False

    half_h = depth * math.tan(camera.yfov / 2.0)
    half_w = half_h * camera.aspect
    # Account for the sphere's radius growing the required half-extent
    # proportionally with depth (a first-order, slightly conservative
    # correction -- exact would need the true distance-to-plane, this is
    # close enough for radius << depth, which always holds here).
    return abs(local[1]) + radius <= half_h and abs(local[0]) + radius <= half_w


def object_support_offset(obj: ObjectSpec, quat_xyzw: np.ndarray, direction_world: np.ndarray) -> float:
    """Exact (orientation-aware) distance from ``obj``'s center to its
    surface along ``direction_world`` (a unit vector), given its current
    orientation ``quat_xyzw``. This is the convex-shape "support function":
    the true offset of the extreme point in that direction, unlike
    :func:`object_bounding_radius`'s orientation-independent (and often
    much larger, e.g. for a box resting flat) over-approximation. Used to
    check ground clearance/penetration correctly regardless of how an
    object has tumbled -- passing the (larger) bounding-sphere radius there
    instead would spuriously flag a perfectly-resting box as "sunk below
    ground" whenever its half-diagonal exceeds its half-height.

    Same local-Z-symmetric cylinder convention as
    ``gltfworld.datagen.mujoco_env`` (see its module docstring).
    """
    x, y, z, w = (float(v) for v in quat_xyzw)
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    d_local = rot.T @ np.asarray(direction_world, dtype=np.float64)

    if obj.shape == "sphere":
        return float(obj.size[0])
    if obj.shape == "box":
        return float(np.abs(d_local) @ obj.size.astype(np.float64))
    if obj.shape == "cylinder":
        radius, half_height, _ = (float(v) for v in obj.size)
        return radius * math.hypot(d_local[0], d_local[1]) + half_height * abs(d_local[2])
    raise ValueError(f"unsupported shape {obj.shape!r}")


# --- sampling helpers ---------------------------------------------------------


def _sample_shape(rng: np.random.Generator) -> str:
    return str(rng.choice(_SHAPE_CHOICES, p=_SHAPE_WEIGHTS))


def _sample_size(shape: str, rng: np.random.Generator) -> np.ndarray:
    characteristic = rng.uniform(_SIZE_MIN, _SIZE_MAX)
    if shape == "sphere":
        return np.array([characteristic, characteristic, characteristic], dtype=np.float32)
    if shape == "box":
        return np.array([characteristic, characteristic, characteristic], dtype=np.float32)
    if shape == "cylinder":
        radius = characteristic * 0.6
        return np.array([radius, characteristic, radius], dtype=np.float32)
    raise ValueError(f"unknown shape {shape!r}")


def _random_unit_quaternion(rng: np.random.Generator) -> np.ndarray:
    """Uniform-random orientation on SO(3) (Shoemake's method), xyzw."""
    u1, u2, u3 = rng.uniform(0.0, 1.0, size=3)
    return np.array(
        [
            math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
            math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
            math.sqrt(u1) * math.sin(2 * math.pi * u3),
            math.sqrt(u1) * math.cos(2 * math.pi * u3),
        ],
        dtype=np.float32,
    )


def _random_velocity(rng: np.random.Generator, max_speed: float) -> np.ndarray:
    direction = rng.normal(size=3)
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float32)
    direction = direction / norm
    speed = rng.uniform(0.0, max_speed)
    return (direction * speed).astype(np.float32)


def _palette(n: int = 8) -> list[np.ndarray]:
    colors = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / n, 0.65, 0.9)
        colors.append(np.array([r, g, b, 1.0], dtype=np.float32))
    return colors


_PALETTE = _palette(8)


def _sample_color(rng: np.random.Generator) -> np.ndarray:
    return _PALETTE[int(rng.integers(0, len(_PALETTE)))].copy()


def _ground_object() -> ObjectSpec:
    return ObjectSpec(
        object_id=_GROUND_OBJECT_ID,
        shape="box",
        size=_GROUND_HALF_EXTENTS.copy(),
        color=np.array([0.55, 0.55, 0.58, 1.0], dtype=np.float32),
        roughness=0.9,
        metallic=0.0,
        mass=1000.0,
        friction=0.8,
        restitution=_RESTITUTION,
        is_static=True,
        category="ground",
    )


def _ground_pose() -> np.ndarray:
    pose = np.zeros(7, dtype=np.float32)
    pose[1] = _GROUND_TOP_Y - float(_GROUND_HALF_EXTENTS[1])
    pose[3:7] = _IDENTITY_QUAT
    return pose


@dataclass
class SampledScene:
    """A ``wm-scenes-v1`` sample: a static ``SceneState`` plus the initial
    per-object pose/velocity ``gltfworld.datagen.mujoco_env.simulate`` needs
    (object order matches ``scene.objects``; row 0 is always the ground)."""

    scene: SceneState
    initial_poses: np.ndarray  # (N, 7) float32, xyz + quat xyzw, gltf convention
    initial_lin_vel: np.ndarray  # (N, 3) float32, world frame, gltf axes
    initial_ang_vel: np.ndarray  # (N, 3) float32, world frame, gltf axes


def sample_scene(seed: int) -> SampledScene:
    """Sample one ``wm-scenes-v1`` scene, deterministic in ``seed``.

    Dynamic objects (1-5, sphere/box mostly, cylinder at 10%) are placed at
    non-overlapping ``(x, z)`` positions inside a 1.5 m x 1.5 m horizontal
    box (``x, z in [-0.75, 0.75]``), each dropped from a height *above the
    ground* (0.2-1.2 m clearance between the object's own bounding sphere
    and the ground plane -- see :func:`object_bounding_radius`) rather than
    an absolute world-Y coordinate, so no object starts inside the ground
    regardless of its (randomly sampled) size -- a deliberate reading of
    the milestone spec's "height 0.2-1.2 m", documented in DESIGN.md.
    Orientation is uniform on SO(3); linear/angular speed are bounded
    (not necessarily uniformly distributed) by 1.5 m/s / 3 rad/s.
    """
    rng = np.random.default_rng(seed)

    n_objects = int(rng.integers(_N_OBJECTS_MIN, _N_OBJECTS_MAX + 1))

    objects: list[ObjectSpec] = [_ground_object()]
    poses: list[np.ndarray] = [_ground_pose()]
    lin_vels: list[np.ndarray] = [np.zeros(3, dtype=np.float32)]
    ang_vels: list[np.ndarray] = [np.zeros(3, dtype=np.float32)]

    placed_positions: list[tuple[np.ndarray, float]] = []  # (xyz position, bounding radius)

    for i in range(n_objects):
        shape = _sample_shape(rng)
        size = _sample_size(shape, rng)
        object_id = i + 1
        placeholder = ObjectSpec(
            object_id=object_id,
            shape=shape,
            size=size,
            color=_sample_color(rng),
            roughness=float(rng.uniform(0.3, 0.8)),
            metallic=0.0,
            mass=1.0,  # filled in below once density is sampled
            friction=float(rng.uniform(_FRICTION_MIN, _FRICTION_MAX)),
            restitution=_RESTITUTION,
            is_static=False,
            category=_CATEGORY_BY_SHAPE[shape],
        )
        radius = object_bounding_radius(placeholder)
        density = rng.uniform(_DENSITY_MIN, _DENSITY_MAX)
        placeholder.mass = float(density * shape_volume(shape, size))

        # Non-overlap uses the true 3D distance (not just xz): two objects
        # dropped from sufficiently different heights are genuinely
        # non-overlapping even with close/coincident xz footprints, and the
        # drop-height range gives plenty of extra room to separate up to 5
        # objects that wouldn't all simultaneously fit non-overlapping in
        # the 1.5 m x 1.5 m footprint alone at max characteristic size
        # (5 circles of the largest allowed radius exceed the footprint's
        # area). Falls back to relaxing the clearance buffer (never the
        # workspace/frustum bounds) if attempts are exhausted, so placement
        # always terminates.
        best_candidate = None
        best_clearance = -np.inf
        for _attempt in range(_MAX_PLACEMENT_ATTEMPTS):
            xz = rng.uniform(-_WORKSPACE_HALF_XZ, _WORKSPACE_HALF_XZ, size=2)
            drop_height = rng.uniform(_DROP_HEIGHT_MIN, _DROP_HEIGHT_MAX)
            y = _GROUND_TOP_Y + radius + _GROUND_CLEARANCE_EPS + drop_height
            candidate = np.array([xz[0], y, xz[1]])

            clearance = min(
                (
                    float(np.linalg.norm(candidate - other_pos) - radius - other_radius)
                    for other_pos, other_radius in placed_positions
                ),
                default=math.inf,
            )
            if clearance > best_clearance:
                best_clearance = clearance
                best_candidate = candidate
            if clearance >= _NON_OVERLAP_BUFFER:
                break
        candidate = best_candidate if best_candidate is not None else np.array([0.0, radius, 0.0])
        placed_positions.append((candidate, radius))

        pose = np.zeros(7, dtype=np.float32)
        pose[0:3] = candidate
        pose[3:7] = _random_unit_quaternion(rng)

        objects.append(placeholder)
        poses.append(pose)
        lin_vels.append(_random_velocity(rng, _MAX_LIN_SPEED))
        ang_vels.append(_random_velocity(rng, _MAX_ANG_SPEED))

    scene = SceneState(
        objects=objects,
        camera=_FIXED_CAMERA,
        lights=_FIXED_LIGHTS,
        gravity=_GRAVITY.copy(),
        dt=_DEFAULT_DT,
        seed=seed,
        scene_version=SCENE_VERSION,
    )

    return SampledScene(
        scene=scene,
        initial_poses=np.stack(poses).astype(np.float32),
        initial_lin_vel=np.stack(lin_vels).astype(np.float32),
        initial_ang_vel=np.stack(ang_vels).astype(np.float32),
    )
