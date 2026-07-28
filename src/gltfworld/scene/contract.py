"""The dynamics/perception tensor contract: ``Episode <-> dict[str, np.ndarray]``.

This is the numeric contract every downstream ML component (V5+ dynamics/
perception models, this milestone's dataset packing/stats) is built against.
It is deliberately *not* a lossless glTF round trip (that's
``gltfworld.scene.convert``'s job, exercised by ``tests/test_provenance.py``
which chains ``episode_to_tensors`` on both ends of a save/load through a
real GLB); it only carries the numeric fields a model needs to consume or
predict, dropping presentation-only fields (color, roughness, metallic,
``parts``, category *string* -- as opposed to the class id -- ...) that a
physics/perception model has no use for.

Per-object dynamic state layout (``states``, one row per non-static object)
-------------------------------------------------------------------------

``D = 22``, column layout:

| slice     | width | field                              | notes |
| --------- | ----- | ---------------------------------- | ----- |
| `[0:3]`   | 3     | `pos`                              | world xyz, meters |
| `[3:7]`   | 4     | `quat` (xyzw)                      | unit quaternion, hemisphere-normalized so `w >= 0` (see below) |
| `[7:10]`  | 3     | `lin_vel`                          | world frame, m/s |
| `[10:13]` | 3     | `ang_vel`                          | world frame, rad/s |
| `[13:16]` | 3     | `shape_onehot`                     | one-hot over `("sphere", "box", "cylinder")`, in that fixed order (matches `gltfworld.scene.scene.SHAPES`) |
| `[16:19]` | 3     | `size`                             | per-shape convention, see `gltfworld.scene.scene.ObjectSpec` |
| `[19:20]` | 1     | `log_mass`                         | `log(mass)`; mass is always > 0 so this is well defined |
| `[20:21]` | 1     | `friction`                         | Coulomb friction coefficient |
| `[21:22]` | 1     | `restitution`                      | restitution coefficient |

Total: 3+4+3+3+3+3+1+1+1 = 22.

**Quaternion hemisphere normalization**: a unit quaternion and its negation
represent the same rotation (double cover of SO(3)); to make raw numeric
comparison (e.g. this milestone's round-trip test) well defined, every
encoded quaternion is flipped to the ``w >= 0`` hemisphere
(``q if q[3] >= 0 else -q``). This is a lossless, exact operation (sign
flip only) as far as the *rotation* is concerned, but it means a raw
encode -> decode -> re-encode of a ``w < 0`` input will not bit-match the
original sign -- callers that need the original sign must keep it out of
band; the contract only promises rotation-equivalence.

**Static objects** (``is_static=True``, e.g. the ground plane) are excluded
from ``states``/``mask``/``class_ids`` entirely -- see the ``static``
sub-dict below.

Output dict (``episode_to_tensors``)
-------------------------------------

- ``states``: ``float32 (T, N, D)``, ``D=22`` as above, ``N`` = number of
  *dynamic* (non-static) objects in ``ep.scene.objects``, in that same
  order.
- ``mask``: ``bool (N,)``, all ``True`` (every row is a real object at this
  per-episode stage; padding to a batch ``N_max`` -- and the corresponding
  ``False`` mask entries -- is added at the dataset-packing level, see
  ``gltfworld.data.pack``).
- ``class_ids``: ``int64 (N,)``, one id per dynamic object from
  ``CATEGORY_TO_CLASS_ID`` (see below), looked up by ``ObjectSpec.category``.
- ``globals``: ``float32 (G,)``, ``G=12``: `gravity` (3, world frame m/s^2),
  `dt` (1, the *actual* recorded seconds/frame -- see
  ``gltfworld.datagen.mujoco_env.simulate``'s docstring for why this isn't
  always the nominal `1/hz`), `camera.position` (3), `camera.rotation` (4,
  xyzw, hemisphere-normalized same as object quats), `camera.yfov` (1).
- ``static``: a sub-dict, one entry per *static* object (ground), each a
  small dict of ``{"shape": str, "size": (3,) float32, "pos": (3,) float32,
  "quat": (4,) float32 (hemisphere-normalized), "category": str}`` -- static
  geometry is constant context a model conditions on, not something it
  predicts, so it isn't packed into the per-frame ``states`` tensor at all.

Category -> class id map
-------------------------

``CATEGORY_TO_CLASS_ID = {"ball": 0, "crate": 1, "cylinder": 2}`` -- the
three dynamic-object categories ``gltfworld.datagen.sample``'s
``wm-scenes-v1`` distribution actually produces (``_CATEGORY_BY_SHAPE``:
sphere -> ball, box -> crate, cylinder -> cylinder). ``"ground"`` (the only
static category) never appears here since static objects are excluded from
``class_ids``. Looking up an unrecognized category raises ``KeyError``
(fail loud rather than silently misclassify).
"""

from __future__ import annotations

import numpy as np

from .episode import Episode
from .scene import ObjectSpec, SceneState

STATE_DIM = 22
GLOBALS_DIM = 12

SHAPE_ORDER = ("sphere", "box", "cylinder")  # matches gltfworld.scene.scene.SHAPES

CATEGORY_TO_CLASS_ID = {"ball": 0, "crate": 1, "cylinder": 2}
CLASS_ID_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_CLASS_ID.items()}


def _hemisphere_normalize(quat: np.ndarray) -> np.ndarray:
    """Flip ``quat`` (..., 4) xyzw so its ``w`` (last) component is >= 0."""
    quat = np.asarray(quat, dtype=np.float32)
    sign = np.where(quat[..., 3:4] < 0, -1.0, 1.0).astype(np.float32)
    return quat * sign


def _shape_onehot(shape: str) -> np.ndarray:
    onehot = np.zeros(3, dtype=np.float32)
    onehot[SHAPE_ORDER.index(shape)] = 1.0
    return onehot


def _shape_from_onehot(onehot: np.ndarray) -> str:
    return SHAPE_ORDER[int(np.argmax(onehot))]


def episode_to_tensors(ep: Episode) -> dict:
    """Encode ``ep`` into the numeric tensor contract (see module docstring).

    Dynamic (non-static) objects populate ``states``/``mask``/``class_ids``,
    in ``ep.scene.objects`` order restricted to ``is_static=False``. Static
    objects populate the ``static`` sub-dict instead.
    """
    scene = ep.scene
    series = ep.series
    t = series.num_frames

    dynamic_indices = [i for i, obj in enumerate(scene.objects) if not obj.is_static]
    static_indices = [i for i, obj in enumerate(scene.objects) if obj.is_static]
    n = len(dynamic_indices)

    states = np.zeros((t, n, STATE_DIM), dtype=np.float32)
    class_ids = np.zeros((n,), dtype=np.int64)

    for row, obj_index in enumerate(dynamic_indices):
        obj = scene.objects[obj_index]
        pos = series.poses[:, obj_index, 0:3]
        quat = _hemisphere_normalize(series.poses[:, obj_index, 3:7])
        lin_vel = (
            series.lin_vel[:, obj_index, :]
            if series.lin_vel is not None
            else np.zeros((t, 3), dtype=np.float32)
        )
        ang_vel = (
            series.ang_vel[:, obj_index, :]
            if series.ang_vel is not None
            else np.zeros((t, 3), dtype=np.float32)
        )
        onehot = _shape_onehot(obj.shape)

        states[:, row, 0:3] = pos
        states[:, row, 3:7] = quat
        states[:, row, 7:10] = lin_vel
        states[:, row, 10:13] = ang_vel
        states[:, row, 13:16] = onehot
        states[:, row, 16:19] = obj.size
        states[:, row, 19] = np.log(obj.mass).astype(np.float32)
        states[:, row, 20] = obj.friction
        states[:, row, 21] = obj.restitution

        if obj.category not in CATEGORY_TO_CLASS_ID:
            raise KeyError(
                f"unrecognized dynamic-object category {obj.category!r}; "
                f"expected one of {sorted(CATEGORY_TO_CLASS_ID)}"
            )
        class_ids[row] = CATEGORY_TO_CLASS_ID[obj.category]

    mask = np.ones((n,), dtype=bool)

    camera = scene.camera
    globals_ = np.concatenate(
        [
            scene.gravity.astype(np.float32),
            np.array([scene.dt], dtype=np.float32),
            camera.position.astype(np.float32),
            _hemisphere_normalize(camera.rotation),
            np.array([camera.yfov], dtype=np.float32),
        ]
    ).astype(np.float32)
    assert globals_.shape == (GLOBALS_DIM,)

    static = {}
    for obj_index in static_indices:
        obj = scene.objects[obj_index]
        static[str(obj.object_id)] = {
            "shape": obj.shape,
            "size": obj.size.copy(),
            "pos": series.poses[0, obj_index, 0:3].copy(),
            "quat": _hemisphere_normalize(series.poses[0, obj_index, 3:7]),
            "category": obj.category,
        }

    return {
        "states": states,
        "mask": mask,
        "class_ids": class_ids,
        "globals": globals_,
        "static": static,
    }


def tensors_to_state(
    states: np.ndarray,
    mask: np.ndarray,
    class_ids: np.ndarray,
    globals_: np.ndarray,
    template: SceneState,
) -> dict:
    """Inverse of :func:`episode_to_tensors` for the dynamic part.

    Reconstructs everything ``states``/``class_ids``/``globals`` actually
    carry (pose, velocities, shape, size, mass, friction, restitution,
    gravity, dt, camera position/rotation/yfov). Fields the tensor contract
    never carries in the first place (color, roughness, metallic, ``parts``,
    the object's original ``object_id``/category *string*, static objects,
    lights, znear/zfar/aspect, seed/scene_version) are pulled from
    ``template`` -- a ``SceneState`` whose *dynamic* objects are assumed to
    be in the same order ``states``' rows are (true for any ``SceneState``
    produced by ``gltfworld.datagen.sample.sample_scene`` and round-tripped
    through the glTF codec, since object order is preserved end to end).

    Returns a dict: ``{"objects": list[ObjectSpec], "poses": (T, N, 7)
    float32, "lin_vel": (T, N, 3) float32, "ang_vel": (T, N, 3) float32,
    "gravity": (3,) float32, "dt": float, "camera": CameraSpec}``.
    """
    states = np.asarray(states, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    globals_ = np.asarray(globals_, dtype=np.float32)
    if globals_.shape != (GLOBALS_DIM,):
        raise ValueError(f"globals must have shape ({GLOBALS_DIM},), got {globals_.shape}")

    t, n, d = states.shape
    if d != STATE_DIM:
        raise ValueError(f"states last dim must be {STATE_DIM}, got {d}")

    template_dynamic = [obj for obj in template.objects if not obj.is_static]
    if len(template_dynamic) < n:
        raise ValueError(
            f"template has {len(template_dynamic)} dynamic objects, fewer than "
            f"the {n} rows in states"
        )

    poses = np.zeros((t, n, 7), dtype=np.float32)
    lin_vel = np.zeros((t, n, 3), dtype=np.float32)
    ang_vel = np.zeros((t, n, 3), dtype=np.float32)
    objects: list[ObjectSpec] = []

    for row in range(n):
        if not mask[row]:
            objects.append(template_dynamic[row])
            continue

        template_obj = template_dynamic[row]
        pos = states[:, row, 0:3]
        quat = _hemisphere_normalize(states[:, row, 3:7])
        poses[:, row, 0:3] = pos
        poses[:, row, 3:7] = quat
        lin_vel[:, row, :] = states[:, row, 7:10]
        ang_vel[:, row, :] = states[:, row, 10:13]

        shape = _shape_from_onehot(states[0, row, 13:16])
        size = states[0, row, 16:19].copy()
        mass = float(np.exp(states[0, row, 19]))
        friction = float(states[0, row, 20])
        restitution = float(states[0, row, 21])

        objects.append(
            ObjectSpec(
                object_id=template_obj.object_id,
                shape=shape,
                size=size,
                color=template_obj.color,
                roughness=template_obj.roughness,
                metallic=template_obj.metallic,
                mass=mass,
                friction=friction,
                restitution=restitution,
                is_static=False,
                category=template_obj.category,
                parts=template_obj.parts,
            )
        )

    gravity = globals_[0:3].copy()
    dt = float(globals_[3])
    camera_position = globals_[4:7].copy()
    camera_rotation = _hemisphere_normalize(globals_[7:11])
    camera_yfov = float(globals_[11])

    from .scene import CameraSpec

    camera = CameraSpec(
        position=camera_position,
        rotation=camera_rotation,
        yfov=camera_yfov,
        znear=template.camera.znear,
        zfar=template.camera.zfar,
        aspect=template.camera.aspect,
    )

    class_ids_expected = np.array(
        [CATEGORY_TO_CLASS_ID[obj.category] for obj in objects if obj.category in CATEGORY_TO_CLASS_ID],
        dtype=np.int64,
    )
    if class_ids_expected.shape[0] == n and not np.array_equal(class_ids_expected, class_ids):
        raise ValueError("class_ids do not match the reconstructed objects' categories")

    return {
        "objects": objects,
        "poses": poses,
        "lin_vel": lin_vel,
        "ang_vel": ang_vel,
        "gravity": gravity,
        "dt": dt,
        "camera": camera,
    }
