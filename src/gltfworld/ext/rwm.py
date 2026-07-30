"""Codec for the custom vendor extension ``RWM_state_series`` + ``extras.rwm``.

Everything glTF (plus the draft KHR physics extensions) can't express rides
here: per-frame velocities, actions, and pose variance (root
``extensions.RWM_state_series``), and object/scene bookkeeping that has no
standard glTF home (``extras.rwm``).

See ``docs/RWM_EXTENSIONS.md`` for the V9-prep ``joint_position`` channel
and the ``extras.rwm`` articulation/semantics fields' full write-up.

``RWM_state_series`` layout::

    extensions.RWM_state_series = {
        "version": "0.1",
        "timesAccessor": <accessor index, shared with the pose animation>,
        "channels": [
            {"target": {"node": <node index>} | {"joint": <physicsJoints index>} | "world",
             "kind": "linear_velocity" | "angular_velocity" | "action" | "pose_variance"
                     | "joint_position",
             "accessor": <accessor index>,
             "component": <int, only present when a value's feature dim > 4
                           and had to be split across multiple VECn accessors>},
            ...
        ],
    }

``joint_position`` channels (V9-prep, one per articulated joint) use
``target = {"joint": i}``, ``i`` indexing the same
``KHR_physics_rigid_bodies.physicsJoints`` array
(``gltfworld.ext.khr_physics``) that the joint's node-level ``joint``
property references -- distinct from the ``{"node": ...}``/``"world"``
targets every other channel kind uses, since a joint isn't itself a node.
Each carries a single SCALAR value per frame (radians for a revolute joint,
meters for a prismatic one -- matching ``ArticulatedSpec.min``/``max``'s own
units), so (unlike ``action``/``pose_variance``) it never needs
multi-accessor chunking.

A channel's accessor always has ``count == len(times)``; its type is SCALAR/
VEC2/VEC3/VEC4 depending on how many feature dims it carries. Values whose
natural width exceeds 4 (pose_variance is 7-wide: 3 position + 4 quaternion;
actions can be arbitrary width) are split into multiple channels of the same
``kind``, each holding up to 4 contiguous feature dims, tagged with a
0-based ``component`` chunk index so decode can concatenate them back in
order. Per-node ``extras.rwm`` additionally carries ``mass``/``is_static``
beyond the plain {object_id, category, parts, schema_version} shape --
documented as a deviation in the milestone report -- so those two
``ObjectSpec`` fields survive a static (KHR_physics_rigid_bodies drops
"motion", hence mass, for static bodies) round trip bit-for-bit. Per-scene
``extras.rwm`` similarly carries ``gravity`` beyond {seed, scene_version,
dt}, since the pinned KHR_physics_rigid_bodies commit has no root gravity
property.

See ``docs/schemas/rwm/RWM_state_series.schema.json`` for the JSON Schema
this module's output is validated against in tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pygltflib

from gltfworld.gltf.accessors import BufferAccumulator, read_accessor
from gltfworld.scene.episode import StateSeries

EXT_STATE_SERIES = "RWM_state_series"
RWM_VERSION = "0.1"
RWM_SCHEMA_VERSION = "0.1"

KIND_LINEAR_VELOCITY = "linear_velocity"
KIND_ANGULAR_VELOCITY = "angular_velocity"
KIND_ACTION = "action"
KIND_POSE_VARIANCE = "pose_variance"
KIND_JOINT_POSITION = "joint_position"

_MAX_CHUNK_WIDTH = 4


def _chunks(total: int) -> list[tuple[int, int]]:
    """Split a feature dimension of size ``total`` into (start, width<=4) chunks."""
    out = []
    start = 0
    while start < total:
        width = min(_MAX_CHUNK_WIDTH, total - start)
        out.append((start, width))
        start += width
    return out


def build_state_series(
    gltf: pygltflib.GLTF2,
    accumulator: BufferAccumulator,
    times_accessor: int,
    node_index_by_object_id: dict[int, int],
    object_ids_in_order: list[int],
    series: StateSeries,
) -> dict[str, Any]:
    """Build the ``RWM_state_series`` root extension dict for ``series``.

    ``object_ids_in_order`` must match the object (N) axis order of
    ``series.poses``. Accessor data is appended to ``accumulator`` (not yet
    finalized -- call ``accumulator.finalize(gltf)`` once, after all
    extensions have added their data).
    """
    channels: list[dict[str, Any]] = []

    def _add_per_node_channel(kind: str, data: np.ndarray) -> None:
        # data: (T, N, F)
        f = data.shape[2]
        chunk_layout = _chunks(f)
        for obj_index, object_id in enumerate(object_ids_in_order):
            node_index = node_index_by_object_id[object_id]
            for start, width in chunk_layout:
                chunk = np.ascontiguousarray(data[:, obj_index, start : start + width])
                accessor_index = accumulator.add_accessor(gltf, chunk)
                channel: dict[str, Any] = {
                    "target": {"node": node_index},
                    "kind": kind,
                    "accessor": accessor_index,
                }
                if len(chunk_layout) > 1:
                    channel["component"] = start // _MAX_CHUNK_WIDTH
                channels.append(channel)

    if series.lin_vel is not None:
        _add_per_node_channel(KIND_LINEAR_VELOCITY, series.lin_vel)
    if series.ang_vel is not None:
        _add_per_node_channel(KIND_ANGULAR_VELOCITY, series.ang_vel)
    if series.pose_var is not None:
        _add_per_node_channel(KIND_POSE_VARIANCE, series.pose_var)

    if series.actions is not None:
        a = series.actions.shape[1]
        for start, width in _chunks(a):
            chunk = np.ascontiguousarray(series.actions[:, start : start + width])
            accessor_index = accumulator.add_accessor(gltf, chunk)
            channel = {
                "target": "world",
                "kind": KIND_ACTION,
                "accessor": accessor_index,
            }
            if len(_chunks(a)) > 1:
                channel["component"] = start // _MAX_CHUNK_WIDTH
            channels.append(channel)

    if series.joint_pos is not None:
        j = series.joint_pos.shape[1]
        for joint_index in range(j):
            chunk = np.ascontiguousarray(series.joint_pos[:, joint_index : joint_index + 1])
            accessor_index = accumulator.add_accessor(gltf, chunk)
            channels.append(
                {
                    "target": {"joint": joint_index},
                    "kind": KIND_JOINT_POSITION,
                    "accessor": accessor_index,
                }
            )

    return {
        "version": RWM_VERSION,
        "timesAccessor": times_accessor,
        "channels": channels,
    }


def write_state_series(gltf: pygltflib.GLTF2, doc: dict[str, Any]) -> None:
    gltf.extensions[EXT_STATE_SERIES] = doc


def read_state_series(gltf: pygltflib.GLTF2) -> dict[str, Any] | None:
    return gltf.extensions.get(EXT_STATE_SERIES)


def decode_channels(
    gltf: pygltflib.GLTF2,
    doc: dict[str, Any],
    node_index_by_object_id: dict[int, int],
    object_ids_in_order: list[int],
    num_frames: int,
) -> dict[str, np.ndarray]:
    """Reconstruct lin_vel/ang_vel/actions/pose_var arrays from a decoded doc.

    Returns a dict with any of the keys ``lin_vel``, ``ang_vel``, ``actions``,
    ``pose_var`` present in ``doc``'s channels, each float32 with shape
    ``(T, N, F)`` (or ``(T, A)`` for actions).
    """
    n = len(object_ids_in_order)
    node_index_to_obj_position = {
        node_index_by_object_id[oid]: i for i, oid in enumerate(object_ids_in_order)
    }

    per_node_widths = {KIND_LINEAR_VELOCITY: 3, KIND_ANGULAR_VELOCITY: 3, KIND_POSE_VARIANCE: 7}
    per_node_chunks: dict[str, dict[int, list[tuple[int, np.ndarray]]]] = {
        kind: {} for kind in per_node_widths
    }
    action_chunks: list[tuple[int, np.ndarray]] = []
    joint_pos_by_index: dict[int, np.ndarray] = {}

    for channel in doc.get("channels", []):
        kind = channel["kind"]
        accessor_index = channel["accessor"]
        component = channel.get("component", 0)
        data = read_accessor(gltf, accessor_index)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if kind == KIND_ACTION:
            action_chunks.append((component, data))
            continue

        if kind == KIND_JOINT_POSITION:
            joint_index = channel["target"]["joint"]
            joint_pos_by_index[joint_index] = data[:, 0]
            continue

        target = channel["target"]
        node_index = target["node"]
        obj_position = node_index_to_obj_position[node_index]
        per_node_chunks[kind].setdefault(obj_position, []).append((component, data))

    out: dict[str, np.ndarray] = {}

    for kind, width in per_node_widths.items():
        chunks_by_obj = per_node_chunks[kind]
        if not chunks_by_obj:
            continue
        arr = np.zeros((num_frames, n, width), dtype=np.float32)
        for obj_position, chunks in chunks_by_obj.items():
            chunks.sort(key=lambda c: c[0])
            arr[:, obj_position, :] = np.concatenate([c[1] for c in chunks], axis=1)
        key = {
            KIND_LINEAR_VELOCITY: "lin_vel",
            KIND_ANGULAR_VELOCITY: "ang_vel",
            KIND_POSE_VARIANCE: "pose_var",
        }[kind]
        out[key] = arr

    if action_chunks:
        action_chunks.sort(key=lambda c: c[0])
        out["actions"] = np.concatenate([c[1] for c in action_chunks], axis=1).astype(np.float32)

    if joint_pos_by_index:
        j = max(joint_pos_by_index) + 1
        arr = np.zeros((num_frames, j), dtype=np.float32)
        for joint_index, data in joint_pos_by_index.items():
            arr[:, joint_index] = data
        out["joint_pos"] = arr

    return out


# --- extras.rwm ---------------------------------------------------------------


def node_extras(
    object_id: int,
    category: str,
    parts: dict[str, Any],
    mass: float,
    is_static: bool,
    semantics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``extras.rwm`` dict for one object's node.

    ``semantics`` (V9-prep, optional -- omitted entirely for ordinary,
    non-articulated objects, so this is fully backward compatible) carries
    the robotics-oriented part/affordance labels for articulated assemblies:
    ``{"labels": ["door"], "affordances": ["openable"]}``. See
    ``docs/RWM_EXTENSIONS.md``'s semantics taxonomy.
    """
    extras: dict[str, Any] = {
        "object_id": int(object_id),
        "category": str(category),
        "parts": parts,
        "schema_version": RWM_SCHEMA_VERSION,
        "mass": float(mass),
        "is_static": bool(is_static),
    }
    if semantics is not None:
        extras["semantics"] = semantics
    return extras


def node_semantics(labels: tuple[str, ...], affordances: tuple[str, ...] = ()) -> dict[str, Any]:
    """Build the ``extras.rwm.semantics`` dict for one articulated object's node."""
    return {"labels": list(labels), "affordances": list(affordances)}


def scene_extras(
    seed: int,
    scene_version: str,
    dt: float,
    gravity: np.ndarray,
    articulations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the ``extras.rwm`` dict for the glTF scene.

    ``gravity`` is included in addition to the {seed, scene_version, dt}
    documented above -- another deviation in the same spirit as node
    extras' mass/is_static: ``KHR_physics_rigid_bodies`` (this pinned
    commit) has no root-level gravity property, so it has no other home.

    ``articulations`` (V9-prep, optional -- omitted when empty) is the list
    of raw ``ArticulatedSpec`` dicts (see
    ``gltfworld.scene.convert.articulation_to_extras``/
    ``articulation_from_extras``): gltfworld's own decode trusts *this*, not
    a re-derivation from the encoded KHR joint dicts, as the source of truth
    for ``SceneState.articulations`` -- the same "extras.rwm is the decode
    source of truth; KHR_* remains a faithful, schema-valid encoding of the
    draft extension in its own right" pattern already used for
    ``mass``/``is_static``/``gravity`` above.
    """
    extras: dict[str, Any] = {
        "seed": int(seed),
        "scene_version": str(scene_version),
        "dt": float(dt),
        "gravity": [float(v) for v in gravity],
    }
    if articulations:
        extras["articulations"] = articulations
    return extras
