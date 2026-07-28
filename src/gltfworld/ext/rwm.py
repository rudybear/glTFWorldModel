"""Codec for the custom vendor extension ``RWM_state_series`` + ``extras.rwm``.

Everything glTF (plus the draft KHR physics extensions) can't express rides
here: per-frame velocities, actions, and pose variance (root
``extensions.RWM_state_series``), and object/scene bookkeeping that has no
standard glTF home (``extras.rwm``).

``RWM_state_series`` layout::

    extensions.RWM_state_series = {
        "version": "0.1",
        "timesAccessor": <accessor index, shared with the pose animation>,
        "channels": [
            {"target": {"node": <node index>} | "world",
             "kind": "linear_velocity" | "angular_velocity" | "action" | "pose_variance",
             "accessor": <accessor index>,
             "component": <int, only present when a value's feature dim > 4
                           and had to be split across multiple VECn accessors>},
            ...
        ],
    }

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

    return out


# --- extras.rwm ---------------------------------------------------------------


def node_extras(
    object_id: int,
    category: str,
    parts: dict[str, Any],
    mass: float,
    is_static: bool,
) -> dict[str, Any]:
    """Build the ``extras.rwm`` dict for one object's node."""
    return {
        "object_id": int(object_id),
        "category": str(category),
        "parts": parts,
        "schema_version": RWM_SCHEMA_VERSION,
        "mass": float(mass),
        "is_static": bool(is_static),
    }


def scene_extras(seed: int, scene_version: str, dt: float, gravity: np.ndarray) -> dict[str, Any]:
    """Build the ``extras.rwm`` dict for the glTF scene.

    ``gravity`` is included in addition to the {seed, scene_version, dt}
    documented above -- another deviation in the same spirit as node
    extras' mass/is_static: ``KHR_physics_rigid_bodies`` (this pinned
    commit) has no root-level gravity property, so it has no other home.
    """
    return {
        "seed": int(seed),
        "scene_version": str(scene_version),
        "dt": float(dt),
        "gravity": [float(v) for v in gravity],
    }
