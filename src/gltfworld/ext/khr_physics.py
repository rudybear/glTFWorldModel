"""Typed codec for the DRAFT ``KHR_physics_rigid_bodies`` + ``KHR_implicit_shapes``
glTF extensions.

Implements the subset gltfworld needs, against the pinned commit recorded in
``DESIGN.md`` under "Pinned specs" (repo: eoineoineoin/glTF_Physics). JSON
schemas for this subset are vendored at ``docs/schemas/khr/`` (see
``docs/schemas/khr/*/PROVENANCE.md`` for repo/commit/date).

Modeled:

- Root ``KHR_implicit_shapes.shapes[]``: sphere / box / cylinder.
- Root ``KHR_physics_rigid_bodies.physicsMaterials[]``: friction + restitution.
- Per-node ``KHR_physics_rigid_bodies``: ``motion`` (initial velocities/mass,
  omitted for static bodies -- a node with no ``motion`` is an immovable
  rigid body per the extension's own semantics) and ``collider`` (geometry
  shape reference + physics material reference).

Everything is read and written as plain dicts (``KhrPhysicsDocument``), so
unknown/extra keys present in a document we didn't author (joints, trigger
volumes, collision filters, capsule/plane shapes, ...) survive a
decode -> encode round trip unchanged; the ``shape_to_object_size`` /
``material_to_friction_restitution`` helpers are read-only typed *views*
over those raw dicts, not a lossy re-parse.

Known deviation (see report): a static (``is_static=True``) ``ObjectSpec``'s
``mass`` value has no home in ``KHR_physics_rigid_bodies`` once ``motion``
is omitted for it -- the extension's own semantics treat a static body's
mass as physically irrelevant. gltfworld still needs it back bit-for-bit
for ``ObjectSpec`` round-tripping, so ``gltfworld.ext.rwm`` additionally
stashes ``mass``/``is_static`` in ``extras.rwm`` per node as the source of
truth for decode; ``KHR_physics_rigid_bodies`` remains a faithful,
schema-valid encoding of the draft extension in its own right.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pygltflib

from gltfworld.scene.scene import ObjectSpec

EXT_IMPLICIT_SHAPES = "KHR_implicit_shapes"
EXT_RIGID_BODIES = "KHR_physics_rigid_bodies"


# --- encode: ObjectSpec -> raw dicts -----------------------------------------


def _shape_key(obj: ObjectSpec) -> tuple:
    return (obj.shape, obj.size.tobytes())


def _shape_dict(obj: ObjectSpec) -> dict[str, Any]:
    if obj.shape == "sphere":
        return {"type": "sphere", "sphere": {"radius": float(obj.size[0])}}
    if obj.shape == "box":
        size = (obj.size.astype(np.float64) * 2.0).tolist()
        return {"type": "box", "box": {"size": [float(v) for v in size]}}
    if obj.shape == "cylinder":
        return {
            "type": "cylinder",
            "cylinder": {
                "height": float(obj.size[1]) * 2.0,
                "radiusBottom": float(obj.size[0]),
                "radiusTop": float(obj.size[2]),
            },
        }
    raise ValueError(f"unknown shape {obj.shape!r}")


def _material_key(obj: ObjectSpec) -> tuple:
    return (float(obj.friction), float(obj.restitution))


def _material_dict(obj: ObjectSpec) -> dict[str, Any]:
    return {
        "staticFriction": float(obj.friction),
        "dynamicFriction": float(obj.friction),
        "restitution": float(obj.restitution),
        "frictionCombine": "average",
        "restitutionCombine": "average",
    }


@dataclass
class KhrPhysicsDocument:
    """Raw (dict-valued) KHR physics state for a whole glTF document.

    Kept as plain dicts (not a lossy typed dataclass) so that decode -> encode
    preserves unknown fields; use the ``shape_to_object_size`` /
    ``material_to_friction_restitution`` module functions for typed reads.
    """

    shapes: list[dict[str, Any]] = field(default_factory=list)
    physics_materials: list[dict[str, Any]] = field(default_factory=list)
    # node index -> raw KHR_physics_rigid_bodies node-extension dict
    node_physics: dict[int, dict[str, Any]] = field(default_factory=dict)


def build_khr_physics(
    objects: list[ObjectSpec],
    node_index_by_object_id: dict[int, int],
    initial_lin_vel: dict[int, np.ndarray] | None = None,
    initial_ang_vel: dict[int, np.ndarray] | None = None,
) -> KhrPhysicsDocument:
    """Encode ``objects`` into root shapes/materials + per-node physics dicts.

    ``initial_lin_vel``/``initial_ang_vel`` map object_id -> (3,) arrays,
    typically ``series.lin_vel[0]``/``series.ang_vel[0]`` when present.
    Shapes and materials are deduplicated by value.
    """
    initial_lin_vel = initial_lin_vel or {}
    initial_ang_vel = initial_ang_vel or {}

    doc = KhrPhysicsDocument()
    shape_index_by_key: dict[tuple, int] = {}
    material_index_by_key: dict[tuple, int] = {}

    for obj in objects:
        shape_key = _shape_key(obj)
        if shape_key not in shape_index_by_key:
            shape_index_by_key[shape_key] = len(doc.shapes)
            doc.shapes.append(_shape_dict(obj))
        shape_index = shape_index_by_key[shape_key]

        material_key = _material_key(obj)
        if material_key not in material_index_by_key:
            material_index_by_key[material_key] = len(doc.physics_materials)
            doc.physics_materials.append(_material_dict(obj))
        material_index = material_index_by_key[material_key]

        node_physics: dict[str, Any] = {
            "collider": {
                "geometry": {"shape": shape_index},
                "physicsMaterial": material_index,
            }
        }
        if not obj.is_static:
            motion: dict[str, Any] = {"mass": float(obj.mass)}
            lin_vel = initial_lin_vel.get(obj.object_id)
            if lin_vel is not None:
                motion["linearVelocity"] = [float(v) for v in lin_vel]
            ang_vel = initial_ang_vel.get(obj.object_id)
            if ang_vel is not None:
                motion["angularVelocity"] = [float(v) for v in ang_vel]
            node_physics["motion"] = motion

        node_index = node_index_by_object_id[obj.object_id]
        doc.node_physics[node_index] = node_physics

    return doc


def write_khr_physics(gltf: pygltflib.GLTF2, doc: KhrPhysicsDocument) -> None:
    """Inject ``doc`` into ``gltf``'s root and per-node extensions dicts."""
    if doc.shapes:
        gltf.extensions.setdefault(EXT_IMPLICIT_SHAPES, {})
        gltf.extensions[EXT_IMPLICIT_SHAPES]["shapes"] = doc.shapes
    if doc.physics_materials:
        gltf.extensions.setdefault(EXT_RIGID_BODIES, {})
        gltf.extensions[EXT_RIGID_BODIES]["physicsMaterials"] = doc.physics_materials

    for node_index, node_physics in doc.node_physics.items():
        node = gltf.nodes[node_index]
        if node.extensions is None:
            node.extensions = {}
        node.extensions[EXT_RIGID_BODIES] = node_physics


# --- decode: raw dicts -> typed views -----------------------------------------


def read_khr_physics(gltf: pygltflib.GLTF2) -> KhrPhysicsDocument:
    """Read back the raw KHR physics dicts from ``gltf``."""
    shapes = list(gltf.extensions.get(EXT_IMPLICIT_SHAPES, {}).get("shapes", []))
    physics_materials = list(gltf.extensions.get(EXT_RIGID_BODIES, {}).get("physicsMaterials", []))

    node_physics: dict[int, dict[str, Any]] = {}
    for index, node in enumerate(gltf.nodes):
        extensions = node.extensions or {}
        if EXT_RIGID_BODIES in extensions:
            node_physics[index] = extensions[EXT_RIGID_BODIES]

    return KhrPhysicsDocument(shapes=shapes, physics_materials=physics_materials, node_physics=node_physics)


def shape_to_object_size(shape_dict: dict[str, Any]) -> tuple[str, np.ndarray]:
    """Typed view: (ObjectSpec.shape name, ObjectSpec.size convention) for a shape dict."""
    shape_type = shape_dict["type"]
    if shape_type == "sphere":
        r = float(shape_dict["sphere"]["radius"])
        return "sphere", np.array([r, r, r], dtype=np.float32)
    if shape_type == "box":
        size = shape_dict["box"]["size"]
        half = [float(v) / 2.0 for v in size]
        return "box", np.array(half, dtype=np.float32)
    if shape_type == "cylinder":
        cyl = shape_dict["cylinder"]
        half_height = float(cyl["height"]) / 2.0
        radius_bottom = float(cyl["radiusBottom"])
        radius_top = float(cyl.get("radiusTop", radius_bottom))
        return "cylinder", np.array([radius_bottom, half_height, radius_top], dtype=np.float32)
    raise ValueError(f"unsupported/unimplemented shape type {shape_type!r}")


def material_to_friction_restitution(material_dict: dict[str, Any]) -> tuple[float, float]:
    """Typed view: (friction, restitution) for a physics material dict.

    ObjectSpec has a single ``friction`` field; we write staticFriction ==
    dynamicFriction on encode, so either is an equally valid read.
    """
    friction = float(material_dict.get("dynamicFriction", material_dict.get("staticFriction", 0.6)))
    restitution = float(material_dict.get("restitution", 0.0))
    return friction, restitution


def node_motion_mass(node_physics: dict[str, Any]) -> float | None:
    """Typed view: mass from a node's ``motion``, or None if the node has no motion (static)."""
    motion = node_physics.get("motion")
    if motion is None:
        return None
    return float(motion.get("mass", 0.0))


def node_motion_velocities(node_physics: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Typed view: (linearVelocity, angularVelocity) from a node's ``motion``, each (3,) float32 or None."""
    motion = node_physics.get("motion")
    if motion is None:
        return None, None
    lin = motion.get("linearVelocity")
    ang = motion.get("angularVelocity")
    lin_arr = np.array(lin, dtype=np.float32) if lin is not None else None
    ang_arr = np.array(ang, dtype=np.float32) if ang is not None else None
    return lin_arr, ang_arr
