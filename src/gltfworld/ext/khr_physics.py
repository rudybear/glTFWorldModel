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

V9-prep: joints
----------------

Also modeled (added for V9-prep, articulated objects): root-level
``KHR_physics_rigid_bodies.physicsJoints[]`` (each a set of ``limits``, per
``glTF.KHR_physics_rigid_bodies.joint.schema.json``/``...joint.limit...``)
and the per-node ``joint`` property (``node.KHR_physics_rigid_bodies.joint``:
``connectedNode`` + ``joint`` index + ``enableCollision``), against the
*same* pinned commit's already-vendored joint schemas (they were fetched
alongside the rest of ``physics_rigid_bodies/schema/*.json`` back in V1 --
see ``docs/schemas/khr/PROVENANCE.md`` -- nothing new to vendor).

Per the pinned spec text (``eoineoineoin/glTF_Physics`` README, "Joints"
section): a joint's two attachment frames are each defined by "the relative
transform between the node and the first parent motion (or the simulation's
fixed reference frame, if no such motion exists)". gltfworld's articulated
objects are flat, top-level nodes (no scene-graph nesting under a common
parent, matching every other gltfworld object) with **dedicated,
motion-less, geometry-less child "joint pivot" nodes** nested one level
under each of ``base``/``part`` (see ``gltfworld.scene.convert``): the
part-side pivot's ancestor-with-motion is the part's own rigid-body node, so
its attachment frame is exactly its fixed local offset within that body
(correctly co-moving as the body's ``motion`` state changes); the base-side
pivot's nearest ancestor has no ``motion`` (base is static), so its
attachment frame is exactly its fixed offset from the world origin. Both
pivots are placed at ``ArticulatedSpec.anchor`` (the shared joint anchor
point, in each body's *own* local frame) with identity local rotation
(gltfworld's articulated scenes keep ``base``/``part`` world-aligned at
rest, see ``ArticulatedSpec``'s docstring), which is exactly what makes a
**hinge** = "3D linear limit, min=max=0" + "1D angular limit about the hinge
axis" + "2D angular limit, min=max=0 about the other two axes" (the pinned
spec's own worked example) and a **slider** = the same pattern with
linear/angular swapped, both literally correct (not approximate) per
:func:`hinge_joint_limits`/:func:`slider_joint_limits`.

Honest gap (see DESIGN.md's V9-prep gap notes): ``joint.limit`` only carries
``stiffness``/``damping`` for the *restorative* force applied once a limit
is exceeded (a soft stop), not a viscous damping coefficient active across
the whole free range the way MJCF/URDF joint ``damping`` (or ``armature``)
is -- there is no way to express "this hinge always has some viscous drag
while swinging, even within its limits" in this pinned commit's joint
schema. Likewise ``joint.drive`` models a persistent, always-on
spring-to-target (position/velocity), not a *finite-duration* external push
impulse -- gltfworld's MJCF "scripted push" actuation (see
``gltfworld.datagen.articulated``) is therefore not encoded as a KHR
``drive`` at all (it would misrepresent a one-shot push as a permanently
active motor); the driving force only exists inside the MuJoCo simulation
that produced the recorded ``poses``/``joint_pos`` trajectory.
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
    # root-level physicsJoints[] (V9-prep)
    joints: list[dict[str, Any]] = field(default_factory=list)


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
    if doc.joints:
        gltf.extensions.setdefault(EXT_RIGID_BODIES, {})
        gltf.extensions[EXT_RIGID_BODIES]["physicsJoints"] = doc.joints

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
    joints = list(gltf.extensions.get(EXT_RIGID_BODIES, {}).get("physicsJoints", []))

    node_physics: dict[int, dict[str, Any]] = {}
    for index, node in enumerate(gltf.nodes):
        extensions = node.extensions or {}
        if EXT_RIGID_BODIES in extensions:
            node_physics[index] = extensions[EXT_RIGID_BODIES]

    return KhrPhysicsDocument(
        shapes=shapes, physics_materials=physics_materials, node_physics=node_physics, joints=joints
    )


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


# --- V9-prep: joints ------------------------------------------------------------
#
# See the module docstring's "V9-prep: joints" section for why the limit
# shapes below (3D + 1D, or 3D + 2D) are exactly the pinned spec's own
# worked hinge example, not an invented approximation.

_ALL_AXES = (0, 1, 2)


def hinge_joint_limits(axis: int, min_rad: float, max_rad: float) -> list[dict[str, Any]]:
    """Limits for a revolute (hinge) joint swinging about local axis ``axis``
    (0=X, 1=Y, 2=Z), within ``[min_rad, max_rad]`` radians.

    Exactly the pinned spec README's own worked example: "a hinged door can
    be constructed by locating the attachment frames at the point where the
    physical hinge would be on each body, adding a 3D linear limit with zero
    maximum distance, a 1D angular limit with min/max describing the swing
    of the door around it's vertical axis, and a 2D angular limit with zero
    limits about the remaining two axes."
    """
    other_axes = [a for a in _ALL_AXES if a != axis]
    return [
        {"linearAxes": list(_ALL_AXES), "min": 0.0, "max": 0.0},
        {"angularAxes": other_axes, "min": 0.0, "max": 0.0},
        {"angularAxes": [axis], "min": float(min_rad), "max": float(max_rad)},
    ]


def slider_joint_limits(axis: int, min_m: float, max_m: float) -> list[dict[str, Any]]:
    """Limits for a prismatic (slider) joint translating along local axis
    ``axis`` (0=X, 1=Y, 2=Z), within ``[min_m, max_m]`` meters.

    The natural translation/rotation-swapped analog of
    :func:`hinge_joint_limits`'s spec-given hinge construction (not itself
    spelled out verbatim in the pinned spec text, but a direct application
    of the same limit-composition primitives it describes): lock all
    rotation (3D angular limit at zero), lock translation on the two
    off-axis directions (2D linear limit at zero), and constrain the slide
    axis to ``[min_m, max_m]`` (1D linear limit).
    """
    other_axes = [a for a in _ALL_AXES if a != axis]
    return [
        {"angularAxes": list(_ALL_AXES), "min": 0.0, "max": 0.0},
        {"linearAxes": other_axes, "min": 0.0, "max": 0.0},
        {"linearAxes": [axis], "min": float(min_m), "max": float(max_m)},
    ]


def build_joint_dict(limits: list[dict[str, Any]]) -> dict[str, Any]:
    """A root-level ``physicsJoints[]`` entry (``glTF.KHR_physics_rigid_bodies.joint``)."""
    return {"limits": limits}


def node_joint_property(connected_node: int, joint_index: int, enable_collision: bool = False) -> dict[str, Any]:
    """A node's ``joint`` property (``node.KHR_physics_rigid_bodies.joint``)."""
    return {
        "connectedNode": int(connected_node),
        "joint": int(joint_index),
        "enableCollision": bool(enable_collision),
    }
