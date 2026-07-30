"""V9-prep: KHR_physics_rigid_bodies joint codec (``gltfworld.ext.khr_physics``).

Covers the subset of the pinned spec's joint machinery this milestone
implements (hinge via a limited rotational DOF, slider via a limited linear
DOF) -- schema validation against the vendored (already-present since V1,
see ``docs/schemas/khr/PROVENANCE.md``) joint schemas, plus the node-level
``joint`` property and the root ``physicsJoints[]`` array.
"""

from __future__ import annotations

from conftest import KHR_SCHEMA_DIR, validate_against_schema

from gltfworld.ext import khr_physics


def test_hinge_joint_limits_matches_spec_worked_example():
    """The pinned spec README's own worked example: "adding a 3D linear
    limit with zero maximum distance, a 1D angular limit with min/max
    describing the swing ... and a 2D angular limit with zero limits about
    the remaining two axes"."""
    limits = khr_physics.hinge_joint_limits(axis=1, min_rad=0.0, max_rad=1.4)
    assert len(limits) == 3

    linear = next(l for l in limits if "linearAxes" in l)
    assert sorted(linear["linearAxes"]) == [0, 1, 2]
    assert linear["min"] == 0.0
    assert linear["max"] == 0.0

    angular_locked = next(l for l in limits if "angularAxes" in l and len(l["angularAxes"]) == 2)
    assert sorted(angular_locked["angularAxes"]) == [0, 2]
    assert angular_locked["min"] == 0.0
    assert angular_locked["max"] == 0.0

    angular_swing = next(l for l in limits if l.get("angularAxes") == [1])
    assert angular_swing["min"] == 0.0
    assert angular_swing["max"] == 1.4


def test_slider_joint_limits_translation_rotation_swapped():
    limits = khr_physics.slider_joint_limits(axis=0, min_m=0.0, max_m=0.3)
    assert len(limits) == 3

    angular = next(l for l in limits if "angularAxes" in l)
    assert sorted(angular["angularAxes"]) == [0, 1, 2]
    assert angular["min"] == 0.0 and angular["max"] == 0.0

    linear_locked = next(l for l in limits if "linearAxes" in l and len(l["linearAxes"]) == 2)
    assert sorted(linear_locked["linearAxes"]) == [1, 2]

    linear_slide = next(l for l in limits if l.get("linearAxes") == [0])
    assert linear_slide["min"] == 0.0
    assert linear_slide["max"] == 0.3


def test_joint_limits_validate_against_vendored_schema(khr_schema_registry):
    for limit in khr_physics.hinge_joint_limits(axis=2, min_rad=-0.1, max_rad=1.9):
        validate_against_schema(
            KHR_SCHEMA_DIR / "physics_rigid_bodies" / "glTF.KHR_physics_rigid_bodies.joint.limit.schema.json",
            khr_schema_registry,
            limit,
        )
    for limit in khr_physics.slider_joint_limits(axis=0, min_m=0.0, max_m=0.4):
        validate_against_schema(
            KHR_SCHEMA_DIR / "physics_rigid_bodies" / "glTF.KHR_physics_rigid_bodies.joint.limit.schema.json",
            khr_schema_registry,
            limit,
        )


def test_joint_dict_validates_against_vendored_schema(khr_schema_registry):
    joint_dict = khr_physics.build_joint_dict(khr_physics.hinge_joint_limits(1, 0.0, 1.2))
    validate_against_schema(
        KHR_SCHEMA_DIR / "physics_rigid_bodies" / "glTF.KHR_physics_rigid_bodies.joint.schema.json",
        khr_schema_registry,
        joint_dict,
    )


def test_node_joint_property_validates_against_vendored_schema(khr_schema_registry):
    node_joint = khr_physics.node_joint_property(connected_node=3, joint_index=0)
    validate_against_schema(
        KHR_SCHEMA_DIR / "physics_rigid_bodies" / "node.KHR_physics_rigid_bodies.joint.schema.json",
        khr_schema_registry,
        node_joint,
    )
    assert node_joint == {"connectedNode": 3, "joint": 0, "enableCollision": False}


def test_joint_limit_requires_linear_or_angular_axes_per_schema(khr_schema_registry):
    """Sanity: an object with neither ``linearAxes`` nor ``angularAxes`` (or
    both) is rejected -- confirms the schema's own ``oneOf`` constraint is
    actually being exercised, not vacuously satisfied."""
    import jsonschema
    import pytest

    bad = {"min": 0.0, "max": 0.0}
    with pytest.raises(jsonschema.ValidationError):
        validate_against_schema(
            KHR_SCHEMA_DIR / "physics_rigid_bodies" / "glTF.KHR_physics_rigid_bodies.joint.limit.schema.json",
            khr_schema_registry,
            bad,
        )


def test_khr_physics_document_round_trips_joints(sample_episode):
    """``KhrPhysicsDocument.joints``/node ``joint`` property survive a raw
    write -> read cycle through ``gltfworld.ext.khr_physics``'s own
    read/write (independent of the higher-level Episode codec, which is
    covered by ``tests/test_articulated_scene.py``)."""
    from gltfworld.scene.convert import episode_to_gltf

    gltf = episode_to_gltf(sample_episode)
    doc = khr_physics.read_khr_physics(gltf)
    assert doc.joints == []  # sample_episode has no articulations

    # Inject a synthetic joint + node property, then round-trip.
    doc.joints.append(khr_physics.build_joint_dict(khr_physics.hinge_joint_limits(1, 0.0, 1.0)))
    node_index = 0
    doc.node_physics.setdefault(node_index, {})
    doc.node_physics[node_index]["joint"] = khr_physics.node_joint_property(1, 0)
    khr_physics.write_khr_physics(gltf, doc)

    doc2 = khr_physics.read_khr_physics(gltf)
    assert doc2.joints == doc.joints
    assert doc2.node_physics[node_index]["joint"] == {"connectedNode": 1, "joint": 0, "enableCollision": False}
