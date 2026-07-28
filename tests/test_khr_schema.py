"""Our encoded KHR_physics_rigid_bodies / KHR_implicit_shapes / RWM_state_series
extension objects validate against the vendored JSON Schemas.

Schemas are vendored (unmodified) at docs/schemas/khr and docs/schemas/rwm;
see PROVENANCE.md in each for the pinned commit. conftest.py builds a
$id-keyed jsonschema Registry from all of them so cross-file $refs resolve.
"""

from __future__ import annotations

from conftest import KHR_SCHEMA_DIR, RWM_SCHEMA_DIR, make_sample_episode, validate_against_schema

from gltfworld.scene.convert import episode_to_gltf


def test_khr_implicit_shapes_root_validates(sample_episode, khr_schema_registry):
    gltf = episode_to_gltf(sample_episode)
    doc = gltf.extensions["KHR_implicit_shapes"]
    validate_against_schema(
        KHR_SCHEMA_DIR / "implicit_shapes" / "glTF.KHR_implicit_shapes.schema.json",
        khr_schema_registry,
        doc,
    )


def test_khr_implicit_shapes_each_shape_validates(sample_episode, khr_schema_registry):
    gltf = episode_to_gltf(sample_episode)
    shapes = gltf.extensions["KHR_implicit_shapes"]["shapes"]
    assert shapes, "sample episode should produce at least one shape"
    for shape in shapes:
        validate_against_schema(
            KHR_SCHEMA_DIR / "implicit_shapes" / "glTF.KHR_implicit_shapes.shape.schema.json",
            khr_schema_registry,
            shape,
        )


def test_khr_physics_rigid_bodies_root_validates(sample_episode, khr_schema_registry):
    gltf = episode_to_gltf(sample_episode)
    doc = gltf.extensions["KHR_physics_rigid_bodies"]
    validate_against_schema(
        KHR_SCHEMA_DIR / "physics_rigid_bodies" / "glTF.KHR_physics_rigid_bodies.schema.json",
        khr_schema_registry,
        doc,
    )


def test_khr_physics_materials_validate(sample_episode, khr_schema_registry):
    gltf = episode_to_gltf(sample_episode)
    materials = gltf.extensions["KHR_physics_rigid_bodies"]["physicsMaterials"]
    assert materials
    for material in materials:
        validate_against_schema(
            KHR_SCHEMA_DIR / "physics_rigid_bodies" / "glTF.KHR_physics_rigid_bodies.material.schema.json",
            khr_schema_registry,
            material,
        )


def test_khr_physics_node_extensions_validate(sample_episode, khr_schema_registry):
    gltf = episode_to_gltf(sample_episode)
    found_static = False
    found_dynamic = False
    for node in gltf.nodes:
        node_physics = (node.extensions or {}).get("KHR_physics_rigid_bodies")
        if node_physics is None:
            continue
        validate_against_schema(
            KHR_SCHEMA_DIR / "physics_rigid_bodies" / "node.KHR_physics_rigid_bodies.schema.json",
            khr_schema_registry,
            node_physics,
        )
        if "motion" in node_physics:
            found_dynamic = True
        else:
            found_static = True
    # Sanity: the sample episode has both a static ground and dynamic bodies,
    # exercising both branches of the schema (motion present/absent).
    assert found_static and found_dynamic


def test_rwm_state_series_validates(sample_episode):
    gltf = episode_to_gltf(sample_episode)
    doc = gltf.extensions["RWM_state_series"]
    validate_against_schema(RWM_SCHEMA_DIR / "RWM_state_series.schema.json", None, doc)


def test_rwm_state_series_validates_with_all_optional_channels():
    import numpy as np

    from gltfworld.scene.episode import Episode, StateSeries

    base = make_sample_episode(n_objects=2, T=4)
    n = base.series.num_objects
    t = base.series.num_frames
    series = StateSeries(
        times=base.series.times,
        poses=base.series.poses,
        lin_vel=np.zeros((t, n, 3), dtype=np.float32),
        ang_vel=np.zeros((t, n, 3), dtype=np.float32),
        actions=np.zeros((t, 5), dtype=np.float32),  # 5 dims: exercises action chunking
        pose_var=np.zeros((t, n, 7), dtype=np.float32),  # 7 dims: exercises pose_variance chunking
    )
    episode = Episode(scene=base.scene, series=series)

    gltf = episode_to_gltf(episode)
    doc = gltf.extensions["RWM_state_series"]
    validate_against_schema(RWM_SCHEMA_DIR / "RWM_state_series.schema.json", None, doc)

    kinds = {channel["kind"] for channel in doc["channels"]}
    assert kinds == {"linear_velocity", "angular_velocity", "action", "pose_variance"}
    # actions (5 dims) and pose_variance (7 dims) must have been split into
    # multiple same-kind channels tagged with an ascending "component".
    action_channels = [c for c in doc["channels"] if c["kind"] == "action"]
    assert len(action_channels) == 2
    assert {c["component"] for c in action_channels} == {0, 1}
