"""V9-prep: articulated-object scene model + transport (``ArticulatedSpec``,
``StateSeries.joint_pos``, ``extras.rwm`` semantics/articulations, the KHR
joint pivot-node encoding in ``gltfworld.scene.convert``).

Pure dataclasses + the glTF codec -- no MuJoCo/``sim`` extra needed (see
``tests/test_articulated_physics.py`` for the MuJoCo-backed physics-sanity
and articulation-consistency checks against real simulated data).
"""

from __future__ import annotations

import socket

import numpy as np
import pytest
from conftest import KHR_SCHEMA_DIR, RWM_SCHEMA_DIR, validate_against_schema

from gltfworld.scene.convert import episode_from_gltf, episode_to_gltf, load_episode, save_episode
from gltfworld.scene.episode import Episode, StateSeries
from gltfworld.scene.scene import ArticulatedSpec, CameraSpec, LightSpec, ObjectSpec, SceneState

_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _axis_angle_quat(axis: int, angle: float) -> np.ndarray:
    k = np.zeros(3, dtype=np.float64)
    k[axis] = 1.0
    return np.array([k[0] * np.sin(angle / 2), k[1] * np.sin(angle / 2), k[2] * np.sin(angle / 2), np.cos(angle / 2)])


def make_articulated_episode(T: int = 10) -> Episode:
    """A small, deterministic, hand-rolled (no MuJoCo) door-hinge episode:
    the door swings open at a constant angular rate, hinged at ``anchor``
    about world Y. Enough to exercise the transport codec end-to-end
    without needing the ``sim`` extra."""
    ground = ObjectSpec(
        object_id=0,
        shape="box",
        size=np.array([2.0, 0.1, 2.0], dtype=np.float32),
        color=np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32),
        roughness=0.9,
        metallic=0.0,
        mass=1000.0,
        friction=0.7,
        restitution=0.05,
        is_static=True,
        category="ground",
    )
    base = ObjectSpec(
        object_id=1,
        shape="box",
        size=np.array([0.3, 0.4, 0.3], dtype=np.float32),
        color=np.array([0.5, 0.35, 0.2, 1.0], dtype=np.float32),
        roughness=0.6,
        metallic=0.0,
        mass=30.0,
        friction=0.6,
        restitution=0.05,
        is_static=True,
        category="cabinet",
    )
    part = ObjectSpec(
        object_id=2,
        shape="box",
        size=np.array([0.02, 0.35, 0.2], dtype=np.float32),
        color=np.array([0.75, 0.6, 0.4, 1.0], dtype=np.float32),
        roughness=0.5,
        metallic=0.0,
        mass=4.0,
        friction=0.5,
        restitution=0.05,
        is_static=False,
        category="door",
    )
    handle = ObjectSpec(
        object_id=3,
        shape="box",
        size=np.array([0.02, 0.02, 0.02], dtype=np.float32),
        color=np.array([0.1, 0.1, 0.1, 1.0], dtype=np.float32),
        roughness=0.3,
        metallic=0.6,
        mass=0.1,
        friction=0.5,
        restitution=0.05,
        is_static=False,
        category="handle",
    )

    anchor = np.array([0.6, 0.4, 0.0], dtype=np.float32)
    articulation = ArticulatedSpec(
        joint_index=0,
        base_object_id=1,
        part_object_id=2,
        joint_type="revolute",
        axis=1,
        min=0.0,
        max=1.4,
        anchor=anchor,
        part_labels=("door",),
        affordances=("openable",),
        handle_object_id=3,
        handle_labels=("handle",),
        handle_affordances=("pullable",),
        base_labels=("cabinet",),
    )

    camera = CameraSpec(
        position=np.array([1.0, 1.2, 2.0], dtype=np.float32),
        rotation=_IDENTITY_QUAT.copy(),
        yfov=0.9,
        znear=0.05,
        zfar=50.0,
        aspect=1.0,
    )
    lights = [
        LightSpec(type="directional", color=np.array([1.0, 1.0, 1.0], dtype=np.float32), intensity=3.0, rotation=_IDENTITY_QUAT.copy())
    ]

    scene = SceneState(
        objects=[ground, base, part, handle],
        camera=camera,
        lights=lights,
        gravity=np.array([0.0, -9.81, 0.0], dtype=np.float32),
        dt=1.0 / 30.0,
        seed=99,
        scene_version="wm-articulated-v1",
        articulations=[articulation],
    )

    handle_local_offset = np.array([0.0, 0.0, 0.02], dtype=np.float32)
    rest_pos = anchor + np.array([0.0, 0.0, 0.3], dtype=np.float32)  # door center at rest (qpos=0)

    times = (np.arange(T, dtype=np.float32) / 30.0).astype(np.float32)
    poses = np.zeros((T, 4, 7), dtype=np.float32)
    joint_pos = np.zeros((T, 1), dtype=np.float32)

    ground_pose = np.array([0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    base_pose = np.array([0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    for t in range(T):
        angle = 0.1 * t  # constant angular rate
        joint_pos[t, 0] = angle
        poses[t, 0] = ground_pose
        poses[t, 1] = base_pose
        rot = _axis_angle_quat(1, angle)
        door_offset = rest_pos - anchor
        # rotate door_offset about anchor by `angle` around axis 1 (Y)
        c, s = np.cos(angle), np.sin(angle)
        rotated = np.array(
            [c * door_offset[0] + s * door_offset[2], door_offset[1], -s * door_offset[0] + c * door_offset[2]]
        )
        door_pos = anchor + rotated
        poses[t, 2, 0:3] = door_pos
        poses[t, 2, 3:7] = rot
        # handle rigidly follows the door
        handle_rot_mat_offset = np.array(
            [c * handle_local_offset[0] + s * handle_local_offset[2], handle_local_offset[1], -s * handle_local_offset[0] + c * handle_local_offset[2]]
        )
        poses[t, 3, 0:3] = door_pos + handle_rot_mat_offset
        poses[t, 3, 3:7] = rot

    series = StateSeries(times=times, poses=poses, joint_pos=joint_pos)
    return Episode(scene=scene, series=series)


@pytest.fixture
def articulated_episode() -> Episode:
    return make_articulated_episode()


# --- scene model validation -----------------------------------------------


def test_articulated_spec_rejects_bad_joint_type():
    with pytest.raises(ValueError):
        ArticulatedSpec(
            joint_index=0, base_object_id=1, part_object_id=2, joint_type="bogus", axis=1, min=0.0, max=1.0,
            anchor=np.zeros(3),
        )


def test_articulated_spec_rejects_bad_axis():
    with pytest.raises(ValueError):
        ArticulatedSpec(
            joint_index=0, base_object_id=1, part_object_id=2, joint_type="revolute", axis=3, min=0.0, max=1.0,
            anchor=np.zeros(3),
        )


def test_scene_state_default_articulations_empty():
    scene = SceneState(
        objects=[],
        camera=CameraSpec(
            position=np.zeros(3), rotation=_IDENTITY_QUAT.copy(), yfov=0.8, znear=0.05, zfar=10.0, aspect=1.0
        ),
        lights=[],
        gravity=np.zeros(3),
        dt=1.0 / 30,
        seed=0,
    )
    assert scene.articulations == []


def test_episode_rejects_joint_count_mismatch():
    ep = make_articulated_episode(T=5)
    bad_series = StateSeries(times=ep.series.times, poses=ep.series.poses, joint_pos=None)
    with pytest.raises(ValueError):
        Episode(scene=ep.scene, series=bad_series)


def test_state_series_joint_pos_shape_validated():
    with pytest.raises(ValueError):
        StateSeries(
            times=np.zeros(3, dtype=np.float32),
            poses=np.zeros((3, 1, 7), dtype=np.float32),
            joint_pos=np.zeros((4, 1), dtype=np.float32),  # wrong T
        )


# --- transport round trip --------------------------------------------------


def test_articulated_episode_roundtrip_in_memory(articulated_episode):
    gltf = episode_to_gltf(articulated_episode)
    decoded = episode_from_gltf(gltf)

    assert len(decoded.scene.articulations) == 1
    art0 = articulated_episode.scene.articulations[0]
    art1 = decoded.scene.articulations[0]
    assert art0.joint_index == art1.joint_index
    assert art0.base_object_id == art1.base_object_id
    assert art0.part_object_id == art1.part_object_id
    assert art0.joint_type == art1.joint_type
    assert art0.axis == art1.axis
    np.testing.assert_allclose(art0.min, art1.min)
    np.testing.assert_allclose(art0.max, art1.max)
    np.testing.assert_allclose(art0.anchor, art1.anchor)
    assert art0.part_labels == art1.part_labels
    assert art0.affordances == art1.affordances
    assert art0.handle_object_id == art1.handle_object_id
    assert art0.handle_labels == art1.handle_labels
    assert art0.handle_affordances == art1.handle_affordances
    assert art0.base_labels == art1.base_labels

    np.testing.assert_array_equal(decoded.series.joint_pos, articulated_episode.series.joint_pos)
    np.testing.assert_array_equal(decoded.series.poses, articulated_episode.series.poses)


def test_articulated_episode_roundtrip_through_glb_file(articulated_episode, tmp_path):
    path = tmp_path / "articulated.glb"
    save_episode(articulated_episode, path)
    decoded = load_episode(path)

    np.testing.assert_array_equal(decoded.series.joint_pos, articulated_episode.series.joint_pos)
    np.testing.assert_array_equal(decoded.series.poses, articulated_episode.series.poses)
    assert len(decoded.scene.articulations) == 1


def test_node_semantics_present_for_base_part_handle(articulated_episode):
    gltf = episode_to_gltf(articulated_episode)
    semantics_by_object_id = {}
    for node in gltf.nodes:
        extras = (node.extras or {}).get("rwm", {})
        if "object_id" in extras and "semantics" in extras:
            semantics_by_object_id[extras["object_id"]] = extras["semantics"]

    assert semantics_by_object_id[1]["labels"] == ["cabinet"]
    assert semantics_by_object_id[2]["labels"] == ["door"]
    assert semantics_by_object_id[2]["affordances"] == ["openable"]
    assert semantics_by_object_id[3]["labels"] == ["handle"]
    assert semantics_by_object_id[3]["affordances"] == ["pullable"]
    # ground has no articulation role -> no semantics key at all
    ground_extras = next(
        (node.extras or {}).get("rwm", {}) for node in gltf.nodes if (node.extras or {}).get("rwm", {}).get("object_id") == 0
    )
    assert "semantics" not in ground_extras


def test_non_articulated_episode_has_no_semantics_key(sample_episode):
    """Backward compatibility: an ordinary (non-articulated) episode's
    ``extras.rwm`` never gains a ``semantics`` key at all."""
    gltf = episode_to_gltf(sample_episode)
    for node in gltf.nodes:
        extras = (node.extras or {}).get("rwm", {})
        if "object_id" in extras:
            assert "semantics" not in extras


def test_scene_roots_exclude_joint_pivot_children(articulated_episode):
    """The two joint-pivot nodes (see gltfworld.scene.convert's
    ``_add_articulation_pivots``) are nested under base/part via
    ``node.children`` and must NOT also appear as scene roots."""
    gltf = episode_to_gltf(articulated_episode)
    root_indices = set(gltf.scenes[gltf.scene].nodes)

    child_indices = set()
    for node in gltf.nodes:
        if node.children:
            child_indices.update(node.children)

    assert child_indices, "expected at least one joint pivot node to be nested"
    assert child_indices.isdisjoint(root_indices)
    # every node still appears exactly once, either as a root or a child
    assert root_indices | child_indices == set(range(len(gltf.nodes)))


def test_non_articulated_scene_roots_unchanged(sample_episode):
    """No articulations -> no children at all -> every node is still a
    scene root, exactly as before V9-prep."""
    gltf = episode_to_gltf(sample_episode)
    assert set(gltf.scenes[gltf.scene].nodes) == set(range(len(gltf.nodes)))


# --- KHR joint schema validation -------------------------------------------


def test_articulated_khr_physics_joints_validate(articulated_episode, khr_schema_registry):
    gltf = episode_to_gltf(articulated_episode)
    doc = gltf.extensions["KHR_physics_rigid_bodies"]
    joints = doc["physicsJoints"]
    assert len(joints) == 1
    for joint in joints:
        validate_against_schema(
            KHR_SCHEMA_DIR / "physics_rigid_bodies" / "glTF.KHR_physics_rigid_bodies.joint.schema.json",
            khr_schema_registry,
            joint,
        )

    found_node_joint = False
    for node in gltf.nodes:
        node_physics = (node.extensions or {}).get("KHR_physics_rigid_bodies")
        if node_physics and "joint" in node_physics:
            validate_against_schema(
                KHR_SCHEMA_DIR / "physics_rigid_bodies" / "node.KHR_physics_rigid_bodies.joint.schema.json",
                khr_schema_registry,
                node_physics["joint"],
            )
            found_node_joint = True
    assert found_node_joint

    # root document itself still validates too
    validate_against_schema(
        KHR_SCHEMA_DIR / "physics_rigid_bodies" / "glTF.KHR_physics_rigid_bodies.schema.json",
        khr_schema_registry,
        doc,
    )


def test_articulated_rwm_joint_position_channel_validates(articulated_episode):
    gltf = episode_to_gltf(articulated_episode)
    doc = gltf.extensions["RWM_state_series"]
    validate_against_schema(RWM_SCHEMA_DIR / "RWM_state_series.schema.json", None, doc)

    joint_channels = [c for c in doc["channels"] if c["kind"] == "joint_position"]
    assert len(joint_channels) == 1
    assert joint_channels[0]["target"] == {"joint": 0}


def test_node_joint_connected_node_points_at_base_pivot(articulated_episode):
    gltf = episode_to_gltf(articulated_episode)
    part_node_index = next(
        i for i, node in enumerate(gltf.nodes) if (node.extras or {}).get("rwm", {}).get("object_id") == 2
    )
    base_node_index = next(
        i for i, node in enumerate(gltf.nodes) if (node.extras or {}).get("rwm", {}).get("object_id") == 1
    )
    part_pivot_index = gltf.nodes[part_node_index].children[0]
    base_pivot_index = gltf.nodes[base_node_index].children[0]

    node_joint = gltf.nodes[part_pivot_index].extensions["KHR_physics_rigid_bodies"]["joint"]
    assert node_joint["connectedNode"] == base_pivot_index


# --- real Khronos glTF-Validator, an articulated sample -------------------


def _network_and_validator_unavailable() -> bool:
    from gltfworld.cli import ensure_validator_binary

    try:
        ensure_validator_binary()
        return False
    except Exception:
        pass
    try:
        socket.create_connection(("github.com", 443), timeout=3).close()
        return False
    except OSError:
        return True


@pytest.mark.skipif(_network_and_validator_unavailable(), reason="no cached glTF-Validator binary and no network to fetch it")
def test_articulated_episode_validates_clean(articulated_episode, tmp_path):
    """Deliverable: a sample articulated GLB (pivot nodes, KHR joints,
    joint_pos channel, semantics extras) passes the real, pinned Khronos
    glTF-Validator with 0 errors -- same acceptance bar as every other
    milestone's transport output (see tests/test_validator.py)."""
    from gltfworld.cli import run_validator

    glb_path = tmp_path / "articulated_episode.glb"
    save_episode(articulated_episode, glb_path)

    report = run_validator(str(glb_path))
    issues = report["issues"]

    assert issues["numErrors"] == 0, issues["messages"]
    warning_codes = {m["code"] for m in issues["messages"] if m["severity"] == 1}
    assert warning_codes <= {"UNSUPPORTED_EXTENSION"}, issues["messages"]
