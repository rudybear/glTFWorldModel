"""Cross-checks that the animation, node TRS, and RWM/KHR data all agree with
``series`` -- they share provenance (all derived from the same Episode), so
this catches encode bugs where one representation silently diverges from
another."""

from __future__ import annotations

import numpy as np

from gltfworld.gltf.accessors import read_accessor
from gltfworld.scene.convert import episode_to_gltf


def test_animation_channel_data_equals_series_poses(sample_episode):
    gltf = episode_to_gltf(sample_episode)
    animation = gltf.animations[0]

    object_nodes = [i for i, node in enumerate(gltf.nodes) if (node.extras or {}).get("rwm", {}).get("object_id") is not None]
    node_to_obj_position = {node_index: i for i, node_index in enumerate(object_nodes)}

    n = len(object_nodes)
    t = sample_episode.series.num_frames
    decoded = np.zeros((t, n, 7), dtype=np.float32)

    for channel in animation.channels:
        node_index = channel.target.node
        if node_index not in node_to_obj_position:
            continue
        obj_position = node_to_obj_position[node_index]
        sampler = animation.samplers[channel.sampler]
        data = read_accessor(gltf, sampler.output)
        if channel.target.path == "translation":
            decoded[:, obj_position, 0:3] = data
        elif channel.target.path == "rotation":
            decoded[:, obj_position, 3:7] = data

    np.testing.assert_array_equal(decoded, sample_episode.series.poses)


def test_animation_samplers_use_step_interpolation(sample_episode):
    gltf = episode_to_gltf(sample_episode)
    animation = gltf.animations[0]
    assert animation.samplers, "expected at least one animation sampler"
    for sampler in animation.samplers:
        assert sampler.interpolation == "STEP"


def test_frame0_node_trs_equals_poses0(sample_episode):
    gltf = episode_to_gltf(sample_episode)
    poses0 = sample_episode.series.poses[0]

    for i, obj in enumerate(sample_episode.scene.objects):
        node_index = next(
            idx
            for idx, node in enumerate(gltf.nodes)
            if (node.extras or {}).get("rwm", {}).get("object_id") == obj.object_id
        )
        node = gltf.nodes[node_index]
        np.testing.assert_array_equal(np.array(node.translation, dtype=np.float32), poses0[i, 0:3])
        np.testing.assert_array_equal(np.array(node.rotation, dtype=np.float32), poses0[i, 3:7])


def test_all_animation_channels_share_one_time_accessor(sample_episode):
    gltf = episode_to_gltf(sample_episode)
    animation = gltf.animations[0]
    input_accessors = {sampler.input for sampler in animation.samplers}
    assert len(input_accessors) == 1

    rwm_doc = gltf.extensions["RWM_state_series"]
    assert rwm_doc["timesAccessor"] in input_accessors


def test_extensions_required_is_absent_or_empty(sample_episode):
    gltf = episode_to_gltf(sample_episode)
    assert not gltf.extensionsRequired


def test_extensions_used_matches_what_is_written(sample_episode):
    gltf = episode_to_gltf(sample_episode)
    used = set(gltf.extensionsUsed)
    assert "RWM_state_series" in used
    assert "KHR_physics_rigid_bodies" in used
    assert "KHR_implicit_shapes" in used
    assert "KHR_lights_punctual" in used  # sample_episode has lights

    # every extension actually referenced somewhere is declared
    referenced = set(gltf.extensions.keys())
    for node in gltf.nodes:
        referenced |= set((node.extensions or {}).keys())
    assert referenced <= used


def test_rwm_channels_reference_valid_node_and_accessor_indices(sample_episode):
    gltf = episode_to_gltf(sample_episode)
    doc = gltf.extensions["RWM_state_series"]
    for channel in doc["channels"]:
        assert 0 <= channel["accessor"] < len(gltf.accessors)
        target = channel["target"]
        if isinstance(target, dict):
            assert 0 <= target["node"] < len(gltf.nodes)
        else:
            assert target == "world"


def test_khr_physics_node_geometry_references_valid_shape_and_material(sample_episode):
    gltf = episode_to_gltf(sample_episode)
    shapes = gltf.extensions["KHR_implicit_shapes"]["shapes"]
    materials = gltf.extensions["KHR_physics_rigid_bodies"]["physicsMaterials"]
    for node in gltf.nodes:
        node_physics = (node.extensions or {}).get("KHR_physics_rigid_bodies")
        if node_physics is None:
            continue
        collider = node_physics["collider"]
        assert 0 <= collider["geometry"]["shape"] < len(shapes)
        assert 0 <= collider["physicsMaterial"] < len(materials)
