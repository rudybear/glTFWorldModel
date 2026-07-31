"""Unit tests for ``gltfworld.data.pack_articulated`` (needs the ``sim``
extra to generate episodes to pack -- no rendering, CPU-only): packed tensor
shapes/values match the source episodes' own ``ArticulatedSpec``/
``joint_pos``, the split scheme agrees with ``gltfworld.data.pack``'s (same
seed -> same split bucket), and a mixed-``T`` directory is rejected loudly
rather than silently truncated.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from gltfworld.data.pack import split_id_for_seed
from gltfworld.data.pack_articulated import JOINT_TYPE_TO_ID, pack_articulated_dataset
from gltfworld.datagen.generate_articulated import generate_articulated_dataset
from gltfworld.scene.convert import load_episode
from safetensors.numpy import load_file


def test_pack_matches_source_episodes(tmp_path):
    gen = generate_articulated_dataset(tmp_path / "episodes", episodes=6, seed=10, steps=12, hz=30.0)
    result = pack_articulated_dataset(tmp_path / "episodes", tmp_path / "packed" / "art.safetensors")

    assert result.count == 6
    assert result.t == 12

    tensors = load_file(str(result.out_file))
    assert tensors["joint_pos"].shape == (6, 12)
    assert tensors["axis"].shape == (6, 3)
    assert tensors["joint_type_id"].shape == (6,)
    assert tensors["limit_min"].shape == (6,)
    assert tensors["limit_max"].shape == (6,)
    assert tensors["camera_pos"].shape == (6, 3)
    assert tensors["camera_rot"].shape == (6, 4)
    assert tensors["camera_yfov"].shape == (6,)

    for i, path in enumerate(gen.episode_paths):
        ep = load_episode(path)
        art = ep.scene.articulations[0]
        np.testing.assert_allclose(tensors["joint_pos"][i], ep.series.joint_pos[:, 0], atol=1e-6)
        assert int(tensors["joint_type_id"][i]) == JOINT_TYPE_TO_ID[art.joint_type]
        assert int(tensors["axis_idx"][i]) == art.axis
        expected_axis = np.zeros(3, dtype=np.float32)
        expected_axis[art.axis] = 1.0
        np.testing.assert_allclose(tensors["axis"][i], expected_axis)
        assert tensors["limit_min"][i] == pytest.approx(art.min)
        assert tensors["limit_max"][i] == pytest.approx(art.max)
        assert int(tensors["seeds"][i]) == ep.scene.seed
        assert int(tensors["split_id"][i]) == split_id_for_seed(ep.scene.seed)


def test_mixed_t_rejected(tmp_path):
    generate_articulated_dataset(tmp_path / "episodes", episodes=2, seed=1, steps=5, hz=30.0)
    # regenerate one episode with a different T, simulating a mixed --steps run.
    generate_articulated_dataset(tmp_path / "episodes2", episodes=1, seed=1, steps=9, hz=30.0)
    import shutil

    shutil.copy(tmp_path / "episodes2" / "ep_000000.glb", tmp_path / "episodes" / "ep_000001.glb")

    with pytest.raises(ValueError, match="T="):
        pack_articulated_dataset(tmp_path / "episodes", tmp_path / "packed" / "art.safetensors")


def test_joint_type_counts_match_kind_split(tmp_path):
    generate_articulated_dataset(tmp_path / "episodes", episodes=8, seed=500, steps=5, hz=30.0)
    result = pack_articulated_dataset(tmp_path / "episodes", tmp_path / "packed" / "art.safetensors")
    # generate_articulated_dataset alternates door(revolute)/drawer(prismatic) exactly 50/50.
    assert result.joint_type_counts == {"revolute": 4, "prismatic": 4}
