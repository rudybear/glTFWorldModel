"""Unit tests for ``gltfworld.datagen.generate_articulated`` (needs the
``sim`` extra, MuJoCo -- same requirement as
``tests/test_articulated_physics.py``): the exact 50/50 door/drawer mix,
per-episode seed determinism, manifest contents, and that every written
episode is a loadable GLB with exactly one articulation. No rendering
(CPU-only, fast) -- the render path is exercised by the real dataset build
and by ``gltfworld.render.renderer``'s own existing tests.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("mujoco")

from gltfworld.datagen.generate_articulated import DATASET_VERSION, generate_articulated_dataset
from gltfworld.scene.convert import load_episode


def test_exact_50_50_kind_split(tmp_path):
    result = generate_articulated_dataset(tmp_path / "episodes", episodes=10, seed=1, steps=5, hz=30.0)
    manifest = json.loads(result.manifest_path.read_text())

    kinds = manifest["episode_kinds"]
    assert kinds.count("door") == 5
    assert kinds.count("drawer") == 5
    # deterministic alternation, not just a 50/50 count.
    assert kinds == ["door", "drawer"] * 5


def test_episode_seeds_are_base_plus_index(tmp_path):
    result = generate_articulated_dataset(tmp_path / "episodes", episodes=4, seed=777, steps=5, hz=30.0)
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["episode_seeds"] == [777, 778, 779, 780]
    assert manifest["dataset_version"] == DATASET_VERSION


def test_written_episodes_load_with_one_articulation(tmp_path):
    result = generate_articulated_dataset(tmp_path / "episodes", episodes=3, seed=42, steps=8, hz=30.0)
    assert len(result.episode_paths) == 3
    for path in result.episode_paths:
        ep = load_episode(path)
        assert len(ep.scene.articulations) == 1
        assert ep.series.joint_pos is not None
        assert ep.series.joint_pos.shape == (8, 1)


def test_deterministic_across_runs(tmp_path):
    r1 = generate_articulated_dataset(tmp_path / "run1", episodes=2, seed=99, steps=5, hz=30.0)
    r2 = generate_articulated_dataset(tmp_path / "run2", episodes=2, seed=99, steps=5, hz=30.0)

    ep1 = load_episode(r1.episode_paths[0])
    ep2 = load_episode(r2.episode_paths[0])
    assert ep1.scene.articulations[0].joint_type == ep2.scene.articulations[0].joint_type
    assert ep1.scene.articulations[0].axis == ep2.scene.articulations[0].axis
    import numpy as np

    np.testing.assert_array_equal(ep1.series.joint_pos, ep2.series.joint_pos)
