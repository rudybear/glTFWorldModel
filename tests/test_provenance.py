"""Provenance: training data (tensor-contract output) must equal what's
actually stored in the glTF files on disk.

For 5 freshly simulated episodes: keep the in-memory ``Episode`` (built
straight from MuJoCo's ``StateSeries``, no glTF involved yet), separately
save it to a real GLB and load it back, then run *both* the in-memory and
the GLB-round-tripped ``Episode`` through ``episode_to_tensors``. If the
resulting tensors don't match to <= 1e-6 absolute (fp32), the glTF files on
disk are not a faithful record of what a training pipeline would actually
consume in memory -- exactly the gap this test exists to catch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from gltfworld.datagen.mujoco_env import simulate
from gltfworld.datagen.sample import sample_scene
from gltfworld.scene.contract import episode_to_tensors
from gltfworld.scene.convert import load_episode, save_episode
from gltfworld.scene.episode import Episode

PROVENANCE_SEEDS = [90210, 90211, 90212, 90213, 90214]


def _simulate_episode(seed: int) -> Episode:
    sampled = sample_scene(seed)
    series = simulate(
        sampled.scene,
        sampled.initial_poses,
        T=40,
        record_hz=30.0,
        initial_lin_vel=sampled.initial_lin_vel,
        initial_ang_vel=sampled.initial_ang_vel,
    )
    return Episode(scene=sampled.scene, series=series)


@pytest.mark.parametrize("seed", PROVENANCE_SEEDS)
def test_tensor_contract_matches_across_glb_round_trip(seed: int, tmp_path: Path):
    in_memory_episode = _simulate_episode(seed)

    glb_path = tmp_path / f"provenance_{seed}.glb"
    save_episode(in_memory_episode, glb_path)
    reloaded_episode = load_episode(glb_path)

    tensors_in_memory = episode_to_tensors(in_memory_episode)
    tensors_reloaded = episode_to_tensors(reloaded_episode)

    for key in ("states", "globals"):
        a = tensors_in_memory[key]
        b = tensors_reloaded[key]
        assert a.shape == b.shape, f"{key}: shape mismatch {a.shape} vs {b.shape}"
        max_abs_diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        assert max_abs_diff <= 1e-6, f"{key}: max abs diff {max_abs_diff} exceeds 1e-6"

    np.testing.assert_array_equal(tensors_in_memory["mask"], tensors_reloaded["mask"])
    np.testing.assert_array_equal(tensors_in_memory["class_ids"], tensors_reloaded["class_ids"])

    assert set(tensors_in_memory["static"].keys()) == set(tensors_reloaded["static"].keys())
    for key in tensors_in_memory["static"]:
        static_in_memory = tensors_in_memory["static"][key]
        static_reloaded = tensors_reloaded["static"][key]
        assert static_in_memory["shape"] == static_reloaded["shape"]
        assert static_in_memory["category"] == static_reloaded["category"]
        for field in ("size", "pos", "quat"):
            diff = float(
                np.max(
                    np.abs(
                        static_in_memory[field].astype(np.float64)
                        - static_reloaded[field].astype(np.float64)
                    )
                )
            )
            assert diff <= 1e-6, f"static[{key}][{field}]: max abs diff {diff} exceeds 1e-6"
