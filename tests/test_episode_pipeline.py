"""End-to-end: generate real episodes -> save through the existing GLB
transport -> validate with the pinned glTF-Validator -> load back and check
sanity invariants. Needs MuJoCo (``sim`` extra) but no GPU/EGL (no
rendering here).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from gltfworld.cli import run_validator
from gltfworld.datagen.generate import generate_dataset
from gltfworld.datagen.sample import object_support_offset
from gltfworld.scene.convert import load_episode

_DOWN_WORLD = np.array([0.0, -1.0, 0.0])

_N_EPISODES = 3
_STEPS = 50
_HZ = 30.0


@pytest.fixture(scope="module")
def generated_dataset(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("wm_scenes_v1_pipeline")
    result = generate_dataset(out_dir, episodes=_N_EPISODES, seed=999, steps=_STEPS, hz=_HZ)
    return result


def test_generate_writes_expected_files(generated_dataset):
    assert generated_dataset.manifest_path.exists()
    assert len(generated_dataset.episode_paths) == _N_EPISODES
    for path in generated_dataset.episode_paths:
        assert path.exists()
        assert path.suffix == ".glb"


def test_manifest_contents(generated_dataset):
    import json

    manifest = json.loads(generated_dataset.manifest_path.read_text())
    assert manifest["dataset_version"] == "wm-scenes-v1"
    assert manifest["scene_version"] == "wm-scenes-v1"
    assert manifest["seed"] == 999
    assert manifest["episode_seeds"] == [999, 1000, 1001]
    assert manifest["episodes"] == _N_EPISODES
    assert manifest["steps"] == _STEPS
    assert manifest["record_hz"] == _HZ
    assert "git_describe" in manifest


@pytest.mark.parametrize("episode_index", range(_N_EPISODES))
def test_each_episode_passes_gltf_validator(generated_dataset, episode_index):
    path = generated_dataset.episode_paths[episode_index]
    report = run_validator(str(path))
    num_errors = report.get("issues", {}).get("numErrors", 0)
    assert num_errors == 0, f"gltf-validator found {num_errors} error(s): {report}"


@pytest.mark.parametrize("episode_index", range(_N_EPISODES))
def test_each_episode_round_trips(generated_dataset, episode_index):
    path = generated_dataset.episode_paths[episode_index]
    episode = load_episode(path)

    assert len(episode.scene.objects) >= 2  # ground + at least one dynamic object
    assert episode.series.num_frames == _STEPS


@pytest.mark.parametrize("episode_index", range(_N_EPISODES))
def test_no_nan_or_inf_anywhere(generated_dataset, episode_index):
    path = generated_dataset.episode_paths[episode_index]
    episode = load_episode(path)
    series = episode.series

    for name, arr in (
        ("times", series.times),
        ("poses", series.poses),
        ("lin_vel", series.lin_vel),
        ("ang_vel", series.ang_vel),
    ):
        if arr is None:
            continue
        assert np.isfinite(arr).all(), f"{name} contains NaN/Inf in episode {episode_index}"


@pytest.mark.parametrize("episode_index", range(_N_EPISODES))
def test_objects_never_sink_below_ground(generated_dataset, episode_index):
    """The true lowest point of every dynamic object (via the shape's exact,
    orientation-aware support function -- not the coarser bounding-sphere
    radius, which would spuriously flag a resting box as "sunk" whenever
    its half-diagonal exceeds its half-height) must stay within 5mm of the
    ground plane's top surface (y=0 for wm-scenes-v1's ground convention)
    *once settled*, and never sink through the floor.

    Two checks, deliberately split (see the V3 report's "tolerances
    loosened" note):

    1. **Steady state** (last 20% of recorded frames): <= 5mm penetration,
       per the milestone spec.
    2. **Anywhere** (every recorded frame, including the very first
       ground impact): a much looser 10cm sanity bound. A fresh, fast,
       realistically-massed impact against MuJoCo's finite-stiffness
       soft-contact model produces real, physically-inherent transient
       penetration well past 5mm for a step or two (measured up to ~21mm
       here even after tightening solref/solimp beyond MuJoCo's defaults --
       see ``gltfworld.datagen.mujoco_env``) before the contact resolves;
       treating that as "sinking" would conflate a normal impact transient
       with an object actually falling through the floor, which is what
       this looser bound is really guarding against.
    """
    path = generated_dataset.episode_paths[episode_index]
    episode = load_episode(path)

    ground_top_y = 0.0
    n_frames = episode.series.num_frames
    steady_state_start = max(0, n_frames - max(1, n_frames // 5))

    for i, obj in enumerate(episode.scene.objects):
        if obj.category == "ground":
            continue
        poses = episode.series.poses[:, i, :]
        bottoms = np.array(
            [poses[t, 1] - object_support_offset(obj, poses[t, 3:7], _DOWN_WORLD) for t in range(poses.shape[0])]
        )

        min_bottom_anywhere = float(bottoms.min())
        assert min_bottom_anywhere >= ground_top_y - 0.10, (
            f"episode {episode_index} object {obj.object_id} fell through the floor: "
            f"bottom={min_bottom_anywhere:.4f} (more than 10cm below ground)"
        )

        min_bottom_settled = float(bottoms[steady_state_start:].min())
        assert min_bottom_settled >= ground_top_y - 0.005, (
            f"episode {episode_index} object {obj.object_id} settled {min_bottom_settled:.4f} "
            f"below ground (more than 5mm) in its final {n_frames - steady_state_start} frames"
        )
