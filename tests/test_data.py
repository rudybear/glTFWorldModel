"""``gltfworld.data.pack``/``gltfworld.data.dataset``: packing determinism,
padding/mask correctness, split determinism, and the torch ``Dataset``
classes. Uses ``conftest.make_sample_episode`` directly (no MuJoCo/``sim``
extra needed -- packing itself has no MuJoCo dependency).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import make_sample_episode

from gltfworld.data.dataset import DynamicsDataset, PerceptionDataset
from gltfworld.data.pack import DEFAULT_N_MAX, PackResult, pack_dataset, split_id_for_seed
from gltfworld.scene.convert import save_episode


def _write_episodes(out_dir: Path, n_objects_list: list[int], seeds: list[int], T: int = 10) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (n_objects, seed) in enumerate(zip(n_objects_list, seeds)):
        ep = make_sample_episode(n_objects=n_objects, T=T)
        ep.scene.seed = seed
        save_episode(ep, out_dir / f"ep_{i:06d}.glb")
    manifest = {
        "dataset_version": "test",
        "scene_version": "wm-scenes-v1",
        "seed": seeds[0],
        "episode_seeds": seeds,
        "episodes": len(seeds),
        "steps": T,
        "record_hz": 30.0,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def test_pack_dataset_pads_and_masks(tmp_path: Path):
    episodes_dir = tmp_path / "episodes"
    _write_episodes(episodes_dir, n_objects_list=[1, 3, 5], seeds=[10, 11, 12])

    out_file = tmp_path / "packed" / "ds.safetensors"
    result = pack_dataset(episodes_dir, out_file)

    assert isinstance(result, PackResult)
    assert result.count == 3
    assert result.n_max == DEFAULT_N_MAX

    from safetensors.numpy import load_file

    tensors = load_file(str(out_file))
    assert tensors["states"].shape == (3, 10, DEFAULT_N_MAX, 22)
    assert tensors["mask"].shape == (3, DEFAULT_N_MAX)
    assert tensors["class_ids"].shape == (3, DEFAULT_N_MAX)
    assert tensors["globals"].shape == (3, 12)

    # episode 0 has 1 dynamic object -> 4 padding rows masked False
    assert tensors["mask"][0].sum() == 1
    assert tensors["mask"][1].sum() == 3
    assert tensors["mask"][2].sum() == 5
    assert not tensors["mask"][0][1:].any()
    assert (tensors["class_ids"][0][1:] == -1).all()
    assert np.all(tensors["states"][0, :, 1:, :] == 0.0)

    meta = json.loads(result.meta_path.read_text())
    assert meta["count"] == 3
    assert meta["n_max"] == DEFAULT_N_MAX
    assert meta["d"] == 22
    assert meta["source_manifest_hash_sha256"] is not None
    assert sum(meta["split_counts"].values()) == 3

    # make_sample_episode's ground is a 5x0.1x5 box at world origin (see
    # conftest.py), top surface at y=+0.1.
    assert meta["ground_top_y"]["mean"] == pytest.approx(0.1, abs=1e-4)
    assert meta["ground_footprint"]["half_extent_x"] == pytest.approx(5.0, abs=1e-4)
    assert meta["ground_footprint"]["half_extent_z"] == pytest.approx(5.0, abs=1e-4)
    assert meta["ground_footprint"]["consistent_across_episodes"] is True


def test_pack_dataset_rejects_mismatched_t(tmp_path: Path):
    episodes_dir = tmp_path / "episodes"
    episodes_dir.mkdir()
    ep1 = make_sample_episode(n_objects=2, T=10)
    ep2 = make_sample_episode(n_objects=2, T=20)
    save_episode(ep1, episodes_dir / "ep_000000.glb")
    save_episode(ep2, episodes_dir / "ep_000001.glb")

    with pytest.raises(ValueError, match="T="):
        pack_dataset(episodes_dir, tmp_path / "packed" / "ds.safetensors")


def test_split_is_deterministic_and_seed_keyed():
    seeds = list(range(1000))
    splits_a = [split_id_for_seed(s) for s in seeds]
    splits_b = [split_id_for_seed(s) for s in seeds]
    assert splits_a == splits_b  # deterministic

    counts = {0: splits_a.count(0), 1: splits_a.count(1), 2: splits_a.count(2)}
    # roughly 90/5/5 over 1000 samples (loose bounds -- this is a hash bucketing test, not exact)
    assert 850 <= counts[0] <= 950
    assert 20 <= counts[1] <= 90
    assert 20 <= counts[2] <= 90


def test_dynamics_dataset_transition_and_sequence_modes(tmp_path: Path):
    episodes_dir = tmp_path / "episodes"
    _write_episodes(episodes_dir, n_objects_list=[2, 3], seeds=[100, 101], T=15)
    out_file = tmp_path / "packed" / "ds.safetensors"
    pack_dataset(episodes_dir, out_file)

    seq_ds = DynamicsDataset(out_file, split=None, mode="sequence")
    assert len(seq_ds) == 2
    states, mask, globals_ = seq_ds[0]
    assert states.shape == (15, DEFAULT_N_MAX, 22)
    assert mask.shape == (DEFAULT_N_MAX,)
    assert globals_.shape == (12,)

    trans_ds = DynamicsDataset(out_file, split=None, mode="transition")
    assert len(trans_ds) == 2 * (15 - 1)
    state_t, state_t1, mask, globals_ = trans_ds[0]
    assert state_t.shape == (DEFAULT_N_MAX, 22)
    assert state_t1.shape == (DEFAULT_N_MAX, 22)


@pytest.mark.gpu
def test_perception_dataset_reads_rendered_frames(tmp_path: Path, episode_renderer):
    # Uses the session-scoped ``episode_renderer`` fixture (not
    # ``render_episode``, which builds+deletes its own renderer -- see
    # DESIGN.md/conftest.py's "one EpisodeRenderer per process" note; a
    # second, ad hoc renderer being deleted here would invalidate the
    # shared EGL display for every other gpu test in this session).
    episodes_dir = tmp_path / "episodes"
    episodes_dir.mkdir()
    ep = make_sample_episode(n_objects=2, T=5)
    save_episode(ep, episodes_dir / "ep_000000.glb")
    manifest = {"episodes": 1, "episode_seeds": [ep.scene.seed], "steps": 5, "record_hz": 30.0}
    (episodes_dir / "manifest.json").write_text(json.dumps(manifest))

    render_dir = episodes_dir / "ep_000000"
    render_dir.mkdir()
    episode_renderer.load(ep)
    rgb = np.empty((5, 256, 256, 3), dtype=np.uint8)
    seg = np.empty((5, 256, 256), dtype=np.uint16)
    depth = np.empty((5, 256, 256), dtype=np.float16)
    for t in range(5):
        episode_renderer.set_frame(t)
        frame = episode_renderer.render()
        rgb[t] = frame.rgb
        seg[t] = frame.seg
        depth[t] = frame.depth.astype(np.float16)
    np.save(render_dir / "rgb.npy", rgb)
    np.save(render_dir / "seg.npy", seg)
    np.save(render_dir / "depth.npy", depth)

    out_file = tmp_path / "packed" / "ds.safetensors"
    pack_dataset(episodes_dir, out_file)

    pd = PerceptionDataset(episodes_dir, out_file, split=None)
    assert len(pd) == 5
    rgb, state, mask, class_ids = pd[0]
    assert rgb.shape == (256, 256, 3)
    assert float(rgb.max()) <= 1.0
    assert float(rgb.min()) >= 0.0
    assert state.shape == (DEFAULT_N_MAX, 22)
