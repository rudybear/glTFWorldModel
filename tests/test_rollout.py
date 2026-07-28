"""``gltfworld.eval.rollout``: rollout shape/finiteness, divergence-curve/
horizon-metrics sanity, and the pred-GLB round trip (load_episode(pred glb)
reproduces the predicted tensors <= 1e-6). Uses ``conftest.make_sample_episode``
(no MuJoCo/GPU needed) packed through the real ``pack_dataset`` pipeline, so
this exercises the exact same code path real data goes through.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import make_sample_episode

from gltfworld.data.dataset import DynamicsDataset
from gltfworld.data.pack import pack_dataset
from gltfworld.eval.rollout import divergence_curve, horizon_metrics, rollout, tensors_to_episode
from gltfworld.models.baselines import BallisticBaseline
from gltfworld.models.dynamics import InteractionTransformer
from gltfworld.scene.contract import episode_to_tensors
from gltfworld.scene.convert import load_episode, save_episode

torch.manual_seed(0)


def _write_episodes(out_dir: Path, n_objects_list: list[int], seeds: list[int], T: int = 12) -> None:
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


def _class_ids_for(pack_file: Path, episode_idx: int) -> np.ndarray:
    """Real ``class_ids`` for a packed-file row (split=None order == packed
    order), needed by ``tensors_to_episode``/``tensors_to_state`` (which
    cross-checks ``class_ids`` against each reconstructed object's actual
    category -- an all-zeros stand-in only works if every object happens to
    be class 0)."""
    from safetensors.numpy import load_file

    tensors = load_file(str(pack_file))
    return tensors["class_ids"][episode_idx]


@pytest.fixture
def tiny_dataset(tmp_path: Path):
    episodes_dir = tmp_path / "episodes"
    _write_episodes(episodes_dir, n_objects_list=[3, 5, 1], seeds=[100, 101, 102], T=12)
    pack_file = tmp_path / "packed" / "ds.safetensors"
    pack_dataset(episodes_dir, pack_file)
    return episodes_dir, pack_file


def test_rollout_shape_and_finite(tiny_dataset):
    _episodes_dir, pack_file = tiny_dataset
    ds = DynamicsDataset(pack_file, split=None, mode="sequence")
    model = InteractionTransformer()

    states0 = ds.states[:, 0]  # (E, N, D)
    out = rollout(model, states0, ds.mask, ds.globals, T=ds.t)

    assert out.shape == (ds.num_episodes, ds.t, ds.n_max, ds.d)
    assert torch.isfinite(out).all()
    # t=0 is untouched (exactly the given initial state).
    assert torch.equal(out[:, 0], states0)


def test_rollout_single_episode_matches_batched(tiny_dataset):
    _episodes_dir, pack_file = tiny_dataset
    ds = DynamicsDataset(pack_file, split=None, mode="sequence")
    model = InteractionTransformer()
    model.eval()

    out_batched = rollout(model, ds.states[:, 0], ds.mask, ds.globals, T=ds.t)
    out_single = rollout(model, ds.states[0, 0], ds.mask[0], ds.globals[0], T=ds.t)

    assert out_single.shape == (ds.t, ds.n_max, ds.d)
    np.testing.assert_allclose(out_single.detach().numpy(), out_batched[0].detach().numpy(), atol=1e-6)


def test_rollout_ballistic_matches_manual_step(tiny_dataset):
    _episodes_dir, pack_file = tiny_dataset
    ds = DynamicsDataset(pack_file, split=None, mode="sequence")
    model = BallisticBaseline()

    out = rollout(model, ds.states[:, 0], ds.mask, ds.globals, T=3)
    # step 0 -> 1 must equal one manual ballistic call.
    with torch.no_grad():
        expected_step1 = model(ds.states[:, 0], ds.mask, ds.globals)
    np.testing.assert_allclose(out[:, 1].numpy(), expected_step1.numpy(), atol=1e-6)


def test_divergence_curve_and_horizon_metrics_zero_for_identical(tiny_dataset):
    _episodes_dir, pack_file = tiny_dataset
    ds = DynamicsDataset(pack_file, split=None, mode="sequence")

    curve = divergence_curve(ds.states, ds.states, ds.mask)
    assert curve.shape == (ds.t - 1,)
    np.testing.assert_allclose(curve, np.zeros_like(curve), atol=1e-6)

    metrics = horizon_metrics(ds.states, ds.states, ds.mask, horizon=5)
    assert metrics["position_error_m"]["median"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["rotation_error_rad"]["median"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["velocity_error_mps"]["median"] == pytest.approx(0.0, abs=1e-6)


def test_divergence_curve_nonzero_for_ballistic_vs_truth(tiny_dataset):
    _episodes_dir, pack_file = tiny_dataset
    ds = DynamicsDataset(pack_file, split=None, mode="sequence")
    model = BallisticBaseline()
    pred = rollout(model, ds.states[:, 0], ds.mask, ds.globals, T=ds.t)

    curve = divergence_curve(pred, ds.states, ds.mask)
    assert curve.shape == (ds.t - 1,)
    assert np.isfinite(curve).all()
    # ballistic (no collision response) diverges from truth (which does
    # include ground contact / non-ballistic motion via the free-fall
    # kinematics fixture) somewhere over a 12-frame rollout.
    assert curve.max() > 0.0


def test_pred_glb_roundtrip(tiny_dataset):
    """load_episode(pred glb) must reproduce the predicted tensors used to
    build it, to <= 1e-6 -- the exact same guarantee test_provenance.py
    checks generically, exercised here specifically for a model rollout's
    re-exported Episode."""
    episodes_dir, pack_file = tiny_dataset
    ds = DynamicsDataset(pack_file, split=None, mode="sequence")
    model = InteractionTransformer()

    pred = rollout(model, ds.states[:, 0], ds.mask, ds.globals, T=ds.t)

    # episode 0 has 3 real dynamic objects (see _write_episodes above).
    ep_idx = 0
    orig_idx = int(ds.episode_indices[ep_idx])
    template = load_episode(episodes_dir / f"ep_{orig_idx:06d}.glb")

    mask_np = ds.mask[ep_idx].numpy()
    n_real = int(mask_np.sum())
    class_ids_np = _class_ids_for(pack_file, ep_idx)
    globals_np = ds.globals[ep_idx].numpy()
    pred_np = pred[ep_idx].detach().numpy()

    pred_episode = tensors_to_episode(pred_np, mask_np, class_ids_np, globals_np, template)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        glb_path = Path(tmp) / "pred.glb"
        save_episode(pred_episode, glb_path)
        loaded = load_episode(glb_path)

    loaded_tensors = episode_to_tensors(loaded)
    # compare only the real (unpadded) dynamic-object rows/columns.
    expected = pred_np[:, :n_real, :]
    np.testing.assert_allclose(loaded_tensors["states"], expected, atol=1e-6)
    np.testing.assert_allclose(loaded_tensors["globals"], globals_np, atol=1e-6)


def test_tensors_to_episode_static_ground_constant(tiny_dataset):
    episodes_dir, pack_file = tiny_dataset
    ds = DynamicsDataset(pack_file, split=None, mode="sequence")
    ep_idx = 0
    orig_idx = int(ds.episode_indices[ep_idx])
    template = load_episode(episodes_dir / f"ep_{orig_idx:06d}.glb")

    mask_np = ds.mask[ep_idx].numpy()
    class_ids_np = _class_ids_for(pack_file, ep_idx)
    globals_np = ds.globals[ep_idx].numpy()
    states_np = ds.states[ep_idx].numpy()

    rebuilt = tensors_to_episode(states_np, mask_np, class_ids_np, globals_np, template)

    assert len(rebuilt.scene.objects) == len(template.scene.objects)
    ground_index = [i for i, obj in enumerate(template.scene.objects) if obj.is_static][0]
    ground_poses = rebuilt.series.poses[:, ground_index, :]
    np.testing.assert_allclose(ground_poses, np.broadcast_to(ground_poses[0], ground_poses.shape), atol=1e-6)
