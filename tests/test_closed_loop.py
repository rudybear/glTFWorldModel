"""CPU-only tests for ``gltfworld.eval.closed_loop``: pure arm-assembly logic
(noise injection, finite-diff velocity, Arm B/Arm C state assembly), dataset
resolution/split filtering, the per-episode glTF-at-every-hop pipeline (Arms
A/B + ballistic, ``--no-perception`` so no renderer/GPU is needed), metric
aggregation shapes/finiteness, and the attribution plot. The gpu-marked
end-to-end smoke (real checkpoints, real Arm C via the renderer) lives in
``tests/test_closed_loop_gpu.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import make_sample_episode

from gltfworld.data.pack import pack_dataset
from gltfworld.eval.closed_loop import (
    ArmAccumulator,
    NoiseParams,
    _DEFAULT_MASS,
    aggregate_results,
    build_arm_b_initial_state,
    build_arm_c_assembly,
    finite_diff_velocity,
    match_detections_across_frames,
    noise_params_from_args,
    noise_params_from_metrics,
    perturb_pose,
    plot_attribution,
    process_episode,
    resolve_episodes_dir,
    select_episodes,
)
from gltfworld.eval.rollout import _load_model_from_ckpt as _load_dynamics_model
from gltfworld.models.baselines import BallisticBaseline
from gltfworld.models.dynamics import InteractionTransformer
from gltfworld.models.rotations import axis_angle_to_quat, quat_hemisphere, quat_multiply, quat_normalize
from gltfworld.scene.convert import save_episode

torch.manual_seed(0)


# --- noise injection: statistics sanity ----------------------------------------


def test_perturb_pose_noise_matches_requested_sigma():
    n = 20000
    pos = torch.zeros(n, 3)
    quat = torch.zeros(n, 4)
    quat[:, 3] = 1.0
    sigma_pos = 0.05
    sigma_rot = torch.full((n,), math.radians(10.0))
    gen = torch.Generator().manual_seed(42)

    noisy_pos, noisy_quat = perturb_pose(pos, quat, sigma_pos, sigma_rot, gen)

    # per-axis position noise std should match sigma_pos closely at this n.
    empirical_sigma_pos = noisy_pos.std().item()
    assert empirical_sigma_pos == pytest.approx(sigma_pos, rel=0.05)

    # rotation: the injected rotvec has per-axis std sigma_rot; recover it by
    # inverting the quaternion delta back to a rotation vector.
    from gltfworld.models.rotations import quat_conjugate, quat_to_axis_angle

    dq = quat_multiply(noisy_quat, quat_conjugate(quat))
    rotvec = quat_to_axis_angle(dq)
    empirical_sigma_rot = rotvec.std().item()
    assert empirical_sigma_rot == pytest.approx(sigma_rot[0].item(), rel=0.05)


def test_perturb_pose_zero_sigma_is_exact():
    pos = torch.randn(10, 3)
    quat = quat_hemisphere(quat_normalize(torch.randn(10, 4)))
    gen = torch.Generator().manual_seed(0)
    noisy_pos, noisy_quat = perturb_pose(pos, quat, 0.0, torch.zeros(10), gen)
    np.testing.assert_allclose(noisy_pos.numpy(), pos.numpy(), atol=1e-6)
    np.testing.assert_allclose(noisy_quat.numpy(), quat.numpy(), atol=1e-6)


def test_perturb_pose_deterministic_given_seed():
    pos = torch.randn(10, 3)
    quat = quat_hemisphere(quat_normalize(torch.randn(10, 4)))
    sigma_rot = torch.full((10,), 0.1)
    out1 = perturb_pose(pos, quat, 0.05, sigma_rot, torch.Generator().manual_seed(7))
    out2 = perturb_pose(pos, quat, 0.05, sigma_rot, torch.Generator().manual_seed(7))
    np.testing.assert_array_equal(out1[0].numpy(), out2[0].numpy())
    np.testing.assert_array_equal(out1[1].numpy(), out2[1].numpy())


# --- finite-diff velocity ------------------------------------------------------


def test_finite_diff_velocity_recovers_constant_motion():
    torch.manual_seed(1)
    n = 50
    pos0 = torch.randn(n, 3)
    vel0 = torch.randn(n, 3) * 0.5
    ang_vel0 = torch.randn(n, 3) * 0.3
    quat0 = quat_hemisphere(quat_normalize(torch.randn(n, 4)))
    dt = 0.033

    pos1 = pos0 + vel0 * dt
    dq = axis_angle_to_quat(ang_vel0 * dt)
    quat1 = quat_hemisphere(quat_normalize(quat_multiply(dq, quat0)))

    lin_vel, ang_vel = finite_diff_velocity(pos0, quat0, pos1, quat1, dt)
    np.testing.assert_allclose(lin_vel.numpy(), vel0.numpy(), atol=1e-4)
    np.testing.assert_allclose(ang_vel.numpy(), ang_vel0.numpy(), atol=1e-4)


# --- Arm B assembly: zero-noise exactness --------------------------------------


def _make_gt_states(n=4, dt=0.033, seed=0):
    torch.manual_seed(seed)
    pos0 = torch.randn(n, 3)
    vel0 = torch.randn(n, 3) * 0.5
    ang_vel0 = torch.randn(n, 3) * 0.3
    quat0 = quat_hemisphere(quat_normalize(torch.randn(n, 4)))
    pos1 = pos0 + vel0 * dt
    dq = axis_angle_to_quat(ang_vel0 * dt)
    quat1 = quat_hemisphere(quat_normalize(quat_multiply(dq, quat0)))

    shape_onehot = torch.zeros(n, 3)
    shape_onehot[:, 1] = 1.0  # all box
    size = torch.full((n, 3), 0.1)
    static = torch.cat(
        [shape_onehot, size, torch.zeros(n, 1), torch.full((n, 1), 0.6), torch.full((n, 1), 0.1)], dim=-1
    )
    states0 = torch.cat([pos0, quat0, vel0, ang_vel0, static], dim=-1)
    states1 = torch.cat([pos1, quat1, vel0, ang_vel0, static], dim=-1)
    return torch.stack([states0, states1], dim=0), dt


def test_build_arm_b_initial_state_zero_noise_matches_gt_exactly():
    states_gt, dt = _make_gt_states()
    noise0 = NoiseParams(sigma_pos_m=0.0, sigma_rot_rad_by_shape={"sphere": 0.0, "box": 0.0, "cylinder": 0.0}, source="test")
    gen = torch.Generator().manual_seed(1)
    init_b = build_arm_b_initial_state(states_gt, dt, noise0, gen)
    np.testing.assert_allclose(init_b.numpy(), states_gt[0].numpy(), atol=1e-4)


def test_build_arm_b_initial_state_deterministic_given_seed():
    states_gt, dt = _make_gt_states()
    noise = NoiseParams(sigma_pos_m=0.05, sigma_rot_rad_by_shape={"sphere": 0.0, "box": 0.1, "cylinder": 0.1}, source="test")
    a = build_arm_b_initial_state(states_gt, dt, noise, torch.Generator().manual_seed(9))
    b = build_arm_b_initial_state(states_gt, dt, noise, torch.Generator().manual_seed(9))
    np.testing.assert_array_equal(a.numpy(), b.numpy())


def test_build_arm_b_initial_state_nonzero_noise_perturbs():
    states_gt, dt = _make_gt_states()
    noise = NoiseParams(sigma_pos_m=0.05, sigma_rot_rad_by_shape={"sphere": 0.0, "box": 0.2, "cylinder": 0.2}, source="test")
    init_b = build_arm_b_initial_state(states_gt, dt, noise, torch.Generator().manual_seed(2))
    assert not np.allclose(init_b.numpy(), states_gt[0].numpy(), atol=1e-4)
    # physics/shape/size fields must stay exact GT (Arm B isolates pose noise only).
    np.testing.assert_allclose(init_b[:, 13:22].numpy(), states_gt[0, :, 13:22].numpy(), atol=1e-6)


# --- Arm C assembly: perfect-perception exactness + degenerate cases -----------


def _make_perfect_pred(pos, quat, size, shape_idx, class_idx, n_max=5):
    n = pos.shape[0]
    existence_logit = torch.full((n_max,), -10.0)
    existence_logit[:n] = 10.0
    position = torch.zeros(n_max, 3)
    position[:n] = pos
    q = torch.zeros(n_max, 4)
    q[:, 3] = 1.0
    q[:n] = quat
    sz = torch.zeros(n_max, 3)
    sz[:n] = size
    shape_logits = torch.zeros(n_max, 3)
    shape_logits[torch.arange(n), shape_idx] = 10.0
    class_logits = torch.zeros(n_max, 3)
    class_logits[torch.arange(n), class_idx] = 10.0
    return {
        "existence_logit": existence_logit,
        "position": position,
        "quat": q,
        "size": sz,
        "shape_logits": shape_logits,
        "class_logits": class_logits,
    }


def test_build_arm_c_assembly_perfect_perception_matches_gt_with_default_physics():
    """The task's own exactness bar: Arm C's assembled state must equal
    Arm A's (GT) state exactly when perception is perfect and noise sigma=0
    -- modulo the one honest, structural exception this milestone documents
    (mass/friction/restitution default to fixed constants, since
    ``PerceptionDETR`` never predicts them at all): construct GT physics
    fields to already equal those same defaults, so the comparison is exact."""
    n = 3
    torch.manual_seed(3)
    pos0 = torch.randn(n, 3)
    vel0 = torch.randn(n, 3) * 0.5
    ang_vel0 = torch.randn(n, 3) * 0.3
    quat0 = quat_hemisphere(quat_normalize(torch.randn(n, 4)))
    dt = 0.033
    pos1 = pos0 + vel0 * dt
    dq = axis_angle_to_quat(ang_vel0 * dt)
    quat1 = quat_hemisphere(quat_normalize(quat_multiply(dq, quat0)))

    shape_idx = torch.tensor([1, 1, 1])  # all box (avoids sphere/cylinder size-canonicalization)
    size = torch.full((n, 3), 0.1)
    class_idx = torch.tensor([1, 1, 1])

    pred0 = _make_perfect_pred(pos0, quat0, size, shape_idx, class_idx)
    pred1 = _make_perfect_pred(pos1, quat1, size, shape_idx, class_idx)

    assembly = build_arm_c_assembly(pred0, pred1, dt, gt_pos0=pos0, gt_class0=class_idx, gt_size0=size)
    assert assembly["n_correspondence"] == n
    query_idx, gt_idx = assembly["armc_to_gt"]
    np.testing.assert_array_equal(sorted(query_idx.tolist()), list(range(n)))
    np.testing.assert_array_equal(sorted(gt_idx.tolist()), list(range(n)))

    from gltfworld.eval.closed_loop import _DEFAULT_FRICTION, _DEFAULT_RESTITUTION

    static = torch.cat(
        [
            torch.nn.functional.one_hot(shape_idx, 3).float(),
            size,
            torch.full((n, 1), math.log(_DEFAULT_MASS)),
            torch.full((n, 1), _DEFAULT_FRICTION),
            torch.full((n, 1), _DEFAULT_RESTITUTION),
        ],
        dim=-1,
    )
    states0_default_phys = torch.cat([pos0, quat0, vel0, ang_vel0, static], dim=-1)
    np.testing.assert_allclose(assembly["initial_state"].numpy(), states0_default_phys.numpy(), atol=1e-4)


def test_build_arm_c_assembly_zero_detections():
    n_max = 5
    empty_pred = {
        "existence_logit": torch.full((n_max,), -10.0),
        "position": torch.zeros(n_max, 3),
        "quat": torch.tile(torch.tensor([0.0, 0.0, 0.0, 1.0]), (n_max, 1)),
        "size": torch.zeros(n_max, 3),
        "shape_logits": torch.zeros(n_max, 3),
        "class_logits": torch.zeros(n_max, 3),
    }
    assembly = build_arm_c_assembly(
        empty_pred, empty_pred, dt=0.033, gt_pos0=torch.zeros(2, 3), gt_class0=torch.zeros(2, dtype=torch.long),
        gt_size0=torch.zeros(2, 3),
    )
    assert assembly["n_correspondence"] == 0
    assert assembly["initial_state"].shape == (0, 22)
    assert assembly["armc_to_gt"][0].size == 0


def test_match_detections_across_frames_basic_correspondence():
    pred0 = {
        "position": torch.tensor([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]),
        "class_logits": torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        "size": torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]),
    }
    pred1 = {
        "position": torch.tensor([[5.05, 5.0, 5.0], [0.05, 0.0, 0.0]]),
        "class_logits": torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        "size": torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]),
    }
    idx0 = torch.tensor([0, 1])
    idx1 = torch.tensor([0, 1])
    m0, m1 = match_detections_across_frames(pred0, idx0, pred1, idx1)
    # frame0's object 0 (near origin) should correspond to frame1's object 1
    # (also near origin); frame0's object 1 to frame1's object 0.
    mapping = dict(zip(m0.tolist(), m1.tolist()))
    assert mapping[0] == 1
    assert mapping[1] == 0


def test_match_detections_across_frames_empty_inputs():
    pred = {
        "position": torch.zeros(0, 3),
        "class_logits": torch.zeros(0, 3),
        "size": torch.zeros(0, 3),
    }
    m0, m1 = match_detections_across_frames(pred, torch.zeros(0, dtype=torch.long), pred, torch.zeros(0, dtype=torch.long))
    assert m0.size == 0 and m1.size == 0


# --- noise calibration from a perception_eval-style metrics.json ---------------


def test_noise_params_from_metrics_chi3_inversion(tmp_path: Path):
    from scipy.stats import chi

    c3 = float(chi(df=3).ppf(0.5))
    pos_median = 0.2095
    box_deg = 8.0
    fake_metrics = {
        "metrics": {
            "PerceptionDETR": {
                "matched_position_error_m": {"median": pos_median},
                "matched_rotation_error_deg_by_shape": {
                    "sphere": {"median": None, "n": 0},
                    "box": {"median": box_deg, "n": 100},
                    "cylinder": {"median": None, "n": 0},
                },
            }
        }
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(fake_metrics))

    noise = noise_params_from_metrics(path)
    assert noise.sigma_pos_m == pytest.approx(pos_median / c3)
    assert noise.sigma_rot_rad_by_shape["box"] == pytest.approx(math.radians(box_deg) / c3)
    assert noise.sigma_rot_rad_by_shape["sphere"] == 0.0
    assert noise.sigma_rot_rad_by_shape["cylinder"] == 0.0  # median is None -> 0


def test_noise_params_from_args():
    noise = noise_params_from_args(0.1, 5.0)
    assert noise.sigma_pos_m == 0.1
    assert noise.sigma_rot_rad_by_shape["box"] == pytest.approx(math.radians(5.0))
    assert noise.sigma_rot_rad_by_shape["cylinder"] == pytest.approx(math.radians(5.0))
    assert noise.sigma_rot_rad_by_shape["sphere"] == 0.0


# --- dataset resolution + split selection --------------------------------------


def _write_glb_episodes(out_dir: Path, seeds: list[int], T: int = 12) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, seed in enumerate(seeds):
        ep = make_sample_episode(n_objects=3, T=T)
        ep.scene.seed = seed
        save_episode(ep, out_dir / f"ep_{i:06d}.glb")


def test_resolve_episodes_dir_raw_glb_directory(tmp_path: Path):
    episodes_dir = tmp_path / "raw"
    _write_glb_episodes(episodes_dir, seeds=[1, 2, 3])
    resolved = resolve_episodes_dir(episodes_dir)
    assert resolved == episodes_dir


def test_resolve_episodes_dir_dataset_root(tmp_path: Path):
    root = tmp_path / "dataset-root"
    episodes_dir = root / "episodes"
    _write_glb_episodes(episodes_dir, seeds=[10, 11])
    (root / "packed").mkdir(parents=True)  # presence alone is enough to identify the root layout
    resolved = resolve_episodes_dir(root)
    assert resolved == episodes_dir


def test_resolve_episodes_dir_packed_file(tmp_path: Path):
    episodes_dir = tmp_path / "episodes"
    _write_glb_episodes(episodes_dir, seeds=[20, 21, 22])
    pack_file = tmp_path / "packed" / "ds.safetensors"
    pack_dataset(episodes_dir, pack_file)

    resolved_from_file = resolve_episodes_dir(pack_file)
    assert resolved_from_file == episodes_dir
    resolved_from_dir = resolve_episodes_dir(pack_file.parent)
    assert resolved_from_dir == episodes_dir


def test_select_episodes_split_all_and_limit(tmp_path: Path):
    episodes_dir = tmp_path / "episodes"
    _write_glb_episodes(episodes_dir, seeds=[1, 2, 3, 4, 5])
    selected = select_episodes(episodes_dir, split="all", n_episodes=3)
    assert len(selected) == 3
    assert selected == sorted(episodes_dir.glob("ep_*.glb"))[:3]


def test_select_episodes_split_filtering_matches_split_id_for_seed(tmp_path: Path):
    from gltfworld.data.pack import split_id_for_seed

    episodes_dir = tmp_path / "episodes"
    seeds = list(range(50))
    _write_glb_episodes(episodes_dir, seeds=seeds, T=3)

    selected = select_episodes(episodes_dir, split="test", n_episodes=1000)
    from gltfworld.scene.convert import load_episode

    for p in selected:
        assert split_id_for_seed(load_episode(p).scene.seed) == 2  # "test" index
    expected_count = sum(1 for s in seeds if split_id_for_seed(s) == 2)
    assert len(selected) == expected_count


# --- process_episode: glTF-at-every-hop (Arms A/B only, no perception/GPU) -----


def _write_fresh_dynamics_ckpt(out_dir: Path) -> Path:
    from safetensors.torch import save_file

    from gltfworld.train.train_dynamics import Config, make_model

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    cfg.save(out_dir / "config.json")
    torch.manual_seed(0)
    model = make_model(cfg)
    ckpt_path = out_dir / "fresh.safetensors"
    save_file(model.state_dict(), ckpt_path)
    return ckpt_path


def test_process_episode_no_perception_roundtrip_and_finite(tmp_path: Path):
    episodes_dir = tmp_path / "episodes"
    _write_glb_episodes(episodes_dir, seeds=[100], T=10)
    ep_path = sorted(episodes_dir.glob("ep_*.glb"))[0]

    ckpt_path = _write_fresh_dynamics_ckpt(tmp_path / "ckpt")
    device = torch.device("cpu")
    model, _ = _load_dynamics_model(ckpt_path, device)

    noise = noise_params_from_args(0.05, 5.0)
    out_dir = tmp_path / "out"
    result = process_episode(
        ep_path, model, per_model=None, device=device, noise=noise, seed=0, out_dir=out_dir, renderer=None,
    )

    assert result.states_A.shape == result.states_gt.shape
    assert result.states_B.shape == result.states_gt.shape
    assert result.n_correspondence == 0
    assert result.episode_C is None
    assert np.isfinite(result.states_A).all()
    assert np.isfinite(result.states_B).all()
    assert np.isfinite(result.states_ballistic).all()

    assert (out_dir / "gt" / f"{ep_path.stem}.glb").exists()
    assert (out_dir / "armA" / f"{ep_path.stem}.glb").exists()
    assert (out_dir / "armB" / f"{ep_path.stem}.glb").exists()
    assert not (out_dir / "armC" / f"{ep_path.stem}.glb").exists()


def test_process_episode_deterministic_given_seed(tmp_path: Path):
    episodes_dir = tmp_path / "episodes"
    _write_glb_episodes(episodes_dir, seeds=[200], T=8)
    ep_path = sorted(episodes_dir.glob("ep_*.glb"))[0]
    ckpt_path = _write_fresh_dynamics_ckpt(tmp_path / "ckpt")
    device = torch.device("cpu")
    model, _ = _load_dynamics_model(ckpt_path, device)
    noise = noise_params_from_args(0.05, 5.0)

    r1 = process_episode(ep_path, model, None, device, noise, seed=42, out_dir=tmp_path / "out1", renderer=None)
    r2 = process_episode(ep_path, model, None, device, noise, seed=42, out_dir=tmp_path / "out2", renderer=None)
    np.testing.assert_array_equal(r1.states_B, r2.states_B)


# --- metric aggregation + attribution plot -------------------------------------


def test_aggregate_results_shapes_finiteness_and_ordering(tmp_path: Path):
    episodes_dir = tmp_path / "episodes"
    _write_glb_episodes(episodes_dir, seeds=[1, 2, 3], T=15)
    ckpt_path = _write_fresh_dynamics_ckpt(tmp_path / "ckpt")
    device = torch.device("cpu")
    model, _ = _load_dynamics_model(ckpt_path, device)
    noise = noise_params_from_args(0.05, 5.0)

    results = []
    for i, ep_path in enumerate(sorted(episodes_dir.glob("ep_*.glb"))):
        results.append(
            process_episode(ep_path, model, None, device, noise, seed=0, out_dir=tmp_path / "out", renderer=None)
        )

    horizons = [1, 5, 10]
    agg = aggregate_results(results, horizons)
    assert agg["n_episodes"] == 3
    for h in horizons:
        for arm in ("A_oracle", "B_oracle_noise", "ballistic"):
            entry = agg["arms"][arm][str(h)]["position_error_m"]
            assert entry["n"] > 0
            assert math.isfinite(entry["median"])
    # Arm C had no perception model, so it must be all-empty (n=0), never crash.
    for h in horizons:
        assert agg["arms"]["C_visual"][str(h)]["position_error_m"]["n"] == 0
    assert agg["arm_c_detection"]["tp"] == 0
    assert agg["arm_c_detection"]["n_zero_correspondence_episodes"] == 3

    for arm, curve in agg["full_curves"].items():
        assert isinstance(curve, list)

    ordering = agg["ordering_check"]
    assert set(ordering.keys()) == {str(h) for h in horizons}


def test_arm_accumulator_full_curve_and_horizon_stats():
    acc = ArmAccumulator()
    pred_pos = np.zeros((5, 2, 3), dtype=np.float32)
    gt_pos = np.zeros((5, 2, 3), dtype=np.float32)
    pred_pos[1:] += 0.1  # constant nonzero error at every horizon
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (5, 2, 1))
    acc.add_episode(pred_pos, gt_pos, quat, quat, horizons=[1, 2, 4])

    curve = acc.full_curve_median()
    assert curve.shape == (4,)
    assert np.isfinite(curve).all()
    np.testing.assert_allclose(curve, 0.1 * np.sqrt(3), atol=1e-5)

    stats = acc.horizon_stats([1, 2, 4])
    for h in (1, 2, 4):
        assert stats[str(h)]["position_error_m"]["n"] == 2
        assert stats[str(h)]["rotation_error_rad"]["median"] == pytest.approx(0.0, abs=1e-6)


def test_plot_attribution_writes_file(tmp_path: Path):
    curves = {
        "A_oracle": [0.01, 0.02, 0.03],
        "B_oracle_noise": [0.02, 0.04, 0.06],
        "C_visual": [0.05, 0.08, 0.12],
        "ballistic": [0.1, 1.0, 10.0],
    }
    out_path = tmp_path / "attribution.png"
    plot_attribution(curves, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_attribution_handles_empty_arm_curve(tmp_path: Path):
    curves = {"A_oracle": [0.01, 0.02], "C_visual": []}
    out_path = tmp_path / "attribution.png"
    plot_attribution(curves, out_path)
    assert out_path.exists()
