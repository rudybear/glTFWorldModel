"""Unit tests for ``gltfworld.eval.articulation_eval``: metrics correctness
on synthetic perfect/corrupted ``FrameRecord``s, the two baselines' scoping
(each only ever scored on the one metric it targets), and -- the
correctness-critical part -- that :func:`build_predicted_episode`'s forward-
kinematics reconstruction agrees with a *real* MuJoCo-simulated articulated
episode when fed that episode's own recorded ``joint_pos`` back in (needs
the ``sim`` extra; no rendering/GPU needed for this file).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from gltfworld.datagen.articulated import sample_articulated_scene, simulate_articulated
from gltfworld.eval.articulation_eval import (
    FrameRecord,
    build_predicted_episode,
    compute_metrics,
    mean_axis_baseline_records,
    midpoint_baseline_records,
)
from gltfworld.scene.convert import load_episode, save_episode
from gltfworld.scene.episode import Episode

# --- compute_metrics correctness -----------------------------------------------


def _record(
    pred_joint_pos_norm, pred_type_id, pred_axis, gt_joint_pos_norm, gt_type_id, gt_axis, limit_min=0.0, limit_max=1.9
):
    return FrameRecord(
        episode_idx=0,
        frame_idx=0,
        pred_joint_pos_norm=pred_joint_pos_norm,
        pred_type_id=pred_type_id,
        pred_axis=np.asarray(pred_axis, dtype=np.float32),
        gt_joint_pos_norm=gt_joint_pos_norm,
        gt_type_id=gt_type_id,
        gt_axis=np.asarray(gt_axis, dtype=np.float32),
        limit_min=limit_min,
        limit_max=limit_max,
    )


def test_compute_metrics_perfect_prediction_hinge():
    records = [_record(0.3, 0, [1.0, 0.0, 0.0], 0.3, 0, [1.0, 0.0, 0.0]) for _ in range(5)]
    m = compute_metrics(records, name="perfect")
    assert m["joint_pos_error_hinge_deg"]["median"] == pytest.approx(0.0, abs=1e-6)
    assert m["joint_pos_error_slider_cm"]["n"] == 0
    assert m["type_accuracy"] == pytest.approx(1.0)
    assert m["axis_error_deg"]["median"] == pytest.approx(0.0, abs=1e-6)


def test_compute_metrics_known_hinge_corruption():
    """A fixed 0.1 rad offset (in normalized units, since limit range is
    1.9) must come back as 0.1 * (180/pi) degrees -- validates the unit
    conversion, not just "some nonzero error"."""
    limit_max = 1.9
    offset_norm = 0.1 / limit_max
    records = [_record(0.3 + offset_norm, 0, [1.0, 0.0, 0.0], 0.3, 0, [1.0, 0.0, 0.0]) for _ in range(10)]
    m = compute_metrics(records, name="corrupted")
    expected_deg = 0.1 * (180.0 / np.pi)
    assert m["joint_pos_error_hinge_deg"]["median"] == pytest.approx(expected_deg, abs=1e-4)


def test_compute_metrics_slider_uses_cm_not_hinge_bucket():
    limit_min, limit_max = 0.0, 0.3
    offset_norm = 0.01 / limit_max  # 1cm offset
    records = [
        _record(0.4 + offset_norm, 1, [0.0, 0.0, 1.0], 0.4, 1, [0.0, 0.0, 1.0], limit_min, limit_max)
        for _ in range(10)
    ]
    m = compute_metrics(records, name="slider")
    assert m["joint_pos_error_hinge_deg"]["n"] == 0
    assert m["joint_pos_error_slider_cm"]["median"] == pytest.approx(1.0, abs=1e-4)  # 1cm


def test_compute_metrics_axis_sign_flip_is_180_degrees():
    records = [_record(0.3, 0, [-1.0, 0.0, 0.0], 0.3, 0, [1.0, 0.0, 0.0])]
    m = compute_metrics(records, name="flipped")
    assert m["axis_error_deg"]["median"] == pytest.approx(180.0, abs=1e-3)


def test_compute_metrics_type_misclassification():
    records = [_record(0.3, 1, [1.0, 0.0, 0.0], 0.3, 0, [1.0, 0.0, 0.0])]
    m = compute_metrics(records, name="wrong-type")
    assert m["type_accuracy"] == pytest.approx(0.0)


# --- baseline scoping -----------------------------------------------------------


def test_midpoint_baseline_only_scores_joint_pos():
    records = [_record(0.9, 0, [1.0, 0.0, 0.0], 0.3, 0, [0.0, 1.0, 0.0])]  # deliberately wrong type/axis
    baseline_records = midpoint_baseline_records(records)
    m = compute_metrics(baseline_records, name="midpoint", include_type=False, include_axis=False)
    # midpoint of [0, 1.9] normalized is 0.5; gt is 0.3 -> raw error = 0.2 * 1.9
    expected_deg = 0.2 * 1.9 * (180.0 / np.pi)
    assert m["joint_pos_error_hinge_deg"]["median"] == pytest.approx(expected_deg, abs=1e-3)
    assert m["type_accuracy"] is None
    assert m["axis_error_deg"] is None


def test_mean_axis_baseline_only_scores_axis():
    records = [_record(0.9, 1, [1.0, 0.0, 0.0], 0.3, 0, [0.0, 1.0, 0.0])]
    mean_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    baseline_records = mean_axis_baseline_records(records, mean_axis)
    m = compute_metrics(baseline_records, name="mean-axis", include_joint_pos=False, include_type=False)
    assert m["axis_error_deg"]["median"] == pytest.approx(0.0, abs=1e-4)
    assert m["joint_pos_error_hinge_deg"] is None
    assert m["type_accuracy"] is None


# --- FK reconstruction cross-checked against a real MuJoCo simulation ----------


@pytest.mark.parametrize("kind,axis", [("door", 1), ("drawer", 0)])
def test_build_predicted_episode_matches_simulated_pose_in_memory(kind, axis):
    """Feed build_predicted_episode the *actual* recorded joint_pos at
    several real frames of a MuJoCo-simulated episode -- the reconstructed
    part/handle pose must match the actually-simulated pose at that frame,
    same tolerance as tests/test_articulated_physics.py's own articulation
    consistency check (this is the same FK formula, run in the predict
    direction instead of the verify direction)."""
    sampled = sample_articulated_scene(seed=7, kind=kind, axis=axis)
    series = simulate_articulated(sampled, T=60, record_hz=30.0)
    ep = Episode(scene=sampled.scene, series=series)

    art = ep.scene.articulations[0]
    obj_ids = [o.object_id for o in ep.scene.objects]
    part_index = obj_ids.index(art.part_object_id)
    handle_index = obj_ids.index(art.handle_object_id)

    for t in (0, 10, 30, 59):
        actual_joint_pos = float(ep.series.joint_pos[t, 0])
        pred_episode = build_predicted_episode(ep, actual_joint_pos)

        actual_part_pos = ep.series.poses[t, part_index, 0:3]
        actual_part_rot = ep.series.poses[t, part_index, 3:7]
        pred_part_pos = pred_episode.series.poses[0, part_index, 0:3]
        pred_part_rot = pred_episode.series.poses[0, part_index, 3:7]

        assert np.linalg.norm(pred_part_pos - actual_part_pos) < 0.03
        rot_err = min(
            np.linalg.norm(pred_part_rot - actual_part_rot), np.linalg.norm(pred_part_rot + actual_part_rot)
        )
        assert rot_err < 0.03

        actual_handle_pos = ep.series.poses[t, handle_index, 0:3]
        pred_handle_pos = pred_episode.series.poses[0, handle_index, 0:3]
        assert np.linalg.norm(pred_handle_pos - actual_handle_pos) < 0.03


def test_build_predicted_episode_roundtrips_through_glb(tmp_path):
    sampled = sample_articulated_scene(seed=3, kind="door", axis=1)
    series = simulate_articulated(sampled, T=30, record_hz=30.0)
    ep = Episode(scene=sampled.scene, series=series)

    pred_episode = build_predicted_episode(ep, pred_joint_pos_raw=0.5)
    glb_path = tmp_path / "pred.glb"
    save_episode(pred_episode, glb_path)
    reloaded = load_episode(glb_path)

    np.testing.assert_allclose(reloaded.series.poses, pred_episode.series.poses, atol=1e-5)
    np.testing.assert_allclose(reloaded.series.joint_pos, pred_episode.series.joint_pos, atol=1e-5)
    assert len(reloaded.scene.articulations) == 1
