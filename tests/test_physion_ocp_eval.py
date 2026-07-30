"""``gltfworld.physion.ocp_eval`` -- unit tests (no data needed) plus
real-data smoke tests against a handful of real Collide trials + the real
``dynamics-v1`` checkpoint. Skips cleanly (like ``test_physion_convert.py``)
when either isn't present on this machine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gltfworld.physion import ocp_eval as oe

REPO_ROOT = Path(__file__).resolve().parent.parent
HDF5_DIR = REPO_ROOT / "data" / "external" / "physion" / "hdf5" / "extracted" / "Collide" / "hdf5s"
DYNAMICS_CKPT = REPO_ROOT / "runs" / "dynamics-v1" / "best.safetensors"


def _require_real_hdf5() -> list[Path]:
    if not HDF5_DIR.exists():
        pytest.skip(f"{HDF5_DIR} not present -- see docs/PHYSION.md")
    files = sorted(HDF5_DIR.glob("*.hdf5"))
    if not files:
        pytest.skip(f"{HDF5_DIR} present but empty")
    return files


def _require_dynamics_ckpt() -> Path:
    if not DYNAMICS_CKPT.exists():
        pytest.skip(f"{DYNAMICS_CKPT} not present -- train dynamics-v1 first (see docs/VERIFICATION.md V5)")
    return DYNAMICS_CKPT


# --- unit tests, no data needed --------------------------------------------------


def test_wilson_interval_bounds_and_midpoint():
    phat, lo, hi = oe.wilson_interval(75, 150)
    assert phat == pytest.approx(0.5)
    assert 0.0 <= lo < phat < hi <= 1.0

    # all successes -> upper bound is 1.0, lower bound strictly < 1.0
    phat, lo, hi = oe.wilson_interval(50, 50)
    assert phat == 1.0
    assert hi == 1.0
    assert lo < 1.0

    # zero trials -> nan, not a crash
    phat, lo, hi = oe.wilson_interval(0, 0)
    assert np.isnan(phat) and np.isnan(lo) and np.isnan(hi)


def test_wilson_interval_narrows_with_more_trials():
    _, lo_small, hi_small = oe.wilson_interval(9, 10)
    _, lo_large, hi_large = oe.wilson_interval(900, 1000)
    assert (hi_large - lo_large) < (hi_small - lo_small)


def test_calibration_split_deterministic_and_sized():
    trial_ids = [f"trial_{i:04d}" for i in range(150)]
    split1 = oe.calibration_split(trial_ids, n_calib=50)
    split2 = oe.calibration_split(trial_ids, n_calib=50)
    assert split1 == split2
    assert len(split1) == 50
    assert split1 <= set(trial_ids)


def test_calibration_split_changes_with_seed():
    trial_ids = [f"trial_{i:04d}" for i in range(150)]
    split_a = oe.calibration_split(trial_ids, n_calib=50, seed="seed-a")
    split_b = oe.calibration_split(trial_ids, n_calib=50, seed="seed-b")
    assert split_a != split_b


def test_collision_radius_matches_sphere_and_averages_box():
    assert oe.collision_radius(np.array([0.2, 0.2, 0.2])) == pytest.approx(0.2)
    assert oe.collision_radius(np.array([0.1, 0.3, 0.2])) == pytest.approx(0.2)


def test_calibrate_threshold_picks_perfectly_separating_threshold():
    curves = {
        "a": np.array([1.0, 0.5, 0.01]),  # dips below 0.05 -> contact
        "b": np.array([1.0, 0.9, 0.8]),  # never dips -> no contact
    }
    labels = {"a": True, "b": False}
    threshold, acc = oe.calibrate_threshold(curves, labels, grid=(0.01, 0.05, 0.5, 1.5))
    assert acc == 1.0
    preds = {tid: oe.predict_label(curves[tid], threshold) for tid in curves}
    assert preds == labels


def test_predict_label_any_frame_below_threshold():
    curve = np.array([1.0, 0.5, 0.02, 0.9])
    assert oe.predict_label(curve, 0.05) is True
    assert oe.predict_label(curve, 0.01) is False


# --- real-data smoke tests --------------------------------------------------------


@pytest.fixture(scope="module")
def five_trials():
    files = _require_real_hdf5()
    from gltfworld.physion.convert import load_trial

    return [load_trial(fp) for fp in files[:5]]


def test_oracle_curve_shape_and_finite(five_trials):
    for trial in five_trials:
        by_id = {obj.object_id: obj for obj in trial.objects}
        agent = by_id[trial.target_id]
        patient = by_id[trial.zone_id]
        curve = oe.oracle_min_distance_curve(agent, patient)
        assert curve.shape == (len(trial.times),)
        assert np.all(np.isfinite(curve))
        assert np.all(curve >= 0.0)  # a distance, never negative


def test_oracle_reproduces_label_on_five_trials(five_trials):
    """Not a strict claim (5 trials, no calibration) -- just checks the
    oracle's distance curve is directionally consistent with the label at a
    reasonable threshold (a hard failure here would mean the geometry/pose
    decode is simply wrong, not just imprecisely calibrated)."""
    threshold = 0.1
    correct = 0
    for trial in five_trials:
        by_id = {obj.object_id: obj for obj in trial.objects}
        agent = by_id[trial.target_id]
        patient = by_id[trial.zone_id]
        curve = oe.oracle_min_distance_curve(agent, patient)
        pred = oe.predict_label(curve, threshold)
        if pred == trial.label:
            correct += 1
    assert correct >= 3  # at least better than chance on this small sample


def test_sanity_check_against_core_labels(five_trials):
    labels = {trial.trial_id: trial.label for trial in five_trials}
    result = oe.sanity_check_against_core_labels(labels)
    if not result:
        pytest.skip("Core archive (data/external/physion/extracted/Physion) not present")
    assert result["n_mismatch"] == 0
    assert result["agreement_rate"] == 1.0


def test_full_pipeline_end_to_end_five_trials(tmp_path):
    _require_real_hdf5()
    ckpt = _require_dynamics_ckpt()

    glb_dir = tmp_path / "glb"
    out_dir = tmp_path / "out"
    exit_code = oe.main(
        [
            "--hdf5-dir",
            str(HDF5_DIR),
            "--glb-dir",
            str(glb_dir),
            "--dynamics-ckpt",
            str(ckpt),
            "--out",
            str(out_dir),
            "--max-trials",
            "6",
            "--n-calib",
            "2",
        ]
    )
    assert exit_code == 0

    import json

    metrics = json.loads((out_dir / "metrics.json").read_text())
    for track in ("oracle", "our_dynamics", "ballistic"):
        assert 0.0 <= metrics[track]["held_out" if track != "oracle" else "held_out"]["accuracy"] <= 1.0
    assert metrics["label_balance"]["true"] + metrics["label_balance"]["false"] == 6
    assert 0.0 <= metrics["threshold"]
