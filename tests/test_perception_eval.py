"""Unit tests for ``gltfworld.eval.perception_eval``: metrics correctness on
a synthetic perfect-prediction case (all errors ~0, F1=1) and a
known-corruption case (injected 5cm position noise -> median error ~5cm,
validating the metric itself, not just the model); the predicted-Episode
builder + its GLB round trip (CPU-only, no renderer/GPU needed -- the
*render* part of the re-render check is exercised separately, gpu-marked)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_sample_episode

from gltfworld.eval.perception_eval import FrameRecord, build_predicted_episode, compute_metrics
from gltfworld.models.matching import SHAPE_ORDER
from gltfworld.scene.contract import episode_to_tensors
from gltfworld.scene.convert import load_episode, save_episode

SPHERE_IDX = SHAPE_ORDER.index("sphere")
BOX_IDX = SHAPE_ORDER.index("box")


def _perfect_record(n_real: int = 3, n_max: int = 5, seed: int = 0) -> FrameRecord:
    rng = np.random.default_rng(seed)
    gt_position = np.zeros((n_max, 3), dtype=np.float32)
    gt_position[:n_real] = rng.uniform(-1, 1, size=(n_real, 3))
    gt_quat = np.zeros((n_max, 4), dtype=np.float32)
    gt_quat[:, 3] = 1.0  # identity everywhere
    gt_size = np.full((n_max, 3), 0.1, dtype=np.float32)
    gt_shape_idx = np.zeros(n_max, dtype=np.int64)
    gt_shape_idx[:n_real] = BOX_IDX
    gt_class_idx = np.zeros(n_max, dtype=np.int64)
    gt_class_idx[:n_real] = 1  # "crate"

    query_idx = np.arange(n_real)
    gt_idx = np.arange(n_real)

    existence_prob = np.zeros(n_max, dtype=np.float64)
    existence_prob[:n_real] = 1.0

    return FrameRecord(
        episode_idx=0,
        frame_idx=0,
        n_real=n_real,
        query_idx=query_idx,
        gt_idx=gt_idx,
        existence_prob=existence_prob,
        pred_position=gt_position.copy(),
        pred_quat=gt_quat.copy(),
        pred_size=gt_size.copy(),
        pred_shape_idx=gt_shape_idx.copy(),
        pred_class_idx=gt_class_idx.copy(),
        gt_position=gt_position,
        gt_quat=gt_quat,
        gt_size=gt_size,
        gt_shape_idx=gt_shape_idx,
        gt_class_idx=gt_class_idx,
    )


def test_compute_metrics_perfect_prediction():
    records = [_perfect_record(n_real=3, seed=s) for s in range(10)]
    metrics = compute_metrics(records, name="perfect")

    assert metrics["existence"]["precision"] == pytest.approx(1.0)
    assert metrics["existence"]["recall"] == pytest.approx(1.0)
    assert metrics["existence"]["f1"] == pytest.approx(1.0)
    assert metrics["matched_position_error_m"]["median"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["matched_position_error_m"]["mean"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["matched_size_error_m"]["mean"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["shape_accuracy"] == pytest.approx(1.0)
    assert metrics["class_accuracy"] == pytest.approx(1.0)
    assert metrics["count_exact_match_rate"] == pytest.approx(1.0)
    # every GT object here is a box at identity orientation predicted at
    # identity orientation -> box rotation error must be ~0 too.
    assert metrics["matched_rotation_error_deg_by_shape"]["box"]["mean"] == pytest.approx(0.0, abs=1e-3)


def test_compute_metrics_known_position_corruption():
    """Inject a known 5cm position offset on every matched prediction --
    the reported median position error must come back at ~5cm, which
    validates the metric computation itself (not the model)."""
    records = []
    for s in range(20):
        r = _perfect_record(n_real=3, seed=s)
        # perturb every real prediction by exactly 5cm along a fixed axis.
        r.pred_position = r.pred_position.copy()
        r.pred_position[: r.n_real, 0] += 0.05
        records.append(r)

    metrics = compute_metrics(records, name="corrupted")
    assert metrics["matched_position_error_m"]["median"] == pytest.approx(0.05, abs=1e-6)
    assert metrics["matched_position_error_m"]["mean"] == pytest.approx(0.05, abs=1e-6)
    # existence/shape/class are untouched by this corruption.
    assert metrics["existence"]["f1"] == pytest.approx(1.0)
    assert metrics["shape_accuracy"] == pytest.approx(1.0)


def test_compute_metrics_false_positive_and_false_negative():
    """1 real GT object; the model predicts it (matched, above threshold)
    plus one spurious extra query above threshold (false positive), and a
    second frame where the real object's matched query falls below
    threshold (false negative)."""
    n_max = 5
    tp_record = _perfect_record(n_real=1, seed=100)
    tp_record.existence_prob = np.zeros(n_max)
    tp_record.existence_prob[0] = 0.9  # matched query, above threshold -> TP
    tp_record.existence_prob[3] = 0.7  # unmatched query, above threshold -> FP

    fn_record = _perfect_record(n_real=1, seed=101)
    fn_record.existence_prob = np.zeros(n_max)
    fn_record.existence_prob[0] = 0.1  # matched query, below threshold -> FN

    metrics = compute_metrics([tp_record, fn_record], name="mixed")
    assert metrics["existence"]["tp"] == 1
    assert metrics["existence"]["fp"] == 1
    assert metrics["existence"]["fn"] == 1
    assert metrics["existence"]["precision"] == pytest.approx(0.5)
    assert metrics["existence"]["recall"] == pytest.approx(0.5)


def test_compute_metrics_sphere_excluded_from_rotation_error():
    n_max = 5
    record = _perfect_record(n_real=2, seed=200)
    record.gt_shape_idx = record.gt_shape_idx.copy()
    record.gt_shape_idx[:2] = SPHERE_IDX
    record.pred_shape_idx = record.gt_shape_idx.copy()
    # scramble the predicted quat -- must not affect anything since sphere
    # rotation is never scored.
    record.pred_quat = record.pred_quat.copy()
    record.pred_quat[0] = [0.5, 0.5, 0.5, np.sqrt(1 - 0.75)]

    metrics = compute_metrics([record], name="sphere-only")
    sphere_stat = metrics["matched_rotation_error_deg_by_shape"]["sphere"]
    assert sphere_stat["n"] == 0
    assert sphere_stat["median"] is None


# --- predicted-Episode builder + GLB round trip (CPU) -------------------------


def test_build_predicted_episode_roundtrip(tmp_path):
    template = make_sample_episode(n_objects=3, T=5)
    n_max = 5
    n_real = 3  # matches make_sample_episode's 3 dynamic objects

    gt_shape_idx = np.array([0, 1, 2, 0, 0], dtype=np.int64)  # sphere, box, cylinder (matches fixture)
    record = FrameRecord(
        episode_idx=0,
        frame_idx=0,
        n_real=n_real,
        query_idx=np.arange(n_real),
        gt_idx=np.arange(n_real),
        existence_prob=np.array([0.9, 0.9, 0.9, 0.1, 0.1]),
        pred_position=np.array(
            [[0.1, 1.0, 0.0], [0.9, 0.5, 0.2], [-0.5, 0.3, -0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        pred_quat=np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n_max, 1)),
        pred_size=np.full((n_max, 3), 0.15, dtype=np.float32),
        pred_shape_idx=gt_shape_idx.copy(),
        pred_class_idx=np.array([0, 1, 2, 0, 0], dtype=np.int64),
        gt_position=np.zeros((n_max, 3), dtype=np.float32),
        gt_quat=np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n_max, 1)),
        gt_size=np.zeros((n_max, 3), dtype=np.float32),
        gt_shape_idx=gt_shape_idx,
        gt_class_idx=np.array([0, 1, 2, 0, 0], dtype=np.int64),
    )

    pred_episode = build_predicted_episode(template, record)
    # 3 existence-thresholded dynamic objects + template's 1 static ground object.
    assert len(pred_episode.scene.objects) == 4
    assert sum(1 for obj in pred_episode.scene.objects if not obj.is_static) == 3
    assert sum(1 for obj in pred_episode.scene.objects if obj.is_static) == 1

    glb_path = tmp_path / "pred.glb"
    save_episode(pred_episode, glb_path)
    reloaded = load_episode(glb_path)

    original_tensors = episode_to_tensors(pred_episode)
    reloaded_tensors = episode_to_tensors(reloaded)
    np.testing.assert_allclose(reloaded_tensors["states"], original_tensors["states"], atol=1e-6)
    np.testing.assert_allclose(reloaded_tensors["globals"], original_tensors["globals"], atol=1e-6)


def test_build_predicted_episode_false_positive_gets_default_color(tmp_path):
    """A thresholded query with no Hungarian match (a false positive) still
    builds a valid, savable object -- using the documented fallback color/
    material rather than crashing for lack of a GT object to copy from."""
    template = make_sample_episode(n_objects=1, T=2)
    n_max = 5
    record = FrameRecord(
        episode_idx=0,
        frame_idx=0,
        n_real=0,
        query_idx=np.array([], dtype=np.int64),
        gt_idx=np.array([], dtype=np.int64),
        existence_prob=np.array([0.9, 0.0, 0.0, 0.0, 0.0]),
        pred_position=np.zeros((n_max, 3), dtype=np.float32),
        pred_quat=np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n_max, 1)),
        pred_size=np.full((n_max, 3), 0.1, dtype=np.float32),
        pred_shape_idx=np.zeros(n_max, dtype=np.int64),
        pred_class_idx=np.zeros(n_max, dtype=np.int64),
        gt_position=np.zeros((n_max, 3), dtype=np.float32),
        gt_quat=np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n_max, 1)),
        gt_size=np.zeros((n_max, 3), dtype=np.float32),
        gt_shape_idx=np.zeros(n_max, dtype=np.int64),
        gt_class_idx=np.zeros(n_max, dtype=np.int64),
    )

    pred_episode = build_predicted_episode(template, record)
    dynamic = [obj for obj in pred_episode.scene.objects if not obj.is_static]
    assert len(dynamic) == 1
    np.testing.assert_allclose(dynamic[0].color, [0.5, 0.5, 0.5, 1.0])

    glb_path = tmp_path / "pred_fp.glb"
    save_episode(pred_episode, glb_path)
    load_episode(glb_path)  # must not raise
