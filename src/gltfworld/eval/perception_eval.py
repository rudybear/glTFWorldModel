"""Eval CLI for ``gltfworld.models.perception.PerceptionDETR``: existence
precision/recall/F1 (+ PR curve), per-object-count breakdown, matched
position/rotation/size error, shape/class accuracy, count-exact-match rate,
a trivial "mean-state" baseline for context, and (optionally, GPU) a
re-render PSNR/SSIM check against the actually-rendered GT frames.

    uv run python -m gltfworld.eval.perception_eval \\
        --ckpt runs/perception-v1/best.safetensors \\
        --data data/perception-v1 --split test \\
        --out runs/perception-v1/eval

Metrics (Hungarian-matched to GT, see ``gltfworld.models.matching
.hungarian_match``) -- ``metrics.json``/``metrics.md``:

- **Existence precision/recall/F1** at the 0.5 threshold, plus a full PR
  curve (101 thresholds). A query is a true positive if it is both (a) the
  Hungarian-assigned match for some real GT object and (b) thresholded
  "existent"; a false positive is an *unmatched* query thresholded
  "existent" (there is no real GT for it, so it can never be a true
  positive); a false negative is a real GT object whose matched query fell
  *below* the threshold.
- **Per-N breakdown** (``N`` = 1..5 real objects in the frame): existence F1
  and matched position error, computed within each ``N`` group.
- **Matched position/rotation/size error, shape/class accuracy**: computed
  over every Hungarian-matched (query, GT) pair -- independent of the
  existence threshold (the same convention ``gltfworld.eval.rollout`` uses:
  a pose-quality metric asks "how good is the best-matched query's pose",
  not "did the detector also fire" -- that's what the existence metrics
  above are for). Rotation error is symmetry-aware
  (``gltfworld.models.matching.symmetry_rotation_angle``, degrees) and split
  by GT shape; sphere rows are excluded (a sphere has no meaningful
  orientation to score at all, per the training loss's own treatment).
- **Count-exact-match rate**: fraction of frames where the thresholded
  predicted-object count equals the real GT count.
- **Mean-state baseline**: a zero-learning dummy predictor -- always predicts
  the (train-split) mean real-object count, all at the (train-split) mean
  position/size and the modal shape/class -- run through the exact same
  metrics pipeline, so the trained model's numbers can be read against a
  trivial prior rather than in a vacuum.

Re-render check (``--render-samples K``, needs the ``render`` extra + GPU)
---------------------------------------------------------------------------

For ``K`` sampled test frames: build a predicted, single-frame ``Episode``
from every existence-thresholded query (predicted pose/size/shape; colors
and physics-material fields are copied from the GT ``ObjectSpec`` of that
query's Hungarian-*matched* real object when it has one -- an intentionally
honest GT-assist, since :class:`~gltfworld.models.perception.PerceptionDETR`
never predicts color/material at all, and this eval's purpose here is only
to sanity-check *geometry* rendering fidelity, not colors; an
existence-thresholded query with no real GT match, i.e. a false positive, is
rendered in a fixed neutral gray -- see :data:`_FALSE_POSITIVE_COLOR` --
since there is no GT object to copy from), render it, and compare against
the actual stored GT rgb frame via PSNR/SSIM
(``gltfworld.eval.metrics``). Predicted frames are also saved as real,
independently loadable ``pred_frames/ep_XXXXXX_fYYYY.glb`` (``T=1``,
``gltfworld.scene.convert.save_episode``) -- each one is round-trip verified
inline (``load_episode`` reproduces the tensor-contract state to ``<= 1e-6``)
and run through the real, pinned glTF-Validator (``gltfworld validate``,
via ``gltfworld.cli.run_validator``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gltfworld.data.dataset import PerceptionDataset
from gltfworld.eval.metrics import psnr, ssim
from gltfworld.models.matching import SPHERE_IDX, MatchCostWeights, hungarian_match, symmetry_rotation_angle
from gltfworld.models.perception import PerceptionDETR, count_params
from gltfworld.scene.contract import CATEGORY_TO_CLASS_ID, SHAPE_ORDER
from gltfworld.scene.convert import load_episode, save_episode
from gltfworld.scene.episode import Episode
from gltfworld.scene.scene import ObjectSpec, SceneState

EXISTENCE_THRESHOLD = 0.5
DEFAULT_RENDER_SAMPLES = 50
RAD2DEG = 180.0 / np.pi

CLASS_ID_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_CLASS_ID.items()}

# Color/material stand-in for a rendered false-positive query (an
# existence-thresholded prediction with no Hungarian-matched real GT object
# to copy an honest color/material from) -- a fixed neutral gray, distinct
# from wm-scenes-v1's 8 sampled hues (gltfworld.datagen.sample) so a
# false-positive is visually identifiable in the rendered PNG, not just in
# the metrics.
_FALSE_POSITIVE_COLOR = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)
_DEFAULT_ROUGHNESS = 0.5
_DEFAULT_METALLIC = 0.0
_DEFAULT_MASS = 1.0
_DEFAULT_FRICTION = 0.6
_DEFAULT_RESTITUTION = 0.1


# --- model / data loading -----------------------------------------------------


def _resolve_paths(data_dir: Path) -> tuple[Path, Path]:
    episodes_dir = data_dir / "episodes"
    packed_dir = data_dir / "packed"
    candidates = sorted(packed_dir.glob("*.safetensors"))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one .safetensors file in {packed_dir}, found {len(candidates)}")
    return episodes_dir, candidates[0]


def _load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> PerceptionDETR:
    from safetensors.torch import load_file

    from gltfworld.train.train_perception import Config, make_model

    config_path = ckpt_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found -- perception eval needs the training run's config.json "
            f"(next to the checkpoint) to know the model architecture"
        )
    cfg = Config.load(config_path)
    model = make_model(cfg).to(device)
    model.load_state_dict(load_file(ckpt_path, device=str(device)))
    model.eval()
    return model


# --- inference: collect every prediction + GT for a split ---------------------


@dataclasses.dataclass
class FrameRecord:
    episode_idx: int
    frame_idx: int
    n_real: int
    query_idx: np.ndarray  # matched query indices (len n_real)
    gt_idx: np.ndarray  # matched gt slot indices (len n_real), into N_max axis
    existence_prob: np.ndarray  # (Q,) all queries
    pred_position: np.ndarray  # (Q, 3)
    pred_quat: np.ndarray  # (Q, 4)
    pred_size: np.ndarray  # (Q, 3)
    pred_shape_idx: np.ndarray  # (Q,)
    pred_class_idx: np.ndarray  # (Q,)
    gt_position: np.ndarray  # (N_max, 3)
    gt_quat: np.ndarray  # (N_max, 4)
    gt_size: np.ndarray  # (N_max, 3)
    gt_shape_idx: np.ndarray  # (N_max,)
    gt_class_idx: np.ndarray  # (N_max,)


@torch.no_grad()
def run_inference(model: torch.nn.Module, ds: PerceptionDataset, device: torch.device, batch_size: int = 64) -> list[FrameRecord]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    records: list[FrameRecord] = []
    row = 0

    for rgb, states, mask, class_ids in loader:
        rgb_d, states_d, mask_d, class_ids_d = rgb.to(device), states.to(device), mask.to(device), class_ids.to(device)
        pred = model(rgb_d)
        existence_prob = torch.sigmoid(pred["existence_logit"])
        gt_position = states_d[..., 0:3]
        gt_quat = states_d[..., 3:7]
        gt_shape_idx = states_d[..., 13:16].argmax(dim=-1)
        gt_size = states_d[..., 16:19]

        matches = hungarian_match(
            pred["position"], pred["class_logits"], pred["size"], gt_position, class_ids_d, gt_size, mask_d,
            MatchCostWeights(),
        )
        pred_shape_idx = pred["shape_logits"].argmax(dim=-1)
        pred_class_idx = pred["class_logits"].argmax(dim=-1)

        b = rgb.shape[0]
        for i in range(b):
            episode_idx, frame_idx = ds._index[row]
            row += 1
            query_idx, gt_idx = matches[i]
            records.append(
                FrameRecord(
                    episode_idx=episode_idx,
                    frame_idx=frame_idx,
                    n_real=int(mask[i].sum()),
                    query_idx=query_idx,
                    gt_idx=gt_idx,
                    existence_prob=existence_prob[i].cpu().numpy(),
                    pred_position=pred["position"][i].cpu().numpy(),
                    pred_quat=pred["quat"][i].cpu().numpy(),
                    pred_size=pred["size"][i].cpu().numpy(),
                    pred_shape_idx=pred_shape_idx[i].cpu().numpy(),
                    pred_class_idx=pred_class_idx[i].cpu().numpy(),
                    gt_position=gt_position[i].cpu().numpy(),
                    gt_quat=gt_quat[i].cpu().numpy(),
                    gt_size=gt_size[i].cpu().numpy(),
                    gt_shape_idx=gt_shape_idx[i].cpu().numpy(),
                    gt_class_idx=class_ids[i].numpy(),
                )
            )
    return records


# --- metrics -------------------------------------------------------------------


def _pr_at_threshold(records: list[FrameRecord], threshold: float) -> tuple[int, int, int]:
    """``(tp, fp, fn)`` at existence threshold ``threshold`` -- see module
    docstring for the true/false positive/negative definitions."""
    tp = fp = fn = 0
    for r in records:
        matched_set = set(r.query_idx.tolist())
        for q in matched_set:
            if r.existence_prob[q] >= threshold:
                tp += 1
            else:
                fn += 1
        for q in range(r.existence_prob.shape[0]):
            if q not in matched_set and r.existence_prob[q] >= threshold:
                fp += 1
    return tp, fp, fn


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _matched_pairs(records: list[FrameRecord]):
    """Yield ``(record, query_idx, gt_idx)`` for every Hungarian-matched
    (query, GT) pair across every frame -- the unit every pose/size/shape/
    class metric below is computed over."""
    for r in records:
        for q, g in zip(r.query_idx, r.gt_idx):
            yield r, int(q), int(g)


def compute_metrics(records: list[FrameRecord], name: str) -> dict:
    tp5, fp5, fn5 = _pr_at_threshold(records, EXISTENCE_THRESHOLD)
    precision, recall, f1 = _precision_recall_f1(tp5, fp5, fn5)

    pr_curve = []
    for t in np.linspace(0.0, 1.0, 101):
        tp, fp, fn = _pr_at_threshold(records, float(t))
        p, rc, _ = _precision_recall_f1(tp, fp, fn)
        pr_curve.append({"threshold": float(t), "precision": p, "recall": rc})

    pos_errs, size_errs = [], []
    rot_errs_by_shape: dict[str, list[float]] = {s: [] for s in SHAPE_ORDER}
    shape_correct, class_correct = [], []

    for r, q, g in _matched_pairs(records):
        pos_err = float(np.linalg.norm(r.pred_position[q] - r.gt_position[g]))
        size_err = float(np.linalg.norm(r.pred_size[q] - r.gt_size[g]))
        pos_errs.append(pos_err)
        size_errs.append(size_err)
        shape_correct.append(int(r.pred_shape_idx[q]) == int(r.gt_shape_idx[g]))
        class_correct.append(int(r.pred_class_idx[q]) == int(r.gt_class_idx[g]))

        gt_shape = int(r.gt_shape_idx[g])
        if gt_shape != SPHERE_IDX:
            angle = symmetry_rotation_angle(
                torch.from_numpy(r.pred_quat[q : q + 1]), torch.from_numpy(r.gt_quat[g : g + 1]),
                torch.tensor([gt_shape]),
            ).item()
            rot_errs_by_shape[SHAPE_ORDER[gt_shape]].append(angle * RAD2DEG)

    # per-N breakdown
    per_n = {}
    for n in range(1, 6):
        group = [r for r in records if r.n_real == n]
        if not group:
            per_n[str(n)] = None
            continue
        tp_n, fp_n, fn_n = _pr_at_threshold(group, EXISTENCE_THRESHOLD)
        p_n, rc_n, f1_n = _precision_recall_f1(tp_n, fp_n, fn_n)
        group_pos_errs = [
            float(np.linalg.norm(r.pred_position[q] - r.gt_position[g])) for r, q, g in _matched_pairs(group)
        ]
        per_n[str(n)] = {
            "n_frames": len(group),
            "existence_precision": p_n,
            "existence_recall": rc_n,
            "existence_f1": f1_n,
            "matched_position_error_m_median": float(np.median(group_pos_errs)) if group_pos_errs else None,
        }

    # count-exact-match rate
    exact_matches = 0
    for r in records:
        predicted_count = int((r.existence_prob >= EXISTENCE_THRESHOLD).sum())
        if predicted_count == r.n_real:
            exact_matches += 1
    count_exact_match_rate = exact_matches / len(records) if records else 0.0

    def _stat(values: list[float]) -> dict:
        if not values:
            return {"median": None, "mean": None, "n": 0}
        arr = np.asarray(values, dtype=np.float64)
        return {"median": float(np.median(arr)), "mean": float(np.mean(arr)), "n": int(arr.size)}

    return {
        "name": name,
        "n_frames": len(records),
        "existence": {
            "threshold": EXISTENCE_THRESHOLD,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp5,
            "fp": fp5,
            "fn": fn5,
            "pr_curve": pr_curve,
        },
        "per_n": per_n,
        "matched_position_error_m": _stat(pos_errs),
        "matched_size_error_m": _stat(size_errs),
        "matched_rotation_error_deg_by_shape": {s: _stat(v) for s, v in rot_errs_by_shape.items()},
        "shape_accuracy": float(np.mean(shape_correct)) if shape_correct else None,
        "class_accuracy": float(np.mean(class_correct)) if class_correct else None,
        "count_exact_match_rate": count_exact_match_rate,
    }


# --- mean-state baseline -------------------------------------------------------


def build_mean_state_baseline(train_ds: PerceptionDataset) -> dict:
    """Dataset statistics (train split) a zero-learning dummy predictor
    always predicts: mean real-object count (rounded), mean position/size
    of every real object seen, and the modal shape/class.

    ``PerceptionDataset.states``/``.mask``/``.class_ids`` are the *whole*
    packed dataset's per-episode tensors (``states (E_total, T, N_max, D)``,
    ``mask``/``class_ids (E_total, N_max)``) -- unlike ``DynamicsDataset``,
    ``PerceptionDataset`` does not itself subset these by split; only its
    ``_index`` list (``(episode_idx, frame_idx)`` pairs) is filtered to the
    requested split's frames (see its docstring). So this function must
    gather rows through ``_index``, not index ``.states``/``.mask``
    directly (those still span every episode in the packed file, train/
    val/test alike, and ``.states`` additionally carries a full ``T`` axis
    this function must also select into).
    """
    states = train_ds.states.numpy()
    mask = train_ds.mask.numpy()
    class_ids = train_ds.class_ids.numpy()

    episode_idxs = np.array([e for e, _f in train_ds._index], dtype=np.int64)
    frame_idxs = np.array([f for _e, f in train_ds._index], dtype=np.int64)

    frame_states = states[episode_idxs, frame_idxs]  # (n_frames, N_max, D)
    frame_mask = mask[episode_idxs]  # (n_frames, N_max)
    frame_class_ids = class_ids[episode_idxs]  # (n_frames, N_max)

    real_positions = frame_states[frame_mask][:, 0:3]
    real_sizes = frame_states[frame_mask][:, 16:19]
    real_shape_idx = frame_states[frame_mask][:, 13:16].argmax(axis=-1)
    real_class_idx = frame_class_ids[frame_mask]

    n_real_per_frame = frame_mask.sum(axis=-1)
    mean_count = int(round(float(n_real_per_frame.mean())))
    mean_count = max(0, min(mean_count, frame_mask.shape[-1]))

    return {
        "mean_count": mean_count,
        "mean_position": real_positions.mean(axis=0),
        "mean_size": real_sizes.mean(axis=0),
        "mode_shape_idx": int(np.bincount(real_shape_idx).argmax()),
        "mode_class_idx": int(np.bincount(real_class_idx).argmax()),
    }


def mean_state_baseline_predict(baseline: dict, batch_size: int, n_queries: int) -> dict[str, torch.Tensor]:
    """Produce a :class:`~gltfworld.models.perception.PerceptionDETR`-shaped
    prediction dict from :func:`build_mean_state_baseline`'s fixed stats --
    identical for every input frame (no learning, no image is even looked
    at)."""
    existence_logit = torch.full((batch_size, n_queries), -10.0)
    existence_logit[:, : baseline["mean_count"]] = 10.0

    position = torch.from_numpy(baseline["mean_position"]).float()[None, None, :].expand(batch_size, n_queries, 3).clone()
    size = torch.from_numpy(baseline["mean_size"]).float()[None, None, :].expand(batch_size, n_queries, 3).clone()
    quat = torch.tensor([0.0, 0.0, 0.0, 1.0])[None, None, :].expand(batch_size, n_queries, 4).clone()

    shape_logits = torch.zeros(batch_size, n_queries, len(SHAPE_ORDER))
    shape_logits[..., baseline["mode_shape_idx"]] = 10.0
    class_logits = torch.zeros(batch_size, n_queries, len(CATEGORY_TO_CLASS_ID))
    class_logits[..., baseline["mode_class_idx"]] = 10.0

    return {
        "existence_logit": existence_logit,
        "position": position,
        "quat": quat,
        "size": size,
        "shape_logits": shape_logits,
        "class_logits": class_logits,
    }


@torch.no_grad()
def run_inference_baseline(baseline: dict, ds: PerceptionDataset, n_queries: int, batch_size: int = 64) -> list[FrameRecord]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    records: list[FrameRecord] = []
    row = 0

    for rgb, states, mask, class_ids in loader:
        b = rgb.shape[0]
        pred = mean_state_baseline_predict(baseline, b, n_queries)
        existence_prob = torch.sigmoid(pred["existence_logit"])
        gt_position = states[..., 0:3]
        gt_quat = states[..., 3:7]
        gt_shape_idx = states[..., 13:16].argmax(dim=-1)
        gt_size = states[..., 16:19]

        matches = hungarian_match(
            pred["position"], pred["class_logits"], pred["size"], gt_position, class_ids, gt_size, mask,
            MatchCostWeights(),
        )
        pred_shape_idx = pred["shape_logits"].argmax(dim=-1)
        pred_class_idx = pred["class_logits"].argmax(dim=-1)

        for i in range(b):
            episode_idx, frame_idx = ds._index[row]
            row += 1
            query_idx, gt_idx = matches[i]
            records.append(
                FrameRecord(
                    episode_idx=episode_idx,
                    frame_idx=frame_idx,
                    n_real=int(mask[i].sum()),
                    query_idx=query_idx,
                    gt_idx=gt_idx,
                    existence_prob=existence_prob[i].numpy(),
                    pred_position=pred["position"][i].numpy(),
                    pred_quat=pred["quat"][i].numpy(),
                    pred_size=pred["size"][i].numpy(),
                    pred_shape_idx=pred_shape_idx[i].numpy(),
                    pred_class_idx=pred_class_idx[i].numpy(),
                    gt_position=gt_position[i].numpy(),
                    gt_quat=gt_quat[i].numpy(),
                    gt_size=gt_size[i].numpy(),
                    gt_shape_idx=gt_shape_idx[i].numpy(),
                    gt_class_idx=class_ids[i].numpy(),
                )
            )
    return records


# --- markdown formatting --------------------------------------------------------


def _format_markdown(all_metrics: dict[str, dict]) -> str:
    lines = ["# Perception eval\n"]
    lines.append("| model | n_frames | existence P | existence R | existence F1 | count-exact | pos err (m) | shape acc | class acc |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, m in all_metrics.items():
        pos = m["matched_position_error_m"]["median"]
        pos_s = f"{pos:.4f}" if pos is not None else "n/a"
        shape_acc = m["shape_accuracy"]
        class_acc = m["class_accuracy"]
        shape_acc_s = f"{shape_acc:.4f}" if shape_acc is not None else "n/a"
        class_acc_s = f"{class_acc:.4f}" if class_acc is not None else "n/a"
        lines.append(
            f"| {name} | {m['n_frames']} | {m['existence']['precision']:.4f} | {m['existence']['recall']:.4f} | "
            f"{m['existence']['f1']:.4f} | {m['count_exact_match_rate']:.4f} | "
            f"{pos_s} | {shape_acc_s} | {class_acc_s} |"
        )
    lines.append("")

    for name, m in all_metrics.items():
        lines.append(f"## {name}: rotation error (deg) by shape\n")
        lines.append("| shape | median | mean | n |")
        lines.append("| --- | --- | --- | --- |")
        for shape, stat in m["matched_rotation_error_deg_by_shape"].items():
            med = f"{stat['median']:.4f}" if stat["median"] is not None else "n/a"
            mean = f"{stat['mean']:.4f}" if stat["mean"] is not None else "n/a"
            lines.append(f"| {shape} | {med} | {mean} | {stat['n']} |")
        lines.append("")

        lines.append(f"## {name}: per-N breakdown\n")
        lines.append("| N | n_frames | existence F1 | matched pos err median (m) |")
        lines.append("| --- | --- | --- | --- |")
        for n, entry in m["per_n"].items():
            if entry is None:
                lines.append(f"| {n} | 0 | n/a | n/a |")
                continue
            pos = entry["matched_position_error_m_median"]
            pos_s = f"{pos:.4f}" if pos is not None else "n/a"
            lines.append(f"| {n} | {entry['n_frames']} | {entry['existence_f1']:.4f} | {pos_s} |")
        lines.append("")

    return "\n".join(lines)


# --- re-render check (gpu) ------------------------------------------------------


def _canonicalize_size(shape_idx: int, size: np.ndarray) -> np.ndarray:
    """Snap a raw, per-axis-independent predicted ``size`` (3,) onto
    ``ObjectSpec.size``'s actual per-shape convention (see
    ``gltfworld.scene.scene``'s module docstring): sphere is isotropic
    (``[r, r, r]``) and cylinder's two radius slots must match
    (``[r, half_height, r]``); ``PerceptionDETR``'s size head has no such
    hard constraint built in (it regresses 3 independent components), so
    without this projection a sphere/cylinder prediction whose components
    happen to disagree would build a physically-inconsistent ``ObjectSpec``
    -- ``mesh_for``/``KHR_implicit_shapes`` only ever read ``size[0]`` for
    the shared radius, so the *other* slot's disagreeing value would
    silently vanish across a save/load round trip (an honest single-radius
    projection here, done once and visibly, instead of a silent per-codec
    truncation nobody documented). Box size is left untouched -- its three
    half-extents are genuinely independent."""
    shape = SHAPE_ORDER[shape_idx]
    size = np.asarray(size, dtype=np.float32)
    if shape == "sphere":
        r = float(size[0])
        return np.array([r, r, r], dtype=np.float32)
    if shape == "cylinder":
        r = float(size[0])
        half_height = float(size[1])
        return np.array([r, half_height, r], dtype=np.float32)
    return size.copy()


def _object_spec_for_query(
    object_id: int,
    shape_idx: int,
    size: np.ndarray,
    color: np.ndarray,
    mass: float,
    friction: float,
    restitution: float,
    category: str,
) -> ObjectSpec:
    return ObjectSpec(
        object_id=object_id,
        shape=SHAPE_ORDER[shape_idx],
        size=_canonicalize_size(shape_idx, size),
        color=color,
        roughness=_DEFAULT_ROUGHNESS,
        metallic=_DEFAULT_METALLIC,
        mass=mass,
        friction=friction,
        restitution=restitution,
        is_static=False,
        category=category,
    )


def build_predicted_episode(template: Episode, record: FrameRecord, threshold: float = EXISTENCE_THRESHOLD) -> Episode:
    """A ``T=1`` predicted :class:`Episode`: one dynamic object per
    existence-thresholded query, plus ``template``'s static (ground) object
    held at its own pose. Color/material for a query come from its
    Hungarian-*matched* GT ``ObjectSpec`` (an honest, documented GT-assist
    for rendering only -- ``PerceptionDETR`` never predicts color/material);
    a thresholded query with no GT match (a false positive) gets
    :data:`_FALSE_POSITIVE_COLOR` and default physics-material values
    instead, since there is no GT object to copy from."""
    template_scene = template.scene
    static_objects = [obj for obj in template_scene.objects if obj.is_static]
    static_indices = [i for i, obj in enumerate(template_scene.objects) if obj.is_static]
    dynamic_objects_template = [obj for obj in template_scene.objects if not obj.is_static]

    matched_query_to_gt = dict(zip(record.query_idx.tolist(), record.gt_idx.tolist()))

    existing_queries = [q for q in range(record.existence_prob.shape[0]) if record.existence_prob[q] >= threshold]

    objects: list[ObjectSpec] = []
    poses = np.zeros((len(existing_queries), 7), dtype=np.float32)
    next_object_id = max((obj.object_id for obj in template_scene.objects), default=-1) + 1

    for row, q in enumerate(existing_queries):
        shape_idx = int(record.pred_shape_idx[q])
        gt_slot = matched_query_to_gt.get(q)
        if gt_slot is not None and gt_slot < len(dynamic_objects_template):
            gt_obj = dynamic_objects_template[gt_slot]
            color, mass, friction, restitution, category = (
                gt_obj.color,
                gt_obj.mass,
                gt_obj.friction,
                gt_obj.restitution,
                gt_obj.category,
            )
        else:
            color, mass, friction, restitution = (
                _FALSE_POSITIVE_COLOR,
                _DEFAULT_MASS,
                _DEFAULT_FRICTION,
                _DEFAULT_RESTITUTION,
            )
            category = CLASS_ID_TO_CATEGORY.get(int(record.pred_class_idx[q]), "ball")

        obj = _object_spec_for_query(
            object_id=next_object_id + row,
            shape_idx=shape_idx,
            size=record.pred_size[q],
            color=color,
            mass=mass,
            friction=friction,
            restitution=restitution,
            category=category,
        )
        objects.append(obj)
        poses[row, 0:3] = record.pred_position[q]
        poses[row, 3:7] = record.pred_quat[q]

    all_objects = objects + static_objects
    all_poses = np.concatenate(
        [poses, np.stack([template.series.poses[0, i] for i in static_indices], axis=0)]
        if static_indices
        else [poses],
        axis=0,
    )

    scene = SceneState(
        objects=all_objects,
        camera=template_scene.camera,
        lights=template_scene.lights,
        gravity=template_scene.gravity,
        dt=template_scene.dt,
        seed=template_scene.seed,
        scene_version=template_scene.scene_version,
    )
    return Episode.single_frame(scene, all_poses)


def render_check(
    records: list[FrameRecord],
    ds: PerceptionDataset,
    episodes_dir: Path,
    out_dir: Path,
    n_samples: int,
    seed: int = 0,
    renderer=None,
) -> dict:
    """``renderer``: an already-open ``EpisodeRenderer`` to reuse (e.g. a
    test's shared session-scoped fixture) -- if ``None`` (the default,
    what the standalone CLI uses), a fresh one is created and deleted here.
    Per ``gltfworld.render.renderer.EpisodeRenderer``'s own documented "one
    live renderer per process" constraint (deleting *any* instance
    terminates the *shared* EGL display for every other still-open
    instance), a caller that already holds one open elsewhere in the same
    process (the test suite's shared ``episode_renderer`` fixture, in
    particular) must pass it in here rather than let this function create
    -- and later delete -- a second one, which would otherwise silently
    break every other renderer left open in that process."""
    from gltfworld.cli import run_validator
    from gltfworld.scene.contract import episode_to_tensors

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(records), size=min(n_samples, len(records)), replace=False)

    pred_dir = out_dir / "pred_frames"
    pred_dir.mkdir(parents=True, exist_ok=True)

    owns_renderer = renderer is None
    if owns_renderer:
        from gltfworld.render.renderer import EpisodeRenderer

        renderer = EpisodeRenderer(width=256, height=256)
    psnrs, ssims = [], []
    validator_errors_total = 0
    roundtrip_max_abs_err = 0.0

    try:
        for idx in sample_idx:
            record = records[int(idx)]
            template = load_episode(episodes_dir / f"ep_{record.episode_idx:06d}.glb")

            pred_episode = build_predicted_episode(template, record)
            glb_path = pred_dir / f"ep_{record.episode_idx:06d}_f{record.frame_idx:04d}.glb"
            save_episode(pred_episode, glb_path)

            # round-trip check: reload and compare tensor-contract states.
            reloaded = load_episode(glb_path)
            reloaded_tensors = episode_to_tensors(reloaded)
            original_tensors = episode_to_tensors(pred_episode)
            if reloaded_tensors["states"].size and original_tensors["states"].size:
                err = float(np.max(np.abs(reloaded_tensors["states"] - original_tensors["states"])))
                roundtrip_max_abs_err = max(roundtrip_max_abs_err, err)

            report = run_validator(str(glb_path))
            validator_errors_total += report.get("issues", {}).get("numErrors", 0)

            renderer.load(pred_episode)
            renderer.set_frame(0)
            pred_rgb = renderer.render().rgb

            rgb_mmap = ds._rgb_mmaps[record.episode_idx]
            gt_rgb = np.asarray(rgb_mmap[record.frame_idx])

            psnrs.append(psnr(pred_rgb, gt_rgb))
            ssims.append(ssim(pred_rgb, gt_rgb))
    finally:
        if owns_renderer:
            renderer.delete()

    def _stat(values: list[float]) -> dict:
        finite = [v for v in values if np.isfinite(v)]
        arr = np.asarray(finite, dtype=np.float64)
        return {
            "median": float(np.median(arr)) if arr.size else None,
            "mean": float(np.mean(arr)) if arr.size else None,
            "n": int(arr.size),
            "n_infinite": len(values) - len(finite),
        }

    return {
        "n_samples": int(sample_idx.size),
        "psnr_db": _stat(psnrs),
        "ssim": _stat(ssims),
        "roundtrip_max_abs_err": roundtrip_max_abs_err,
        "validate_clean": validator_errors_total == 0,
        "validator_total_errors": validator_errors_total,
        "pred_frames_dir": str(pred_dir),
    }


# --- CLI -------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Eval CLI for gltfworld's PerceptionDETR model.")
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="dataset root (e.g. data/perception-v1)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--render-samples", type=int, default=DEFAULT_RENDER_SAMPLES, help="0 disables the GPU re-render check"
    )
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    episodes_dir, pack_file = _resolve_paths(args.data)
    ds = PerceptionDataset(episodes_dir, pack_file, split=args.split)
    train_ds = PerceptionDataset(episodes_dir, pack_file, split="train")
    print(f"eval split={args.split!r}: {len(ds)} frames; train split (for baseline stats): {len(train_ds)} frames")

    model = _load_model_from_ckpt(args.ckpt, device)
    n_params = count_params(model)
    print(f"model params={n_params:,}")

    model_records = run_inference(model, ds, device, batch_size=args.batch_size)
    model_metrics = compute_metrics(model_records, name="PerceptionDETR")

    baseline_stats = build_mean_state_baseline(train_ds)
    baseline_records = run_inference_baseline(baseline_stats, ds, n_queries=model.n_queries, batch_size=args.batch_size)
    baseline_metrics = compute_metrics(baseline_records, name="mean-state baseline")

    all_metrics = {"PerceptionDETR": model_metrics, "mean-state baseline": baseline_metrics}

    result = {
        "split": args.split,
        "n_frames": len(ds),
        "n_params": n_params,
        "baseline_stats": {
            "mean_count": baseline_stats["mean_count"],
            "mean_position": baseline_stats["mean_position"].tolist(),
            "mean_size": baseline_stats["mean_size"].tolist(),
            "mode_shape": SHAPE_ORDER[baseline_stats["mode_shape_idx"]],
            "mode_class": CLASS_ID_TO_CATEGORY[baseline_stats["mode_class_idx"]],
        },
        "metrics": all_metrics,
    }

    if args.render_samples > 0:
        render_result = render_check(model_records, ds, episodes_dir, args.out, args.render_samples)
        result["render_check"] = render_result
        print(
            f"render check: {render_result['n_samples']} frames, "
            f"psnr median={render_result['psnr_db']['median']}, ssim median={render_result['ssim']['median']}, "
            f"roundtrip_max_abs_err={render_result['roundtrip_max_abs_err']:.2e}, "
            f"validate_clean={render_result['validate_clean']}"
        )

    metrics_path = args.out / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2))
    md_path = args.out / "metrics.md"
    md_path.write_text(_format_markdown(all_metrics))
    print(f"wrote {metrics_path}, {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
