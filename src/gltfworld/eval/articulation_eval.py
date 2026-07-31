"""Eval CLI for ``gltfworld.models.articulation.ArticulationEstimator``
(V9, joint-state estimation): joint-position error (degrees for hinges, cm
for sliders -- reported separately, never mixed since they're different
physical quantities), joint-type accuracy, axis angular error (degrees), two
context baselines, and (optionally, GPU) a re-render PSNR/SSIM check against
the actually-rendered GT frames.

    uv run python -m gltfworld.eval.articulation_eval \\
        --ckpt runs/articulation-v1/best.safetensors \\
        --data data/articulated-v1 --split test \\
        --out runs/articulation-v1/eval

Metrics (``metrics.json``/``metrics.md``):

- **Joint position error**: over every hinge (revolute) test frame, the
  absolute difference between the model's denormalized prediction and the
  recorded ``joint_pos`` (see ``gltfworld.models.articulation``'s "Joint
  position normalization" note for why the model regresses a *normalized*
  target and how it's denormalized back to raw units here), reported in
  **degrees**; separately, over every slider (prismatic) test frame, the
  same in **centimeters**. These are never averaged together -- a hinge's
  radians and a slider's meters aren't the same unit, let alone the same
  physical quantity.
- **Type accuracy**: fraction of frames where ``argmax(type_logits)``
  matches the recorded joint type.
- **Axis angular error**: ``arccos(pred_axis . gt_axis)`` in degrees, over
  every test frame (``gt_axis`` is one of the world X/Y/Z basis vectors, see
  ``gltfworld.datagen.articulated``'s sampler).

Baselines (context, not a bar to clear -- see below):

- **predict-midpoint-of-range**: always predicts the joint at the midpoint
  of its own known ``[limit_min, limit_max]`` (normalized 0.5) -- scored
  only on joint-position error (it makes no type/axis prediction at all).
- **predict-dataset-mean-axis**: always predicts the (train-split) mean
  axis vector, re-normalized to unit length -- scored only on axis error.

**Ditto context, not a bar** (Jiang et al. 2022, "Ditto: Building Digital
Twins of Articulated Objects from Interaction"): Ditto reports a median
revolute-axis error of **1.36 degrees**. That number is not directly
comparable to this milestone's axis-error metric -- Ditto's task is a
different *input modality and problem shape* (point clouds, a **before/
after** interaction pair given as input, and axis estimation via explicit
3D geometric fitting to the observed part motion) solving for the same
physical quantity from much richer, motion-disambiguating input than a
*single* RGB frame with no temporal/interaction signal at all. This
milestone's task (a single rendered frame -> joint type/axis/position, no
point cloud, no before/after pair) is intentionally simpler-input and is
not attempting to match or beat Ditto's number; it is reported here purely
as external context for the reader, not as an acceptance bar (see
DESIGN.md's V9 section for the same caveat in prose).

Re-render check (``--render-samples K``, needs the ``render`` extra + GPU)
---------------------------------------------------------------------------

For ``K`` sampled test frames: denormalize the model's predicted joint
position back to raw units (using that episode's own known
``[limit_min, limit_max]``), forward-kinematically reconstruct the moving
``part``'s (and, if present, the rigidly-following ``handle``'s) pose at
that predicted joint position -- the exact same anchor/axis composition
``tests/test_articulated_physics.py``'s articulation-consistency check
verifies against MuJoCo's own simulated trajectory, just run in the
*predict* direction instead of the *verify* direction -- build a ``T=1``
predicted :class:`~gltfworld.scene.episode.Episode` (with a real
``joint_position`` channel, using the same verified transport codec, not a
new encoding), render it, and compare against the actual stored GT rgb
frame via PSNR/SSIM (``gltfworld.eval.metrics``). Predicted frames are also
saved as real, independently loadable
``pred_frames/ep_XXXXXX_fYYYY.glb`` (``gltfworld.scene.convert.save_episode``)
-- each one is round-trip verified inline (reloading reproduces the exact
poses/joint_pos that built it) and run through the real, pinned
glTF-Validator (``gltfworld validate``, via ``gltfworld.cli.run_validator``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gltfworld.data.dataset import ArticulationDataset
from gltfworld.eval.metrics import psnr, ssim
from gltfworld.models.articulation import JOINT_TYPE_NAMES, ArticulationEstimator, count_params, denormalize_joint_pos
from gltfworld.scene.convert import load_episode, save_episode
from gltfworld.scene.episode import Episode, StateSeries

RAD2DEG = 180.0 / np.pi
DEFAULT_RENDER_SAMPLES = 50

DITTO_CONTEXT_NOTE = (
    "Ditto (Jiang et al. 2022) reports a median revolute-axis error of 1.36 degrees, from point-cloud "
    "before/after-interaction input -- a different, motion-disambiguating input modality than this "
    "milestone's single-RGB-frame task. Reported here as external context only, not as an acceptance bar."
)

# --- acceptance bar (see DESIGN.md's V9 section / docs/VERIFICATION.md's V9 checkpoint) ---
ACCEPTANCE_HINGE_MEDIAN_DEG = 5.0
ACCEPTANCE_SLIDER_MEDIAN_CM = 2.0
ACCEPTANCE_TYPE_ACC = 0.98
ACCEPTANCE_AXIS_MEDIAN_DEG = 10.0


# --- model / data loading -----------------------------------------------------


def _resolve_paths(data_dir: Path) -> tuple[Path, Path]:
    episodes_dir = data_dir / "episodes"
    packed_dir = data_dir / "packed"
    candidates = sorted(packed_dir.glob("*.safetensors"))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one .safetensors file in {packed_dir}, found {len(candidates)}")
    return episodes_dir, candidates[0]


def _load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> ArticulationEstimator:
    from safetensors.torch import load_file

    from gltfworld.train.train_articulation import Config, make_model

    config_path = ckpt_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found -- articulation eval needs the training run's config.json "
            f"(next to the checkpoint) to know the model architecture"
        )
    cfg = Config.load(config_path)
    model = make_model(cfg).to(device)
    model.load_state_dict(load_file(ckpt_path, device=str(device)))
    model.eval()
    return model


# --- inference: collect every prediction + GT for a split ---------------------


class FrameRecord:
    __slots__ = (
        "episode_idx",
        "frame_idx",
        "pred_joint_pos_norm",
        "pred_type_id",
        "pred_axis",
        "gt_joint_pos_norm",
        "gt_type_id",
        "gt_axis",
        "limit_min",
        "limit_max",
    )

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@torch.no_grad()
def run_inference(model: torch.nn.Module, ds: ArticulationDataset, device: torch.device, batch_size: int = 64):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    records: list[FrameRecord] = []
    row = 0
    for rgb, joint_pos_norm, joint_type_id, axis, limit_min, limit_max in loader:
        rgb_d = rgb.to(device)
        pred = model(rgb_d)
        pred_type_id = pred["type_logits"].argmax(dim=-1).cpu().numpy()
        pred_joint_pos_norm = pred["joint_pos_norm"].cpu().numpy()
        pred_axis = pred["axis"].cpu().numpy()

        b = rgb.shape[0]
        for i in range(b):
            episode_idx, frame_idx = ds._index[row]
            row += 1
            records.append(
                FrameRecord(
                    episode_idx=episode_idx,
                    frame_idx=frame_idx,
                    pred_joint_pos_norm=float(pred_joint_pos_norm[i]),
                    pred_type_id=int(pred_type_id[i]),
                    pred_axis=pred_axis[i],
                    gt_joint_pos_norm=float(joint_pos_norm[i]),
                    gt_type_id=int(joint_type_id[i]),
                    gt_axis=axis[i].numpy(),
                    limit_min=float(limit_min[i]),
                    limit_max=float(limit_max[i]),
                )
            )
    return records


# --- metrics -------------------------------------------------------------------


def _stat(values: list[float]) -> dict:
    if not values:
        return {"median": None, "mean": None, "n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(arr)), "mean": float(np.mean(arr)), "n": int(arr.size)}


def compute_metrics(
    records: list[FrameRecord],
    name: str,
    *,
    include_joint_pos: bool = True,
    include_type: bool = True,
    include_axis: bool = True,
) -> dict:
    hinge_errs_deg: list[float] = []
    slider_errs_cm: list[float] = []
    type_correct: list[bool] = []
    axis_errs_deg: list[float] = []

    for r in records:
        if include_joint_pos:
            raw_pred = r.pred_joint_pos_norm * (r.limit_max - r.limit_min) + r.limit_min
            raw_gt = r.gt_joint_pos_norm * (r.limit_max - r.limit_min) + r.limit_min
            abs_err = abs(raw_pred - raw_gt)
            if JOINT_TYPE_NAMES[r.gt_type_id] == "revolute":
                hinge_errs_deg.append(abs_err * RAD2DEG)
            else:
                slider_errs_cm.append(abs_err * 100.0)

        if include_type:
            type_correct.append(r.pred_type_id == r.gt_type_id)

        if include_axis:
            cos_sim = float(np.clip(np.dot(r.pred_axis, r.gt_axis), -1.0, 1.0))
            axis_errs_deg.append(float(np.arccos(cos_sim)) * RAD2DEG)

    return {
        "name": name,
        "n_frames": len(records),
        "joint_pos_error_hinge_deg": _stat(hinge_errs_deg) if include_joint_pos else None,
        "joint_pos_error_slider_cm": _stat(slider_errs_cm) if include_joint_pos else None,
        "type_accuracy": (float(np.mean(type_correct)) if type_correct else None) if include_type else None,
        "axis_error_deg": _stat(axis_errs_deg) if include_axis else None,
    }


def check_acceptance(model_metrics: dict) -> dict:
    hinge_median = (model_metrics["joint_pos_error_hinge_deg"] or {}).get("median")
    slider_median = (model_metrics["joint_pos_error_slider_cm"] or {}).get("median")
    type_acc = model_metrics["type_accuracy"]
    axis_median = (model_metrics["axis_error_deg"] or {}).get("median")

    checks = {
        "hinge_median_deg": {
            "value": hinge_median,
            "bar": ACCEPTANCE_HINGE_MEDIAN_DEG,
            "pass": hinge_median is not None and hinge_median <= ACCEPTANCE_HINGE_MEDIAN_DEG,
        },
        "slider_median_cm": {
            "value": slider_median,
            "bar": ACCEPTANCE_SLIDER_MEDIAN_CM,
            "pass": slider_median is not None and slider_median <= ACCEPTANCE_SLIDER_MEDIAN_CM,
        },
        "type_accuracy": {
            "value": type_acc,
            "bar": ACCEPTANCE_TYPE_ACC,
            "pass": type_acc is not None and type_acc >= ACCEPTANCE_TYPE_ACC,
        },
        "axis_median_deg": {
            "value": axis_median,
            "bar": ACCEPTANCE_AXIS_MEDIAN_DEG,
            "pass": axis_median is not None and axis_median <= ACCEPTANCE_AXIS_MEDIAN_DEG,
        },
    }
    checks["all_pass"] = all(c["pass"] for c in checks.values())
    return checks


# --- baselines ------------------------------------------------------------------


def midpoint_baseline_records(records: list[FrameRecord]) -> list[FrameRecord]:
    """``predict-midpoint-of-range``: joint_pos_norm always 0.5 -- scored
    only on joint-position error (see module docstring)."""
    out = []
    for r in records:
        out.append(
            FrameRecord(
                episode_idx=r.episode_idx,
                frame_idx=r.frame_idx,
                pred_joint_pos_norm=0.5,
                pred_type_id=r.gt_type_id,  # not scored (include_type=False below)
                pred_axis=r.gt_axis,  # not scored (include_axis=False below)
                gt_joint_pos_norm=r.gt_joint_pos_norm,
                gt_type_id=r.gt_type_id,
                gt_axis=r.gt_axis,
                limit_min=r.limit_min,
                limit_max=r.limit_max,
            )
        )
    return out


def mean_axis_baseline(train_ds: ArticulationDataset) -> np.ndarray:
    """Mean of every train-split episode's own axis vector (one per
    episode, not per frame -- every frame of an episode shares the same
    axis, so weighting by frame would just repeat the same vector T times
    for no reason), re-normalized to unit length."""
    episode_idxs = sorted({e for e, _f in train_ds._index})
    axes = np.stack([np.asarray(train_ds.axis[e]) for e in episode_idxs], axis=0)
    mean_axis = axes.mean(axis=0)
    norm = np.linalg.norm(mean_axis)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (mean_axis / norm).astype(np.float32)


def mean_axis_baseline_records(records: list[FrameRecord], mean_axis: np.ndarray) -> list[FrameRecord]:
    """``predict-dataset-mean-axis``: always predicts the fixed
    ``mean_axis`` -- scored only on axis error (see module docstring)."""
    out = []
    for r in records:
        out.append(
            FrameRecord(
                episode_idx=r.episode_idx,
                frame_idx=r.frame_idx,
                pred_joint_pos_norm=r.gt_joint_pos_norm,  # not scored
                pred_type_id=r.gt_type_id,  # not scored
                pred_axis=mean_axis,
                gt_joint_pos_norm=r.gt_joint_pos_norm,
                gt_type_id=r.gt_type_id,
                gt_axis=r.gt_axis,
                limit_min=r.limit_min,
                limit_max=r.limit_max,
            )
        )
    return out


# --- markdown formatting --------------------------------------------------------


def _format_markdown(all_metrics: dict[str, dict], acceptance: dict) -> str:
    lines = ["# Articulation (V9) eval\n"]
    lines.append("| model | n_frames | hinge err (deg) | slider err (cm) | type acc | axis err (deg) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for name, m in all_metrics.items():
        hinge = m["joint_pos_error_hinge_deg"]
        slider = m["joint_pos_error_slider_cm"]
        hinge_s = f"{hinge['median']:.4f}" if hinge and hinge["median"] is not None else "n/a"
        slider_s = f"{slider['median']:.4f}" if slider and slider["median"] is not None else "n/a"
        type_s = f"{m['type_accuracy']:.4f}" if m["type_accuracy"] is not None else "n/a"
        axis = m["axis_error_deg"]
        axis_s = f"{axis['median']:.4f}" if axis and axis["median"] is not None else "n/a"
        lines.append(f"| {name} | {m['n_frames']} | {hinge_s} | {slider_s} | {type_s} | {axis_s} |")
    lines.append("")

    lines.append("## Acceptance\n")
    lines.append("| check | value | bar | pass |")
    lines.append("| --- | --- | --- | --- |")
    for key in ("hinge_median_deg", "slider_median_cm", "type_accuracy", "axis_median_deg"):
        c = acceptance[key]
        val_s = f"{c['value']:.4f}" if c["value"] is not None else "n/a"
        lines.append(f"| {key} | {val_s} | {c['bar']} | {c['pass']} |")
    lines.append(f"\n**all_pass: {acceptance['all_pass']}**\n")

    lines.append(f"## Context\n\n{DITTO_CONTEXT_NOTE}\n")
    return "\n".join(lines)


# --- re-render check (gpu) ------------------------------------------------------


def _rotate_about_cardinal_axis(v: np.ndarray, axis: int, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula, specialized to a cardinal (X/Y/Z) axis --
    identical math to ``tests/test_articulated_physics.py``'s helper of the
    same name (that test independently verifies this composition against
    MuJoCo's own simulated trajectory; this module runs the same formula in
    the *predict* direction)."""
    k = np.zeros(3)
    k[axis] = 1.0
    return v * np.cos(angle) + np.cross(k, v) * np.sin(angle) + k * np.dot(k, v) * (1.0 - np.cos(angle))


def _axis_angle_quat_xyzw(axis: int, angle: float) -> np.ndarray:
    k = np.zeros(3)
    k[axis] = 1.0
    half = angle / 2.0
    return np.array([k[0] * np.sin(half), k[1] * np.sin(half), k[2] * np.sin(half), np.cos(half)])


def _quat_conj_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qxyz = q[0:3]
    w = q[3]
    t = 2.0 * np.cross(qxyz, v)
    return v + w * t + np.cross(qxyz, t)


def build_predicted_episode(template: Episode, pred_joint_pos_raw: float) -> Episode:
    """A ``T=1`` predicted :class:`Episode`: reconstructs the moving
    ``part`` (and its rigidly-following ``handle``, if present)'s pose at
    ``pred_joint_pos_raw`` via forward kinematics from ``ArticulatedSpec``'s
    own ``anchor``/``axis`` metadata plus ``template``'s own frame-0 recorded
    pose/``joint_pos`` (to derive the rest-configuration offset and, for the
    handle, its fixed local attachment offset) -- no privileged access to
    the original ``SampledArticulatedScene`` that generated ``template``.
    Every other object (ground/base, static) keeps its own frame-0 pose."""
    scene = template.scene
    art = scene.articulations[0]
    obj_ids = [o.object_id for o in scene.objects]
    part_index = obj_ids.index(art.part_object_id)

    poses0 = template.series.poses[0].astype(np.float64)
    jp0 = float(template.series.joint_pos[0, 0])
    anchor = art.anchor.astype(np.float64)
    axis_idx = int(art.axis)

    part_pos0 = poses0[part_index, 0:3]
    if art.joint_type == "revolute":
        rest_offset = _rotate_about_cardinal_axis(part_pos0 - anchor, axis_idx, -jp0)
        pred_pos = anchor + _rotate_about_cardinal_axis(rest_offset, axis_idx, pred_joint_pos_raw)
        pred_rot = _axis_angle_quat_xyzw(axis_idx, pred_joint_pos_raw)
    else:
        axis_vec = np.zeros(3)
        axis_vec[axis_idx] = 1.0
        rest_offset = part_pos0 - anchor - axis_vec * jp0
        pred_pos = anchor + rest_offset + axis_vec * pred_joint_pos_raw
        pred_rot = np.array([0.0, 0.0, 0.0, 1.0])

    poses = poses0.copy()
    poses[part_index, 0:3] = pred_pos
    poses[part_index, 3:7] = pred_rot

    if art.handle_object_id is not None:
        handle_index = obj_ids.index(art.handle_object_id)
        part_rot0 = poses0[part_index, 3:7]
        handle_pos0 = poses0[handle_index, 0:3]
        handle_local_offset = _quat_rotate_xyzw(_quat_conj_xyzw(part_rot0), handle_pos0 - part_pos0)
        poses[handle_index, 0:3] = pred_pos + _quat_rotate_xyzw(pred_rot, handle_local_offset)
        poses[handle_index, 3:7] = pred_rot

    series = StateSeries(
        times=np.array([0.0], dtype=np.float32),
        poses=poses[None, ...].astype(np.float32),
        joint_pos=np.array([[pred_joint_pos_raw]], dtype=np.float32),
    )
    return Episode(scene=scene, series=series)


def render_check(
    records: list[FrameRecord],
    ds: ArticulationDataset,
    episodes_dir: Path,
    out_dir: Path,
    n_samples: int,
    seed: int = 0,
    renderer=None,
) -> dict:
    """See ``gltfworld.eval.perception_eval.render_check``'s identical
    ``renderer=`` reuse note (deleting one ``EpisodeRenderer`` kills the
    shared EGL display for every other still-open instance in the same
    process)."""
    from gltfworld.cli import run_validator

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

            raw_pred = record.pred_joint_pos_norm * (record.limit_max - record.limit_min) + record.limit_min
            pred_episode = build_predicted_episode(template, raw_pred)

            glb_path = pred_dir / f"ep_{record.episode_idx:06d}_f{record.frame_idx:04d}.glb"
            save_episode(pred_episode, glb_path)

            reloaded = load_episode(glb_path)
            pos_err = float(
                np.max(np.abs(reloaded.series.poses - pred_episode.series.poses))
            )
            jp_err = float(np.max(np.abs(reloaded.series.joint_pos - pred_episode.series.joint_pos)))
            roundtrip_max_abs_err = max(roundtrip_max_abs_err, pos_err, jp_err)

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

    def _finite_stat(values: list[float]) -> dict:
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
        "psnr_db": _finite_stat(psnrs),
        "ssim": _finite_stat(ssims),
        "roundtrip_max_abs_err": roundtrip_max_abs_err,
        "validate_clean": validator_errors_total == 0,
        "validator_total_errors": validator_errors_total,
        "pred_frames_dir": str(pred_dir),
    }


# --- CLI -------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Eval CLI for gltfworld's ArticulationEstimator model.")
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="dataset root (e.g. data/articulated-v1)")
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
    ds = ArticulationDataset(episodes_dir, pack_file, split=args.split)
    train_ds = ArticulationDataset(episodes_dir, pack_file, split="train")
    print(f"eval split={args.split!r}: {len(ds)} frames; train split (for baseline stats): {len(train_ds)} frames")

    model = _load_model_from_ckpt(args.ckpt, device)
    n_params = count_params(model)
    print(f"model params={n_params:,}")

    model_records = run_inference(model, ds, device, batch_size=args.batch_size)
    model_metrics = compute_metrics(model_records, name="ArticulationEstimator")

    midpoint_records = midpoint_baseline_records(model_records)
    midpoint_metrics = compute_metrics(
        midpoint_records, name="predict-midpoint-of-range", include_type=False, include_axis=False
    )

    mean_axis = mean_axis_baseline(train_ds)
    mean_axis_records = mean_axis_baseline_records(model_records, mean_axis)
    mean_axis_metrics = compute_metrics(
        mean_axis_records, name="predict-dataset-mean-axis", include_joint_pos=False, include_type=False
    )

    all_metrics = {
        "ArticulationEstimator": model_metrics,
        "predict-midpoint-of-range": midpoint_metrics,
        "predict-dataset-mean-axis": mean_axis_metrics,
    }
    acceptance = check_acceptance(model_metrics)

    result = {
        "split": args.split,
        "n_frames": len(ds),
        "n_params": n_params,
        "mean_axis_baseline_vector": mean_axis.tolist(),
        "metrics": all_metrics,
        "acceptance": acceptance,
        "ditto_context_note": DITTO_CONTEXT_NOTE,
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
    md_path.write_text(_format_markdown(all_metrics, acceptance))
    print(f"wrote {metrics_path}, {md_path}")
    print(f"acceptance: {json.dumps(acceptance, indent=2)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
