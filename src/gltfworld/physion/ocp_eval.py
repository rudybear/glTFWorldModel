"""Object Contact Prediction (OCP) evaluation on Physion Collide (V8 stage 3).

"Will the red (agent/target) object touch the yellow (patient/zone) region
at any point during the trial?" -- evaluated three ways, all against the
same real per-frame `target_contacting_zone` ground truth (see
docs/PHYSION.md's schema section):

- **GT-contact oracle**: full ground-truth trajectories (real per-object
  mesh geometry, transformed by real per-frame pose -- no model, no
  approximation beyond triangulated-mesh nearest-vertex distance), a
  proximity-threshold ceiling this milestone calibrates on a held-out 50
  trials. Bounds everything below it.
- **Our dynamics rollout**: the first ``K_SEED_FRAMES`` of real GT state,
  mapped into gltfworld's own D=22 tensor contract via the *existing*
  ``gltfworld.scene.contract``/``gltfworld.physion.convert`` pipeline
  (stage 2's ``trial_to_episode`` -> ``episode_to_tensors``, no new
  conversion logic here), rolled forward to the trial's last recorded frame
  with ``InteractionTransformer`` (``runs/dynamics-v1/best.safetensors``),
  contact predicted via the *same* calibrated threshold but on a coarse
  bounding-*sphere* approximation (mean AABB half-extent -- a strong,
  deliberately simple approximation, see :func:`collision_radius`), since
  the model has no concept of real mesh geometry at all.
- **Ballistic control**: identical protocol, ``BallisticBaseline`` instead
  of the learned model -- isolates what learned dynamics adds over pure
  physics extrapolation.

CLI:

    uv run python -m gltfworld.physion.ocp_eval \\
        --hdf5-dir data/external/physion/hdf5/extracted/Collide/hdf5s \\
        --glb-dir data/external/physion/glb/Collide \\
        --dynamics-ckpt runs/dynamics-v1/best.safetensors \\
        --out runs/physion-ocp-v1
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pygltflib
import torch

from gltfworld.eval.rollout import rollout
from gltfworld.models.baselines import BallisticBaseline
from gltfworld.physion.convert import (
    PhysionTrial,
    load_trial,
    quat_rotate_points,
    save_trial_gltf,
)
from gltfworld.scene.contract import episode_to_tensors
from gltfworld.scene.convert import episode_from_gltf
from gltfworld.scene.episode import Episode

# gltfworld.scene.contract.episode_to_tensors (built for wm-scenes-v1's own
# 3-category world) raises on any ObjectSpec.category outside
# CATEGORY_TO_CLASS_ID -- real Physion model names ("cone", "torus",
# "648972_chair_poliform_harmony", ...) are never in that set (a real domain-
# gap finding in its own right, see docs/PHYSION.md). ``class_ids`` isn't
# consumed by the dynamics model at all (only ``shape_onehot``, already
# sphere/box per stage 2's classifier), so this remap is purely to satisfy
# episode_to_tensors' own contract -- it never touches the GLB's real,
# already-encoded ``category``/``extras.rwm`` (see gltfworld.physion.convert),
# only a throwaway copy built just for this tensor-mapping call.
_SHAPE_TO_CONTRACT_CATEGORY = {"sphere": "ball", "box": "crate", "cylinder": "cylinder"}


def _episode_for_contract(episode: Episode) -> Episode:
    remapped_objects = [
        dataclasses.replace(obj, category=_SHAPE_TO_CONTRACT_CATEGORY[obj.shape]) for obj in episode.scene.objects
    ]
    remapped_scene = dataclasses.replace(episode.scene, objects=remapped_objects)
    return Episode(scene=remapped_scene, series=episode.series)


K_SEED_FRAMES = 15
N_CALIBRATION = 50
CALIBRATION_SEED = "gltfworld-physion-ocp-v1"

# A log-ish spaced grid over gltfworld's own object-scale range
# (wm-scenes-v1 objects are 0.05-0.25m characteristic size, DESIGN.md) plus
# Physion's (broadly similar-order) object sizes -- wide enough to include
# both "basically touching" and "same-room" thresholds.
DEFAULT_THRESHOLD_GRID = (
    0.0,
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
)


# --- binomial confidence interval -----------------------------------------------


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    """Wilson score interval (95% by default) for a binomial proportion --
    better-behaved than the normal approximation at small `n` or extreme
    `p` (both apply here: n=50-150, accuracy plausibly near/at 1.0 for the
    oracle). Returns ``(phat, lo, hi)``."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return phat, max(0.0, center - half), min(1.0, center + half)


# --- calibration split -----------------------------------------------------------


def calibration_split(trial_ids: list[str], n_calib: int = N_CALIBRATION, seed: str = CALIBRATION_SEED) -> set[str]:
    """Deterministic pseudo-random 50-trial calibration subset (sha256 of a
    fixed seed + trial_id, same style as ``gltfworld.data.pack``'s
    seed-keyed split scheme) -- reproducible without depending on file
    iteration order."""

    def _key(trial_id: str) -> str:
        return hashlib.sha256(f"{seed}:{trial_id}".encode()).hexdigest()

    ordered = sorted(trial_ids, key=_key)
    return set(ordered[:n_calib])


# --- GT-contact oracle (real mesh geometry, full trajectory) --------------------


def oracle_min_distance_curve(agent, patient) -> np.ndarray:
    """Per-frame nearest-vertex world-space distance between two
    ``PhysionObject``s' *real* mesh geometry (transformed by their real
    per-frame pose) -- the GT-contact oracle's proximity signal. A
    real, honest approximation: nearest *vertex*-to-vertex, not true
    surface-to-surface distance (slightly overestimates true minimum
    surface separation for coarse meshes) -- absorbed by the empirical
    threshold calibration below, not corrected for separately."""
    t = agent.positions.shape[0]
    # patient (zone) is empirically static in every Collide trial checked
    # (docs/PHYSION.md) -- transform its vertices once, not per frame.
    patient_static = bool(np.max(np.abs(patient.positions - patient.positions[0])) < 1e-6) and bool(
        np.max(np.abs(patient.rotations - patient.rotations[0])) < 1e-6
    )
    out = np.empty(t, dtype=np.float64)
    if patient_static:
        patient_world = quat_rotate_points(patient.rotations[0], patient.vertices) + patient.positions[0]
        for i in range(t):
            agent_world = quat_rotate_points(agent.rotations[i], agent.vertices) + agent.positions[i]
            d = np.linalg.norm(agent_world[:, None, :] - patient_world[None, :, :], axis=-1)
            out[i] = d.min()
    else:
        for i in range(t):
            agent_world = quat_rotate_points(agent.rotations[i], agent.vertices) + agent.positions[i]
            patient_world = quat_rotate_points(patient.rotations[i], patient.vertices) + patient.positions[i]
            d = np.linalg.norm(agent_world[:, None, :] - patient_world[None, :, :], axis=-1)
            out[i] = d.min()
    return out


# --- our-dynamics / ballistic: D=22 contract + rollout ---------------------------


def collision_radius(size: np.ndarray) -> float:
    """Scalar bounding-*sphere* radius from an ``ObjectSpec.size``-shaped
    array -- collapses the sphere/box distinction stage 2's
    ``classify_bounding_shape`` makes for KHR physics into a single number
    for contact-distance purposes (exact for "sphere", a simple, honestly
    coarse approximation for "box"). Shared by the our-dynamics and
    ballistic tracks (never the oracle, which uses real mesh geometry)."""
    return float(np.mean(size))


def role_position_track(trial_episode, tensors: dict, role_object_id: int, model, k_seed: int) -> np.ndarray:
    """World position of ``role_object_id`` at every recorded frame of the
    trial: if it's a static object (per ``gltfworld.scene.contract``'s
    is_static exclusion), its position never changes at all -- return the
    constant. If it's dynamic, the first ``k_seed`` frames are real GT and
    the rest are ``model``'s own autoregressive rollout (never re-fed
    ground truth beyond frame ``k_seed - 1``, per
    ``gltfworld.eval.rollout.rollout``'s own semantics).
    """
    static_key = str(role_object_id)
    if static_key in tensors["static"]:
        t_total = tensors["states"].shape[0]
        pos = tensors["static"][static_key]["pos"]
        return np.tile(pos, (t_total, 1))

    dynamic_object_ids = [obj.object_id for obj in trial_episode.scene.objects if not obj.is_static]
    row = dynamic_object_ids.index(role_object_id)

    states = tensors["states"]
    mask = tensors["mask"]
    globals_ = tensors["globals"]
    t_total = states.shape[0]
    k = min(k_seed, t_total)

    initial_state = torch.from_numpy(states[k - 1]).float()
    mask_t = torch.from_numpy(mask)
    globals_t = torch.from_numpy(globals_).float()
    n_roll = t_total - (k - 1)

    pred = rollout(model, initial_state, mask_t, globals_t, n_roll).detach().cpu().numpy()
    full_states = np.concatenate([states[: k - 1], pred], axis=0) if k > 1 else pred
    return full_states[:, row, 0:3]


def dynamics_track_contact_curve(trial: PhysionTrial, glb_path: Path, model, k_seed: int = K_SEED_FRAMES) -> np.ndarray:
    """Center-to-center-minus-radii distance curve for the agent/patient
    pair, over the whole trial, using ``model`` for the unobserved tail
    (see :func:`role_position_track`) -- shared by the our-dynamics and
    ballistic tracks (only ``model`` differs)."""
    gltf = pygltflib.GLTF2.load(str(glb_path))
    episode = episode_from_gltf(gltf)
    tensors = episode_to_tensors(_episode_for_contract(episode))

    agent_pos = role_position_track(episode, tensors, trial.target_id, model, k_seed)
    patient_pos = role_position_track(episode, tensors, trial.zone_id, model, k_seed)

    obj_by_id = {obj.object_id: obj for obj in episode.scene.objects}
    r_agent = collision_radius(obj_by_id[trial.target_id].size)
    r_patient = collision_radius(obj_by_id[trial.zone_id].size)

    center_dist = np.linalg.norm(agent_pos - patient_pos, axis=-1)
    return center_dist - (r_agent + r_patient)


# --- scoring -----------------------------------------------------------------------


def predict_label(distance_curve: np.ndarray, threshold: float) -> bool:
    return bool(np.any(distance_curve <= threshold))


def accuracy(preds: list[bool], labels: list[bool]) -> tuple[int, int]:
    k = sum(1 for p, y in zip(preds, labels) if p == y)
    return k, len(labels)


def calibrate_threshold(curves: dict[str, np.ndarray], labels: dict[str, bool], grid=DEFAULT_THRESHOLD_GRID) -> tuple[float, float]:
    """Grid search over ``grid``, maximizing accuracy on the given
    (calibration) trials. Ties broken toward the *smaller* threshold (a
    tighter contact definition, the more conservative choice)."""
    best_threshold = grid[0]
    best_acc = -1.0
    for threshold in grid:
        preds = [predict_label(curves[tid], threshold) for tid in labels]
        k, n = accuracy(preds, list(labels.values()))
        acc = k / n
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    return best_threshold, best_acc


# --- GT-contact oracle sanity: cross-check against the Core archive's labels.csv --


def sanity_check_against_core_labels(trial_labels: dict[str, bool], core_root: Path | None = None) -> dict:
    """Cross-check each trial's HDF5-derived label (``any(target_contacting_zone)``)
    against the *independently-shipped* PhysionTest-Core archive's
    ``labels.csv`` (via ``gltfworld.physion.ingest.PhysionIndex``) -- the
    "GT-contact oracle sanity" this milestone's spec asks for: if these two,
    differently-packaged copies of the same underlying simulator label
    disagreed, that would bound the entire OCP exercise below 100% before
    any modeling even starts. Returns ``{"n_checked", "n_missing",
    "n_mismatch", "agreement_rate"}``; skips cleanly (empty dict) if the
    Core archive isn't extracted on this machine."""
    from gltfworld.physion.ingest import PhysionIndex

    try:
        core_index = PhysionIndex(core_root) if core_root else PhysionIndex()
    except FileNotFoundError:
        return {}

    core_by_id = {t.trial_id: t.label for t in core_index if t.scenario == "Collide"}
    n_missing = 0
    n_mismatch = 0
    n_checked = 0
    for trial_id, trial_label in trial_labels.items():
        core_id = f"{trial_id}_img"
        if core_id not in core_by_id:
            n_missing += 1
            continue
        n_checked += 1
        if bool(core_by_id[core_id]) != bool(trial_label):
            n_mismatch += 1

    return {
        "n_checked": n_checked,
        "n_missing": n_missing,
        "n_mismatch": n_mismatch,
        "agreement_rate": (1.0 - n_mismatch / n_checked) if n_checked else float("nan"),
    }


# --- batch conversion (stage 2 applied to all 150 test trials) ------------------


def convert_all_trials(hdf5_dir: Path, glb_dir: Path) -> list[str]:
    """Convert every ``*.hdf5`` trial in ``hdf5_dir`` to a GLB in
    ``glb_dir`` (skips a trial if its GLB already exists -- convert-all is
    idempotent/resumable). Returns the sorted list of trial ids converted
    or already present."""
    glb_dir.mkdir(parents=True, exist_ok=True)
    trial_ids = []
    for hdf5_path in sorted(glob.glob(str(hdf5_dir / "*.hdf5"))):
        trial_id = Path(hdf5_path).stem
        glb_path = glb_dir / f"{trial_id}.glb"
        if not glb_path.exists():
            trial = load_trial(hdf5_path)
            save_trial_gltf(trial, glb_path)
        trial_ids.append(trial_id)
    return sorted(trial_ids)


# --- CLI -----------------------------------------------------------------------------


def _load_dynamics_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    from safetensors.torch import load_file

    from gltfworld.train.train_dynamics import Config, make_model

    config_path = ckpt_path.parent / "config.json"
    cfg = Config.load(config_path)
    model = make_model(cfg).to(device)
    model.load_state_dict(load_file(ckpt_path, device=str(device)))
    model.eval()
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Physion Collide OCP evaluation (V8 stage 3).")
    parser.add_argument("--hdf5-dir", required=True, type=Path)
    parser.add_argument("--glb-dir", required=True, type=Path)
    parser.add_argument("--dynamics-ckpt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--k-seed", type=int, default=K_SEED_FRAMES)
    parser.add_argument("--n-calib", type=int, default=N_CALIBRATION)
    parser.add_argument("--max-trials", type=int, default=None, help="debug: cap the number of trials processed")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print(f"converting/loading trials from {args.hdf5_dir} -> {args.glb_dir}")
    trial_ids = convert_all_trials(args.hdf5_dir, args.glb_dir)
    if args.max_trials is not None:
        trial_ids = trial_ids[: args.max_trials]
    print(f"{len(trial_ids)} trials")

    model = _load_dynamics_model(args.dynamics_ckpt, device)
    ballistic = BallisticBaseline().to(device)

    labels: dict[str, bool] = {}
    oracle_curves: dict[str, np.ndarray] = {}
    dynamics_curves: dict[str, np.ndarray] = {}
    ballistic_curves: dict[str, np.ndarray] = {}

    for i, trial_id in enumerate(trial_ids):
        hdf5_path = args.hdf5_dir / f"{trial_id}.hdf5"
        glb_path = args.glb_dir / f"{trial_id}.glb"
        trial = load_trial(hdf5_path)
        labels[trial_id] = trial.label

        by_id = {obj.object_id: obj for obj in trial.objects}
        agent = by_id[trial.target_id]
        patient = by_id[trial.zone_id]
        oracle_curves[trial_id] = oracle_min_distance_curve(agent, patient)
        dynamics_curves[trial_id] = dynamics_track_contact_curve(trial, glb_path, model, args.k_seed)
        ballistic_curves[trial_id] = dynamics_track_contact_curve(trial, glb_path, ballistic, args.k_seed)

        if (i + 1) % 25 == 0 or i == len(trial_ids) - 1:
            print(f"  {i + 1}/{len(trial_ids)} trials processed")

    print("cross-checking HDF5-derived labels against Core archive's labels.csv...")
    core_sanity = sanity_check_against_core_labels(labels)
    if core_sanity:
        print(f"  Core labels.csv agreement: {core_sanity}")
    else:
        print("  Core archive not available on this machine -- skipped")

    calib_ids = calibration_split(trial_ids, n_calib=args.n_calib)
    eval_ids = [tid for tid in trial_ids if tid not in calib_ids]

    calib_labels = {tid: labels[tid] for tid in calib_ids}
    threshold, calib_acc = calibrate_threshold(oracle_curves, calib_labels)
    print(f"calibrated threshold: {threshold} (calibration-set oracle accuracy {calib_acc:.4f}, n={len(calib_ids)})")

    def _score(curves: dict[str, np.ndarray], ids: list[str]) -> dict:
        preds = [predict_label(curves[tid], threshold) for tid in ids]
        gt = [labels[tid] for tid in ids]
        k, n = accuracy(preds, gt)
        phat, lo, hi = wilson_interval(k, n)
        return {
            "k": k,
            "n": n,
            "accuracy": phat,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "n_predicted_contact": int(sum(preds)),
        }

    def _divergence_diagnostic(curves: dict[str, np.ndarray], ids: list[str]) -> dict:
        # median/IQR of the *final-frame* distance (not just the binary
        # contact call) -- diagnoses whether a track's rollout is at least
        # in the right ballpark even when the binary OCP accuracy collapses
        # to chance (see docs/RESULTS.md's V8 section).
        finals = np.array([curves[tid][-1] for tid in ids], dtype=np.float64)
        return {
            "median_final_distance_m": float(np.median(finals)),
            "p25_final_distance_m": float(np.percentile(finals, 25)),
            "p75_final_distance_m": float(np.percentile(finals, 75)),
        }

    results = {
        "threshold": threshold,
        "k_seed_frames": args.k_seed,
        "n_calibration": len(calib_ids),
        "calibration_trial_ids": sorted(calib_ids),
        "oracle": {
            "calibration_set": _score(oracle_curves, sorted(calib_ids)),
            "held_out": _score(oracle_curves, eval_ids),
            "full_150": _score(oracle_curves, trial_ids),
        },
        "our_dynamics": {
            "held_out": _score(dynamics_curves, eval_ids),
            "full_150": _score(dynamics_curves, trial_ids),
            "divergence_diagnostic": _divergence_diagnostic(dynamics_curves, trial_ids),
        },
        "ballistic": {
            "held_out": _score(ballistic_curves, eval_ids),
            "full_150": _score(ballistic_curves, trial_ids),
            "divergence_diagnostic": _divergence_diagnostic(ballistic_curves, trial_ids),
        },
        "label_balance": {"true": sum(labels.values()), "false": sum(1 for v in labels.values() if not v)},
        "core_labels_csv_sanity_check": core_sanity,
    }

    out_path = args.out / "metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
