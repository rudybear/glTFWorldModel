"""V7 closed-loop demo: perceive -> roll forward -> re-render, with real glTF
at every hop, plus a 3-arm attribution analysis that separates
perception-induced from dynamics-induced rollout error.

    uv run python -m gltfworld.eval.closed_loop \\
        --episodes data/perception-v1 \\
        --dyn-ckpt runs/dynamics-v1/best.safetensors \\
        --per-ckpt runs/perception-v3-cnn/best.safetensors \\
        --per-metrics runs/perception-v3-cnn/eval/metrics.json \\
        --out runs/closed-loop-v1 --n-episodes 20 --video 5

Three arms, per selected test episode (see DESIGN.md's V7 section for the
full design writeup)
-------------------------------------------------------------------------

- **Arm A (oracle)**: the exact ground-truth state at ``t=0`` (already
  carries the simulator's true velocity/angular-velocity, not a finite
  difference) rolled forward by the dynamics model. This is the dynamics
  model's own error ceiling -- no perception involved at all.
- **Arm B (oracle+noise)**: the same ground-truth *poses* at ``t=0,1``,
  independently perturbed per frame by Gaussian noise matched to the
  perception model's *measured* error distribution (position sigma from a
  ``perception_eval`` ``metrics.json``, rotation sigma per shape from the
  same file's ``matched_rotation_error_deg_by_shape`` -- see
  :func:`noise_params_from_metrics`), then finite-differenced into a
  velocity/angular-velocity the same way Arm C's real detections have to be.
  Object identity/count/physics-material fields (mass/friction/restitution,
  shape, size) stay exact GT -- this arm isolates *pose measurement noise
  alone*, with no detection/correspondence error mixed in.
- **Arm C (visual, the real closed loop)**: render GT frames 0 and 1 (the
  vendored ``EpisodeRenderer``), run the real perception model on each
  independently, Hungarian-match frame 0's existence-thresholded detections
  to frame 1's (by position + class + size proximity -- reusing
  ``gltfworld.models.matching.hungarian_match`` verbatim, see
  :func:`match_detections_across_frames`) to get a cross-frame
  correspondence, finite-difference velocity/angular-velocity from the
  matched pairs, and assemble the initial state from *only* the
  correspondences that survived (unmatched detections in either frame are
  dropped from the rollout and reported as detection-level stats instead --
  see :func:`build_arm_c_assembly`). Since ``PerceptionDETR`` never predicts
  mass/friction/restitution, every Arm C object gets the same fixed default
  physics values ``gltfworld.eval.perception_eval`` already uses for its own
  false-positive rendering fallback -- a real, honest, structural blind spot
  of the perception model, not a bug, and a real (documented, not hidden)
  confound in the B->C gap alongside pure detection/correspondence noise.

Every arm's rollout is reconstructed into a full ``Episode`` (reusing
``gltfworld.eval.rollout.tensors_to_episode``), saved via ``save_episode``,
and reloaded via ``load_episode`` *before* metric computation -- the same
"glTF at every hop" transport-exercising discipline every other milestone in
this project follows. Arm A/B preserve GT object identity/count 1:1 so their
rollout is compared directly against GT; Arm C's object set is whatever
survived detection + cross-frame correspondence, so its *trajectory* error is
reported only over objects additionally Hungarian-matched against the real
GT frame-0 state (for scoring only -- this second match is never fed back
into what Arm C's rollout actually saw). Unmatched/missing/spurious objects
are reported separately as detection-level precision/recall/F1, per the
milestone's own "be precise about what's averaged" requirement.

Attribution plot (``attribution.png``): median position error vs. horizon,
one curve per arm plus a ``BallisticBaseline`` reference (rolled out from
Arm A's own exact initial state -- reuses ``gltfworld.models.baselines
.BallisticBaseline`` and ``gltfworld.eval.rollout.rollout`` directly). The
A->B gap is the perception-noise cost; the B->C gap is the
detection/correspondence (+ missing-physics-params) cost; Arm A alone is the
dynamics model's own ceiling.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gltfworld.data.pack import SPLIT_NAMES, _pack_meta_path, split_id_for_seed
from gltfworld.eval.perception_eval import (
    _DEFAULT_FRICTION,
    _DEFAULT_MASS,
    _DEFAULT_METALLIC,
    _DEFAULT_RESTITUTION,
    _DEFAULT_ROUGHNESS,
    _FALSE_POSITIVE_COLOR,
    EXISTENCE_THRESHOLD,
    _canonicalize_size,
)
from gltfworld.eval.perception_eval import _load_model_from_ckpt as _load_perception_model
from gltfworld.eval.rollout import _load_model_from_ckpt as _load_dynamics_model
from gltfworld.eval.rollout import rollout, tensors_to_episode
from gltfworld.models.baselines import BallisticBaseline
from gltfworld.models.matching import MatchCostWeights, hungarian_match
from gltfworld.models.rotations import (
    axis_angle_to_quat,
    quat_conjugate,
    quat_geodesic_angle,
    quat_hemisphere,
    quat_multiply,
    quat_normalize,
    quat_to_axis_angle,
)
from gltfworld.scene.contract import CATEGORY_TO_CLASS_ID, CLASS_ID_TO_CATEGORY, SHAPE_ORDER, episode_to_tensors
from gltfworld.scene.convert import load_episode, save_episode
from gltfworld.scene.episode import Episode
from gltfworld.scene.scene import ObjectSpec, SceneState

DEFAULT_HORIZONS = (1, 5, 10, 30, 60, 99)
ARM_NAMES = ("A_oracle", "B_oracle_noise", "C_visual", "ballistic")


# --- noise calibration (Arm B) -------------------------------------------------


def _chi3_median() -> float:
    """Median of a standard (scale=1) 3-DOF chi distribution -- the
    distribution of ``||x||`` for ``x ~ N(0, I_3)``. Used to invert a
    *measured median error magnitude* (perception's own reported
    ``matched_position_error_m``/``matched_rotation_error_deg_by_shape``,
    both non-negative 3D-vector norms under an assumed isotropic Gaussian
    noise model) back into the per-axis Gaussian sigma that would produce
    that exact median -- an exact, closed-form inversion (via
    ``scipy.stats.chi``, already an ``ml``-extra dependency) rather than an
    approximate RMS/sqrt(3) rule of thumb.
    """
    from scipy.stats import chi

    return float(chi(df=3).ppf(0.5))


@dataclasses.dataclass
class NoiseParams:
    sigma_pos_m: float
    sigma_rot_rad_by_shape: dict  # SHAPE_ORDER name -> radians
    source: str


def noise_params_from_metrics(metrics_path: Path) -> NoiseParams:
    """Derive Arm B's noise sigmas from a real ``gltfworld.eval.perception_eval``
    ``metrics.json`` (position sigma from the overall ``matched_position_error_m``
    median; rotation sigma per shape from ``matched_rotation_error_deg_by_shape``
    -- sphere is always 0, matching this project's own "a sphere has no
    meaningful orientation" convention, since ``matched_rotation_error_deg_by_shape``
    never reports one for sphere either, see DESIGN.md's V6 section)."""
    c3 = _chi3_median()
    data = json.loads(Path(metrics_path).read_text())
    m = data["metrics"]["PerceptionDETR"]
    pos_median = m["matched_position_error_m"]["median"]
    if pos_median is None:
        raise ValueError(f"{metrics_path}: matched_position_error_m.median is null -- no matched pairs?")
    sigma_pos = pos_median / c3

    sigma_rot = {}
    for shape in SHAPE_ORDER:
        stat = m["matched_rotation_error_deg_by_shape"].get(shape)
        med = stat["median"] if stat else None
        if shape == "sphere" or med is None:
            sigma_rot[shape] = 0.0
        else:
            sigma_rot[shape] = math.radians(med) / c3
    return NoiseParams(sigma_pos_m=sigma_pos, sigma_rot_rad_by_shape=sigma_rot, source=str(metrics_path))


def noise_params_from_args(sigma_pos_m: float, sigma_rot_deg: float) -> NoiseParams:
    """Explicit-override construction (no ``metrics.json`` needed): the same
    ``sigma_rot_deg`` is applied to both box and cylinder (sphere fixed at 0,
    same rationale as :func:`noise_params_from_metrics`)."""
    sigma_rot_rad = {shape: math.radians(sigma_rot_deg) for shape in SHAPE_ORDER}
    sigma_rot_rad["sphere"] = 0.0
    return NoiseParams(sigma_pos_m=sigma_pos_m, sigma_rot_rad_by_shape=sigma_rot_rad, source="cli-args")


def resolve_noise_params(
    per_metrics: Path | None, noise_sigma_pos: float | None, noise_sigma_rot_deg: float | None
) -> NoiseParams:
    if per_metrics is not None:
        noise = noise_params_from_metrics(per_metrics)
        if noise_sigma_pos is not None:
            noise.sigma_pos_m = noise_sigma_pos
        if noise_sigma_rot_deg is not None:
            for shape in ("box", "cylinder"):
                noise.sigma_rot_rad_by_shape[shape] = math.radians(noise_sigma_rot_deg)
        return noise
    if noise_sigma_pos is None:
        raise ValueError("either --per-metrics or --noise-sigma-pos must be given")
    return noise_params_from_args(noise_sigma_pos, noise_sigma_rot_deg or 0.0)


# --- pose perturbation + finite-difference velocity (shared by Arm B and C) ---


def _generator_for(seed: int, episode_seed: int, tag: str) -> torch.Generator:
    """Deterministic per-(global seed, episode, purpose) RNG stream -- so a
    fixed ``--seed`` always reproduces the exact same Arm B noise draw for a
    given episode, independent of iteration order."""
    digest = hashlib.sha256(f"gltfworld-closed-loop-v1:{seed}:{episode_seed}:{tag}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little", signed=False) % (2**63))


def perturb_pose(
    pos: torch.Tensor, quat: torch.Tensor, sigma_pos: float, sigma_rot_rad: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """``pos`` (N, 3), ``quat`` (N, 4) xyzw, ``sigma_rot_rad`` (N,) per-object
    rotation sigma -> independently perturbed ``(pos, quat)``. Rotation noise
    is composed on the left (``dq * quat``), the same convention
    ``gltfworld.train.train_dynamics.add_input_noise`` uses for its own
    training-time input-noise injection."""
    pos = pos.to(torch.float32)
    quat = quat.to(torch.float32)
    noisy_pos = pos + torch.randn(pos.shape, generator=generator, dtype=torch.float32) * sigma_pos
    rotvec = torch.randn(pos.shape, generator=generator, dtype=torch.float32) * sigma_rot_rad[:, None]
    dq = axis_angle_to_quat(rotvec)
    noisy_quat = quat_hemisphere(quat_normalize(quat_multiply(dq, quat)))
    return noisy_pos, noisy_quat


def finite_diff_velocity(
    pos0: torch.Tensor, quat0: torch.Tensor, pos1: torch.Tensor, quat1: torch.Tensor, dt: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-frame finite difference -> ``(lin_vel, ang_vel)``, both world
    frame (the tensor contract's own convention). Angular velocity: the
    world-frame delta rotation ``dq`` such that ``quat1 = dq * quat0``
    (``dq = quat1 * quat0^-1``), converted to a rotation vector via the log
    map (:func:`gltfworld.models.rotations.quat_to_axis_angle`) and divided
    by ``dt`` -- the exact inverse of ``gltfworld.models.dynamics.integrate``'s
    own ``quat_new = dq * quat`` update, so a constant-angular-velocity
    two-frame pair recovers that velocity exactly."""
    lin_vel = (pos1 - pos0) / dt
    dq_world = quat_multiply(quat1, quat_conjugate(quat0))
    ang_vel = quat_to_axis_angle(dq_world) / dt
    return lin_vel, ang_vel


# --- Arm B: oracle + measured-noise, finite-diffed ----------------------------


def build_arm_b_initial_state(
    states_gt: torch.Tensor, dt: float, noise: NoiseParams, generator: torch.Generator
) -> torch.Tensor:
    """``states_gt`` (T>=2, N, 22) GT tensor-contract states -> Arm B's
    ``(N, 22)`` initial state: GT poses at frames 0/1 independently
    perturbed (per-shape rotation sigma), finite-diffed into velocity, with
    every other field (shape/size/mass/friction/restitution) held at exact
    GT -- see the module docstring's Arm B description."""
    pos0, quat0 = states_gt[0, :, 0:3], states_gt[0, :, 3:7]
    pos1, quat1 = states_gt[1, :, 0:3], states_gt[1, :, 3:7]
    shape_idx = states_gt[0, :, 13:16].argmax(dim=-1)
    sigma_rot = torch.tensor(
        [noise.sigma_rot_rad_by_shape[SHAPE_ORDER[int(i)]] for i in shape_idx], dtype=torch.float32
    )

    pos0_n, quat0_n = perturb_pose(pos0, quat0, noise.sigma_pos_m, sigma_rot, generator)
    pos1_n, quat1_n = perturb_pose(pos1, quat1, noise.sigma_pos_m, sigma_rot, generator)

    lin_vel, ang_vel = finite_diff_velocity(pos0_n, quat0_n, pos1_n, quat1_n, dt)
    static = states_gt[0, :, 13:22]
    return torch.cat([pos0_n, quat0_n, lin_vel, ang_vel, static], dim=-1)


# --- Arm C: real perception + cross-frame correspondence ----------------------


@torch.no_grad()
def run_perception(model: torch.nn.Module, rgb_u8: np.ndarray, device: torch.device) -> dict:
    """One rendered frame -> ``PerceptionDETR``'s raw per-query output dict
    (``N_MAX`` queries, batch dim squeezed back out, moved to CPU)."""
    rgb = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0)[None].to(device)
    pred = model(rgb)
    return {k: v[0].detach().cpu() for k, v in pred.items()}


def existent_indices(pred: dict, threshold: float = EXISTENCE_THRESHOLD) -> torch.Tensor:
    prob = torch.sigmoid(pred["existence_logit"])
    return torch.nonzero(prob >= threshold, as_tuple=True)[0]


def match_detections_across_frames(
    pred0: dict, idx0: torch.Tensor, pred1: dict, idx1: torch.Tensor, cost_weights: MatchCostWeights = MatchCostWeights()
) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian-match frame 0's existence-thresholded detections (``idx0``,
    indices into ``pred0``'s ``N_MAX`` query axis) to frame 1's (``idx1``) by
    position + class + size proximity -- reusing
    ``gltfworld.models.matching.hungarian_match`` verbatim, with frame 1's
    detections standing in as that generic function's "GT" side (a real GT
    row that "exists" per its mask; frame 1's detections are already the
    ``idx1``-filtered existent subset, so every one of them is eligible).
    Returns ``(matched_idx0, matched_idx1)``, both indices back into the
    original ``N_MAX`` query axes -- parallel arrays, one entry per
    surviving correspondence (unmatched detections in either frame are
    simply absent from the output, per Hungarian assignment on a
    rectangular cost matrix)."""
    if idx0.numel() == 0 or idx1.numel() == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    pos0 = pred0["position"][idx0][None]
    logits0 = pred0["class_logits"][idx0][None]
    size0 = pred0["size"][idx0][None]
    pos1 = pred1["position"][idx1][None]
    class1 = pred1["class_logits"][idx1].argmax(dim=-1)[None]
    size1 = pred1["size"][idx1][None]
    mask1 = torch.ones(1, idx1.numel(), dtype=torch.bool)

    matches = hungarian_match(pos0, logits0, size0, pos1, class1, size1, mask1, cost_weights)
    m0, m1 = matches[0]
    return idx0[m0].numpy(), idx1[m1].numpy()


def match_armc_to_gt(
    armc_pos: torch.Tensor,
    armc_class_logits: torch.Tensor,
    armc_size: torch.Tensor,
    gt_pos: torch.Tensor,
    gt_class: torch.Tensor,
    gt_size: torch.Tensor,
    cost_weights: MatchCostWeights = MatchCostWeights(),
) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian-match Arm C's assembled objects against the real GT
    frame-0 state, for *scoring only* -- this correspondence is never fed
    back into Arm C's actual rolled-out state, only used afterward to know
    which of Arm C's objects (if any) a given horizon's trajectory error
    should be measured against. Reuses ``hungarian_match`` a second time
    (the same generic query<->GT matching primitive
    ``gltfworld.eval.perception_eval.run_inference`` already uses for its
    own eval)."""
    if armc_pos.shape[0] == 0 or gt_pos.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    mask = torch.ones(1, gt_pos.shape[0], dtype=torch.bool)
    matches = hungarian_match(
        armc_pos[None], armc_class_logits[None], armc_size[None], gt_pos[None], gt_class[None], gt_size[None], mask,
        cost_weights,
    )
    return matches[0]


def build_arm_c_assembly(
    pred0: dict,
    pred1: dict,
    dt: float,
    gt_pos0: torch.Tensor,
    gt_class0: torch.Tensor,
    gt_size0: torch.Tensor,
    threshold: float = EXISTENCE_THRESHOLD,
) -> dict:
    """Assemble Arm C's initial ``(N_C, 22)`` state from two independent
    perception passes: existence-threshold each frame, Hungarian-match
    frame0<->frame1 for correspondence, finite-diff velocity from the
    matched pairs, fix mass/friction/restitution at
    ``gltfworld.eval.perception_eval``'s own false-positive-rendering
    defaults (``PerceptionDETR`` never predicts these -- see module
    docstring), and separately Hungarian-match the result against the real
    GT frame-0 state for later scoring (see :func:`match_armc_to_gt`).

    Returns a dict: ``initial_state (N_C, 22)``, ``shape_idx``/``class_idx``
    (N_C,), ``size`` (N_C, 3) (canonicalized), ``pos0``/``quat0`` (N_C, 3/4),
    ``corr0`` (N_C,) -- the original frame-0 query-slot index each row came
    from (for GT-assist color lookup when building glTF ``ObjectSpec``s),
    ``armc_to_gt`` (query_idx, gt_idx) from :func:`match_armc_to_gt`, and
    detection-level counts (``n_det0_exist``, ``n_det1_exist``,
    ``n_correspondence``).
    """
    idx0 = existent_indices(pred0, threshold)
    idx1 = existent_indices(pred1, threshold)
    corr0, corr1 = match_detections_across_frames(pred0, idx0, pred1, idx1)
    n_c = int(corr0.shape[0])

    empty_common = {
        "n_det0_exist": int(idx0.numel()),
        "n_det1_exist": int(idx1.numel()),
        "n_correspondence": n_c,
    }
    if n_c == 0:
        return {
            "initial_state": torch.zeros(0, 22, dtype=torch.float32),
            "shape_idx": torch.zeros(0, dtype=torch.int64),
            "class_idx": torch.zeros(0, dtype=torch.int64),
            "size": torch.zeros(0, 3, dtype=torch.float32),
            "pos0": torch.zeros(0, 3, dtype=torch.float32),
            "quat0": torch.zeros(0, 4, dtype=torch.float32),
            "corr0": np.array([], dtype=np.int64),
            "armc_to_gt": (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
            **empty_common,
        }

    corr0_t = torch.from_numpy(corr0).long()
    corr1_t = torch.from_numpy(corr1).long()
    pos0 = pred0["position"][corr0_t]
    quat0 = pred0["quat"][corr0_t]
    pos1 = pred1["position"][corr1_t]
    quat1 = pred1["quat"][corr1_t]
    size_raw = pred0["size"][corr0_t]
    shape_idx = pred0["shape_logits"][corr0_t].argmax(dim=-1)
    class_idx = pred0["class_logits"][corr0_t].argmax(dim=-1)
    class_logits0 = pred0["class_logits"][corr0_t]

    lin_vel, ang_vel = finite_diff_velocity(pos0, quat0, pos1, quat1, dt)

    size_canon = torch.stack(
        [torch.from_numpy(_canonicalize_size(int(s), sz.numpy())) for s, sz in zip(shape_idx.tolist(), size_raw)]
    )
    shape_onehot = F.one_hot(shape_idx, num_classes=len(SHAPE_ORDER)).to(torch.float32)

    log_mass = torch.full((n_c, 1), math.log(_DEFAULT_MASS), dtype=torch.float32)
    friction = torch.full((n_c, 1), _DEFAULT_FRICTION, dtype=torch.float32)
    restitution = torch.full((n_c, 1), _DEFAULT_RESTITUTION, dtype=torch.float32)

    initial_state = torch.cat(
        [pos0, quat0, lin_vel, ang_vel, shape_onehot, size_canon, log_mass, friction, restitution], dim=-1
    )

    armc_to_gt = match_armc_to_gt(pos0, class_logits0, size_canon, gt_pos0, gt_class0, gt_size0)

    return {
        "initial_state": initial_state,
        "shape_idx": shape_idx,
        "class_idx": class_idx,
        "size": size_canon,
        "pos0": pos0,
        "quat0": quat0,
        "corr0": corr0,
        "armc_to_gt": armc_to_gt,
        **empty_common,
    }


def build_arm_c_objects(assembly: dict, gt_objects_dynamic: list, next_object_id: int) -> list:
    """One ``ObjectSpec`` per Arm C object, for the synthetic template a
    glTF export needs (see :func:`build_synthetic_template`). Shape/size
    always come from the model's own prediction; color/category are an
    honest GT-assist for matched objects (copied from the corresponding real
    GT object, exactly the same convention
    ``gltfworld.eval.perception_eval.build_predicted_episode`` already
    established for its re-render check) or a fixed neutral-gray fallback
    for a genuine false positive (no GT match) -- rendering-only, never fed
    into the tensor-contract state or any metric."""
    query_idx, gt_idx = assembly["armc_to_gt"]
    query_to_gt = dict(zip(query_idx.tolist(), gt_idx.tolist()))
    n_c = assembly["n_correspondence"]

    objects = []
    for row in range(n_c):
        shape_idx = int(assembly["shape_idx"][row])
        gt_row = query_to_gt.get(row)
        if gt_row is not None and gt_row < len(gt_objects_dynamic):
            gt_obj = gt_objects_dynamic[gt_row]
            color, category = gt_obj.color, gt_obj.category
        else:
            color = _FALSE_POSITIVE_COLOR
            category = CLASS_ID_TO_CATEGORY.get(int(assembly["class_idx"][row]), "ball")
        objects.append(
            ObjectSpec(
                object_id=next_object_id + row,
                shape=SHAPE_ORDER[shape_idx],
                size=assembly["size"][row].numpy(),
                color=color,
                roughness=_DEFAULT_ROUGHNESS,
                metallic=_DEFAULT_METALLIC,
                mass=_DEFAULT_MASS,
                friction=_DEFAULT_FRICTION,
                restitution=_DEFAULT_RESTITUTION,
                is_static=False,
                category=category,
            )
        )
    return objects


def build_synthetic_template(real_template: Episode, arm_objects: list, arm_frame0_poses: np.ndarray) -> Episode:
    """A ``T=1`` synthetic ``Episode`` combining ``arm_objects`` (Arm C's own
    dynamic objects, count may differ from GT) with ``real_template``'s real
    static (ground) objects -- everything ``gltfworld.eval.rollout
    .tensors_to_episode`` needs as a source of "metadata the tensor contract
    doesn't carry" (color/material/camera/lights/gravity/dt/seed), without
    requiring its dynamic-object count to match the real episode's (which
    ``tensors_to_episode`` would otherwise reject -- Arm C's detected/
    corresponded object count is generally *not* the real GT count)."""
    real_scene = real_template.scene
    static_objects = [o for o in real_scene.objects if o.is_static]
    static_indices = [i for i, o in enumerate(real_scene.objects) if o.is_static]
    static_poses0 = (
        np.stack([real_template.series.poses[0, i, :] for i in static_indices], axis=0)
        if static_indices
        else np.zeros((0, 7), dtype=np.float32)
    )
    all_objects = list(arm_objects) + static_objects
    all_poses0 = (
        np.concatenate([arm_frame0_poses, static_poses0], axis=0)
        if all_objects
        else np.zeros((0, 7), dtype=np.float32)
    )
    scene = SceneState(
        objects=all_objects,
        camera=real_scene.camera,
        lights=real_scene.lights,
        gravity=real_scene.gravity,
        dt=real_scene.dt,
        seed=real_scene.seed,
        scene_version=real_scene.scene_version,
    )
    return Episode.single_frame(scene, all_poses0)


# --- dataset resolution --------------------------------------------------------


def resolve_episodes_dir(path: Path) -> Path:
    """``--episodes`` accepts a raw directory of ``ep_*.glb`` files, a
    dataset root (``<root>/episodes`` + ``<root>/packed``, e.g.
    ``data/perception-v1``), or a packed dataset dir/file (``pack_meta.json``'s
    ``source_dir`` gives the real episodes directory, same convention
    ``gltfworld.eval.rollout``'s ``--emit-gltf`` uses)."""
    path = Path(path)
    if path.is_file() and path.suffix == ".safetensors":
        meta = json.loads(_pack_meta_path(path).read_text())
        return Path(meta["source_dir"])
    if path.is_dir():
        if any(path.glob("ep_*.glb")):
            return path
        if (path / "episodes").is_dir():
            return path / "episodes"
        candidates = sorted(path.glob("*.safetensors"))
        if len(candidates) == 1:
            meta = json.loads(_pack_meta_path(candidates[0]).read_text())
            return Path(meta["source_dir"])
    raise ValueError(f"could not resolve an episodes directory (ep_*.glb) from {path}")


def select_episodes(episodes_dir: Path, split: str, n_episodes: int) -> list[Path]:
    """Sorted ``ep_*.glb`` paths, filtered to ``split`` (the same
    deterministic ``split_id_for_seed`` bucketing every packed dataset in
    this project uses) -- ``split="all"`` disables filtering (handy for
    small synthetic test fixtures that don't target a particular split)."""
    paths = sorted(episodes_dir.glob("ep_*.glb"))
    if split == "all":
        selected = paths
    else:
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be 'all' or one of {SPLIT_NAMES}, got {split!r}")
        target = SPLIT_NAMES.index(split)
        selected = [p for p in paths if split_id_for_seed(load_episode(p).scene.seed) == target]
    return selected[:n_episodes]


# --- per-episode error accumulation --------------------------------------------


def _pos_err_curve(pred_pos: np.ndarray, gt_pos: np.ndarray) -> np.ndarray:
    """``pred_pos``/``gt_pos`` (T, n, 3) -> per-horizon position error
    ``(T-1, n)`` (row ``h-1`` is horizon ``h``, horizon 0 skipped since it's
    the (identical, by construction) initial condition)."""
    return np.linalg.norm(pred_pos[1:] - gt_pos[1:], axis=-1)


def _rot_err_at(pred_quat: np.ndarray, gt_quat: np.ndarray, horizon: int) -> np.ndarray:
    return quat_geodesic_angle(torch.from_numpy(pred_quat[horizon]), torch.from_numpy(gt_quat[horizon])).numpy()


class ArmAccumulator:
    """Collects raw per-object position errors at every horizon
    ``1..T_max-1`` (for the full attribution curve) plus rotation error at
    the discrete ``DEFAULT_HORIZONS``-style set, across every episode --
    median/IQR are computed once at the very end, over the whole
    (episode, object) population, matching ``gltfworld.eval.rollout
    .horizon_metrics``'s own "median + IQR over every unmasked
    (episode, object) pair" convention."""

    def __init__(self) -> None:
        self.pos_curve: list[list[float]] = []  # index h-1 -> flat list of errors
        self.rot_by_horizon: dict[int, list[float]] = {}

    def add_episode(self, pred_pos: np.ndarray, gt_pos: np.ndarray, pred_quat: np.ndarray, gt_quat: np.ndarray, horizons: list[int]) -> None:
        curve = _pos_err_curve(pred_pos, gt_pos)  # (T-1, n)
        while len(self.pos_curve) < curve.shape[0]:
            self.pos_curve.append([])
        for h_idx in range(curve.shape[0]):
            self.pos_curve[h_idx].extend(curve[h_idx].tolist())
        t = pred_pos.shape[0]
        for h in horizons:
            if h >= t:
                continue
            self.rot_by_horizon.setdefault(h, []).extend(_rot_err_at(pred_quat, gt_quat, h).tolist())

    def full_curve_median(self) -> np.ndarray:
        return np.array(
            [float(np.median(vals)) if vals else float("nan") for vals in self.pos_curve], dtype=np.float64
        )

    def horizon_stats(self, horizons: list[int]) -> dict:
        out = {}
        for h in horizons:
            h_idx = h - 1
            pos_vals = self.pos_curve[h_idx] if 0 <= h_idx < len(self.pos_curve) else []
            rot_vals = self.rot_by_horizon.get(h, [])
            out[str(h)] = {
                "position_error_m": _stat_dict(pos_vals),
                "rotation_error_rad": _stat_dict(rot_vals),
            }
        return out


def _stat_dict(values: list[float]) -> dict:
    if not values:
        return {"median": None, "p25": None, "p75": None, "n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "n": int(arr.size),
    }


# --- per-episode processing -----------------------------------------------------


@dataclasses.dataclass
class EpisodeResult:
    orig_name: str
    episode_gt: Episode
    episode_A: Episode
    episode_B: Episode
    episode_C: Episode
    states_gt: np.ndarray  # (T, N, 22) reloaded
    states_A: np.ndarray
    states_B: np.ndarray
    states_C: np.ndarray  # (T, N_C, 22) reloaded
    states_ballistic: np.ndarray
    armc_to_gt: tuple  # (query_idx, gt_idx) into (N_C, N)
    n_det0_exist: int
    n_det1_exist: int
    n_correspondence: int
    n_gt_real: int


def _roundtrip_episode(ep: Episode, out_path: Path) -> tuple[Episode, np.ndarray]:
    """Save -> load -> convert back to the tensor contract, asserting the
    round trip is exact (<=1e-6) -- the "transport exercised at every hop"
    discipline this milestone requires. Returns ``(reloaded_episode,
    reloaded_states)`` -- callers use the *reloaded* tensor for downstream
    metrics, not the in-memory one, per the milestone's own instructions."""
    original_states = episode_to_tensors(ep)["states"]
    save_episode(ep, out_path)
    reloaded = load_episode(out_path)
    reloaded_states = episode_to_tensors(reloaded)["states"]
    if original_states.size and reloaded_states.size:
        err = float(np.max(np.abs(original_states - reloaded_states)))
        assert err <= 1e-6, f"{out_path}: glTF round trip error {err:.3e} exceeds 1e-6"
    return reloaded, reloaded_states


def process_episode(
    ep_path: Path,
    dyn_model: torch.nn.Module,
    per_model: torch.nn.Module | None,
    device: torch.device,
    noise: NoiseParams,
    seed: int,
    out_dir: Path,
    existence_threshold: float = EXISTENCE_THRESHOLD,
    renderer=None,
) -> EpisodeResult:
    """Run all 3 arms + the ballistic reference for one real episode,
    emitting each arm's rollout as a real, round-trip-verified GLB under
    ``out_dir/{arm}/<name>.glb``. ``per_model``/``renderer`` are only needed
    for Arm C -- pass ``None`` to skip it (e.g. a CPU-only smoke where no
    renderer/perception ckpt is available); ``episode_C``/``states_C`` are
    then ``None``/an empty array and detection counts are all 0.
    """
    ep = load_episode(ep_path)
    name = ep_path.stem
    tensors = episode_to_tensors(ep)
    states_gt_np = tensors["states"]
    class_ids = tensors["class_ids"]
    globals_np = tensors["globals"]
    t = states_gt_np.shape[0]
    n_gt = states_gt_np.shape[1]

    # All assembly logic (noise injection, finite-diff, matching) happens on
    # CPU regardless of `device` -- only the tensors actually fed to a model
    # (dyn_model/per_model, which may live on a CUDA device) are moved there,
    # immediately before each `rollout()`/`run_perception()` call, and moved
    # straight back to CPU right after (every downstream consumer here is
    # numpy/CPU-based: glTF export, matching, metric accumulation).
    states_gt = torch.from_numpy(states_gt_np)
    globals_t = torch.from_numpy(globals_np)
    mask_gt = torch.ones(n_gt, dtype=torch.bool)
    dt = float(ep.series.times[1] - ep.series.times[0]) if t > 1 else float(ep.scene.dt)

    globals_dev = globals_t.to(device)
    mask_gt_dev = mask_gt.to(device)

    for sub in ("gt", "armA", "armB", "armC"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # --- Arm A: oracle ----------------------------------------------------
    rollout_A = rollout(dyn_model, states_gt[0].to(device), mask_gt_dev, globals_dev, t).detach().cpu()

    # --- Arm B: oracle + measured noise -------------------------------------
    gen_b = _generator_for(seed, ep.scene.seed, "armB")
    if t > 1:
        init_b = build_arm_b_initial_state(states_gt, dt, noise, gen_b)
    else:
        init_b = states_gt[0].clone()
    rollout_B = rollout(dyn_model, init_b.to(device), mask_gt_dev, globals_dev, t).detach().cpu()

    # --- ballistic reference (from Arm A's exact initial state) -------------
    rollout_ballistic = (
        rollout(BallisticBaseline().to(device), states_gt[0].to(device), mask_gt_dev, globals_dev, t).detach().cpu()
    )

    # --- Arm C: real visual closed loop --------------------------------------
    n_c = 0
    n_det0 = n_det1 = 0
    armc_to_gt: tuple = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    episode_C = None
    states_C_np = np.zeros((t, 0, 22), dtype=np.float32)

    if per_model is not None and renderer is not None and t > 1:
        renderer.load(ep)
        renderer.set_frame(0)
        rgb0 = renderer.render().rgb
        renderer.set_frame(1)
        rgb1 = renderer.render().rgb

        pred0 = run_perception(per_model, rgb0, device)
        pred1 = run_perception(per_model, rgb1, device)

        gt_pos0 = states_gt[0, :, 0:3]
        gt_class0 = torch.from_numpy(class_ids)
        gt_size0 = states_gt[0, :, 16:19]

        assembly = build_arm_c_assembly(pred0, pred1, dt, gt_pos0, gt_class0, gt_size0, existence_threshold)
        n_c = assembly["n_correspondence"]
        n_det0, n_det1 = assembly["n_det0_exist"], assembly["n_det1_exist"]
        armc_to_gt = assembly["armc_to_gt"]

        gt_objects_dynamic = [o for o in ep.scene.objects if not o.is_static]
        next_object_id = max((o.object_id for o in ep.scene.objects), default=-1) + 1
        armc_objects = build_arm_c_objects(assembly, gt_objects_dynamic, next_object_id)

        armc_frame0_poses = np.zeros((n_c, 7), dtype=np.float32)
        if n_c:
            armc_frame0_poses[:, 0:3] = assembly["pos0"].numpy()
            armc_frame0_poses[:, 3:7] = assembly["quat0"].numpy()

        rollout_C = (
            rollout(
                dyn_model, assembly["initial_state"].to(device), torch.ones(n_c, dtype=torch.bool, device=device),
                globals_dev, t,
            )
            .detach()
            .cpu()
        )

        synthetic_template = build_synthetic_template(ep, armc_objects, armc_frame0_poses)
        armc_class_ids = np.array(
            [CATEGORY_TO_CLASS_ID[o.category] for o in armc_objects], dtype=np.int64
        )
        episode_C_prewrite = tensors_to_episode(
            rollout_C.numpy(),
            np.ones(n_c, dtype=bool),
            armc_class_ids,
            globals_np,
            synthetic_template,
        )
        episode_C, states_C_np = _roundtrip_episode(episode_C_prewrite, out_dir / "armC" / f"{name}.glb")

    # --- glTF at every hop: GT / Arm A / Arm B --------------------------------
    episode_gt_prewrite = tensors_to_episode(states_gt_np, np.ones(n_gt, dtype=bool), class_ids, globals_np, ep)
    episode_gt, states_gt_reloaded = _roundtrip_episode(episode_gt_prewrite, out_dir / "gt" / f"{name}.glb")

    episode_A_prewrite = tensors_to_episode(
        rollout_A.detach().numpy(), np.ones(n_gt, dtype=bool), class_ids, globals_np, ep
    )
    episode_A, states_A_np = _roundtrip_episode(episode_A_prewrite, out_dir / "armA" / f"{name}.glb")

    episode_B_prewrite = tensors_to_episode(
        rollout_B.detach().numpy(), np.ones(n_gt, dtype=bool), class_ids, globals_np, ep
    )
    episode_B, states_B_np = _roundtrip_episode(episode_B_prewrite, out_dir / "armB" / f"{name}.glb")

    return EpisodeResult(
        orig_name=name,
        episode_gt=episode_gt,
        episode_A=episode_A,
        episode_B=episode_B,
        episode_C=episode_C,
        states_gt=states_gt_reloaded,
        states_A=states_A_np,
        states_B=states_B_np,
        states_C=states_C_np,
        states_ballistic=rollout_ballistic.detach().numpy(),
        armc_to_gt=armc_to_gt,
        n_det0_exist=n_det0,
        n_det1_exist=n_det1,
        n_correspondence=n_c,
        n_gt_real=n_gt,
    )


# --- aggregation + attribution plot --------------------------------------------


def aggregate_results(results: list[EpisodeResult], horizons: list[int]) -> dict:
    accum = {name: ArmAccumulator() for name in ARM_NAMES}
    det_tp = det_fp = det_fn = 0
    n_zero_correspondence = 0

    for r in results:
        accum["A_oracle"].add_episode(
            r.states_A[..., 0:3], r.states_gt[..., 0:3], r.states_A[..., 3:7], r.states_gt[..., 3:7], horizons
        )
        accum["B_oracle_noise"].add_episode(
            r.states_B[..., 0:3], r.states_gt[..., 0:3], r.states_B[..., 3:7], r.states_gt[..., 3:7], horizons
        )
        accum["ballistic"].add_episode(
            r.states_ballistic[..., 0:3], r.states_gt[..., 0:3], r.states_ballistic[..., 3:7], r.states_gt[..., 3:7],
            horizons,
        )

        query_idx, gt_idx = r.armc_to_gt
        n_matched = len(query_idx)
        det_tp += n_matched
        det_fp += r.n_correspondence - n_matched
        det_fn += r.n_gt_real - n_matched
        if r.n_correspondence == 0:
            n_zero_correspondence += 1
        if n_matched:
            pred_pos = r.states_C[:, query_idx, 0:3]
            gt_pos = r.states_gt[:, gt_idx, 0:3]
            pred_quat = r.states_C[:, query_idx, 3:7]
            gt_quat = r.states_gt[:, gt_idx, 3:7]
            accum["C_visual"].add_episode(pred_pos, gt_pos, pred_quat, gt_quat, horizons)

    precision = det_tp / (det_tp + det_fp) if (det_tp + det_fp) > 0 else 1.0
    recall = det_tp / (det_tp + det_fn) if (det_tp + det_fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    arms_out = {name: accum[name].horizon_stats(horizons) for name in ARM_NAMES}
    curves = {name: accum[name].full_curve_median().tolist() for name in ARM_NAMES}

    # ordering check at the longest common horizon: A <= B <= C median error
    # (allowing statistical noise -- reported, never silently gated/hidden).
    ordering = {}
    for h in horizons:
        vals = {}
        for name in ("A_oracle", "B_oracle_noise", "C_visual"):
            entry = arms_out[name].get(str(h))
            vals[name] = entry["position_error_m"]["median"] if entry else None
        ordering[str(h)] = {
            **vals,
            "ordering_ok": (
                vals["A_oracle"] is not None
                and vals["B_oracle_noise"] is not None
                and vals["C_visual"] is not None
                and vals["A_oracle"] <= vals["B_oracle_noise"] <= vals["C_visual"]
            ),
        }

    return {
        "n_episodes": len(results),
        "horizons": horizons,
        "arms": arms_out,
        "full_curves": curves,
        "arm_c_detection": {
            "tp": det_tp,
            "fp": det_fp,
            "fn": det_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_zero_correspondence_episodes": n_zero_correspondence,
        },
        "ordering_check": ordering,
    }


def plot_attribution(curves: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = {
        "A_oracle": "A: oracle (dynamics ceiling)",
        "B_oracle_noise": "B: oracle + perception noise",
        "C_visual": "C: visual closed loop",
        "ballistic": "ballistic reference",
    }
    for name, curve in curves.items():
        arr = np.asarray(curve, dtype=np.float64)
        if arr.size == 0:
            continue
        ax.plot(np.arange(1, len(arr) + 1), arr, label=labels.get(name, name))
    ax.set_yscale("log")
    ax.set_xlabel("horizon (frames)")
    ax.set_ylabel("median position error (m)")
    ax.set_title("Closed-loop attribution: A->B = perception-noise cost, B->C = detection cost")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- video (gpu) ----------------------------------------------------------------


def _render_all_frames(renderer, ep: Episode) -> list:
    renderer.load(ep)
    frames = []
    for frame_idx in range(ep.series.num_frames):
        renderer.set_frame(frame_idx)
        frames.append(renderer.render().rgb)
    return frames


def render_videos(results: list[EpisodeResult], out_dir: Path, renderer) -> None:
    import imageio

    video_dir = out_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        if r.episode_C is None:
            continue
        gt_frames = _render_all_frames(renderer, r.episode_gt)
        c_frames = _render_all_frames(renderer, r.episode_C)
        a_frames = _render_all_frames(renderer, r.episode_A)

        two_panel = [np.concatenate([g, c], axis=1) for g, c in zip(gt_frames, c_frames)]
        imageio.mimwrite(video_dir / f"{r.orig_name}.mp4", two_panel, fps=30, macro_block_size=None)

        three_panel = [np.concatenate([g, a, c], axis=1) for g, a, c in zip(gt_frames, a_frames, c_frames)]
        imageio.mimwrite(video_dir / f"{r.orig_name}_3panel.mp4", three_panel, fps=30, macro_block_size=None)
    print(f"wrote {sum(1 for r in results if r.episode_C is not None)} video pairs to {video_dir}")


# --- CLI -------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V7 closed-loop demo + 3-arm attribution.")
    parser.add_argument("--episodes", required=True, type=Path, help="glb dir, dataset root, or packed dataset")
    parser.add_argument("--dyn-ckpt", required=True, type=Path)
    parser.add_argument("--per-ckpt", required=True, type=Path)
    parser.add_argument("--per-metrics", type=Path, default=None, help="perception_eval metrics.json for noise calibration")
    parser.add_argument("--noise-sigma-pos", type=float, default=None, help="override/fallback position noise sigma (m)")
    parser.add_argument("--noise-sigma-rot-deg", type=float, default=None, help="override/fallback rotation noise sigma (deg, box+cylinder)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--split", default="test")
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    parser.add_argument("--existence-threshold", type=float, default=EXISTENCE_THRESHOLD)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", type=int, default=0, help="render N GT|armC (+3-panel) mp4s (needs GPU)")
    parser.add_argument("--no-perception", action="store_true", help="skip Arm C entirely (CPU-only smoke)")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise = resolve_noise_params(args.per_metrics, args.noise_sigma_pos, args.noise_sigma_rot_deg)

    episodes_dir = resolve_episodes_dir(args.episodes)
    ep_paths = select_episodes(episodes_dir, args.split, args.n_episodes)
    if not ep_paths:
        raise ValueError(f"no episodes selected from {episodes_dir} (split={args.split!r})")

    dyn_model, _dyn_type = _load_dynamics_model(args.dyn_ckpt, device)
    per_model = None
    renderer = None
    if not args.no_perception:
        per_model = _load_perception_model(args.per_ckpt, device)
        from gltfworld.render.renderer import EpisodeRenderer

        renderer = EpisodeRenderer(width=256, height=256)

    try:
        results = []
        for ep_path in ep_paths:
            r = process_episode(
                ep_path, dyn_model, per_model, device, noise, args.seed, args.out,
                existence_threshold=args.existence_threshold, renderer=renderer,
            )
            results.append(r)
            print(f"{ep_path.name}: n_gt={r.n_gt_real} n_det0={r.n_det0_exist} n_det1={r.n_det1_exist} n_corr={r.n_correspondence}")

        agg = aggregate_results(results, args.horizons)
        agg["split"] = args.split
        agg["noise"] = dataclasses.asdict(noise)

        metrics_path = args.out / "metrics.json"
        metrics_path.write_text(json.dumps(agg, indent=2))
        plot_attribution(agg["full_curves"], args.out / "attribution.png")
        print(f"wrote {metrics_path}, {args.out / 'attribution.png'}")

        if args.video > 0:
            if renderer is None:
                raise ValueError("--video needs Arm C's renderer; do not pass --no-perception")
            render_videos(results[: args.video], args.out, renderer)
    finally:
        if renderer is not None:
            renderer.delete()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
