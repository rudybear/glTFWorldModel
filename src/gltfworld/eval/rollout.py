"""Autoregressive rollout + divergence eval for the dynamics models.

``rollout`` is the reusable core (any ``forward(states, mask, globals) ->
next_states`` model -- ``InteractionTransformer``, ``BallisticBaseline``,
``NoInteractionMLP`` all qualify). The CLI (``__main__``) evaluates a
checkpoint against the ``BallisticBaseline`` and (optionally) a
``NoInteractionMLP`` checkpoint on a packed dataset's test split: per-horizon
position/rotation/velocity error (median + IQR over objects x episodes),
``metrics.json``/``metrics.md``, a log-y divergence-curve PNG, and (per
DESIGN.md's "glTF at every hop" principle) predicted/ground-truth rollouts
re-exported as real, loadable ``.glb`` episodes.

    uv run python -m gltfworld.eval.rollout \\
        --ckpt runs/dynamics-v1/best.safetensors \\
        --data data/dynamics-v1/packed --split test \\
        --out runs/dynamics-v1/eval \\
        --mlp-ckpt runs/dynamics-mlp/best.safetensors \\
        --emit-gltf 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gltfworld.data.dataset import DynamicsDataset, _load_packed, _split_indices
from gltfworld.data.pack import _pack_meta_path
from gltfworld.models.baselines import BallisticBaseline
from gltfworld.models.rotations import quat_geodesic_angle
from gltfworld.scene.contract import tensors_to_state
from gltfworld.scene.convert import load_episode, save_episode
from gltfworld.scene.episode import Episode, StateSeries
from gltfworld.scene.scene import SceneState

DEFAULT_HORIZONS = (1, 5, 10, 30, 99)


# --- rollout ---------------------------------------------------------------------


@torch.no_grad()
def rollout(
    model: torch.nn.Module,
    initial_state: torch.Tensor,
    mask: torch.Tensor,
    globals_: torch.Tensor,
    T: int,
) -> torch.Tensor:
    """Autoregressive rollout: ``initial_state`` is the ``t=0`` frame.

    Accepts either a single episode (``initial_state (N, D)``, ``mask
    (N,)``, ``globals_ (G,)``) -> returns ``(T, N, D)``, or a batch
    (``initial_state (B, N, D)``, ...) -> returns ``(B, T, N, D)``. Index 0
    of the output is always ``initial_state`` itself (unmodified); indices
    ``1..T-1`` are the model's own successive predictions (never re-fed
    ground truth -- true autoregressive rollout).
    """
    single = initial_state.dim() == 2
    if single:
        initial_state = initial_state.unsqueeze(0)
        mask = mask.unsqueeze(0)
        globals_ = globals_.unsqueeze(0)

    model.eval()
    states = [initial_state]
    cur = initial_state
    for _ in range(T - 1):
        cur = model(cur, mask, globals_)
        states.append(cur)
    out = torch.stack(states, dim=1)  # (B, T, N, D)

    if single:
        out = out.squeeze(0)
    return out


# --- metrics -----------------------------------------------------------------------


def _median_iqr(x: torch.Tensor) -> dict:
    if x.numel() == 0:
        return {"median": None, "p25": None, "p75": None, "n": 0}
    arr = x.detach().cpu().numpy().astype(np.float64)
    return {
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "n": int(arr.size),
    }


def horizon_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, horizon: int) -> dict:
    """``pred``/``gt`` ``(E, T, N, D)``, ``mask (E, N)`` -> position/rotation/
    velocity error distribution (median + IQR) at frame index ``horizon``,
    over every ``(episode, object)`` pair where ``mask`` is True."""
    pred_h = pred[:, horizon]
    gt_h = gt[:, horizon]
    m = mask

    pos_err = torch.linalg.norm(pred_h[..., 0:3] - gt_h[..., 0:3], dim=-1)[m]
    vel_err = torch.linalg.norm(pred_h[..., 7:10] - gt_h[..., 7:10], dim=-1)[m]
    rot_err = quat_geodesic_angle(pred_h[..., 3:7], gt_h[..., 3:7])[m]

    return {
        "position_error_m": _median_iqr(pos_err),
        "rotation_error_rad": _median_iqr(rot_err),
        "velocity_error_mps": _median_iqr(vel_err),
    }


def divergence_curve(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Median position error at every horizon ``1..T-1``, as a ``(T-1,)``
    numpy array -- the full curve behind the divergence-curve PNG (the
    designated ``DEFAULT_HORIZONS`` in ``metrics.json``/``.md`` are just a
    handful of samples off this same curve)."""
    t = pred.shape[1]
    out = np.zeros(t - 1, dtype=np.float64)
    for h in range(1, t):
        pos_err = torch.linalg.norm(pred[:, h, ..., 0:3] - gt[:, h, ..., 0:3], dim=-1)[mask]
        out[h - 1] = float(torch.median(pos_err).item()) if pos_err.numel() else float("nan")
    return out


# --- glTF-at-every-hop: rebuild a full Episode from rolled-out tensors ------------


def tensors_to_episode(
    states: np.ndarray,
    mask: np.ndarray,
    class_ids: np.ndarray,
    globals_: np.ndarray,
    template_episode: Episode,
) -> Episode:
    """Rebuild a full, loadable ``Episode`` (same scene -- ground included --
    same object identities) from a ``(T, N_max, D)`` tensor-contract rollout
    (real or predicted) plus the *original* per-episode ``Episode`` this
    packed row came from (source of the ground/static objects, camera
    extras, lights, and the real per-episode dynamic-object count -- packed
    tensors are padded to ``N_max``, but a real ``Episode`` only ever
    carries its own actual object count).

    ``pose_variance`` is always omitted (no model here predicts it).
    """
    template_scene = template_episode.scene
    n_real = int(mask.sum())
    states_real = states[:, :n_real, :]  # padding, if any, is always trailing (see gltfworld.data.pack)
    class_ids_real = class_ids[:n_real]
    mask_real = np.ones(n_real, dtype=bool)

    result = tensors_to_state(states_real, mask_real, class_ids_real, globals_, template_scene)

    t = states_real.shape[0]
    dynamic_indices = [i for i, obj in enumerate(template_scene.objects) if not obj.is_static]
    static_indices = [i for i, obj in enumerate(template_scene.objects) if obj.is_static]
    n_total = len(template_scene.objects)
    if len(dynamic_indices) != n_real:
        raise ValueError(
            f"template episode has {len(dynamic_indices)} dynamic objects, mask marks {n_real} real rows"
        )

    poses_full = np.zeros((t, n_total, 7), dtype=np.float32)
    lin_vel_full = np.zeros((t, n_total, 3), dtype=np.float32)
    ang_vel_full = np.zeros((t, n_total, 3), dtype=np.float32)
    objects_full = list(template_scene.objects)

    for row, obj_index in enumerate(dynamic_indices):
        poses_full[:, obj_index, :] = result["poses"][:, row, :]
        lin_vel_full[:, obj_index, :] = result["lin_vel"][:, row, :]
        ang_vel_full[:, obj_index, :] = result["ang_vel"][:, row, :]
        objects_full[obj_index] = result["objects"][row]

    for obj_index in static_indices:
        # static objects don't evolve; hold their template frame-0 pose for
        # every frame (the tensor contract never carries static poses at all).
        poses_full[:, obj_index, :] = template_episode.series.poses[0, obj_index, :]

    times = np.arange(t, dtype=np.float32) * result["dt"]
    new_scene = SceneState(
        objects=objects_full,
        camera=result["camera"],
        lights=template_scene.lights,
        gravity=result["gravity"],
        dt=result["dt"],
        seed=template_scene.seed,
        scene_version=template_scene.scene_version,
    )
    series = StateSeries(times=times, poses=poses_full, lin_vel=lin_vel_full, ang_vel=ang_vel_full)
    return Episode(scene=new_scene, series=series)


# --- CLI -----------------------------------------------------------------------------


def _load_class_ids(pack_file: Path, split: str, n: int | None = None) -> torch.Tensor:
    """``class_ids (E_split, N_max)``, in the exact same split-filtered row
    order as ``DynamicsDataset(pack_file, split=split, mode=...)``'s own
    ``states``/``mask``/``globals`` -- ``DynamicsDataset`` itself doesn't
    keep ``class_ids`` (only ``PerceptionDataset`` does), so this reads it
    directly off the packed file the same way ``DynamicsDataset`` reads
    everything else."""
    tensors = _load_packed(pack_file)
    indices = _split_indices(tensors["split_id"], split)
    class_ids = tensors["class_ids"][indices]
    return class_ids[:n] if n is not None else class_ids


def _resolve_pack_file(data_path: Path) -> Path:
    if data_path.is_file():
        return data_path
    candidates = sorted(data_path.glob("*.safetensors"))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one .safetensors file in {data_path}, found {len(candidates)}")
    return candidates[0]


def _load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    from safetensors.torch import load_file

    from gltfworld.train.train_dynamics import Config, make_model

    config_path = ckpt_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found -- rollout eval needs the training run's config.json "
            f"(next to the checkpoint) to know the model architecture/type"
        )
    cfg = Config.load(config_path)
    model = make_model(cfg).to(device)
    model.load_state_dict(load_file(ckpt_path, device=str(device)))
    model.eval()
    return model, cfg.model


def _format_markdown_table(all_metrics: dict, horizons: list[int]) -> str:
    lines = []
    for metric_key, title in [
        ("position_error_m", "Position error (m)"),
        ("rotation_error_rad", "Rotation geodesic error (rad)"),
        ("velocity_error_mps", "Velocity error (m/s)"),
    ]:
        lines.append(f"## {title}\n")
        header = "| model | " + " | ".join(f"h={h}" for h in horizons) + " |"
        sep = "| --- | " + " | ".join("---" for _ in horizons) + " |"
        lines.append(header)
        lines.append(sep)
        for model_name, per_horizon in all_metrics.items():
            cells = []
            for h in horizons:
                entry = per_horizon.get(str(h)) or per_horizon.get(h)
                if entry is None:
                    cells.append("n/a")
                    continue
                m = entry[metric_key]
                if m["median"] is None:
                    cells.append("n/a")
                else:
                    cells.append(f"{m['median']:.4f} [{m['p25']:.4f}, {m['p75']:.4f}]")
            lines.append(f"| {model_name} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rollout eval for gltfworld dynamics models.")
    parser.add_argument("--ckpt", required=True, type=Path, help="InteractionTransformer checkpoint (.safetensors)")
    parser.add_argument("--data", required=True, type=Path, help="packed dataset dir or .safetensors file")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mlp-ckpt", type=Path, default=None, help="optional NoInteractionMLP checkpoint")
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--emit-gltf", type=int, default=0, help="write N test episodes as pred/gt .glb pairs")
    parser.add_argument("--video", type=int, default=0, help="render N GT-vs-pred side-by-side mp4s (needs GPU)")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pack_file = _resolve_pack_file(args.data)
    ds = DynamicsDataset(pack_file, split=args.split, mode="sequence")
    n_episodes = ds.num_episodes if args.max_episodes is None else min(args.max_episodes, ds.num_episodes)

    states_gt = ds.states[:n_episodes].to(device)
    mask = ds.mask[:n_episodes].to(device)
    globals_ = ds.globals[:n_episodes].to(device)
    class_ids_all = _load_class_ids(pack_file, args.split, n=n_episodes)
    t = states_gt.shape[1]
    horizons = [h for h in args.horizons if h < t]

    models: dict[str, torch.nn.Module] = {}
    model, model_type = _load_model_from_ckpt(args.ckpt, device)
    models[f"model({model_type})"] = model
    models["ballistic"] = BallisticBaseline().to(device)
    if args.mlp_ckpt is not None:
        mlp_model, mlp_type = _load_model_from_ckpt(args.mlp_ckpt, device)
        models[f"mlp({mlp_type})"] = mlp_model

    all_metrics: dict[str, dict] = {}
    divergence_curves: dict[str, np.ndarray] = {}
    rollouts: dict[str, torch.Tensor] = {}

    for name, m in models.items():
        pred = rollout(m, states_gt[:, 0], mask, globals_, t)
        rollouts[name] = pred
        all_metrics[name] = {str(h): horizon_metrics(pred, states_gt, mask, h) for h in horizons}
        divergence_curves[name] = divergence_curve(pred, states_gt, mask)

    metrics_path = args.out / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "split": args.split,
                "n_episodes": n_episodes,
                "horizons": horizons,
                "metrics": all_metrics,
            },
            indent=2,
        )
    )

    md = _format_markdown_table(all_metrics, horizons)
    (args.out / "metrics.md").write_text(md)

    # divergence-curve PNG
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, curve in divergence_curves.items():
        ax.plot(np.arange(1, len(curve) + 1), curve, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("horizon (frames)")
    ax.set_ylabel("median position error (m)")
    ax.set_title(f"Rollout divergence, {args.split} split ({n_episodes} episodes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "divergence_curve.png", dpi=150)
    plt.close(fig)

    print(f"wrote {metrics_path}, {args.out / 'metrics.md'}, {args.out / 'divergence_curve.png'}")

    if args.emit_gltf > 0:
        meta = json.loads(_pack_meta_path(pack_file).read_text())
        episodes_dir = Path(meta["source_dir"])
        pred_dir = args.out / "pred"
        gt_dir = args.out / "gt"
        pred_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        primary_name = next(iter(models))
        pred_rollout = rollouts[primary_name].detach().cpu().numpy()

        for i in range(min(args.emit_gltf, n_episodes)):
            orig_idx = int(ds.episode_indices[i])
            template_path = episodes_dir / f"ep_{orig_idx:06d}.glb"
            template_episode = load_episode(template_path)

            mask_i = mask[i].detach().cpu().numpy()
            class_ids_i = class_ids_all[i].numpy()
            globals_i = globals_[i].detach().cpu().numpy()

            pred_ep = tensors_to_episode(pred_rollout[i], mask_i, class_ids_i, globals_i, template_episode)
            gt_ep = tensors_to_episode(
                states_gt[i].detach().cpu().numpy(), mask_i, class_ids_i, globals_i, template_episode
            )
            save_episode(pred_ep, pred_dir / f"ep_{orig_idx:06d}.glb")
            save_episode(gt_ep, gt_dir / f"ep_{orig_idx:06d}.glb")
        print(f"wrote {min(args.emit_gltf, n_episodes)} pred/gt episode pairs to {pred_dir}, {gt_dir}")

    if args.video > 0:
        _render_side_by_side_videos(
            args.out,
            pack_file,
            ds,
            rollouts[next(iter(models))],
            states_gt,
            mask,
            class_ids_all,
            min(args.video, n_episodes),
        )

    return 0


def _render_side_by_side_videos(
    out_dir: Path,
    pack_file: Path,
    ds: DynamicsDataset,
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    class_ids_all: torch.Tensor,
    n_videos: int,
) -> None:
    """Render GT-vs-pred side-by-side mp4s for the first ``n_videos``
    episodes. Needs the ``render`` extra (a real GPU + EGL context);
    imported lazily so the rest of the CLI works without it."""
    import imageio

    from gltfworld.render.renderer import EpisodeRenderer

    meta = json.loads(_pack_meta_path(pack_file).read_text())
    episodes_dir = Path(meta["source_dir"])
    video_dir = out_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    renderer = EpisodeRenderer(width=256, height=256)
    try:
        for i in range(n_videos):
            orig_idx = int(ds.episode_indices[i])
            template = load_episode(episodes_dir / f"ep_{orig_idx:06d}.glb")
            class_ids_i = class_ids_all[i].numpy()
            globals_i = ds.globals[i].detach().cpu().numpy()

            pred_ep = tensors_to_episode(pred_np[i], mask_np[i], class_ids_i, globals_i, template)
            gt_ep = tensors_to_episode(gt_np[i], mask_np[i], class_ids_i, globals_i, template)

            def _render_all_frames(ep: Episode) -> list[np.ndarray]:
                renderer.load(ep)
                frames = []
                for frame_idx in range(ep.series.num_frames):
                    renderer.set_frame(frame_idx)
                    frames.append(renderer.render().rgb)
                return frames

            gt_frames = _render_all_frames(gt_ep)
            pred_frames = _render_all_frames(pred_ep)
            frames = [
                np.concatenate([gt_frame, pred_frame], axis=1) for gt_frame, pred_frame in zip(gt_frames, pred_frames)
            ]

            out_path = video_dir / f"ep_{orig_idx:06d}.mp4"
            imageio.mimwrite(out_path, frames, fps=30, macro_block_size=None)
        print(f"wrote {n_videos} side-by-side videos to {video_dir}")
    finally:
        renderer.delete()


if __name__ == "__main__":
    raise SystemExit(main())
