"""Training harness for ``ArticulationEstimator`` (single RGB frame -> joint
state), driven by a JSON-loadable :class:`Config`. Shares
``gltfworld.train.train_perception``/``train_dynamics``'s harness *contract*
(config dataclass, resumable checkpoints, ``--smoke``/``--smoke-val``,
``log.csv``, safetensors checkpoints) -- an even simpler single-phase
schedule than either of those (no Hungarian matching, no autoregressive
rollout phase; see ``gltfworld.models.articulation``'s module docstring for
why this task collapses to plain regression/classification instead of
set-prediction).

Epoch-equivalent guard (same philosophy as ``train_perception``'s, see its
module docstring's V6.1/V6.2 postmortem for the incident this general
pattern exists to prevent): training refuses to start if the configured
step budget would train on each packed train frame more than
:data:`MAX_EPOCH_EQUIVALENT` times over, unless explicitly overridden. With
``articulated-v1``'s ~1,500 episodes x 100 frames (~150k total, ~90% train)
and a 15k-step / batch-128 default schedule, this lands at roughly 14x --
under the 15x guard with real but thin margin (see :func:`epoch_equivalent`),
which is exactly why the milestone spec calls for "budget steps
accordingly" rather than reusing ``train_perception``'s own 25k-step
default unmodified.

Two fast correctness checks, mirroring ``train_perception``'s: ``--smoke``
(500 steps, train-loss-only) and ``--smoke-val`` (a few thousand steps,
asserts val ``joint_pos_norm`` MAE actually improves) -- see
:data:`SMOKE_VAL_MIN_RELATIVE_IMPROVEMENT`/:data:`SMOKE_VAL_MAE_BOUND` for
the acceptance bar and how it was calibrated (empirically, against this
project's own real ``articulated-v1`` run -- not guessed).

CLI
---

    uv run python -m gltfworld.train.train_articulation \\
        --config configs/articulation_v1.json --out runs/articulation-v1

    # resume an interrupted run:
    uv run python -m gltfworld.train.train_articulation \\
        --config configs/articulation_v1.json --out runs/articulation-v1 --resume

    # fast correctness check (500 steps, asserts >=30% EMA train loss drop):
    uv run python -m gltfworld.train.train_articulation \\
        --config configs/articulation_v1.json --out runs/articulation-v1-smoke --smoke

    # fast generalization check (asserts val joint_pos_norm MAE improves):
    uv run python -m gltfworld.train.train_articulation \\
        --config configs/articulation_v1.json --out runs/articulation-v1-smoke-val --smoke-val

Artifacts written to ``--out``: ``config.json``, ``log.csv`` (``step, split,
lr, loss_total, loss_joint_pos, loss_type, loss_axis, joint_pos_norm_mae,
type_acc, axis_err_deg`` -- the last three val-only columns are blank on
train rows), and checkpoints (``step_{N:07d}.safetensors`` every
``ckpt_every`` steps, ``best.safetensors``, ``last.safetensors``, each with a
matching ``*.train_state.pt``) -- identical scheme to ``train_perception``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from gltfworld.data.dataset import ArticulationDataset
from gltfworld.models.articulation import ArticulationEstimator, LossWeights, compute_articulation_losses, count_params

RAD2DEG = 180.0 / np.pi


# --- config --------------------------------------------------------------------


@dataclass
class Config:
    # data
    episodes_dir: str = "data/articulated-v1/episodes"
    pack_file: str = "data/articulated-v1/packed/articulated-v1.safetensors"

    # model
    d_model: int = 256

    # optimization
    seed: int = 0
    batch_size: int = 128
    lr: float = 2e-4
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    bf16: bool = True
    num_workers: int = 4

    steps: int = 15_000

    # augmentation (RGB only -- geometry/state targets are never touched)
    aug_brightness: float = 0.2
    aug_contrast: float = 0.2
    aug_noise_sigma: float = 0.02

    # loss weights
    loss_w_joint_pos: float = 1.0
    loss_w_type: float = 1.0
    loss_w_axis: float = 1.0

    # logging / checkpointing
    val_every: int = 500
    val_batches: int = 10
    ckpt_every: int = 2500
    log_every: int = 50

    # dataset-scale guard (see check_dataset_scale below)
    allow_high_epoch_equivalent: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = json.loads(Path(path).read_text())
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=2) + "\n")

    def loss_weights(self) -> LossWeights:
        return LossWeights(joint_pos=self.loss_w_joint_pos, type_=self.loss_w_type, axis=self.loss_w_axis)


# --- dataset-scale guard -------------------------------------------------------
#
# Same general lesson as train_perception.check_dataset_scale (see its
# module-level comment / DESIGN.md's V6.1 postmortem): a step budget trained
# too many times over a too-small dataset silently memorizes instead of
# generalizing, and a train-loss-only smoke check can't tell the difference.
# MAX_EPOCH_EQUIVALENT is kept at the same 15x threshold train_perception
# uses -- this milestone's own default config (15k steps, batch 128, against
# articulated-v1's ~135k train frames) lands at ~14.2x, real but thin margin
# under the guard, per this module's own docstring.
MAX_EPOCH_EQUIVALENT = 15.0


def epoch_equivalent(steps: int, batch_size: int, n_train_frames: int) -> float:
    if n_train_frames <= 0:
        return float("inf")
    return steps * batch_size / n_train_frames


def check_dataset_scale(cfg: "Config", n_train_frames: int) -> None:
    eq = epoch_equivalent(cfg.steps, cfg.batch_size, n_train_frames)
    if eq > MAX_EPOCH_EQUIVALENT and not cfg.allow_high_epoch_equivalent:
        raise ValueError(
            f"articulation training dataset too small for this step budget: {n_train_frames} train "
            f"frames, steps={cfg.steps}, batch_size={cfg.batch_size} -> epoch-equivalent={eq:.1f}x "
            f"(each frame would be trained on ~{eq:.1f} times), exceeds MAX_EPOCH_EQUIVALENT="
            f"{MAX_EPOCH_EQUIVALENT:.1f}. See train_perception.py's identical guard / DESIGN.md's V6.1 "
            "postmortem for the failure mode this prevents: a too-small train set trained on too long "
            "memorizes instead of generalizing. Generate more episodes (`gltfworld generate-articulated` "
            "+ `gltfworld pack-articulated`) or, if this is a deliberately small-scale run (e.g. a unit "
            "test), set allow_high_epoch_equivalent=True in the config."
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(cfg: Config) -> ArticulationEstimator:
    return ArticulationEstimator(d_model=cfg.d_model)


# --- RGB-only augmentation (identical scheme to train_perception.augment_rgb) --


def augment_rgb(rgb: torch.Tensor, cfg: Config) -> torch.Tensor:
    b = rgb.shape[0]
    device = rgb.device

    brightness = 1.0 + (torch.rand(b, 1, 1, 1, device=device) * 2 - 1) * cfg.aug_brightness
    out = rgb * brightness

    mean = out.mean(dim=(1, 2, 3), keepdim=True)
    contrast = 1.0 + (torch.rand(b, 1, 1, 1, device=device) * 2 - 1) * cfg.aug_contrast
    out = (out - mean) * contrast + mean

    out = out + torch.randn_like(out) * cfg.aug_noise_sigma
    return out.clamp(0.0, 1.0)


# --- checkpoint IO (identical scheme to train_perception/train_dynamics) ------


def _ckpt_paths(out_dir: Path, tag: str) -> tuple[Path, Path]:
    return out_dir / f"{tag}.safetensors", out_dir / f"{tag}.train_state.pt"


def save_checkpoint(
    out_dir: Path, tag: str, model: torch.nn.Module, global_step: int, optimizer, scheduler, best_val: float
) -> None:
    weights_path, state_path = _ckpt_paths(out_dir, tag)
    save_file(model.state_dict(), weights_path)
    torch.save(
        {
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val": best_val,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "np_rng_state": np.random.get_state(),
            "py_rng_state": random.getstate(),
        },
        state_path,
    )


def load_checkpoint(
    out_dir: Path, tag: str, model: torch.nn.Module, optimizer, scheduler, device: torch.device
) -> tuple[int, float]:
    weights_path, state_path = _ckpt_paths(out_dir, tag)
    model.load_state_dict(load_file(weights_path, device=str(device)))
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng_state"])
    if state.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    np.random.set_state(state["np_rng_state"])
    random.setstate(state["py_rng_state"])
    return state["global_step"], state["best_val"]


# --- CSV logging ---------------------------------------------------------------

_CSV_FIELDS = [
    "step",
    "split",
    "lr",
    "loss_total",
    "loss_joint_pos",
    "loss_type",
    "loss_axis",
    "joint_pos_norm_mae",
    "type_acc",
    "axis_err_deg",
]


def _csv_writer(log_path: Path, resuming: bool):
    is_new = not (resuming and log_path.exists())
    f = log_path.open("w" if is_new else "a", newline="")
    writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
    if is_new:
        writer.writeheader()
    return f, writer


# --- data loading ----------------------------------------------------------------


def _infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def make_loader(cfg: Config, split: str, shuffle: bool) -> DataLoader:
    ds = ArticulationDataset(cfg.episodes_dir, cfg.pack_file, split=split)
    if len(ds) == 0:
        raise ValueError(
            f"ArticulationDataset(split={split!r}) has 0 frames -- check {cfg.episodes_dir}/{cfg.pack_file}"
        )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
    )


# --- evaluation ------------------------------------------------------------------


@torch.no_grad()
def evaluate(model: torch.nn.Module, val_iter, cfg: Config, device: torch.device, n_batches: int) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "joint_pos": 0.0, "type": 0.0, "axis": 0.0}
    abs_errs: list[float] = []
    type_correct: list[bool] = []
    axis_angle_errs_deg: list[float] = []

    for _ in range(n_batches):
        rgb, joint_pos_norm, joint_type_id, axis, limit_min, limit_max = next(val_iter)
        rgb, joint_pos_norm, joint_type_id, axis = (
            rgb.to(device),
            joint_pos_norm.to(device),
            joint_type_id.to(device),
            axis.to(device),
        )
        pred = model(rgb)
        total, comp = compute_articulation_losses(pred, joint_pos_norm, joint_type_id, axis, cfg.loss_weights())
        totals["total"] += float(total.detach())
        for k in ("joint_pos", "type", "axis"):
            totals[k] += comp[k]

        abs_errs.extend((pred["joint_pos_norm"] - joint_pos_norm).abs().detach().cpu().numpy().tolist())
        pred_type = pred["type_logits"].argmax(dim=-1)
        type_correct.extend((pred_type == joint_type_id).detach().cpu().numpy().tolist())
        cos_sim = (pred["axis"] * axis).sum(dim=-1).clamp(-1.0, 1.0)
        axis_angle_errs_deg.extend((torch.acos(cos_sim) * RAD2DEG).detach().cpu().numpy().tolist())

    model.train()
    out = {k: v / n_batches for k, v in totals.items()}
    out["joint_pos_norm_mae"] = float(np.mean(abs_errs)) if abs_errs else float("nan")
    out["type_acc"] = float(np.mean(type_correct)) if type_correct else float("nan")
    out["axis_err_deg"] = float(np.mean(axis_angle_errs_deg)) if axis_angle_errs_deg else float("nan")
    return out


# --- main training loop -----------------------------------------------------------


def train(cfg: Config, out_dir: Path, resume: bool, smoke: bool, smoke_val: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16
    autocast_enabled = cfg.bf16

    # See train_perception.train's identical note: the LR schedule's horizon
    # must stay pinned to the *configured* full-run step count even under
    # --smoke/--smoke-val's truncated loop, or a short run's LR trajectory
    # stops being a faithful proxy for the real run's early portion.
    lr_schedule_steps = cfg.steps

    if smoke:
        cfg = dataclasses.replace(cfg, steps=500, val_every=100, val_batches=2, ckpt_every=500, log_every=10)
    elif smoke_val:
        cfg = dataclasses.replace(cfg, steps=3000, val_every=250, val_batches=8, ckpt_every=3000, log_every=50)

    cfg.save(out_dir / "config.json")

    train_loader = make_loader(cfg, "train", shuffle=True)
    val_loader = make_loader(cfg, "val", shuffle=True)
    check_dataset_scale(cfg, len(train_loader.dataset))
    train_iter = _infinite_loader(train_loader)
    val_iter = _infinite_loader(val_loader)

    model = make_model(cfg).to(device)
    n_params = count_params(model)
    print(f"model=ArticulationEstimator device={device} params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(lr_schedule_steps, 1), eta_min=cfg.lr * cfg.min_lr_ratio
    )

    global_step = 0
    best_val = float("inf")
    loss_ema: float | None = None
    ema_decay = 0.98

    if resume and (out_dir / "last.train_state.pt").exists():
        global_step, best_val = load_checkpoint(out_dir, "last", model, optimizer, scheduler, device)
        print(f"resumed from step {global_step}")
    elif resume:
        print("no checkpoint found to resume from; starting fresh")

    log_file, log_writer = _csv_writer(out_dir / "log.csv", resuming=(global_step > 0))

    train_loss_history: list[tuple[int, float, float]] = []  # (step, raw_loss, ema_loss)
    val_metric_history: list[tuple[int, float]] = []  # (step, joint_pos_norm_mae)
    t_start = time.time()
    model.train()

    try:
        while global_step < cfg.steps:
            rgb, joint_pos_norm, joint_type_id, axis, _limit_min, _limit_max = next(train_iter)
            rgb, joint_pos_norm, joint_type_id, axis = (
                rgb.to(device),
                joint_pos_norm.to(device),
                joint_type_id.to(device),
                axis.to(device),
            )
            rgb = augment_rgb(rgb, cfg)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                pred = model(rgb)
                loss, comp = compute_articulation_losses(pred, joint_pos_norm, joint_type_id, axis, cfg.loss_weights())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            loss_value = float(loss.detach())
            loss_ema = loss_value if loss_ema is None else ema_decay * loss_ema + (1 - ema_decay) * loss_value
            train_loss_history.append((global_step, loss_value, loss_ema))

            if global_step % cfg.log_every == 0 or global_step == 1:
                log_writer.writerow(
                    {
                        "step": global_step,
                        "split": "train",
                        "lr": optimizer.param_groups[0]["lr"],
                        "loss_total": loss_value,
                        "loss_joint_pos": comp["joint_pos"],
                        "loss_type": comp["type"],
                        "loss_axis": comp["axis"],
                        "joint_pos_norm_mae": "",
                        "type_acc": "",
                        "axis_err_deg": "",
                    }
                )
                log_file.flush()

            if global_step % cfg.val_every == 0 or global_step == cfg.steps:
                val_metrics = evaluate(model, val_iter, cfg, device, cfg.val_batches)
                val_metric_history.append((global_step, val_metrics["joint_pos_norm_mae"]))
                log_writer.writerow(
                    {
                        "step": global_step,
                        "split": "val",
                        "lr": optimizer.param_groups[0]["lr"],
                        "loss_total": val_metrics["total"],
                        "loss_joint_pos": val_metrics["joint_pos"],
                        "loss_type": val_metrics["type"],
                        "loss_axis": val_metrics["axis"],
                        "joint_pos_norm_mae": val_metrics["joint_pos_norm_mae"],
                        "type_acc": val_metrics["type_acc"],
                        "axis_err_deg": val_metrics["axis_err_deg"],
                    }
                )
                log_file.flush()
                elapsed = time.time() - t_start
                print(
                    f"step {global_step}/{cfg.steps} train_loss={loss_value:.5f} "
                    f"val_loss={val_metrics['total']:.5f} joint_pos_norm_mae={val_metrics['joint_pos_norm_mae']:.4f} "
                    f"type_acc={val_metrics['type_acc']:.4f} axis_err_deg={val_metrics['axis_err_deg']:.2f} "
                    f"elapsed={elapsed:.1f}s"
                )
                if val_metrics["total"] < best_val:
                    best_val = val_metrics["total"]
                    save_checkpoint(out_dir, "best", model, global_step, optimizer, scheduler, best_val)

            if global_step % cfg.ckpt_every == 0 or global_step == cfg.steps:
                save_checkpoint(out_dir, f"step_{global_step:07d}", model, global_step, optimizer, scheduler, best_val)
                save_checkpoint(out_dir, "last", model, global_step, optimizer, scheduler, best_val)
    finally:
        log_file.close()

    save_checkpoint(out_dir, "last", model, global_step, optimizer, scheduler, best_val)

    return {
        "global_step": global_step,
        "n_params": n_params,
        "train_loss_history": train_loss_history,
        "val_metric_history": val_metric_history,
        "best_val": best_val,
    }


# --smoke-val's two-part pass bar, same shape as train_perception's (see its
# module docstring's V6.2 recalibration note for the general rationale: a
# relative-improvement-only bar can pass from a terrible baseline, an
# absolute-only bar can pass on a flat-but-already-low curve). Calibrated
# against this project's own real articulated-v1 --smoke-val run (see
# DESIGN.md's V9 section for the measured numbers this bar was set below/
# above with margin, not tuned to just barely pass).
SMOKE_VAL_MIN_RELATIVE_IMPROVEMENT = 0.15
SMOKE_VAL_MAE_BOUND = 0.25


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the gltfworld articulation-state estimator.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="500-step fast correctness check; exits 0/1")
    parser.add_argument(
        "--smoke-val",
        action="store_true",
        help="~3,000-step fast generalization check; exits 0/1 (see module docstring)",
    )
    args = parser.parse_args(argv)
    if args.smoke and args.smoke_val:
        parser.error("--smoke and --smoke-val are mutually exclusive")

    cfg = Config.load(args.config)
    result = train(cfg, args.out, resume=args.resume, smoke=args.smoke, smoke_val=args.smoke_val)

    if args.smoke:
        history = result["train_loss_history"]
        n = len(history)
        window = max(1, n // 10)
        start_raw = sum(v for _, v, _ in history[:window]) / window
        end_raw = sum(v for _, v, _ in history[-window:]) / window
        start_ema = history[window - 1][2]
        end_ema = history[-1][2]
        drop_raw = (start_raw - end_raw) / start_raw if start_raw > 0 else 0.0
        drop_ema = (start_ema - end_ema) / start_ema if start_ema > 0 else 0.0
        print(f"smoke: raw start_loss={start_raw:.5f} end_loss={end_raw:.5f} drop={drop_raw * 100:.1f}%")
        print(f"smoke: ema start_loss={start_ema:.5f} end_loss={end_ema:.5f} drop={drop_ema * 100:.1f}%")
        if drop_ema < 0.30:
            print(f"SMOKE FAIL: ema loss only dropped {drop_ema * 100:.1f}%, need >= 30%")
            return 1
        print("SMOKE PASS")
        return 0

    if args.smoke_val:
        history = result["val_metric_history"]
        if len(history) < 2:
            print(f"SMOKE-VAL FAIL: only {len(history)} val evaluation(s), need >= 2 to see a trend")
            return 1
        baseline_step, baseline_mae = history[0]
        if baseline_step != 250:
            print(
                f"SMOKE-VAL FAIL: first val evaluation was at step {baseline_step}, expected step 250 "
                "(val_every misconfigured?) -- cannot compute the step-250 baseline the acceptance bar needs"
            )
            return 1
        last_step, last_mae = history[-1]
        relative_improvement = (baseline_mae - last_mae) / baseline_mae if baseline_mae > 0 else 0.0
        print(
            f"smoke-val: val joint_pos_norm_mae @ step {baseline_step} = {baseline_mae:.4f} -> "
            f"@ step {last_step} = {last_mae:.4f} (relative improvement {relative_improvement * 100:.1f}%)"
        )
        if relative_improvement < SMOKE_VAL_MIN_RELATIVE_IMPROVEMENT:
            print(
                f"SMOKE-VAL FAIL: val joint_pos_norm_mae improved only {relative_improvement * 100:.1f}% from its "
                f"step-{baseline_step} value ({baseline_mae:.4f} -> {last_mae:.4f}), need >= "
                f"{SMOKE_VAL_MIN_RELATIVE_IMPROVEMENT * 100:.1f}%"
            )
            return 1
        if not (last_mae < SMOKE_VAL_MAE_BOUND):
            print(f"SMOKE-VAL FAIL: final val joint_pos_norm_mae {last_mae:.4f} >= bound {SMOKE_VAL_MAE_BOUND:.4f}")
            return 1
        print("SMOKE-VAL PASS")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
