"""Training harness for the dynamics models (``InteractionTransformer`` /
``NoInteractionMLP``), driven by a JSON-loadable :class:`Config`.

Two-phase schedule (see DESIGN.md's V5 section):

- **Phase 1** (``phase1_steps``, default 40k): single-step teacher forcing.
  Every step samples a random ``(state_t, state_t+1)`` transition, adds
  small Gaussian noise to ``state_t`` (position/velocity/rotation -- see
  :func:`add_input_noise`), predicts one step, and computes the masked
  weighted loss (:func:`compute_losses`) against the *clean* ``state_t+1``.
  AdamW, cosine-annealed ``lr`` (default 3e-4 -> a floor).
- **Phase 2** (``phase2_steps``, default 10k): ``K``-step autoregressive
  rollout finetuning, ``K`` annealed linearly from ``phase2_k_min`` to
  ``phase2_k_max`` across the phase. A fresh AdamW at a lower, fixed-ish
  ``lr`` (default 1e-4, itself lightly cosine-annealed). No input noise (the
  model's own rollout errors are the only "noise" here).

CLI
---

    uv run python -m gltfworld.train.train_dynamics \\
        --config configs/dynamics_v1.json --out runs/dynamics-v1

    # resume an interrupted run:
    uv run python -m gltfworld.train.train_dynamics \\
        --config configs/dynamics_v1.json --out runs/dynamics-v1 --resume

    # fast correctness check (500 steps, tiny val, asserts >=30% loss drop):
    uv run python -m gltfworld.train.train_dynamics \\
        --config configs/dynamics_v1.json --out runs/dynamics-v1-smoke --smoke

    # the NoInteractionMLP baseline, same harness:
    uv run python -m gltfworld.train.train_dynamics \\
        --config configs/dynamics_mlp.json --out runs/dynamics-mlp --model mlp

Artifacts written to ``--out``: ``config.json`` (a copy of the resolved
config), ``log.csv`` (``step,phase,split,lr,k,loss_total,loss_pos,loss_vel,
loss_angvel,loss_rot``, appended to -- never truncated -- across resumes),
and checkpoints: ``step_{N:07d}.safetensors`` every ``ckpt_every`` steps,
``best.safetensors`` (lowest val total loss seen so far), ``last.safetensors``
(always the most recent step) -- each with a matching ``*.train_state.pt``
(optimizer/scheduler/step/rng state; not itself a model checkpoint, just
what ``--resume`` needs, hence plain ``torch.save`` rather than safetensors,
which only stores flat tensor maps).

**Determinism / nondeterminism**: ``set_seed`` seeds Python's ``random``,
``numpy``, and every torch RNG (CPU + all CUDA devices) it's given, and
training-state checkpoints round-trip that state so ``--resume`` continues
the same RNG stream. What is *not* pinned down (documented, not silently
ignored): cuDNN kernel selection/algorithm nondeterminism (this model uses
no convolutions, so this is mostly moot in practice, but ``torch.backends.
cudnn.benchmark`` is left at its default rather than forced to a
deterministic-but-slower mode) and the inherent nonassociativity of
floating-point reduction order in CUDA's parallel sum/attention kernels
(bf16 autocast matmuls in particular) -- bit-identical reruns on GPU are not
guaranteed even with every seed fixed, only statistically equivalent runs.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from gltfworld.data.dataset import DynamicsDataset
from gltfworld.models.baselines import NoInteractionMLP
from gltfworld.models.dynamics import (
    ANGVEL_SCALE,
    POS_SCALE,
    VEL_SCALE,
    InteractionTransformer,
    count_params,
)
from gltfworld.models.rotations import (
    axis_angle_to_quat,
    quat_geodesic_angle,
    quat_hemisphere,
    quat_multiply,
    quat_normalize,
)

DEG2RAD = math.pi / 180.0


# --- config --------------------------------------------------------------------


@dataclass
class Config:
    # data
    pack_file: str = "data/dynamics-v1/packed/dynamics-v1.safetensors"

    # model
    model: str = "transformer"  # "transformer" | "mlp"
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    mlp_ratio: int = 4
    mlp_hidden: int = 256

    # optimization
    seed: int = 0
    batch_size: int = 1024
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    bf16: bool = True

    # phase 1: single-step teacher forcing
    phase1_steps: int = 40_000
    phase1_min_lr_ratio: float = 0.05

    # phase 2: K-step rollout finetune
    phase2_steps: int = 10_000
    phase2_lr: float = 1e-4
    phase2_min_lr_ratio: float = 0.1
    phase2_batch_size: int = 128
    phase2_k_min: int = 2
    phase2_k_max: int = 8

    # input noise (phase 1 only), in raw physical units
    noise_sigma_pos: float = 0.005  # meters
    noise_sigma_vel: float = 0.02  # m/s
    noise_sigma_rot_deg: float = 0.5  # degrees

    # loss weights
    loss_weight_pos: float = 1.0
    loss_weight_vel: float = 1.0
    loss_weight_angvel: float = 1.0
    loss_weight_rot: float = 1.0

    # logging / checkpointing
    val_every: int = 1000
    val_batches: int = 20
    ckpt_every: int = 5000
    log_every: int = 50

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(cfg: Config) -> torch.nn.Module:
    if cfg.model == "transformer":
        return InteractionTransformer(
            d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads, mlp_ratio=cfg.mlp_ratio
        )
    if cfg.model == "mlp":
        return NoInteractionMLP(hidden=cfg.mlp_hidden)
    raise ValueError(f"unknown model {cfg.model!r}, expected 'transformer' or 'mlp'")


# --- in-memory samplers (avoid DataLoader overhead: the whole packed split -----
# --- fits comfortably in RAM/VRAM, see module docstring) -----------------------


class TransitionSampler:
    """Random ``(state_t, state_t+1, mask, globals)`` transitions, sampled by
    vectorized gather directly off a :class:`DynamicsDataset` (``mode=
    "transition"``)'s already-loaded tensors -- no per-item Python loop."""

    def __init__(self, dataset: DynamicsDataset, device: torch.device) -> None:
        self.states = dataset.states.to(device)
        self.mask = dataset.mask.to(device)
        self.globals = dataset.globals.to(device)
        self.num_episodes, self.t, self.n_max, self.d = self.states.shape
        self.device = device

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ep_idx = torch.randint(0, self.num_episodes, (batch_size,), device=self.device)
        t_idx = torch.randint(0, self.t - 1, (batch_size,), device=self.device)
        state_t = self.states[ep_idx, t_idx]
        state_t1 = self.states[ep_idx, t_idx + 1]
        mask_b = self.mask[ep_idx]
        globals_b = self.globals[ep_idx]
        return state_t, state_t1, mask_b, globals_b


class SequenceSampler:
    """Random full-episode ``(states, mask, globals)`` sequences, for
    phase-2 rollout finetuning (``mode="sequence"``)."""

    def __init__(self, dataset: DynamicsDataset, device: torch.device) -> None:
        self.states = dataset.states.to(device)
        self.mask = dataset.mask.to(device)
        self.globals = dataset.globals.to(device)
        self.num_episodes, self.t, self.n_max, self.d = self.states.shape
        self.device = device

    def sample(self, batch_size: int, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ep_idx = torch.randint(0, self.num_episodes, (batch_size,), device=self.device)
        t0 = torch.randint(0, self.t - k, (batch_size,), device=self.device)
        # gather a (batch, k+1, N, D) window per-episode, one start index each.
        offsets = torch.arange(k + 1, device=self.device)
        t_idx = t0[:, None] + offsets[None, :]  # (batch, k+1)
        window = self.states[ep_idx[:, None], t_idx]  # (batch, k+1, N, D)
        mask_b = self.mask[ep_idx]
        globals_b = self.globals[ep_idx]
        return window, mask_b, globals_b


# --- noise / loss ----------------------------------------------------------------


def add_input_noise(states: torch.Tensor, mask: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Gaussian noise on position/velocity/rotation, real (unmasked) object
    rows only (padded rows are all-zero and never contribute to the loss;
    perturbing them wastes nothing but there's no reason to)."""
    pos = states[..., 0:3]
    quat = states[..., 3:7]
    vel = states[..., 7:10]
    rest = states[..., 10:22]

    mask_f = mask[..., None].to(states.dtype)

    pos_noisy = pos + torch.randn_like(pos) * cfg.noise_sigma_pos * mask_f
    vel_noisy = vel + torch.randn_like(vel) * cfg.noise_sigma_vel * mask_f

    sigma_rad = cfg.noise_sigma_rot_deg * DEG2RAD
    rotvec_noise = torch.randn_like(pos) * sigma_rad * mask_f
    dq = axis_angle_to_quat(rotvec_noise)
    quat_noisy = quat_hemisphere(quat_normalize(quat_multiply(dq, quat)))

    return torch.cat([pos_noisy, quat_noisy, vel_noisy, rest], dim=-1)


def compute_losses(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, cfg: Config
) -> tuple[torch.Tensor, dict[str, float]]:
    """Masked, weighted loss: MSE on normalized-unit pos/vel/ang_vel plus a
    squared-geodesic-angle rotation term. Returns ``(total, components)``
    with ``components`` detached floats for logging."""
    mask_f = mask.to(pred.dtype)
    denom = mask_f.sum().clamp(min=1.0)

    pos_sq = ((pred[..., 0:3] - target[..., 0:3]) / POS_SCALE) ** 2
    pos_loss = (pos_sq.sum(-1) * mask_f).sum() / denom

    vel_sq = ((pred[..., 7:10] - target[..., 7:10]) / VEL_SCALE) ** 2
    vel_loss = (vel_sq.sum(-1) * mask_f).sum() / denom

    angvel_sq = ((pred[..., 10:13] - target[..., 10:13]) / ANGVEL_SCALE) ** 2
    angvel_loss = (angvel_sq.sum(-1) * mask_f).sum() / denom

    rot_angle = quat_geodesic_angle(pred[..., 3:7], target[..., 3:7])
    rot_loss = ((rot_angle**2) * mask_f).sum() / denom

    total = (
        cfg.loss_weight_pos * pos_loss
        + cfg.loss_weight_vel * vel_loss
        + cfg.loss_weight_angvel * angvel_loss
        + cfg.loss_weight_rot * rot_loss
    )
    components = {
        "pos": float(pos_loss.detach()),
        "vel": float(vel_loss.detach()),
        "angvel": float(angvel_loss.detach()),
        "rot": float(rot_loss.detach()),
    }
    return total, components


# --- checkpoint IO ---------------------------------------------------------------


def _ckpt_paths(out_dir: Path, tag: str) -> tuple[Path, Path]:
    return out_dir / f"{tag}.safetensors", out_dir / f"{tag}.train_state.pt"


def save_checkpoint(
    out_dir: Path,
    tag: str,
    model: torch.nn.Module,
    global_step: int,
    optimizer1: torch.optim.Optimizer,
    scheduler1,
    optimizer2: torch.optim.Optimizer,
    scheduler2,
    best_val: float,
) -> None:
    weights_path, state_path = _ckpt_paths(out_dir, tag)
    save_file(model.state_dict(), weights_path)
    torch.save(
        {
            "global_step": global_step,
            "optimizer1": optimizer1.state_dict(),
            "scheduler1": scheduler1.state_dict(),
            "optimizer2": optimizer2.state_dict(),
            "scheduler2": scheduler2.state_dict(),
            "best_val": best_val,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "np_rng_state": np.random.get_state(),
            "py_rng_state": random.getstate(),
        },
        state_path,
    )


def load_checkpoint(
    out_dir: Path,
    tag: str,
    model: torch.nn.Module,
    optimizer1: torch.optim.Optimizer,
    scheduler1,
    optimizer2: torch.optim.Optimizer,
    scheduler2,
    device: torch.device,
) -> tuple[int, float]:
    weights_path, state_path = _ckpt_paths(out_dir, tag)
    model.load_state_dict(load_file(weights_path, device=str(device)))
    # map_location="cpu" (not `device`): RNG state tensors must stay CPU
    # ByteTensors (torch.cuda.set_rng_state_all rejects a CUDA tensor), and
    # optimizer.load_state_dict already re-casts its own tensors to each
    # parameter's actual device, so loading everything to CPU first is both
    # correct and simplest. weights_only=False: this is gltfworld's own
    # training-state checkpoint (optimizer/scheduler/step/rng state,
    # including a numpy RNG state tuple torch.load's default weights-only
    # unpickler doesn't allowlist), never an untrusted third-party file.
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer1.load_state_dict(state["optimizer1"])
    scheduler1.load_state_dict(state["scheduler1"])
    optimizer2.load_state_dict(state["optimizer2"])
    scheduler2.load_state_dict(state["scheduler2"])
    torch.set_rng_state(state["rng_state"])
    if state.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    np.random.set_state(state["np_rng_state"])
    random.setstate(state["py_rng_state"])
    return state["global_step"], state["best_val"]


# --- CSV logging -----------------------------------------------------------------

_CSV_FIELDS = ["step", "phase", "split", "lr", "k", "loss_total", "loss_pos", "loss_vel", "loss_angvel", "loss_rot"]


def _csv_writer(log_path: Path, resuming: bool):
    is_new = not (resuming and log_path.exists())
    f = log_path.open("w" if is_new else "a", newline="")
    writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
    if is_new:
        writer.writeheader()
    return f, writer


# --- evaluation --------------------------------------------------------------------


@torch.no_grad()
def evaluate(model: torch.nn.Module, val_sampler: TransitionSampler, cfg: Config, n_batches: int) -> dict[str, float]:
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(12345)  # fixed, reproducible val subset each call
    totals = {"total": 0.0, "pos": 0.0, "vel": 0.0, "angvel": 0.0, "rot": 0.0}
    batch_size = min(cfg.batch_size, max(1, val_sampler.num_episodes * (val_sampler.t - 1)))
    for _ in range(n_batches):
        ep_idx = torch.randint(0, val_sampler.num_episodes, (batch_size,), generator=gen).to(val_sampler.device)
        t_idx = torch.randint(0, val_sampler.t - 1, (batch_size,), generator=gen).to(val_sampler.device)
        state_t = val_sampler.states[ep_idx, t_idx]
        state_t1 = val_sampler.states[ep_idx, t_idx + 1]
        mask_b = val_sampler.mask[ep_idx]
        globals_b = val_sampler.globals[ep_idx]
        pred = model(state_t, mask_b, globals_b)
        total, comp = compute_losses(pred, state_t1, mask_b, cfg)
        totals["total"] += float(total.detach())
        for k in ("pos", "vel", "angvel", "rot"):
            totals[k] += comp[k]
    model.train()
    return {k: v / n_batches for k, v in totals.items()}


# --- main training loop -------------------------------------------------------------


def train(cfg: Config, out_dir: Path, resume: bool, smoke: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16
    autocast_enabled = cfg.bf16

    if smoke:
        cfg = dataclasses.replace(
            cfg,
            phase1_steps=500,
            phase2_steps=0,
            val_every=100,
            val_batches=2,
            ckpt_every=500,
            log_every=10,
        )

    cfg.save(out_dir / "config.json")

    train_ds = DynamicsDataset(cfg.pack_file, split="train", mode="transition")
    val_ds = DynamicsDataset(cfg.pack_file, split="val", mode="transition")

    train_sampler = TransitionSampler(train_ds, device)
    val_sampler = TransitionSampler(val_ds, device)
    seq_sampler = SequenceSampler(DynamicsDataset(cfg.pack_file, split="train", mode="sequence"), device)

    model = make_model(cfg).to(device)
    n_params = count_params(model)
    print(f"model={cfg.model} device={device} params={n_params:,}")

    optimizer1 = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, T_max=max(cfg.phase1_steps, 1), eta_min=cfg.lr * cfg.phase1_min_lr_ratio
    )
    optimizer2 = torch.optim.AdamW(model.parameters(), lr=cfg.phase2_lr, weight_decay=cfg.weight_decay)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=max(cfg.phase2_steps, 1), eta_min=cfg.phase2_lr * cfg.phase2_min_lr_ratio
    )

    total_steps = cfg.phase1_steps + cfg.phase2_steps
    global_step = 0
    best_val = float("inf")
    loss_ema: float | None = None
    ema_decay = 0.98

    if resume and (out_dir / "last.train_state.pt").exists():
        global_step, best_val = load_checkpoint(out_dir, "last", model, optimizer1, scheduler1, optimizer2, scheduler2, device)
        print(f"resumed from step {global_step}")
    elif resume:
        print("no checkpoint found to resume from; starting fresh")

    log_file, log_writer = _csv_writer(out_dir / "log.csv", resuming=(global_step > 0))

    train_loss_history: list[tuple[int, float, float]] = []  # (step, raw_loss, ema_loss)
    t_start = time.time()
    model.train()

    try:
        while global_step < total_steps:
            phase = 1 if global_step < cfg.phase1_steps else 2

            if phase == 1:
                state_t, state_t1, mask_b, globals_b = train_sampler.sample(cfg.batch_size)
                state_t_noisy = add_input_noise(state_t, mask_b, cfg)
                optimizer, scheduler = optimizer1, scheduler1
                k = 1
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                    pred = model(state_t_noisy, mask_b, globals_b)
                    loss, comp = compute_losses(pred, state_t1, mask_b, cfg)
            else:
                progress = (global_step - cfg.phase1_steps) / max(cfg.phase2_steps - 1, 1)
                k = int(round(cfg.phase2_k_min + (cfg.phase2_k_max - cfg.phase2_k_min) * progress))
                k = max(1, min(k, seq_sampler.t - 1))
                window, mask_b, globals_b = seq_sampler.sample(cfg.phase2_batch_size, k)
                optimizer, scheduler = optimizer2, scheduler2
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                    cur = window[:, 0]
                    step_losses = []
                    comps_accum = {"pos": 0.0, "vel": 0.0, "angvel": 0.0, "rot": 0.0}
                    for step in range(k):
                        cur = model(cur, mask_b, globals_b)
                        target = window[:, step + 1]
                        step_loss, comp = compute_losses(cur, target, mask_b, cfg)
                        step_losses.append(step_loss)
                        for kk in comps_accum:
                            comps_accum[kk] += comp[kk]
                    loss = torch.stack(step_losses).mean()
                    comp = {kk: v / k for kk, v in comps_accum.items()}

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            loss_value = float(loss.detach())
            loss_ema = loss_value if loss_ema is None else ema_decay * loss_ema + (1 - ema_decay) * loss_value
            train_loss_history.append((global_step, loss_value, loss_ema))

            if global_step % cfg.log_every == 0 or global_step == 1:
                lr = optimizer.param_groups[0]["lr"]
                log_writer.writerow(
                    {
                        "step": global_step,
                        "phase": phase,
                        "split": "train",
                        "lr": lr,
                        "k": k,
                        "loss_total": loss_value,
                        "loss_pos": comp["pos"],
                        "loss_vel": comp["vel"],
                        "loss_angvel": comp["angvel"],
                        "loss_rot": comp["rot"],
                    }
                )
                log_file.flush()

            if global_step % cfg.val_every == 0 or global_step == total_steps:
                val_metrics = evaluate(model, val_sampler, cfg, cfg.val_batches)
                log_writer.writerow(
                    {
                        "step": global_step,
                        "phase": phase,
                        "split": "val",
                        "lr": optimizer.param_groups[0]["lr"],
                        "k": k,
                        "loss_total": val_metrics["total"],
                        "loss_pos": val_metrics["pos"],
                        "loss_vel": val_metrics["vel"],
                        "loss_angvel": val_metrics["angvel"],
                        "loss_rot": val_metrics["rot"],
                    }
                )
                log_file.flush()
                elapsed = time.time() - t_start
                print(
                    f"step {global_step}/{total_steps} phase={phase} k={k} "
                    f"train_loss={loss_value:.5f} val_loss={val_metrics['total']:.5f} "
                    f"elapsed={elapsed:.1f}s"
                )
                if val_metrics["total"] < best_val:
                    best_val = val_metrics["total"]
                    save_checkpoint(out_dir, "best", model, global_step, optimizer1, scheduler1, optimizer2, scheduler2, best_val)

            if global_step % cfg.ckpt_every == 0 or global_step == total_steps:
                save_checkpoint(
                    out_dir, f"step_{global_step:07d}", model, global_step, optimizer1, scheduler1, optimizer2, scheduler2, best_val
                )
                save_checkpoint(out_dir, "last", model, global_step, optimizer1, scheduler1, optimizer2, scheduler2, best_val)
    finally:
        log_file.close()

    # always leave a final "last" checkpoint even if ckpt_every didn't land exactly on total_steps
    save_checkpoint(out_dir, "last", model, global_step, optimizer1, scheduler1, optimizer2, scheduler2, best_val)

    return {
        "global_step": global_step,
        "n_params": n_params,
        "train_loss_history": train_loss_history,
        "best_val": best_val,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a gltfworld dynamics model.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="500-step fast correctness check; exits 0/1")
    parser.add_argument("--model", choices=["transformer", "mlp"], default=None, help="override config's model field")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    if args.model is not None:
        cfg = dataclasses.replace(cfg, model=args.model)

    result = train(cfg, args.out, resume=args.resume, smoke=args.smoke)

    if args.smoke:
        history = result["train_loss_history"]
        n = len(history)
        window = max(1, n // 10)
        # raw per-step loss is high-variance (single random batch each step);
        # the pass/fail check uses an EMA-smoothed loss (decay 0.98, ~50-step
        # time constant) so noise doesn't dominate a 500-step comparison.
        # Both raw and EMA start/end are printed either way, for the record.
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
