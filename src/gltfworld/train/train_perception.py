"""Training harness for ``PerceptionDETR`` (single RGB frame -> object set),
driven by a JSON-loadable :class:`Config`. Shares
``gltfworld.train.train_dynamics``'s harness *contract* (config dataclass,
resumable checkpoints, ``--smoke``, ``log.csv``, safetensors checkpoints) but
is otherwise a much simpler single-phase schedule -- there is no analogue of
the dynamics model's autoregressive rollout finetune phase here.

Schedule
--------

AdamW, ``lr=2e-4`` cosine-annealed to a floor, batch 128, default 25k steps,
bf16 autocast, grad-norm clip. Every step samples a random batch of
``(rgb, state, mask, class_ids)`` frames (via a plain ``torch.utils.data
.DataLoader`` over ``gltfworld.data.dataset.PerceptionDataset`` -- unlike
``train_dynamics``'s fully-in-VRAM vectorized samplers, image tensors are too
large to preload the whole split at once, so ordinary ``DataLoader``
shuffling + worker processes handle the (memory-mapped) frame I/O instead),
applies RGB-only augmentation (:func:`augment_rgb` -- brightness/contrast
jitter + Gaussian noise; geometry/state targets are never touched, since the
perception task is only ever asked to be robust to *appearance* nuisance, not
lied to about where things actually are), runs the model, Hungarian-matches
predictions to GT (``gltfworld.models.matching.compute_perception_losses``),
and takes one AdamW step on the resulting weighted set-prediction loss.

CLI
---

    uv run python -m gltfworld.train.train_perception \\
        --config configs/perception_v1.json --out runs/perception-v1

    # resume an interrupted run:
    uv run python -m gltfworld.train.train_perception \\
        --config configs/perception_v1.json --out runs/perception-v1 --resume

    # fast correctness check (500 steps, tiny val, asserts >=30% loss drop):
    uv run python -m gltfworld.train.train_perception \\
        --config configs/perception_v1.json --out runs/perception-v1-smoke --smoke

Artifacts written to ``--out``: ``config.json``, ``log.csv`` (``step, split,
lr, loss_total, loss_existence, loss_position, loss_size, loss_shape,
loss_cls, loss_rotation, matched_pos_err_m`` -- the last two val-only columns
are blank on train rows), and checkpoints (``step_{N:07d}.safetensors`` every
``ckpt_every`` steps, ``best.safetensors``, ``last.safetensors``, each with a
matching ``*.train_state.pt``) -- identical scheme to ``train_dynamics``.
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

from gltfworld.data.dataset import PerceptionDataset
from gltfworld.models.matching import LossWeights, MatchCostWeights, compute_perception_losses
from gltfworld.models.perception import PerceptionDETR, count_params


# --- config --------------------------------------------------------------------


@dataclass
class Config:
    # data
    episodes_dir: str = "data/perception-v1/episodes"
    pack_file: str = "data/perception-v1/packed/perception-v1.safetensors"

    # model
    d_model: int = 256
    enc_layers: int = 6
    dec_layers: int = 3
    n_heads: int = 8
    mlp_ratio: int = 4

    # optimization
    seed: int = 0
    batch_size: int = 128
    lr: float = 2e-4
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    bf16: bool = True
    num_workers: int = 4

    steps: int = 25_000

    # augmentation (RGB only -- see module docstring)
    aug_brightness: float = 0.2
    aug_contrast: float = 0.2
    aug_noise_sigma: float = 0.02

    # Hungarian matching cost weights
    match_w_pos: float = 1.0
    match_w_cls: float = 1.0
    match_w_size: float = 1.0

    # loss weights
    loss_w_existence: float = 1.0
    loss_w_position: float = 1.0
    loss_w_size: float = 1.0
    loss_w_shape: float = 1.0
    loss_w_cls: float = 1.0
    loss_w_rotation: float = 1.0

    # normalization scales for the position/size loss terms (see
    # gltfworld.models.matching.compute_perception_losses)
    pos_loss_scale: float = 2.0
    size_loss_scale: float = 0.25

    # logging / checkpointing
    val_every: int = 1000
    val_batches: int = 10
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

    def match_cost_weights(self) -> MatchCostWeights:
        return MatchCostWeights(w_pos=self.match_w_pos, w_cls=self.match_w_cls, w_size=self.match_w_size)

    def loss_weights(self) -> LossWeights:
        return LossWeights(
            existence=self.loss_w_existence,
            position=self.loss_w_position,
            size=self.loss_w_size,
            shape=self.loss_w_shape,
            cls=self.loss_w_cls,
            rotation=self.loss_w_rotation,
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(cfg: Config) -> PerceptionDETR:
    return PerceptionDETR(
        d_model=cfg.d_model,
        enc_layers=cfg.enc_layers,
        dec_layers=cfg.dec_layers,
        n_heads=cfg.n_heads,
        mlp_ratio=cfg.mlp_ratio,
    )


# --- RGB-only augmentation ----------------------------------------------------


def augment_rgb(rgb: torch.Tensor, cfg: Config) -> torch.Tensor:
    """``rgb (B, H, W, 3)`` float32 in ``[0, 1]`` -> brightness/contrast
    jitter + Gaussian noise, clamped back to ``[0, 1]``. Geometry/state
    targets are never touched by this function -- see module docstring for
    why the perception task must stay truthful about *where* things are even
    while its *appearance* robustness is trained up."""
    b = rgb.shape[0]
    device = rgb.device

    brightness = 1.0 + (torch.rand(b, 1, 1, 1, device=device) * 2 - 1) * cfg.aug_brightness
    out = rgb * brightness

    mean = out.mean(dim=(1, 2, 3), keepdim=True)
    contrast = 1.0 + (torch.rand(b, 1, 1, 1, device=device) * 2 - 1) * cfg.aug_contrast
    out = (out - mean) * contrast + mean

    out = out + torch.randn_like(out) * cfg.aug_noise_sigma
    return out.clamp(0.0, 1.0)


# --- checkpoint IO (identical scheme to train_dynamics) -----------------------


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
    # weights_only=False: gltfworld's own training-state checkpoint (see
    # train_dynamics.load_checkpoint's identical note).
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng_state"])
    if state.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    np.random.set_state(state["np_rng_state"])
    random.setstate(state["py_rng_state"])
    return state["global_step"], state["best_val"]


# --- CSV logging -----------------------------------------------------------------

_CSV_FIELDS = [
    "step",
    "split",
    "lr",
    "loss_total",
    "loss_existence",
    "loss_position",
    "loss_size",
    "loss_shape",
    "loss_cls",
    "loss_rotation",
    "matched_pos_err_m",
]


def _csv_writer(log_path: Path, resuming: bool):
    is_new = not (resuming and log_path.exists())
    f = log_path.open("w" if is_new else "a", newline="")
    writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
    if is_new:
        writer.writeheader()
    return f, writer


# --- data loading ---------------------------------------------------------------


def _infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def make_loader(cfg: Config, split: str, shuffle: bool) -> DataLoader:
    ds = PerceptionDataset(cfg.episodes_dir, cfg.pack_file, split=split)
    if len(ds) == 0:
        raise ValueError(
            f"PerceptionDataset(split={split!r}) has 0 frames -- check {cfg.episodes_dir}/{cfg.pack_file}"
        )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
    )


# --- evaluation --------------------------------------------------------------------


@torch.no_grad()
def evaluate(model: torch.nn.Module, val_iter, cfg: Config, device: torch.device, n_batches: int) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "existence": 0.0, "position": 0.0, "size": 0.0, "shape": 0.0, "cls": 0.0, "rotation": 0.0}
    pos_errs: list[float] = []

    for _ in range(n_batches):
        rgb, states, mask, class_ids = next(val_iter)
        rgb, states, mask, class_ids = rgb.to(device), states.to(device), mask.to(device), class_ids.to(device)
        pred = model(rgb)
        total, comp, matches = compute_perception_losses(
            pred, states, mask, class_ids, cfg.match_cost_weights(), cfg.loss_weights(),
            pos_scale=cfg.pos_loss_scale, size_scale=cfg.size_loss_scale,
        )
        totals["total"] += float(total.detach())
        for k in ("existence", "position", "size", "shape", "cls", "rotation"):
            totals[k] += comp[k]

        gt_position = states[..., 0:3]
        for b, (query_idx, gt_idx) in enumerate(matches):
            if query_idx.size == 0:
                continue
            pred_pos = pred["position"][b, query_idx]
            gt_pos = gt_position[b, gt_idx]
            err = torch.linalg.norm(pred_pos - gt_pos, dim=-1)
            pos_errs.extend(err.detach().cpu().numpy().tolist())

    model.train()
    out = {k: v / n_batches for k, v in totals.items()}
    out["matched_pos_err_m"] = float(np.median(pos_errs)) if pos_errs else float("nan")
    return out


# --- main training loop -------------------------------------------------------------


def train(cfg: Config, out_dir: Path, resume: bool, smoke: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16
    autocast_enabled = cfg.bf16

    if smoke:
        cfg = dataclasses.replace(cfg, steps=500, val_every=100, val_batches=2, ckpt_every=500, log_every=10)

    cfg.save(out_dir / "config.json")

    train_loader = make_loader(cfg, "train", shuffle=True)
    val_loader = make_loader(cfg, "val", shuffle=True)
    train_iter = _infinite_loader(train_loader)
    val_iter = _infinite_loader(val_loader)

    model = make_model(cfg).to(device)
    n_params = count_params(model)
    print(f"model=PerceptionDETR device={device} params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.steps, 1), eta_min=cfg.lr * cfg.min_lr_ratio
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
    t_start = time.time()
    model.train()

    try:
        while global_step < cfg.steps:
            rgb, states, mask, class_ids = next(train_iter)
            rgb, states, mask, class_ids = rgb.to(device), states.to(device), mask.to(device), class_ids.to(device)
            rgb = augment_rgb(rgb, cfg)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                pred = model(rgb)
                loss, comp, _matches = compute_perception_losses(
                    pred, states, mask, class_ids, cfg.match_cost_weights(), cfg.loss_weights(),
                    pos_scale=cfg.pos_loss_scale, size_scale=cfg.size_loss_scale,
                )

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
                        "loss_existence": comp["existence"],
                        "loss_position": comp["position"],
                        "loss_size": comp["size"],
                        "loss_shape": comp["shape"],
                        "loss_cls": comp["cls"],
                        "loss_rotation": comp["rotation"],
                        "matched_pos_err_m": "",
                    }
                )
                log_file.flush()

            if global_step % cfg.val_every == 0 or global_step == cfg.steps:
                val_metrics = evaluate(model, val_iter, cfg, device, cfg.val_batches)
                log_writer.writerow(
                    {
                        "step": global_step,
                        "split": "val",
                        "lr": optimizer.param_groups[0]["lr"],
                        "loss_total": val_metrics["total"],
                        "loss_existence": val_metrics["existence"],
                        "loss_position": val_metrics["position"],
                        "loss_size": val_metrics["size"],
                        "loss_shape": val_metrics["shape"],
                        "loss_cls": val_metrics["cls"],
                        "loss_rotation": val_metrics["rotation"],
                        "matched_pos_err_m": val_metrics["matched_pos_err_m"],
                    }
                )
                log_file.flush()
                elapsed = time.time() - t_start
                print(
                    f"step {global_step}/{cfg.steps} train_loss={loss_value:.5f} "
                    f"val_loss={val_metrics['total']:.5f} matched_pos_err={val_metrics['matched_pos_err_m']:.4f}m "
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
        "best_val": best_val,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the gltfworld perception model (PerceptionDETR).")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="500-step fast correctness check; exits 0/1")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    result = train(cfg, args.out, resume=args.resume, smoke=args.smoke)

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
