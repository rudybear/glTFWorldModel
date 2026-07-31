"""gpu-marked: ``train_articulation --smoke``/``--smoke-val`` against the
*real* packed ``articulated-v1`` dataset (not a synthetic fixture --
``data/`` is git-ignored and only present on a machine that actually ran
``gltfworld generate-articulated --render`` + ``gltfworld pack-articulated``).

Purpose: confirm the training harness (``ArticulationDataset`` loading,
RGB-only augmentation, optimizer/scheduler, checkpoint IO) works end-to-end
against real rendered frames, that 500 steps measurably reduces the
(EMA-smoothed) training loss within a 5-minute budget (``--smoke``), and
that a ~3,000-step run shows real (not memorized-then-flat) improvement in
val ``joint_pos_norm_mae`` (``--smoke-val``) -- the same category of check
``train_perception``'s V6.1/V6.2 postmortem established is necessary (a
train-loss-only smoke check cannot tell "the model generalizes" apart from
"the model memorized a too-small train set").
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EPISODES_DIR = REPO_ROOT / "data" / "articulated-v1" / "episodes"
PACK_FILE = REPO_ROOT / "data" / "articulated-v1" / "packed" / "articulated-v1.safetensors"

pytestmark = pytest.mark.gpu


def _require_real_dataset() -> None:
    if not PACK_FILE.exists() or not EPISODES_DIR.exists():
        pytest.skip(
            f"{PACK_FILE} not present (run `gltfworld generate-articulated --render`+`pack-articulated` first)"
        )


def test_smoke_on_real_articulated_v1(tmp_path: Path):
    _require_real_dataset()

    from gltfworld.train.train_articulation import main

    config_path = REPO_ROOT / "configs" / "articulation_v1.json"
    out_dir = tmp_path / "smoke-articulation"
    t0 = time.time()
    exit_code = main(["--config", str(config_path), "--out", str(out_dir), "--smoke"])
    elapsed = time.time() - t0

    assert exit_code == 0, f"--smoke exited {exit_code}"
    assert elapsed < 300.0, f"--smoke took {elapsed:.1f}s, budget is 300s (5 min)"

    assert (out_dir / "log.csv").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "last.safetensors").exists()
    assert (out_dir / "last.train_state.pt").exists()


def test_smoke_val_on_real_articulated_v1(tmp_path: Path):
    _require_real_dataset()

    from gltfworld.train.train_articulation import main

    config_path = REPO_ROOT / "configs" / "articulation_v1.json"
    out_dir = tmp_path / "smoke-val-articulation"
    t0 = time.time()
    exit_code = main(["--config", str(config_path), "--out", str(out_dir), "--smoke-val"])
    elapsed = time.time() - t0

    assert exit_code == 0, f"--smoke-val exited {exit_code}"
    assert elapsed < 1200.0, f"--smoke-val took {elapsed:.1f}s, budget is 1200s (20 min)"

    assert (out_dir / "log.csv").exists()
