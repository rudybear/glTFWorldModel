"""gpu-marked: ``train_perception --smoke`` against the *real* packed
``perception-v1`` dataset (not a synthetic fixture -- ``data/`` is
git-ignored and only present on a machine that actually ran
``gltfworld generate --render`` + ``gltfworld pack``).

Purpose: confirm the training harness (PerceptionDataset loading, RGB-only
augmentation, Hungarian matching, optimizer/scheduler, checkpoint IO) works
end-to-end against real rendered frames, and that 500 steps measurably
reduces the (EMA-smoothed) training loss within the 5-minute budget -- not
just that the code runs without raising.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EPISODES_DIR = REPO_ROOT / "data" / "perception-v1" / "episodes"
PACK_FILE = REPO_ROOT / "data" / "perception-v1" / "packed" / "perception-v1.safetensors"

pytestmark = pytest.mark.gpu


def _require_real_dataset() -> None:
    if not PACK_FILE.exists() or not EPISODES_DIR.exists():
        pytest.skip(f"{PACK_FILE} not present (run `gltfworld generate --render`+pack first, see data/README.md)")


def test_smoke_on_real_perception_v1(tmp_path: Path):
    _require_real_dataset()

    from gltfworld.train.train_perception import main

    config_path = REPO_ROOT / "configs" / "perception_v1.json"
    out_dir = tmp_path / "smoke-perception"
    t0 = time.time()
    exit_code = main(["--config", str(config_path), "--out", str(out_dir), "--smoke"])
    elapsed = time.time() - t0

    assert exit_code == 0, f"--smoke exited {exit_code}"
    assert elapsed < 300.0, f"--smoke took {elapsed:.1f}s, budget is 300s (5 min)"

    assert (out_dir / "log.csv").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "last.safetensors").exists()
    assert (out_dir / "last.train_state.pt").exists()
