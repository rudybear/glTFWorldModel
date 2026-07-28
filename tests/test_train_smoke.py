"""gpu-marked: ``train_dynamics --smoke`` against the *real* packed
``dynamics-v1`` dataset (not a synthetic fixture -- ``data/`` is git-ignored
and only present on a machine that actually ran ``gltfworld generate`` +
``gltfworld pack``, hence ``@pytest.mark.gpu`` alongside the GPU/timing
requirement: a real GPU is what gets this under the 3-minute budget).

Purpose: confirm the training harness (data loading, noise injection,
optimizer/scheduler, checkpoint IO) actually works end-to-end against real
data, and that 500 steps measurably reduces the training loss -- not just
that the code runs without raising.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACK_FILE = REPO_ROOT / "data" / "dynamics-v1" / "packed" / "dynamics-v1.safetensors"

pytestmark = pytest.mark.gpu


def _require_real_dataset() -> None:
    if not PACK_FILE.exists():
        pytest.skip(f"{PACK_FILE} not present (run `gltfworld generate`+pack first, see data/README.md)")


@pytest.mark.parametrize("model", ["transformer", "mlp"])
def test_smoke_on_real_dynamics_v1(tmp_path: Path, model: str):
    _require_real_dataset()

    from gltfworld.train.train_dynamics import Config, main

    config_name = "dynamics_v1.json" if model == "transformer" else "dynamics_mlp.json"
    config_path = REPO_ROOT / "configs" / config_name
    cfg = Config.load(config_path)
    assert Path(cfg.pack_file) == Path("data/dynamics-v1/packed/dynamics-v1.safetensors")

    out_dir = tmp_path / f"smoke-{model}"
    t0 = time.time()
    exit_code = main(["--config", str(config_path), "--out", str(out_dir), "--smoke"])
    elapsed = time.time() - t0

    assert exit_code == 0, f"--smoke exited {exit_code} for model={model}"
    assert elapsed < 180.0, f"--smoke took {elapsed:.1f}s, budget is 180s (3 min)"

    assert (out_dir / "log.csv").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "last.safetensors").exists()
    assert (out_dir / "last.train_state.pt").exists()
