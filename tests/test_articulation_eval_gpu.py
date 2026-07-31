"""gpu-marked: the ``gltfworld.eval.articulation_eval`` pipeline end-to-end
against the real ``articulated-v1`` dataset -- inference, metrics, both
baselines, and (separately) the GPU re-render check (needs a real EGL
context + the ``render`` extra). Not a claim about the *model*'s trained
quality when run against a freshly-initialized checkpoint (see
``test_articulation_eval_cli_metrics_and_baselines_fresh_ckpt``); the real
trained-checkpoint numbers are reported by the orchestrator's actual eval
run, not by this test.

Renderer-reuse pattern identical to ``tests/test_perception_eval_gpu.py``
(see its module docstring): the render check is exercised via a *direct*
call to ``articulation_eval.render_check`` using the shared, session-scoped
``episode_renderer`` fixture, never through ``main()``'s own ad hoc
renderer (deleting one ``EpisodeRenderer`` mid-session invalidates the
shared EGL display for every other still-open renderer in this process).
"""

from __future__ import annotations

import json
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


def _write_fresh_checkpoint(out_dir: Path) -> Path:
    import torch
    from safetensors.torch import save_file

    from gltfworld.train.train_articulation import Config, make_model

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    cfg.save(out_dir / "config.json")
    torch.manual_seed(0)
    model = make_model(cfg)
    ckpt_path = out_dir / "fresh.safetensors"
    save_file(model.state_dict(), ckpt_path)
    return ckpt_path


def test_articulation_eval_cli_metrics_and_baselines_fresh_ckpt(tmp_path: Path):
    _require_real_dataset()

    ckpt_path = _write_fresh_checkpoint(tmp_path / "ckpt")
    out_dir = tmp_path / "eval"

    from gltfworld.eval.articulation_eval import main

    exit_code = main(
        [
            "--ckpt", str(ckpt_path),
            "--data", str(REPO_ROOT / "data" / "articulated-v1"),
            "--split", "test",
            "--out", str(out_dir),
            "--batch-size", "16",
            "--render-samples", "0",
        ]
    )
    assert exit_code == 0

    metrics_path = out_dir / "metrics.json"
    assert metrics_path.exists()
    assert (out_dir / "metrics.md").exists()

    result = json.loads(metrics_path.read_text())
    assert result["n_frames"] > 0
    assert "ArticulationEstimator" in result["metrics"]
    assert "predict-midpoint-of-range" in result["metrics"]
    assert "predict-dataset-mean-axis" in result["metrics"]
    assert "render_check" not in result
    assert "acceptance" in result


def test_render_check_with_shared_renderer(tmp_path: Path, episode_renderer):
    _require_real_dataset()

    import torch

    from gltfworld.data.dataset import ArticulationDataset
    from gltfworld.eval.articulation_eval import render_check, run_inference
    from gltfworld.train.train_articulation import Config, make_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    torch.manual_seed(0)
    model = make_model(cfg).to(device)
    model.eval()

    ds = ArticulationDataset(EPISODES_DIR, PACK_FILE, split="test")
    records = run_inference(model, ds, device, batch_size=16)

    out_dir = tmp_path / "eval"
    out_dir.mkdir(parents=True)
    result = render_check(records, ds, EPISODES_DIR, out_dir, n_samples=3, renderer=episode_renderer)

    assert result["n_samples"] == 3
    assert result["roundtrip_max_abs_err"] < 1e-5
    assert result["validate_clean"] is True
    pred_frames_dir = out_dir / "pred_frames"
    assert pred_frames_dir.exists()
    assert len(list(pred_frames_dir.glob("*.glb"))) == 3

    # the shared renderer must still be usable afterward (not deleted).
    from conftest import make_sample_episode

    episode_renderer.load(make_sample_episode(n_objects=1, T=2))
    episode_renderer.set_frame(0)
    frame = episode_renderer.render()
    assert frame.rgb.shape == (256, 256, 3)
