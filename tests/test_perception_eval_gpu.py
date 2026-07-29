"""gpu-marked: the ``gltfworld.eval.perception_eval`` pipeline end-to-end
against the real ``perception-v1`` dataset -- inference, metrics, the
mean-state baseline, and (separately) the GPU re-render check (needs a real
EGL context + the ``render`` extra), all exercised against a small number of
samples/frames to stay fast. Not a claim about the *model*'s trained quality
(this milestone ships an untrained-but-freshly-constructed checkpoint here
purely to exercise the eval pipeline -- full training is explicitly out of
this milestone's scope, per DESIGN.md/VERIFICATION.md).

The render check is exercised via a *direct* call to
``perception_eval.render_check`` using the shared, session-scoped
``episode_renderer`` fixture (``tests/conftest.py``) -- not through
``main()``'s own ``--render-samples`` path, which builds and deletes its
own ad hoc ``EpisodeRenderer``. Per ``EpisodeRenderer``'s documented "one
live renderer per process" constraint, that ad hoc renderer being deleted
mid-session would invalidate the shared EGL display for every other
still-open renderer in this test session (`tests/test_data.py`'s
``test_perception_dataset_reads_rendered_frames`` holds one open via that
same shared fixture) -- exactly the failure mode ``test_data.py`` itself
warns about in its own comment. ``main()``'s CLI path is still exercised
directly with ``--render-samples 0`` (its own dedicated renderer is only
built for a real standalone process, never inside this shared session).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import make_sample_episode

REPO_ROOT = Path(__file__).resolve().parent.parent
EPISODES_DIR = REPO_ROOT / "data" / "perception-v1" / "episodes"
PACK_FILE = REPO_ROOT / "data" / "perception-v1" / "packed" / "perception-v1.safetensors"

pytestmark = pytest.mark.gpu


def _require_real_dataset() -> None:
    if not PACK_FILE.exists() or not EPISODES_DIR.exists():
        pytest.skip(f"{PACK_FILE} not present (run `gltfworld generate --render`+pack first, see data/README.md)")


def _write_fresh_checkpoint(out_dir: Path) -> Path:
    """A freshly-initialized (untrained) PerceptionDETR checkpoint + its
    config.json -- enough for perception_eval's CLI to load and run, without
    paying for a real training run (out of this milestone's scope)."""
    import torch
    from safetensors.torch import save_file

    from gltfworld.train.train_perception import Config, make_model

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    cfg.save(out_dir / "config.json")
    torch.manual_seed(0)
    model = make_model(cfg)
    ckpt_path = out_dir / "fresh.safetensors"
    save_file(model.state_dict(), ckpt_path)
    return ckpt_path


def test_perception_eval_cli_metrics_and_baseline(tmp_path: Path):
    """The metrics/baseline pipeline, with the GPU re-render check
    disabled (``--render-samples 0``) -- exercised separately below with a
    shared renderer, see module docstring."""
    _require_real_dataset()

    ckpt_path = _write_fresh_checkpoint(tmp_path / "ckpt")
    out_dir = tmp_path / "eval"

    from gltfworld.eval.perception_eval import main

    exit_code = main(
        [
            "--ckpt", str(ckpt_path),
            "--data", str(REPO_ROOT / "data" / "perception-v1"),
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
    assert "PerceptionDETR" in result["metrics"]
    assert "mean-state baseline" in result["metrics"]
    assert "render_check" not in result


def test_render_check_with_shared_renderer(tmp_path: Path, episode_renderer):
    _require_real_dataset()

    import torch

    from gltfworld.data.dataset import PerceptionDataset
    from gltfworld.eval.perception_eval import render_check, run_inference
    from gltfworld.train.train_perception import Config, make_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    torch.manual_seed(0)
    model = make_model(cfg).to(device)
    model.eval()

    ds = PerceptionDataset(EPISODES_DIR, PACK_FILE, split="test")
    records, _workspace_stats = run_inference(model, ds, device, batch_size=16)

    out_dir = tmp_path / "eval"
    out_dir.mkdir(parents=True)
    result = render_check(records, ds, EPISODES_DIR, out_dir, n_samples=3, renderer=episode_renderer)

    assert result["n_samples"] == 3
    assert result["roundtrip_max_abs_err"] < 1e-6
    assert result["validate_clean"] is True
    pred_frames_dir = out_dir / "pred_frames"
    assert pred_frames_dir.exists()
    assert len(list(pred_frames_dir.glob("*.glb"))) == 3

    # the shared renderer must still be usable afterward (not deleted).
    episode_renderer.load(make_sample_episode(n_objects=1, T=2))
    episode_renderer.set_frame(0)
    frame = episode_renderer.render()
    assert frame.rgb.shape == (256, 256, 3)
