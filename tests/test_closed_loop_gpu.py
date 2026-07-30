"""gpu-marked: a short (3-episode) end-to-end smoke of the full V7
closed-loop CLI against the real, trained checkpoints -- ``runs/dynamics-v1``
(InteractionTransformer, V5) and ``runs/perception-v3-cnn`` (PerceptionDETR,
CNN encoder, V6.3) -- exercising every real hop: render GT frames 0/1, run
the real perception model, Hungarian-match detections across frames, roll
forward with the real dynamics model, save every arm as a real GLB, reload
it, and score. Not a claim about attribution-curve *shape* at scale (that's
the orchestrator's full 20-episode run, out of this milestone's scope, see
DESIGN.md's V7 section) -- this only confirms the pipeline runs end-to-end
against real data/checkpoints, produces finite metrics, and that every
emitted GLB is glTF-Validator-clean.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "perception-v1"
DYN_CKPT = REPO_ROOT / "runs" / "dynamics-v1" / "best.safetensors"
PER_CKPT = REPO_ROOT / "runs" / "perception-v3-cnn" / "best.safetensors"
PER_METRICS = REPO_ROOT / "runs" / "perception-v3-cnn" / "eval" / "metrics.json"

pytestmark = pytest.mark.gpu


def _require_real_assets() -> None:
    missing = [
        p for p in (DATA_DIR / "episodes", DATA_DIR / "packed", DYN_CKPT, PER_CKPT, PER_METRICS) if not p.exists()
    ]
    if missing:
        pytest.skip(f"missing real assets for closed-loop smoke: {missing}")


def test_closed_loop_cli_end_to_end_3_episodes(tmp_path: Path):
    _require_real_assets()

    from gltfworld.cli import run_validator
    from gltfworld.eval.closed_loop import main

    out_dir = tmp_path / "closed-loop"
    exit_code = main(
        [
            "--episodes", str(DATA_DIR),
            "--dyn-ckpt", str(DYN_CKPT),
            "--per-ckpt", str(PER_CKPT),
            "--per-metrics", str(PER_METRICS),
            "--out", str(out_dir),
            "--n-episodes", "3",
            "--split", "test",
            "--horizons", "1", "5", "10", "30",
            "--video", "0",
            "--seed", "0",
        ]
    )
    assert exit_code == 0

    metrics_path = out_dir / "metrics.json"
    assert metrics_path.exists()
    assert (out_dir / "attribution.png").exists()

    result = json.loads(metrics_path.read_text())
    assert result["n_episodes"] == 3

    # every reported statistic must be finite (n=0 groups report median=None,
    # which is fine and expected -- only non-null medians must be finite).
    for arm in ("A_oracle", "B_oracle_noise", "C_visual", "ballistic"):
        for h_entry in result["arms"][arm].values():
            for metric in ("position_error_m", "rotation_error_rad"):
                median = h_entry[metric]["median"]
                if median is not None:
                    assert median == median and abs(median) != float("inf"), f"{arm}/{metric} non-finite: {median}"

    for curve in result["full_curves"].values():
        for v in curve:
            assert v == v  # not NaN -- a NaN entry (no data at that horizon) would only
            # be expected if an arm had zero episodes contribute at every horizon,
            # which shouldn't happen for A/B/ballistic with 3 real episodes.

    # sanity direction: at h=30 (if present -- some episodes may be shorter),
    # the oracle dynamics-only arm should not do *worse* than the real visual
    # closed loop by more than noise -- reported as a finding either way, not
    # silently swallowed (see DESIGN.md's V7 acceptance section).
    h30_A = result["arms"]["A_oracle"].get("30", {}).get("position_error_m", {}).get("median")
    h30_C = result["arms"]["C_visual"].get("30", {}).get("position_error_m", {}).get("median")
    print(f"h=30 median position error: arm A (oracle)={h30_A}, arm C (visual)={h30_C}")
    if h30_A is not None and h30_C is not None:
        assert h30_A <= h30_C, (
            f"expected arm A (oracle) error <= arm C (visual) error at h=30, got A={h30_A} C={h30_C} "
            "-- a real finding to report, not to silently paper over, if it ever fires"
        )

    # every emitted GLB, across every arm, validates clean.
    n_checked = 0
    for sub in ("gt", "armA", "armB", "armC"):
        for glb_path in sorted((out_dir / sub).glob("*.glb")):
            report = run_validator(str(glb_path))
            n_errors = report.get("issues", {}).get("numErrors", 0)
            assert n_errors == 0, f"{glb_path}: glTF-Validator reported {n_errors} errors"
            n_checked += 1
    assert n_checked >= 3 * 3  # at least gt/armA/armB for all 3 episodes (armC depends on detections)
