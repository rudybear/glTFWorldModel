"""``gltfworld.physion.ingest`` against the *real* extracted PhysionTest-Core
archive (not a synthetic fixture -- ``data/external/physion/extracted/`` is
git-ignored and only present on a machine that actually unzipped
``data/external/physion/Physion.zip``; CI won't have it, so this whole module
skips cleanly when the directory is absent).

Purpose: confirm the index actually enumerates the real archive's 8
scenarios/1200 trials with correct counts, that at least one real mp4
decodes to the expected shape/dtype, that ``labels.csv``'s ``"True"``/
``"False"`` strings parse to real Python bools, and that iteration order is
deterministic across repeated construction (no reliance on filesystem
listing order).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gltfworld.physion.ingest import SCENARIOS, PhysionIndex, load_frames

REPO_ROOT = Path(__file__).resolve().parent.parent
PHYSION_ROOT = REPO_ROOT / "data" / "external" / "physion" / "extracted" / "Physion"


def _require_real_archive() -> None:
    if not PHYSION_ROOT.exists():
        pytest.skip(
            f"{PHYSION_ROOT} not present -- unzip data/external/physion/Physion.zip "
            "there first (see docs/PHYSION.md); this is real external benchmark data, "
            "not something CI generates or ships"
        )


@pytest.fixture(scope="module")
def index() -> PhysionIndex:
    _require_real_archive()
    return PhysionIndex(PHYSION_ROOT)


def test_index_nonempty_and_counts_per_present_scenario(index: PhysionIndex):
    assert len(index) > 0
    counts = index.scenario_counts()
    assert counts, "expected at least one scenario present"
    for scenario in counts:
        assert scenario in SCENARIOS
    for scenario, count in counts.items():
        assert count > 0, f"scenario {scenario!r} indexed with zero trials"


def test_index_matches_full_archive_when_all_scenarios_present(index: PhysionIndex):
    # The real archive (as downloaded) has all 8 scenarios x 150 trials = 1200,
    # a perfectly balanced binary label split. Assert the strong claim only
    # when the full archive is actually there, so a partial/local extraction
    # still exercises the weaker per-scenario check above without failing here.
    counts = index.scenario_counts()
    if set(counts) != set(SCENARIOS):
        pytest.skip("not all 8 scenarios present in this extraction; skipping the full-archive count check")
    assert len(index) == 1200
    for scenario, count in counts.items():
        assert count == 150, f"{scenario}: expected 150 trials, got {count}"
    n_true = sum(1 for t in index if t.label is True)
    n_false = sum(1 for t in index if t.label is False)
    assert n_true == 600
    assert n_false == 600


def test_trial_paths_exist_and_labels_are_real_bools(index: PhysionIndex):
    for t in index:
        assert t.video_path.exists(), t.video_path
        assert t.redyellow_path.exists(), t.redyellow_path
        assert t.map_path.exists(), t.map_path
        assert isinstance(t.label, bool)


def test_deterministic_ordering(index: PhysionIndex):
    reindexed = PhysionIndex(PHYSION_ROOT)
    assert [t.trial_id for t in index] == [t.trial_id for t in reindexed]
    # Also stable across an independent third construction (not just 2 vs 2).
    reindexed_again = PhysionIndex(PHYSION_ROOT)
    assert [t.trial_id for t in reindexed] == [t.trial_id for t in reindexed_again]
    # Scenario-major, trial_id-minor order.
    scenario_order = [SCENARIOS.index(t.scenario) for t in index]
    assert scenario_order == sorted(scenario_order)


def test_by_id_lookup_roundtrip(index: PhysionIndex):
    first = index[0]
    assert index.by_id(first.trial_id) == first


def test_load_frames_expected_shape_and_dtype(index: PhysionIndex):
    trial = index[0]
    frames = load_frames(trial, max_frames=5)
    assert frames.dtype == np.uint8
    assert frames.ndim == 4
    t, h, w, c = frames.shape
    assert t == 5
    assert c == 3
    assert h > 0 and w > 0


def test_load_frames_stride_and_redyellow_variant(index: PhysionIndex):
    trial = index[0]
    strided = load_frames(trial, max_frames=3, stride=2)
    assert strided.shape[0] == 3

    redyellow = load_frames(trial, max_frames=3, use_redyellow=True)
    assert redyellow.dtype == np.uint8
    assert redyellow.shape[1:] == load_frames(trial, max_frames=3).shape[1:]


def test_load_frames_by_path_directly(index: PhysionIndex):
    trial = index[0]
    frames = load_frames(trial.video_path, max_frames=2)
    assert frames.shape[0] == 2
