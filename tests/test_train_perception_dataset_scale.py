"""V6.1 regression test for ``gltfworld.train.train_perception``'s
dataset-scale guard (``check_dataset_scale``/``epoch_equivalent``/
``MAX_EPOCH_EQUIVALENT``).

No GPU, no real dataset needed -- these are pure arithmetic checks against
``Config``, so they run in the fast (non-``gpu``) lane and would have failed
loudly, immediately (rather than 25,000 steps and several hours later), had
they existed before the V6 incident: a 25k-step run was launched against a
500-episode (45,800 train-frame) ``perception-v1`` dataset -- an
epoch-equivalent of ~69.9x -- and train loss fell smoothly the entire time
while val ``matched_pos_err`` sat flat at the ~0.6m mean-predictor floor from
step 1000 onward, because the model simply memorized the tiny train set. The
``--smoke`` correctness check (train-loss-only, 500 steps) never looked at
val and passed without incident.

Root-cause evidence supporting the dataset-too-small diagnosis (not a
matching/eval-pipeline bug): running the exact val evaluation code path
(``gltfworld.models.matching.compute_perception_losses``) on 500 held-out
*train* frames from the V6 checkpoint gave median matched_pos_err 0.12m,
while the same code path on 500 real val frames gave 0.61m -- the eval
pipeline is correct; the model just never had enough distinct scenes to
learn anything generalizable.
"""

from __future__ import annotations

import pytest

from gltfworld.train.train_perception import (
    MAX_EPOCH_EQUIVALENT,
    Config,
    check_dataset_scale,
    epoch_equivalent,
)


def test_epoch_equivalent_matches_v6_incident_numbers():
    # The actual V6 run: 25,000 steps, batch_size=128, 458 train episodes x
    # 100 rendered frames/episode = 45,800 train frames.
    eq = epoch_equivalent(steps=25_000, batch_size=128, n_train_frames=45_800)
    assert eq == pytest.approx(69.9, abs=0.1)
    assert eq > MAX_EPOCH_EQUIVALENT


def test_epoch_equivalent_zero_frames_is_infinite():
    assert epoch_equivalent(steps=100, batch_size=128, n_train_frames=0) == float("inf")


def test_check_dataset_scale_raises_on_the_v6_incident_config():
    cfg = Config(steps=25_000, batch_size=128)
    with pytest.raises(ValueError, match="too small"):
        check_dataset_scale(cfg, n_train_frames=45_800)


def test_check_dataset_scale_passes_on_a_properly_sized_dataset():
    # A dataset scaled up to match dynamics-v1 (~10,000 episodes): even the
    # full 25k-step/batch-128 schedule stays well under the guard.
    cfg = Config(steps=25_000, batch_size=128)
    check_dataset_scale(cfg, n_train_frames=900_000)  # does not raise


def test_check_dataset_scale_passes_on_the_500_step_smoke_schedule():
    # --smoke overrides steps=500 regardless of dataset size -- even against
    # the original (too-small-for-25k) 45,800-frame dataset, 500 steps alone
    # is not enough to trip the guard (consistent with --smoke having passed
    # cleanly during the V6 incident; the guard's job is to catch a *full*
    # run against too little data, not to forbid ever touching a small
    # dataset for a quick correctness check).
    cfg = Config(steps=500, batch_size=128)
    check_dataset_scale(cfg, n_train_frames=45_800)  # does not raise


def test_check_dataset_scale_override_flag_suppresses_the_error():
    cfg = Config(steps=25_000, batch_size=128, allow_high_epoch_equivalent=True)
    check_dataset_scale(cfg, n_train_frames=45_800)  # does not raise
