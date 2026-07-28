"""``gltfworld.data.stats``/``gltfworld stats`` CLI: sanity-checked against a
small deterministically simulated dataset (MuJoCo, ``sim`` extra needed to
*generate* fixtures -- ``compute_stats`` itself has no MuJoCo dependency).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from gltfworld.data.pack import pack_dataset
from gltfworld.data.stats import compute_stats, format_human
from gltfworld.datagen.generate import generate_dataset


@pytest.fixture(scope="module")
def packed_dataset(tmp_path_factory) -> Path:
    out_dir = tmp_path_factory.mktemp("stats_episodes")
    generate_dataset(out_dir, episodes=12, seed=555, steps=30, hz=30.0)
    pack_file = tmp_path_factory.mktemp("stats_packed") / "ds.safetensors"
    pack_dataset(out_dir, pack_file)
    return pack_file


def test_compute_stats_basic_shape(packed_dataset: Path):
    report = compute_stats(packed_dataset)
    assert report["episode_count"] == 12
    assert report["transition_count"] == 12 * (30 - 1)
    assert report["nan_inf_count"] == 0
    assert sum(report["split_counts"].values()) == 12
    # per_shape_counts is over (episode, frame, object) rows; class_id_histogram
    # is over (episode, object) instances only -- related by a factor of T.
    assert sum(report["per_shape_counts"].values()) == report["frames_per_episode"] * sum(
        report["class_id_histogram"].values()
    )
    assert set(report["per_shape_counts"]) <= {"sphere", "box", "cylinder"}


def test_compute_stats_ranges_are_sane(packed_dataset: Path):
    report = compute_stats(packed_dataset)
    friction = report["friction_range"]
    assert 0.0 <= friction["min"] and friction["max"] <= 1.01

    mass = report["mass_range"]
    assert mass["min"] > 0.0

    # wm-scenes-v1 objects start within a modest workspace and never travel
    # implausibly far in a 30-frame (~1s) episode.
    for axis_range in report["position_range"].values():
        assert abs(axis_range["max"]) < 20.0
        assert abs(axis_range["min"]) < 20.0


def test_compute_stats_ground_penetration_and_energy_present(packed_dataset: Path):
    report = compute_stats(packed_dataset)
    ge = report["ground_penetration_and_energy"]
    assert "error" not in ge
    assert 0.0 <= ge["steady_state_within_5mm_fraction"] <= 1.0
    assert 0.0 <= ge["transient_within_100mm_fraction"] <= 1.0
    assert 0.0 <= ge["energy_non_increasing_fraction"] <= 1.0
    assert 0.0 <= ge["off_plate_object_frame_fraction"] <= 1.0
    assert 0.0 <= ge["episodes_with_departure_fraction"] <= 1.0
    # The loose "didn't fall through the floor" bound should hold for every
    # episode in a short, in-workspace wm-scenes-v1 run (see DESIGN.md) --
    # frames where an object has left the finite ground plate are excluded
    # from this check (see DESIGN.md "Finite ground plate"), not counted
    # as penetration.
    assert ge["transient_within_100mm_fraction"] == 1.0


def test_format_human_is_nonempty_text(packed_dataset: Path):
    report = compute_stats(packed_dataset)
    text = format_human(report)
    assert "episodes:" in text
    assert "NaN/Inf count: 0" in text


def test_stats_cli_json_matches_compute_stats(packed_dataset: Path):
    result = subprocess.run(
        [sys.executable, "-m", "gltfworld.cli", "stats", str(packed_dataset), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    cli_report = json.loads(result.stdout)
    direct_report = compute_stats(packed_dataset)
    assert cli_report == direct_report


def test_stats_cli_human_exits_zero(packed_dataset: Path):
    result = subprocess.run(
        [sys.executable, "-m", "gltfworld.cli", "stats", str(packed_dataset)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "NaN/Inf count: 0" in result.stdout
