"""End-to-end: a sample episode, saved as GLB, validates clean against the
real (pinned) Khronos glTF-Validator binary via the CLI code path.

Draft/unknown extensions (KHR_implicit_shapes, KHR_physics_rigid_bodies,
RWM_state_series) are expected to produce UNSUPPORTED_EXTENSION *info*-level
messages, not errors -- the validator doesn't know about them. Those, and
any other non-error messages, are allowed; only ``numErrors`` must be zero.
"""

from __future__ import annotations

import socket

import pytest

from gltfworld.cli import ensure_validator_binary, run_validator
from gltfworld.scene.convert import save_episode


def _network_and_binary_unavailable() -> bool:
    try:
        ensure_validator_binary()
        return False
    except Exception:
        pass
    # Already-cached binary would have returned above; only skip if we
    # genuinely can't fetch it (no cached copy, no network).
    try:
        socket.create_connection(("github.com", 443), timeout=3).close()
        return False
    except OSError:
        return True


@pytest.mark.skipif(_network_and_binary_unavailable(), reason="no cached glTF-Validator binary and no network to fetch it")
def test_sample_episode_validates_clean(sample_episode, tmp_path):
    glb_path = tmp_path / "sample_episode.glb"
    save_episode(sample_episode, glb_path)

    report = run_validator(str(glb_path))
    issues = report["issues"]

    assert issues["numErrors"] == 0, issues["messages"]

    # Warnings, if any, must only be about our own unknown/draft extensions.
    warning_codes = {m["code"] for m in issues["messages"] if m["severity"] == 1}
    assert warning_codes <= {"UNSUPPORTED_EXTENSION"}, issues["messages"]


@pytest.mark.skipif(_network_and_binary_unavailable(), reason="no cached glTF-Validator binary and no network to fetch it")
def test_validator_reports_unsupported_extensions_as_non_errors(sample_episode, tmp_path):
    glb_path = tmp_path / "sample_episode.glb"
    save_episode(sample_episode, glb_path)

    report = run_validator(str(glb_path))
    messages = report["issues"]["messages"]
    unsupported = [m for m in messages if m["code"] == "UNSUPPORTED_EXTENSION"]
    assert unsupported
    for m in unsupported:
        assert m["severity"] != 0  # not an error


def test_cli_validate_exit_code(sample_episode, tmp_path, capsys):
    if _network_and_binary_unavailable():
        pytest.skip("no cached glTF-Validator binary and no network to fetch it")

    from gltfworld.cli import main

    glb_path = tmp_path / "sample_episode.glb"
    save_episode(sample_episode, glb_path)

    exit_code = main(["validate", str(glb_path)])
    assert exit_code == 0


def test_cli_inspect_smoke(sample_episode, tmp_path, capsys):
    from gltfworld.cli import main

    glb_path = tmp_path / "sample_episode.glb"
    save_episode(sample_episode, glb_path)

    exit_code = main(["inspect", str(glb_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "objects:" in out
    assert "frames (T):" in out
    assert "extensions used:" in out
