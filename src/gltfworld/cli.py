"""Command-line entry point for gltfworld.

``validate`` and ``inspect`` are real as of milestone V1 (the glTF transport
codec); ``render``/``generate``/``stats``/``crosscheck`` remain V0-style
stubs until their milestones land.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

_STUB_SUBCOMMANDS = ("render", "generate", "stats", "crosscheck")

GLTF_VALIDATOR_VERSION = "2.0.0-dev.3.10"
_VALIDATOR_URL_TEMPLATE = (
    "https://github.com/KhronosGroup/glTF-Validator/releases/download/"
    "{version}/gltf_validator-{version}-linux64.tar.xz"
)


def _cache_dir() -> Path:
    override = os.environ.get("GLTFWORLD_CACHE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "gltfworld"


def ensure_validator_binary() -> Path:
    """Download (if needed) and return the path to the pinned glTF-Validator binary.

    Cached under ``~/.cache/gltfworld/`` (override with ``GLTFWORLD_CACHE_DIR``),
    same release used by CI (see ``.github/workflows/ci.yml``).
    """
    cache_dir = _cache_dir()
    install_dir = cache_dir / f"gltf-validator-{GLTF_VALIDATOR_VERSION}-linux64"
    binary_path = install_dir / "gltf_validator"
    if binary_path.exists():
        return binary_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    install_dir.mkdir(parents=True, exist_ok=True)
    url = _VALIDATOR_URL_TEMPLATE.format(version=GLTF_VALIDATOR_VERSION)
    archive_path = cache_dir / f"gltf_validator-{GLTF_VALIDATOR_VERSION}-linux64.tar.xz"
    urllib.request.urlretrieve(url, archive_path)  # noqa: S310 (pinned https github release URL)
    with tarfile.open(archive_path) as tar:
        tar.extractall(install_dir, filter="data")  # noqa: S202 (trusted pinned release archive)

    mode = binary_path.stat().st_mode
    binary_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return binary_path


def run_validator(path: str) -> dict:
    """Run the pinned glTF-Validator on ``path``, returning its parsed JSON report."""
    binary = ensure_validator_binary()
    result = subprocess.run(
        [str(binary), "-o", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"glTF-Validator did not produce JSON output.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        ) from exc


def _cmd_validate(path: str) -> int:
    report = run_validator(path)
    issues = report.get("issues", {})
    num_errors = issues.get("numErrors", 0)
    num_warnings = issues.get("numWarnings", 0)

    print(json.dumps(report, indent=2))
    print(f"gltfworld validate: {num_errors} error(s), {num_warnings} warning(s)", file=sys.stderr)
    return 0 if num_errors == 0 else 1


def _cmd_inspect(path: str) -> int:
    # Imported lazily: keeps `gltfworld --help`/stub subcommands cheap and
    # avoids a hard numpy/pygltflib import for commands that don't need it.
    from gltfworld.scene.convert import load_episode

    episode = load_episode(path)
    scene = episode.scene
    series = episode.series

    print(f"objects: {len(scene.objects)}")
    for obj in scene.objects:
        print(
            f"  - id={obj.object_id} shape={obj.shape} category={obj.category!r} "
            f"mass={obj.mass:.6g} static={obj.is_static}"
        )

    duration = float(series.times[-1] - series.times[0]) if series.num_frames > 1 else 0.0
    print(f"frames (T): {series.num_frames}")
    print(f"duration: {duration:.6g}s (dt={scene.dt:.6g})")

    channels = [
        name
        for name, present in (
            ("lin_vel", series.lin_vel is not None),
            ("ang_vel", series.ang_vel is not None),
            ("actions", series.actions is not None),
            ("pose_var", series.pose_var is not None),
        )
        if present
    ]
    print(f"optional channels: {', '.join(channels) if channels else '(none)'}")

    import pygltflib

    gltf = pygltflib.GLTF2.load(str(path))
    used = gltf.extensionsUsed or []
    print(f"extensions used: {', '.join(used) if used else '(none)'}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gltfworld")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate a GLB/glTF file with the pinned glTF-Validator")
    validate_parser.add_argument("path", help="path to a .glb or .gltf file")

    inspect_parser = subparsers.add_parser("inspect", help="print a summary of an episode GLB/glTF file")
    inspect_parser.add_argument("path", help="path to a .glb or .gltf file")

    for name in _STUB_SUBCOMMANDS:
        sub = subparsers.add_parser(name)
        sub.add_argument("args", nargs=argparse.REMAINDER)

    parsed = parser.parse_args(argv)

    if parsed.command == "validate":
        return _cmd_validate(parsed.path)
    if parsed.command == "inspect":
        return _cmd_inspect(parsed.path)

    print(f"gltfworld {parsed.command}: not implemented yet (milestone V1+)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
