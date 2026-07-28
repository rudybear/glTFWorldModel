"""Command-line entry point for gltfworld.

``validate`` and ``inspect`` are real as of milestone V1 (the glTF transport
codec); ``render`` and ``crosscheck`` are real as of milestone V2 (the
headless renderer); ``generate`` is real as of milestone V3 (MuJoCo
data generation); ``stats`` is real as of the dataset/provenance/stats
milestone (V4 per the pre-training-gate scheme).
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

_STUB_SUBCOMMANDS: tuple[str, ...] = ()

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


def _cmd_render(path: str, out: str, size: int) -> int:
    # Imported lazily: the `render` extra (pyrender's runtime deps) isn't
    # required just to run `validate`/`inspect`.
    from gltfworld.render.renderer import render_episode

    out_dir = render_episode(path, out, width=size, height=size)
    print(f"gltfworld render: wrote rgb.npy, seg.npy, depth.npy, frame_000.png to {out_dir}")
    return 0


def _cmd_crosscheck(path: str, size: int, out: str | None) -> int:
    from gltfworld.render.crosscheck import crosscheck_frame0, write_side_by_side_png
    from gltfworld.render.renderer import EpisodeRenderer
    from gltfworld.scene.convert import load_episode

    episode = load_episode(path)
    renderer = EpisodeRenderer(width=size, height=size)
    try:
        result = crosscheck_frame0(episode, renderer, width=size, height=size)
    finally:
        renderer.delete()

    if out is None:
        out = str(Path("/tmp") / "gltfworld_crosscheck" / f"{Path(path).stem}.png")
    png_path = write_side_by_side_png(result, out)

    print(f"frame 0 binary silhouette IoU: {result.iou:.4f}")
    for object_id, iou in sorted(result.per_object_iou.items()):
        print(f"  object_id={object_id} IoU: {iou:.4f}")
    print(f"side-by-side + diff PNG: {png_path}")
    return 0 if result.iou >= 0.98 else 1


def _cmd_generate(out: str, episodes: int, seed: int, steps: int, hz: float, render: bool, size: int) -> int:
    # Imported lazily: the `sim` extra (mujoco) isn't required just to run
    # validate/inspect/render/crosscheck.
    from gltfworld.datagen.generate import generate_dataset

    result = generate_dataset(out, episodes, seed, steps=steps, hz=hz, render=render, size=size)
    print(f"gltfworld generate: wrote {len(result.episode_paths)} episode(s) + manifest to {result.out_dir}")
    return 0


def _cmd_pack(episodes_dir: str, out_file: str, n_max: int) -> int:
    from gltfworld.data.pack import pack_dataset

    result = pack_dataset(episodes_dir, out_file, n_max=n_max)
    print(
        f"gltfworld pack: wrote {result.out_file} ({result.count} episodes, "
        f"T={result.t}, N_max={result.n_max}) + {result.meta_path}"
    )
    print(f"split sizes: {result.split_counts}")
    return 0


def _cmd_stats(pack_file: str, as_json: bool) -> int:
    from gltfworld.data.stats import compute_stats, format_human

    report = compute_stats(pack_file)
    if as_json:
        print(json.dumps(report, indent=2))
    else:
        print(format_human(report))
    return 0 if report["nan_inf_count"] == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gltfworld")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate a GLB/glTF file with the pinned glTF-Validator")
    validate_parser.add_argument("path", help="path to a .glb or .gltf file")

    inspect_parser = subparsers.add_parser("inspect", help="print a summary of an episode GLB/glTF file")
    inspect_parser.add_argument("path", help="path to a .glb or .gltf file")

    render_parser = subparsers.add_parser("render", help="render an episode GLB to rgb/seg/depth .npy files")
    render_parser.add_argument("path", help="path to a .glb episode file")
    render_parser.add_argument("--out", required=True, help="output directory")
    render_parser.add_argument("--size", type=int, default=256, help="square render size in pixels (default 256)")

    crosscheck_parser = subparsers.add_parser(
        "crosscheck", help="cross-render frame 0 against MuJoCo and report silhouette IoU"
    )
    crosscheck_parser.add_argument("path", help="path to a .glb episode file")
    crosscheck_parser.add_argument("--size", type=int, default=256, help="square render size in pixels (default 256)")
    crosscheck_parser.add_argument(
        "--out", default=None, help="side-by-side PNG path (default: /tmp/gltfworld_crosscheck/<stem>.png)"
    )

    generate_parser = subparsers.add_parser(
        "generate", help="sample wm-scenes-v1 scenes, simulate in MuJoCo, write GLB episodes + manifest"
    )
    generate_parser.add_argument("--out", required=True, help="output directory")
    generate_parser.add_argument("--episodes", type=int, required=True, help="number of episodes to generate")
    generate_parser.add_argument("--seed", type=int, required=True, help="base seed (episode i uses seed + i)")
    generate_parser.add_argument("--steps", type=int, default=100, help="recorded frames per episode (default 100)")
    generate_parser.add_argument("--hz", type=float, default=30.0, help="recording rate in Hz (default 30)")
    generate_parser.add_argument(
        "--render", action="store_true", help="also render each episode's frames (needs the render extra + GPU)"
    )
    generate_parser.add_argument(
        "--size", type=int, default=256, help="square render size in pixels when --render is passed (default 256)"
    )

    pack_parser = subparsers.add_parser("pack", help="pack a directory of ep_*.glb episodes into one safetensors file")
    pack_parser.add_argument("episodes_dir", help="directory of ep_*.glb episodes (+ manifest.json)")
    pack_parser.add_argument("--out", required=True, dest="out_file", help="output .safetensors path")
    pack_parser.add_argument("--n-max", type=int, default=5, help="pad object dimension to this many dynamic objects (default 5)")

    stats_parser = subparsers.add_parser("stats", help="report dataset statistics for a packed dataset")
    stats_parser.add_argument("pack_file", help="path to a packed .safetensors file (see `gltfworld pack`)")
    stats_parser.add_argument("--json", action="store_true", dest="as_json", help="emit machine-readable JSON")

    for name in _STUB_SUBCOMMANDS:
        sub = subparsers.add_parser(name)
        sub.add_argument("args", nargs=argparse.REMAINDER)

    parsed = parser.parse_args(argv)

    if parsed.command == "validate":
        return _cmd_validate(parsed.path)
    if parsed.command == "inspect":
        return _cmd_inspect(parsed.path)
    if parsed.command == "render":
        return _cmd_render(parsed.path, parsed.out, parsed.size)
    if parsed.command == "crosscheck":
        return _cmd_crosscheck(parsed.path, parsed.size, parsed.out)
    if parsed.command == "generate":
        return _cmd_generate(
            parsed.out, parsed.episodes, parsed.seed, parsed.steps, parsed.hz, parsed.render, parsed.size
        )
    if parsed.command == "pack":
        return _cmd_pack(parsed.episodes_dir, parsed.out_file, parsed.n_max)
    if parsed.command == "stats":
        return _cmd_stats(parsed.pack_file, parsed.as_json)

    print(f"gltfworld {parsed.command}: not implemented yet (milestone V1+)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
