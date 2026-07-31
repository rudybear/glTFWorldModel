"""``gltfworld generate-articulated`` (V9): sample ``wm-articulated-v1``
scenes (a cabinet with a hinged door, or a chest/table with a sliding
drawer -- see ``gltfworld.datagen.articulated``), drive each with a
scripted push in MuJoCo, and write each as a GLB episode (joints +
``joint_position`` channel + ``extras.rwm`` semantics, all through the
existing, independently-verified V9-prep transport codec) -- mirrors
``gltfworld.datagen.generate``'s structure/manifest scheme for the flat
``wm-scenes-v1`` distribution, just over
``gltfworld.datagen.articulated``'s sampler/simulator instead. No new
encoding, no ML, no dataset packing (see ``gltfworld.data.pack_articulated``
for that).

Writes ``ep_{i:06d}.glb`` per episode plus one ``manifest.json`` describing
the whole run (dataset/scene version, base seed, per-episode seeds/kinds,
dt, T, git revision).

**Exact 50/50 door/drawer mix**: episode ``i`` is pinned to ``kind="door"``
for even ``i``, ``"drawer"`` for odd ``i`` (rather than leaving ``kind`` to
``sample_articulated_scene``'s own internal random draw) -- this guarantees
an exact 50/50 split regardless of episode count, rather than a
statistically-close-to-50/50 one. ``axis`` is left to the sampler's own
per-seed random draw (uniform over ``{0, 1, 2}``, see
``gltfworld.datagen.articulated``'s "axis coverage over realism" design
note), so the joint-axis distribution is whatever that sampler already
produces.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from gltfworld.datagen.articulated import SCENE_VERSION, sample_articulated_scene, simulate_articulated
from gltfworld.scene.convert import save_episode
from gltfworld.scene.episode import Episode

DATASET_VERSION = "wm-articulated-v1"


def _git_describe() -> str:
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except OSError:
        pass
    return "unknown"


@dataclass
class GenerateArticulatedResult:
    out_dir: Path
    manifest_path: Path
    episode_paths: list[Path]


def generate_articulated_dataset(
    out_dir: str | Path,
    episodes: int,
    seed: int,
    *,
    steps: int = 100,
    hz: float = 30.0,
    render: bool = False,
    size: int = 256,
) -> GenerateArticulatedResult:
    """Generate ``episodes`` ``wm-articulated-v1`` episodes seeded from
    ``seed`` (episode ``i`` uses ``seed + i``, recorded in the manifest so
    any single episode is independently reproducible from its own seed)
    into ``out_dir``.

    ``render=True`` additionally renders each episode's frames (rgb+seg+
    depth) via ``gltfworld.render.renderer.render_episode`` into an
    ``ep_{i:06d}/`` subdirectory next to its GLB (needs the ``render``
    extra + a working GPU/EGL context -- not exercised by any non-gpu test).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_paths: list[Path] = []
    episode_seeds: list[int] = []
    episode_kinds: list[str] = []

    for i in range(episodes):
        episode_seed = seed + i
        kind = "door" if i % 2 == 0 else "drawer"
        episode_seeds.append(episode_seed)
        episode_kinds.append(kind)

        sampled = sample_articulated_scene(episode_seed, kind=kind)
        series = simulate_articulated(sampled, T=steps, record_hz=hz)
        episode = Episode(scene=sampled.scene, series=series)

        episode_path = out_dir / f"ep_{i:06d}.glb"
        save_episode(episode, episode_path)
        episode_paths.append(episode_path)

        if render:
            from gltfworld.render.renderer import render_episode

            render_episode(episode_path, out_dir / f"ep_{i:06d}", width=size, height=size)

    manifest = {
        "dataset_version": DATASET_VERSION,
        "scene_version": SCENE_VERSION,
        "seed": seed,
        "episode_seeds": episode_seeds,
        "episode_kinds": episode_kinds,
        "episodes": episodes,
        "steps": steps,
        "record_hz": hz,
        "rendered": render,
        "render_size": size if render else None,
        "git_describe": _git_describe(),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return GenerateArticulatedResult(out_dir=out_dir, manifest_path=manifest_path, episode_paths=episode_paths)
