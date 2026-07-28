"""``gltfworld generate``: sample ``wm-scenes-v1`` scenes, simulate them in
MuJoCo, and write each as a GLB episode through the existing transport
codec (``gltfworld.scene.convert.save_episode``) -- no new encoding, no ML,
no dataset packing (those are later milestones).

Writes ``ep_{i:06d}.glb`` per episode plus one ``manifest.json`` describing
the whole run (dataset/scene version, base seed, per-episode seeds, dt, T,
git revision).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from gltfworld.datagen.mujoco_env import simulate
from gltfworld.datagen.sample import SCENE_VERSION, sample_scene
from gltfworld.scene.convert import save_episode
from gltfworld.scene.episode import Episode

DATASET_VERSION = "wm-scenes-v1"


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
class GenerateResult:
    out_dir: Path
    manifest_path: Path
    episode_paths: list[Path]


def generate_dataset(
    out_dir: str | Path,
    episodes: int,
    seed: int,
    *,
    steps: int = 100,
    hz: float = 30.0,
    render: bool = False,
    size: int = 256,
) -> GenerateResult:
    """Generate ``episodes`` ``wm-scenes-v1`` episodes seeded from ``seed``
    (episode ``i`` uses ``seed + i``, recorded in the manifest so any single
    episode is independently reproducible from its own seed) into
    ``out_dir``.

    ``render=True`` additionally renders each episode's frames via
    ``gltfworld.render.renderer.render_episode`` into an ``ep_{i:06d}/``
    subdirectory next to its GLB (needs the ``render`` extra + a working
    GPU/EGL context -- not exercised by any non-gpu test).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_paths: list[Path] = []
    episode_seeds: list[int] = []

    for i in range(episodes):
        episode_seed = seed + i
        episode_seeds.append(episode_seed)

        sampled = sample_scene(episode_seed)
        series = simulate(
            sampled.scene,
            sampled.initial_poses,
            T=steps,
            record_hz=hz,
            initial_lin_vel=sampled.initial_lin_vel,
            initial_ang_vel=sampled.initial_ang_vel,
        )
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
        "episodes": episodes,
        "steps": steps,
        "record_hz": hz,
        "rendered": render,
        "render_size": size if render else None,
        "git_describe": _git_describe(),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return GenerateResult(out_dir=out_dir, manifest_path=manifest_path, episode_paths=episode_paths)
