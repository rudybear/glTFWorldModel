"""``pack_dataset``: turn a directory of ``ep_*.glb`` episodes (as written by
``gltfworld generate``, see ``gltfworld.datagen.generate``) into one packed
``safetensors`` file plus a ``pack_meta.json`` sidecar, ready for
``gltfworld.data.dataset``'s torch ``Dataset`` classes and ``gltfworld stats``.

No new encoding: every episode is loaded through the real transport codec
(``gltfworld.scene.convert.load_episode``) and converted via the tensor
contract (``gltfworld.scene.contract.episode_to_tensors``) -- packing only
stacks/pads what the contract already produces across episodes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from gltfworld.datagen.sample import object_support_offset
from gltfworld.scene.contract import (
    CATEGORY_TO_CLASS_ID,
    GLOBALS_DIM,
    SHAPE_ORDER,
    STATE_DIM,
    episode_to_tensors,
)
from gltfworld.scene.convert import load_episode
from gltfworld.scene.scene import ObjectSpec

DEFAULT_N_MAX = 5

# --- deterministic 90/5/5 split ------------------------------------------------

SPLIT_NAMES = ("train", "val", "test")
_SPLIT_THRESHOLDS = (0.90, 0.95, 1.0)  # cumulative: train < 0.90 <= val < 0.95 <= test


def split_id_for_seed(episode_seed: int) -> int:
    """Deterministic 90/5/5 train(0)/val(1)/test(2) split, keyed by an
    episode's own ``SceneState.seed`` (not its position in the dataset, so
    packing order/subsetting never perturbs which split an episode lands in).

    ``sha256(f"gltfworld-split-v1:{episode_seed}")``'s first 8 hex digits,
    read as a uint32 and divided by ``2**32``, gives a value uniform in
    ``[0, 1)``; thresholded at 0.90/0.95 for train/val/test. This is a
    standard hash-bucketing scheme (deterministic, no stored state needed,
    stable under adding more episodes later).
    """
    digest = hashlib.sha256(f"gltfworld-split-v1:{episode_seed}".encode()).hexdigest()
    u = int(digest[:8], 16) / float(0x1_0000_0000)
    for split_index, threshold in enumerate(_SPLIT_THRESHOLDS):
        if u < threshold:
            return split_index
    return len(_SPLIT_THRESHOLDS) - 1  # pragma: no cover - u < 1.0 always


@dataclass
class PackResult:
    out_file: Path
    meta_path: Path
    count: int
    n_max: int
    t: int
    split_counts: dict[str, int]


def _pack_meta_path(out_file: Path) -> Path:
    return out_file.with_suffix("").with_suffix(".pack_meta.json")


def _static_top_y(static_entry: dict) -> float:
    """The world-Y height of ``static_entry``'s top surface (its highest
    point along world +Y, given its actual orientation) -- used by
    ``gltfworld stats`` for ground-penetration checks. Reuses
    ``gltfworld.datagen.sample.object_support_offset``'s orientation-aware
    "support function" (exact for the box/sphere/cylinder shapes gltfworld
    supports) rather than assuming an axis-aligned box, even though
    ``wm-scenes-v1``'s ground happens to always be one.
    """
    dummy = ObjectSpec(
        object_id=-1,
        shape=static_entry["shape"],
        size=static_entry["size"],
        color=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        roughness=1.0,
        metallic=0.0,
        mass=1.0,
        friction=0.0,
        restitution=0.0,
        is_static=True,
        category=static_entry["category"],
    )
    offset = object_support_offset(dummy, static_entry["quat"], direction_world=np.array([0.0, 1.0, 0.0]))
    return float(static_entry["pos"][1]) + offset


def _static_footprint_half_extents(static_entry: dict) -> tuple[float, float] | None:
    """``(half_extent_x, half_extent_z)`` of ``static_entry``'s horizontal
    footprint, centered on its own ``pos`` -- only meaningful (and only
    computed) for a box-shaped static object (``wm-scenes-v1``'s ground is
    always one; see DESIGN.md "Finite ground plate"). Used by
    ``gltfworld stats`` to tell "genuinely penetrating the ground" apart
    from "walked/rolled/bounced off a *finite* plate and is now free-falling
    with nothing underneath it" -- the latter is a documented scope
    boundary of ``wm-scenes-v1``, not a physics bug, and would otherwise
    show up as arbitrarily large "penetration". Returns ``None`` for a
    non-box static object, since there's no single "footprint half-extent"
    for a sphere/cylinder in this axis-aligned sense.
    """
    if static_entry["shape"] != "box":
        return None
    size = static_entry["size"]
    return float(size[0]), float(size[2])


def pack_dataset(episodes_dir: str | Path, out_file: str | Path, *, n_max: int = DEFAULT_N_MAX) -> PackResult:
    """Pack every ``ep_*.glb`` in ``episodes_dir`` into one safetensors file.

    Written tensors (padded to ``n_max`` dynamic objects along the object
    axis; ``mask`` is ``False`` on padding rows):

    - ``states``    float32 ``(E, T, n_max, D)``, ``D=22`` (see
      ``gltfworld.scene.contract``)
    - ``mask``      bool    ``(E, n_max)``
    - ``class_ids`` int64   ``(E, n_max)`` (``-1`` on padding rows)
    - ``globals``   float32 ``(E, G)``, ``G=12``
    - ``split_id``  int64   ``(E,)`` -- 0=train, 1=val, 2=test (see
      :func:`split_id_for_seed`)
    - ``seeds``     int64   ``(E,)`` -- each episode's own ``SceneState.seed``

    Every episode in ``episodes_dir`` must have the same number of frames
    ``T`` (true for any single ``gltfworld generate`` run, which uses one
    fixed ``--steps``) and no more than ``n_max`` dynamic objects (true for
    ``wm-scenes-v1``, whose sampler caps dynamic object count at 5 --
    ``DEFAULT_N_MAX``); a mismatch raises ``ValueError`` rather than
    silently truncating/reshaping.
    """
    episodes_dir = Path(episodes_dir)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    episode_paths = sorted(episodes_dir.glob("ep_*.glb"))
    if not episode_paths:
        raise ValueError(f"no ep_*.glb files found in {episodes_dir}")

    manifest_path = episodes_dir / "manifest.json"
    if manifest_path.exists():
        manifest_bytes = manifest_path.read_bytes()
        source_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes)
    else:
        source_manifest_hash = None
        manifest = None

    all_states: list[np.ndarray] = []
    all_mask: list[np.ndarray] = []
    all_class_ids: list[np.ndarray] = []
    all_globals: list[np.ndarray] = []
    all_seeds: list[int] = []
    ground_top_ys: list[float] = []
    ground_footprints: list[tuple[float, float]] = []

    t_ref: int | None = None
    for path in episode_paths:
        ep = load_episode(path)
        tensors = episode_to_tensors(ep)
        states = tensors["states"]
        mask = tensors["mask"]
        class_ids = tensors["class_ids"]

        t, n, d = states.shape
        if t_ref is None:
            t_ref = t
        elif t != t_ref:
            raise ValueError(f"{path}: T={t} != first episode's T={t_ref} (mixed --steps run?)")
        if n > n_max:
            raise ValueError(f"{path}: has {n} dynamic objects, exceeds n_max={n_max}")

        pad_n = n_max - n
        states_p = np.pad(states, ((0, 0), (0, pad_n), (0, 0)), constant_values=0.0).astype(np.float32)
        mask_p = np.pad(mask, (0, pad_n), constant_values=False)
        class_ids_p = np.pad(class_ids, (0, pad_n), constant_values=-1).astype(np.int64)

        all_states.append(states_p)
        all_mask.append(mask_p)
        all_class_ids.append(class_ids_p)
        all_globals.append(tensors["globals"])
        all_seeds.append(int(ep.scene.seed))

        for static_entry in tensors["static"].values():
            ground_top_ys.append(_static_top_y(static_entry))
            footprint = _static_footprint_half_extents(static_entry)
            if footprint is not None:
                center_x, center_z = float(static_entry["pos"][0]), float(static_entry["pos"][2])
                ground_footprints.append((center_x, center_z, footprint[0], footprint[1]))

    states_arr = np.stack(all_states, axis=0)
    mask_arr = np.stack(all_mask, axis=0)
    class_ids_arr = np.stack(all_class_ids, axis=0)
    globals_arr = np.stack(all_globals, axis=0)
    seeds_arr = np.array(all_seeds, dtype=np.int64)
    split_arr = np.array([split_id_for_seed(s) for s in all_seeds], dtype=np.int64)

    save_file(
        {
            "states": states_arr,
            "mask": mask_arr,
            "class_ids": class_ids_arr,
            "globals": globals_arr,
            "split_id": split_arr,
            "seeds": seeds_arr,
        },
        out_file,
    )

    split_counts = {name: int(np.sum(split_arr == i)) for i, name in enumerate(SPLIT_NAMES)}

    ground_top_ys_arr = np.array(ground_top_ys, dtype=np.float64) if ground_top_ys else np.zeros(0)
    ground_top_y_stats = {
        "min": float(ground_top_ys_arr.min()) if ground_top_ys_arr.size else None,
        "max": float(ground_top_ys_arr.max()) if ground_top_ys_arr.size else None,
        "mean": float(ground_top_ys_arr.mean()) if ground_top_ys_arr.size else None,
    }

    ground_footprint_stats = None
    if ground_footprints:
        fp = np.array(ground_footprints, dtype=np.float64)  # (num_static_ground_objects, 4)
        # wm-scenes-v1's ground is a single fixed box across every episode;
        # if that ever stops being true, report the (still usable) mean
        # rather than silently picking one episode's value.
        ground_footprint_stats = {
            "center_x": float(fp[:, 0].mean()),
            "center_z": float(fp[:, 1].mean()),
            "half_extent_x": float(fp[:, 2].mean()),
            "half_extent_z": float(fp[:, 3].mean()),
            "consistent_across_episodes": bool(np.allclose(fp, fp[0], atol=1e-4)),
        }

    meta = {
        "source_dir": str(episodes_dir),
        "source_manifest_hash_sha256": source_manifest_hash,
        "source_manifest": manifest,
        "count": len(episode_paths),
        "n_max": n_max,
        "d": STATE_DIM,
        "globals_dim": GLOBALS_DIM,
        "t": t_ref,
        "shape_order": list(SHAPE_ORDER),
        "category_to_class_id": CATEGORY_TO_CLASS_ID,
        "split_scheme": (
            "sha256(f'gltfworld-split-v1:{episode_seed}')[:8 hex digits] as uint32 / 2**32, "
            "thresholded at 0.90 (train) / 0.95 (val) / 1.0 (test); keyed by each episode's "
            "own SceneState.seed, not its position in the packed file"
        ),
        "split_names": list(SPLIT_NAMES),
        "split_counts": split_counts,
        "ground_top_y": ground_top_y_stats,
        "ground_footprint": ground_footprint_stats,
    }
    meta_path = _pack_meta_path(out_file)
    meta_path.write_text(json.dumps(meta, indent=2))

    return PackResult(
        out_file=out_file, meta_path=meta_path, count=len(episode_paths), n_max=n_max, t=t_ref, split_counts=split_counts
    )
