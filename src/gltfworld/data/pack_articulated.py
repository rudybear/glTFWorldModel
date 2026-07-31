"""``pack_articulated_dataset``: turn a directory of ``ep_*.glb``
articulated episodes (as written by ``gltfworld generate-articulated``, see
``gltfworld.datagen.generate_articulated``) into one packed ``safetensors``
file plus a ``pack_meta.json`` sidecar, ready for
``gltfworld.data.dataset.ArticulationDataset``.

Unlike ``gltfworld.data.pack.pack_dataset`` (V4, built for
``wm-scenes-v1``'s general multi-object tensor contract), this pack is
purpose-built for the V9 articulation task: joint-state estimation is a
*single-joint* regression/classification problem (one door/drawer joint per
episode), not a set-of-objects one, so there's no ``N_max``-object padding,
no class/shape one-hot, no Hungarian matching -- just each episode's own
``ArticulatedSpec`` (type/axis/limits) + its recorded ``joint_pos`` time
series + its (fixed, but recorded per-episode for completeness) camera.

No new encoding here either: every episode is loaded through the real
transport codec (``gltfworld.scene.convert.load_episode``); packing only
stacks/pads what ``SceneState.articulations``/``StateSeries.joint_pos``
already carry across episodes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from gltfworld.data.pack import SPLIT_NAMES, split_id_for_seed
from gltfworld.scene.convert import load_episode

# joint_type_id: 0 = revolute (hinge, door), 1 = prismatic (slider, drawer) --
# matches gltfworld.datagen.articulated's own "door" -> revolute / "drawer"
# -> prismatic convention 1:1, so this column is redundant with (but cheaper
# to consume than) re-deriving it from ArticulatedSpec.joint_type strings at
# training time.
JOINT_TYPE_NAMES = ("revolute", "prismatic")
JOINT_TYPE_TO_ID = {name: i for i, name in enumerate(JOINT_TYPE_NAMES)}


@dataclass
class PackArticulatedResult:
    out_file: Path
    meta_path: Path
    count: int
    t: int
    split_counts: dict[str, int]
    joint_type_counts: dict[str, int]


def _unit_axis(axis: int) -> np.ndarray:
    v = np.zeros(3, dtype=np.float32)
    v[axis] = 1.0
    return v


def pack_articulated_dataset(episodes_dir: str | Path, out_file: str | Path) -> PackArticulatedResult:
    """Pack every ``ep_*.glb`` in ``episodes_dir`` (each with exactly one
    ``ArticulatedSpec``, per ``gltfworld.datagen.articulated``) into one
    safetensors file.

    Written tensors (``E`` = episode count, ``T`` = frames per episode,
    fixed across the whole directory -- same "no mixed --steps run"
    contract as ``gltfworld.data.pack.pack_dataset``):

    - ``joint_pos``      float32 ``(E, T)`` -- radians (revolute) or meters
      (prismatic), raw units, straight off ``StateSeries.joint_pos[:, 0]``
    - ``joint_type_id``  int64   ``(E,)`` -- 0=revolute, 1=prismatic
    - ``axis``           float32 ``(E, 3)`` -- unit vector, one of the world
      X/Y/Z basis vectors (``wm-articulated-v1``'s own convention, see
      ``ArticulatedSpec``'s docstring)
    - ``axis_idx``       int64   ``(E,)`` -- 0/1/2, which world axis
    - ``limit_min``      float32 ``(E,)``
    - ``limit_max``      float32 ``(E,)``
    - ``camera_pos``     float32 ``(E, 3)``
    - ``camera_rot``     float32 ``(E, 4)`` -- xyzw
    - ``camera_yfov``    float32 ``(E,)``
    - ``split_id``       int64   ``(E,)`` -- 0=train/1=val/2=test, same
      ``gltfworld.data.pack.split_id_for_seed`` scheme (keyed by each
      episode's own ``SceneState.seed``), so a train/val/test split_id is
      directly comparable in meaning across the flat and articulated
      datasets even though the packed tensors themselves are unrelated.
    - ``seeds``          int64   ``(E,)``
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

    all_joint_pos: list[np.ndarray] = []
    all_joint_type_id: list[int] = []
    all_axis: list[np.ndarray] = []
    all_axis_idx: list[int] = []
    all_limit_min: list[float] = []
    all_limit_max: list[float] = []
    all_camera_pos: list[np.ndarray] = []
    all_camera_rot: list[np.ndarray] = []
    all_camera_yfov: list[float] = []
    all_seeds: list[int] = []

    t_ref: int | None = None
    for path in episode_paths:
        ep = load_episode(path)
        if len(ep.scene.articulations) != 1:
            raise ValueError(
                f"{path}: expected exactly 1 articulation, found {len(ep.scene.articulations)} "
                "-- pack_articulated_dataset is built for wm-articulated-v1's single-joint-per-episode scenes"
            )
        if ep.series.joint_pos is None:
            raise ValueError(f"{path}: episode has an articulation but no recorded joint_pos series")

        art = ep.scene.articulations[0]
        jp = ep.series.joint_pos[:, 0]

        t = jp.shape[0]
        if t_ref is None:
            t_ref = t
        elif t != t_ref:
            raise ValueError(f"{path}: T={t} != first episode's T={t_ref} (mixed --steps run?)")

        if art.joint_type not in JOINT_TYPE_TO_ID:
            raise ValueError(f"{path}: unknown joint_type {art.joint_type!r}")

        all_joint_pos.append(jp.astype(np.float32))
        all_joint_type_id.append(JOINT_TYPE_TO_ID[art.joint_type])
        all_axis.append(_unit_axis(art.axis))
        all_axis_idx.append(int(art.axis))
        all_limit_min.append(float(art.min))
        all_limit_max.append(float(art.max))
        all_camera_pos.append(ep.scene.camera.position.astype(np.float32))
        all_camera_rot.append(ep.scene.camera.rotation.astype(np.float32))
        all_camera_yfov.append(float(ep.scene.camera.yfov))
        all_seeds.append(int(ep.scene.seed))

    joint_pos_arr = np.stack(all_joint_pos, axis=0)
    joint_type_id_arr = np.array(all_joint_type_id, dtype=np.int64)
    axis_arr = np.stack(all_axis, axis=0)
    axis_idx_arr = np.array(all_axis_idx, dtype=np.int64)
    limit_min_arr = np.array(all_limit_min, dtype=np.float32)
    limit_max_arr = np.array(all_limit_max, dtype=np.float32)
    camera_pos_arr = np.stack(all_camera_pos, axis=0)
    camera_rot_arr = np.stack(all_camera_rot, axis=0)
    camera_yfov_arr = np.array(all_camera_yfov, dtype=np.float32)
    seeds_arr = np.array(all_seeds, dtype=np.int64)
    split_arr = np.array([split_id_for_seed(s) for s in all_seeds], dtype=np.int64)

    save_file(
        {
            "joint_pos": joint_pos_arr,
            "joint_type_id": joint_type_id_arr,
            "axis": axis_arr,
            "axis_idx": axis_idx_arr,
            "limit_min": limit_min_arr,
            "limit_max": limit_max_arr,
            "camera_pos": camera_pos_arr,
            "camera_rot": camera_rot_arr,
            "camera_yfov": camera_yfov_arr,
            "split_id": split_arr,
            "seeds": seeds_arr,
        },
        out_file,
    )

    split_counts = {name: int(np.sum(split_arr == i)) for i, name in enumerate(SPLIT_NAMES)}
    joint_type_counts = {
        name: int(np.sum(joint_type_id_arr == i)) for i, name in enumerate(JOINT_TYPE_NAMES)
    }
    axis_counts = {str(i): int(np.sum(axis_idx_arr == i)) for i in range(3)}

    meta = {
        "source_dir": str(episodes_dir),
        "source_manifest_hash_sha256": source_manifest_hash,
        "source_manifest": manifest,
        "count": len(episode_paths),
        "t": t_ref,
        "joint_type_names": list(JOINT_TYPE_NAMES),
        "split_scheme": (
            "sha256(f'gltfworld-split-v1:{episode_seed}')[:8 hex digits] as uint32 / 2**32, "
            "thresholded at 0.90 (train) / 0.95 (val) / 1.0 (test); keyed by each episode's "
            "own SceneState.seed, not its position in the packed file -- same scheme as "
            "gltfworld.data.pack.split_id_for_seed"
        ),
        "split_names": list(SPLIT_NAMES),
        "split_counts": split_counts,
        "joint_type_counts": joint_type_counts,
        "axis_counts": axis_counts,
        "limit_min_range": [float(limit_min_arr.min()), float(limit_min_arr.max())],
        "limit_max_range": [float(limit_max_arr.min()), float(limit_max_arr.max())],
    }
    meta_path = out_file.with_suffix("").with_suffix(".pack_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    return PackArticulatedResult(
        out_file=out_file,
        meta_path=meta_path,
        count=len(episode_paths),
        t=t_ref,
        split_counts=split_counts,
        joint_type_counts=joint_type_counts,
    )
