"""Torch ``Dataset`` classes over a packed dataset (see ``gltfworld.data.pack``).

``DynamicsDataset`` reads only the packed safetensors file (states/mask/
class_ids/globals/split_id) -- it needs no rendered frames. ``PerceptionDataset``
additionally needs the *rendered* frame stacks (``rgb.npy``/``seg.npy``/
``depth.npy``) written by ``gltfworld generate --render`` next to each
episode's GLB (``ep_{i:06d}/rgb.npy`` etc, see
``gltfworld.datagen.generate.generate_dataset``); it reads them
memory-mapped (``np.load(..., mmap_mode="r")``) rather than loading every
episode's frames into RAM up front. ``ArticulationDataset`` (V9) is the same
"one item per rendered frame, memory-mapped rgb" pattern applied to
``gltfworld.data.pack_articulated``'s simpler single-joint-per-episode
packed tensors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch.utils.data import Dataset

from gltfworld.data.pack import SPLIT_NAMES


def _load_packed(pack_file: str | Path) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(str(pack_file), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def _split_indices(split_id: torch.Tensor, split: str | None) -> torch.Tensor:
    if split is None:
        return torch.arange(split_id.shape[0])
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES} or None, got {split!r}")
    target = SPLIT_NAMES.index(split)
    return torch.nonzero(split_id == target, as_tuple=True)[0]


class DynamicsDataset(Dataset):
    """Dynamics-model training data from a packed dataset.

    ``mode="transition"`` (default): one item per ``(episode, t)`` pair with
    ``t`` in ``[0, T-2]``, returning ``(state_t, state_t+1, mask, globals)``
    -- ``state_t``/``state_t+1`` are ``(N_max, D)`` float32, ``mask`` is
    ``(N_max,)`` bool, ``globals`` is ``(G,)`` float32.

    ``mode="sequence"``: one item per episode, returning ``(states, mask,
    globals)`` with ``states`` the full ``(T, N_max, D)`` sequence.
    """

    def __init__(self, pack_file: str | Path, split: str | None = None, mode: str = "transition") -> None:
        if mode not in ("transition", "sequence"):
            raise ValueError(f"mode must be 'transition' or 'sequence', got {mode!r}")
        self.mode = mode
        self.pack_file = Path(pack_file)

        tensors = _load_packed(self.pack_file)
        indices = _split_indices(tensors["split_id"], split)

        self.states = tensors["states"][indices]
        self.mask = tensors["mask"][indices]
        self.globals = tensors["globals"][indices]
        self.episode_indices = indices

        self.num_episodes, self.t, self.n_max, self.d = self.states.shape

        if mode == "transition":
            self._pairs = [(ep, t) for ep in range(self.num_episodes) for t in range(self.t - 1)]

    def __len__(self) -> int:
        if self.mode == "sequence":
            return self.num_episodes
        return len(self._pairs)

    def __getitem__(self, idx: int):
        if self.mode == "sequence":
            return self.states[idx], self.mask[idx], self.globals[idx]

        ep, t = self._pairs[idx]
        return self.states[ep, t], self.states[ep, t + 1], self.mask[ep], self.globals[ep]


class PerceptionDataset(Dataset):
    """Perception-model training data: one item per rendered frame.

    ``episodes_dir`` must be the same directory ``pack_dataset`` packed
    (episode order is re-derived by the same ``sorted(glob("ep_*.glb"))``
    used by ``pack_dataset``, so packed-tensor rows line up with
    ``ep_{i:06d}/`` render directories by position). Only episodes that
    actually have a rendered ``ep_{i:06d}/rgb.npy`` are included (a packed
    dataset generated with ``--no-render`` for some/all episodes simply
    contributes no frames).

    Returns ``(rgb, state, mask, class_ids)``:

    - ``rgb``: ``(H, W, 3)`` float32 in ``[0, 1]`` (cast up from the stored
      uint8 frame)
    - ``state``: ``(N_max, D)`` float32, the object states at that frame
    - ``mask``: ``(N_max,)`` bool
    - ``class_ids``: ``(N_max,)`` int64
    """

    def __init__(self, episodes_dir: str | Path, pack_file: str | Path, split: str | None = None) -> None:
        self.episodes_dir = Path(episodes_dir)
        episode_paths = sorted(self.episodes_dir.glob("ep_*.glb"))

        tensors = _load_packed(pack_file)
        split_indices = set(_split_indices(tensors["split_id"], split).tolist())

        self.states = tensors["states"]
        self.mask = tensors["mask"]
        self.class_ids = tensors["class_ids"]

        self._rgb_mmaps: dict[int, np.ndarray] = {}
        self._index: list[tuple[int, int]] = []  # (episode_idx, frame_idx)

        for episode_idx, glb_path in enumerate(episode_paths):
            if episode_idx not in split_indices:
                continue
            rgb_path = glb_path.parent / glb_path.stem / "rgb.npy"
            if not rgb_path.exists():
                continue
            rgb = np.load(rgb_path, mmap_mode="r")
            self._rgb_mmaps[episode_idx] = rgb
            for frame_idx in range(rgb.shape[0]):
                self._index.append((episode_idx, frame_idx))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        episode_idx, frame_idx = self._index[idx]
        rgb_u8 = np.asarray(self._rgb_mmaps[episode_idx][frame_idx])  # copy out of the mmap
        rgb = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0)
        state = self.states[episode_idx, frame_idx]
        mask = self.mask[episode_idx]
        class_ids = self.class_ids[episode_idx]
        return rgb, state, mask, class_ids


class ArticulationDataset(Dataset):
    """V9 articulation-state training data: one item per rendered frame of a
    ``wm-articulated-v1`` episode (see ``gltfworld.data.pack_articulated``).

    Mirrors ``PerceptionDataset``'s "one item per rendered frame,
    memory-mapped rgb" pattern, but the packed tensors are the simpler
    single-joint-per-episode set ``pack_articulated_dataset`` writes (no
    ``N_max`` object padding -- there is exactly one joint per episode).

    ``episodes_dir`` must be the same directory ``pack_articulated_dataset``
    packed (episode order is re-derived by the same
    ``sorted(glob("ep_*.glb"))``, so packed-tensor rows line up with
    ``ep_{i:06d}/`` render directories by position). Only episodes that
    actually have a rendered ``ep_{i:06d}/rgb.npy`` are included.

    Returns ``(rgb, joint_pos_norm, joint_type_id, axis, limit_min, limit_max)``:

    - ``rgb``: ``(H, W, 3)`` float32 in ``[0, 1]``
    - ``joint_pos_norm``: scalar float32, ``(joint_pos - limit_min) /
      (limit_max - limit_min)`` -- the regression target (see
      ``gltfworld.models.articulation`` for why normalized-by-range rather
      than raw radians/meters)
    - ``joint_type_id``: scalar int64, 0=revolute (hinge) / 1=prismatic (slider)
    - ``axis``: ``(3,)`` float32 unit vector (one of the world X/Y/Z basis vectors)
    - ``limit_min``/``limit_max``: scalars float32, raw units (radians or
      meters) -- kept alongside the normalized target so eval can denormalize
      predictions back into real, reportable units without re-reading the GLB
    """

    def __init__(self, episodes_dir: str | Path, pack_file: str | Path, split: str | None = None) -> None:
        self.episodes_dir = Path(episodes_dir)
        episode_paths = sorted(self.episodes_dir.glob("ep_*.glb"))

        tensors = _load_packed(pack_file)
        split_indices = set(_split_indices(tensors["split_id"], split).tolist())

        self.joint_pos = tensors["joint_pos"]
        self.joint_type_id = tensors["joint_type_id"]
        self.axis = tensors["axis"]
        self.limit_min = tensors["limit_min"]
        self.limit_max = tensors["limit_max"]

        self._rgb_mmaps: dict[int, np.ndarray] = {}
        self._index: list[tuple[int, int]] = []  # (episode_idx, frame_idx)

        for episode_idx, glb_path in enumerate(episode_paths):
            if episode_idx not in split_indices:
                continue
            rgb_path = glb_path.parent / glb_path.stem / "rgb.npy"
            if not rgb_path.exists():
                continue
            rgb = np.load(rgb_path, mmap_mode="r")
            self._rgb_mmaps[episode_idx] = rgb
            for frame_idx in range(rgb.shape[0]):
                self._index.append((episode_idx, frame_idx))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        episode_idx, frame_idx = self._index[idx]
        rgb_u8 = np.asarray(self._rgb_mmaps[episode_idx][frame_idx])  # copy out of the mmap
        rgb = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0)

        limit_min = self.limit_min[episode_idx]
        limit_max = self.limit_max[episode_idx]
        joint_pos = self.joint_pos[episode_idx, frame_idx]
        joint_pos_norm = (joint_pos - limit_min) / (limit_max - limit_min)

        joint_type_id = self.joint_type_id[episode_idx]
        axis = self.axis[episode_idx]
        return rgb, joint_pos_norm, joint_type_id, axis, limit_min, limit_max
