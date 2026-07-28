# data/

This directory is gitignored (`.gitignore`: `/data/*` + `!/data/README.md`)
-- generated datasets live here locally but are never committed. This file
*is* committed so the datasets are always reproducible from source.

Both datasets below were generated from commit `27dea4be734e2bafcee8d1e0cff4a09f82acdfef`
("V3.1: cylinder Y-axis convention fix (KHR interop) + doc corrections"),
the last commit before this milestone's own changes (`git describe --always
--dirty` recorded `27dea4b-dirty` at generation time -- the dataset/pack/
stats/metrics code added by this milestone does not change episode
generation itself, only what's done with episodes afterwards, so
regenerating against this milestone's own final commit reproduces bit
-identical episodes).

## `dynamics-v1` -- 10,000 episodes, states only, no rendered frames

```bash
uv run gltfworld generate \
  --out data/dynamics-v1/episodes \
  --episodes 10000 --seed 20260727 --steps 100 --hz 30
uv run gltfworld pack data/dynamics-v1/episodes --out data/dynamics-v1/packed/dynamics-v1.safetensors
```

- Wall time: 271.91s (4.53 min) to generate; 115.18s to pack.
- Disk: episodes 636M (10,000 `.glb` files + `manifest.json`); packed
  safetensors 421M (`dynamics-v1.safetensors`) + a 161K `pack_meta.json`
  sidecar (see `gltfworld.data.pack.pack_dataset`'s docstring for the exact
  tensor layout/padding/split scheme).
- Split (90/5/5 by `SceneState.seed` hash bucketing, see
  `gltfworld.data.pack.split_id_for_seed`): train 8992 / val 532 / test 476.
- Source manifest sha256 (`data/dynamics-v1/episodes/manifest.json`, also
  recorded in `data/dynamics-v1/packed/dynamics-v1.pack_meta.json`'s
  `source_manifest_hash_sha256`):
  `b6e8c86c4ce66e83f0e490bc44faa6889f211e7c1ab8d571985934e57a13a516`

## `perception-v1` -- 500 episodes, WITH rendered frames (256x256 rgb+seg+depth)

```bash
uv run gltfworld generate \
  --out data/perception-v1/episodes \
  --episodes 500 --seed 20260728 --steps 100 --hz 30 \
  --render --size 256
uv run gltfworld pack data/perception-v1/episodes --out data/perception-v1/packed/perception-v1.safetensors
```

- Wall time: 127.87s (2.13 min) to generate + render (~391 frames/s across
  generation+render combined, on this machine's RTX PRO 6000 Blackwell);
  5.59s to pack (packing only touches the tensor contract, not the
  rendered frames themselves).
- Disk: episodes+frames 22G (500 episodes, each with `ep_{i:06d}.glb` +
  an `ep_{i:06d}/` directory of `rgb.npy` (T,256,256,3 uint8) / `seg.npy`
  (T,256,256 uint16) / `depth.npy` (T,256,256 float16) / `frame_000.png`);
  packed safetensors 22M + a 9K `pack_meta.json` (the packed tensors are
  states/mask/class_ids/globals only -- rendered frames are read directly
  from each episode's own `rgb.npy`/etc by `gltfworld.data.dataset.PerceptionDataset`,
  memory-mapped, not duplicated into the packed file).
- Split: train 458 / val 27 / test 15.
- Source manifest sha256:
  `d2bc671d8a5bbb0f48cd82107e6728c8f494cc56535f51be1d872839204af5a0`

## Both datasets

- `--steps 100 --hz 30` (~3.3s/episode) is long enough that a real (small)
  fraction of episodes have an object roll/bounce off `wm-scenes-v1`'s
  *finite* 6m x 6m ground plate and free-fall indefinitely afterwards --
  a known, documented scope boundary (DESIGN.md "Finite ground plate"),
  not a bug. `gltfworld stats` excludes those off-plate frames from its
  ground-penetration checks and reports the departure rate separately
  (measured: ~14.0% of `dynamics-v1` episodes, ~12.6% of `perception-v1`
  episodes have >=1 object leave the plate at some point -- see
  `docs/PRETRAINING_GATE.md` for the full numbers).
- Regenerating with the commands above from the same commit and the same
  `--seed` reproduces bit-identical `.glb` episodes (`wm-scenes-v1`'s
  sampler and MuJoCo simulation are both fully deterministic given a seed,
  see DESIGN.md); packing is a pure function of the episodes plus
  `gltfworld.scene.contract`/`gltfworld.data.pack`'s fixed padding/split
  logic, so the packed `.safetensors` files (and the hashes/stats above)
  reproduce identically too.
