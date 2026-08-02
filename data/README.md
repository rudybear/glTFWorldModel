# data/

This directory is gitignored (`.gitignore`: `/data/*` + `!/data/README.md`)
-- generated datasets live here locally but are never committed. This file
*is* committed so the datasets are always reproducible from source.

`dynamics-v1`/`perception-v1` below (the `wm-scenes-v1` flat-object
distribution, V4) were generated from commit
`27dea4be734e2bafcee8d1e0cff4a09f82acdfef` ("V3.1: cylinder Y-axis
convention fix (KHR interop) + doc corrections"), the last commit before
that milestone's own changes (`git describe --always --dirty` recorded
`27dea4b-dirty` at generation time -- the dataset/pack/stats/metrics code
added by that milestone does not change episode generation itself, only
what's done with episodes afterwards, so regenerating against that
milestone's own final commit reproduces bit-identical episodes).
`articulated-v1` (the `wm-articulated-v1` door/drawer distribution, V9) has
its own generation commit recorded in its own section below.

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

## `perception-v1` -- originally 500 episodes; regenerated at 4,000 (V6.1)

**V4's original generation** (numbers below are this original run):

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

**V6.1 regenerated `perception-v1` at production scale** (4,000 episodes,
same base seed `20260728` so the original 500 are a strict prefix), after
the 500-episode dataset was found too small (69.9x epoch-equivalent over
25k steps caused memorization rather than generalization -- see DESIGN.md's
"V6.1 postmortem"):

```bash
uv run gltfworld generate \
  --out data/perception-v1/episodes \
  --episodes 4000 --seed 20260728 --steps 100 --hz 30 \
  --render --size 256
uv run gltfworld pack data/perception-v1/episodes --out data/perception-v1/packed/perception-v1.safetensors
```

This is the dataset every reported V6-V7 result (docs/RESULTS.md) actually
trains/evaluates against -- not the original 500-episode run above, which
is kept documented here only for provenance/reproducibility of the
memorization postmortem itself. Exact wall-time/disk numbers for the
4,000-episode regeneration were not separately recorded in this file at
generation time; DESIGN.md's V6.1/V6.2 sections have the full narrative
(epoch-equivalent guard, out-of-box GT filter) this regeneration fed into.

## `articulated-v1` -- 1,500 episodes (750 door / 750 drawer), WITH rendered frames (V9)

```bash
uv run gltfworld generate-articulated \
  --out data/articulated-v1/episodes \
  --episodes 1500 --seed 20260730 --steps 100 --hz 30 \
  --render --size 256
uv run gltfworld pack-articulated data/articulated-v1/episodes --out data/articulated-v1/packed/articulated-v1.safetensors
```

Generated from commit `54e818e-dirty` (the V9-prep merge commit this
milestone's own changes are built on top of; `git describe --always
--dirty` recorded at generation time).

- Wall time: 318.1s (5.30 min) to generate + render 150,000 frames (1,500
  episodes x 100 steps); 16.1s to pack (the packed tensors here are just
  each episode's own `ArticulatedSpec`/`joint_pos`/camera -- see
  `gltfworld.data.pack_articulated.pack_articulated_dataset`'s docstring --
  not a general per-object tensor contract, so packing is much cheaper than
  `dynamics-v1`/`perception-v1`'s).
- Disk: ~65G (rendered rgb+seg+depth frame stacks dominate, same pattern as
  `perception-v1`).
- Split (same `sha256`-bucketing scheme as `gltfworld.data.pack
  .split_id_for_seed`, keyed by each episode's own seed): train 1,384 /
  val 64 / test 52.
- Joint type: exactly 750 revolute (door) / 750 prismatic (drawer) --
  `generate_articulated_dataset` alternates `kind` by episode index for an
  exact 50/50 mix, not a statistically-close-to-50/50 random draw.
- Axis distribution: X 527 / Y 474 / Z 499 (close to the sampler's uniform
  `{0,1,2}` draw, as expected).
- Source manifest sha256:
  `92b8c8cdeb22c1b2f68f9d9c7a67c1a7f53e044cce4a5054bc81e863bdb726aa`

## `dynamics-v1`/`perception-v1` (`wm-scenes-v1`)

- `--steps 100 --hz 30` (~3.3s/episode) is long enough that a real (small)
  fraction of episodes have an object roll/bounce off `wm-scenes-v1`'s
  *finite* 6m x 6m ground plate and free-fall indefinitely afterwards --
  a known, documented scope boundary (DESIGN.md "Finite ground plate"),
  not a bug. `gltfworld stats` excludes those off-plate frames from its
  ground-penetration checks and reports the departure rate separately
  (measured: ~14.0% of `dynamics-v1` episodes, ~12.6% of `perception-v1`
  episodes have >=1 object leave the plate at some point -- see
  `docs/PRETRAINING_GATE.md` for the full numbers).

## All three datasets

Regenerating any of the commands above from the same commit and the same
`--seed` reproduces bit-identical `.glb` episodes (both `wm-scenes-v1`'s and
`wm-articulated-v1`'s samplers, and the MuJoCo simulation underneath both,
are fully deterministic given a seed, see DESIGN.md); packing is a pure
function of the episodes plus the relevant pack module's fixed padding/
split logic, so the packed `.safetensors` files (and the hashes/stats
above) reproduce identically too.
