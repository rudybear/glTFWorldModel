# glTFWorldModel

A world-model proof of concept that uses **glTF 2.0 as the transport format
between simulation, rendering, and learned perception/dynamics models** —
not just as a static-asset interchange format. MuJoCo generates ground-truth
rigid-body (and articulated door/drawer) episodes; those episodes are
serialized as glTF/GLB — standard pose animation, plus rigid-body physics
and joints via draft Khronos extensions, plus a custom `RWM_state_series`
extension for velocity/action/uncertainty/joint-state time series; a
vendored, patched renderer turns glTF scenes into RGB/depth/segmentation
frames; a perception model (frame -> scene state) and a dynamics model
(state[t] -> state[t+1]) train on that data; and inference emits real glTF
at every hop, closing the loop back through the renderer.

Alongside the working pipeline, this repo produced a full **gap analysis**
of glTF as a transport for *dynamic* world state — the things core glTF and
its draft physics extensions don't express, what workaround this project
shipped for each, and what a proper Khronos-track extension would need.
See **[docs/GAP_REPORT.md](docs/GAP_REPORT.md)**, the flagship deliverable.

## Status

**V10 (final)**. All ten milestones (V0-V9.1) complete and independently
verified; see [docs/VERIFICATION.md](docs/VERIFICATION.md) for every
checkpoint and [DESIGN.md](DESIGN.md) for the full architecture writeup.

| Milestone | What it delivered |
| --- | --- |
| V0 | Project scaffold, CI, verification protocol |
| V1 | glTF transport codec: pose animation + `KHR_physics_rigid_bodies`/`KHR_implicit_shapes` + `RWM_state_series`, schema-validated |
| V2 | Headless renderer (rgb/seg/depth), MuJoCo cross-render oracle (IoU 0.9962) |
| V3 | MuJoCo episode generation, `wm-scenes-v1` distribution |
| V4 | Dataset build + provenance + stats + cross-validated metrics (pre-training gate) |
| V5 | Dynamics model (`InteractionTransformer`) + baselines, trained + evaluated |
| V6 | Perception model (`PerceptionDETR`) + Hungarian matching, trained + evaluated (honest miss) |
| V7 | Closed-loop demo + 3-arm attribution (perception vs. dynamics error) |
| V8 | Physion external anchor: HDF5 -> real glTF conversion + OCP evaluation |
| V9 / V9-prep | Articulated objects: KHR joints, trained joint-state estimator, all bars pass |
| V9.1 | EGL context-lifecycle fix (single-process GPU test suite) |
| V10 | Gap report, README, final documentation pass (this milestone) |

## Results highlights

| Result | Number | Detail |
| --- | --- | --- |
| Dynamics vs. ballistic baseline | **42x** (h=30), **176x** (h=99) | `InteractionTransformer` beats unbounded constant-gravity extrapolation past first contact — [RESULTS.md V5](docs/RESULTS.md) |
| Perception (final, honest) | existence F1 **0.870** (target 0.95), pos err **0.180m** (target 0.05m) | Data-limited, not broken: CNN encoder fixed a ViT memorization crisis; position signal is real (4.3x the mean-state baseline) — [RESULTS.md V6](docs/RESULTS.md) |
| Closed-loop (visual) vs. ballistic | **34x** better at h=99 (1.62m vs. 55.5m) | Learned dynamics keeps imperfect detections physically plausible — [RESULTS.md V7](docs/RESULTS.md) |
| Correlated-noise finding | i.i.d.-noise arm **17x worse** than the real visual arm | Real detector error is frame-correlated (lag-1 autocorr **0.55-0.82**), not i.i.d. — naive Gaussian noise models overestimate closed-loop degradation; see [docs/GAP_REPORT.md](docs/GAP_REPORT.md) G6 |
| Physion external anchor | GT-contact oracle **92.0%** held-out; our dynamics zero-shot **49.0%** (chance) | State-based OCP track, 150 real Collide trials, 100% validator-clean — honest chance-level zero-shot transfer, reported not hidden — [RESULTS.md V8](docs/RESULTS.md) |
| Articulation (joint-state estimation) | **all 4 acceptance bars pass** | Hinge 3.35°, slider 1.45cm, type acc 0.982, axis 1.84° — all vs. their respective bars — [RESULTS.md/VERIFICATION.md V9](docs/RESULTS.md) |

Every number above is independently verified (a different agent than the
one that produced it re-ran the checkpoint) — see
[docs/VERIFICATION.md](docs/VERIFICATION.md) for the full checkpoint-by-
checkpoint protocol and [docs/PRETRAINING_GATE.md](docs/PRETRAINING_GATE.md)
for the pre-training data-quality gate every model trained against.

## Documentation map

- **[docs/GAP_REPORT.md](docs/GAP_REPORT.md)** — the flagship deliverable:
  ~25 evidence-backed findings on where glTF succeeds/fails as a dynamic
  world-model transport, plus ranked recommendations for Khronos-track
  extension work.
- **[DESIGN.md](DESIGN.md)** — full architecture writeup, milestone by
  milestone, including every documented deviation from the original spec
  text and every honestly-reported "didn't meet the bar" finding.
- **[docs/RESULTS.md](docs/RESULTS.md)** — recorded, measured results for
  the trained models through V8.1, reported honestly whether or not they
  met their acceptance bar (V9 articulation numbers live in
  VERIFICATION.md's and DESIGN.md's V9 sections).
- **[docs/VERIFICATION.md](docs/VERIFICATION.md)** — the independent
  verification protocol: purpose/command/expected-result for every
  checkpoint in every milestone, with actual observed values recorded next
  to each.
- **[docs/PRETRAINING_GATE.md](docs/PRETRAINING_GATE.md)** — the 10-item
  checklist every dataset had to pass before any model-training code was
  allowed to consume it.
- **[docs/PHYSION.md](docs/PHYSION.md)** — Physion benchmark format
  reconnaissance + 14 concrete HDF5-to-glTF conversion findings (the
  primary evidence base for the gap report's Part B/C findings).
- **[docs/RWM_EXTENSIONS.md](docs/RWM_EXTENSIONS.md)** — channel-by-channel/
  field-by-field reference for the custom `RWM_state_series` extension and
  `extras.rwm` bookkeeping.
- **[docs/EXTERNAL_VALIDITY.md](docs/EXTERNAL_VALIDITY.md)** — two
  independent experiments testing this project's own claims from the
  outside: a spec-only decoder reimplementation (bitwise-identical, 6
  under-specified conventions found and fixed) and a clean-room
  reproduction from the public clone (exact-digit smoke reproduction,
  bit-identical seeded generation, onboarding friction found and fixed).

## Stack

| Piece | Role |
| --- | --- |
| [pygltflib](https://gitlab.com/dodgyville/pygltflib) | glTF/GLB read + write |
| [trimesh](https://trimesh.org/) | mesh generation only (not the scene graph or transport) |
| [MuJoCo](https://mujoco.org/) | ground-truth physics simulation, episode generation |
| vendored [pyrender](https://github.com/mmatl/pyrender) (patched, pinned commit) | headless rendering of glTF scenes |
| `KHR_physics_rigid_bodies`, `KHR_implicit_shapes` (draft, pinned commit) | rigid-body physics + colliders + joints on top of glTF |
| `RWM_state_series` (custom) | time-series world state (velocity/action/uncertainty/joint state) carried alongside pose animation |
| PyTorch | perception (`PerceptionDETR`), dynamics (`InteractionTransformer`), articulation (`ArticulationEstimator`) models |
| h5py | Physion HDF5 ground-truth ingest |

## Setup

```bash
uv sync --all-extras --dev
uv run pytest -m "not gpu"   # CI-equivalent: no GPU/EGL required
uv run pytest                # full suite, needs a GPU + working EGL offscreen context
```

### Expected results on a fresh clone

Verified by an independent clean-room reproduction from the public clone
(see [docs/EXTERNAL_VALIDITY.md](docs/EXTERNAL_VALIDITY.md), Experiment B) --
what a stranger with no local data/checkpoints actually sees:

- **Fast lane** (`uv run pytest -m "not gpu" -q`, right after `uv sync
  --all-extras --dev`, no GPU needed): **336 passed, 17 skipped
  (Physion-data-dependent), 19 deselected** (gpu-marked).
- **GPU lane** (`uv run pytest -q`, needs a GPU + working EGL context): **8
  passed, 11 skipped** until datasets/checkpoints exist locally.

What unlocks the full figures (**353 passed / 19 deselected** on the fast
lane once the Physion archive is present -- 0 skipped, since all 17
Physion-dependent skips convert to real passes; **18 passed + 1 xfail** on
the gpu lane, i.e. every gpu-marked test, once datasets + trained
checkpoints exist locally):

1. **Physion archive** (~33 GB) -- unlocks the 17 Physion-dependent fast-lane
   skips:

   ```bash
   mkdir -p data/external/physion/hdf5/extracted
   curl -o data/external/physion/hdf5/Collide_testing_HDF5s.tar.gz \
     https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/Collide_testing_HDF5s.tar.gz
   tar -xzf data/external/physion/hdf5/Collide_testing_HDF5s.tar.gz \
     -C data/external/physion/hdf5/extracted
   ```

   (see [docs/PHYSION.md](docs/PHYSION.md) for the full archive-structure
   writeup and where the `Physion.zip` "Core" tier fits in).

2. **Dataset generation** -- unlocks GPU-lane tests that need real rendered
   data (see "Reproduce everything" below for the exact `generate`/`pack`
   commands for `dynamics-v1`/`perception-v1`/`articulated-v1`).
3. **Training** -- unlocks GPU-lane tests that need real checkpoints (see
   "Reproduce everything" below for the `train_dynamics`/`train_perception`/
   `train_articulation` commands).

## Reproduce everything

Datasets (see [data/README.md](data/README.md) for exact hashes/splits):

```bash
# dynamics-v1 (10,000 episodes, states only)
uv run gltfworld generate --out data/dynamics-v1/episodes \
  --episodes 10000 --seed 20260727 --steps 100 --hz 30
uv run gltfworld pack data/dynamics-v1/episodes --out data/dynamics-v1/packed/dynamics-v1.safetensors

# perception-v1 (4,000 episodes, rendered 256x256 rgb+seg+depth)
uv run gltfworld generate --out data/perception-v1/episodes \
  --episodes 4000 --seed 20260728 --steps 100 --hz 30 --render --size 256
uv run gltfworld pack data/perception-v1/episodes --out data/perception-v1/packed/perception-v1.safetensors

# articulated-v1 (1,500 episodes, 750 door / 750 drawer, rendered)
uv run gltfworld generate-articulated --out data/articulated-v1/episodes \
  --episodes 1500 --seed 20260730 --steps 100 --hz 30 --render --size 256
uv run gltfworld pack-articulated data/articulated-v1/episodes \
  --out data/articulated-v1/packed/articulated-v1.safetensors
```

Training (each resumable via `--resume`; see `configs/`):

```bash
uv run python -m gltfworld.train.train_dynamics --config configs/dynamics_v1.json --out runs/dynamics-v1
uv run python -m gltfworld.train.train_dynamics --config configs/dynamics_mlp.json --out runs/dynamics-mlp --model mlp
uv run python -m gltfworld.train.train_perception --config configs/perception_v2_cnn_40k.json --out runs/perception-v4-cnn-40k
uv run python -m gltfworld.train.train_articulation --config configs/articulation_v1.json --out runs/articulation-v1
```

Evaluation:

```bash
uv run python -m gltfworld.eval.rollout --ckpt runs/dynamics-v1/best.safetensors \
  --data data/dynamics-v1/packed --split test --out runs/dynamics-v1/eval \
  --mlp-ckpt runs/dynamics-mlp/best.safetensors --emit-gltf 5

uv run python -m gltfworld.eval.perception_eval --ckpt runs/perception-v4-cnn-40k/best.safetensors \
  --data data/perception-v1 --split test --out runs/perception-v4-cnn-40k/eval

uv run python -m gltfworld.eval.closed_loop --episodes data/perception-v1 \
  --dyn-ckpt runs/dynamics-v1/best.safetensors \
  --per-ckpt runs/perception-v4-cnn-40k/best.safetensors \
  --per-metrics runs/perception-v4-cnn-40k/eval/metrics.json \
  --out runs/closed-loop-v1 --n-episodes 20 --video 5

uv run python -m gltfworld.eval.articulation_eval --ckpt runs/articulation-v1/best.safetensors \
  --data data/articulated-v1 --split test --out runs/articulation-v1/eval --render-samples 50
```

Physion external anchor (needs `Collide_testing_HDF5s.tar.gz`, ~33GB, see
[docs/PHYSION.md](docs/PHYSION.md) for the download URL):

```bash
uv run python -m gltfworld.physion.ocp_eval \
  --hdf5-dir data/external/physion/hdf5/extracted/Collide/hdf5s \
  --glb-dir data/external/physion/glb/Collide \
  --dynamics-ckpt runs/dynamics-v1/best.safetensors \
  --out runs/physion-ocp-v1
```

Full GPU test lane (per-module isolation, see V9.1's EGL fix):

```bash
./scripts/run_gpu_tests.sh
```

## License

MIT.
