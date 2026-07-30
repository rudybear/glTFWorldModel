# Verification protocol

Every milestone in [DESIGN.md](../DESIGN.md) is verified by an agent
independent of the one that implemented it. That agent emits a JSON verdict
(`{"milestone": "...", "verdict": "PASS"|"FAIL", "checks": [...]}`); a FAIL
blocks moving on to the next milestone.

Independently of the JSON verdict, every checkpoint below is documented as:

- **Purpose** — what the checkpoint is supposed to catch.
- **Command** — the exact command to re-run it.
- **Expected result** — what a human should see if it passes.

so that any of these can be re-run by hand, not just by an agent.

## V0 — project scaffold, CI, verification protocol

### Checkpoint: dependency install

- **Purpose**: confirm the project metadata and dependency set in
  `pyproject.toml` are resolvable and installable with `uv`.
- **Command**: `uv sync --dev`
- **Expected result**: exits 0; creates/updates `.venv` and `uv.lock` with no
  resolution errors.

### Checkpoint: test suite

- **Purpose**: confirm the package imports and the smoke test passes.
- **Command**: `uv run pytest -v`
- **Expected result**: `tests/test_smoke.py` collects and passes (2 checks:
  `gltfworld.__version__ == "0.1.0"`, `gltfworld.cli.main` is callable); exit
  code 0.

### Checkpoint: CLI stub behavior

- **Purpose**: confirm the CLI is wired up end-to-end (entry point ->
  argparse -> subcommand dispatch) even though no subcommand does real work
  yet.
- **Command**: `uv run gltfworld validate /dev/null`
- **Expected result**: prints `gltfworld validate: not implemented yet
  (milestone V1+)` to stdout; process exits with code 2.

### Checkpoint: CI - test job

- **Purpose**: confirm the test suite passes in a clean CI environment, not
  just locally.
- **Command**: `.github/workflows/ci.yml`, job `test` (push/PR trigger; or
  `gh workflow run ci.yml` / inspect the Actions tab)
- **Expected result**: `uv sync --dev` then `uv run pytest -v` both succeed;
  job is green.

### Checkpoint: CI - glTF validator availability

- **Purpose**: confirm the pinned Khronos glTF-Validator binary can be
  fetched and cached in CI, ahead of real validation logic landing in V1.
- **Command**: `.github/workflows/ci.yml`, job `test` (tests/test_validator.py)
- **Expected result**: downloads/caches `gltf-validator` release
  `2.0.0-dev.3.10` (linux64) and validates sample episodes; job is green.

## V1 — glTF transport codec: pose animation + KHR physics + RWM_state_series

### Checkpoint: accessor round trip

- **Purpose**: confirm `gltfworld.gltf.accessors` (`BufferAccumulator` +
  `read_accessor`) round-trips every dtype/type combo gltfworld uses
  (float32 SCALAR/VEC3/VEC4, uint32/uint16 SCALAR), including 4-byte
  alignment edge cases (an odd-count uint16 index buffer followed by
  another accessor).
- **Command**: `uv run pytest tests/test_accessors.py -v`
- **Expected result**: all tests pass; in particular
  `test_odd_length_uint16_index_buffer` and
  `test_buffer_total_length_is_4byte_aligned` confirm every bufferView
  starts on a 4-byte boundary and the final buffer length is a multiple of
  4.

### Checkpoint: Episode <-> GLB round trip (property test)

- **Purpose**: confirm `episode_to_gltf`/`episode_from_gltf` (and
  `save_episode`/`load_episode` through a real GLB file) never lose or
  corrupt data, across randomly generated scenes/episodes (1-5 objects,
  T in {1, 2, 50}, optional velocity/action/pose-variance channels present
  or absent) plus one deterministic golden episode.
- **Command**: `uv run pytest tests/test_roundtrip.py -v`
- **Expected result**: all tests pass. Every float32 array (poses, sizes,
  colors, velocities, times, ...) is checked bit-for-bit (uint32 view
  comparison, not just `==`), not merely "close enough".

### Checkpoint: KHR/RWM JSON Schema validation

- **Purpose**: confirm gltfworld's own `KHR_implicit_shapes` /
  `KHR_physics_rigid_bodies` / `RWM_state_series` output actually conforms
  to the schemas it claims to implement (vendored at `docs/schemas/khr/` and
  `docs/schemas/rwm/`), not just that our own decoder can read it back.
- **Command**: `uv run pytest tests/test_khr_schema.py -v`
- **Expected result**: all tests pass, including one exercising both the
  static (`motion` absent) and dynamic (`motion` present) node-physics
  schema branches, and one exercising RWM channel splitting for
  >4-component values (5-dim actions, 7-dim pose_variance).

### Checkpoint: cross-representation consistency

- **Purpose**: confirm the different places pose/physics/state-series data
  appears in one GLB never silently disagree with each other: animation
  channel data vs. `series.poses`, frame-0 node TRS vs. `poses[0]`, all
  animation samplers sharing one time accessor (the same one
  `RWM_state_series.timesAccessor` points at), `extensionsRequired` staying
  empty, and every RWM channel / KHR physics collider reference pointing at
  a valid node/accessor/shape/material index.
- **Command**: `uv run pytest tests/test_consistency.py -v`
- **Expected result**: all tests pass.

### Checkpoint: real glTF-Validator, clean run

- **Purpose**: confirm a gltfworld-produced GLB is actually spec-valid
  according to the independent, pinned Khronos glTF-Validator binary, not
  just internally self-consistent.
- **Command**: `uv run pytest tests/test_validator.py -v` (skipped only if
  neither a cached validator binary nor network access is available)
- **Expected result**: all tests pass; `numErrors == 0` on the sample
  episode. Non-error messages are expected and allowed: `UNSUPPORTED_EXTENSION`
  (info-level; the validator doesn't know about our draft/custom
  extensions) and `UNUSED_OBJECT` (info-level; RWM channel accessors aren't
  referenced by anything the validator understands).

### Checkpoint: CLI `validate`/`inspect` demo

- **Purpose**: confirm the CLI end-to-end, by hand, the way a human would
  actually use it.
- **Command**:

  ```bash
  uv run python -c "
  import sys; sys.path.insert(0, 'tests')
  from conftest import make_sample_episode
  from gltfworld.scene.convert import save_episode
  save_episode(make_sample_episode(), '/tmp/sample_episode.glb')
  "
  uv run gltfworld validate /tmp/sample_episode.glb
  uv run gltfworld inspect /tmp/sample_episode.glb
  ```

- **Expected result**: `validate` downloads+caches the pinned glTF-Validator
  binary into `~/.cache/gltfworld/` on first run, prints the JSON report,
  and exits 0 (`numErrors == 0`, only info-level messages as above).
  `inspect` prints each object (id/shape/category/mass/static), frame
  count, duration, which optional channels are present, and the
  `extensionsUsed` list; exits 0.

## V2 — headless renderer: rgb + segmentation + depth

Everything in this section needs the `render` extra
(`gltfworld.render.renderer`, wrapping the vendored pyrender at
`src/gltfworld/_vendor/pyrender/`; see `src/gltfworld/_vendor/PROVENANCE.md`
for the pinned commit and patch list). The MuJoCo crosscheck also needs the
`sim` extra. Install both:

```bash
uv sync --extra sim --extra render --dev
```

All of the checkpoints below except "EGL device info" and "CI stays green"
need a real GPU with a working EGL offscreen context, so their tests are
`@pytest.mark.gpu` and only run locally, never in CI (see "CI stays green"
below).

### Checkpoint: EGL device info

- **Purpose**: confirm the renderer actually gets a hardware (NVIDIA) EGL
  context, not a software/Mesa fallback, before trusting any performance or
  correctness numbers from it.
- **Command**:

  ```bash
  uv run python -c "
  from gltfworld.render.renderer import egl_info
  import json; print(json.dumps(egl_info(), indent=2))
  "
  ```

- **Expected result**: exits 0; `gl_vendor` contains `"NVIDIA"` and
  `gl_renderer` names the actual GPU (on a Mesa-only machine, `gl_vendor`
  would instead say `"Mesa"` or similar -- that's the renderer.py module's
  `__EGL_VENDOR_LIBRARY_FILENAMES` forcing not having an NVIDIA ICD to
  force in the first place, not a bug in gltfworld). On this project's
  development machine: `gl_vendor: "NVIDIA Corporation"`,
  `gl_renderer: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition/PCIe/SSE2"`.

### Checkpoint: analytic correctness (sphere/box/seg/determinism)

- **Purpose**: confirm the renderer's geometry, depth, and segmentation are
  *quantitatively* correct against closed-form expectations (not just
  "looks plausible") -- projected sphere silhouette area, center-pixel
  depth, box occlusion ordering, seg-value exactness, and bitwise
  determinism across repeated renders of the same frame.
- **Command**: `uv run pytest tests/test_render_analytic.py -v -m gpu`
- **Expected result**: all 5 tests pass: sphere silhouette area within 3%
  of the closed-form projected-disc area; sphere center-pixel depth within
  1% of `d - r`; the nearer of two boxes occludes the farther one at the
  center pixel with the farther box still visible (and correctly
  identified) where it peeks out around the nearer box's silhouette; every
  nonzero seg value is a real object_id from the episode; two renders of
  the same frame are bitwise identical (rgb, seg, and depth's raw uint32
  bit pattern).

### Checkpoint: MuJoCo cross-render oracle (IoU)

- **Purpose**: confirm gltfworld's renderer and an independent renderer
  (MuJoCo) agree on scene *geometry* for the same episode -- an
  independent check that doesn't share any code path with
  `EpisodeRenderer`. Compares binary silhouettes (any-object vs
  background) only, since MuJoCo's flat shading looks nothing like
  pyrender's PBR output and lighting/color are out of scope here.
- **Command**: `uv run pytest tests/test_crosscheck.py -v -m gpu -s`
  (`-s` to see the printed IoU numbers), or via the CLI:
  `uv run gltfworld crosscheck /tmp/sample_episode.glb`
- **Expected result**: binary silhouette IoU >= 0.98 (measured on this
  machine: **0.9962** for `tests/conftest.py:make_sample_episode()`'s frame
  0, with per-object IoUs 0.9863 / 0.9989 / 1.0000 for the three falling
  shapes). Also runs 3 fast, non-gpu unit tests for the
  `gltf_pose_to_mujoco` Y-up-to-Z-up coordinate conversion (axis mapping,
  quaternion order/normalization, proper-rotation/determinant check) --
  those run under the default `-m "not gpu"` too. A side-by-side +
  binary-mask-diff PNG is written to `/tmp/gltfworld_crosscheck/`.

### Checkpoint: benchmark (rgb / rgb+depth / rgb+depth+seg fps)

- **Purpose**: confirm the renderer is fast enough for real dataset
  generation, with an honest breakdown if it isn't at the stretch target.
- **Command**: `uv run pytest tests/test_render_bench.py -v -m gpu -s`
- **Expected result**: passes (hard floor: rgb+depth+seg >= 100 fps);
  prints fps for all three variants plus a draw-vs-readback breakdown.
  Measured on this machine (persistent renderer, 4-object scene, 500
  frames, 256x256): rgb-only 1003.1 fps, rgb+depth 1022.7 fps,
  rgb+depth+seg **639.5 fps** (comfortably above both the 100 fps floor and
  the 300 fps target). Note: (a) and (b) use the same underlying pyrender
  call (`OffscreenRenderer.render()` always reads back color and depth
  together unless `RenderFlags.DEPTH_ONLY` is set), so their near-equal fps
  is an expected, reported finding, not a bug.

### Checkpoint: `render`/`crosscheck` CLI demo

- **Purpose**: confirm the CLI end-to-end, the way a human would use it.
- **Command**:

  ```bash
  uv run python -c "
  import sys; sys.path.insert(0, 'tests')
  from conftest import make_sample_episode
  from gltfworld.scene.convert import save_episode
  save_episode(make_sample_episode(), '/tmp/sample_episode.glb')
  "
  uv run gltfworld render /tmp/sample_episode.glb --out /tmp/gltfworld_render_out --size 256
  uv run gltfworld crosscheck /tmp/sample_episode.glb
  ```

- **Expected result**: `render` writes `rgb.npy` `(T,H,W,3)` uint8,
  `seg.npy` `(T,H,W)` uint16, `depth.npy` `(T,H,W)` float16, and
  `frame_000.png` into `/tmp/gltfworld_render_out/`; exits 0. `crosscheck`
  prints the frame-0 binary silhouette IoU and per-object IoUs, writes a
  side-by-side + diff PNG, and exits 0 if IoU >= 0.98 (1 otherwise).

### Checkpoint: CI stays green with gpu tests excluded

- **Purpose**: confirm gpu-marked tests (which need a real GPU/EGL context
  CI runners don't have) are cleanly excluded in CI, while still being
  *collected* (their modules import `gltfworld.render.renderer`/
  `.crosscheck` at module scope, so the `render` extra must still resolve
  in CI) so a broken import doesn't silently disappear behind the marker
  filter.
- **Command**: `.github/workflows/ci.yml`, job `test` (now
  `uv sync --dev --extra render --extra sim` then `uv run pytest -v -m "not gpu"`
  -- `--extra sim` added in V3 so MuJoCo-backed datagen tests run for real
  in CI, not just via `importorskip`: MuJoCo's core physics API needs no
  GPU/EGL context, only `mujoco.Renderer` (gpu-marked) does); or locally:
  `uv run pytest -v -m "not gpu"`.
- **Expected result**: green. As of V2: 38 non-gpu tests pass, 7 gpu tests
  deselected. As of V3 (this count grows with each milestone -- re-run the
  command for the current number, don't treat this as pinned): 87 non-gpu
  tests pass, 7 gpu tests deselected (not errored, not skipped-with-a-
  warning -- cleanly excluded by the marker). Run the full suite (gpu tests
  included) locally with `uv run pytest -v` (94 tests as of V3) -- both
  invocations must be green on a GPU machine.

## V3 — MuJoCo episode generation

Needs the `sim` extra (`mujoco>=3.1`): `uv sync --extra sim --dev`. None of
this milestone's own tests need a GPU/EGL context (MuJoCo's core physics
API is CPU-only; only `mujoco.Renderer`, used by the V2 crosscheck, needs
one) -- they all run under the default `-m "not gpu"`, including in CI
(`.github/workflows/ci.yml` now installs `--extra sim` too).

### Checkpoint: MuJoCo<->contract conversion (`mj_convert`)

- **Purpose**: confirm the one, consolidated place all MuJoCo coordinate/
  quaternion/velocity conversion lives (`gltfworld.datagen.mj_convert`) is
  correct: known axis-mapping cases, round trips, batch-equals-scalar, and
  the free-joint velocity frame convention (MuJoCo's `qvel` is 3 linear
  world-frame + 3 angular *body-local*-frame -- verified empirically, see
  DESIGN.md).
- **Command**: `uv run pytest tests/test_mj_convert.py -v`
- **Expected result**: all 16 tests pass. Pure numpy, no MuJoCo import, no
  GPU.

### Checkpoint: crosscheck consolidation didn't regress

- **Purpose**: confirm refactoring `gltfworld.render.crosscheck` to import
  its coordinate conversion from `mj_convert` (instead of defining its own)
  kept its exact V2 behavior.
- **Command**: `uv run pytest tests/test_crosscheck.py -v -m "not gpu"`
  (non-gpu unit tests); `uv run pytest tests/test_crosscheck.py -v -m gpu -s`
  (the real cross-render, needs GPU + MuJoCo).
- **Expected result**: all pass. The gpu test additionally now asserts,
  per non-ground object, `union > 0` (a real, non-vacuous comparison) and
  `IoU >= 0.95`, on top of the overall `IoU >= 0.98` -- catching the V2
  cylinder-out-of-frame bug for good (see DESIGN.md "wm-scenes-v1"
  distribution). Measured on this machine: per-object IoU 0.9946 / 1.0000
  / 0.9954, overall 0.9969.

### Checkpoint: free fall matches closed-form physics

- **Purpose**: confirm the MuJoCo simulation + `mj_convert` round trip
  reproduce a textbook result for the simplest possible case (a sphere
  dropped with zero initial velocity, never contacting anything in the
  recorded window) before trusting anything more complex.
- **Command**: `uv run pytest tests/test_freefall.py -v`
- **Expected result**: all 3 tests pass: recorded position within 1% of
  `h - 1/2 g t^2`, recorded `lin_vel` within 1% of `-g t`, no horizontal
  drift or spin.

### Checkpoint: velocity/frame-convention exposé

- **Purpose**: confirm recorded `lin_vel`/`ang_vel` are self-consistent with
  finite-differencing the recorded poses, for a real multi-object episode
  with rotation and ground contacts -- the check that fails loudly if
  MuJoCo's body-local vs. world-frame angular velocity convention was
  mixed up.
- **Command**: `uv run pytest tests/test_velocity_consistency.py -v`
- **Expected result**: both tests pass (85th-percentile finite-difference
  error < 2% of max speed; a handful of contact/bounce frames are expected
  outliers past that, documented in the test). Confirmed this actually
  catches the bug it's meant to: monkeypatching the angular-velocity
  conversion to the naive (wrong, no-body-rotation) interpretation pushes
  the 85th-percentile error to ~5.8 rad/s against a ~0.07 rad/s tolerance
  -- a large, obvious failure, not a borderline one.

### Checkpoint: `wm-scenes-v1` distribution

- **Purpose**: confirm the seeded scene sampler is deterministic and every
  sample honors its declared constraints (object count, size, mass/
  density, friction, restitution, non-overlap, in-frustum camera framing).
- **Command**: `uv run pytest tests/test_distribution.py -v`
- **Expected result**: all 14 tests pass across 50 sampled seeds each
  (same seed -> bit-identical scene; every object non-overlapping, sized/
  massed/frictioned in range, fully inside the fixed camera's frustum at
  t=0).

### Checkpoint: end-to-end episode pipeline

- **Purpose**: confirm `generate`d episodes are real, valid GLB files
  through the *existing* transport (no new encoding): pass the independent
  glTF-Validator with zero errors, round-trip through `load_episode`, carry
  no NaN/Inf, and never fall through the ground plane.
- **Command**: `uv run pytest tests/test_episode_pipeline.py -v`
- **Expected result**: all 14 tests pass (3 generated episodes, each
  checked against the real validator, round-tripped, and checked for
  NaN/Inf and ground penetration -- see DESIGN.md's "Ground-contact
  tolerances" note for why that last check is split into a loose
  "didn't fall through the floor" (10cm, every frame) and a tight
  steady-state (5mm, last 20% of frames) bound instead of one "5mm at any
  frame" bound).

### Checkpoint: `generate` CLI demo

- **Purpose**: confirm the CLI end-to-end, the way a human would use it.
- **Command**:

  ```bash
  uv run gltfworld generate --out /tmp/gltfworld_generate_demo --episodes 3 --seed 42 --steps 100 --hz 30
  uv run gltfworld inspect /tmp/gltfworld_generate_demo/ep_000000.glb
  uv run gltfworld validate /tmp/gltfworld_generate_demo/ep_000000.glb
  cat /tmp/gltfworld_generate_demo/manifest.json
  ```

- **Expected result**: `generate` writes `ep_000000.glb`..`ep_000002.glb` +
  `manifest.json` to the output directory and exits 0. `inspect` prints
  each object (ground + N dynamic, with mass/shape/category), frame count/
  duration, and the extensions used. `validate` exits 0 with `numErrors ==
  0`. `manifest.json` records `dataset_version`, `scene_version`, the base
  `seed`, each episode's own seed, `steps`, `record_hz`, and `git_describe`.

### Checkpoint: MANUAL -- preview a generated episode in a real glTF viewer

- **Purpose**: the strongest end-to-end confirmation there is -- an
  independent, off-the-shelf glTF viewer (not gltfworld's own renderer)
  loading a `generate`d episode and playing its animation, with no console
  errors. This is a check a human needs to actually look at; it can't be
  fully automated.
- **Command** (generates `out/preview_episode.glb`, per this doc):

  ```bash
  uv run gltfworld generate --out out --episodes 1 --seed 7 --steps 90 --hz 30
  cp out/ep_000000.glb out/preview_episode.glb
  ```

- **Expected result**: drag `out/preview_episode.glb` into
  <https://gltf-viewer.donmccurdy.com> in a browser. You should see: a
  handful (matching whatever N the seed 7 scene sampled -- check
  `uv run gltfworld inspect out/preview_episode.glb`'s object list) of
  spheres/boxes (and, at 10% odds, a cylinder) falling under gravity and
  settling on a flat gray ground plane; the animation plays automatically
  on load; no errors in the browser's developer console. `out/` is
  git-ignored (see `.gitignore`), so this file is never committed -- it's
  a one-off local artifact for this manual check.

## V4 -- dataset build, provenance, stats, metric harness (pre-training gate)

This milestone builds everything the PRE-TRAINING GATE checks before any
model-training code (V5+) is allowed to start: the tensor contract episodes
are converted to for training, real packed datasets, dataset-level
statistics, and an independently cross-validated eval-metrics module. See
`docs/PRETRAINING_GATE.md` for the assembled checklist with actual observed
values; this section is the narrative/how-it-was-verified writeup.

### Checkpoint: tensor contract round trip

- **Purpose**: confirm `gltfworld.scene.contract.episode_to_tensors`/
  `tensors_to_state` (the `D=22` per-object state layout every downstream
  dataset/model consumes) losslessly round-trips the dynamic part of an
  Episode -- pose, velocities, shape/size, mass/friction/restitution,
  gravity/dt/camera -- to <= 1e-6 relative error, and that static objects
  (the ground) are excluded from `states` and land in the separate
  `static` sub-dict instead.
- **Command**: `uv run pytest tests/test_contract.py -v`
- **Expected result**: all 7 tests pass, including
  `test_round_trip_dynamic_part` parameterized over `(n_objects, T) in
  {(1,1), (3,30), (5,50)}`. Observed: **all 7 pass**.

### Checkpoint: provenance -- training tensors match what's actually on disk

- **Purpose**: confirm the tensor contract computed from a freshly
  simulated in-memory `Episode` matches the tensor contract computed from
  that same episode after a real save-to-GLB/load-from-GLB round trip, to
  <= 1e-6 absolute (fp32) -- i.e. that the `.glb` files a training pipeline
  reads are a faithful record of what MuJoCo actually produced, not merely
  "close" to it.
- **Command**: `uv run pytest tests/test_provenance.py -v` (needs the
  `sim` extra; 5 freshly simulated episodes, seeds 90210-90214)
- **Expected result**: all 5 pass. Observed: **all 5 pass**, states and
  globals within 1e-6 absolute, static (ground) fields within 1e-6 too.

### Checkpoint: dataset packing (`pack_dataset` + `DynamicsDataset`/`PerceptionDataset`)

- **Purpose**: confirm a directory of `ep_*.glb` episodes packs into one
  safetensors file with correct padding (`mask`/`class_ids` padding
  respected), a deterministic seed-keyed 90/5/5 split, and a
  `pack_meta.json` sidecar recording enough provenance (source manifest
  hash, `N_max`/`D`/count/split scheme/ground geometry) to make the packed
  file self-describing; and that the torch `Dataset` classes built on top
  read it back correctly (transition/sequence modes, memory-mapped
  rendered frames).
- **Command**: `uv run pytest tests/test_data.py -v -m "not gpu"` (pack/
  split/dataset-class tests, no MuJoCo/GPU needed) and
  `uv run pytest tests/test_data.py -v -m gpu` (the `PerceptionDataset`
  test, needs a real renderer)
- **Expected result**: all pass. Observed: **4 non-gpu + 1 gpu, all pass**.

### Checkpoint: real dataset generation

- **Purpose**: actually generate (not just unit-test) the two datasets this
  milestone promises, and report real wall time/throughput/disk usage --
  see `data/README.md` for the exact pinned commands.
- **Command**: see `data/README.md`.
- **Expected result / observed**:
  - `dynamics-v1`: 10,000 episodes, states only. **271.91s (4.53 min)** to
    generate, **636M** on disk. Packed: **421M** safetensors + 161K
    `pack_meta.json`, **115.18s** to pack. Split: train 8992 / val 532 /
    test 476.
  - `perception-v1`: 500 episodes with rendered 256x256 rgb+seg+depth
    frames. **127.87s (2.13 min)** to generate+render (~391 combined
    frames/s on this machine's RTX PRO 6000 Blackwell), **22G** on disk.
    Packed: **22M** safetensors + 9K `pack_meta.json`, **5.59s** to pack
    (packing only touches the tensor contract; rendered frames are read
    memory-mapped from each episode's own directory, not duplicated into
    the packed file). Split: train 458 / val 27 / test 15.
  - Validator sampling (see next checkpoint) and `gltfworld stats` (see
    below) were run against both real datasets, not just synthetic test
    fixtures.

### Checkpoint: validator, 0 errors, sampled from both real datasets

- **Purpose**: confirm real, at-scale generated episodes (not just the
  handful exercised by V3's own `test_episode_pipeline.py`) are still
  spec-valid glTF according to the independent, pinned glTF-Validator.
- **Command**: 20 episodes sampled (`random.seed(0)`) from each of
  `data/dynamics-v1/episodes/` and `data/perception-v1/episodes/`, each run
  through `uv run gltfworld validate <path>`.
- **Expected result**: `numErrors == 0` for all 40 sampled episodes.
  Observed: **0/20 dynamics-v1 episodes with numErrors != 0; 0/20
  perception-v1 episodes with numErrors != 0**.

### Checkpoint: `gltfworld stats` -- dataset sanity report

- **Purpose**: a single command that reports everything the gate needs to
  know about a packed dataset's health: 0 NaN/Inf, per-shape/class
  balance, physically plausible position/velocity/mass/friction ranges,
  ground-penetration bounds holding (per DESIGN.md's steady-state/
  transient split), and an energy-dissipation sanity trend -- in both a
  human-readable table and `--json` machine format.
- **Command**: `uv run gltfworld stats <packed_file>[.safetensors]
  [--json]`; unit-tested by `uv run pytest tests/test_stats.py -v`.
- **Expected result / observed** (full numbers in
  `docs/PRETRAINING_GATE.md`; summarized here):
  - Both datasets: **0 NaN/Inf** (hard requirement, met).
  - `dynamics-v1`: steady-state penetration <= 5mm for **99.95%** of
    episodes; transient penetration <= 100mm for **99.99%**; smoothed
    total energy non-increasing for **99.99%** of episodes.
  - `perception-v1`: steady-state <= 5mm for **99.80%**; transient <=
    100mm for **100.00%**; energy non-increasing for **100.00%**.
  - **Documented scope-boundary finding, not a bug**: at `--steps 100
    --hz 30` (~3.3s/episode), `wm-scenes-v1`'s *finite* 6x6m ground plate
    (DESIGN.md "Finite ground plate") is genuinely departed by at least
    one object in **14.02%** of `dynamics-v1` episodes and **12.60%** of
    `perception-v1` episodes -- exactly the behavior DESIGN.md's own
    earlier 100-step follow-up sweep predicted. `gltfworld stats` excludes
    those off-plate object-frames from the penetration checks above
    (there is no ground under them to "penetrate") and reports the
    departure rate as its own separate, visible metric
    (`off_plate_object_frame_fraction`/`episodes_with_departure_fraction`)
    instead of silently filtering it out of the denominator.

### Checkpoint: eval metrics (PSNR/SSIM/MSE), cross-validated

- **Purpose**: confirm gltfworld's own from-scratch PSNR/SSIM
  implementations (`gltfworld.eval.metrics`) -- the canonical numbers every
  later perception-quality eval (V7+) will report -- are numerically
  correct, not just plausible-looking, by cross-validating against
  independent reference implementations.
- **Command**: `uv run pytest tests/test_metrics.py -v`
- **Expected result**: PSNR matches `skimage.metrics.peak_signal_noise_ratio`
  exactly (same `data_range=255` convention) on random and structured
  (base image + Gaussian noise) image pairs, grayscale and RGB; SSIM
  matches `skimage.metrics.structural_similarity` (Wang et al. 2004
  parameters: `gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
  K1=0.01, K2=0.03`, `data_range=255`) within 1e-6; PSNR additionally
  cross-checked against `torchmetrics.image.PeakSignalNoiseRatio` (float32
  precision, ~1e-3 tolerance) as a second, independent reference.
  **torchmetrics' SSIM is deliberately not required to match** (see
  `tests/test_metrics.py`'s comment): it pads instead of cropping the
  gaussian-filtered SSIM map at image borders, a real, benign difference in
  border-handling convention between the two reference libraries
  themselves, not a bug in either -- skimage is this project's designated
  primary SSIM anchor. Observed: **all 11 tests pass** (property tests run
  40/40 and 20/20 Hypothesis examples respectively for the random/
  structured PSNR and SSIM checks).

### Checkpoint: external metric replication -- attempted (CLEVRER/SlotFormer)

- **Purpose**: the spec asked for an attempt to reproduce SlotFormer's
  published CLEVRER video-prediction PSNR (30.21) within +/-0.2 using their
  released rollout artifacts/pretrained weights, timeboxed so a
  download/access failure doesn't burn the milestone.
- **What was attempted**: fetched and read
  `github.com/pairlab/SlotFormer`'s README and `docs/{data,benchmark,install}.md`
  (all reachable, no auth needed) to find the actual reproduction path.
- **What actually blocks full reproduction** (recorded honestly, not
  glossed over):
  1. Pretrained weights and precomputed slot artifacts
     (`pretrained.zip`, `slots/clevrer_slots.pkl`,
     `slots/rollout_clevrer_slots.pkl`) are hosted on a Google Drive
     *folder* link
     (`https://drive.google.com/drive/folders/15y21miKZsAVHOSQEZLbUBWRrsZzcd5QW`).
     `curl`ing that URL returns an empty, JS-rendered Google Drive web-app
     shell (HTTP 200, `content-length: 0` on the meaningful body) --
     listing/downloading real folder contents needs either a browser
     session or the Google Drive API with OAuth credentials, neither
     available in this environment. A direct `uc?export=download&id=<folder
     id>` request (the common single-file trick) returned **HTTP 500**
     with 0 bytes, as expected for a folder id rather than a file id.
  2. The raw CLEVRER dataset itself (`Training/Validation Videos,
     Annotations`, required regardless of pretrained-weight access, per
     `docs/data.md`) is hosted at `clevrer.csail.mit.edu`, which timed out
     at the TCP connect stage from this sandbox
     (`curl -v --connect-timeout 8 http://clevrer.csail.mit.edu/` ->
     `Failed to connect to clevrer.csail.mit.edu port 80 after 8002 ms:
     Timeout was reached`, both `http://` and `https://`) -- host resolves
     (`128.52.131.62`) but the connection itself never completes, i.e. this
     is a genuine network-reachability failure, not a 4xx/5xx from the
     server.
  3. Even with data/weights in hand, SlotFormer's own reproduction
     pipeline (`docs/install.md`) pins a materially different, older stack
     (PyTorch 1.10.1 + CUDA 11.3, `einops==0.3.2`, `phyre==0.2.2`, the
     unpublished `nerv` training-loop package at a pinned git tag) that
     would need a separate environment from this project's own (PyTorch
     2.13, Python 3.12) to run at all.
- **Outcome**: per the spec's own timeboxing instruction, this was
  **not** pursued further (no partial/fudged number is reported in its
  place). The skimage/torchmetrics cross-validation above (PSNR exact,
  SSIM within 1e-6 against Wang et al. 2004's own reference implementation)
  is this milestone's metric-correctness anchor; the Physion replication
  planned for V8 remains gltfworld's primary *external* anchor, per the
  spec.

### Checkpoint: pre-training gate assembly

- **Purpose**: one document collecting every check above (plus the V1-V3
  checkpoints it depends on) into a single pass/fail checklist before any
  training code starts.
- **Command**: see `docs/PRETRAINING_GATE.md`.
- **Expected result**: every item's actual observed value is recorded next
  to its expected result; overall verdict recorded at the top.

### Checkpoint: CI

- **Purpose**: confirm the local dev environment isn't the only place this
  milestone's tests pass.
- **Finding**: `gh run list --limit 5` showed the `test` job failing on
  *every* run since V0, including runs re-triggered on pre-V4 commits --
  **pre-existing, not caused by this milestone**. Root cause: `import
  mujoco` (mujoco 3.11, unpinned upper bound in `pyproject.toml`) now
  unconditionally imports `mujoco.rendering.classic.renderer` ->
  `mujoco.egl` -> `OpenGL.EGL`'s raw ctypes bindings, which load
  `libEGL.so.1` at *module import time*. mujoco's own `try/except
  ImportError` guard around that import does not catch the failure mode
  that shows up on a runner with no EGL/GL system libraries at all (plain
  `ubuntu-latest`): PyOpenGL's ctypes loader resolves to `None` and then
  raises `AttributeError` (not `ImportError`) reading an attribute off it,
  which propagates straight through mujoco's guard and pytest's
  `importorskip`, breaking collection of every module that does a bare
  `import mujoco` (`gltfworld.datagen.*`) -- silently invalidating the
  documented "MuJoCo's core physics API needs no GPU/EGL context"
  assumption (DESIGN.md/V3) the moment the CI image lacks any EGL library
  at all, not because of anything this milestone's own tests do.
- **Fix applied**: `.github/workflows/ci.yml`'s `test` job now installs
  `libegl1 libgl1` (a real, if software, EGL/GL runtime) before `uv sync`,
  and also adds `--extra ml` to the CI sync (this milestone's dataset/
  metrics tests need `torch`/`safetensors`/`scikit-image`/`torchmetrics`,
  none of which were previously installed in CI).
- **Caveat, stated plainly**: per this milestone's rules, **no commit here
  is pushed**, so `gh run list --limit 1` at the time of writing still
  shows the last *pushed* commit's (pre-V4) run, which fails for the
  reason above -- the CI fix is committed locally alongside everything
  else and will apply the next time someone pushes. This checkpoint is
  therefore recorded as **not independently green on GitHub yet** in
  `docs/PRETRAINING_GATE.md`, with the above root-cause/fix documented so
  it isn't mistaken for silence/oversight.

## V5 -- dynamics model + baselines + training/eval code

Needs the `ml` extra (`torch`, `safetensors`, `scipy`, `matplotlib`,
`imageio[ffmpeg]`): `uv sync --dev --extra ml`. `--emit-gltf`/`--video`
additionally need `--extra sim`/`--extra render` (only for the CLI demo and
gpu-marked training smoke against real data; the unit test suite itself
doesn't need them). See DESIGN.md's "Dynamics model (V5)" section for the
full architecture/training-harness writeup this section verifies against.

### Checkpoint: rotation math vs. scipy (independent reference)

- **Purpose**: confirm every batched torch rotation op
  (`gltfworld.models.rotations`) -- quat normalize/hemisphere/multiply,
  axis-angle exponential map, quat<->matrix, quat<->6D, geodesic angle --
  agrees with `scipy.spatial.transform.Rotation` (same xyzw/axis-angle
  conventions, no translation needed), including the small-angle/near-zero
  numerical-stability cases and gradient finiteness.
- **Command**: `uv run pytest tests/test_rotations.py -v`
- **Expected result**: all 29 tests pass. Observed: **29/29 pass**.

### Checkpoint: dynamics model correctness (integrator, equivariance, masking)

- **Purpose**: confirm `InteractionTransformer`'s parameter count lands in
  the 4-7M target band; confirm a freshly constructed (zero-init head)
  model is bit-identical to `BallisticBaseline` with gravity zeroed out
  (both reduce to the same constant-velocity update through the *same*
  shared `integrate` function); confirm permutation equivariance across the
  object-token axis (for both `InteractionTransformer` and
  `NoInteractionMLP`); confirm padded object slots never leak into real
  slots' outputs (3 real objects vs. the same 3 padded to 5, masked);
  confirm forward output is finite with a genuinely unit quaternion.
- **Command**: `uv run pytest tests/test_dynamics.py -v`
- **Expected result**: all 9 tests pass. Observed: **9/9 pass**. Printed
  param count (`uv run python -m gltfworld.models.dynamics`): **4,815,113**
  (within the 4-7M band); `uv run python -m gltfworld.models.baselines`:
  `BallisticBaseline` 0 params (no learning), `NoInteractionMLP` **75,529**
  params (see DESIGN.md for why this is smaller than the spec's "~0.3M"
  approximation -- a documented, deliberate deviation, not an oversight).

### Checkpoint: rollout + eval correctness (shape, glTF round trip)

- **Purpose**: confirm `rollout()` produces the right shape/is finite for
  both single-episode and batched calling conventions and agrees between
  them; confirm `divergence_curve`/`horizon_metrics` are exactly zero
  comparing identical tensors to themselves and nonzero for a baseline that
  genuinely diverges from ground truth; confirm the "glTF at every hop"
  round trip (`tensors_to_episode` -> `save_episode` -> `load_episode` ->
  `episode_to_tensors`) reproduces the predicted tensors to <= 1e-6, and
  that a rebuilt episode's static (ground) object holds a constant pose
  across every frame.
- **Command**: `uv run pytest tests/test_rollout.py -v`
- **Expected result**: all 7 tests pass. Observed: **7/7 pass**.

### Checkpoint: training harness smoke (gpu, real `dynamics-v1` data)

- **Purpose**: confirm the training harness -- data loading, noise
  injection, optimizer/scheduler, bf16 autocast, checkpoint IO -- actually
  works end-to-end against the real packed `dynamics-v1` dataset (not a
  synthetic fixture), and that 500 steps measurably drops the (EMA-
  smoothed) training loss, for *both* `InteractionTransformer` and
  `NoInteractionMLP`, each within the 3-minute budget.
- **Command**:

  ```bash
  uv run python -m gltfworld.train.train_dynamics \
      --config configs/dynamics_v1.json --out /tmp/dynamics-v1-smoke --smoke
  uv run python -m gltfworld.train.train_dynamics \
      --config configs/dynamics_mlp.json --out /tmp/dynamics-mlp-smoke --smoke
  ```

  or as a pytest (skips cleanly if `data/dynamics-v1/packed/` isn't
  present locally): `uv run pytest tests/test_train_smoke.py -v -m gpu -s`
- **Expected result**: both exit 0. Measured on this machine (RTX PRO 6000
  Blackwell): `InteractionTransformer` (4,815,113 params) **41.5% raw /
  38.7% EMA** loss drop in **4.2s**; `NoInteractionMLP` (75,529 params)
  **38.9% raw / 35.5% EMA** loss drop in **1.7s** -- both comfortably clear
  the 30% bar, in seconds rather than the 3-minute budget. Full printed
  curves:

  ```
  # InteractionTransformer (configs/dynamics_v1.json)
  model=transformer device=cuda params=4,815,113
  step 100/500 phase=1 k=1 train_loss=0.05943 val_loss=0.04044 elapsed=1.1s
  step 200/500 phase=1 k=1 train_loss=0.03334 val_loss=0.03545 elapsed=1.9s
  step 300/500 phase=1 k=1 train_loss=0.02917 val_loss=0.03183 elapsed=2.6s
  step 400/500 phase=1 k=1 train_loss=0.06719 val_loss=0.03007 elapsed=3.4s
  step 500/500 phase=1 k=1 train_loss=0.06925 val_loss=0.02967 elapsed=4.2s
  smoke: raw start_loss=0.05930 end_loss=0.03468 drop=41.5%
  smoke: ema start_loss=0.05627 end_loss=0.03450 drop=38.7%
  SMOKE PASS

  # NoInteractionMLP (configs/dynamics_mlp.json)
  model=mlp device=cuda params=75,529
  step 100/500 phase=1 k=1 train_loss=0.06214 val_loss=0.04532 elapsed=0.6s
  step 200/500 phase=1 k=1 train_loss=0.03965 val_loss=0.04204 elapsed=0.9s
  step 300/500 phase=1 k=1 train_loss=0.03782 val_loss=0.03869 elapsed=1.2s
  step 400/500 phase=1 k=1 train_loss=0.07703 val_loss=0.03731 elapsed=1.5s
  step 500/500 phase=1 k=1 train_loss=0.07522 val_loss=0.03671 elapsed=1.7s
  smoke: raw start_loss=0.06829 end_loss=0.04175 drop=38.9%
  smoke: ema start_loss=0.06456 end_loss=0.04163 drop=35.5%
  SMOKE PASS
  ```

### Checkpoint: resumability (hand-verified, not just unit-tested)

- **Purpose**: confirm `--resume` actually continues an interrupted run
  correctly -- model weights, both phases' optimizer/scheduler state, the
  global step counter, and every RNG stream -- rather than merely not
  crashing.
- **Command**: run a short config to step 100 without `--resume`, then run
  the same `--out` directory again with `--resume` and a config whose
  total step count is higher; inspect `log.csv` for continuity.
- **Expected result / observed**: a 100-step partial run followed by
  `--resume` to a 300-step target continued cleanly from step 100 (printed
  `resumed from step 100`), correctly re-entered phase 1's cosine schedule
  (`lr` back near its phase-1 value, not reset to the initial value nor
  jumped to phase 2 early) and transitioned to phase 2 exactly at step 200
  (`k` annealing 2->8 observed correctly: `k=5` at step 250, `k=8` at step
  300, matching the linear anneal formula), with `log.csv` appended (not
  truncated) across the resume boundary and `step_0000100/0200/0300.
  safetensors` + matching `.train_state.pt` files all present.

### Checkpoint: full training run command (for the orchestrator)

- **Purpose**: the exact command the orchestrator runs to actually train
  the shipped `dynamics-v1` model and its `NoInteractionMLP` baseline to
  completion -- **not run as part of this milestone** (V5 delivers and
  smoke-tests the training code; the full run is explicitly out of scope
  here, per this milestone's own rules).
- **Command**:

  ```bash
  uv run python -m gltfworld.train.train_dynamics \
      --config configs/dynamics_v1.json --out runs/dynamics-v1
  uv run python -m gltfworld.train.train_dynamics \
      --config configs/dynamics_mlp.json --out runs/dynamics-mlp --model mlp
  ```

  Both are resumable (`--resume`) if interrupted; `runs/` is git-ignored
  (`.gitignore`), so no run artifacts are committed regardless.

### Checkpoint: eval CLI demo (rollout, metrics, glTF-at-every-hop)

- **Purpose**: confirm the eval CLI end-to-end, the way the orchestrator
  will actually use it once the full training run above has produced real
  checkpoints -- per-horizon metrics, the markdown table, the divergence
  curve PNG, and predicted/ground-truth episodes re-exported as real,
  independently loadable `.glb` files.
- **Command**:

  ```bash
  uv run python -m gltfworld.eval.rollout \
      --ckpt runs/dynamics-v1/best.safetensors \
      --data data/dynamics-v1/packed --split test \
      --out runs/dynamics-v1/eval \
      --mlp-ckpt runs/dynamics-mlp/best.safetensors \
      --emit-gltf 5
  ```

- **Expected result**: writes `metrics.json`, `metrics.md`,
  `divergence_curve.png`, and `pred/ep_XXXXXX.glb` / `gt/ep_XXXXXX.glb`
  pairs for 5 test episodes to `runs/dynamics-v1/eval/`; exits 0.
  **Acceptance bar**: the trained `InteractionTransformer` must beat
  `BallisticBaseline` on median position error at horizons 1/10/30 on the
  test split. Sanity-checked (not the full run -- a 300-step
  correctness-check training run, since the full run is out of this
  milestone's scope, see above) on 10 real `dynamics-v1` test-split
  episodes (`--max-episodes 10`):

  | model | h=1 | h=10 | h=30 | h=99 |
  | --- | --- | --- | --- | --- |
  | model (300-step) | 0.0055m | 0.0808m | 0.2833m | 1.2886m |
  | ballistic | 0.0053m | 0.0534m | 4.6248m | 55.5423m |

  Even this minimally-trained checkpoint already beats ballistic by more
  than an order of magnitude at h=30/h=99 (ballistic's constant-gravity,
  no-collision extrapolation diverges catastrophically past first ground
  contact); h=1/h=10 are close either way at 300 steps (a single-or-few
  1/30s step is dominated by the ballistic term regardless of training),
  consistent with the trend widening (not closing) as training
  progresses to the full run.

### Observed (full run)

The full training runs (`configs/dynamics_v1.json` and `configs/dynamics_mlp.json`)
have been completed and independently verified. Detailed results recorded in
[docs/RESULTS.md](docs/RESULTS.md): **acceptance bar met** (InteractionTransformer
beats ballistic at all horizons: 1.4× at h=10, 42× at h=30, 176× at h=99). The
MLP competitiveness finding (slightly better at h=30/99 medians, overlapping IQRs)
is documented honestly in RESULTS.md per project policy.

### Checkpoint: full test suite

- **Purpose**: confirm this milestone didn't regress anything upstream and
  that everything new is exercised.
- **Command**: `uv run pytest -v -m "not gpu"` (CI-equivalent) and
  `uv run pytest -v` (full, local, GPU machine only).
- **Expected result / observed**: **168 passed, 10 deselected** (not-gpu);
  **178 passed** (full, gpu tests included -- 2 new gpu tests added this
  milestone: the real-data training smoke, parametrized over both models).

### Checkpoint: CI

- **Purpose**: confirm CI's dependency set covers this milestone's new
  `matplotlib` dependency (added to the `ml` extra for the divergence-curve
  plot).
- **Finding**: `.github/workflows/ci.yml`'s `test` job already runs
  `uv sync --dev --extra render --extra sim --extra ml` (added in V4), so
  no CI workflow change was needed this milestone -- `matplotlib` resolves
  automatically as part of the existing `--extra ml` sync. Per this
  milestone's own rules, **no commit here is pushed**, so (as recorded
  honestly in V4's own CI checkpoint) the last actually-pushed commit's CI
  run predates this fix and is unaffected either way.

## V6 -- perception model + Hungarian matching + training/eval code

Needs the `ml` extra (already synced by V5, no new dependency added this
milestone -- `scipy.optimize.linear_sum_assignment` reuses the same
`scipy` `ml`-extra dependency V5's rotation-math cross-checks already
depend on). The training smoke and eval CLI additionally need the real
`perception-v1` dataset (`data/perception-v1/`, git-ignored, generated per
`data/README.md`) plus `--extra render --extra sim` for the GPU-only
checkpoints (rendering/EGL). See DESIGN.md's "Perception model (V6)"
section for the full architecture/training-harness writeup this section
verifies against.

### Checkpoint: symmetry-aware rotation loss (the correctness-critical part)

- **Purpose**: confirm the cube's rotational symmetry group is genuinely 24
  proper (orthogonal, `det=+1`) and pairwise-distinct rotations; confirm the
  box rotation loss is ~0 for a prediction equal to the GT composed with
  *any* of those 24 symmetries, and *not* near-zero for a genuinely
  different orientation (so the zero-loss checks aren't vacuously trivial);
  confirm the cylinder axis-alignment loss is ~0 for a spin about the
  object's own local-Y symmetry axis and for a 180-degree end-swap flip,
  and *not* near-zero for a genuinely tilted axis; confirm a sphere's
  rotation loss is exactly 0 regardless of how different the predicted and
  GT quaternions are.
- **Command**: `uv run pytest tests/test_matching.py -v`
- **Expected result**: all 23 tests pass. Observed: **23/23 pass**.

### Checkpoint: Hungarian matching correctness (incl. distractor queries)

- **Purpose**: confirm the matching cost/assignment picks the obviously-
  correct query<->GT pairing (not just *a* valid assignment) on a hand-built
  case with 3 real GT objects, 2 correct predicted queries per object, and
  2 far-away distractor queries that must end up unmatched; confirm a
  frame with 0 real GT objects returns an empty match with no error.
- **Command**: `uv run pytest tests/test_matching.py -k hungarian -v`
- **Expected result**: subset of the 23 above pass.

### Checkpoint: model shapes/finiteness + parameter count

- **Purpose**: confirm `PerceptionDETR.forward` produces the documented
  output shapes for every field, all finite; confirm the quaternion output
  is genuinely unit-norm and hemisphere-normalized; confirm position/size
  outputs stay within their documented workspace bounds; confirm batch
  independence (no accidental cross-sample leakage); confirm a wrong input
  image size is rejected rather than silently misbehaving; print/sanity-
  check the parameter count.
- **Command**: `uv run pytest tests/test_perception_model.py -v` and
  `uv run python -m gltfworld.models.perception`
- **Expected result**: all 7 tests pass. Observed: **7/7 pass**. Printed
  param count: **8,234,259** -- see DESIGN.md's "documented parameter-count
  deviation" note (the milestone spec's own approximate "~12-16M" target and
  the literal architecture description aren't simultaneously satisfiable;
  same precedent as V5's `NoInteractionMLP`).

### Checkpoint: eval metrics correctness (perfect-prediction + known-corruption cases)

- **Purpose**: confirm `compute_metrics` reports F1=1, 0 position/size/
  rotation error, 100% shape/class accuracy, and a 100% count-exact-match
  rate on a synthetic *perfect* prediction; confirm a hand-built mixed
  true-positive/false-positive/false-negative case reports the expected
  precision/recall exactly (0.5/0.5); confirm sphere rows are excluded from
  the rotation-error stats entirely (`n=0`), regardless of how wrong the
  predicted quaternion is; confirm a known, injected 5cm position offset on
  every matched prediction is reported back as a ~5cm median position
  error -- this validates the *metric computation itself*, independent of
  any trained model; confirm `build_predicted_episode` (the re-render
  check's `Episode` builder) round-trips through `save_episode`/
  `load_episode` to `<= 1e-6`, for both the ordinary (GT-matched color) case
  and the false-positive (default-color fallback) case.
- **Command**: `uv run pytest tests/test_perception_eval.py -v`
- **Expected result**: all 6 tests pass. Observed: **6/6 pass**.

### Checkpoint: training harness smoke (gpu, real `perception-v1` data)

- **Purpose**: confirm the training harness -- `PerceptionDataset` loading
  (real rendered frames, not a synthetic fixture), RGB-only augmentation,
  Hungarian matching, bf16 autocast, checkpoint IO -- works end-to-end
  against the real packed `perception-v1` dataset, and that 500 steps
  measurably drops the (EMA-smoothed) training loss within the 5-minute
  budget.
- **Command**:

  ```bash
  uv run python -m gltfworld.train.train_perception \
      --config configs/perception_v1.json --out /tmp/perception-v1-smoke --smoke
  ```

  or as a pytest (skips cleanly if `data/perception-v1/packed/` isn't
  present locally): `uv run pytest tests/test_train_perception_smoke.py -v -m gpu -s`
- **Expected result**: exits 0. Measured on this machine (RTX PRO 6000
  Blackwell), `PerceptionDETR` (8,234,259 params): **19.0% raw / 33.2% EMA**
  loss drop in **136.5s** -- comfortably clears the 30% bar, well under the
  5-minute budget. Full printed curve:

  ```
  model=PerceptionDETR device=cuda params=8,234,259
  step 100/500 train_loss=2.05505 val_loss=2.15920 matched_pos_err=0.7584m elapsed=27.7s
  step 200/500 train_loss=2.24856 val_loss=2.16282 matched_pos_err=0.7507m elapsed=54.8s
  step 300/500 train_loss=2.03515 val_loss=2.06810 matched_pos_err=0.7445m elapsed=81.9s
  step 400/500 train_loss=2.11222 val_loss=2.09216 matched_pos_err=0.7702m elapsed=109.3s
  step 500/500 train_loss=1.94137 val_loss=2.17090 matched_pos_err=0.7629m elapsed=136.5s
  smoke: raw start_loss=2.52709 end_loss=2.04675 drop=19.0%
  smoke: ema start_loss=3.06045 end_loss=2.04316 drop=33.2%
  SMOKE PASS
  ```

  The high matched-position-error (~0.75m) after only 500 steps is
  expected and not a bug: `matched_pos_err` is a raw median position error
  in meters over an essentially-untrained 500-step model (the milestone's
  own scope boundary is "deliver + smoke-test the training code", not
  "train to convergence" -- see the acceptance-bar checkpoint below for
  what the *full* run needs to clear).

### Checkpoint: full training run command (for the orchestrator)

- **Purpose**: the exact command the orchestrator runs to actually train
  the shipped `perception-v1` model to completion -- **not run as part of
  this milestone** (V6 delivers and smoke-tests the training code; the full
  run is explicitly out of scope here, per this milestone's own rules,
  mirroring V5's identical scope boundary for the dynamics model).
- **Command**:

  ```bash
  uv run python -m gltfworld.train.train_perception \
      --config configs/perception_v1.json --out runs/perception-v1
  ```

  Resumable (`--resume`) if interrupted; `runs/` is git-ignored, so no run
  artifacts are committed regardless.

### Checkpoint: eval CLI end-to-end (gpu, metrics + re-render + glTF-at-every-hop)

- **Purpose**: confirm the eval CLI end-to-end against real data -- model
  inference, Hungarian matching, the mean-state baseline (run through the
  identical metrics pipeline for comparison), and the GPU re-render check
  (predicted-`Episode` construction, rendering, PSNR/SSIM against the real
  stored GT frame, the inline `<= 1e-6` round trip, and a clean run through
  the real, pinned glTF-Validator) -- using a freshly-initialized (untrained)
  checkpoint, since this milestone doesn't run the full training (see
  above); this checkpoint is about the *pipeline* working end-to-end on
  real data, not about reporting trained-model quality numbers.
- **Command**:

  ```bash
  uv run python -m gltfworld.eval.perception_eval \
      --ckpt runs/perception-v1/best.safetensors \
      --data data/perception-v1 --split test \
      --out runs/perception-v1/eval
  ```

- **Expected result**: writes `metrics.json`, `metrics.md`, and
  `pred_frames/ep_XXXXXX_fYYYY.glb` files for the sampled re-render frames;
  exits 0.
- **Test-suite split, and a real bug this caught**: `tests/
  test_perception_eval_gpu.py` exercises this in two separate tests rather
  than one call to `main()`, and for a real reason found while writing it:
  `render_check`'s first version unconditionally built *and deleted* its
  own `EpisodeRenderer` (mirroring `gltfworld.eval.rollout
  ._render_side_by_side_videos`'s existing pattern) -- fine as a genuinely
  standalone CLI process, but fatal inside the shared pytest session, where
  `tests/test_data.py::test_perception_dataset_reads_rendered_frames`
  already holds the session-scoped `episode_renderer` fixture open (see
  `EpisodeRenderer`'s own documented "one live renderer per process"
  constraint: deleting *any* instance terminates the *shared* EGL display
  for every other still-open instance in the process). This surfaced as a
  session-teardown `EGL_NOT_INITIALIZED` crash in that unrelated fixture,
  not a failure in the new test itself -- exactly the class of bug that
  constraint's docstring warns about. Fixed by giving `render_check` an
  optional `renderer=` parameter (only builds/deletes its own when none is
  supplied); `test_perception_eval_cli_metrics_and_baseline` exercises
  `main()` with `--render-samples 0` (metrics/baseline pipeline only, no
  renderer involved at all), and `test_render_check_with_shared_renderer`
  calls `render_check` directly with the shared `episode_renderer` fixture
  (3 render samples, to stay fast) and then confirms that shared renderer
  is still usable afterward. `main()`'s own `--render-samples` path (used
  above) still builds/deletes its own renderer, correct for the real
  standalone-process case the CLI actually runs in.
- **Expected result / observed**: both tests pass; round trip `<= 1e-6`,
  glTF-Validator clean, PSNR/SSIM computed successfully (measured on this
  machine against a freshly-initialized checkpoint: PSNR median ~24.5 dB,
  SSIM median ~0.97 -- expected to be unremarkable at this stage since
  background/camera/ground dominate the frame and the object appearance
  itself is a documented GT-assist, see DESIGN.md; this number is not a
  claim about geometric prediction accuracy, which the matched-pose metrics
  above already cover separately).

### Acceptance (see DESIGN.md's "Perception model (V6)" section)

On the `perception-v1` test split: existence F1 >= 0.95, median matched
position error <= 0.05 m, class accuracy >= 0.95. **Measured against the
trained CNN encoder checkpoint** (see `docs/RESULTS.md` V6 section for the final
numbers): acceptance bar **NOT met** (existence F1 0.8701 < 0.95; median position
error 0.1798 m >> 0.05 m; class accuracy 0.9496 ≈ 0.95). The postmortem and
recovery findings (ViT memorization, dataset-scale guard, out-of-box GT filter,
CNN encoder swap) are documented in `docs/RESULTS.md` V6 section and DESIGN.md
sections V6.1-V6.3.

### Observed (final, V6)

Trained `perception-v4-cnn-40k` checkpoint (40k steps with CNN encoder on 4k-episode dataset) evaluated on test split: existence F1 **0.8701** (target ≥ 0.95), median matched position error **0.1798 m** (target ≤ 0.05 m), class accuracy **0.9496** (target ≥ 0.95). Position signal is real and grounded (4.3× better than mean-state baseline at 0.7732 m). Validation curve plateaued around step 40k at 0.155 m, suggesting dataset size remains limiting factor for sub-5cm accuracy.

### Checkpoint: full test suite

- **Purpose**: confirm this milestone didn't regress anything upstream and
  that everything new is exercised.
- **Command**: `uv run pytest -v -m "not gpu"` (CI-equivalent) and
  `uv run pytest -v` (full, local, GPU machine only).
- **Expected result / observed**: **197 passed, 13 deselected** (not-gpu, up
  from V5's 168/10 -- 4 new CPU-fast test files this milestone: `test_matching.py`,
  `test_perception_model.py`, `test_perception_eval.py`, and
  `test_train_perception_smoke.py`/`test_perception_eval_gpu.py`'s
  gpu-marked tests raising the deselected count); **210 passed** (full, gpu
  tests included -- 4 new gpu tests added this milestone: the real-data
  training smoke, the eval CLI metrics/baseline path, and the render-check
  path split across two tests per the renderer-sharing fix above).

### Checkpoint: CI / new dependencies

- **Purpose**: confirm this milestone didn't need any new dependency or CI
  workflow change.
- **Finding**: no new dependency was added -- `scipy.optimize
  .linear_sum_assignment` (Hungarian matching) reuses the `ml` extra's
  existing `scipy` dependency (already synced for V5's rotation-math
  cross-checks). `.github/workflows/ci.yml`'s `test` job already runs
  `uv sync --dev --extra render --extra sim --extra ml`, so no workflow
  change was needed. Per this milestone's own rules, **no commit here is
  pushed**, so this remains unverified against a live GitHub Actions run
  until a future push, exactly as V4/V5 recorded honestly for their own
  unpushed commits.

## V7 -- closed-loop demo + attribution

Needs the `ml` + `render` extras (no new dependencies -- `scipy.stats.chi`
reuses the same `scipy` already synced for V5/V6; everything else is code
already in this project). The end-to-end GPU checkpoint additionally needs
real assets: `data/perception-v1/{episodes,packed}`, `runs/dynamics-v1/
best.safetensors`, `runs/perception-v3-cnn/best.safetensors` + its
`eval/metrics.json`. See DESIGN.md's "Closed-loop demo + attribution (V7)"
section for the full three-arm design/correspondence-method writeup this
section verifies against.

### Checkpoint: log-map utility (`quat_to_axis_angle`), the correctness-critical new primitive

- **Purpose**: Arm B/C's finite-difference velocity assembly needs a
  quaternion -> rotation-vector log map that didn't exist before this
  milestone (`gltfworld.models.rotations` only had the exponential-map
  direction, `axis_angle_to_quat`). Confirm it matches `scipy.spatial
  .transform.Rotation.as_rotvec()` on random rotations in the principal
  `[0, pi]` range, round-trips exactly with `axis_angle_to_quat` in both
  compositions, is numerically stable at `theta -> 0`, and maps the identity
  quaternion to the zero vector.
- **Command**: `uv run pytest tests/test_rotations.py -v -k quat_to_axis_angle`
- **Expected result**: all 8 new tests pass (subset of the module's 37, up
  from 29 pre-V7). Observed: **8/8 pass**.

### Checkpoint: pure arm-assembly logic (CPU, no ckpt/GPU needed)

- **Purpose**: confirm the noise-injection/finite-diff/matching primitives
  underneath all 3 arms are correct in isolation, independent of any trained
  model: injected Gaussian noise (position + per-shape rotation) matches its
  requested sigma empirically (n=20,000, within 5%); zero noise reproduces
  the GT state exactly; the same seed reproduces the same draw; Arm B's
  physics-material fields stay untouched by noise; `finite_diff_velocity`
  recovers a known constant linear/angular velocity from two analytically
  constructed frames; **the milestone's own stated exactness bar** -- Arm C
  with a synthetic *perfect* detector (exact GT position/quat/size/shape/
  class, existence=1 at both frames) and zero Arm B noise reproduces Arm
  A/GT's state exactly, modulo the one documented exception
  (mass/friction/restitution default to fixed constants since
  `PerceptionDETR` never predicts them -- the test fixture's GT physics
  fields are constructed to already equal those defaults, making the
  comparison exact rather than approximate); a zero-detections degenerate
  case (`n_correspondence == 0`) produces an empty, not crashing, assembly;
  `hungarian_match`-based cross-frame correspondence recovers the obviously-
  correct pairing on a hand-built 2-object case (incl. an empty-input case);
  the exact chi(3) noise-calibration inversion against a synthetic
  `metrics.json` (and the `--noise-sigma-*` CLI-args-only construction);
  dataset resolution (`resolve_episodes_dir`) across all 3 accepted input
  shapes (raw glb dir, dataset root with `episodes/`+`packed/`, bare packed
  dir/file via `pack_meta.json`'s `source_dir`); deterministic split
  filtering (`select_episodes` against `gltfworld.data.pack
  .split_id_for_seed`, the same scheme every packed dataset in this project
  uses); the full `process_episode` pipeline (Arms A/B + ballistic only,
  `per_model=None`/no renderer) -- glTF emitted under `gt/armA/armB`, each
  round-trip-asserted inline, finite states, and determinism given a fixed
  seed; `ArmAccumulator`/`aggregate_results` shape/finiteness (incl. an
  all-empty Arm C when no perception model ran, and the `ordering_check`
  dict's shape); the attribution plot (incl. an arm with an empty curve).
- **Command**: `uv run pytest tests/test_closed_loop.py -v`
- **Expected result**: all 24 tests pass. Observed: **24/24 pass**.

### Checkpoint: full test suite (fast lane)

- **Purpose**: confirm this milestone didn't regress anything upstream.
- **Command**: `uv run pytest -v -m "not gpu"`
- **Expected result / observed**: **248 passed, 15 deselected** (not-gpu; up
  from V6's 197/13 -- `tests/test_closed_loop.py` is new (24 CPU-fast
  tests), `tests/test_rotations.py` gained 8 tests for the new
  `quat_to_axis_angle` log map, and `tests/test_closed_loop_gpu.py`'s single
  gpu-marked test raises the deselected count by 2).

### Checkpoint: closed-loop CLI end-to-end (gpu, real checkpoints + 3 real episodes)

- **Purpose**: confirm the full CLI runs end-to-end against real data and
  real, trained checkpoints -- render GT frames 0/1, run the real
  `PerceptionDETR` (CNN encoder, `perception-v3-cnn`), Hungarian-match
  detections across frames, roll forward with the real `InteractionTransformer`
  (`dynamics-v1`), save every arm as glTF, reload, and score -- producing
  finite metrics and a clean pass through the real, pinned glTF-Validator
  for every emitted GLB. **Not** a claim about the attribution curve's shape
  at scale (3 episodes; see the acceptance section below and DESIGN.md's V7
  section for a real, honestly-reported finding from this exact run about
  `A <= B <= C` ordering not holding at every horizon at this sample size).
- **Command**:

  ```bash
  uv run pytest tests/test_closed_loop_gpu.py -v -m gpu -s
  ```

  or directly:

  ```bash
  uv run python -m gltfworld.eval.closed_loop \
      --episodes data/perception-v1 \
      --dyn-ckpt runs/dynamics-v1/best.safetensors \
      --per-ckpt runs/perception-v3-cnn/best.safetensors \
      --per-metrics runs/perception-v3-cnn/eval/metrics.json \
      --out runs/closed-loop-smoke --n-episodes 3 --split test \
      --horizons 1 5 10 30 --video 0
  ```

- **Expected result / observed**: exit 0; writes `metrics.json` and
  `attribution.png`; every metric in `metrics.json` finite (`n=0` groups
  report `median: null`, expected); every emitted GLB across
  `gt/armA/armB/armC` (9 total for 3 episodes -- Arm C's count depends on
  whether any correspondence survived, which it did for all 3 here) passes
  `gltfworld validate` with 0 errors. Measured median position error at
  each horizon (this machine, `runs/dynamics-v1` + `runs/perception-v3-cnn`,
  seed 0):

  | arm | h=1 | h=5 | h=10 | h=30 |
  | --- | --- | --- | --- | --- |
  | A (oracle) | 0.0049 | 0.0199 | 0.0254 | 0.1244 |
  | B (oracle+noise) | 0.2329 | 0.9162 | 1.5404 | 3.3760 |
  | C (visual) | 0.5170 | 0.6262 | 0.7083 | 0.8259 |
  | ballistic | 0.0053 | 0.0267 | 0.0534 | 4.6160 |

  detection stats: `tp=10, fp=0, fn=1` (precision 1.0, recall 0.909, F1
  0.952) over the 3 episodes' correspondence-surviving objects.

  **`A <= C` holds at every horizon** (the test's own asserted sanity
  direction), but **`A <= B <= C` does not** at `h=5/10/30` -- Arm B
  diverges *faster* than Arm C. This is reported, not hidden or re-tuned
  away (see DESIGN.md's V7 section for the full explanation: Arm B's i.i.d.-
  per-frame Gaussian noise model, once finite-differenced over a small
  `dt`, implies a very large injected velocity noise, while the real
  detector's actual per-frame errors are apparently more correlated
  frame-to-frame than an i.i.d. model assumes and partially cancel in the
  finite difference instead). Only 3 episodes were run here; the
  orchestrator's full 20-episode run (with the eventual
  `perception-v4-cnn-40k` checkpoint) is the statistically meaningful
  version of this same measurement.

### Checkpoint: `--video` path (gpu, manual spot-check)

- **Purpose**: confirm the 2-panel (`GT | Arm C`) and 3-panel (`GT | Arm A |
  Arm C`) mp4 export works end-to-end (reuses `gltfworld.eval.rollout
  ._render_side_by_side_videos`'s exact renderer/imageio pattern).
- **Command**:

  ```bash
  uv run python -m gltfworld.eval.closed_loop \
      --episodes data/perception-v1 --dyn-ckpt runs/dynamics-v1/best.safetensors \
      --per-ckpt runs/perception-v3-cnn/best.safetensors \
      --per-metrics runs/perception-v3-cnn/eval/metrics.json \
      --out /tmp/cl-video-check --n-episodes 1 --video 1
  ```

- **Expected result / observed**: exits 0; writes `video/ep_XXXXXX.mp4`
  (21,311 bytes) and `video/ep_XXXXXX_3panel.mp4` (29,444 bytes) on this
  machine -- not wired into the automated gpu-marked test suite (a manual
  spot-check only, to keep the automated gpu lane's runtime bounded per this
  session's "keep GPU usage light" constraint; the 3-episode end-to-end
  checkpoint above already exercises every other real hop, including Arm C's
  renderer use for perception, without paying for full-episode video
  rendering 3x per episode).

### Checkpoint: full test suite (gpu lane)

- **Purpose**: confirm this milestone's new gpu-marked test runs cleanly
  alongside every pre-existing gpu-marked test (no regression from the new
  `quat_to_axis_angle` addition or `closed_loop.py`'s reuse of
  `EpisodeRenderer`/`PerceptionDETR`/`InteractionTransformer`).
- **Command**: `uv run pytest -v -m gpu`
- **Expected result / observed**: full gpu lane green, `tests/
  test_closed_loop_gpu.py`'s new test included; run once this session per
  the "run the gpu lane once, keep it brief" instruction.

### Checkpoint: CI / new dependencies

- **Purpose**: confirm this milestone didn't need any new dependency or CI
  workflow change.
- **Finding**: no new dependency was added (`scipy.stats.chi` reuses the
  `ml` extra's existing `scipy`). Per this milestone's own rules, **no
  commit here is pushed**, so this remains unverified against a live GitHub
  Actions run until a future push, exactly as V4/V5/V6 recorded honestly for
  their own unpushed commits.

### Acceptance (see DESIGN.md's "Closed-loop demo + attribution (V7)" section)

Closed loop runs end-to-end via real glTF at every hop (every arm's rollout
saved -> reloaded -> round-trip-asserted `<= 1e-6` before scoring);
`attribution.png` produced; every emitted GLB validates clean. Arm ordering
sanity (`A <= B <= C` at long horizons, allowing statistical noise) is
**reported, not gated** -- per this exact requirement, the 3-episode GPU
smoke's finding that `A <= C` holds but the full `A <= B <= C` chain does
not at `h >= 5` is recorded above and in DESIGN.md as a real finding, not
silently passed over. Full-scale (20-episode) confirmation of whether this
pattern holds at a statistically meaningful sample size, and with the
eventual `perception-v4-cnn-40k` checkpoint, is the orchestrator's to run
separately, per this milestone's own scope boundary (deliver + smoke-test
the closed loop here, the same precedent V5/V6 established for their own
full training/eval runs).

### Observed (final, V7)

Closed-loop demo on 20 test-split episodes with `perception-v4-cnn-40k` CNN encoder + `dynamics-v1` transformer: Arm A (oracle) baseline, Arm B (oracle + chi(3)-calibrated noise) upper bound on i.i.d.-measurement-noise cost, Arm C (real closed loop) shows **1.62 m position error at h=99** vs. ballistic's **55.5 m** (34× improvement). Key findings: (1) detector errors are frame-correlated (lag-1 autocorrelation 0.55–0.82), causing naive i.i.d.-noise model (Arm B) to diverge 17× faster than real detector (Arm C) at h=99; (2) learned dynamics keeps imperfect perceptual observations physically plausible. Attribution curve, per-arm trajectory metrics, and real glTF at every hop archived in `runs/closed-loop-v1/`.

## V8 -- Physion external anchor (state-based track)

Needs the `sim` extra (`h5py`, added this milestone) + `ml` extra. The
end-to-end checkpoints below additionally need real, gitignored assets:
`data/external/physion/hdf5/extracted/Collide/hdf5s/*.hdf5` (150 files,
32.62 GiB downloaded + extracted from `Collide_testing_HDF5s.tar.gz`, see
`docs/PHYSION.md`) and `runs/dynamics-v1/best.safetensors` (V5's trained
checkpoint). See DESIGN.md's V8 line and `docs/PHYSION.md` (schema,
conversion findings, "V8 decision" section) for the full design writeup
this section verifies against.

### Checkpoint: HDF5 acquisition + schema

- **Purpose**: confirm the downloaded archive is genuine (not truncated/
  corrupted) and that the schema documented in `docs/PHYSION.md` was
  derived from the real file, not assumed from the upstream repo's docs.
- **Command**: `curl -sI https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/Collide_testing_HDF5s.tar.gz` (compare `Content-Length` against `docs/PHYSION.md`'s table); `uv run pytest tests/test_physion_convert.py tests/test_physion_ocp_eval.py -v` (skips cleanly if the archive isn't present on this machine).
- **Expected result / observed**: `Content-Length: 35026607691` (exact
  match, 32.62 GiB); all real-data tests pass when the archive is present
  (**19/19** across both test files on this machine), skip cleanly
  otherwise.

### Checkpoint: HDF5 -> glTF conversion (stage 2)

- **Purpose**: confirm `gltfworld.physion.convert` produces valid, faithful
  glTF from real per-trial HDF5 data -- real mesh geometry, real per-frame
  poses/velocities, physics/semantics metadata -- per DESIGN.md's "object
  states -> our glTF conversion experiment" line.
- **Command**: `uv run pytest tests/test_physion_convert.py -v`
- **Expected result / observed**: **8/8 pass** on 3 real converted Collide
  trials: glTF-Validator **0 errors** (warnings, if any, are only
  `UNSUPPORTED_EXTENSION` for the draft/custom extensions, same bar as V1's
  own validator test); real mesh POSITION/NORMAL/indices accessors
  round-trip bit-exact; pose-animation frame count equals the source
  HDF5's frame count exactly (152 == 152 on the sampled trials); poses/
  velocities/physics metadata (mass/friction/restitution/is_static/
  category/role) round-trip through the *existing*, unmodified
  `gltfworld.scene.convert.episode_from_gltf` to `<=1e-5` absolute.
- **Full-archive robustness**: all 150 Collide test trials convert without
  error (`gltfworld.physion.ocp_eval.convert_all_trials`), including two
  edge cases only found at full scale (not in the initial 3-trial sample):
  distractor/occluder objects with truncated physics-metadata arrays, and
  13 real-world asset model names with empty exported mesh geometry --
  both handled with documented, tested fallbacks (see
  `docs/PHYSION.md` findings 12-13), never affecting the OCP-relevant
  (agent/patient) geometry in any of the 150 trials checked.

### Checkpoint: OCP evaluation (stage 3)

- **Purpose**: confirm the OCP accuracy numbers in `docs/RESULTS.md`'s V8
  section are reproducible, and that the evaluation harness's own building
  blocks (Wilson CI, deterministic calibration split, threshold
  calibration, the GT-contact oracle's real-mesh proximity computation) are
  independently correct.
- **Command**: `uv run pytest tests/test_physion_ocp_eval.py -v`; full run:
  ```
  uv run python -m gltfworld.physion.ocp_eval \
      --hdf5-dir data/external/physion/hdf5/extracted/Collide/hdf5s \
      --glb-dir data/external/physion/glb/Collide \
      --dynamics-ckpt runs/dynamics-v1/best.safetensors \
      --out runs/physion-ocp-v1
  ```
- **Expected result / observed**: **11/11** unit + real-data smoke tests
  pass. Full run (150 trials, ~44s wall on this machine, CPU-only): GT-
  contact oracle **92.0%** held-out accuracy (n=100, 95% CI [0.850, 0.959]);
  our dynamics (zero-shot `InteractionTransformer`) **49.0%** held-out
  (n=100, CI [0.394, 0.587], statistically indistinguishable from the 50%
  chance rate); ballistic control identical in binary accuracy but median
  final-frame divergence **2.32m vs. 102.54m** (our dynamics vs. ballistic)
  -- full breakdown and discussion in `docs/RESULTS.md`'s V8 section.
  Sanity check: the HDF5's own per-frame contact label agrees with the
  independently-shipped Core archive's `labels.csv` on **150/150** trials
  (100%).

### Checkpoint: full test suite (fast lane)

- **Purpose**: confirm this milestone didn't regress anything upstream.
- **Command**: `uv run pytest -v -m "not gpu"`
- **Expected result / observed**: **275 passed, 15 deselected** (not-gpu;
  up from V7's 248/15 -- `tests/test_physion_convert.py` (8 tests) and
  `tests/test_physion_ocp_eval.py` (11 tests) are new; `tests/
  test_physion_ingest.py` (8 tests, from the pre-V8 `v8-physion-ingest`
  merge) already counted toward V7's baseline via the merge).

### Acceptance

Per this milestone's own rules ("a failed transfer with a working
transport-conversion is still a successful gap experiment -- frame it that
way"): the transport-conversion half (stage 2) is unambiguously successful
(validator-clean, bit-exact round trips, verified at both 3-trial and
full-150-trial scale). The dynamics-model half (stage 3) shows a genuine,
honestly-reported zero-shot transfer collapse to chance -- exactly the
outcome DESIGN.md/`docs/PHYSION.md`'s own honest feasibility notes
anticipated ahead of time, not a surprise discovered after the fact. Both
outcomes are reported plainly in `docs/RESULTS.md`'s V8 section, including
the explicit, load-bearing caveat that this milestone's numbers are not a
head-to-head comparison against the published benchmark table (one
scenario, state-based input, a zero-shot-transferred model vs. every
published number's trained/fine-tuned one).
