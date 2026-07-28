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
