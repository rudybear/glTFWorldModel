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
- **Command**: `.github/workflows/ci.yml`, job `gltf-validator`
- **Expected result**: downloads/caches `gltf-validator` release
  `2.0.0-dev.3.10` (linux64) and `gltf_validator --version` prints a version
  string; job is green. (No glTF files are validated yet — that starts in
  V1.)

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
