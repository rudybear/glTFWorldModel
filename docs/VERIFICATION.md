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
  `uv sync --dev --extra render` then `uv run pytest -v -m "not gpu"`); or
  locally: `uv run pytest -v -m "not gpu"`.
- **Expected result**: green; 38 non-gpu tests pass, 7 gpu tests deselected
  (not errored, not skipped-with-a-warning -- cleanly excluded by the
  marker). Run the full suite (gpu tests included) locally with
  `uv run pytest -v` -- both invocations must be green on a GPU machine.
