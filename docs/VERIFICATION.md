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
