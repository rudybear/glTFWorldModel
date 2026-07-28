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
