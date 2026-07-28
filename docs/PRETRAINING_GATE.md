# Pre-training gate

Status: **9/10 items PASS, 1 item BLOCKED on a pending push** (see the CI
item below -- the underlying fix is committed, it just hasn't been
independently re-run on GitHub because this milestone's rules say not to
push). See `docs/VERIFICATION.md`'s V4 section for the full narrative
writeup behind each item.

Every item below must be checked before any model-training code (V5+) is
allowed to start consuming `data/dynamics-v1` or `data/perception-v1`. Each
item states its purpose in plain language, the exact command to reproduce
it, what a human should expect to see, and the actual value observed on
this run.

## 1. Transport round-trip suite

- [x] **Purpose**: confirm the underlying glTF transport (poses, KHR
  physics, `RWM_state_series`) that every episode -- and therefore every
  packed dataset -- rides on has never lost or corrupted data, across
  randomly generated episodes plus a deterministic golden one.
- **Command**: `uv run pytest tests/test_roundtrip.py tests/test_accessors.py tests/test_khr_schema.py tests/test_consistency.py -v`
- **Expected result**: all pass, every float32 array bit-for-bit equal.
- **Observed**: all pass (part of the 123 non-gpu tests below).

## 2. Validator: 0 errors on N sampled episodes from BOTH datasets

- [x] **Purpose**: confirm real, at-scale generated episodes (not just a
  handful of hand-built fixtures) are still spec-valid glTF, independently
  of gltfworld's own reader.
- **Command**:

  ```bash
  uv run python3 - <<'EOF'
  import random, subprocess, sys, json
  from pathlib import Path
  random.seed(0)
  def sample_and_validate(episodes_dir, n=20):
      paths = sorted(Path(episodes_dir).glob("ep_*.glb"))
      sample = random.sample(paths, min(n, len(paths)))
      results = []
      for p in sample:
          r = subprocess.run([sys.executable, "-m", "gltfworld.cli", "validate", str(p)],
                              capture_output=True, text=True)
          report = json.loads(r.stdout)
          results.append(report.get("issues", {}).get("numErrors", -1))
      return results
  for name, d in [("dynamics-v1", "data/dynamics-v1/episodes"), ("perception-v1", "data/perception-v1/episodes")]:
      res = sample_and_validate(d, 20)
      print(name, "numErrors != 0:", sum(1 for e in res if e != 0), "/", len(res))
  EOF
  ```

- **Expected result**: `numErrors == 0` for every sampled episode in both
  datasets.
- **Observed**: `dynamics-v1`: **0/20** with `numErrors != 0`.
  `perception-v1`: **0/20** with `numErrors != 0`.

## 3. Renderer analytic + crosscheck

- [x] **Purpose**: confirm the renderer used to produce `perception-v1`'s
  frames is quantitatively correct (closed-form sphere/box/seg checks) and
  agrees with an independent renderer (MuJoCo) on scene geometry.
- **Command**: `uv run pytest tests/test_render_analytic.py tests/test_crosscheck.py -v -m gpu`
- **Expected result**: all pass; silhouette IoU >= 0.98 overall.
- **Observed**: all pass (part of the 8 gpu tests below); IoU **0.9962**
  overall (unchanged from V2/V3, no rendering-path code touched by this
  milestone).

## 4. Velocity consistency

- [x] **Purpose**: confirm recorded `lin_vel`/`ang_vel` (the values that
  feed straight into the `states` tensor's columns 7:13) are self-consistent
  with finite-differencing recorded poses -- the check that would catch a
  MuJoCo body-local/world-frame angular-velocity mixup.
- **Command**: `uv run pytest tests/test_velocity_consistency.py -v`
- **Expected result**: both tests pass (85th-percentile error < 2% of max
  speed).
- **Observed**: both pass (part of the 123 non-gpu tests below; unchanged
  from V3, no simulation code touched by this milestone).

## 5. Provenance

- [x] **Purpose**: confirm the tensor contract computed from a freshly
  simulated in-memory Episode matches the tensor contract computed from
  that same episode after a real GLB save/load round trip, to <= 1e-6
  absolute (fp32) -- proof that training data == what's actually in the
  `.glb` files on disk.
- **Command**: `uv run pytest tests/test_provenance.py -v`
- **Expected result**: all 5 (seeds 90210-90214) pass, max abs diff <= 1e-6.
- **Observed**: all 5 pass.

## 6. Stats sanity (0 NaN, bounds hold, energy trend)

- [x] **Purpose**: confirm both real packed datasets are actually sane
  before any training starts: no NaN/Inf, physically plausible ranges,
  ground-penetration bounds holding, energy dissipating (not blowing up).
- **Command**: `uv run gltfworld stats data/dynamics-v1/packed/dynamics-v1.safetensors` and
  `uv run gltfworld stats data/perception-v1/packed/perception-v1.safetensors`
  (`--json` for the machine-readable form); unit tests:
  `uv run pytest tests/test_stats.py -v`
- **Expected result**: `NaN/Inf count: 0` for both; steady-state
  ground-penetration and energy-trend fractions high (not necessarily
  100%, per DESIGN.md's own documented contact-transient/finite-plate
  caveats); `tests/test_stats.py` passes.
- **Observed**:

  | metric | dynamics-v1 | perception-v1 |
  | --- | --- | --- |
  | episodes | 10,000 | 500 |
  | transitions | 990,000 | 49,500 |
  | NaN/Inf count | **0** | **0** |
  | steady-state penetration <= 5mm | **99.95%** | **99.80%** |
  | transient penetration <= 100mm | **99.99%** | **100.00%** |
  | energy non-increasing (smoothed) | **99.99%** | **100.00%** |
  | off-plate object-frames (excluded above) | 1.06% | 1.00% |
  | episodes with >=1 object leaving the finite plate | 14.02% | 12.60% |

  The last two rows are a documented, pre-existing scope boundary
  (DESIGN.md "Finite ground plate"), not a defect: at `--steps 100 --hz 30`
  (~3.3s/episode) a real fraction of episodes have an object roll/bounce
  off the 6x6m plate and free-fall afterwards, exactly as DESIGN.md's own
  earlier 100-step sweep predicted. `gltfworld stats` excludes those
  frames from the penetration checks (there's no ground under them) and
  reports the departure rate separately rather than hiding it in the
  denominator. `tests/test_stats.py`: **6/6 pass**.

## 7. Metrics cross-validated

- [x] **Purpose**: confirm the canonical PSNR/SSIM/MSE implementations
  every later perception-quality eval will report (`gltfworld.eval.metrics`)
  are numerically correct against independent references, not merely
  plausible.
- **Command**: `uv run pytest tests/test_metrics.py -v`
- **Expected result**: PSNR matches `skimage.metrics.peak_signal_noise_ratio`
  exactly; SSIM matches `skimage.metrics.structural_similarity` (Wang et
  al. 2004 parameters) within 1e-6; PSNR additionally cross-checked
  against `torchmetrics` (~1e-3, float32).
- **Observed**: **11/11 pass**. External CLEVRER/SlotFormer replication
  was attempted and honestly documented as blocked (Google Drive
  folder gating + `clevrer.csail.mit.edu` unreachable from this sandbox +
  an incompatible pinned training stack) -- see
  `docs/VERIFICATION.md`'s "external metric replication -- attempted"
  checkpoint for the exact URLs/errors. The skimage/torchmetrics
  cross-validation above is this milestone's metric-correctness anchor per
  the spec's own instruction (Physion replication in V8 remains the
  primary *external* anchor).

## 8. Datasets packed with recorded hashes

- [x] **Purpose**: confirm both datasets are packed into training-ready
  tensors with enough recorded provenance (source manifest hash, split
  scheme, ground geometry) that they're independently reproducible and
  self-describing.
- **Command**: `cat data/dynamics-v1/packed/dynamics-v1.pack_meta.json data/perception-v1/packed/perception-v1.pack_meta.json`;
  see `data/README.md` for the exact generation+pack commands.
- **Expected result**: `source_manifest_hash_sha256` present and non-null;
  `count`/`n_max`/`d`/`t`/split_counts consistent with the generation run.
- **Observed**:
  - `dynamics-v1`: sha256 `b6e8c86c4ce66e83f0e490bc44faa6889f211e7c1ab8d571985934e57a13a516`;
    10,000 episodes, `N_max=5`, `D=22`, `T=100`; split train 8992 / val 532 / test 476.
  - `perception-v1`: sha256 `d2bc671d8a5bbb0f48cd82107e6728c8f494cc56535f51be1d872839204af5a0`;
    500 episodes, `N_max=5`, `D=22`, `T=100`; split train 458 / val 27 / test 15.

## 9. Full local test suite green (both pytest lanes)

- [x] **Purpose**: confirm every automated check above (and everything
  from V0-V3) is green together, not just individually.
- **Command**: `uv run pytest -q -m "not gpu"` and `uv run pytest -q -m gpu`
- **Expected result**: both exit 0.
- **Observed**: **123 passed, 8 deselected** (`not gpu` lane); **8 passed,
  123 deselected** (`gpu` lane). 131 tests total, all green.

## 10. CI green on GitHub

- [x] **Purpose**: confirm the test suite passes in a clean environment,
  not just this machine.
- **Command**: `gh run list --limit 1`
- **Expected result**: latest run's status `completed`/`success`.
- **Observed**: **PASS** (verified post-push by the orchestrator,
  2026-07-28). History: every CI run from V0 through V3.1 had been failing
  for two stacked reasons, both found during V4: (1) `import mujoco`
  transitively imports `OpenGL.EGL`'s raw ctypes bindings at module load
  time, which raise `AttributeError` (not `ImportError`, so mujoco's own
  guard and pytest's `importorskip` both miss it) on a runner with no
  EGL/GL system libraries — fixed by installing `libegl1 libgl1` and
  adding `--extra ml` to the CI sync (commit fff0715); (2) the V0-era
  standalone `gltf-validator` job invoked the validator binary with an
  unsupported `--version` flag (usage + exit 1 on every run) and was
  redundant — real validation runs inside pytest via
  `tests/test_validator.py` — so the job was removed (commit ce3e1a1).
  After pushing both fixes: run 30333212629 on commit ce3e1a1 concluded
  `success` (job `test`: success). Details in `docs/VERIFICATION.md` V4
  "Checkpoint: CI".
