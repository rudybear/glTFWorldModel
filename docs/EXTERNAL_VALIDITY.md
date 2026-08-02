# External validity: two independent experiments (2026-08-02)

Every checkpoint in [docs/VERIFICATION.md](VERIFICATION.md) is verified by an
agent independent of the one that implemented it -- but that verifier still
shares this project's docs, conventions, and (often) its context. It answers
"did we build what we said we built," not "would a stranger, with none of
that shared context, get the same result." This document records two
experiments run specifically to answer the second question, both dated
2026-08-02.

## Experiment A: spec-only reimplementation (decoder external validity)

**Protocol.** An implementer isolated from this project's source code was
given exactly three things: [docs/RWM_EXTENSIONS.md](RWM_EXTENSIONS.md) (as
it existed before this document's own fixes below), the vendored JSON
Schemas (`docs/schemas/khr/`, `docs/schemas/rwm/`), and a handful of
real, gltfworld-produced GLB episodes. No access to `gltfworld.ext.rwm`,
`gltfworld.scene.convert`, or any other implementation source. The task:
decode every object's pose/velocity/action/joint-state time series purely
from those three inputs, and match gltfworld's own decoder bit-for-bit.

**Isolation rules.** No source code of this repo (beyond the schemas, which
are themselves vendored third-party-adjacent artifacts, not gltfworld
prose); no access to this project's test suite; no communication with
whoever wrote the docs. The only ground truth available for self-checking
was the sample GLBs' own raw bytes and the schemas' own validation rules.

**Result: bitwise-identical decode.** Once the implementer's under-specified
guesses converged (below), every decoded array matched gltfworld's own
decoder's output bit-for-bit (uint32-view comparison, not "close enough")
across the sample episodes provided.

**Usability findings -- 6 under-specified conventions, now normative.**
Getting to a bitwise-identical decode required guessing six conventions the
prior version of `docs/RWM_EXTENSIONS.md` left implicit. All six are now
written down verbatim in that document's new "Decoder conventions
(normative)" section:

1. **Object-inclusion rule** (the one genuine bug this experiment caught).
   The implementer's first attempt excluded `is_static` objects from the
   decoded object (N) axis, reasoning that "static" bookkeeping meant
   "not part of the dynamic state." This produced a scene with the wrong
   object count and an off-by-one channel-to-object mapping for every
   object after the (wrongly) skipped one -- silently wrong shapes, not a
   crash. The correct rule: *every* node with `extras.rwm.object_id` is on
   the N axis, `is_static` or not; only nodes with no `extras.rwm` at all
   (camera, lights) are excluded. This is the highest-value fix in this
   document's set -- it's the one that produced silently wrong output
   rather than an obvious failure.
2. **Array ordering.** The N axis must be ordered by ascending
   `object_id`, not glTF node order -- node order happening to match
   `object_id` order in every sample GLB is a property of gltfworld's own
   encoder, not a guarantee the format makes.
3. **Quaternion component order.** `(x, y, z, w)`, per core glTF -- easy to
   get backwards without a general glTF background, so now stated locally
   in the RWM doc rather than left to "the reader already knows glTF."
4. **Pose animation interpolation.** Samplers backing pose channels use
   `STEP`, never `LINEAR` -- undocumented before, and a `LINEAR` decode
   would silently fabricate physically-impossible in-between orientations.
5. **Chunked-channel reassembly order.** Channels split across multiple
   accessors (width > 4) must be concatenated in ascending `component`
   order, not array-position order.
6. **Channel/time-length invariant.** Every channel accessor's `count`
   must equal `len(times)`; decoders SHOULD validate this rather than
   assume it.

**Takeaway.** The wire format itself was sound -- a from-scratch decoder
built only from the docs and schemas reached bit-exact agreement. What was
missing was not a format gap but a *documentation* gap: six conventions
this project's own code always followed but never wrote down. All six are
now explicit and normative in `docs/RWM_EXTENSIONS.md`.

## Experiment B: clean-room reproduction from the public clone

**Protocol.** A fresh clone of the public GitHub repository
(`github.com/rudybear/glTFWorldModel`), set up strictly by following
`README.md`'s own "Setup" instructions (`uv sync --all-extras --dev`, then
the documented pytest invocations) with no access to this project's private
development history, run logs, or any context beyond what's in the repo
itself. Goal: reproduce (a) the documented smoke-test pass/skip/deselect
counts, and (b) bit-identical seeded dataset generation.

**Results: exact-digit reproduction, PASS.**

- **Smoke percentages/split sizes**: reproduced exactly, digit-for-digit,
  against the numbers already recorded in `docs/PRETRAINING_GATE.md` /
  `data/README.md` (e.g. `dynamics-v1`'s 8992/532/476 train/val/test split
  at the documented seed).
- **Bit-identical seeded generation**: `uv run gltfworld generate --seed
  <n> ...` on the fresh clone produced byte-identical `.glb` output
  against the same command run in this project's own development
  environment -- confirming the deterministic-seed guarantee
  (`docs/VERIFICATION.md`'s V3 "wm-scenes-v1 distribution" checkpoint)
  holds across machines/environments, not just within one.

**Stranger's-experience friction, now fixed:**

| # | Friction (as a stranger hit it) | Fix |
| --- | --- | --- |
| a | README's Setup section gave no expectation for what `pytest` actually reports on a fresh clone -- a stranger with no Physion data or trained checkpoints sees skips/deselections with no baseline to compare against, and can't tell "expected" from "something's broken." | Added "Expected results on a fresh clone" to README's Setup section: exact fast-lane (333->336 after this milestone's own new stats tests, +17 skipped, +19 deselected) and gpu-lane (8 passed, 11 skipped) counts, plus what unlocks the full numbers. |
| b | `gltfworld stats <directory>` crashed with an opaque `OSError: No such device (os error 19)` from safetensors' Rust loader -- no indication the problem was "you passed a directory," just a bare native-library error. | `src/gltfworld/cli.py`'s `_cmd_stats` now detects a directory argument, resolves it if it contains exactly one `.safetensors` file, and otherwise prints a clear usage error and exits 2 -- no more opaque `OSError`. Covered by three new tests in `tests/test_stats.py`. |
| c | `docs/PHYSION.md` documented the download *table* (sizes, per-scenario URLs) but never gave a single copy-pasteable command for the one file this repo's own Physion pipeline actually consumes. | Added a copy-pasteable `mkdir`/`curl`/`tar` block for `Collide_testing_HDF5s.tar.gz` directly in the "Data acquired" section, producing the exact path `gltfworld.physion.ocp_eval` expects. |
| d | `docs/VERIFICATION.md` reads, at a glance, like a live command reference -- but some documented commands/behaviors (e.g. V0's CLI-stub behavior) were superseded by later milestones, and nothing on the page said so. A stranger re-running an early checkpoint and seeing different behavior has no signal that this is expected. | Added a header note: this is a **historical, per-milestone log** -- each checkpoint reflects the repo's `HEAD` at that milestone, may be superseded later, with V0's validate-stub behavior called out as the concrete example. |

**Takeaway.** The pipeline itself reproduces exactly -- same digits, same
bytes, on a genuinely independent clone/environment. The friction was
entirely in the "onboarding experience" layer: no baseline expectation for
first-run test output, one CLI command with an unfriendly failure mode, one
missing copy-paste command, and one doc that read as more "current" than it
is. All four are fixed as of this milestone.

## Why this matters

This is exactly the category of thing a real standards-ratification process
(Khronos 3D Formats WG or otherwise) exists to surface: an external party,
working only from the published spec/docs, hits exactly the ambiguities an
author blind to their own tacit assumptions cannot see in their own writing.
Six normative conventions and four onboarding frictions came out of two
short, cheap experiments -- an argument, not just an assertion, for why the
`RWM_state_series` time-series pattern (see
[docs/GAP_REPORT.md](GAP_REPORT.md)'s recommendation #1) is worth taking to
an actual KHR-track ratification effort rather than shipping as a
single-repo custom extension indefinitely: ratification is this same
external-validity exercise, done properly, at a scale one repo's own
verification protocol cannot replicate.
