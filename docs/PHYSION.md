# Physion benchmark: format reconnaissance (V8 prep)

Status: 2026-07-28. This is ingest + reconnaissance only, ahead of V8
("external eval anchor: Physion replication", see DESIGN.md's milestone
list). No OCP evaluation runs here -- see "V8 options" below for what still
needs deciding before evaluation code gets written.

Source: [cogtoolslab/physics-benchmarking-neurips2021](https://github.com/cogtoolslab/physics-benchmarking-neurips2021)
(MIT), NeurIPS 2021 Datasets & Benchmarks track. Published headline numbers
(the benchmark's own Table, human vs. model performance on the "will the red
object touch the yellow zone" Object Contact Prediction (OCP) task): humans
**71%**, particle-based GNS **~71%**, DPI **~70%**, pixel-based models
**55-65%**. These are the numbers V8 replicates against, not numbers this
milestone reproduces itself.

## Archive structure (`data/external/physion/Physion.zip`, "PhysionTest-Core")

Verified by actually unzipping the archive (`data/external/physion/extracted/`,
gitignored) and running `gltfworld.physion.ingest.PhysionIndex` /
`ffprobe`-equivalent (`imageio`) inspection against every file, not just
reading the benchmark repo's docs.

```
Physion/
  labels.csv                       # 1200 rows: trial_id -> "True"/"False"
  {Scenario}/
    mp4s/{trial_id}.mp4             # human-viewed stimulus video
    mp4s-redyellow/{id_ry}.mp4      # model-input video (see below)
    maps/{stem}_map.png             # single-frame agent/patient map
```

- **Scenarios present**: all 8 -- Dominoes, Support, Collide, Contain, Drop,
  Roll, Link, Drape. **150 trials each, 1200 total.**
- **Videos**: h264 mp4, **512x512**, **30 fps** (`imageio`/ffmpeg-read
  `r_frame_rate`), frame counts vary by trial (observed 150-570 frames,
  i.e. ~5-19s; Support's "towers" trials run longest). Every trial has a
  matching `mp4s` and `mp4s-redyellow` file (verified: 1200/1200 pairs
  resolve, zero missing).
- **`mp4s` vs. `mp4s-redyellow`**: two renders of the *same* trial, not two
  different things. `mp4s` is what was shown to human MTurk participants --
  the two objects that matter for the trial (the "agent", the thing that
  moves/is pushed, and the "patient", the thing that may or may not get
  touched) appear in **random** colors so a human doesn't get an unfair
  visual hint. `mp4s-redyellow` is the version **passed to models** -- the
  same trial, but the agent is recolored red and the patient yellow,
  consistently, so a model doesn't have to *identify* which two objects
  matter (a separate, harder problem this benchmark isn't testing) before it
  can predict whether they touch. This is the direct source of DESIGN.md's
  framing, "will the red object touch the yellow zone" -- it's literally a
  fixed color-coding convention of the input video, not a metaphor.
- **Filename convention** (verified against all 1200 trials, zero exceptions):
  `trial_id` (the `labels.csv` key and the `mp4s` filename stem, e.g.
  `pilot_it2_collision_assorted_targets_box_1_dis_1_occ_0000_img`) always
  ends in a trailing 4-digit index; the `mp4s-redyellow` filename is
  identical except `"-redyellow"` is spliced in immediately before that
  trailing index (`..._occ_0000_img.mp4` -> `..._occ-redyellow_0000_img.mp4`).
  The `maps/` filename is `trial_id` with the `_img` suffix swapped for
  `_map` (`..._occ_0000_map.png`). `trial_id` is globally unique across all
  1200 trials (no scenario-qualification needed to disambiguate).
- **`maps/*.png`**: 512x512 RGBA, **one static image per trial**, not a
  per-frame segmentation sequence. Per the benchmark repo: "PNG segmentation
  maps for each test stimulus, indicating location of `agent` object in red
  and `patient` object in yellow" -- effectively a single reference frame of
  the same red/yellow convention `mp4s-redyellow` uses throughout the video.
- **`labels.csv`**: 1200 data rows, columns `("", "ground truth outcome")`
  (the unnamed first column is the trial id, exactly matching each `mp4s`
  filename minus `.mp4`). Values are literal Python-style `"True"`/`"False"`
  strings. **Perfectly balanced**: 600 `True` / 600 `False` across the whole
  archive. This is the OCP ground truth: did the patient (yellow) object
  ever get contacted by the agent (red) object during the trial.

### What ground truth this "Core" archive does and does not carry

**Available (sufficient for pixel-based OCP evaluation, matching how the
benchmark's own pixel-model baselines -- e.g. a CNN/video-transformer
readout -- were run):**

- The rendered RGB video, at the exact frames a pixel/video model would
  consume, in both the human-viewing and model-input (red/yellow) colorings.
- The binary OCP label per trial (contact: yes/no).
- A single agent/patient reference map per trial.

**Not available here, only in the full per-scenario HDF5 releases:**

- Per-frame object state (3D pose, velocity, mesh/asset identity) -- this
  Core archive has *no* structured scene-state ground truth at all, only
  rendered pixels + the final binary label. There is nothing here to feed a
  *state-based* dynamics model (`gltfworld.models.dynamics
  .InteractionTransformer`, V5) or to convert into gltfworld's own glTF
  transport -- doing that needs the HDF5 tier.
- Per-frame depth, surface normals, optical flow, per-frame instance
  segmentation (vs. this archive's one-static-map-per-trial).
- Camera parameters (intrinsics/extrinsics) per trial/frame.
- Any train-time data at all -- this archive is **test-only**. Both a
  "dynamics training" and a "readout training" split exist, but only as
  separate HDF5/mp4 downloads (see below), not inside `Physion.zip`.

## Why this matters for the glTF-transport gap report

DESIGN.md's whole thesis is that the transport format between simulation,
rendering, and model IO shouldn't be an implementation-detail throwaway --
worth stating plainly what happened here as one more data point for that
argument: this physics-ML benchmark team, needing to carry rendered video +
per-frame object/camera state + depth/flow/segmentation + a scalar label
through a research pipeline, **did not reach for an existing structured
transport format** (glTF or otherwise). They invented a bespoke combination
instead -- ad hoc HDF5 for the rich per-frame state tier, plain MP4 for the
video tier models actually consume, and a hand-rolled CSV for the one scalar
label everything is graded against -- with the two tiers ("Core" vs. HDF5)
carrying genuinely different information and no single self-describing file
tying video frame `t` to its corresponding object/camera state. That's a real
gap: a general-purpose, self-describing transport (frames + object state +
camera + labels, one file, one loader) would have let this benchmark ship a
single artifact instead of a two-tier "video-only" / "everything, but 380GB"
split with no format in between.

## Download URLs + sizes (verified by direct HTTP HEAD request, 2026-07-28)

All hosted at `https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/`.

| file | `Content-Length` | size |
| --- | --- | --- |
| `Physion.zip` (already downloaded, this archive) | 284,049,930 | 270.9 MiB (matches repo's "270 MB") |
| `PhysionTestHDF5.tar.gz` (full test tier, all scenarios) | 408,171,036,204 | 380.14 GiB (matches repo's "~380GB") |
| `PhysionTrainMP4s.tar.gz` | 803,488,088 | 766.3 MiB |

**Per-scenario `*_testing_HDF5s.tar.gz`** (individually addressable --
confirmed by HEAD, each returns `200 OK` with a real `Content-Length`, not a
redirect/error):

| scenario | bytes | size (GiB) | rigid? |
| --- | ---: | ---: | --- |
| Collide | 35,026,607,691 | 32.62 | rigid |
| Drape | 33,391,071,156 | 31.10 | **non-rigid (cloth)** |
| Dominoes | 39,333,918,824 | 36.63 | rigid |
| Drop | 42,749,717,976 | 39.81 | rigid |
| Roll | 43,562,865,197 | 40.57 | rigid |
| Link | 63,530,489,135 | 59.17 | rigid |
| Support | 73,083,671,475 | 68.06 | rigid |
| Contain | 77,492,764,925 | 72.17 | rigid |

**Finding: no per-scenario test HDF5 file was downloaded.** The task's own
bar was "individually addressable at reasonable size <=15GB -- if one is,
download the smallest rigid scenario only." Every rigid-scenario test HDF5
is directly addressable (all `200 OK`, real sizes, no auth/redirect needed),
but **all of them are far over 15GB** -- the smallest rigid scenario,
Collide, is 32.62 GiB, more than 2x the ceiling. (Drape at 31.10 GiB is
nominally smaller but is the one non-rigid/cloth scenario, excluded by the
task's own "rigid scenario" qualifier.) Per the task's explicit fallback,
this is documented rather than downloaded.

Spot-checked two of the smaller-tier alternatives too, in case a lighter-weight
per-scenario file existed: `Collide_dynamics_training_HDF5s.tar.gz` (31.32
GiB) and `Collide_readout_training_HDF5s.tar.gz` (15.77 GiB) -- both still at
or above the 15GB bar, and both are *training* splits (irrelevant to
Core's test-only OCP replication) rather than a lighter version of the
*test* tier. No file at any tier/scenario combination checked clears 15GB.

Per-scenario training-tier files follow the same naming convention:
`{Scenario}_dynamics_training_HDF5s.tar.gz` / `{Scenario}_readout_training_HDF5s.tar.gz`
/ `{Scenario}_testing_HDF5s.tar.gz`, for `Scenario` in Dominoes, Support,
Collide, Contain, Drop, Roll, Link, Drape (all confirmed to resolve via the
two representative HEAD checks above; not every one of the 24 combinations
was individually HEAD-checked, since the smallest confirmed data point
already rules out downloading any of them under the 15GB bar).

## Ingest module

`gltfworld.physion.ingest` (`PhysionIndex`, `load_frames`) -- see the
module's own docstring for the full API. Enumerates all 1200 trials
(scenario, trial id, video/redyellow/map paths, parsed boolean label) from
an extracted `Physion.zip`; `load_frames` decodes an mp4 to an in-memory
`np.uint8 (T, H, W, 3)` stack via `imageio` (already a project dependency,
`ml` extra) with optional frame-stride/max-frame limits and a
`use_redyellow` switch. No `h5py` dependency -- this Core archive is pure
mp4/PNG/CSV, so the module stays dependency-light per this milestone's own
scope (state-based HDF5 ingest, if V8 chooses that path, is separate,
follow-on work -- see option (b) below).

Verified directly against the real, fully extracted archive (not a
synthetic fixture): `tests/test_physion_ingest.py`, skips cleanly (with a
clear reason) if `data/external/physion/extracted/` isn't present.

## Honest feasibility note for V8

**The perception model (`gltfworld.models.perception.PerceptionDETR`, V6)
will not transfer zero-shot to Physion's videos.** It's trained (once
training completes) exclusively on `perception-v1`, itself rendered by
gltfworld's own vendored-pyrender pipeline from `wm-scenes-v1` scenes --
simple primitives (sphere/box/cylinder), flat-shaded/basic materials, a
fixed camera, a bare ground plane. Physion's videos are rendered in
ThreeDWorld (TDW): textured household-object-like meshes, room interiors,
varied cameras per trial, realistic lighting. The domain gap (asset style,
texture, lighting, camera diversity, scene complexity) is large enough that
running `PerceptionDETR` directly on Physion frames and expecting usable
detections would not be a meaningful test of anything -- this is a modeling
fact to design around, not a bug to fix inside this milestone.

Three options for V8, laid out without picking one (each has a different
cost/what-it-actually-validates tradeoff):

**(a) Readout-probe protocol, matching the benchmark's own baselines.**
The benchmark's own pixel-based models don't work zero-shot either -- they
train a lightweight linear/shallow readout on top of frozen features,
using the "readout training" HDF5/mp4 split (a `{Scenario}
_readout_training_HDF5s.tar.gz` per scenario, see table above; sizes not
individually re-verified beyond the Collide spot-check, but same tier as
the training-tier numbers already confirmed). Requires: downloading at
least one scenario's readout-training data (still tens of GB per the
Collide/Dominoes spot-checks above -- none of the scenarios clear the 15GB
bar even at this tier), extracting whatever frozen features gltfworld can
produce from the readout-train videos (e.g. `PerceptionDETR`'s encoder
activations, even out-of-domain), and training a small probe against the
readout-train labels, then evaluating on Physion Core's own 1200 test
trials via `mp4s-redyellow` + `labels.csv` (both already ingested by this
milestone's `PhysionIndex`). This is the closest apples-to-apples comparison
to the benchmark's own published pixel-model numbers (55-65%), since it's
the same experimental protocol.

**(b) State-based track: HDF5 ground-truth states through our dynamics
model.** Bypasses perception entirely by using Physion's HDF5-tier
*ground-truth* object states (position/velocity/pose per
frame) as if they were `gltfworld`'s own tensor contract, run through
`gltfworld.models.dynamics.InteractionTransformer` (V5) instead of
`PerceptionDETR`. This tests the *dynamics* model's transfer, not
perception's, and sidesteps the TDW-vs-gltfworld visual domain gap
entirely -- but requires actually downloading and parsing a per-scenario
test HDF5 (32.62-77.49 GiB per the rigid-scenario table above, i.e. a real,
deliberate storage/bandwidth commitment this milestone did not make), and
requires building an HDF5 state -> gltfworld tensor-contract adapter
(different object/feature representation, different units/conventions,
untested) -- exactly the "object states -> our glTF conversion experiment"
DESIGN.md's V8 line anticipates, but as a new, nontrivial conversion layer,
not something this ingest milestone's `mp4`/`labels.csv`-only module covers.

**(c) Domain-adapted perception.** Fine-tune (or few-shot adapt)
`PerceptionDETR` itself on some Physion supervision before evaluating it on
the OCP task -- the readout-training tier again supplies frames, but this
option asks the *perception* model to adapt (weights change) rather than
training a probe on frozen features as in (a). Most faithful to "does our
actual perception model generalize" but the most expensive: needs enough
Physion frames with usable supervision for fine-tuning (not just a linear
probe's worth), a training run, and a real risk that closing the domain gap
this way just reproduces a TDW-specialized perception model rather than
demonstrating gltfworld's approach transfers -- the result would need
honest framing either way.

None of (a)/(b)/(c) is implemented or started by this milestone; this
section exists so V8 doesn't have to re-derive the tradeoffs from scratch.
