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

## V8 decision: option (b), state-based track (2026-07-30)

V8 implements **option (b)**: bypass perception entirely, convert Physion's
HDF5 ground-truth object state into gltfworld's own glTF transport, and run
`gltfworld.models.dynamics.InteractionTransformer` (V5, `dynamics-v1`
checkpoint) on the converted trials for the OCP task. This tests the
*dynamics* model's zero-shot transfer, not perception's, and produces the
"object states -> our glTF conversion experiment" DESIGN.md's V8 line
anticipates.

### Data acquired

`Collide_testing_HDF5s.tar.gz` (the smallest rigid-scenario test tier,
32.62 GiB per the table above -- verified again via a direct HTTP HEAD
immediately before download: `Content-Length: 35026607691`, exact match),
downloaded to `data/external/physion/hdf5/` and extracted to
`data/external/physion/hdf5/extracted/Collide/` (both gitignored). The
archive contains `Collide/hdf5s/` (150 per-trial `.hdf5` files, one per
Collide test trial, matching Core's 150-trial count for this scenario) plus
`Collide/hdf5s-redyellow/` (a mirrored 150 files -- appears to be the same
per-trial structure with red/yellow-coded `_id` segmentation, not
independently explored since this milestone's track needs no pixel data at
all) and two scenario-level JSON summaries, `trial_labels.json`/
`trial_labels_by_field.json` (150 entries each, keyed by `stimulus_name`,
carrying `target_id`/`zone_id`/segmentation colors/`does_target_contact_zone`
-- the same OCP outcome as Core's `labels.csv`, from inside the HDF5 tier
itself; see "cross-check" below). `h5py>=3.10` added to the `sim` extra in
`pyproject.toml` for this.

### Physion HDF5 schema (verified against real files, `Collide/hdf5s/*.hdf5`)

Each per-trial file has two top-level groups, `static` and `frames`. No
`labels.csv`-style file lives inside an individual trial's HDF5 -- OCP
ground truth is either the per-frame `target_contacting_zone` label (below)
or the scenario-level `trial_labels.json` alongside the `hdf5s/` directory.

**`static/`** (one entry per rigid body tracked in the whole trial, `N`
objects, arrays in a fixed order shared across every `static/` dataset --
verified index-consistent against `model_names`/`mesh/vertices_i`):

| dataset | shape | dtype | meaning |
| --- | --- | --- | --- |
| `object_ids` | `(N,)` | int64 | stable per-trial object id (1-based; matches `frames/*/objects/*` row order and `target_id`/`zone_id`/`probe_id`) |
| `model_names` | `(N,)` | object (bytes) | TDW asset name, e.g. `cube`, `cone`, `sphere`, `torus`, or a real furniture asset name for occluders/distractors (e.g. `linbrazil_diz_armchair`) |
| `mass` | `(N,)` | float64 | kg |
| `static_friction` / `dynamic_friction` | `(N,)` | float64 | **two** separate coefficients (`ObjectSpec.friction` only has one -- see conversion findings) |
| `bounciness` | `(N,)` | float64 | restitution |
| `scale` (+ `scale_x`/`scale_y`/`scale_z`) | `(N,3)` / `(N,)` | float32/float64 | per-axis scale applied to the unit mesh below |
| `color` | `(N,3)` | float64 | RGB in `[0,1]` (no alpha) |
| `object_segmentation_colors` | `(N,3)` | uint8 | instance-segmentation color, unrelated to the separate red/yellow OCP convention |
| `initial_position` / `initial_rotation` | `(N,3)` / `(N,4)` | float32 | redundant with `frames/0000/objects/positions`/`rotations` |
| `mesh/vertices_{i}` | `(V_i,3)` | float32 | **real per-object mesh geometry**, local unit space, index `i` = position in `object_ids` (0-based) |
| `mesh/faces_{i}` | `(F_i,3)` | int32 | triangle indices into `vertices_i` (every file checked is already triangulated) |
| `target_id` / `zone_id` / `probe_id` | scalar | int64 | which `object_ids` entry is the agent-to-be ("target", pushed into contact), the patient ("zone", a flat marker plane), and the launched impactor ("probe") -- see roles below |
| `distractors` / `occluders` | `(k,)` | object (bytes) | **model names** (not ids) of decorative objects irrelevant to the OCP question |
| `push_force` / `push_position` / `push_time` | `(3,)`/`(3,)`/scalar | float32/float32/int64 | the initial impulse applied to the probe |
| `seed` / `trial_seed` / `trial_num` / `room` / `stimulus_name` / `git_commit` | scalar | various | provenance, matches the `trial_id` used elsewhere in this project |

No `gravity` or `dt`/`framerate` field anywhere in `static/` -- see below.

**`frames/{i:04d}/`** (one group per recorded frame, `i` zero-padded to 4
digits; Collide trials are 151-152 frames, i.e. ~5.0-5.07s):

| path | shape | dtype | meaning |
| --- | --- | --- | --- |
| `objects/positions` | `(N,3)` | float32 | world position of each object's *pivot* (not its geometric center -- TDW primitives are base-pivoted, `y in [0, height]` locally, not `[-height/2, height/2]`; see conversion findings) |
| `objects/rotations` | `(N,4)` | float32 | quaternion, component order `(x,y,z,w)` -- verified empirically (not assumed): reproduces `objects/forwards` exactly via the standard right-handed quaternion-rotate formula applied to the raw, unmodified coordinates (see conversion findings for the handedness caveat this glosses over) |
| `objects/velocities` / `angular_velocities` | `(N,3)` | float32 | **real, simulator-native per-frame velocities** -- no finite-differencing needed (unlike `wm-scenes-v1` episodes without a `lin_vel`/`ang_vel` series) |
| `objects/forwards`/`front`/`back`/`top`/`bottom`/`left`/`right` | `(N,3)` | float32 | world-space points on the object's oriented bounding box (face centers) plus the forward direction; not used by this milestone's conversion (real mesh + `positions`/`rotations` already fully determine these) |
| `objects/center` | `(N,3)` | float32 | world-space true geometric center (vs. `positions`' pivot) |
| `camera_matrices/camera_matrix` | `(16,)` | float32 | row-major 4x4 world-to-camera view matrix; **fixed for the whole trial** (verified bit-identical frame 0 vs. frame 50); rotation block has **determinant -1** under a naive right-handed read (see conversion findings) |
| `camera_matrices/projection_matrix` | `(16,)` | float32 | row-major 4x4; `[1,1] = 1.9209819` -> a ~59.6 degree vertical FOV under the standard `1/tan(fovy/2)` OpenGL convention |
| `collisions/*`, `env_collisions/*` | varies | float32/int32/int64/`\|S1` | discrete contact-event records (object-object and object-environment); not carried into the glTF conversion (continuous per-frame state is; see findings) |
| `images/_img`, `_id`, `_depth`, `_flow`, `_normals` | varies (compressed byte streams, `_depth` raw `(512,512,3)` uint8) | uint8 | rendered pixel data -- **not read at all** by this milestone (state-based track, no pixels needed; also the overwhelming majority of each file's size) |
| `labels/target_contacting_zone` | scalar | bool | **the actual OCP signal, computed by the simulator itself, per frame** -- the trial-level OCP label is `any(target_contacting_zone across all frames)` |
| `labels/has_target`/`has_zone`/`target_has_moved`/`target_on_ground`/`target_delta_position`/`trial_end`/`trial_complete`/`trial_timeout` | scalar/`(3,)` | bool/float32 | auxiliary simulator bookkeeping, not otherwise used here |

**Roles** (agent/patient framing from the top of this doc, in this HDF5's
own vocabulary): `target_id` = **agent** (red in `mp4s-redyellow`; the object
that might touch the zone), `zone_id` = **patient** (yellow; a flat marker
region, empirically static -- see findings), `probe_id` = the launched
impactor that starts the chain reaction (not specially colored in
`mp4s-redyellow`, per `trial_labels.json`'s `target_segmentation_color`/
`zone_segmentation_color` fields only ever naming target+zone). Any
remaining `object_ids` not in `{target_id, zone_id, probe_id}` are
distractors/occluders (by `model_names` membership in `static/distractors`/
`static/occluders`).

**Cross-check**: `trial_labels.json`'s `does_target_contact_zone` is used
directly as this milestone's OCP ground truth (matches Core's `labels.csv`
by construction -- same benchmark, same 150 Collide trials; not
independently re-diffed row-by-row here since `trial_labels.json` lives
inside the exact tier this milestone downloads, one join away from Core's
labels.csv, and is more convenient -- keyed by `stimulus_name`, no
`_img`-suffix stripping needed).

**`dt` is not stored anywhere in the file.** Inferred as **1/30 s**: (a)
Collide trials run 151-152 frames, i.e. 5.03-5.07s at 1/30s/frame, matching
docs' own "5-19s" duration range for the *shortest* scenario; (b) a direct
per-frame finite-difference check (`dpos` vs `v_avg * dt`) does *not*
reproduce 1/30s cleanly (order 0.009-0.014s instead) -- but this is expected,
not contradictory: like `wm-scenes-v1`'s own generator (500 Hz internal,
~30 Hz recorded, see DESIGN.md), TDW's underlying physics almost certainly
substeps faster than the recorded-frame rate, so naive single-step
finite-differencing of *recorded* frames doesn't recover the *true* internal
integrator step. `dt=1/30` is what this milestone uses for every downstream
per-recorded-frame computation (the same semantics `wm-scenes-v1`'s
`scene.dt`/`series.times` already use).

**Gravity is not stored anywhere in the file either.** Used as TDW/Unity's
documented default, `(0, -9.81, 0)` -- an assumption, not independently
re-derived per trial (a direct finite-difference re-derivation was tried and
rejected as unreliable for the same substep reason as `dt` above).

## Physion conversion findings (V8 stage 2, `gltfworld.physion.convert`)

Every real impedance mismatch hit converting HDF5 -> glTF, in the order a
reader would hit them. This is the primary gap-report evidence this
milestone produces (DESIGN.md's V9 scope draws on this section directly).
Verified against 3 real converted Collide trials (see
`tests/test_physion_convert.py`): validator **0 errors**, real mesh
POSITION/NORMAL/indices accessors round-trip bit-exact, pose-animation frame
count equals the source HDF5's frame count exactly, and poses/velocities/
physics metadata round-trip through the *existing*, unmodified
`gltfworld.scene.convert.episode_from_gltf` to <=1e-5 absolute.

1. **No normals at all.** HDF5 carries only `mesh/vertices_i`/`faces_i` --
   raw positions and triangle indices, no per-vertex or per-face normal
   data anywhere. `compute_vertex_normals` derives them (area-weighted
   per-vertex average of adjacent face normals, an un-normalized
   cross-product magnitude as the area weight) -- a real, lossy
   approximation of whatever smooth/authored normals the original TDW
   asset actually used for shading, unrecoverable from triangulated face
   data alone. Every converted mesh's normals are therefore *faceted*
   relative to the original renders (visible on any smoothly-curved asset,
   e.g. the `sphere`/`torus`/`cone` primitives seen in Collide), not a
   faithful reproduction.

2. **Mesh pivot vs. geometric center.** TDW's own primitives are
   base-pivoted: local vertex Y ranges `[0, height]`, not `[-height/2,
   height/2]` (verified: the `cube`/`cone`/`sphere` assets in Collide all
   have `vmin.y == 0`). `frames/*/objects/positions` is this pivot, not the
   object's true center (`frames/*/objects/center` is a *separate*,
   unused-by-this-conversion dataset that *is* the true center). gltfworld
   reuses the raw pivot position as the glTF node's translation directly
   (so the *visual* mesh -- real vertices, unmodified -- renders in exactly
   the right place), but this means the **KHR-physics collider
   approximation is generally displaced from the visual mesh's true
   center** along the local pivot-to-center vector: our pinned
   `KHR_implicit_shapes` subset (`gltfworld.ext.khr_physics`) has no
   local-offset/local-transform field for a shape, only an implicit
   "centered on the node origin" assumption -- a real schema limitation
   this conversion surfaces, not something gltfworld's codec chose to
   paper over. (`classify_bounding_shape`'s returned `center` *is* computed
   correctly and is available to any caller that wants the true center for
   other purposes -- see stage 3's contact-geometry math, which does use
   it -- it's specifically the *KHR-physics encoding* that has nowhere to
   put it.)

3. **Two friction coefficients collapse into one.** HDF5 carries
   `static_friction`/`dynamic_friction` as two independent per-object
   values (real Coulomb-friction physics distinguishes them);
   `ObjectSpec.friction` (and `KHR_physics_rigid_bodies`' physics-material
   schema, as gltfworld encodes it) has exactly one friction field. This
   conversion averages the two -- a real, lossy information collapse (not
   a bug in either format; gltfworld's own physics-material encoding was
   built against `wm-scenes-v1`'s single-friction-value physics distribution
   and never needed a second coefficient before).

4. **No generic/convex mesh collider type, so *some* shape-type
   approximation for physics is unavoidable regardless of transport
   quality.** `classify_bounding_shape`'s AABB-isotropy sphere/box
   classifier (shared verbatim with stage 3's dynamics-tensor mapping, one
   documented approximation instead of two) cannot represent a cylinder at
   all (gltfworld's whole shape vocabulary, `gltfworld.scene.scene.SHAPES`,
   trained-model-compatible or not, is only sphere/box/cylinder) --
   distinguishing a true cylinder from a box by AABB alone isn't possible
   without also inspecting the mesh's actual radial symmetry, not attempted
   here. Every non-sphere Physion asset (`cone`, `torus`, `dumbbell`,
   `linbrazil_diz_armchair`, ...) is therefore boxed, however round it
   visually is -- confirmed directly against the 3 converted trials:
   `cone`/`torus`/`dumbbell` (the `target`/agent role in each) all
   classified `"box"`.

5. **Chirality/handedness: TDW/Unity is left-handed; gltfworld's transport
   convention is right-handed.** Verified two ways: (a) the per-object
   quaternion (`objects/rotations`, component order `x,y,z,w`) reproduces
   `objects/forwards` *exactly* via the standard right-handed
   quaternion-rotate formula applied to the raw, unmodified HDF5
   coordinates -- self-consistent within TDW's own frame; (b) the camera's
   `camera_matrix` rotation block has **determinant -1** under that same
   naive right-handed read (not a numerical artifact -- checked
   `R @ R.T == I` to confirm it's a genuine orthonormal-but-improper
   transform, not noise), consistent with a left-handed world composed
   with a chirality-flipping camera projection. gltfworld deliberately
   does **not** apply any chirality-correcting flip anywhere in this
   conversion: positions/velocities/quaternions/mesh vertices are all
   reused as-is. This has **zero effect on every metric stage 3 computes**
   (Euclidean distance, contact thresholds, velocity magnitude, and the
   dynamics model's own features are all invariant under a single global
   mirror of all coordinates together) -- but it does mean a GLB from this
   conversion, rendered by an ordinary right-handed glTF viewer, would show
   a left-right-mirrored scene relative to Physion's own rendered
   `mp4s`/`mp4s-redyellow` for the *same* trial. Documented as a known,
   deliberately-unaddressed, low-impact limitation, not a silent bug.

6. **Camera orientation is synthesized, not decoded.** Because of finding
   5, `_decode_camera` does not attempt to turn the real `camera_matrix`
   into a glTF camera rotation (that would require resolving which axis
   the chirality flip belongs to, which the data alone doesn't disambiguate
   without a rendered-pixel ground truth to check against -- out of this
   milestone's scope). It uses the matrix's *translation* half (handedness-
   independent: `camera_position = -R^T @ t`, a real, correctly-decoded
   world position) and synthesizes a look-at orientation aimed at the
   scene's frame-0 object centroid instead. `yfov` is read from the
   projection matrix's `[1,1]` entry under the standard OpenGL
   `1/tan(fovy/2)` convention (not independently cross-checked against a
   rendered frame). The encoded camera is therefore a plausible stand-in,
   not a faithful reproduction of Physion's actual render camera --
   immaterial to stage 3 (state-based, no rendering involved) but relevant
   to anyone using these GLBs for visual debugging.

7. **No true ground/floor object at all.** TDW's room/level geometry
   (floor, walls) is static level geometry, not a tracked rigid body --
   it never appears in `static/object_ids` and has no per-frame pose. The
   `zone` object (patient role) is a thin, per-trial-positioned flat marker
   plate, empirically confirmed static (position/rotation bit-identical
   across every frame checked) but **not** a stand-in for the floor (it
   sits *on* the floor, at an arbitrary per-trial position/height).
   Converted trials therefore carry **no ground-plane object whatsoever**
   -- a real, load-bearing mismatch against `wm-scenes-v1`'s convention
   (every episode has exactly one static ground box the dynamics model's
   learned "ground" token was trained to attend against, per DESIGN.md's
   V5 section); see docs/RESULTS.md's V8 section for how this plays out in
   the dynamics-transfer numbers.

8. **Discrete collision-event records dropped.** `frames/*/collisions/*`
   and `frames/*/env_collisions/*` (contact points, per-event relative
   velocity, discrete enter/stay/exit state strings) are real, richer
   physics-engine output than gltfworld's transport has any channel for
   (`RWM_state_series` carries continuous per-frame state, not discrete
   events) -- not read or carried into the GLB at all. A real gap in the
   *reverse* direction from most findings above: here it's gltfworld's
   schema, not Physion's, that has nothing to receive the richer
   information Physion actually offers.

9. **One real positive finding (not a mismatch): velocities need no
   finite-differencing.** Unlike a `wm-scenes-v1` episode that lacks a
   velocity series, Physion's HDF5 tier writes real, simulator-native
   per-frame `velocities`/`angular_velocities` directly -- gltfworld's
   `RWM_state_series` channel for these is populated from genuine physics-
   engine state, not a numerical approximation, for every converted trial.

10. **Pixels dropped entirely, by design.** `frames/*/images/*` (rgb/seg/
    depth/flow/normals -- also the overwhelming majority of each file's
    bytes) is never read. Consistent with this milestone's state-based
    scope (option (b) from the "V8 options" section above) and with a
    pre-existing, more general observation: glTF itself has no
    video-frame-sequence concept, so this isn't specific to Physion.

11. **Variable per-trial object count (3, 5, or 7 in Collide alone,
    `target`+`zone`+`probe` plus 0/1/2 distractor-occluder pairs).**
    Not actually a problem for `InteractionTransformer` (permutation-
    equivariant, no hardcoded `N_max` in the model itself -- only
    `wm-scenes-v1`'s training-data *packer* padded to `N_max=5`, see
    DESIGN.md's V4 section) but a real domain-gap dimension worth flagging
    for stage 3: some other Physion scenarios (not Collide) go well beyond
    `wm-scenes-v1`'s `N<=5` training distribution.

Three more findings, hit only once every one of the 150 Collide test trials
was actually converted (not just the first 3):

12. **Distractor/occluder objects have no physics metadata at all.**
    `static/mass`/`static_friction`/`dynamic_friction`/`bounciness` are only
    ever as long as the physically-relevant object count
    (`target`+`zone`+`probe` -- always 3), *not* the full `object_ids`
    length (5 or 7 when distractors/occluders are present -- 23 trials have
    3 objects, 64 have 5, 63 have 7, across the full 150). Verified this is
    benign, not silently wrong: every such object is also empirically static
    (zero pose change across every frame checked, same test as the finding
    on `zone` above), so a placeholder mass/friction/restitution (documented
    in code, `convert.py`) is physically inert -- `KHR_physics_rigid_bodies`
    omits `motion` for a static body regardless of what its mass field says.
13. **Some occluder/distractor real-world asset meshes are simply empty.**
    13 model names (`amphora_jar_vase`, `648972_chair_poliform_harmony`,
    `shar_pei`, `animal_dog_rtsit_1280`, ...) have `static/mesh/vertices_i`
    with **zero rows** in every trial that uses them -- TDW's mesh-export
    step evidently can't (or doesn't) dump geometry for every asset kind,
    even though the object is fully tracked (position/rotation/scale) like
    any other. `gltfworld.physion.convert.load_trial` substitutes a
    placeholder unit box (scaled like the real object would be) so the
    trial still converts to a complete, valid GLB -- never affects the
    OCP-relevant geometry (never `target`/`zone`/`probe` in any of the 150
    trials checked).
14. **Real Physion model names don't fit gltfworld's own tensor-contract
    category vocabulary.** `gltfworld.scene.contract.episode_to_tensors`
    (built for `wm-scenes-v1`'s `{"ball", "crate", "cylinder"}` world) raises
    on any other `ObjectSpec.category` string -- real Physion names like
    `"cone"`, `"torus"`, `"648972_chair_poliform_harmony"` all fail it.
    Stage 3 (`gltfworld.physion.ocp_eval`) works around this with a
    shape-based remap (sphere -> ball, box -> crate) applied to a throwaway
    copy of the decoded `Episode` used only for the tensor-contract call --
    the GLB's own `extras.rwm.category` (the real model name) is never
    touched. `class_ids` (the only thing this remap affects) isn't consumed
    by the dynamics model at all (only `shape_onehot` is), so this is a
    zero-cost workaround for a real, narrower-than-it-looks contract
    assumption -- worth fixing generally (accept an arbitrary category
    string, or make `class_ids` optional) if gltfworld's tensor contract is
    ever extended beyond `wm-scenes-v1`.
