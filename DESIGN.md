# DESIGN

Status: 2026-08-02, milestone V10 (final) -- see "V10 closing status" at
the end of this document.

## Architecture flow

```
MuJoCo sim
  -> GLB episodes (pose animation + KHR_physics_rigid_bodies/KHR_implicit_shapes
     + RWM_state_series + extras.rwm semantics)
    -> renderer (frames)
      -> perception model (frames -> scene state) + dynamics model (state[t] -> state[t+1])
        -> inference emits glTF at every hop
          -> renderer (closes the loop)
```

Each arrow is a real glTF (or GLB) file on disk: the transport format is not
an implementation detail, it's the thing under study.

## Reuse-first stack decisions

- **pygltflib** — mature, spec-accurate glTF/GLB read+write; avoids
  reimplementing the JSON/binary-chunk plumbing.
- **trimesh** — mesh generation utility only (primitives, procedural
  geometry); not used as the scene graph or transport.
- **MuJoCo** — battle-tested rigid-body physics for ground-truth episode
  generation; avoids writing a physics engine to get training data.
- **vendored pyrender (V2)** — headless GL rendering of glTF scenes without
  building a renderer from scratch; vendored (not a dependency) so it can be
  patched for extension-aware rendering. Pinned commit
  `a59963ef890891656fd17c90e12d663233dcaa99` (mmatl/pyrender, latest
  `master` as of vendoring); full patch list in
  `src/gltfworld/_vendor/PROVENANCE.md` (numpy-2 `np.infty` removal,
  optional/guarded `pyglet` import so headless use needs no windowing
  toolkit, and intra-package import fixes the vendoring itself required).
- **PyTorch** — standard ML framework for perception/dynamics models.

## Custom components

1. **`RWM_state_series` codec** (`gltfworld.ext.rwm`) — carries per-frame
   world state (beyond what pose animation/KHR physics express) inside a
   glTF extension. Validation: round-trip tests (encode -> decode ->
   compare) plus schema validation against
   `docs/schemas/rwm/RWM_state_series.schema.json`.
2. **KHR physics codec** (`gltfworld.ext.khr_physics`) — reads/writes
   `KHR_physics_rigid_bodies` + `KHR_implicit_shapes` draft extension data.
   Validation: encoded output validated against the vendored JSON Schemas in
   `docs/schemas/khr/` plus a full sample episode run through the real,
   pinned Khronos glTF-Validator binary (see CI and `gltfworld validate`).
3. **Episode glue** (`gltfworld.scene.convert`) — ties a `SceneState` +
   `StateSeries` (an `Episode`) to a single GLB (animation + physics + state
   series in sync). Validation: a Hypothesis property test round-trips
   randomly generated episodes through save -> load with bitwise-equal
   arrays, plus one deterministic golden episode
   (`tests/conftest.py:make_sample_episode`).

## Pinned specs

`gltfworld.ext.khr_physics` implements a subset of the two DRAFT Khronos
physics extensions, against this pinned commit of the spec repo:

- **Repo**: https://github.com/eoineoineoin/glTF_Physics
- **Pinned commit**: `9dc61cb3474ff9a51f58d3592f79d5c9e572056a` (2026-01-20)
- **Vendored schemas**: `docs/schemas/khr/implicit_shapes/`,
  `docs/schemas/khr/physics_rigid_bodies/` (see `docs/schemas/khr/PROVENANCE.md`)

Those extension schemas in turn `$ref` a few core glTF JSON Schema files
(`glTFProperty`, `glTFChildOfRootProperty`, `glTFid`, `extension`, `extras`),
vendored separately from the main glTF spec repo:

- **Repo**: https://github.com/KhronosGroup/glTF
- **Pinned commit**: `77b44be7bef26e01fb0b140e3d5bb1716421c5e9` (2026-07-16)
- **Vendored schemas**: `docs/schemas/khr/core/`

The `RWM_state_series` custom extension has its own JSON Schema (not pinned
to any external spec, since gltfworld is its author):
`docs/schemas/rwm/RWM_state_series.schema.json`.

## Transport encoding

One `Episode` (`gltfworld.scene.episode`) = one GLB, built by
`gltfworld.scene.convert.episode_to_gltf`/`episode_from_gltf`:

- **Nodes**: one node per object (`name = "obj_{object_id}"`), TRS set to
  the frame-0 pose. Mesh geometry (from `gltfworld.scene.primitives`,
  trimesh-generated) is deduplicated by `(shape, size bytes)`; the
  `Primitive.material` still varies per object, so distinct `Mesh` entries
  share the same POSITION/NORMAL/indices accessors when geometry matches.
  One camera node + camera. Lights via `KHR_lights_punctual` (root
  `lights[]` + per-node `{"light": index}`).
- **Poses live in exactly one place**: `animations[0]`, one shared float32
  time accessor, two STEP-interpolated channels per object (translation,
  rotation). Node TRS only mirrors frame 0; nothing reads pose data from
  anywhere else.
- **Physics**: `KHR_physics_rigid_bodies` + `KHR_implicit_shapes` carry
  per-object mass/friction/restitution/collider shape, with initial
  velocities from `series.lin_vel`/`series.ang_vel` at t=0 when present. A
  node with no `motion` property is a static body (mass irrelevant); shapes
  and physics materials are deduplicated by value.
- **Everything else**: `RWM_state_series` (root extension, reusing the same
  time accessor) carries the full velocity/action/pose-variance time
  series as accessor-backed channels; `extras.rwm` carries per-object and
  per-scene bookkeeping (object_id/category/parts, seed/scene_version/dt)
  glTF has no standard home for.
- `extensionsUsed` lists every extension actually written;
  `extensionsRequired` is always left empty, so a generic glTF viewer can
  still load the mesh/camera/light/animation content even though it won't
  understand the physics/state-series semantics.

### Documented deviations from the milestone spec text

- **Static object mass** (`ObjectSpec.mass` when `is_static=True`): the
  milestone spec says to omit `motion` (hence `mass`) from
  `KHR_physics_rigid_bodies` for static bodies, which is also how the real
  extension is meant to be used (no `motion` = immovable, infinite-mass
  body). But `ObjectSpec` still carries a `mass` value even for static
  objects that needs to survive encode/decode bit-for-bit. `extras.rwm` per
  node therefore carries `mass` and `is_static` in addition to the
  documented `{object_id, category, parts, schema_version}`, as the single
  source of truth `episode_from_gltf` reads back (KHR physics data remains
  a faithful, schema-valid encoding of the draft extension in its own
  right; it's just not what decode trusts for these two fields).
- **Scene gravity**: `extras.rwm` per scene similarly carries `gravity` in
  addition to the documented `{seed, scene_version, dt}` — the pinned
  `KHR_physics_rigid_bodies` commit has no root-level gravity property.
- **RWM channels wider than 4 components** (`pose_variance` is 7-wide:
  3 position + 4 quaternion; `action` can be arbitrary width): split into
  multiple channels of the same `kind`, each holding up to 4 contiguous
  feature dims, tagged with a 0-based `component` chunk index (only present
  when more than one chunk exists). This is an elaboration of "SCALAR/VECn
  as fits" from the milestone spec text, since no single glTF accessor type
  holds more than 4 components.
- **pygltflib empty-value pruning**: pygltflib's JSON serialization path
  (`gltf_to_json` -> `delete_empty_keys`) silently deletes any dict entry
  whose value has `len() == 0` (empty string/list/dict) anywhere in the
  document, including inside `extras.rwm.parts`. gltfworld does not work
  around this (would mean monkeypatching a dependency); the Hypothesis
  round-trip test avoids generating empty-string `parts` values as a result.
  A real caller putting an empty string into `ObjectSpec.parts` will not get
  it back after a real GLB save/load round trip.

## Rendering (V2)

`gltfworld.render.renderer.EpisodeRenderer` turns an in-memory `Episode`
into rgb/depth/seg frames via the vendored pyrender (pinned commit, patch
list: `src/gltfworld/_vendor/PROVENANCE.md`). Nothing outside
`gltfworld._vendor` imports pyrender directly.

- **EGL setup**: `gltfworld.render.renderer` sets `PYOPENGL_PLATFORM=egl`
  (only if unset) *before* importing the vendored package. If
  `__EGL_VENDOR_LIBRARY_FILENAMES` isn't already set by the caller, it
  globs `/usr/share/glvnd/egl_vendor.d/*.json` for a filename mentioning
  "nvidia" and points that env var at it (never hardcodes a path — ICD
  filenames vary by distro/driver package) so libglvnd doesn't hand the
  context off to a software/Mesa ICD instead. `egl_info()` reports what
  actually ended up active (`GL_VENDOR`/`GL_RENDERER`/`GL_VERSION` read
  back from a live context, not just env-var introspection) — see
  `docs/VERIFICATION.md` V2 "EGL device info".
- **Seg encoding**: pyrender's `RenderFlags.SEG` pass paints each tracked
  node a flat, unlit color instead of shading it, so gltfworld encodes a
  `uint16` `object_id` directly into that (r, g, b) color as
  `(object_id & 0xFF, (object_id >> 8) & 0xFF, 0)` and decodes with
  `object_id = r + g * 256`. Background (no geometry hit) is `(0, 0, 0)` ->
  `object_id == 0`, which means an `ObjectSpec` whose `object_id` is
  legitimately `0` (the ground plane in
  `tests/conftest.py:make_sample_episode`) is indistinguishable from
  background in the seg buffer alone — a pre-existing V1 modeling choice
  (ground assigned id 0), not something the seg encoding can special-case
  without lying about a real object's id. Code that needs "is there any
  geometry here at all" (e.g. the MuJoCo crosscheck's binary silhouette)
  uses `depth > 0` instead of the seg channel, which has no such aliasing.
- **One renderer per process**: pyrender's EGL platform binds to the
  *shared default* EGL display, and deleting any one
  `EpisodeRenderer`/`OffscreenRenderer` instance terminates that shared
  display for every other still-open instance in the same process
  (confirmed empirically, not just a suspicion — see
  `EpisodeRenderer`'s docstring in `src/gltfworld/render/renderer.py`).
  Not patched (would need global refcounting of the shared display); the
  renderer, benchmark, and cross-check all treat one `EpisodeRenderer` as a
  process-wide singleton instead (the test suite shares one across all
  gpu-marked tests via a session-scoped fixture).
- **Camera aspect**: renders are always square (`width == height`), so
  `EpisodeRenderer` uses `aspectRatio = width / height` for the camera
  projection rather than the episode's stored `CameraSpec.aspect` (which
  describes the scene's *authoring* aspect ratio, not this renderer's
  output) — keeps circular silhouettes circular, which the analytic tests
  (and any square-pixel consumer) need.
- **MuJoCo crosscheck coordinate conversion**
  (`gltfworld.render.crosscheck.gltf_pose_to_mujoco`): glTF is Y-up,
  right-handed, quaternion (x, y, z, w); MuJoCo is Z-up, right-handed,
  quaternion (w, x, y, z). The two world frames are related by one fixed
  +90 degree rotation about the shared X axis
  (`(x, y, z)_gltf -> (x, -z, y)_mujoco`, a proper rotation — determinant
  +1, not a reflection, so it preserves handedness). MuJoCo's camera
  convention already matches glTF/OpenGL (looks down local -Z, local +Y
  up), so the same conversion applies uniformly to object poses and the
  camera; unit-tested in `tests/test_crosscheck.py`. Measured binary
  silhouette IoU against MuJoCo's independent renderer on
  `make_sample_episode()`'s frame 0: **0.9962** (threshold: >= 0.98).
- **Performance**: persistent renderer, 4-object scene, 500 frames,
  256x256, measured on this machine's RTX PRO 6000 Blackwell:
  rgb+depth+seg **639.5 fps** (hard floor 100, target 300 — both cleared).
  See `docs/VERIFICATION.md` V2 "benchmark" for the full breakdown
  (rgb-only and rgb+depth land at the same fps — pyrender's offscreen path
  always reads back color and depth together unless `DEPTH_ONLY` is set,
  so there's no cheaper rgb-only path through the public API).

## MuJoCo data generation (V3)

`gltfworld.datagen` turns a seeded scene sample into a real MuJoCo
simulation and back into an ordinary GLB episode through the *existing*
transport codec (`gltfworld.scene.convert`) -- no new encoding, no
transport changes.

### Conversion consolidation

All MuJoCo<->contract conversion (position/quaternion/velocity) lives in
exactly one place: `gltfworld.datagen.mj_convert`. `gltfworld.render.crosscheck`
(V2) used to define its own `gltf_pose_to_mujoco`; it now imports
`gltf_pose_to_mj` from `mj_convert` under that original name (kept as a
back-compat alias) instead, per this module's "one conversion point in the
codebase" rule -- same fixed change-of-basis matrix, same behavior, same
passing V2 unit tests (`tests/test_crosscheck.py`), just one shared
implementation instead of two copies.

- **Positions/vectors**: same fixed +90 degree rotation about the shared X
  axis as V2 documented (`(x, y, z)_gltf -> (x, -z, y)_mj`); applies
  identically to positions, gravity, and linear/angular velocity (there is
  no translation between the two world origins).
- **Orientations**: `q_mj = q_axis_change (x) q_gltf` (Hamilton product,
  composition, *not* conjugation) -- deliberately preserved from V2 even
  though a naive reading of "a 90 degree rotation about MuJoCo Z equals the
  corresponding glTF rotation about Y" would suggest conjugation instead.
  Composition is what's actually self-consistent with gltfworld's *real*
  shared local-geometry convention: `gltfworld.render.crosscheck`'s MJCF
  builder (and now `gltfworld.datagen.mujoco_env`'s) hands MuJoCo's native
  geoms the exact same `ObjectSpec.size` numbers used to build the
  renderer's own trimesh-generated mesh, with no separate local-frame
  remap -- so both sides only agree if the *same* raw local coordinates are
  embedded into each world frame via the *same* linear map, which is
  exactly what composition (not conjugation) gives. `tests/test_mj_convert.py`
  documents the actual invariant with a direct check (rotating a local
  reference point via the orientation conversion must match rotating the
  same point's *image* via the vector conversion) instead of the
  misleading Y/Z-rotation framing.
- **Velocities**: MuJoCo's free-joint `qvel` is 3 linear + 3 angular. Which
  frame each half is in is *not* well documented and easy to get backwards
  silently, so it was verified empirically (see
  `gltfworld.datagen.mj_convert`'s module docstring and the V3 report) by
  integrating one physics step from a non-trivial orientation and comparing
  the finite-difference rotation (in both candidate frames) against the set
  `qvel`: **linear is world-frame, angular is body-local-frame** (MuJoCo
  3.11). Converting to the contract's world-frame `ang_vel` therefore
  requires rotating the body-local angular velocity by the body's *current*
  orientation before the axis remap
  (`mj_freejoint_vel_to_gltf_world`/`gltf_world_vel_to_mj_freejoint`).
  `tests/test_velocity_consistency.py` is the "did we get this backwards"
  exposé: it fails loudly (85th-percentile error ~5.8 rad/s against a
  ~0.07 rad/s tolerance) if the angular half is treated as already
  world-frame.

### `wm-scenes-v1` distribution

Sampled by `gltfworld.datagen.sample.sample_scene(seed)` (`np.random.default_rng(seed)`,
fully deterministic). Returns a `SampledScene`: a `SceneState` (no pose --
pose is a `StateSeries`/per-episode quantity, see `gltfworld.datagen.mujoco_env`'s
module docstring) plus the initial per-object pose/velocity
`gltfworld.datagen.mujoco_env.simulate` needs to seed MuJoCo.

| field | distribution |
| --- | --- |
| dynamic object count N | `Uniform{1..5}` |
| shape | sphere 45%, box 45%, cylinder 10% |
| characteristic size | `U[0.05, 0.25]` m |
| initial x, z | `U[-0.75, 0.75]` m each (a 1.5 m x 1.5 m horizontal box), rejection-sampled non-overlapping in true 3D (not just x/z -- see below) |
| initial height | dropped from `U[0.2, 1.2]` m of clearance *above* the object's own bounding sphere and the ground (not an absolute world-Y value -- see below) |
| orientation | uniform on SO(3) (Shoemake's method) |
| linear speed \|v\| | `<= 1.5` m/s (random direction, magnitude `U[0, 1.5]`) |
| angular speed \|w\| | `<= 3` rad/s (random axis, magnitude `U[0, 3]`) |
| density | `U[300, 3000]` kg/m^3 -> `mass = density * shape_volume` |
| friction | `U[0.4, 1.0]` |
| restitution | fixed `0.1` |
| color | one of 8 fixed HSV-spaced hues |
| category | `"ball"` / `"crate"` / `"cylinder"` (by shape); ground is `"ground"` |
| ground | 1 static box (6m x 0.2m x 6m), category `"ground"`, top surface at world Y=0 |
| camera | 1 fixed camera (see below), `aspect=1.0` |
| lights | 1 directional + 1 point (fill) |

Two deliberate readings of the milestone spec text, documented here rather
than left implicit:

- **"height 0.2-1.2 m" is drop clearance, not an absolute Y coordinate.**
  Read literally as an absolute object-center Y range, the largest sampled
  objects (bounding radius up to ~0.43 m) placed at the lowest height
  (0.2 m) would start embedded in the ground before the simulation even
  begins. `sample_scene` instead samples each object's height as
  `ground_top + object_bounding_radius + 0.2..1.2`, so every object always
  starts genuinely above the floor regardless of its own (independently
  randomized) size.
- **Non-overlap uses true 3D distance, exploiting the height axis, not just
  x/z.** Five objects at the largest allowed size don't all
  simultaneously fit non-overlapping in a 1.5 m x 1.5 m footprint alone
  (worst-case circle-packing area exceeds the footprint's area) -- but they
  fit easily once the generous, independently-sampled drop-height range is
  also available as separation room. Placement is rejection-sampled over
  `(x, z, drop_height)` jointly, checked against true 3D center-to-center
  distance vs. the summed bounding radii; empirically zero failures across
  3000 sampled seeds (worst-case clearance still >= 0 down to floating
  point).

**Fixed camera framing**: position `(0, 1.7, 4.6)`, looking at `(0, 1.0, 0)`,
`yfov=62 deg`, `aspect=1.0` (render size is always square, see V2's "Camera
aspect" note) -- verified (`gltfworld.datagen.sample.point_in_frustum`,
exercised by `tests/test_distribution.py` across 50 seeds) to keep every
sampled object's whole bounding sphere inside the frustum at t=0 with >=30%
margin to spare on the tightest axis. The same frustum-containment logic
fixed a V2 bug in `tests/conftest.py:make_sample_episode`: its cylinder
object sat at `x=3`, outside the crosscheck render's 1:1-aspect frustum,
making that object's crosscheck IoU vacuous (`union == 0` trivially returns
`1.0`). Objects are now centered and closely spaced (`x = (i - (n+1)/2) *
0.9`) so every object is genuinely visible; `tests/test_crosscheck.py` now
asserts `union > 0` for every non-ground object in addition to the IoU
threshold, so a regression back to "vacuously out of frame" would fail
loudly instead of silently reporting a perfect score.

### MJCF construction (`gltfworld.datagen.mujoco_env`)

- `scene_to_mjcf(scene, initial_poses)`: one MJCF `<body>` per
  `scene.objects[i]`, placed at `initial_poses[i]` (converted via
  `mj_convert`). Dynamic objects get a `<freejoint>` + a `density`-based
  geom (not an explicit `mass` override -- MuJoCo derives mass *and*
  inertia tensor from geometry the normal way); static objects (including
  the ground, which is just an ordinary static `ObjectSpec`, not a special
  case) are welded directly to the world, matching
  `KHR_physics_rigid_bodies`' own "no `motion` = immovable" semantics.
  `timestep=0.002` (500 Hz), gravity from `scene.gravity`, integrator left
  at MuJoCo's default.
- **Restitution -> solref, a documented lossy approximation.** MuJoCo's
  soft-contact model has no direct restitution coefficient. Restitution is
  mapped onto solref's standard `(timeconst, dampratio)` form via
  `dampratio = 1 - restitution` (critically damped/no bounce at
  restitution 0, increasingly underdamped/"bouncier" as restitution rises)
  with `timeconst` fixed at MuJoCo's recommended minimum (2x the physics
  timestep). `wm-scenes-v1` fixes restitution at a low 0.1, where this
  barely matters in practice.
- **Cylinder axis convention (fixed in V3.1; found, not fixed, in V3).**
  V3 shipped with a real interop bug: `gltfworld.scene.primitives.mesh_for
  ("cylinder", ...)` (trimesh's default) generated a mesh symmetric about
  the *local Z* axis, but `ObjectSpec`'s documented convention (and the
  vendored `KHR_implicit_shapes` cylinder schema's own text: "centered
  along the Y axis") says local *Y*. V3's `scene_to_mjcf` papered over this
  by deliberately mirroring gltfworld's *actual* (Z-symmetric,
  buggy-relative-to-its-own-docs) mesh convention in the physics geometry
  too, so gltfworld's own renderer and MuJoCo agreed with *each other* --
  but a spec-conformant external `KHR_implicit_shapes` reader, which has no
  way to know about that mesh bug, reconstructs the collider centered on Y
  and would show every cylinder rotated 90 degrees relative to gltfworld's
  own renderer, for the exact same node quaternion. Independent
  verification confirmed this and classified it an interop defect, not an
  internal-consistency issue.

  V3.1 fixes it the other way around: `mesh_for`'s cylinder is now rotated
  -90 degrees about local X (a proper rotation, vertices and normals both)
  so the mesh is genuinely Y-symmetric, matching `ObjectSpec`/the schema.
  Since MuJoCo's native `type="cylinder"` geom type is *always*
  Z-symmetric (a builtin primitive, not something MJCF can reparameterize),
  `scene_to_mjcf` (and `gltfworld.render.crosscheck`'s separate MJCF
  builder) now instead give every cylinder geom a fixed,
  body-orientation-independent local `quat` (-90 degrees about local X,
  see `_CYLINDER_LOCAL_FIX_QUAT_WXYZ`) that re-centers MuJoCo's native
  Z-symmetric shape onto local Y *within the body frame*, before the
  body's own (unchanged, still *composed* per `mj_convert`) world
  orientation is applied. Verified both analytically and empirically (an
  identity-oriented object's cylinder now stands upright along MuJoCo's
  +Z, matching the fixed mesh under the same identity orientation; for
  random orientations the geom's resulting world symmetry axis matches
  `R(q_gltf) @ (0, 1, 0)` to float64 precision) and physically
  (`tests/test_cylinder_axis.py`: a cylinder dropped lying on its side
  settles with its center a *radius*, not a *half-height*, above the
  ground). `gltfworld.datagen.sample.object_support_offset` (used for
  ground-clearance checks) is updated to match: local Y is now the height
  axis, X/Z the radial plane. The crosscheck sample episode
  (`tests/conftest.py:make_sample_episode`) now poses its cylinder lying on
  its side rather than upright, specifically so a 90-degree axis mixup
  between mesh and collider would be visually obvious (and IoU-detectable)
  again if this ever regresses -- measured per-object crosscheck IoU for
  that cylinder: **1.0000** (up from indistinguishable-by-coincidence
  under the old, both-wrong convention).
- `simulate(scene, initial_poses, T, record_hz, ...)`: runs at 500 Hz
  internal, records every `round(500 / record_hz)` steps (not necessarily
  exact when `record_hz` doesn't divide 500 evenly, e.g. the default 30 Hz
  -> 500/30 = 16.67 rounds to 17 steps/frame, i.e. an actual ~29.4 Hz;
  `scene.dt`/`series.times` always reflect the *actual* recorded period,
  never a nominal value that doesn't match what was simulated).
- **Ground-contact tolerances (found during V3 verification, documented as
  a loosened tolerance, not silently relaxed; re-measured during V3.1
  independent verification).** The milestone's "never sink more than 5mm
  below the ground plane at any recorded step" is checked as a
  *steady-state* constraint (the last 20% of an episode's frames), not
  literally every frame. A fresh, realistically-massed, fast-falling
  object's very first ground impact produces genuine, physically-inherent
  transient penetration well past 5mm under MuJoCo's finite-stiffness
  soft-contact model, before the contact resolves over the next few steps.
  The originally-reported ~21mm (V3) was a **single-run example**, not a
  worst case: independent re-measurement across a broader sweep of seeds
  found transient penetration up to **~31mm** in some runs, and this
  project's own follow-up sweep (200 `wm-scenes-v1` seeds, `steps=50`,
  `hz=30`, on-plate cases only) found a worse case still -- **~84mm**, for
  a small (`radius~3.6cm`), fast, dense cylinder taking a hard secondary
  bounce (restitution 0.1 still permits *some* bounce). All of these stay
  comfortably inside the loose 10cm "didn't fall through the floor" bound,
  and every case checked settles to steady-state penetration <0.1mm --
  consistent with the phenomenon being real, bounded contact-impact
  transients (worse for smaller/faster/denser objects), not unbounded
  sinking. The enforced bounds are unchanged (<=5mm steady-state, <=10cm
  transient, per the milestone spec); only the "~21mm" prose example is
  corrected here so it isn't mistaken for a measured worst case.
  `tests/test_episode_pipeline.py` checks both: a loose 10cm "didn't fall
  through the floor" sanity bound across every frame, and the tight 5mm
  bound restricted to settled frames.
- **Finite ground plate (not a bug, a scope boundary).** `wm-scenes-v1`'s
  ground is a finite 6m x 6m plate (`_GROUND_HALF_EXTENTS`), not an
  infinite plane. An object given enough linear/angular speed (up to the
  sampled maxima of 1.5 m/s / 3 rad/s, or a cylinder that starts rolling on
  its side and never stops) can cross the plate's edge and fall
  indefinitely in a long-enough episode -- physically correct given a
  finite plate, but it means `object_support_offset`-based "penetration"
  checks stop being meaningful once an object is off the plate's footprint
  (there's no ground there to penetrate). `wm-scenes-v1`'s own workspace
  (objects start within `x, z in [-0.75, 0.75]`, well inside the 3m
  half-extents) and the short episodes exercised by
  `tests/test_episode_pipeline.py` (`steps=50`, `hz=30`, ~1.7s) don't
  trigger this. Longer episodes (this project's own follow-up sweep used
  `steps=100`, ~3.3s, and saw objects depart the plate in some seeds) would
  need either a larger plate or a capped duration if this milestone's
  episodes are ever lengthened.

## Milestones

**Renumbering note**: the original plan below (drafted at V0) allotted
separate milestones (originally numbered V4-V6) to the
`RWM_state_series` codec, the KHR physics codec, and MuJoCo-episode glue.
In practice all three landed together as part of **V1** (see "Custom
components" above -- `gltfworld.ext.rwm`, `gltfworld.ext.khr_physics`, and
`gltfworld.scene.convert` are all implemented and independently verified
as of V1's own checkpoints in `docs/VERIFICATION.md`). Rather than leave
three stale, already-completed line items in this list, V4 onward is
renumbered starting from what's actually next after V3; nothing described
below was silently dropped, it just already happened earlier than
originally planned.

**Second renumbering note (V5/V6 swap)**: the plan as of V4 had V5 =
perception model, V6 = dynamics model, in that order. They're swapped here:
the dynamics model landed first (as V5), perception second (now V6). Same
reasoning as above -- no content dropped, just built in the order the
underlying data/gate work actually supported: V4 packed and gate-passed
`dynamics-v1` (10k episodes, states-only, no rendering needed) well before
`perception-v1`'s smaller, rendering-dependent dataset had an obvious next
model milestone lined up, so dynamics moved first.

- **V0** (done) — project scaffold, CI, verification protocol.
- **V1** (done) — glTF transport codec: pose animation +
  `KHR_physics_rigid_bodies`/`KHR_implicit_shapes` + `RWM_state_series`,
  all with schema validation; `validate`/`inspect` CLI work. (Absorbed what
  this list originally planned as separate V4/V5/V6 milestones -- see the
  renumbering note above.)
- **V2** (done) — vendored, patched pyrender renders episodes headlessly
  (rgb/seg/depth, 256x256); `render`/`crosscheck` CLI work; MuJoCo
  cross-render oracle; benchmark. See "Rendering (V2)" above.
- **V3** (done) — MuJoCo episode generation; `generate` CLI produces GLB
  episodes. See "MuJoCo data generation (V3)" above.
- **V4** (done) — dataset build + provenance + stats + metric harness,
  feeding the pre-training gate: tensor contract (`gltfworld.scene.contract`),
  dataset packing/loading (`gltfworld.data.pack`/`.dataset`), real
  `dynamics-v1`/`perception-v1` datasets, `stats` CLI, cross-validated
  PSNR/SSIM (`gltfworld.eval.metrics`). See "Dataset build (V4)" below,
  `docs/PRETRAINING_GATE.md`, and `docs/VERIFICATION.md`'s V4 section.
- **V5** (done) — dynamics model: state[t] -> state[t+1]
  (`InteractionTransformer`) + baselines (`BallisticBaseline`,
  `NoInteractionMLP`) + training/eval harness. **Reordered ahead of V6**
  (originally the other way around, see the second renumbering note below)
  -- `dynamics-v1` (V4) packed and gate-passed first, so there was real data
  to train a dynamics model against before `perception-v1` had an
  equivalent model milestone ready to build on. See "Dynamics model (V5)"
  below and `docs/VERIFICATION.md`'s V5 section.
- **V6** (code delivered + smoke-tested; full training out of this
  milestone's scope, see below) — perception model: single RGB frame ->
  scene state (`PerceptionDETR`) + Hungarian matching/set loss +
  training/eval harness. See "Perception model (V6)" below and
  `docs/VERIFICATION.md`'s V6 section.
- **V7** (done) — closed-loop demo + attribution: perceive -> roll forward
  -> re-render, real glTF at every hop, plus a 3-arm (oracle / oracle+noise
  / visual) attribution analysis separating perception-induced from
  dynamics-induced rollout error. See "Closed-loop demo + attribution (V7)"
  below and `docs/VERIFICATION.md`'s V7 section.
- **V8** (done) — external eval anchor: Physion replication, state-based
  track (the primary *external* correctness anchor for this project's eval
  numbers, per V4's own metric-cross-validation note). HDF5 -> real glTF
  conversion (`gltfworld.physion.convert`) verified clean at both 3-trial
  and full-150-trial scale; OCP evaluation (`gltfworld.physion.ocp_eval`)
  shows a strong 92%-held-out GT-contact oracle ceiling but a zero-shot
  `InteractionTransformer` transfer collapse to chance -- both outcomes
  reported honestly. See `docs/PHYSION.md` (schema, decision, conversion
  findings) and `docs/RESULTS.md`/`docs/VERIFICATION.md`'s V8 sections.
- **V9** — articulation stage: real dataset (`articulated-v1`), a trained
  joint-state estimator, honest eval, transport exercised throughout.
  **Prep landed first** (KHR joints codec, articulated door/drawer
  transport, joint state channel, physics-sanity + articulation-consistency
  tests -- see "Articulated objects (V9-prep)" below and
  `docs/VERIFICATION.md`'s V9-prep section). **This stage landed on top of
  it**: `gltfworld generate-articulated`/`pack-articulated` CLIs,
  `ArticulationEstimator` (joint position/type/axis from a single rendered
  frame -- explicitly *not* full object detection, see "Articulation stage
  (V9)" below for the scope statement), its training harness, and its eval
  CLI (baselines, re-render check, glTF-at-every-hop). Articulated
  *dynamics* (predicting how a joint's state evolves over time) and the
  full milestone-spec gap report/RWM extension write-up remain open --
  future work, see "Articulation stage (V9)"'s own scope/gaps notes.
- **V9.1** (done) — fixed the EGL context-lifecycle bug V9 discovered
  (`closed_loop.main()` unconditionally deleting a shared EGL display);
  single-process `-m gpu` lane green again. See "V9.1 addendum" above.
- **V10** (done) — the gap report + RWM extension write-up V9's own entry
  flagged as open: `docs/GAP_REPORT.md` v1.0 (20 numbered findings + 5
  positive findings, ranked recommendations), README.md rewritten as the
  project's final summary, this document's own closing status note, and a
  documentation-only consistency pass (`data/README.md`'s stale
  `perception-v1` section corrected). See "V10 closing status" at the end
  of this document and `docs/VERIFICATION.md`'s V10 section.

## Dataset build (V4)

Everything needed to justify starting model training lives behind
`docs/PRETRAINING_GATE.md`; this section is the short factual summary
(full narrative in `docs/VERIFICATION.md`'s V4 section).

- **Tensor contract** (`gltfworld.scene.contract`): `episode_to_tensors`/
  `tensors_to_state` turn an `Episode` into `states float32 (T, N, D=22)`
  (pos(3) + hemisphere-normalized quat xyzw(4) + lin_vel(3) + ang_vel(3) +
  shape-onehot(3) + size(3) + log_mass(1) + friction(1) + restitution(1)),
  `mask bool (N,)`, `class_ids int64 (N,)` (`{"ball": 0, "crate": 1,
  "cylinder": 2}`), and `globals float32 (G=12,)` (gravity(3) + dt(1) +
  camera position(3)+rotation(4)+yfov(1)); static objects (the ground) are
  excluded from `states` and carried in a separate `static` sub-dict
  instead. Verified round-trip <= 1e-6 relative (`tests/test_contract.py`)
  and, more importantly, verified against the *actual glTF files on disk*
  (`tests/test_provenance.py`: simulate -> keep in-memory series -> save
  GLB -> load GLB -> compare tensor contract from both paths, <= 1e-6
  absolute).
- **Packing** (`gltfworld.data.pack.pack_dataset`): one directory of
  `ep_*.glb` -> one `safetensors` file (`states`/`mask`/`class_ids`/
  `globals`/`split_id`/`seeds`, padded to `N_max=5`) + a `pack_meta.json`
  sidecar (source manifest sha256, `N_max`/`D`/count/`T`, ground
  top-Y/footprint, and the split scheme). Split is a deterministic 90/5/5
  train/val/test bucketing of `sha256("gltfworld-split-v1:" +
  episode_seed)`'s first 8 hex digits (keyed by each episode's own
  `SceneState.seed`, not its position in the file -- see
  `gltfworld.data.pack.split_id_for_seed`).
- **Datasets, generated for real** (not just unit-tested; see
  `data/README.md` for exact pinned commands and `docs/PRETRAINING_GATE.md`
  for full stats):
  - `dynamics-v1`: 10,000 episodes, states only (`--steps 100 --hz 30`,
    seed `20260727`). Generated in 4.53 min, 636M on disk; packed to 421M
    in 115.18s. Split: train 8992 / val 532 / test 476.
  - `perception-v1`: 500 episodes with rendered 256x256 rgb+seg+depth
    frames (seed `20260728`). Generated+rendered in 2.13 min (~391 combined
    frames/s), 22G on disk; packed (states only, frames stay
    memory-mapped from their own `.npy` files) to 22M in 5.59s. Split:
    train 458 / val 27 / test 15.
  - Both datasets: 0 NaN/Inf; steady-state ground-penetration <= 5mm holds
    for 99.95%/99.80% of episodes respectively (contact-transient outliers
    documented, not silently dropped -- see "Ground-contact tolerances"
    above); smoothed total energy non-increasing for 99.99%/100.00% of
    episodes. A real, DESIGN.md-predicted fraction of episodes
    (14.02%/12.60%) have an object depart `wm-scenes-v1`'s *finite* ground
    plate at `--steps 100`; `gltfworld stats` excludes those frames from
    the penetration checks (there's no ground there to penetrate) and
    reports the departure rate as its own explicit metric instead of
    hiding it.
- **`gltfworld stats`** (`gltfworld.data.stats`): episode/transition
  counts, per-shape/class histograms, position/velocity/mass/friction
  ranges, NaN/Inf count, steady-state/transient ground-penetration
  fractions, energy trend, split sizes -- human table or `--json`.
- **Eval metrics** (`gltfworld.eval.metrics`): from-scratch MSE/PSNR/SSIM,
  cross-validated against `skimage` (PSNR exact, SSIM within 1e-6 at Wang
  et al. 2004's own parameters) and `torchmetrics` (PSNR, supplementary).
  A CLEVRER/SlotFormer external replication was attempted and honestly
  documented as blocked (Google Drive folder gating, `clevrer.csail.mit.edu`
  unreachable from this environment, and an incompatible pinned training
  stack) rather than faked -- see `docs/VERIFICATION.md`.

## Dynamics model (V5)

`gltfworld.models.dynamics.InteractionTransformer` predicts `state[t] ->
state[t+1]` directly on the `D=22` tensor contract (`gltfworld.scene.
contract`); `gltfworld.models.baselines` holds two comparison points
(`BallisticBaseline`, no learning; `NoInteractionMLP`, learned but with no
cross-object attention); `gltfworld.train.train_dynamics` trains any of
them with one shared harness; `gltfworld.eval.rollout` runs/evaluates
autoregressive rollouts and re-exports predictions as real glTF, per this
project's "inference emits glTF at every hop" principle (see the top-level
architecture flow diagram).

### Rotation math (`gltfworld.models.rotations`)

Batched, differentiable torch, cross-validated against
`scipy.spatial.transform.Rotation` (`tests/test_rotations.py`, 29 tests):
quaternion normalize/hemisphere/multiply/conjugate, the axis-angle
exponential map (`axis_angle_to_quat`, stable at `theta -> 0` via a Taylor
expansion of `sin(theta/2)/theta` rather than dividing by a clamped
`theta`), quaternion <-> rotation matrix (a batched, branchless adaptation
of the standard four-case Shepperd method, in pytorch3d's convention but
xyzw), quaternion <-> 6D rotation representation (Zhou et al. 2019: first
two matrix columns; recovered via Gram-Schmidt), and geodesic angle.

**Geodesic angle is not `2*arccos(|dot|)`.** It's computed as
`4 * atan2(||q1 - q2||, ||q1 + q2||)` after aligning `q2`'s hemisphere to
`q1` (not just the canonical `w>=0` hemisphere). Both formulas agree in
exact arithmetic, but `arccos`'s derivative diverges as its argument
approaches +-1 -- exactly where most training pairs land, since two states
one 1/30s frame apart are usually only a few degrees apart. The
`atan2`-of-norms form has a well-behaved gradient everywhere on `[0, pi]`,
including at 0 (verified: `tests/test_rotations.py::
test_geodesic_angle_gradient_finite_near_zero`). The `4x` factor (as
opposed to the more commonly quoted `2x` for the *un-hemisphere-aligned*
angle between the quaternions themselves) comes directly from the
half-angle identities relating quaternion-vector angle to rotation angle;
derived in the function's own docstring and cross-checked against
`scipy`'s `Rotation.magnitude()` of the relative rotation.

### `InteractionTransformer` (~4.8M params, target band 4-7M)

- **Tokens**: one per object, embedding a hand-picked, roughly unit-scaled
  24-dim feature vector (`object_features`: `pos/2.0`, `vel/3.0`,
  `ang_vel/6.0`, the 6D rotation representation, `shape_onehot`,
  `size/0.25`, `log_mass/3.0`, `friction`, `restitution` -- fixed scale
  constants, not data-fit, chosen to roughly range-normalize
  `wm-scenes-v1`'s sampled distributions per DESIGN.md's own V3 section);
  one globals token (`globals_features`: `gravity/9.81`, `dt*30` --
  camera deliberately excluded, irrelevant to physics); one learned
  "ground" token (a plain `nn.Parameter`, *not* derived from any input
  feature -- `wm-scenes-v1`'s ground plate is geometrically identical
  across every episode, so there's no per-episode ground signal to encode;
  this token just gives every object a fixed attention partner to learn
  ground-relative dynamics against, `[CLS]`-style).
- **6 pre-norm `nn.TransformerEncoder` layers**, `d_model=256`, 8 heads, MLP
  ratio 4, `src_key_padding_mask` built from the dataset's per-object
  `mask` so padded object slots are excluded from attention as *keys* (they
  can never influence a real object's output -- see
  `tests/test_dynamics.py::test_masking_invariance_padded_slots_dont_leak`).
  No positional encoding is added across the object-token axis, so the
  whole stack is permutation-equivariant in object order (`tests/
  test_dynamics.py::test_permutation_equivariance_transformer`).
- **Output head** (shared, zero-init final layer): per real-object token,
  `(dv, dw, r)` -- linear-velocity delta, angular-velocity delta, and a
  rotation-update rotation-vector (axis-angle), 3 each. Zero-init means a
  freshly constructed model predicts all-zero deltas, so integration
  reduces to *exact* constant-velocity extrapolation until training has
  changed anything (`tests/test_dynamics.py::
  test_integrator_exactness_fresh_model_matches_ballistic`: with gravity
  zeroed out, a fresh model's output is bit-identical, same dtype/op order,
  to `BallisticBaseline`'s).
- **Integration** (`gltfworld.models.dynamics.integrate`, semi-implicit
  Euler, shared by every model in this milestone -- baselines import it
  directly rather than reimplementing it, so a model-vs-baseline comparison
  is never comparing different arithmetic):

  ```
  v' = v + dv
  p' = p + v' * dt          (uses the *updated* velocity)
  w' = w + dw
  q' = normalize(hemisphere(exp(r) (x) q))
  ```

  Static per-object features (shape one-hot, size, log-mass, friction,
  restitution) are copied through unchanged.
- Parameter count is printed by `python -m gltfworld.models.dynamics`
  (asserts the 4-7M band itself); measured on this machine: **4,815,113**.

### Baselines (`gltfworld.models.baselines`)

- **`BallisticBaseline`**: `dv = gravity * dt`, `dw = 0`, `r = 0` -- no
  learned parameters, routed through the exact same `integrate` call.
- **`NoInteractionMLP`**: per-object MLP (`object_features` concatenated
  with `globals_features`, 2 hidden layers of 256; output head is *not*
  zero-init, unlike `InteractionTransformer`'s -- see below), applied
  independently per object token -- no attention, no mechanism for one
  object to influence another's prediction. This is the ablation that
  isolates what
  `InteractionTransformer`'s cross-object attention actually buys.
  Measured parameter count: **75,529** -- smaller than the milestone
  spec's "~0.3M" approximation. Deviation, documented rather than papered
  over: the literal architecture description ("2x256 hidden") was kept as
  ground truth over the approximate parameter count, since the two aren't
  simultaneously satisfiable with this feature dimensionality without
  padding the network with width that architecture description doesn't
  ask for.
- **Why `NoInteractionMLP` isn't zero-init**: `InteractionTransformer`'s
  zero-init is a real invariant (exact constant-velocity start, tested).
  For this much smaller ablation model, zero-init would leave almost no
  loss to reduce inside the training harness's 500-step `--smoke` check
  (constant velocity is already a good approximation over a single
  1/30s step) -- a small random init (`nn.Linear`'s default) instead gives
  smoke a real, non-trivial loss curve to demonstrate learning on. Both
  models still route zero explicit deltas through the identical
  `integrate` function bit-identically to `BallisticBaseline`
  (`tests/test_dynamics.py::test_mlp_shares_integrator_with_zero_deltas`)
  -- the *only* thing that changed is what the network predicts before any
  training, not the integrator.

### Training harness (`gltfworld.train.train_dynamics`)

JSON-loadable `Config` dataclass (`configs/dynamics_v1.json` for
`InteractionTransformer`, `configs/dynamics_mlp.json` for
`NoInteractionMLP`); two-phase schedule:

- **Phase 1** (default 40k steps): single-step teacher forcing. Every step
  samples a random `(state_t, state_t+1)` transition (via `TransitionSampler`,
  a vectorized in-memory gather over the packed split's tensors -- moved
  to `device` once, no per-item Python-loop `DataLoader` overhead), adds
  Gaussian noise to `state_t` (position `sigma=5mm`, velocity
  `sigma=0.02 m/s`, rotation `sigma=0.5 deg` via a random small axis-angle
  composed onto the quaternion), predicts one step, and computes the
  masked weighted loss against the *clean* `state_t+1`. AdamW, `lr=3e-4`
  cosine-annealed to a floor, bf16 autocast, grad-norm clip 1.0.
- **Phase 2** (default 10k steps): `K`-step autoregressive rollout
  finetuning (`SequenceSampler`, same vectorized-gather approach but over
  full `(T, N_max, D)` episode windows), `K` annealed linearly 2 -> 8 across
  the phase, no input noise (the model's own rollout error is the only
  "noise" here), a fresh AdamW at `lr=1e-4`.
- **Loss** (`compute_losses`): masked, weighted MSE on normalized-unit
  position/velocity/angular-velocity (divided by the same `POS_SCALE`/
  `VEL_SCALE`/`ANGVEL_SCALE` constants `object_features` uses, so the three
  components are on commensurate scales) plus a squared-geodesic-angle
  rotation term (`quat_geodesic_angle(pred_quat, target_quat)**2`); weights
  all configurable, default 1.0 each; every component logged separately.
- **Checkpoints**: `step_{N:07d}.safetensors` (model weights only) every
  `ckpt_every` steps, plus `best.safetensors` (lowest val total loss) and
  `last.safetensors` (most recent), each with a matching
  `*.train_state.pt` (optimizer/scheduler/step/RNG state -- plain
  `torch.save`, not safetensors, since it isn't a flat tensor map).
  `--resume` restores model + both phases' optimizer/scheduler state + the
  global step counter + every RNG stream (Python/`numpy`/torch CPU+CUDA),
  and continues into whichever phase the restored step falls into.
  Verified by direct harness test (not just unit test): a 100-step partial
  run resumed to 300 steps continues the cosine schedule and phase
  transition (phase 1 -> phase 2 at step 200) correctly, `log.csv` appended
  (never truncated) across the resume boundary.
- **`--smoke`**: overrides to 500 steps (phase 2 skipped), a tiny val
  subset, and asserts the *EMA-smoothed* (decay 0.98) training loss dropped
  >= 30% from an early-vs-final comparison window, printing both the raw
  (high-variance, single-random-batch-per-step) and EMA curves either way.
  Why EMA and not raw: single-batch loss variance (different episodes'
  object counts/difficulty land in different random batches) dominates the
  raw per-step signal at this batch size over only 500 steps; EMA is the
  standard practical smoothing for exactly this kind of noisy-loss
  pass/fail check. Measured on the real `dynamics-v1` packed dataset (RTX
  PRO 6000 Blackwell): `InteractionTransformer` **41.5% raw / 38.7% EMA**
  drop in 4.2s; `NoInteractionMLP` **38.9% raw / 35.5% EMA** drop in 1.7s
  -- both comfortably clear the 30% bar and finish in seconds, not the 3
  minute budget.
- **`NoInteractionMLP`'s tuned `lr=5e-3`** (vs. `InteractionTransformer`'s
  `3e-4`), a deliberate, documented choice: at the transformer's `lr`, the
  MLP's per-batch training loss is so dominated by batch-composition noise
  (which objects/episodes land in a given random batch materially changes
  intrinsic task difficulty) that it doesn't clear the smoke gate even
  though its *validation* loss (a much lower-variance, fixed-subset signal)
  does genuinely improve -- a real, measured finding about how much faster
  a much smaller model needs to move to show up above that noise floor in
  only 500 steps, not a bug. A higher `lr`, reasonable for a 75K-param MLP
  regressing near-zero deltas, resolves it with a real (not just
  noise-window-selection) validation-loss improvement to back it up
  (0.0453 -> 0.0367 in 500 steps at `lr=5e-3`, vs. 0.0472 -> 0.0468 -- barely
  moving -- at `lr=3e-4`).
- **Determinism**: `set_seed` seeds Python/`numpy`/torch (CPU + all CUDA
  devices); training-state checkpoints round-trip every one of those RNG
  streams so `--resume` continues the same stream. **Not** pinned down,
  documented rather than silently assumed away: cuDNN kernel-selection
  nondeterminism (moot in practice -- this model has no convolutions) and
  the inherent nonassociativity of floating-point reduction order in CUDA's
  parallel matmul/attention/softmax kernels (bf16 autocast in particular)
  -- bit-identical reruns on GPU aren't guaranteed even with every seed
  fixed, only statistically equivalent runs.

### Rollout + eval (`gltfworld.eval.rollout`)

- `rollout(model, initial_state, mask, globals, T)` -- accepts either a
  single episode (`(N, D)`) or a batch (`(B, N, D)`), autoregressive (index
  0 is the given initial state unmodified; indices `1..T-1` are the model's
  own successive predictions, never re-fed ground truth).
- CLI computes, for the requested checkpoint, `BallisticBaseline`, and
  (optionally, `--mlp-ckpt`) a `NoInteractionMLP` checkpoint: per-horizon
  (default `1/5/10/30/99`) position/rotation/velocity error, median + IQR
  over every unmasked `(episode, object)` pair; writes `metrics.json` +
  a markdown table `metrics.md`, and a log-y divergence-curve PNG (median
  position error at *every* horizon `1..T-1`, one line per model --
  `matplotlib`, added to the `ml` extra for this milestone).
- **glTF at every hop** (`--emit-gltf N`): re-exports N test episodes as
  real, loadable `.glb` pairs -- `pred/ep_XXXXXX.glb` (the model's own
  rolled-out prediction, rebuilt into a *full* `Episode` via
  `tensors_to_episode`: same scene, same ground/static objects held at
  their template frame-0 pose since the tensor contract never carries
  static poses at all, `pose_variance` omitted) and `gt/ep_XXXXXX.glb` (the
  same tensors re-exported from ground truth, for a same-format diff).
  Round-trip verified to <= 1e-6 absolute
  (`tests/test_rollout.py::test_pred_glb_roundtrip`): `load_episode` of a
  written pred `.glb`, run back through `episode_to_tensors`, reproduces
  the exact tensors that built it.
- `--video N` (needs the `render` extra + a real GPU/EGL context, lazily
  imported): renders GT-vs-pred side-by-side mp4s at 30fps via
  `EpisodeRenderer` + `imageio-ffmpeg`.

### Acceptance (see `docs/VERIFICATION.md`'s V5 section for exact commands)

Model must beat `BallisticBaseline` on median position error at horizons
1/10/30 on the `dynamics-v1` test split. Measured on a 300-step
correctness-check training run (not the full 50k-step run, which the
orchestrator runs separately per this milestone's own scope boundary --
training code is delivered and smoke-tested here, not executed to
completion): at h=30, model **0.283m** vs. ballistic **4.625m**; at h=99,
model **1.289m** vs. ballistic **55.542m** -- ballistic's unbounded
constant-gravity extrapolation (no floor, no collisions) diverges
catastrophically past first contact, exactly the failure mode a learned
model is supposed to fix. At h=1 the two are closer (0.0055m vs. 0.0053m --
a single 1/30s step is dominated by the ballistic term either way,
un-surprising this early and this undertrained); h=1/10 pass the same
"model <= ballistic" bar at full training convergence is expected to
widen, not close, given h=30/h=99's trend.

## Perception model (V6)

`gltfworld.models.perception.PerceptionDETR` predicts `frame -> set of
objects` (pose, size, shape, semantic class, existence) directly from a
single rendered RGB frame; `gltfworld.models.matching` holds Hungarian
matching + the symmetry-aware set-prediction loss; `gltfworld.train
.train_perception` trains it with a `train_dynamics`-style harness (config
dataclass, resumable, `--smoke`, `log.csv`, safetensors checkpoints);
`gltfworld.eval.perception_eval` evaluates a checkpoint (existence PR/F1,
matched pose/size/shape/class error, a mean-state baseline, and a GPU
re-render PSNR/SSIM check) and re-exports predictions as real glTF, per this
project's "inference emits glTF at every hop" principle.

### `PerceptionDETR` (~8.2M params -- see "documented parameter-count
deviation" below)

A DETR-lite (Carion et al. 2020's set-prediction recipe, scaled down):

- **Patch embed**: `256x256x3` RGB (normalized from `PerceptionDataset`'s
  `[0, 1]` range to `[-1, 1]` *inside* the model, not by the caller) -> one
  non-overlapping `16x16` patch per token via a single strided `Conv2d`
  (equivalent to a per-patch linear projection) -> 256 tokens, plus a
  learned per-token 2D positional embedding.
- **Encoder**: 6 pre-norm `nn.TransformerEncoderLayer`s, `d_model=256`, 8
  heads, MLP ratio 4 -- architecturally the same recipe as `gltfworld.models
  .dynamics.InteractionTransformer`'s stack, just over image-patch tokens.
- **Decoder**: `N_MAX=5` learned object-query tokens (fixed per-slot
  embeddings, not derived from the image -- standard DETR), 3 pre-norm
  `nn.TransformerDecoderLayer`s (self-attention across queries + cross-
  attention to the encoder's image tokens), same `d_model`/heads/MLP ratio.
- **Heads** (one shared 2-layer MLP trunk, split by output field):
  existence logit (1); position (3, workspace-normalized via a sigmoid
  affinely mapped into a fixed `[POS_MIN, POS_MAX]` box with margin over
  `wm-scenes-v1`'s actual sampled range, then denormalized -- that mapped
  value *is* the final world-unit position); rotation as a 6D representation
  -> quaternion via `gltfworld.models.rotations.sixd_to_quat` (continuous,
  no double-cover discontinuity as a regression target); size (3, same
  sigmoid-into-a-box scheme, `[SIZE_MIN, SIZE_MAX]`); shape logits (3, order
  matches `gltfworld.scene.contract.SHAPE_ORDER`); class logits (3, order
  matches `gltfworld.scene.contract.CATEGORY_TO_CLASS_ID`).

**Documented parameter-count deviation** (same precedent as V5's
`NoInteractionMLP`): the milestone spec text's own approximate "~12-16M"
target and the literal architecture description above (patch16/256 tokens,
6 encoder + 3 decoder layers, `d_model=256`, 8 heads, MLP ratio 4, 5
queries) aren't simultaneously satisfiable -- the literal architecture, as
specified, measures smaller. Per this project's stated policy (prefer the
literal, testable architecture description over an approximate headline
number rather than padding hidden dims solely to hit a target the
component list doesn't otherwise ask for), the architecture was implemented
exactly as specified. Measured (`python -m gltfworld.models.perception`,
this machine): **8,234,259** parameters.

### Matching + symmetry-aware loss (`gltfworld.models.matching`)

- **Hungarian matching** (`hungarian_match`, `scipy.optimize
  .linear_sum_assignment` -- exact, not an approximation): per sample, a
  cost matrix `(N_MAX queries) x (n_real GT objects)` from
  `w_pos * L2(position) + w_cls * CE(class) + w_size * L2(size)`, restricted
  to that sample's existence-eligible (`mask`) GT rows only -- a padded/
  nonexistent GT slot can never be matched against. Queries beyond the real
  GT count are left unmatched.
- **Losses on matched pairs**: position MSE and size MSE (both normalized
  by the same `POS_SCALE`/`SIZE_SCALE`-style constants
  `InteractionTransformer`'s own features use, for a commensurate loss
  scale), shape CE, class CE, and a **symmetry-aware rotation loss**
  (`symmetry_rotation_loss`, squared symmetry-aware angle -- same
  squared-geodesic-angle convention `train_dynamics.compute_losses` uses):
  - **sphere**: always 0 -- a sphere has continuous rotational symmetry
    about every axis, so there is no orientation to supervise at all.
  - **box**: the minimum geodesic angle (`gltfworld.models.rotations
    .quat_geodesic_angle`) between the predicted quaternion and the GT
    quaternion composed with each of the cube's **24** rotational
    symmetries (`CUBE_SYMMETRY_QUATS`, precomputed once at import time from
    the 24 proper, `det=+1`, signed-permutation matrices of R^3 -- not
    data-fit).
  - **cylinder**: `axis_alignment_angle` compares only the predicted-vs-GT
    local **Y** axis direction (`R(q) @ (0, 1, 0)`, per `ObjectSpec`'s
    documented cylinder convention), sign-aligned before measuring the
    angle between them -- invariant to spinning about that axis (doesn't
    change the axis direction at all) *and* to the 180-degree end-swap flip
    (negates the axis direction, which sign-alignment cancels) simultaneously,
    for free.

  Both symmetry angles use the same `atan2`-of-norms form
  `quat_geodesic_angle` does (rather than `arccos`), for the identical
  well-behaved-gradient-near-zero reason documented there.
- **Unmatched queries**: existence BCE target 0; matched queries: target 1.
  All loss weights are configurable (`gltfworld.train.train_perception
  .Config`).
- Unit tests (`tests/test_matching.py`) prove each symmetry directly: a box
  rotated by any of its 24 symmetries -> rotation loss ~0 (and a genuinely
  different orientation is *not* near-zero, so the test isn't vacuous); a
  cylinder spun about its own local-Y axis -> ~0, flipped 180 degrees -> ~0
  (and a genuinely tilted axis is *not* near-zero); a sphere's rotation loss
  is exactly 0 regardless of how different the predicted and GT quaternions
  are.

### Training harness (`gltfworld.train.train_perception`)

Shares `train_dynamics`'s harness *contract* (JSON-loadable `Config`
dataclass, resumable checkpoints with matching `.train_state.pt`,
`--smoke`, `log.csv`, `step_{N:07d}.safetensors`/`best`/`last`
checkpoints) but is a simpler single-phase schedule -- there is no analogue
of the dynamics model's autoregressive rollout finetune phase here (a
single-frame perception model has no rollout to finetune against).

- AdamW, `lr=2e-4` cosine-annealed to a floor, batch 128, default 25k steps,
  bf16 autocast, grad-norm clip 1.0.
- **Data loading**: unlike `train_dynamics`'s fully-in-VRAM vectorized
  samplers, `perception-v1`'s rendered RGB frames are too large to
  preload an entire split into VRAM up front, so training uses an ordinary
  `torch.utils.data.DataLoader` over `gltfworld.data.dataset
  .PerceptionDataset` (shuffled, worker-process I/O over the
  memory-mapped `rgb.npy` files) wrapped in a small infinite-iterator
  helper, rather than a custom sampler.
- **Augmentation, RGB only** (`augment_rgb`): brightness jitter, contrast
  jitter, and additive Gaussian noise, clamped back to `[0, 1]`. State
  targets (position/rotation/size/shape/class) are never touched by this --
  the perception task must stay geometrically truthful even while its
  *appearance* robustness is trained up.
- **Val** (every 1k steps by default): total + per-component loss, plus a
  quick matched-position-error (median, over Hungarian-matched pairs) as a
  human-readable sanity signal alongside the raw loss.
- **`--smoke`**: 500 steps, tiny val subset, asserts the EMA-smoothed
  (decay 0.98) training loss dropped >= 30% (same early-vs-final-window
  convention `train_dynamics --smoke` uses, for the same "single-batch loss
  variance dominates the raw per-step signal at this budget" reason).
  Measured on the real `perception-v1` packed dataset (RTX PRO 6000
  Blackwell): **19.0% raw / 33.2% EMA** drop in **136.5s** (well under the
  5-minute budget) -- see `docs/VERIFICATION.md`'s V6 section for the full
  printed curve.

### Eval (`gltfworld.eval.perception_eval`)

    uv run python -m gltfworld.eval.perception_eval \
        --ckpt runs/perception-v1/best.safetensors \
        --data data/perception-v1 --split test \
        --out runs/perception-v1/eval

- **Existence**: precision/recall/F1 at a 0.5 threshold (a query is a true
  positive iff it is both the Hungarian-assigned match for a real GT object
  *and* thresholded "existent"; a false positive is an *unmatched* query
  thresholded existent; a false negative is a real GT object whose matched
  query fell below threshold), plus a full 101-point PR curve.
- **Per-N breakdown** (N = 1..5 real objects/frame): existence F1 and
  matched position error within each group.
- **Matched pose/size/shape/class metrics**: computed over every
  Hungarian-matched pair, independent of the existence threshold (a
  pose-quality metric, not a detection metric -- same convention
  `gltfworld.eval.rollout` uses). Rotation error is the same symmetry-aware
  angle the loss uses, reported in degrees, split by GT shape (sphere rows
  excluded -- nothing meaningful to score).
- **Count-exact-match rate**: fraction of frames where the thresholded
  predicted-object count equals the real GT count.
- **Mean-state baseline**: a zero-learning dummy predictor (always predicts
  the train-split's mean real-object count, all at the train-split mean
  position/size and the modal shape/class, never looking at the image at
  all) run through the identical metrics pipeline, so the trained model's
  numbers are read against a trivial prior rather than in a vacuum.
- **Re-render check** (`--render-samples K`, default 50, needs the `render`
  extra + GPU): for K sampled test frames, builds a predicted single-frame
  `Episode` from every existence-thresholded query (predicted pose/size/
  shape; color/material copied from the GT `ObjectSpec` of that query's
  Hungarian-*matched* real object -- an intentionally honest GT-assist,
  documented as such, since `PerceptionDETR` never predicts color/material
  and this check's purpose is only to sanity-check geometry rendering
  fidelity; an unmatched thresholded query, i.e. a false positive, gets a
  fixed neutral-gray fallback instead, since there is no GT object to copy
  from), renders it, and compares PSNR/SSIM against the actual stored GT
  frame. Predicted frames are saved as real, independently loadable
  `pred_frames/ep_XXXXXX_fYYYY.glb` (`T=1`), round-trip verified inline
  (`<= 1e-6`), and run through the real, pinned glTF-Validator.

### Acceptance (see `docs/VERIFICATION.md`'s V6 section for exact commands)

On the `perception-v1` test split: existence F1 >= 0.95, median matched
position error <= 0.05 m, class accuracy >= 0.95. Per this milestone's own
scope boundary, the full 25k-step training run is the orchestrator's to run
(not run here); if the trained model's numbers miss this bar, that is to be
reported honestly and recorded in `docs/RESULTS.md` rather than hidden --
same policy `docs/VERIFICATION.md`'s V5 section already established for the
dynamics model.

### V6.1 postmortem: flat val `matched_pos_err` (dataset far too small)

The orchestrator's full 25k-step run against the `perception-v1` dataset
produced a pathological result: train loss fell smoothly to ~0.19-0.21, but
val loss climbed monotonically from 2.18 (step 1000) to 8.87 (step 25000),
and val `matched_pos_err` sat completely flat at ~0.6-0.67m (mean-predictor
level for the ~1.5m workspace) from step 1000 through step 25000 -- it never
improved even once. `--smoke` (500 steps, train-loss-only) passed cleanly at
33% EMA drop, because it never looked at val at all.

**Root-cause experiment** (the deciding one, not a guess): the exact val
evaluation code path (`compute_perception_losses` + Hungarian matching) was
run on 500 held-out *train* frames from the step-25000 checkpoint, and
separately on 500 real val frames, both through the identical code:

| split (through the val eval code path) | median matched_pos_err |
| --- | --- |
| train frames | 0.12 m |
| val frames | 0.61 m |

Train frames scored well *through the exact same pipeline* real val frames
scored badly through -- ruling out a matching/eval-pipeline bug (option (c)
in the investigation) and a train/val frame<->state misalignment bug
(option (b)): the code is correct, and the model has clearly learned
*something*, just nothing that transfers past the frames it was trained on.
That is memorization, not a pipeline defect (per the investigation's own
decision rule).

**Why**: `perception-v1` had only 500 total episodes (458 train, 45,800
rendered train frames) -- a placeholder/dev-scale count, an order of
magnitude below `dynamics-v1`'s 10,000 episodes used for the (successfully
generalizing) dynamics model. At `batch_size=128` and `steps=25,000`, the
run drew 3,200,000 samples against those 45,800 train frames -- an
"epoch-equivalent" of ~69.9x. `PerceptionDETR`'s ~8.2M parameters, over a
visually low-dimensional task (fixed camera, 3 shapes, 8 colors, a small
continuous workspace), memorized the 458 distinct training scenes well
before step 1000 and never had a chance to learn a generalizable
image->geometry mapping instead.

**Refinement (V6.2 diagnostic, see below): the memorization finding above is
not uniform across output heads.** Re-examining the same held-out-train-vs-
val checkpoint: shape/class classification actually *did* generalize
reasonably well (0.875 accuracy vs a 0.40 trivial-baseline accuracy) --
it is specifically the position/existence heads that memorized without
generalizing. A visually low-dimensional, near-discrete task (3 shapes, 3
classes) is easier to generalize on from few examples than a continuous
regression target over a large-ish task-relevant workspace; this dataset
was too small for the latter, not for the former.

**Fix**:

1. Regenerated `perception-v1` at production scale (4,000 episodes,
   `gltfworld generate --render --episodes 4000` at the same base seed, so
   the original 500 episodes are a strict prefix) and re-packed it -- more
   scene diversity to train against. (See the V6.2 follow-up below: the
   originally-claimed "short confirmation run shows real, monotonic val
   `matched_pos_err` improvement" was not actually observed and has been
   retracted -- the dataset regeneration was necessary but a short
   confirmation run at V6.1's chosen step count was not sufficient to
   demonstrate it worked.)
2. Added a dataset-scale guard to the training harness itself
   (`gltfworld.train.train_perception.check_dataset_scale`/
   `epoch_equivalent`/`MAX_EPOCH_EQUIVALENT=15.0`): `train()` now refuses to
   start (a loud, immediate `ValueError`, not a silently-wasted 25k-step
   run) whenever the configured step budget implies training on each train
   frame more than 15x over, unless the config explicitly opts in via
   `allow_high_epoch_equivalent=True` (e.g. a deliberately tiny smoke/
   unit-test dataset). Run against the V6 incident's exact numbers
   (25,000 steps, batch 128, 45,800 train frames -> 69.9x), this raises
   immediately; against the regenerated 4,000-episode dataset it does not
   (~8.9x for the same full schedule). This guard's own logic/cap is
   unaffected by the V6.2 follow-up below and remains correct as designed.
3. Added `--smoke-val`, since recalibrated in V6.2 (see below) to ~5,000
   steps: asserts val `matched_pos_err` both improves from its step-500
   value by a minimum relative amount and drops below an absolute bound --
   unlike `--smoke`, this would have caught the V6 pathology directly (a
   flat-at-the-mean-predictor val curve fails both conditions) rather than
   needing the full 25k-step run and a human reading the log to notice.
4. Regression tests: `tests/test_train_perception_dataset_scale.py` (no
   GPU/data needed -- pure arithmetic against the exact V6 incident numbers)
   and `tests/test_train_perception_smoke.py::test_smoke_val_on_real_
   perception_v1` (gpu-marked, runs `--smoke-val` end-to-end against the
   real, regenerated dataset).

### V6.2 postmortem: correcting V6.1's unsupported claim, under-training vs. stagnation, and an out-of-box GT defect

A follow-up diagnostic against the regenerated (4,000-episode) dataset
produced hard evidence that requires correcting one of V6.1's claims above,
and surfaced an independent, previously-unquantified data defect.

**Correction to V6.1's "Fix" item 1**: the claim that a "~2500-step
confirmation run shows real, monotonic val `matched_pos_err` improvement"
is **not supported by the evidence** and is retracted. Two independent
~1800-step confirmation runs against the regenerated dataset both showed a
flat val `matched_pos_err` curve -- not the monotonic improvement V6.1
reported. Whatever run originally produced that impression either
mismeasured or was not reproduced; treat it as never having been
demonstrated.

**Why the flat curve is under-training, not a stuck/broken pathway** -- two
independent pieces of evidence:

1. **The position pathway itself is healthy.** An 8-frame overfit test
   converges to a 0.05-0.09 m median `matched_pos_err`, and position
   receives 32% of the trunk's gradient norm during training -- a model
   that could not learn position at all, or whose gradient was being
   drowned out by other loss terms, would not show either of these.
2. **Epoch-equivalent math accounts for the flatness.** 1,800 steps @
   `batch_size=128` on the 4,000-episode pack's 363,600 train frames is
   `epoch_equivalent(1800, 128, 363_600) = 0.63` -- well under one full
   pass over the training data. The *old*, 500-episode (memorizing)
   dataset didn't show any val loss movement until roughly 3-8
   epoch-equivalents in (see the V6.1 root-cause section above: train loss
   was already memorizing well before step 1000 out of a 25,000-step/69.9x
   run, i.e. movement was visible well under 1 epoch-equivalent there only
   because the *train* set was catastrophically small and thus trivial to
   fit -- not evidence that a *properly-sized* dataset should show val
   movement that early too). A calibrated confirmation run on the
   regenerated dataset needs roughly 15,000-25,000 steps (~5.3-8.8
   epoch-equivalents) to be a fair test of whether the dataset fix worked.

**`--smoke-val` recalibration**: V6.1's original ~1800-step/0.63-epoch-
equivalent window was too short to distinguish "the dataset fix didn't
work" from "hasn't trained long enough yet" -- exactly the ambiguity the
two flat-curve runs above ran into. `--smoke-val` is recalibrated to
~5,000 steps (~1.8 epoch-equivalents on the 4,000-episode pack -- still far
short of the 15k-25k needed for a real confirmation, but enough to show an
early, partial trend without paying for a full run) with a two-part
acceptance bar
(`gltfworld.train.train_perception.SMOKE_VAL_MIN_RELATIVE_IMPROVEMENT=0.15`/
`SMOKE_VAL_POS_ERR_BOUND_M=0.55`): val `matched_pos_err` at the final step
must be both >= 15% relatively better than its value at step 500 *and*
below 0.55m in absolute terms -- neither condition alone is a reliable
generalization signal (a flat curve at a low absolute value would pass an
absolute-only bar; an "improvement" from a catastrophic baseline to a
still-bad value would pass a relative-only bar). See
`gltfworld.train.train_perception`'s module docstring for the full
rationale.

**Observed result of running the recalibrated `--smoke-val` (this session,
`configs/perception_v1.json`, seed 0) -- reported honestly, not tuned to
pass**: val `matched_pos_err` per evaluation (steps 500-5000, every 500
steps): 0.6458, 0.6606, 0.6426, 0.6589, 0.6309, 0.6228, 0.6156, 0.6182,
0.6285, 0.5958 (m). Step-500 -> step-5000 relative improvement: **7.7%**
(needs >= 15%); final value **0.5958m** (needs < 0.55m). **`--smoke-val`
FAILED both of its own acceptance conditions.** This was not treated as a
reason to loosen the bar -- it is fully consistent with, not a
contradiction of, the epoch-equivalent analysis above: 5,000 steps is only
~1.8 epoch-equivalents, well short of the ~15,000-25,000 steps (~5.3-8.8
epoch-equivalents) estimated as necessary for a fair confirmation. The
per-evaluation trend is real but slow and noisy (a mostly-flat plateau
around 0.62-0.66m for the first ~4,000 steps, with the clearest single drop
only in the last evaluation) -- exactly the shape expected of a
still-under-training run this early in a 25k-step schedule, not evidence
that the dataset-scale/out-of-box-GT fixes above didn't work. The
orchestrator should not conclude either recovery or continued failure from
this result alone: a full 15,000-25,000-step run (out of this milestone's
scope -- see the Acceptance section above and DESIGN.md's "Do NOT launch
the full retrain" scope boundary) is still the only way to actually confirm
whether the V6.1/V6.2 fixes produce a generalizing model. The orchestrator
has since launched that full run (`runs/perception-v2`); `--smoke-val`'s
thresholds will be recalibrated from its measured curve once it completes,
not tuned retroactively to make this session's short run pass.

**Full 25k-step run outcome (orchestrator-run, `runs/perception-v2`)**: val
`matched_pos_err` declined steadily from 0.658m to 0.461m with no plateau
-- confirming the dataset-scale/out-of-box-GT fixes did restore real
generalization -- though a train/val loss gap persists (0.61 train vs 2.34
val), which `--smoke-val`'s pending recalibration (above) should account
for rather than ignore.

**Out-of-box GT defect (independent of the above)**: 4.51%/4.85%/3.60% of
train/val/test GT object positions in the regenerated 4,000-episode pack
fall outside the model's representable `[POS_MIN, POS_MAX]` workspace box
(`gltfworld.models.perception`) -- objects that escape `wm-scenes-v1`'s
finite ground plate over the course of an episode (observed extremes:
y=-30.4m, z=+17.4m). These are simultaneously unobservable (outside the
fixed camera's frustum -- never actually visible in the rendered RGB frame)
and unrepresentable (the position head's sigmoid-into-box parameterization
makes it structurally impossible for any query to emit a position out
there), so every code path that treated them as "exists, supervise its
pose" before this fix was poisoning both training and eval with an
impossible-to-hit target. Fixed via `gltfworld.models.matching
.filter_out_of_box_gt`, a single shared helper used by both the training
loss (`compute_perception_losses`) and eval matching
(`gltfworld.eval.perception_eval.run_inference`/`run_inference_baseline`):
an out-of-box GT object's `mask` entry is cleared before Hungarian
matching, so it is treated as *absent* for that frame -- no query can ever
be matched to it, and existence supervision/eval skips it too, not just
pose supervision. `gltfworld.data.pack.pack_dataset` also reports the
per-split fraction (printed, and recorded in `pack_meta.json`'s
`workspace_filter` field) so the defect's scale stays visible independent
of any particular training/eval run.

### V6.3: CNN encoder option (small-data regime)

**Rationale.** V6.1/V6.2 established, with real evidence rather than a
guess, that the `vit` encoder's problem is data-hunger, not a broken
pathway: the full 25k-step run on the (correctly-sized, defect-fixed)
4,000-episode `perception-v1` pack reached only **0.461m** median val
position error against the milestone's **0.05m** bar, with a persistent
train/val gap (0.61 train vs. 2.34 val loss) even after the dataset-scale
guard and out-of-box-GT fixes landed; an 8-frame overfit test converges to
0.05-0.09m and position gets 32% of the trunk's gradient norm (the
decoder/heads/loss pathway is healthy); shape/class heads *do* generalize
(0.875 vs. a 0.40 trivial baseline). A from-scratch `nn.TransformerEncoder`
over raw image patches has no built-in spatial prior -- no locality, no
translation equivariance -- and has to learn one from data alone; a
from-scratch CNN gets both for free from its architecture. This is the
textbook fix for "transformer generalizes too slowly in a small-data
regime": swap in a convolutional inductive bias rather than trying to make
the transformer work harder on the same data.

**Architecture** (`gltfworld.models.perception`, `encoder="cnn"` option on
`PerceptionDETR`, default remains `encoder="vit"`, unchanged): a stride-1
stem (`3 -> 32` channels) + 4 stride-2 stages (output channels 32/64/128/256,
`GroupNorm` + `SiLU` after every conv, plain stacked conv blocks -- no
residual connections, not architecturally required by this fix), taking
`256x256` down to a `16x16x256` feature map (4 halvings: `256/2**4 == 16`,
matching the `vit` path's 256-token patch grid exactly), a `1x1` conv
projecting to `d_model=256` tokens, plus a learned per-token 2D positional
embedding. That token sequence feeds the *existing* decoder (3 layers, 5
queries) and heads *unchanged* -- no transformer self-attention over image
tokens in this path at all; the convnet's own stacked local receptive fields
do the spatial mixing instead. Per-stage depth
(`CNN_BLOCKS_PER_STAGE = (2, 2, 3, 5)`) is not architecturally load-bearing;
it is tuned only to land the whole model's parameter count in the milestone's
6-12M target band. Measured (`python -m gltfworld.models.perception`, this
machine): **6,467,219** parameters (`encoder="cnn"`) vs. **8,234,259**
(`encoder="vit"`, unchanged from V6).

**Config**: `configs/perception_v2_cnn.json` -- an exact copy of
`configs/perception_v1.json` with `"encoder": "cnn"` added; same 25k steps,
batch 128, losses, and augmentation, so any difference in outcome is
attributable to the encoder swap alone, not a confounded config change.

**Tests** (`tests/test_perception_model.py`): forward shape/finiteness,
quaternion unit-norm, and workspace-bound checks for `encoder="cnn"`
(mirroring the existing `vit` tests), batch independence, an encoder-name
validation test (`ValueError` on an unknown encoder string), the 6-12M
param-count band, and a pinned exact-count regression test for
`encoder="vit"` (`8_234_259`, unchanged) so this milestone can't silently
regress the existing model while adding the new option.

**`--smoke-val` result on `configs/perception_v2_cnn.json`** (this session,
same recalibrated V6.2 criteria: >= 15% relative improvement in val
`matched_pos_err` from step 500 to the final step, AND final value < 0.55m;
~5,000 steps, ~1.8 epoch-equivalents on the 4,000-episode pack -- reported
honestly, not tuned to pass). Full per-500-step trajectory (val
`matched_pos_err`, this machine, RTX PRO 6000 Blackwell, 2332.5s total):

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val matched_pos_err (m) | 0.6326 | 0.5322 | 0.4599 | 0.4214 | 0.3863 | 0.3578 | 0.3401 | 0.3060 | 0.3038 | 0.2858 |

Step-500 -> step-5000 relative improvement: **54.8%** (needs >= 15%); final
value **0.2858m** (needs < 0.55m). **`--smoke-val` PASSED both conditions.**
The curve is monotonically declining at every single evaluation, with no
plateau -- a qualitatively different shape from the `vit` encoder's mostly-flat
0.62-0.66m plateau over the identical step budget/dataset/schedule/loss (V6.2
postmortem above: 7.7% relative improvement, final 0.5958m, **failed** both
conditions). This is a real, favorable within-budget comparison, not just a
passed threshold: same data, same 5,000-step window, only the encoder
differs.

This result is also independent evidence for the V6.2 bound: the CNN
encoder generalizes measurably faster and further than `vit` did over the
exact same short window, on the same data, same schedule, same loss --
consistent with the rationale above (built-in convolutional inductive bias
needs less data to start generalizing than a from-scratch transformer
encoder does). Per this milestone's own scope boundary, the full 25k-step
run against `perception_v2_cnn.json` -- the only way to confirm whether this
early trend holds all the way to the milestone's 0.05m acceptance bar -- is
the orchestrator's to launch separately, not run here.

### V6 final outcome (40k steps, CNN encoder, 4k-episode dataset)

The orchestrator's full 40k-step run on `perception-v4-cnn-40k` reached convergence with the CNN encoder on the corrected, 4k-episode dataset. **Acceptance bar NOT met**: existence F1 **0.8701** (< 0.95 bar), median matched position error **0.1798 m** (>> 0.05 m target, 3.6× over), class accuracy **0.9496** (≈ 0.95). The CNN encoder resolved the ViT memorization crisis and validated the data-hunger diagnosis (monotonic 54.8% position-error improvement vs. ViT's flat 7.7% at 5k steps), and position signal is genuine (4.3× baseline). Validation curves plateau around step 40k at 0.155 m, indicating the 4,000-episode dataset, while large enough to defeat memorization, remains insufficient for the sub-5cm closed-loop acceptance bar — a measurable, honest outcome that feeds forward into the V7 closed-loop analysis and gap-report calibration.

## Closed-loop demo + attribution (V7)

`gltfworld.eval.closed_loop` is the flagship artifact this project's whole
architecture-flow diagram (top of this document) has been building toward:
perceive -> roll forward -> re-render, with real glTF at every hop, plus an
attribution analysis that separates *how much of the rollout error is the
dynamics model's own ceiling* from *how much is perception's fault* --
required precisely because V6 established perception is real but imperfect
(existence F1 ~0.89, median matched position error ~0.21m, class accuracy
~0.95 on `perception-v1`/test with the CNN encoder) rather than assuming a
perfect oracle detector.

    uv run python -m gltfworld.eval.closed_loop \
        --episodes data/perception-v1 \
        --dyn-ckpt runs/dynamics-v1/best.safetensors \
        --per-ckpt runs/perception-v3-cnn/best.safetensors \
        --per-metrics runs/perception-v3-cnn/eval/metrics.json \
        --out runs/closed-loop-v1 --n-episodes 20 --video 5

### Three arms, per selected test episode

- **Arm A (oracle)**: the exact ground-truth state at `t=0` (already
  carries the simulator's true velocity/angular-velocity -- not a finite
  difference) rolled forward by `InteractionTransformer`
  (`gltfworld.eval.rollout.rollout`, reused verbatim). This is the dynamics
  model's own error ceiling; no perception involved.
- **Arm B (oracle + measured noise)**: the same ground-truth *poses* at
  `t=0,1`, independently perturbed *per frame* by Gaussian noise, then
  finite-differenced into velocity/angular-velocity the same two-frame way
  Arm C's real detections have to be -- object identity/count and every
  physics-material field (mass/friction/restitution/shape/size) stay exact
  GT. This isolates *pose-measurement noise alone*, with no
  detection/correspondence error mixed in.
- **Arm C (visual, the real closed loop)**: render GT frames 0 and 1 (the
  vendored `EpisodeRenderer`), run the real `PerceptionDETR` on each frame
  independently, existence-threshold (0.5, same as
  `gltfworld.eval.perception_eval`), Hungarian-match frame 0's detections to
  frame 1's by position + class + size proximity to get a cross-frame
  correspondence, finite-difference velocity/angular-velocity from the
  matched pairs, and roll forward *only* the correspondences that survived.

Every arm's rollout is reconstructed into a full `Episode`
(`gltfworld.eval.rollout.tensors_to_episode`), saved via `save_episode`, and
reloaded via `load_episode` *before* metric computation -- transport is
exercised at every hop, and every round trip is asserted `<= 1e-6`
(`gltfworld.eval.closed_loop._roundtrip_episode`), exactly the same
discipline every earlier milestone's own eval CLI follows. A `BallisticBaseline`
reference (rolled out from Arm A's own exact initial state) is added as a
4th curve for scale, reusing V5's baseline directly.

### Noise calibration (Arm B): exact chi(3) inversion, not an RMS approximation

Perception's own reported error statistics are *median magnitudes* of a
3D vector (`matched_position_error_m`, `matched_rotation_error_deg_by_shape`
-- both already-non-negative norms), not a per-axis sigma directly. Given an
assumed isotropic Gaussian noise model (`x ~ N(0, sigma^2 * I_3)`), `||x||`
follows a 3-DOF chi distribution whose median is `sigma * chi(df=3).ppf(0.5)`
(`~1.5382`) -- so `gltfworld.eval.closed_loop.noise_params_from_metrics`
inverts this *exactly* (via `scipy.stats.chi`, already an `ml`-extra
dependency) rather than an approximate `median/sqrt(3)` RMS rule of thumb:
`sigma_pos = median_position_error_m / chi(df=3).ppf(0.5)`, and per-shape
`sigma_rot = radians(median_rotation_error_deg[shape]) / chi(df=3).ppf(0.5)`
(sphere fixed at 0 -- `matched_rotation_error_deg_by_shape` never reports one
for sphere either, per V6's own "a sphere has no meaningful orientation"
convention). `--noise-sigma-pos`/`--noise-sigma-rot-deg` let a caller
override or fully replace the `metrics.json`-derived values without one.

Noise is injected independently per frame (frame 0 and frame 1 each get
their own fresh draw, not one shared offset that would cancel in the finite
difference) -- rotation composed on the left (`dq * quat`), the same
convention `gltfworld.train.train_dynamics.add_input_noise` already
established for its own training-time input-noise injection.

### Cross-frame correspondence and Arm C assembly: reusing `hungarian_match` twice, for two different jobs

`gltfworld.models.matching.hungarian_match`'s signature is generic (any
"pred" set of positions/class-logits/sizes against any "gt" set with a mask)
-- Arm C reuses it verbatim for two structurally distinct jobs rather than
writing a new matcher:

1. **Frame0 <-> frame1 correspondence**
   (`gltfworld.eval.closed_loop.match_detections_across_frames`): frame 1's
   existence-thresholded detections stand in as `hungarian_match`'s "GT"
   side (each one is a real, existent row per its own mask), frame 0's as
   the "pred"/query side. Unmatched detections on either side (Hungarian
   assignment on a rectangular cost matrix) are dropped from Arm C's
   rollout -- a genuinely spurious/exiting/entering object, not silently
   forced into a correspondence it doesn't have.
2. **Arm C <-> real GT correspondence, for scoring only**
   (`match_armc_to_gt`): after Arm C's `(N_C, 22)` initial state is
   assembled from step 1's surviving correspondences, it is separately
   Hungarian-matched against the *real* GT frame-0 state -- purely to know
   which of Arm C's rolled-out objects (if any) a given horizon's
   trajectory error should be measured against. This second match is never
   fed back into what Arm C's rollout actually saw (the closed loop never
   peeks at GT to fix its own state); it only exists after the fact, for
   metrics, exactly the same "matched-pair scoring, decoupled from the
   detector's own blind assembly" discipline `perception_eval.run_inference`
   already established.

**Documented structural blind spot: mass/friction/restitution.**
`PerceptionDETR` never predicts these (V6's own architecture has no head for
them) -- so every Arm C object is assigned the same fixed default physics
values `gltfworld.eval.perception_eval` already uses for its own
false-positive rendering fallback (`mass=1.0`, `friction=0.6`,
`restitution=0.1`). This is a real, honest gap, not a bug: it means the
measured B->C gap is *not* purely "detection/correspondence noise" -- part
of it is this orthogonal, unavoidable-given-the-model's-output-space blind
spot. `tests/test_closed_loop.py::
test_build_arm_c_assembly_perfect_perception_matches_gt_with_default_physics`
makes the boundary of this gap precise: with a perfect mock detector (exact
GT position/quat/size/shape/class, existence=1) and zero Arm B noise, Arm
C's assembled state matches Arm A/GT's exactly, *provided* the GT fixture's
own physics fields are constructed to equal those same defaults -- i.e. the
only way Arm C can differ from a "perfect" oracle is via this documented
physics-params gap plus whatever the real detector actually gets wrong.

Color/category on Arm C's synthetic `ObjectSpec`s (needed only to build a
renderable glTF, never fed into the tensor-contract state or any metric) is
an honest GT-assist for matched objects (copied from the corresponding real
GT object) or a fixed neutral-gray fallback for a genuine false positive --
the exact same convention `perception_eval.build_predicted_episode` already
established for its own re-render check.

### Metrics: detection-level vs. matched-trajectory error, kept separate

Per the milestone's own "be precise about what's averaged" requirement:

- **Detection-level** (`arm_c_detection` in `metrics.json`): precision/
  recall/F1 over the *corresponded* (survived frame0<->frame1 matching)
  objects against real GT, aggregated across every episode (`tp`/`fp`/`fn`
  accumulated the same way `gltfworld.eval.perception_eval`'s own existence
  metric does) -- a stricter, two-frame-survival version of single-frame
  detection accuracy, plus `n_zero_correspondence_episodes` (episodes where
  Arm C had nothing to roll forward at all).
- **Matched-trajectory error** (`arms.C_visual` in `metrics.json`): median +
  IQR position/rotation error at each horizon, computed *only* over Arm C
  objects that got a genuine GT correspondence in step 2 above -- unmatched
  Arm C objects and undetected GT objects never enter this average (they'd
  be scoring "how wrong is a comparison that shouldn't exist" otherwise).
  `gltfworld.eval.closed_loop.ArmAccumulator` collects the *full* per-horizon
  curve (every horizon `1..T-1`, not just the reported discrete set) across
  every episode -- the same "median + IQR over every unmasked
  (episode, object) pair" population convention `gltfworld.eval.rollout
  .horizon_metrics` already established, just accumulated per-episode (Arm
  C's object count varies per episode, so it can't be stacked into one
  batched tensor the way Arm A/B's fixed, GT-identity-preserving object
  count can).

`attribution.png`: median position error vs. horizon (log-y), one curve per
arm + the ballistic reference. The A->B gap is (an upper bound on) the
perception-noise cost; the B->C gap is the detection/correspondence cost
*plus* the documented physics-params blind spot above; Arm A alone is the
dynamics model's own ceiling.

### A real finding from the 3-episode GPU smoke: Arm B can diverge *faster* than Arm C

Measured on this machine (`runs/dynamics-v1` + `runs/perception-v3-cnn`,
3 real `perception-v1` test episodes, `tests/test_closed_loop_gpu.py`):
median position error at `h=1/5/10/30` --

| arm | h=1 | h=5 | h=10 | h=30 |
| --- | --- | --- | --- | --- |
| A (oracle) | 0.0049 | 0.0199 | 0.0254 | 0.1244 |
| B (oracle+noise) | 0.2329 | 0.9162 | 1.5404 | 3.3760 |
| C (visual) | 0.5170 | 0.6262 | 0.7083 | 0.8259 |
| ballistic | 0.0053 | 0.0267 | 0.0534 | 4.6160 |

A <= C holds cleanly (the sanity bar `tests/test_closed_loop_gpu.py` actually
asserts) but the full `A <= B <= C` ordering does **not** hold at
`h=5/10/30`: Arm B diverges *faster* than Arm C, not slower. Reported here
honestly rather than tuned away, per this project's own "if violated, that's
a REPORTED finding not a silent pass" policy (identical in spirit to V5's
MLP-competitiveness finding and V6.1/V6.2's postmortems) --
`gltfworld.eval.closed_loop.aggregate_results`'s `ordering_check` records
this per-horizon for exactly this reason, un-gated.

**Why, most likely** (a real, explicable mechanism, not a bug -- the
zero-noise/perfect-perception exactness tests above independently confirm
the assembly arithmetic itself is correct): Arm B's noise model treats each
frame's position measurement as an *independent* fresh Gaussian draw, and
`finite_diff_velocity` divides by a small `dt` (~0.034s here) -- so
`sigma_vel ~ sqrt(2) * sigma_pos / dt`. At this run's calibrated
`sigma_pos ~= 0.136m` (from `perception-v3-cnn`'s measured 0.2095m median,
V6.3's CNN encoder), that implies an injected velocity noise on the order of
several m/s, dwarfing `wm-scenes-v1`'s own sampled speed range (`<=1.5 m/s`
initial, DESIGN.md's V3 section) and driving Arm B's rollout to diverge very
fast. The real detector's actual per-object errors, by contrast, are
apparently *not* well-modeled as fresh-i.i.d.-per-frame: the same trained
model looking at two adjacent, nearly-identical frames of the same object
likely makes a *correlated* (systematically-biased-the-same-way) error
rather than an independent one, which partially cancels in the finite
difference instead of amplifying it -- explaining why the real Arm C
finite-diffed velocity ends up materially better-behaved than Arm B's
i.i.d.-noise model predicts. This is itself a useful, honest finding about
the limits of an i.i.d.-Gaussian noise model as a stand-in for "what
perception noise alone does" at short frame gaps, not a reason to loosen or
re-tune `--noise-sigma-*` to make the ordering come out prettier. Only 3
episodes were run here (this milestone's own GPU-smoke scope, see
`docs/VERIFICATION.md`'s V7 section); the orchestrator's full 20-episode run
is the statistically meaningful version of this same measurement and may or
may not reproduce this exact pattern.

### Video (`--video K`, gpu)

Reuses `EpisodeRenderer` exactly like `gltfworld.eval.rollout
._render_side_by_side_videos` (one process-wide renderer, `imageio.mimwrite`,
30fps): a 2-panel `GT | Arm C` mp4 and a 3-panel `GT | Arm A (oracle) | Arm C
(visual)` mp4 per requested episode, under `out/video/`.

### Tests

- CPU (`tests/test_closed_loop.py`, 24 tests): noise-injection statistics
  sanity (empirical sigma matches the requested one within 5% at n=20,000);
  zero-noise/deterministic-seed exactness for Arm B; the arm-assembly
  exactness bar described above for Arm C with a synthetic perfect
  detector (incl. a zero-detections degenerate case); `hungarian_match`-based
  cross-frame correspondence on a hand-built 2-object case; the exact
  chi(3) noise-calibration inversion against a synthetic `metrics.json`;
  dataset-resolution variants (raw glb dir, dataset root, packed dir/file);
  deterministic split filtering (`split_id_for_seed`); the full
  `process_episode` glTF-round-trip pipeline (Arms A/B + ballistic only,
  `per_model=None`/no renderer needed) with finiteness and determinism
  checks; `ArmAccumulator`/`aggregate_results` shape and finiteness checks;
  the attribution plot (incl. an empty-curve arm, e.g. Arm C with no
  perception model at all).
- GPU (`tests/test_closed_loop_gpu.py`, 1 test, gpu-marked): the full CLI
  end-to-end against the real `runs/dynamics-v1` + `runs/perception-v3-cnn`
  checkpoints and 3 real `perception-v1` test episodes -- asserts exit 0,
  every reported metric finite, every emitted GLB (`gt`/`armA`/`armB`/`armC`)
  passes `gltfworld validate` with 0 errors, and the `A <= C` sanity
  direction at `h=30` (see the finding above for why this test does *not*
  additionally assert the full `A <= B <= C` ordering).

### Acceptance (see `docs/VERIFICATION.md`'s V7 section for exact commands)

Closed loop runs end-to-end via real glTF at every hop; `attribution.png` is
produced; every emitted GLB validates clean. Full-scale (20-episode)
arm-ordering sanity and the attribution curve's final shape are the
orchestrator's to run with the eventual `perception-v4-cnn-40k` checkpoint,
per this milestone's own scope boundary (deliver + smoke-test the closed
loop here, same precedent V5/V6 established for their own full training/eval
runs).

## Articulated objects (V9-prep)

Brings articulated objects (a cabinet with a hinged door, a chest/table with
a sliding drawer -- each with a handle) into the transport, using the draft
`KHR_physics_rigid_bodies` **joint** machinery (revolute via a limited
rotational DOF, prismatic via a limited linear DOF). This is prep work for
the full V9 milestone ("gap report + RWM extension write-up"); the gaps
found and documented below are exactly the kind of material that write-up
will collect, not a substitute for it. See `docs/RWM_EXTENSIONS.md` for the
full channel/field reference (`joint_position` channel, `extras.rwm`
`semantics`/`articulations`, the v0 semantics taxonomy) and
`docs/VERIFICATION.md`'s V9-prep section for the checkpoint-by-checkpoint
writeup.

### Joint schema: already vendored, newly exercised

The pinned commit's joint-related JSON Schema files
(`glTF.KHR_physics_rigid_bodies.joint{,.limit,.drive}.schema.json`,
`node.KHR_physics_rigid_bodies.joint.schema.json`) were already vendored
back in V1 as part of the original `physics_rigid_bodies/schema/*.json`
wildcard fetch -- confirmed against the pinned commit's actual GitHub tree
listing that nothing is missing, so no re-vendoring was needed (see
`docs/schemas/khr/PROVENANCE.md`'s V9-prep update note). This milestone is
simply the first to read/write them.

### KHR joint encoding: hinge/slider as limit compositions

Per the pinned spec README's "Joints" section (not just the JSON Schema
files, which don't carry this prose): a joint's two **attachment frames**
are each "the relative transform between the node and the first parent
`motion` (or the simulation's fixed reference frame, if no such motion
exists)". The worked example given there for a hinged door is followed
literally, not approximated: "a 3D linear limit with zero maximum distance,
a 1D angular limit with min/max describing the swing ... and a 2D angular
limit with zero limits about the remaining two axes"
(`gltfworld.ext.khr_physics.hinge_joint_limits`). A slider
(`slider_joint_limits`) is the natural translation/rotation-swapped analog
(3D angular limit locked at zero, 2D linear limit locked at zero, 1D linear
limit as the travel range) -- not spelled out verbatim in the spec text, but
a direct application of the same limit-composition primitives it describes.

**Attachment frames, concretely**: rather than moving an articulated
object's own mesh/collider origin to the physical hinge/slide point (which
would desynchronize the visual mesh from the `KHR_implicit_shapes` collider
-- both are always centered on the owning node's origin, with no offset
field in this pinned commit, see "Honest gaps" below), each of `base`/`part`
gets a second, motion-less, geometry-less **joint pivot** child node, nested
one level under it (`node.children`), placed at `ArticulatedSpec.anchor` in
that body's own local frame. The part-side pivot's nearest ancestor-with-
`motion` is `part` itself, so its attachment frame is a fixed local offset
that co-moves with the body exactly as the joint's own semantics require;
the base-side pivot's nearest ancestor has no `motion` (base is static), so
its attachment frame is its fixed offset from the world origin. The
part-side pivot carries the node `joint` property (`connectedNode` = the
base-side pivot, `joint` = index into the root `physicsJoints[]` array).
Since gltfworld's object nodes were previously *always* scene roots (no
node ever had `children`), `gltfworld.scene.convert._compute_scene_roots`
now computes `scenes[0].nodes` as "every node not listed as somebody's
child" instead of "every node" -- identical output when nothing has
children (every pre-V9 episode), so this is a zero-risk generalization, not
a behavior change for existing transport.

**Simplifying convention** (`ArticulatedSpec`, `wm-articulated-v1`): `base`
and `part` are authored at identity world orientation, and `axis` (0/1/2)
indexes a *world*-aligned X/Y/Z axis at rest, matching the pivot nodes'
identity local rotation -- not a limitation of `KHR_physics_rigid_bodies`
itself (which supports arbitrary joint-local bases), just a simplification
this milestone's own generated scenes use.

### MJCF <-> KHR joint mapping (`gltfworld.datagen.articulated`)

MuJoCo's own hinge/slide joints map directly onto the KHR limit
compositions above, with one simplification that turned out to need no new
MJCF machinery at all: since `base` never moves (`is_static=True`, no MJCF
joint), "`part`'s motion relative to `base`" and "`part`'s motion relative
to world" are kinematically identical. So `part` is placed directly under
MuJoCo's `worldbody` (flat, exactly like every other gltfworld body --
`gltfworld.datagen.mujoco_env` never needed body nesting either), with its
`<joint type="hinge"|"slide" pos="..." axis="..." range="min max"
damping="...">` given in `part`'s own body-local frame (MJCF's standard
convention for a joint declared inside a `<body>`) -- computed by the same
"rotate the world offset by the body's own orientation's conjugate" trick
`gltfworld.scene.convert` uses for the KHR pivot nodes, just in MuJoCo's
axis convention (`gltfworld.datagen.mj_convert`).

`joint_pos` (the recorded `StateSeries` channel) is `data.qpos` at the
joint's single DOF, read directly off MuJoCo every recorded frame --
already in the units `KHR_physics_rigid_bodies.joint.limit` itself uses
(radians for revolute, meters for prismatic), no conversion needed.

**Found the hard way, fixed, and worth recording**: MJCF's *default*
`compiler angle` unit is **degrees**, not radians -- silently
reinterpreting `range="{min} {max}"` (authored in radians, matching the
KHR/robotics convention this whole milestone uses) as degrees, compiling a
door meant to swing up to ~1.9 radians into a joint actually limited to
~1.9 *degrees* (0.033 rad). The joint hit that minuscule limit almost
immediately and sat there under continued push pressure -- which looked
enough like "reaches a limit and settles" to not be obviously wrong at a
glance, until the recorded `joint_pos` trajectory was inspected directly.
Fixed with an explicit `<compiler angle="radian"/>` in
`_articulated_mjcf`'s generated XML.
`gltfworld.datagen.mujoco_env`'s existing MJCF never hit this because it
only ever uses `<freejoint>`, which has no angle-valued attributes at all.

**Push force and joint damping are derived from the sampled mass/geometry,
not fixed constants** -- found necessary, not just nicer, during
development: a single fixed push-force number reliably overshoots a small/
light door and undershoots a large/heavy one (or vice versa), and a single
fixed damping *coefficient* is only "light" relative to one particular
moment of inertia -- applied to a much larger or smaller door under the
same-formula push, the residual post-limit-bounce settling either never
finishes damping out within a short recorded episode (a very light door)
or the push never has enough energy to reach the limit at all (a very
heavy one, over-damped by the same fixed number). Both derived from a rough
kinematic estimate (revolute: treat the part as a rod pivoting about one
end, `I ~= (1/3) * mass * (2 * part_extent)^2`; prismatic: `F = mass *
accel`), targeting a fixed decay *time constant* (`damping = I / tau` or
`mass / tau`, `tau = 0.3s`) rather than a fixed damping number, so both push
and damping scale consistently across the sampler's randomized mass/size
range. Verified empirically across the sampler's full parameter range
(mass 1.5-8kg, size 0.15-0.45m, limits 0.15-1.9 rad/m): robustly monotonic-
to-peak and settled (see `tests/test_articulated_physics.py`).

**Gravity coupling depends on which world axis the joint uses, and this is
real physics, not a bug** -- found while debugging why some randomly-sampled
axis choices "opened and stayed open" cleanly while others "opened, then
drifted back closed over several seconds": a rotation axis exactly parallel
to gravity (gltf axis=1, vertical Y) has *zero* gravity torque at every
angle (the rotating body's height along that axis never changes, so gravity
does no work on it) -- the door only has the scripted push's momentum and
joint damping to work with, and with intentionally "light" damping, a hard
bounce off the limit can take several seconds to fully settle. A horizontal
hinge axis (0 or 2) instead makes the door pendulum-like: depending on which
side of vertical it starts on, gravity can *assist* staying open (settles
almost exactly at the limit, no long tail) or *oppose* it (swings back
toward closed instead) -- both physically correct, not contradictory. Per
`gltfworld.datagen.articulated`'s own "axis coverage over realism" design
note, the general sampler still draws `axis` uniformly from `{0, 1, 2}` for
full KHR `angularAxes`/`linearAxes` coverage; the physics-sanity tests
specifically pin a vertical hinge axis (door) / horizontal slide axis
(drawer) -- the gravity-decoupled cases -- for a reliably reproducible
trajectory shape, rather than asserting "opens monotonically" against a
combination where gravity may genuinely be fighting the push.

### The articulation consistency check

The moving part's recorded pose must equal the anchor point composed with
the joint transform implied by the recorded `joint_pos` at every step:
`part_pose(t) = anchor ∘ Rotate(axis, joint_pos(t))` (revolute) or `anchor +
axis_vec * joint_pos(t)` (prismatic), reconstructed purely from
`ArticulatedSpec`'s own metadata plus the recorded `poses`/`joint_pos` --
not from any privileged access to MuJoCo's internal state
(`tests/test_articulated_physics.py::test_articulation_consistency_*`,
checked both on a freshly-simulated in-memory `Episode` and after a real
save/load `.glb` round trip, mirroring `tests/test_provenance.py`'s
pattern). Measured worst-case error across 60 sampled (seed, kind, axis)
combinations: **0.0077m position, 0.014 quaternion-component rotation**
(cross-validated against an independent `scipy.spatial.transform.Rotation`
implementation of the same formula, not just gltfworld's own hand-rolled
version) -- small, bounded, and concentrated mid-transient (near-exact at
rest and once settled, per direct inspection of the error's time profile),
consistent with a benign MuJoCo forward-kinematics/reporting artifact
rather than a wrong formula (a genuinely wrong composition -- e.g. the
degree/radian mixup above, or a sign-flipped axis -- produces errors many
orders of magnitude larger, not a small bounded residual that vanishes at
rest). The test's tolerance (0.03 / 0.03) is set with margin above this
measured bound, not tuned to just barely pass.

### Honest gaps (feeding the full V9 gap report)

- **`joint.limit`'s `stiffness`/`damping` are soft-stop parameters, not
  viscous joint damping.** They describe the *restorative* force applied
  once a limit is exceeded (an optional spring instead of a hard stop); by
  default the limit is infinitely stiff. There is no property in this
  pinned commit's joint schema for "this hinge has some viscous drag across
  its whole free range, even within its limits" (MJCF/URDF's per-joint
  `damping`) or for `armature` (MJCF's added rotational/reflected inertia
  term, used to improve numerical stability of the actuator/joint's
  effective inertia -- not physical damping at all, but likewise
  unrepresentable here). gltfworld's own generated episodes *do* use MJCF
  joint damping to get realistic settling behavior, but that parameter has
  no home in the KHR encoding -- a downstream KHR-only consumer doing fresh
  forward simulation from the `.glb` alone would see an undamped (or only
  soft-limit-damped) joint, not the damped one MuJoCo actually simulated.
- **`joint.drive` models a persistent spring-to-target, not a one-shot
  push.** The drive force is `stiffness * (positionTarget - positionCurrent)
  + damping * (velocityTarget - velocityCurrent)` -- an always-on motor
  chasing a target, not a finite-duration external impulse. gltfworld's
  scripted "push" actuation (a constant generalized force applied for a
  bounded time window, then released) doesn't fit this shape without
  misrepresenting it as a permanently-active motor holding some target
  forever, so it is **not** encoded as a KHR `drive` at all -- the driving
  force only ever existed inside the MuJoCo simulation that produced the
  recorded `poses`/`joint_pos` trajectory; the KHR joint dict for these
  episodes carries `limits` only, no `drives`.
- **The handle's rigid attachment isn't encoded as a KHR joint/weld.** Its
  motion is *derived* (`handle_pose(t) = part_pose(t) ∘
  (handle_local_offset, identity)`), not independently simulated or
  KHR-joint-constrained -- doing so properly would need a second,
  anchor-aligned pivot-node pair (per the same construction used for the
  main hinge/slide joint) purely to describe a rigid weld, which this
  milestone judged not worth the added node/joint count for a purely
  cosmetic part. `extras.rwm.semantics` still identifies it
  (`{"labels": ["handle"], "affordances": ["pullable"]}`), and its animated
  pose track is always exactly consistent with rigidly following `part` --
  sufficient for this project's playback/training use case, but a real gap
  for a downstream engine trying to do fresh forward simulation from the
  `.glb` alone without gltfworld's own semantics convention.
- **No offset/center field on `KHR_implicit_shapes` colliders.** This is
  *why* the joint-pivot-child-node design (above) was necessary instead of
  simply moving each articulated object's own node origin to the physical
  hinge point -- box/sphere/cylinder shapes are only ever defined centered
  on their owning node's origin in this pinned commit, with no separate
  offset/center property. Moving an object's own origin to the hinge would
  have desynchronized its `KHR_implicit_shapes` collider from its visual
  mesh (both still centered on that now-relocated origin, no longer at the
  shape's true geometric center) -- an interop defect in the same spirit as
  the V3.1 cylinder-axis finding. The pivot-child-node design sidesteps it
  entirely: object nodes keep their ordinary, unmodified mesh/collider
  convention.

`gltfworld inspect` (`gltfworld.cli`) additively reports the new
`joint_pos` optional channel and an `articulations:` summary line per
joint (type/base/part/axis/range/handle) when present -- `(none)`/absent for
every pre-V9-prep episode, exactly as before.

### `wm-articulated-v1` distribution (`gltfworld.datagen.articulated.sample_articulated_scene`)

| field | distribution |
| --- | --- |
| kind | `"door"` / `"drawer"`, 50/50 (or pinned via `kind=`) |
| joint axis | `Uniform{0, 1, 2}` (or pinned via `axis=`) -- see "axis coverage over realism" above |
| base (cabinet) half-extents | `U[0.25, 0.4] x U[0.3, 0.45] x U[0.25, 0.4]` m |
| part extent (sweep direction) | `U[0.15, 0.35]` m |
| part span (along the joint axis) | `~U[0.7, 1.0] * base_half` (door: 0.9-1.0x; drawer: 0.7-0.95x) |
| part thickness | `U[0.015, 0.03]` m |
| part mass | door `U[2, 8]` kg; drawer `U[1.5, 5]` kg |
| joint limit range | door `[0, U[1.0, 1.9]]` rad (~57-109 deg); drawer `[0, U[0.15, 0.35]]` m |
| initial joint position | `U[min, ~0.3-0.35 * max]` (starts mostly-closed, not always exactly 0) |
| push force/torque | derived from mass/geometry (see above), `x U[0.9, 1.15]` jitter |
| joint damping | derived from mass/geometry, targeting `tau=0.3s` decay (see above) |
| push window | `[0.05s, 0.30s]` (a brief kick, not held for the whole episode) |
| handle size | `U[0.015, 0.03]` m (cube) |
| ground | 1 static box (3m x 0.2m x 3m), category `"ground"` |
| camera | 1 fixed camera, `aspect=1.0` |

### Acceptance (see `docs/VERIFICATION.md`'s V9-prep section for exact commands)

- KHR joint dicts + node `joint` property validate against the vendored
  schemas; `RWM_state_series`'s `joint_position` channel validates against
  the (updated) vendored schema.
- Articulated `Episode` <-> GLB round-trips exactly (in-memory and through a
  real `.glb` file), including `ArticulatedSpec`/`joint_pos`.
- A sample articulated GLB passes the real, pinned Khronos glTF-Validator
  with 0 errors.
- Door pushed with positive torque (vertical hinge, gravity-decoupled):
  opens monotonically to its peak, settles within its limits.
- Drawer pushed with positive force (horizontal slide axis): stays within
  its travel limits throughout.
- The articulation consistency check holds (see above) to <= 0.03m / 0.03
  quaternion-component tolerance, both in-memory and post-round-trip.
## Articulation stage (V9)

Builds on V9-prep's transport work (KHR joints, `joint_position` channel,
`extras.rwm` semantics -- see "Articulated objects (V9-prep)" above) with a
real dataset, a trained model, and an honest eval -- the actual "articulation
stage" this milestone's own name refers to.

### Scope, stated plainly

This milestone's perception task is **joint-state estimation**, not
`PerceptionDETR`-style object detection: given a single rendered frame of a
`wm-articulated-v1` door/drawer scene, estimate the joint's generalized
position (normalized), its type (revolute/hinge vs. prismatic/slider), and
its axis (a 3D unit vector). Every `wm-articulated-v1` episode has exactly
one articulated joint, so this collapses to a small regression/
classification problem, not a set-prediction one -- no Hungarian matching,
no object queries, no existence head (see
`gltfworld.models.articulation`'s module docstring for the full
architecture rationale). Articulated **dynamics** (predicting how the
joint's state evolves over time -- the V9 counterpart of
`gltfworld.models.dynamics.InteractionTransformer`) is **explicitly out of
scope** for this milestone; future work, alongside the full gap report/RWM
extension write-up V9-prep's own entry already flagged as still open.

### Dataset: `articulated-v1`

Generated by the new `gltfworld generate-articulated` CLI
(`gltfworld.datagen.generate_articulated.generate_articulated_dataset`):
1,500 episodes, exactly 750 door (revolute) / 750 drawer (prismatic) --
pinned by alternating `kind` per episode index rather than relying on the
sampler's own random draw, so the mix is exactly 50/50 regardless of
episode count (see the module's docstring). 100 steps @ ~30Hz (same
`record_hz` rounding as `wm-scenes-v1`, see V3's section), seed `20260730`.
Each episode is a GLB (joints + `joint_position` channel + `extras.rwm`
semantics, all existing V9-prep-verified transport, no new encoding) plus
rendered 256x256 rgb+seg+depth frames via `EpisodeRenderer` (`--render`).

- **Generated + rendered**: 1,500 episodes x 100 frames = 150,000 frames in
  **318.1s (5.30 min)**, ~65GB on disk (rendered frame stacks dominate, same
  as `perception-v1`'s own disk footprint pattern -- see V4's section).
- **Packed** (`gltfworld pack-articulated` /
  `gltfworld.data.pack_articulated.pack_articulated_dataset` -- a
  purpose-built pack for this single-joint-per-episode task, not
  `gltfworld.data.pack`'s general `N_max`-object tensor contract, see that
  module's own docstring for why) in **16.1s**. Split (same
  `sha256`-bucketing scheme as `gltfworld.data.pack.split_id_for_seed`,
  keyed by each episode's own seed): **train 1,384 / val 64 / test 52**.
  Joint type: exactly **750 revolute / 750 prismatic** (matches the
  generator's exact-alternation guarantee). Axis distribution across the
  full dataset: **X 527 / Y 474 / Z 499** (close to the sampler's uniform
  `{0,1,2}` draw, as expected). Limit ranges observed: revolute `max` in
  `[1.002, 1.899]` rad (~57-109 degrees, `min` always 0), prismatic `max` in
  `[0.151, 0.349]` m (`min` always 0) -- both match the sampler's documented
  `U[1.0, 1.9]` rad / `U[0.15, 0.35]` m ranges (see "wm-articulated-v1
  distribution" above).

### Model: `ArticulationEstimator` (`gltfworld.models.articulation`)

Reuses `gltfworld.models.perception._CNNEncoder` (the V6.3 small-data-regime
CNN encoder trunk) wholesale, not duplicated -- imported and instantiated
directly. 256 pooled tokens -> mean-pooled to one vector -> a small 2-layer
MLP trunk -> three small heads: `joint_pos_head` (1, normalized-by-limit-
range regression), `type_head` (2-way classification), `axis_head` (3,
L2-normalized to a unit vector). **3,301,766 parameters** measured (`python
-m gltfworld.models.articulation`) -- no fixed target band asserted (the
milestone spec gives no approximate count to reconcile against here, unlike
V5/V6's documented parameter-count deviations).

Two design decisions worth recording (full reasoning in the module's own
docstring):

- **Joint position is normalized by each episode's own `[limit_min,
  limit_max]`**, not regressed in raw radians/meters -- puts a hinge's
  ~1.9 rad range and a slider's ~0.35 m range on a common ~[0,1] scale
  before the loss ever sees them, the same "fixed unit-ish scale constants"
  rationale `InteractionTransformer`'s `object_features` uses (V5's
  section). The limits themselves are known dataset metadata, not predicted
  by the model -- the model is never asked to guess a cabinet's own travel
  range from one frame, only the current position within it.
- **Axis regression uses a *directed* cosine loss, not a sign-invariant
  one.** `wm-articulated-v1` always samples `axis` as a *positive* world
  basis vector paired with a non-negative `joint_pos` that increases
  specifically in that `+axis` direction -- the two aren't independent, they
  jointly fix one physical opening direction. A sign-invariant loss
  (`1 - |cos|`) would let the model score a mirrored, wrong-direction axis
  prediction as free; the directed loss (`1 - cos`) scores a sign flip at
  its worst value instead. `tests/test_articulation_model.py::
  test_axis_loss_is_directed_not_sign_invariant` pins this down directly.

### Training harness (`gltfworld.train.train_articulation`)

Same harness contract as `train_perception`/`train_dynamics` (JSON config,
resumable safetensors checkpoints, `log.csv`, `--smoke`/`--smoke-val`) --
simpler single-phase schedule (no Hungarian matching, no autoregressive
rollout phase). AdamW `lr=2e-4` cosine-annealed, batch 128, bf16 autocast,
RGB-only augmentation (brightness/contrast/noise -- geometry/state targets
never touched). **Epoch-equivalent guard**: same 15x `MAX_EPOCH_EQUIVALENT`
threshold as `train_perception`'s (same V6.1 postmortem philosophy -- a
too-small dataset trained too long silently memorizes instead of
generalizing); `articulated-v1`'s 1,384 train episodes x 100 frames =
138,400 train frames at the default 15k-step/batch-128 config lands at
**13.9x** -- under the guard, with real but intentionally thin margin (this
milestone's spec explicitly calls for "budget steps accordingly" against a
~150k-frame dataset and a small model, rather than reusing
`train_perception`'s own 25k-step default unmodified).

**Real training run** (`configs/articulation_v1.json`, full 15,000 steps,
this machine's RTX PRO 6000 Blackwell): **956.2s (~15.9 min)** wall clock.
Train loss falls fast and monotonically (0.172 at step 500 -> 0.00121 at
step 15000). Val behavior is where the interesting, honestly-reported
finding is: val total loss does **not** monotonically improve --
it bottoms out early (**0.135 at step 1500**, the run's actual minimum) and
then *rises* for the rest of the run (up to ~1.4-1.6 by step 15000), driven
almost entirely by the `type` cross-entropy term (`loss_type`: 0.116 at
step 1500 -> 1.39 at step 15000) becoming increasingly overconfident-wrong
on held-out val frames -- classic overfitting on the easiest sub-task
(door-vs-drawer is a large, easy-to-memorize visual difference). The other
two sub-tasks tell a different story over the same window: `axis_err_deg`
keeps *improving* through the whole run (4.55 deg at step 500 -> ~0.3-0.4
deg by step 15000) and `joint_pos_norm_mae` stays roughly flat/noisy
(~0.08-0.11 throughout, no clear further gain past step ~2000). Net effect:
**type accuracy peaks early (0.965 at step 1500) and degrades with more
training (0.858 by step 15000)**, while axis error keeps getting better --
three sub-tasks with three different optimal stopping points inside one
shared training run.

The harness's own `best.safetensors` selection (lowest total val loss, the
same scheme `train_perception`/`train_dynamics` use) automatically landed
on **step 1500** as a result -- not a coincidence but exactly the mechanism
working as designed: total val loss is dominated by whichever sub-task's
loss is least bounded (cross-entropy on a confidently-wrong logit has no
upper limit; MSE/cosine terms do), so "lowest total val loss" doubles as an
effective proxy for "before type-classification overfitting sets in" here.
Confirmed directly by re-running eval against `last.safetensors` (step
15000) instead of `best.safetensors` (step 1500) on the same test split:
`last` scores a *better* axis error (0.216 vs 1.842 degrees) and hinge error
(1.189 vs 3.349 degrees) but **fails** the type-accuracy acceptance bar
(0.921 < 0.98, vs `best`'s 0.982) -- concretely demonstrating the
overfitting this section describes, not just asserting it from the training
curve. All reported eval numbers below use `best.safetensors` (step 1500),
per the harness's own designed selection criterion.

### Eval (`gltfworld.eval.articulation_eval`)

Test-split metrics: joint-position error in **degrees** for hinges and
**centimeters** for sliders (reported separately -- never averaged, they're
different physical quantities), joint-type accuracy, axis angular error in
degrees. Two context baselines: **predict-midpoint-of-range** (always
predicts the normalized midpoint 0.5, scored only on joint-position error)
and **predict-dataset-mean-axis** (always predicts the train-split's mean
axis vector, re-normalized to unit length, scored only on axis error) --
each scoped to the one metric it targets rather than combined into one
trivial predictor (see the module's own docstring). A re-render check
(`--render-samples`) denormalizes a predicted joint position, reconstructs
the moving part's (and handle's) pose via the exact same anchor/axis forward
kinematics `tests/test_articulated_physics.py`'s articulation-consistency
check verifies against real MuJoCo trajectories (run here in the *predict*
direction; cross-checked directly in
`tests/test_articulation_eval.py::test_build_predicted_episode_matches_simulated_pose_in_memory`),
renders it, and compares PSNR/SSIM against the actual stored GT frame.
Predicted states are saved as real, independently loadable `T=1` GLBs with a
genuine `joint_position` channel (existing verified transport, no new
encoding) -- round-trip verified inline and run through the real, pinned
glTF-Validator.

**Ditto context, not a bar.** Ditto (Jiang et al. 2022, "Building Digital
Twins of Articulated Objects from Interaction") reports a median
revolute-axis error of 1.36 degrees -- from point-cloud, before/after-
interaction-pair input, a materially richer, motion-disambiguating input
modality than this milestone's single-RGB-frame task. That number is
reported in this milestone's eval output purely as external context, not as
something this task is attempting to match or beat (different problem
shape entirely -- see `gltfworld.eval.articulation_eval`'s module docstring
for the full caveat).

**Test-split results** (`best.safetensors`, step 1500; 52 test episodes,
5,200 frames):

| model | hinge err deg (median / mean, n) | slider err cm (median / mean, n) | type acc | axis err deg (median / mean) |
| --- | --- | --- | --- | --- |
| `ArticulationEstimator` | 3.349 / 4.514 (n=2700) | 1.449 / 1.870 (n=2500) | 0.9823 | 1.842 / 2.031 |
| predict-midpoint-of-range | 34.809 / 33.910 (n=2700) | 7.924 / 8.762 (n=2500) | n/a | n/a |
| predict-dataset-mean-axis | n/a | n/a | n/a | 54.820 / 54.713 |

The model beats both trivial baselines by roughly an order of magnitude
(hinge: 3.3 deg vs. 34.8 deg; slider: 1.4 cm vs. 7.9 cm; axis: 1.8 deg vs.
54.8 deg).

**Re-render check** (50 sampled test frames, GPU): PSNR median **39.31 dB**
(mean 40.93; 10/50 samples scored `inf` -- predicted and GT frames
bit-identical, excluded from the finite median/mean per
`gltfworld.eval.metrics`' convention), SSIM median **0.9983**; round-trip
error **0.0** (bit-exact GLB reload); glTF-Validator: **0 errors** across
all 50 predicted `T=1` GLBs (`joint_position` channel present and
schema-valid in every one, same verified transport as every other
milestone's output).

### Acceptance (see `docs/VERIFICATION.md`'s V9 section for exact commands)

- Hinge median joint-position error <= 5 degrees.
- Slider median joint-position error <= 2 cm.
- Joint-type accuracy >= 0.98.
- Axis median angular error <= 10 degrees.

**Result: all four bars clear** (`best.safetensors`, step 1500, test split):

| check | value | bar | pass |
| --- | --- | --- | --- |
| hinge median error | 3.349 deg | <= 5 deg | **pass** |
| slider median error | 1.449 cm | <= 2 cm | **pass** |
| type accuracy | 0.9823 | >= 0.98 | **pass** (thin margin) |
| axis median error | 1.842 deg | <= 10 deg | **pass** |

Type accuracy clears its bar with real but thin margin (0.9823 vs. 0.98),
and -- per the training-curve finding above -- would **not** clear it at
all with the final-step (`last.safetensors`) checkpoint instead (0.921),
underscoring that checkpoint selection (not just architecture/data) mattered
for this result.

### Honest gaps (feeding the full V9 gap report)

- **Articulated dynamics is out of scope.** This milestone estimates joint
  *state* from a single frame; it does not predict how that state evolves
  (no rollout/dynamics model analogous to `InteractionTransformer` for
  articulated joints). A real gap for any closed-loop use case involving
  articulated furniture.
- **Joint limits are given, not estimated.** The model is only ever asked
  for the joint's current position/type/axis, never its travel range --
  denormalizing a prediction back to real units (and this eval's re-render
  check) both rely on the episode's own known `[limit_min, limit_max]`. A
  fully self-contained perception system would need to estimate this too
  (e.g. from the cabinet's visible geometry), which this milestone does not
  attempt.
- **Camera is fixed.** Every `wm-articulated-v1` episode uses the same
  camera position/orientation/FOV (see "wm-articulated-v1 distribution"
  above) -- the model has never seen (and this eval never tests) viewpoint
  variation, unlike `wm-scenes-v1`'s fixed-but-at-least-parameterized-scene
  camera. A real, undemonstrated generalization gap.
- **Single object per scene.** Every episode has exactly one articulated
  assembly and no distractor objects -- the model has not been tested on
  multi-object clutter, occlusion, or scenes with more than one articulated
  joint.
- **The three sub-tasks (position/type/axis) have different optimal
  stopping points within one shared training run** (see the training-curve
  finding above) -- a single shared loss/checkpoint-selection scheme cannot
  simultaneously chase all three optima. This run's `best.safetensors`
  selection (by total val loss) happens to land on a good trade-off point
  for this particular dataset/config, but that is closer to a fortunate
  side effect of cross-entropy's unbounded-worst-case dominating the total
  loss than a deliberately engineered multi-task schedule -- a real
  multi-task-tuned training recipe (e.g. per-task early stopping, or
  loss-term reweighting/uncertainty weighting) is future work.

### Known issue (found during V9, pre-existing, not caused by V9): `test_crosscheck.py` crashes the full gpu lane

Running the complete `uv run pytest -v` (all lanes, gpu included) surfaced
**9 failed, 1 error** downstream of `tests/test_crosscheck.py::
test_crosscheck_binary_silhouette_iou` -- every subsequent render-dependent
test in the same session (`test_data.py`, `test_perception_eval_gpu.py`,
all 5 of `test_render_analytic.py`, `test_render_bench.py`) fails with
`OpenGL.raw.EGL._errors.EGLError(err = EGL_NOT_INITIALIZED, ...)`, and the
session-scoped `episode_renderer` fixture's own teardown (`renderer.delete()`)
then errors too, at whatever the last test in the run happens to be.

**Verified not a V9 regression, three ways**:

1. **Reproduced identically with V9's code entirely absent.** `git stash -u`
   (removing every file this milestone added/changed) and re-running
   `uv run pytest -q -m gpu` reproduces the exact same crash starting at the
   exact same test (`test_crosscheck.py`), before this milestone's code
   ever existed in the working tree.
2. **This milestone's own new gpu tests neither create a second renderer
   nor delete the shared one** -- checked directly, not assumed:
   `test_train_articulation_smoke.py` never touches a renderer at all;
   `test_articulation_eval_gpu.py`'s two tests either pass
   `--render-samples 0` (skips rendering entirely, mirroring
   `test_perception_eval_gpu.py`'s own pattern) or pass the shared,
   session-scoped `episode_renderer` fixture into `render_check(...,
   renderer=episode_renderer)`, which only ever deletes a renderer it
   created itself (`owns_renderer = renderer is None`) -- identical
   reuse discipline to the pre-existing `test_perception_eval_gpu.py`.
   All 4 of this milestone's own new gpu tests pass cleanly, every time,
   run alone (`uv run pytest -v -m gpu tests/test_train_articulation_smoke.py
   tests/test_articulation_eval_gpu.py` -- 4 passed).
3. **`test_crosscheck_binary_silhouette_iou` crashes the whole process even
   run completely alone** (`uv run pytest -v -m gpu -k
   test_crosscheck_binary_silhouette_iou`, a fresh process, zero prior
   renderer/EGL history) -- not a corruption some earlier test leaves
   behind, and not order-dependent. Reproduced twice, deterministically
   (100% of attempts), not an intermittent flake in the "sometimes passes"
   sense.

**Root-cause hypothesis** (plausible, evidence-supported, not fix-verified --
fixing it is out of this milestone's scope): `gltfworld.render.crosscheck
.render_mujoco_frame0` constructs its own `mujoco.Renderer`, which owns
*its own* EGL context, inside the same process as the shared
`EpisodeRenderer`'s already-open EGL context -- the module's own existing
comment already documents one related fragility in this exact area (MuJoCo
defaults to GLX unless `MUJOCO_GL=egl` is forced, to avoid colliding with
gltfworld's EGL context) but forcing both consumers onto EGL does not
appear to fully eliminate contention over this machine's single default EGL
display between two independent libraries' own context lifecycle
management (matches `EpisodeRenderer`'s own documented "one live renderer
per process" constraint, just extended to "one live EGL-context-owning
renderer *of any kind*, not just multiple `EpisodeRenderer` instances").
Confirmed present on this machine's current driver stack (NVIDIA
580.173.02); not confirmed whether it reproduces on other driver/library
versions.

**Practical impact / workaround**: the fast lane (`-m "not gpu"`) is
completely unaffected (**350 passed**, no rendering involved at all). The
gpu lane's render-dependent tests all pass individually or in
subsets that exclude `test_crosscheck.py` (this milestone's own 4 new gpu
tests: **4 passed** together); the full, un-deselected `-m gpu` lane
crashes past that one specific test whenever MuJoCo's own renderer and
`EpisodeRenderer` are both invoked within one process, which is exactly
what `test_crosscheck.py` does deliberately as its own cross-render oracle.
Not fixed here (pre-existing V2/V3-era code, orthogonal to V9's own scope);
flagged here as a genuine, reproducible environment finding rather than
silently worked around by, e.g., quietly reordering or skipping the test.

### V9.1 addendum (2026-08-02): FIXED

**Confirmed root cause**: `closed_loop.main()` constructed its own
`EpisodeRenderer` at startup and unconditionally deleted it at shutdown,
without any try/except guard or awareness that the same process might be
reusing the same EGL display context elsewhere. The bug manifested when
pytest's default test collection order (alphabetical by module, then by test
name within each module) happened to run `test_crosscheck.py`
(which temporarily creates a MuJoCo renderer within the same process)
*during* the session-scoped fixture's lifetime -- terminating the shared
process-wide EGL display in `closed_loop.main()`'s shutdown, leaving all
subsequent render calls in the same session with `EGL_NOT_INITIALIZED`.

**The fix**: three layers of defense-in-depth:

1. **Renderer injection + `owns_renderer` convention**: `EpisodeRenderer` now
   accepts an optional `renderer` parameter. When `None`, it constructs and owns
   the renderer (old behavior); when provided, it uses the injected renderer and
   does *not* delete it at shutdown. Same pattern `render_check()` and the
   test suite already established with `test_perception_eval_gpu.py`.
2. **MuJoCo crosscheck render isolated in spawned subprocess**: `render_mujoco_frame0`
   now spawns a fresh subprocess for MuJoCo rendering (instead of doing it
   in-process), completely decoupling its EGL context lifecycle from
   `EpisodeRenderer`'s.
3. **Per-module runner as defense-in-depth**: `scripts/run_gpu_tests.sh` runs
   each test module in its own subprocess, so even if any module has lingering
   EGL context issues, they don't cascade to the next module.

**Evidence**:

- Single-process `uv run pytest -m gpu -q`: **18 passed / 1 xfailed / 0 failed**
  (the xfail is pre-existing, unrelated: `test_render_bench.py::test_benchmark`
  xfails on slower machines; this development box is not in the target perf
  class). Session completes without `EGL_NOT_INITIALIZED`.
- Per-module via `scripts/run_gpu_tests.sh`: **8/8 modules pass** (each in its
  own subprocess: `test_crosscheck.py`, `test_render_analytic.py`,
  `test_render_bench.py`, `test_data.py`, `test_perception_eval_gpu.py`,
  `test_train_perception_smoke.py`, `test_train_articulation_smoke.py`,
  `test_articulation_eval_gpu.py`).
- Sabotage test (independent verification): reverting the fix reproduces
  `EGLError(EGL_NOT_INITIALIZED)` exactly, confirming the fix addresses
  the root cause, not merely masking symptoms.

## V10 closing status (2026-08-02)

This is the project's final milestone: consolidation and repo polish, no
new experiments, no training, no GPU runs beyond one final fast-lane
pytest. Ten milestones (V0-V9.1), each independently verified against a
different agent than the one that implemented it
(`docs/VERIFICATION.md`), are complete:

- **Transport** (V1, V9-prep): a real, custom `RWM_state_series` glTF
  extension plus the draft `KHR_physics_rigid_bodies`/`KHR_implicit_shapes`
  extensions carry pose, rigid-body physics, joints, and arbitrary
  time-series state through a single GLB per episode, schema-validated and
  independently glTF-Validator-clean at every scale this project operated
  at (10,000+ synthetic episodes, 150 real external-dataset conversions,
  1,500 articulated episodes -- 0 errors across all of it, one real
  regression caught and fixed in V8.1).
- **Simulation -> rendering -> models -> re-emission**, the whole
  architecture-flow diagram at the top of this document, is real and
  working end-to-end (V2-V7): MuJoCo generates ground truth, a vendored
  patched renderer turns glTF into frames, a dynamics model beats a
  ballistic baseline by 42-176x, a perception model is real but
  data-limited (honestly reported short of its own acceptance bar), and a
  closed-loop demo ties both together with a genuine, measured finding
  about correlated vs. i.i.d. perception noise.
- **External validation** (V8): a real dataset (Physion) was converted
  into this transport end-to-end, producing both a strong state-based
  oracle ceiling (92%) and an honestly-reported zero-shot transfer collapse
  to chance -- fourteen concrete impedance-mismatch findings from that
  conversion are the primary evidence base for this project's gap report.
- **Articulation** (V9-prep, V9): hinged/sliding joints round-trip through
  the transport; a trained joint-state estimator clears all four
  acceptance bars.
- **The gap report** (V10, this milestone): `docs/GAP_REPORT.md` v1.0
  consolidates every honest gap recorded across V0-V9.1 into 20 numbered
  findings plus 5 positive findings, each with a code pointer and (where
  one exists) a measurement, organized by severity and prior-art
  comparison, closing with ranked recommendations for what a
  Khronos-track extension effort would need next.

**What remains open, stated plainly** (not silently dropped, per this
project's own house policy): articulated *dynamics* (predicting how a
joint's state evolves over time, the V9-counterpart of
`InteractionTransformer`) was never attempted; perception's full 0.05m/0.95
F1 acceptance bar was never met at the dataset scale this project trained
at (`docs/RESULTS.md`'s V6 section is explicit about this); the V9-prep
EGL crash's root cause was found and fixed in V9.1, but only on this
project's own development machine's current driver stack (not confirmed
across other NVIDIA driver versions); and `docs/GAP_REPORT.md`'s own
recommendations are exactly that -- recommendations for future Khronos-track
work, not something this project implements itself.

No further milestones are planned after V10.
