# DESIGN

Status: 2026-07-27, milestone V2.

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
- **V5** — perception model: frames -> scene state.
- **V6** — dynamics model: state[t] -> state[t+1].
- **V7** — inference loop: model output -> glTF -> renderer, closed loop.
- **V8** — external eval anchor: Physion replication (the primary
  *external* correctness anchor for this project's eval numbers, per V4's
  own metric-cross-validation note).
- **V9** — gap report + RWM extension write-up; PoC evaluation wrap-up.

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
