# DESIGN

Status: 2026-07-28, milestone V6.

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
