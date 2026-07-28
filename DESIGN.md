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

## Milestones

- **V0** — project scaffold, CI, verification protocol.
- **V1** — glTF I/O layer: load/save/validate real GLB files; `validate`/`inspect` CLI work.
- **V2** (done) — vendored, patched pyrender renders episodes headlessly
  (rgb/seg/depth, 256x256); `render`/`crosscheck` CLI work; MuJoCo
  cross-render oracle; benchmark. See "Rendering (V2)" above.
- **V3** — MuJoCo episode generation; `generate` CLI produces GLB episodes.
- **V4** — `RWM_state_series` extension: encode/decode + schema validation.
- **V5** — KHR physics extension codec (rigid bodies + implicit shapes).
- **V6** — episode glue: full MuJoCo -> GLB round trip, `crosscheck` CLI.
- **V7** — perception model: frames -> scene state.
- **V8** — dynamics model: state[t] -> state[t+1].
- **V9** — inference loop: model output -> glTF -> renderer, closed loop.
- **V10** — gap report + RWM extension write-up; PoC evaluation (`stats`, `eval`).
