# DESIGN

Status: 2026-07-27, milestone V0.

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
  patched for extension-aware rendering.
- **PyTorch** — standard ML framework for perception/dynamics models.

## Custom components

1. **`RWM_state_series` codec** (`gltfworld.ext`) — carries per-frame world
   state (beyond what pose animation/KHR physics express) inside a glTF
   extension. Validation: round-trip tests (encode -> decode -> compare) plus
   schema validation against a JSON Schema checked into `tests/goldens/`.
2. **KHR physics codec** (`gltfworld.ext`) — reads/writes
   `KHR_physics_rigid_bodies` + `KHR_implicit_shapes` draft extension data.
   Validation: fixtures cross-checked against the Khronos glTF-Validator
   (pinned release, see CI) plus targeted unit tests per shape/body type.
3. **Episode glue** (`gltfworld.datagen`/`gltfworld.scene`) — ties a MuJoCo
   rollout to a single GLB episode (animation + physics + state series in
   sync). Validation: end-to-end smoke test that a generated episode
   round-trips through load -> inspect -> re-serialize with no data loss.

## Milestones

- **V0** — project scaffold, CI, verification protocol.
- **V1** — glTF I/O layer: load/save/validate real GLB files; `validate`/`inspect` CLI work.
- **V2** — vendored renderer produces frames from a static glTF scene.
- **V3** — MuJoCo episode generation; `generate` CLI produces GLB episodes.
- **V4** — `RWM_state_series` extension: encode/decode + schema validation.
- **V5** — KHR physics extension codec (rigid bodies + implicit shapes).
- **V6** — episode glue: full MuJoCo -> GLB round trip, `crosscheck` CLI.
- **V7** — perception model: frames -> scene state.
- **V8** — dynamics model: state[t] -> state[t+1].
- **V9** — inference loop: model output -> glTF -> renderer, closed loop.
- **V10** — gap report + RWM extension write-up; PoC evaluation (`stats`, `eval`).
