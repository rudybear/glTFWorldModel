# glTFWorldModel

A world-model proof of concept that uses **glTF 2.0 as the transport format
between training and inference**. MuJoCo generates ground-truth rigid-body
episodes; those episodes are serialized as glTF/GLB (standard pose animation,
plus physics and time-series state carried in a mix of draft Khronos
extensions and a custom vendor extension); a renderer turns glTF scenes into
frames; a perception model (frames -> scene state) and a dynamics model
(state t -> t+1) are trained on that data; and inference emits real glTF at
every hop, closing the loop back through the renderer.

Alongside the PoC, this repo tracks a gap analysis of glTF as a transport for
*dynamic* world state (rigid-body physics, time-series state, robotics-style
semantics) — the things core glTF 2.0 doesn't express and where draft or
custom extensions are needed to fill in.

## Stack

| Piece | Role |
|---|---|
| [pygltflib](https://gitlab.com/dodgyville/pygltflib) | glTF/GLB read + write |
| [trimesh](https://trimesh.org/) | mesh generation only |
| [MuJoCo](https://mujoco.org/) | physics simulation, episode generation |
| vendored [pyrender](https://github.com/mmatl/pyrender) (V2) | headless rendering of glTF scenes |
| `KHR_physics_rigid_bodies`, `KHR_implicit_shapes` (draft) | rigid-body + collider semantics on top of glTF |
| `RWM_state_series` (custom) | time-series world state carried alongside pose animation |

## Status

V2 — headless renderer (rgb + segmentation + depth via vendored, patched
pyrender; MuJoCo cross-render oracle; `render`/`crosscheck` CLI) on top of
the V1 glTF transport codec (pose animation + KHR physics +
`RWM_state_series`) — see [docs/VERIFICATION.md](docs/VERIFICATION.md).

## Setup

```bash
uv sync --all-extras
uv run pytest -m "not gpu"   # CI-equivalent: no GPU/EGL required
uv run pytest                # full suite, needs a GPU + working EGL offscreen context
```

## License

MIT.
