# Provenance

The JSON Schema files under `implicit_shapes/` and `physics_rigid_bodies/`
in this directory are vendored, unmodified, from the draft Khronos
extensions repo:

- **Repo**: https://github.com/eoineoineoin/glTF_Physics
- **Pinned commit**: `9dc61cb3474ff9a51f58d3592f79d5c9e572056a`
- **Commit date**: 2026-01-20T15:38:51Z
- **Vendored on**: 2026-07-27

Source paths (relative to the repo root at the pinned commit):

- `extensions/2.0/Khronos/KHR_implicit_shapes/schema/*.json` -> `implicit_shapes/`
- `extensions/2.0/Khronos/KHR_physics_rigid_bodies/schema/*.json` -> `physics_rigid_bodies/`

gltfworld implements a subset of these two draft extensions; see
`src/gltfworld/ext/khr_physics.py` for exactly which properties are read and
written, and `DESIGN.md` ("Pinned specs") for the same commit hash recorded
alongside the transport-encoding writeup.

These files are not modified from upstream. Do not hand-edit them; re-fetch
from the pinned commit if they ever need to change.

## Core glTF schema dependencies

The extension schemas above `$ref` a handful of core glTF JSON Schema files
(`glTFProperty.schema.json`, `glTFChildOfRootProperty.schema.json`,
`glTFid.schema.json`, `extension.schema.json`, `extras.schema.json`), which
live in the main glTF spec repo, not the physics repo. Vendored into
`core/` from:

- **Repo**: https://github.com/KhronosGroup/glTF
- **Pinned commit**: `77b44be7bef26e01fb0b140e3d5bb1716421c5e9`
- **Commit date**: 2026-07-16T23:26:47Z
- **Vendored on**: 2026-07-27
- **Source path**: `specification/2.0/schema/*.json`

Also unmodified from upstream.
