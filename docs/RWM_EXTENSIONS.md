# RWM extensions reference

The custom, gltfworld-authored parts of the transport: `RWM_state_series`
(root glTF extension) and `extras.rwm` (per-node/per-scene bookkeeping glTF
and the draft KHR physics extensions have no standard home for). See
`gltfworld.ext.rwm` for the implementation and `DESIGN.md` for the prose
architecture writeup; this document is the channel-by-channel/field-by-field
reference, kept current as new channels/fields are added (V9-prep added
`joint_position`/`semantics`/`articulations`, documented below alongside the
pre-existing fields).

## `RWM_state_series` (root extension)

```json
{
  "version": "0.1",
  "timesAccessor": <accessor index, shared with the pose animation>,
  "channels": [ { "target": ..., "kind": "...", "accessor": <index>, "component": <int, optional> }, ... ]
}
```

Each channel's `accessor` always has `count == len(times)`; its glTF
accessor type (SCALAR/VEC2/VEC3/VEC4) matches how many feature dims it
carries. A `target` is one of three shapes:

| `target` | meaning | used by |
| --- | --- | --- |
| `{"node": i}` | per-object channel, node index `i` | `linear_velocity`, `angular_velocity`, `pose_variance` |
| `{"joint": j}` | per-joint channel, index `j` into `KHR_physics_rigid_bodies.physicsJoints[]` (V9-prep) | `joint_position` |
| `"world"` | per-episode (not per-object/joint) channel | `action` |

| `kind` | width | units | notes |
| --- | --- | --- | --- |
| `linear_velocity` | 3 | m/s | world frame |
| `angular_velocity` | 3 | rad/s | world frame |
| `pose_variance` | 7 | position variance (3) + quaternion variance (4) | split into 2 chunks (`component` 0, 1) since 7 > 4 |
| `action` | arbitrary (A) | task-defined | split into `ceil(A/4)` chunks when A > 4 |
| `joint_position` | 1 | radians (revolute joint) or meters (prismatic joint) | **V9-prep**; never chunked (always width 1) |

Channels whose natural feature width exceeds 4 (the widest glTF accessor
type, VEC4) are split into multiple channels of the same `kind`, each
holding up to 4 contiguous feature dims, tagged with a 0-based `component`
chunk index so decode can concatenate them back in order (see
`gltfworld.ext.rwm._chunks`).

### `joint_position` (V9-prep)

One channel per articulated joint (`SceneState.articulations[j]`, same
ordering as `KHR_physics_rigid_bodies.physicsJoints[]`), each a plain SCALAR
accessor: the joint's own generalized position at every recorded frame --
the hinge angle (radians) for a `"revolute"` `ArticulatedSpec.joint_type`,
or the slide displacement (meters) for `"prismatic"`. Reassembled into
`StateSeries.joint_pos` (`(T, J)` float32) by
`gltfworld.ext.rwm.decode_channels`.

Why a dedicated `{"joint": j}` target rather than reusing `{"node": i}`
(e.g. the moving part's own node): a joint is not itself a node in the
pinned `KHR_physics_rigid_bodies` extension (see
`gltfworld.ext.khr_physics`'s module docstring, "V9-prep: joints") -- it's
an entry in the root `physicsJoints[]` array, referenced *from* a node's
`joint` property (`connectedNode` + `joint` index). Targeting the joint
index directly keeps `joint_position` channels meaningful even though the
actual KHR encoding routes the reference through a dedicated, geometry-less
"joint pivot" child node (see `gltfworld.scene.convert`), not the part's own
visible node.

## `extras.rwm` (per-node and per-scene)

### Per-object node (`node.extras.rwm`)

```json
{
  "object_id": <int>, "category": "<str>", "parts": {...},
  "schema_version": "0.1", "mass": <float>, "is_static": <bool>,
  "semantics": {"labels": ["door"], "affordances": ["openable"]}
}
```

`semantics` (V9-prep) is present **only** for objects that are part of an
articulated assembly (`base`/`part`/`handle` roles in some
`ArticulatedSpec`) -- omitted entirely (not even a `null`) for every
ordinary object, so a non-articulated episode's `extras.rwm` is byte-for-
byte unchanged from pre-V9-prep. See "Semantics taxonomy v0" below for the
label/affordance vocabulary.

### Per-scene (`scenes[i].extras.rwm`)

```json
{
  "seed": <int>, "scene_version": "<str>", "dt": <float>,
  "gravity": [gx, gy, gz],
  "articulations": [ {...ArticulatedSpec fields...}, ... ]
}
```

`articulations` (V9-prep) is present only when `SceneState.articulations`
is non-empty; each entry is the raw-dict encoding of one `ArticulatedSpec`
(`gltfworld.scene.convert.articulation_to_extras`/`articulation_from_extras`):
`joint_index`, `base_object_id`, `part_object_id`, `joint_type`
(`"revolute"`/`"prismatic"`), `axis` (0/1/2), `min`/`max` (radians or
meters), `anchor` (`[x, y, z]`, world position at t=0), `part_labels`,
`affordances`, `handle_object_id` (nullable), `handle_labels`,
`handle_affordances`, `base_labels`.

**`extras.rwm` is gltfworld's own decode source of truth for
`SceneState.articulations`** -- the same pattern already established for
`mass`/`is_static`/`gravity` (see DESIGN.md's "Documented deviations"): the
encoded `KHR_physics_rigid_bodies.physicsJoints[]` + node `joint` property
remain a faithful, independently schema-valid encoding of the draft
extension in their own right (what an external KHR-aware consumer would
read), but gltfworld's own `episode_from_gltf` reconstructs
`ArticulatedSpec` directly from `extras.rwm.articulations`, not by
re-deriving it from the KHR joint dicts.

## Semantics taxonomy v0 (V9-prep)

The label/affordance vocabulary `node.extras.rwm.semantics` draws from,
introduced for `gltfworld.datagen.articulated`'s two archetypes (a cabinet
with a hinged door, a chest/table with a sliding drawer):

| field | value | meaning |
| --- | --- | --- |
| `labels` (base) | `"cabinet"` | the static body the moving part is attached to |
| `labels` (part) | `"door"` / `"drawer"` | the moving, jointed body |
| `labels` (handle) | `"handle"` | cosmetic, rigidly-attached grasp point |
| `affordances` (part) | `"openable"` | the part can be moved by an external agent along its joint's free DOF |
| `affordances` (handle) | `"pullable"` | the handle is the intended point of contact for actuating the part |

This is a **v0** vocabulary (five categories, two affordances) scoped to
exactly the two articulated archetypes this milestone generates -- not a
general object/affordance ontology. Extending it (more furniture
categories, more affordances such as `"graspable"` for ordinary rigid
objects) is future work, out of this milestone's scope.
