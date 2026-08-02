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

## Decoder conventions (normative)

These are load-bearing for anyone decoding `RWM_state_series`/`extras.rwm`
independently of `gltfworld.ext.rwm`/`gltfworld.scene.convert` -- surfaced by
an isolated, spec-only reimplementation (given only this document + the
vendored schemas + sample GLBs, no source code) that decoded bitwise-
identically but had to *guess* every rule below; one guess (object
inclusion, below) was initially wrong and produced silently-wrong shapes
before being corrected against the schema/GLBs. See
[docs/EXTERNAL_VALIDITY.md](EXTERNAL_VALIDITY.md) for the full experiment
writeup. Each rule was already true of every gltfworld-produced GLB; this
section makes the rules explicit instead of implicit.

1. **Object-inclusion rule.** The N (object) axis is *every* node carrying
   `extras.rwm.object_id` -- **including** `is_static` objects (e.g. the
   ground plate). Do not filter on `is_static`; a static object still gets
   a row in `poses`/`states` (it just never moves). The *only* nodes
   excluded from the N axis are ones with no `extras.rwm` at all (the
   camera node, light nodes) -- those are structurally different (no
   `object_id`, no physics collider), not merely "static objects to skip".
   Getting this wrong (e.g. excluding `is_static` objects, or trying to
   detect "the ground" by convention rather than by the presence/absence
   of `extras.rwm`) silently produces a scene with the wrong object count
   and, worse, an `RWM_state_series` channel-to-object mapping that's off
   by one for every object node that comes after the (wrongly) skipped one.
2. **Array ordering: ascending `object_id`.** Whatever you decode into a
   `(..., N, ...)` array (poses, per-node RWM channels, per-node
   `extras.rwm`), order the N axis by **ascending `extras.rwm.object_id`**,
   not by glTF node order. glTF does not guarantee node array order
   reflects any semantic ordering, and nothing in this format's schema
   promises object nodes appear in `object_id` order -- `object_id` itself
   is the only stable, order-independent key. (gltfworld's own encoder
   happens to emit nodes in ascending-`object_id` order today, so node
   order and `object_id` order coincide in every GLB this project has ever
   produced -- but a decoder that relies on that coincidence rather than
   sorting explicitly is relying on an implementation detail, not the
   spec.)
3. **Quaternion component order: `(x, y, z, w)`.** `RWM_state_series` and
   the pose animation both use core glTF's own quaternion convention
   (`rotation` accessors, and any RWM channel carrying quaternion-shaped
   data such as `pose_variance`'s 4 quaternion-variance components) --
   scalar component last, not first. This isn't an RWM-specific rule (it's
   just core glTF), but it's easy to get backwards when reading
   `pose_variance`/`joint_position`-adjacent data in isolation from a
   general glTF background, so it's stated here explicitly rather than
   left implicit.
4. **Pose animation samplers MUST use `STEP` interpolation.** Every
   `AnimationSampler` backing an object's `translation`/`rotation` channel
   uses `"interpolation": "STEP"`, never the glTF default (`"LINEAR"`).
   This is deliberate, not an oversight: gltfworld's animation is a
   faithful record of *sampled simulator states*, not a keyframed
   authored animation meant to be smoothly interpolated between samples --
   `LINEAR`-interpolating quaternions in particular would silently
   fabricate intermediate orientations the simulator never produced. A
   decoder that assumes `LINEAR` (or ignores `interpolation` and always
   linearly blends) will read plausible-looking but physically fabricated
   in-between frames whenever it samples off the exact recorded times.
5. **Chunked channels concatenate in ascending `component` order.**
   Channels whose natural feature width exceeds 4 (`pose_variance` at 7,
   `action` when its task-defined width A > 4) are split into multiple
   channels of the same `kind`/`target`, each tagged with a 0-based
   `component` field (see the chunking table above). Reassemble a
   channel's full feature vector by concatenating its chunks **in
   ascending `component` order** -- `component` is not guaranteed to
   appear in a channel's array position, i.e. don't assume the first
   `pose_variance` channel you encounter in `channels[]` is `component: 0`.
6. **Channel accessor `count` MUST equal `len(times)`.** Every channel's
   `accessor` (as referenced by `RWM_state_series.channels[i].accessor`)
   has `count` exactly equal to the length of the shared `timesAccessor`
   array -- one value per recorded frame, no channel-specific subsampling
   or padding. Decoders SHOULD validate this (`accessor.count ==
   len(times)`) before indexing into a channel's data, rather than
   silently truncating/overrunning if a malformed or hand-edited document
   violates it.

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
