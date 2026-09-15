# EXT_state_series

## Contributors

- rudybear, glTFWorldModel project (github.com/rudybear/glTFWorldModel) — proposal author, reference implementation, reference dataset

## Status

Draft proposal. Not ratified, not submitted to a Khronos working group ballot.
This document is written in Khronos extension-spec format to be
ballot-ready, but no 3D Formats WG vote has occurred. See
[Appendix B](#appendix-b-conformance-methodology) for the conformance
methodology already exercised against a prior, non-normative version of
this extension (`RWM_state_series`, this project's own vendor prefix).

## Dependencies

- Written against glTF 2.0.
- **Interacts with `KHR_animation_pointer`**: `EXT_state_series` channels
  target objects using the same JSON Pointer (RFC 6901) resolution rules
  `KHR_animation_pointer` uses for its `pointer` extension property (see
  "Targeting" below). `EXT_state_series` does not require
  `KHR_animation_pointer` to be present — a glTF document may combine
  ordinary `animation.channel` TRS targeting (`node`/`path`) with
  `EXT_state_series`, or may combine `KHR_animation_pointer`-targeted
  animation with `EXT_state_series`. When a document also declares
  `KHR_physics_rigid_bodies`, `EXT_state_series` channels commonly target
  its `physicsJoints[]` array (see Appendix A).
- No dependency on any other draft KHR physics extension; `EXT_state_series`
  is deliberately usable by a document with no physics extension at all
  (e.g. carrying only `linear_velocity` alongside plain node poses).

This extension MAY be used with `extensionsUsed`; because it carries state
that a naive viewer cannot render, it MUST NOT be listed under
`extensionsRequired` — a viewer with no knowledge of `EXT_state_series`
MUST still load and display the document's geometry and any core-glTF
animation correctly.

## Overview

Core glTF's `animation` object interpolates a node's `translation`,
`rotation`, `scale`, or (via `KHR_animation_pointer`) any other *mutable* pointer-
addressable property, over time. It has no concept of a *recorded, sampled
observation* — a value describing what *happened or was measured*, as
opposed to a value a player should *apply*. The distinction is not "pose
vs. non-pose": the draft `KHR_physics_rigid_bodies` publishes Object Model
pointer templates making `motion.linearVelocity`/`angularVelocity` and
joint drive targets mutable, so a *prescriptive* velocity track is already
expressible via `KHR_animation_pointer`. What has no home is the
*descriptive* counterpart — a measured velocity history, an applied
action, a joint's observed position, an estimate of pose uncertainty:
quantities a consumer must read but never execute (see "Relationship to
`KHR_animation_pointer`" below). `EXT_state_series` fills that gap: a root-
level glTF extension carrying named, time-sampled channels of arbitrary
(non-pose) per-object state, sharing a single time accessor with any
co-existing pose animation.

`EXT_state_series` generalizes beyond rigid-body simulation. Its channels
do not target "the node this body corresponds to" as a special case; they
target **any glTF object identifiable by a JSON Pointer** — a node, a
material, a camera, a light, an entry in another extension's array (e.g.
`KHR_physics_rigid_bodies.physicsJoints[]`), or the document's active
scene. This lets, for example, a light's intensity time series, a
material's animated-but-uncertain roughness estimate, or a physics joint's
generalized position all be expressed with the same channel shape,
without inventing a new per-object-type sub-extension for each.

This is a direct generalization of `RWM_state_series`, a project-specific
vendor extension exercised in production across 12,000+ generated episodes
and a real external-dataset conversion (see
[docs/GAP_REPORT.md](../../GAP_REPORT.md), recommendation #1, and
[Appendix A](#appendix-a-migration-from-rwm_state_series) below for the
exact mapping).

Requirement levels ("MUST", "MUST NOT", "SHOULD", "SHOULD NOT", "MAY") are
to be interpreted as described in RFC 2119.

## glTF Schema Updates

### Root extension: `EXT_state_series`

Declared once, at the document root, under `extensions.EXT_state_series`:

```json
{
  "extensionsUsed": ["EXT_state_series"],
  "extensions": {
    "EXT_state_series": {
      "version": "1.0.0-draft.1",
      "timesAccessor": 4,
      "channels": [ /* channel objects, see below */ ]
    }
  }
}
```

| Field | Type | Description | Required |
|---|---|---|---|
| `version` | `string` | `EXT_state_series` schema version. | Yes |
| `timesAccessor` | `integer` | Index of the shared time accessor (SCALAR, float32, seconds). The same accessor MAY also be used as an `AnimationSampler.input` for a co-existing pose animation. | Yes |
| `channels` | `channel[]` | The document's time-series channels. MAY be empty. | Yes |

### Channel object

```json
{
  "pointer": "/nodes/3",
  "kind": "angular_velocity",
  "accessor": 7,
  "frame": "world",
  "units": "rad/s"
}
```

| Field | Type | Description | Required |
|---|---|---|---|
| `pointer` | `string` | JSON Pointer (RFC 6901) identifying the glTF object this channel's quantity is *about*. Resolved with the same rules `KHR_animation_pointer` uses to resolve its `pointer` property (see "Targeting" below) — but unlike `KHR_animation_pointer`, the pointer here identifies an *object* (a node, a material, an array entry, the active scene), not the specific property being animated; `kind` (below) supplies that. | Yes |
| `kind` | `string` | The quantity this channel carries, from the registered vocabulary (below) or a vendor-specific `x-*` kind. | Yes |
| `accessor` | `integer` | Index of the accessor holding this channel's sampled data. `accessor.count` MUST equal `len(times)` (the shared `timesAccessor`'s count) — see "Decoder conventions", rule 6. Component type SCALAR/VEC2/VEC3/VEC4 per the `kind`'s natural width (chunked if wider, see `component`). | Yes |
| `component` | `integer` | 0-based chunk index, present only when a `kind`'s natural feature width exceeds 4 and has been split across multiple channels of the same `kind`/`pointer`. | No |
| `units` | `string` | UCUM-style unit string (e.g. `"m/s"`, `"rad/s"`, `"N"`). If omitted, the `kind`'s default unit (below) applies. | No |
| `frame` | `string` | One of `"world"`, `"parent"`, `"local"`. MUST be present for every vector-valued kind whose components are directionally meaningful (`linear_velocity`, `angular_velocity`, `applied_force`, `applied_torque`; see the vocabulary table). MUST NOT be present for scalar or frame-independent kinds (`action`, `joint_position`, `joint_velocity`). | Conditional, see vocabulary table |
| `sampling` | `string` | One of `"sampled"` (a recorded observation at each `times[i]`, no implied interpolation between samples — the default) or `"interpolable"` (values MAY be meaningfully interpolated between samples, e.g. a slowly-varying scalar parameter). Default: `"sampled"`. | No |
| `temporalCorrelation` | `object` | Optional metadata describing the temporal correlation structure of this channel's values across consecutive samples. Meaningful primarily for uncertainty-shaped kinds (`pose_variance`). See "Uncertainty and temporal correlation" below. | No |

### Targeting

Each channel's `pointer` is a JSON Pointer (RFC 6901), resolved against the
glTF document root exactly as `KHR_animation_pointer` resolves its own
`pointer` extension property: a `/`-delimited path of object keys and
array indices, walked from the document root. As in `KHR_animation_pointer`,
a pointer resolves to a **defined** location — present in the document
explicitly, or implied by a spec/extension default whose enclosing object
is present. Because `EXT_state_series` pointers commonly identify objects
that live inside another extension's own root-level array (most notably
`KHR_physics_rigid_bodies.physicsJoints[]`), pointers routinely traverse
into `extensions` objects — this is expected and permitted, exactly as
`KHR_animation_pointer` permits (and, for `extras`, explicitly
application-defines) pointers into non-core-schema locations.

Unlike `KHR_animation_pointer`, where the pointer identifies the exact
*property* being animated (e.g. `/nodes/0/rotation`), an `EXT_state_series`
channel's `pointer` identifies the **object the quantity is about** (e.g.
`/nodes/0`), and `kind` supplies the quantity — because most of this
extension's vocabulary (velocity, applied force, joint position,
uncertainty) has no corresponding settable glTF property for
`KHR_animation_pointer` to target in the first place. This is the
generalization this extension makes over its `RWM_state_series`
predecessor, which enumerated a closed `{node|joint|scene}` target shape
instead of an open pointer: any current or future glTF object type —
node, material, camera, light, or an array entry contributed by another
extension — can carry `EXT_state_series` channels without a schema change
to this extension.

Episode/scene-wide channels (a quantity that is not about any single
object, e.g. a discrete per-frame `action` an external agent took) target
the document's active scene, e.g. `"pointer": "/scenes/0"`.

### Relationship to `KHR_animation_pointer` (and why animation is not sufficient)

The first question a reviewer should ask — and one a post-publication
review did ask — is whether the ratified `KHR_animation_pointer` already
covers this ground. It permits an animation to target "any mutable
property in a glTF asset" (its gate: the property "MUST be mutable as
defined by the glTF 2.0 Asset Object Model"), and `KHR_physics_rigid_bodies`
publishes Object Model pointer templates for `motion/linearVelocity`,
`motion/angularVelocity`, and joint drive `positionTarget`/`velocityTarget`.
A time-varying velocity or drive-target track is therefore expressible in
glTF **today**, with no new extension.

`EXT_state_series` exists because that mechanism is the wrong *plane* for
recordings, for four reasons grounded in the specs' own text:

1. **Animation is prescriptive.** Players apply it; `KHR_physics_rigid_bodies`
   states that animations "should take priority over the physics
   simulation." A measured-velocity log encoded as a pointer animation is
   indistinguishable from kinematic velocity *control* — a physics-aware
   importer will drive the scene with the measurements. Observation data
   must be readable without ever being executed.
2. **No multi-track coexistence.** Core glTF: within one animation a target
   "MUST NOT be used more than once"; across animations, same-property
   behavior is explicitly left runtime-undefined. Ground truth, a model's
   prediction, and that prediction's variance for one object therefore
   cannot coexist as pointer tracks; as `EXT_state_series` channels they
   are simply parallel data.
3. **The observation vocabulary has no properties.** No mutable property
   exists for applied actions, pose uncertainty, *measured* joint position
   (drive targets are commands, not observations), or contact phenomena —
   a pointer needs something to point at, and defining those properties
   would itself require an extension.
4. **No observation metadata.** Animation samplers carry no units,
   reference frame, sampled-vs-interpolable semantics, or temporal-
   correlation structure — precisely the conventions whose absence this
   project measured as silent-corruption risks.

**Considered alternative (rejected):** define the missing quantities as
static extension properties and animate them via `KHR_animation_pointer`.
This costs the same extension surface as `EXT_state_series` while
inheriting problems 1, 2, and 4 unchanged.

**Normative guidance:** for quantities that *are* mutable, settable
properties — camera/material/light parameters, physics initial conditions,
drive commands — producers SHOULD use `KHR_animation_pointer` (or core
animation) and SHOULD NOT mirror them as `EXT_state_series` channels.
`EXT_state_series` is exclusively the *descriptive* plane: recorded,
sampled observations.

### Registered kind vocabulary

| `kind` | Width | Default units | `frame` | Notes |
|---|---|---|---|---|
| `linear_velocity` | 3 (VEC3) | `m/s` | **Required** | Measured/observed value — distinct from the *settable* `KHR_physics_rigid_bodies` `motion.linearVelocity` (an initial condition/command; see "Relationship to `KHR_animation_pointer`"). |
| `angular_velocity` | 3 (VEC3) | `rad/s` | **Required** | Measured/observed value (same distinction as `linear_velocity`). See "Implementation Notes" on the cost of an ambiguous frame for this kind specifically. |
| `applied_force` | 3 (VEC3) | `N` | **Required** | |
| `applied_torque` | 3 (VEC3) | `N·m` | **Required** | |
| `action` | Task-defined (A); chunked if A > 4 | Task-defined; producer MUST document units out-of-band (e.g. `extras`) | MUST NOT be present | `pointer` targets the active scene, not a per-object node, in the common case of a whole-episode action. |
| `joint_position` | 1 (SCALAR) | `rad` (revolute) or `m` (prismatic), per the targeted joint's type | MUST NOT be present | `pointer` targets a `KHR_physics_rigid_bodies.physicsJoints[]` entry, e.g. `/extensions/KHR_physics_rigid_bodies/physicsJoints/2`. |
| `joint_velocity` | 1 (SCALAR) | `rad/s` (revolute) or `m/s` (prismatic) | MUST NOT be present | Same targeting as `joint_position`. |
| `pose_variance` | 7 (3 position-variance + 4 quaternion-variance components); chunked (`component` 0, 1) since 7 > 4 | `m^2` (position variance components), dimensionless (quaternion-variance components) | Required, and MUST match the frame of the pose this channel estimates uncertainty for | Diagonal-only (no cross-time or cross-component correlation term in the accessor itself); see "Uncertainty and temporal correlation" for the accompanying `temporalCorrelation` metadata. |
| `x-*` | Vendor-defined | Vendor-defined; MUST be documented (e.g. in `extras` alongside the channel, or in accompanying producer documentation) | Vendor-defined | Custom kind. `kind` MUST match `^x-[A-Za-z0-9](?:[A-Za-z0-9_-]*[A-Za-z0-9])?$`. Consumers MUST ignore `x-*` channels whose kind they do not recognize rather than treat the document as invalid. |

Channels whose natural feature width exceeds 4 (the widest glTF accessor
type, VEC4) MUST be split into multiple channels of the same `kind` and
`pointer`, each holding up to 4 contiguous feature dims, each tagged with
a 0-based `component` chunk index (see "Decoder conventions", rule 5).

### Uncertainty and temporal correlation

A `pose_variance` channel MAY carry an optional `temporalCorrelation`
object alongside its per-frame diagonal variance data:

```json
{
  "pointer": "/nodes/3",
  "kind": "pose_variance",
  "accessor": 9,
  "component": 0,
  "frame": "world",
  "temporalCorrelation": { "model": "ar1", "coefficient": 0.71 }
}
```

| Field | Type | Description |
|---|---|---|
| `model` | `string` | Correlation model identifier. v0 of this extension defines exactly one: `"ar1"` (lag-1 autoregressive). |
| `coefficient` | `number` | For `"ar1"`: the lag-1 autocorrelation coefficient, in `[0, 1]`. |

**Why this field exists, and why it is not optional advice buried in
prose.** A per-frame variance channel with no temporal-correlation
metadata invites consumers to treat consecutive samples as independent
(i.i.d.) noise when reasoning about multi-frame quantities (e.g.
finite-differenced velocity, or a multi-step rollout's compounding error).
This project's own closed-loop measurement
([docs/GAP_REPORT.md](../../GAP_REPORT.md), finding G6) found this
assumption to be not merely imprecise but **quantitatively wrong by an
order of magnitude**: an i.i.d.-Gaussian noise model calibrated to the
*correct* per-frame error magnitude diverged 17x faster than the real,
measured closed loop at a 99-step horizon (27.6m vs. 1.62m median position
error), because real perception-model error is strongly frame-correlated
(measured lag-1 autocorrelation 0.55–0.82 across position/rotation) rather
than independent — finite-differenced velocity noise from correlated
per-frame error **partially cancels**, rather than compounding via the
`sqrt(2)` amplification an i.i.d. model assumes. `temporalCorrelation` is
this extension's answer: a `pose_variance` channel that omits it is making
no claim about temporal structure at all (and a consumer MUST NOT assume
i.i.d. in that case either — absence of the field is not evidence of
independence); a channel that includes it gives a consumer a
minimally-sufficient basis to model correlated rather than independent
rollout noise. This is `EXT_state_series`'s central point of departure
from `UsdPhysics` and every other uncertainty representation surveyed for
this proposal (see G6's "Prior art" note): none carry any temporal-
correlation metadata at all.

## Shared time accessor and invariants

- All channels within one `EXT_state_series` root extension object share
  exactly one `timesAccessor`. A document with multiple independent
  series (e.g. per-episode) uses multiple `EXT_state_series`-bearing
  documents, not multiple `timesAccessor` values within one.
- Every channel's `accessor.count` MUST equal `len(times)` (the shared
  `timesAccessor`'s `count`). See "Decoder conventions", rule 6.
- **STEP-equivalent semantics for co-existing pose animation.** When a
  document's node pose (`translation`/`rotation`/`scale`, whether
  targeted by ordinary `animation.channel.target.node`/`path` or by
  `KHR_animation_pointer`) is sampled at the same instants as an
  `EXT_state_series` channel (i.e. its `AnimationSampler.input` is the
  same accessor as, or an index-for-index-equal accessor to, the
  `EXT_state_series` `timesAccessor`), the corresponding
  `AnimationSampler.interpolation` MUST be `"STEP"`. `EXT_state_series`
  channels are recordings of sampled, discrete simulator/observation
  state, not authored keyframes meant to be smoothly interpolated; a pose
  animation carrying the same sampled states must not silently imply
  intermediate states the source never produced by defaulting to (or
  being read as) `"LINEAR"`.

## Non-goals

`EXT_state_series` deliberately does not attempt to standardize:

- **Streaming or append semantics.** Every channel accessor is a complete,
  fixed-length array over the document's full time range. A document
  representing a live or growing series is out of scope; producers of
  live state should emit successive complete documents/snapshots by
  convention.
- **Ragged, event-shaped channels** (e.g. discrete contact/collision
  events with a variable per-frame count). `EXT_state_series` channels are
  uniformly sampled, one value per `times[i]`; event data does not fit
  that shape without either padding (misrepresenting "no event" as a
  measured zero) or a variable-length encoding this extension does not
  define.
- **Video or frame-sequence-as-scene-content.** Rendered frame data is out
  of scope for a state-series extension.

Each of these was considered and rejected as *this extension's* scope, not
because the underlying need is unimportant: [docs/GAP_REPORT.md](../../GAP_REPORT.md)'s
G4 (discrete collision/contact events), G5 (video/frame-sequence content),
and the recommendations section's explicit "not recommended" note both
conclude that convention-level or sidecar-file solutions are adequate for
these needs at present, and that widening this extension's own scope (or
glTF core's type system, per G3) to absorb them is not justified by the
measured cost of the status quo. A future, separate extension MAY address
ragged event channels; this one does not.

## Decoder conventions (normative)

These conventions are load-bearing for any independent decoder — ported
and generalized from `RWM_state_series`'s own normative conventions
(`docs/RWM_EXTENSIONS.md`), which were themselves surfaced by an isolated,
spec-only reimplementation exercise (see
[Appendix B](#appendix-b-conformance-methodology) and
[docs/EXTERNAL_VALIDITY.md](../../EXTERNAL_VALIDITY.md)).

1. **Object-inclusion rule.** Every glTF object that is the resolved target
   of at least one `EXT_state_series` channel's `pointer` is in scope for
   decode — including objects whose other properties or extensions (e.g.
   `KHR_physics_rigid_bodies`'s `motion.type: "static"`) might otherwise
   suggest they are "inert" or non-dynamic. Decoders MUST NOT filter the
   set of decoded objects using any property outside `EXT_state_series`
   itself (e.g. excluding static bodies, or objects lacking a mesh) — scope
   is determined solely by which objects have channels pointed at them.
2. **Ordering by target pointer's canonical order.** When reconstructing an
   array over multiple like-kinded objects (e.g. all nodes'
   `linear_velocity`), decoders MUST order that array by the resolved
   pointer's own canonical array index (e.g. ascending glTF node index for
   `/nodes/i` targets, ascending material index for `/materials/i`
   targets) — not by `channels[]` array position, which carries no
   ordering guarantee. This is a direct improvement over
   `RWM_state_series`'s equivalent rule (ascending `extras.rwm.object_id`,
   an out-of-band bookkeeping key): because `EXT_state_series` targets are
   JSON Pointers into glTF's own arrays, the pointer's own index *is* the
   canonical, order-independent key, and no separate ID scheme is needed.
3. **Quaternion component order: `(x, y, z, w)`.** Any channel carrying
   quaternion-shaped data (e.g. `pose_variance`'s 4 quaternion-variance
   components) uses core glTF's own quaternion convention — scalar
   component last. This is not an `EXT_state_series`-specific rule, but is
   stated here explicitly because it is easy to get backwards when reading
   variance-shaped data in isolation.
4. **Pose-animation samplers co-sampled with `EXT_state_series` MUST use
   `STEP` interpolation.** See "Shared time accessor and invariants" above.
5. **Chunked channels concatenate in ascending `component` order.**
   Channels split across multiple accessors (natural width > 4) MUST be
   reassembled by concatenating their chunks in ascending `component`
   order — `component` is not guaranteed to correlate with a channel's
   position in the `channels[]` array.
6. **Channel accessor `count` MUST equal `len(times)`.** Decoders SHOULD
   validate `accessor.count == len(times)` before indexing into a
   channel's data, rather than silently truncating or overrunning if a
   malformed or hand-edited document violates the invariant.

## JSON structure with examples

### Example: velocity + action + pose uncertainty for a two-object scene

```json
{
  "extensionsUsed": ["EXT_state_series", "KHR_physics_rigid_bodies"],
  "extensions": {
    "EXT_state_series": {
      "version": "1.0.0-draft.1",
      "timesAccessor": 10,
      "channels": [
        { "pointer": "/nodes/0", "kind": "linear_velocity", "accessor": 11, "frame": "world" },
        { "pointer": "/nodes/0", "kind": "angular_velocity", "accessor": 12, "frame": "world" },
        { "pointer": "/nodes/1", "kind": "linear_velocity", "accessor": 13, "frame": "world" },
        { "pointer": "/nodes/1", "kind": "angular_velocity", "accessor": 14, "frame": "world" },
        {
          "pointer": "/nodes/0", "kind": "pose_variance", "accessor": 15,
          "component": 0, "frame": "world"
        },
        {
          "pointer": "/nodes/0", "kind": "pose_variance", "accessor": 16,
          "component": 1, "frame": "world",
          "temporalCorrelation": { "model": "ar1", "coefficient": 0.71 }
        },
        { "pointer": "/scenes/0", "kind": "action", "accessor": 17 }
      ]
    }
  }
}
```

### Example: an articulated joint's generalized position

```json
{
  "pointer": "/extensions/KHR_physics_rigid_bodies/physicsJoints/2",
  "kind": "joint_position",
  "accessor": 22
}
```

### Example: a vendor-specific channel

```json
{
  "pointer": "/nodes/4",
  "kind": "x-grip_force",
  "accessor": 23,
  "units": "N",
  "frame": "local"
}
```

## Implementation Notes

- **Frame ambiguity is a real, measured cost, not a hypothetical.** This
  project's own MuJoCo-to-glTF pipeline hit a concrete hazard converting
  MuJoCo's body-frame angular velocity output: MuJoCo reports a body's
  angular velocity components in that body's own local frame by
  convention, not world frame, and a converter that copies the raw values
  into a channel documented (or assumed) to be world-frame silently
  produces plausible-looking but wrong data for any rotated body — the
  error is not a crash, it is a silently wrong number. `frame` being a
  required, explicit field for every directional vector kind (rather than
  a fixed, spec-wide convention or an omittable default) is a direct
  response to this measured hazard, not a hypothetical generalization.
- **`pointer` resolution reuses `KHR_animation_pointer` tooling.** An
  implementation that has already implemented `KHR_animation_pointer`'s
  JSON Pointer resolver (property-definedness rules, traversal through
  `extensions`/`extras`) can reuse that same resolver for
  `EXT_state_series` channel `pointer`s with no additional pointer-syntax
  work; only the interpretation (identify an *object*, not a *property*)
  differs.
- **`joint_position`/`joint_velocity` targeting is deliberately indirect.**
  A physics joint is not itself a glTF node in
  `KHR_physics_rigid_bodies` (a draft extension pinned by this project at
  commit `9dc61cb3474ff9a51f58d3592f79d5c9e572056a`) — it is an entry in
  the root `physicsJoints[]` array, referenced *from* a node's `joint`
  property. Pointing directly at the `physicsJoints[j]` entry (rather than,
  say, the node that owns the joint) keeps the channel meaningful
  regardless of whether an implementation additionally routes the
  reference through a geometry-less "joint pivot" child node, as this
  project's own encoder does (see `gltfworld.scene.convert`,
  `docs/RWM_EXTENSIONS.md`'s `joint_position` section).
- **Reference implementation**: this project's `gltfworld.ext.rwm` module
  implements the predecessor `RWM_state_series` vendor extension at
  production scale (12,000+ episodes, 0 schema-validation failures); a
  migration to `EXT_state_series` proper is a mechanical target-shape
  change (Appendix A), not a redesign.

## Conformance

A document is conformant to `EXT_state_series` if and only if:

1. It is a valid glTF 2.0 document per the core glTF 2.0 schema.
2. Its root `extensions.EXT_state_series` object validates against
   [`schema/glTF.EXT_state_series.schema.json`](schema/glTF.EXT_state_series.schema.json)
   and each of its channels validates against
   [`schema/glTF.EXT_state_series.channel.schema.json`](schema/glTF.EXT_state_series.channel.schema.json).
3. Every channel's `pointer` resolves to a defined glTF object (per the
   property-definedness rule inherited from `KHR_animation_pointer`).
4. Every channel's `accessor.count` equals `len(times)` (rule 6).
5. All applicable rules in "Decoder conventions (normative)" hold.

A decoder is conformant if, given a conformant document, it reproduces the
sampled-state arrays a reference decoder would produce, bit-for-bit, by
following only this specification, its referenced schemas, and no
undocumented convention. See [Appendix B](#appendix-b-conformance-methodology)
for the methodology this project uses to test exactly that claim.

---

## Appendix A: Migration from `RWM_state_series`

`RWM_state_series` (this project's pre-standardization vendor extension,
`docs/RWM_EXTENSIONS.md`) maps onto `EXT_state_series` as follows. The
underlying accessor data (times, velocities, variances, actions, joint
positions) is **byte-identical**; only the targeting shape and a small set
of field names change.

| `RWM_state_series` | `EXT_state_series` | Notes |
|---|---|---|
| `target: {"node": i}` | `pointer: "/nodes/i"` | Direct translation. |
| `target: {"joint": j}` | `pointer: "/extensions/KHR_physics_rigid_bodies/physicsJoints/j"` | `RWM_state_series` referenced the joint index directly (uninterpreted); `EXT_state_series` uses the equivalent JSON Pointer into the same array. |
| `target: "world"` | `pointer: "/scenes/i"` (the episode's scene) | `RWM_state_series`'s `"world"` sentinel becomes an explicit pointer to the active scene. |
| `kind` (all five v0.1 values: `linear_velocity`, `angular_velocity`, `action`, `pose_variance`, `joint_position`) | Same `kind` names, same widths, same units, same chunking | Unchanged. `EXT_state_series` additionally registers `applied_force`, `applied_torque`, `joint_velocity`, and vendor `x-*` kinds, none of which existed in `RWM_state_series` v0.1. |
| `accessor`, `component` | `accessor`, `component` | Unchanged: same field names, same semantics, same chunking-order rule. |
| (no `frame` field; velocity kinds were world-frame by fixed convention) | `frame` (required for vector kinds) | `RWM_state_series` fixed `linear_velocity`/`angular_velocity` to world frame implicitly; `EXT_state_series` makes this explicit and per-channel (see Implementation Notes on the MuJoCo body-frame hazard this generalizes away from). Migrated `RWM_state_series` documents' velocity channels are all `"frame": "world"`. |
| (no `units` field; fixed per `kind` by convention) | `units` (optional, defaults per `kind`) | Migrated documents may omit `units` and rely on the same defaults `RWM_state_series` used implicitly. |
| (no uncertainty temporal-correlation metadata) | `temporalCorrelation` (optional, `pose_variance` only) | New in `EXT_state_series`; absent in any migrated `RWM_state_series` document until a producer adds it. |
| `version: "0.1"` | `version: "1.0.0-draft.1"` | Versioning scheme changes; not otherwise semantically comparable. |

What is **unchanged**: the shared-`timesAccessor` design; the
`count == len(times)` invariant; ascending-`component` chunk-reassembly
order; the `(x, y, z, w)` quaternion convention; the requirement that
co-sampled pose animation use `STEP` interpolation; the reference decoder's
underlying array shapes and dtypes.

## Appendix B: Conformance methodology

`EXT_state_series`'s conformance methodology is the same blind,
spec-only reimplementation protocol this project already ran against
`RWM_state_series` and reports in full in
[docs/EXTERNAL_VALIDITY.md](../../EXTERNAL_VALIDITY.md) ("Experiment A:
spec-only reimplementation"), adopted here as this extension's own
normative conformance test:

1. **Isolate an implementer from the reference implementation.** Give them
   exactly: this specification, its JSON Schemas
   (`schema/glTF.EXT_state_series.schema.json`,
   `schema/glTF.EXT_state_series.channel.schema.json`), and a sample asset
   corpus — this project's generated episode corpus (12,000+ GLBs across
   `dynamics-v1`, `perception-v1`, `articulated-v1`) is the corpus this
   project uses for its own conformance runs. No access to
   `gltfworld.ext.rwm`/`gltfworld.scene.convert` or any other reference
   decoder source, no access to this project's test suite.
2. **Task**: decode every sample asset's full state series (poses,
   velocities, actions, joint state, uncertainty) using only the inputs in
   step 1.
3. **Pass criterion**: the independent decoder's output matches the
   reference decoder's output **bitwise** (uint32-view comparison, not a
   tolerance-based "close enough") across the full sample corpus.
4. **On failure or ambiguity**: every place the independent implementer had
   to *guess* rather than read an explicit rule is a specification defect,
   regardless of whether the guess happened to be correct. `RWM_state_series`'s
   own run of this protocol found six such defects (all now folded into
   this document's "Decoder conventions" section in generalized form); one
   guess was initially wrong and produced silently-incorrect output before
   correction (the object-inclusion rule) — the single highest-value
   finding the protocol produced, precisely because it failed silently
   rather than loudly.

This project regards this protocol — not schema validation alone — as the
real conformance bar for `EXT_state_series`: a document that merely
validates against the JSON Schema can still be under-specified in ways
that only a from-scratch, isolated reimplementation surfaces. A future
KHR-track ratification of this extension should budget for running this
same protocol against the final spec text before ratification, not only
against implementations that already share context with the spec's
authors.
