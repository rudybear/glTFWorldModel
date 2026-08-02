# glTF as a world-model transport: gap report v1.0

Status: 2026-08-02, milestone V10 (final). This is the project's flagship
deliverable: a consolidated, evidence-backed account of where glTF 2.0 (core
+ draft Khronos physics extensions + this project's own custom extension)
succeeds and fails as the transport format between simulation, rendering,
and learned perception/dynamics models for a *dynamic* world model — as
opposed to glTF's native use case, a static (or purely kinematic-animation)
3D scene.

Every finding below is drawn from a real, working pipeline
(`MuJoCo -> GLB -> renderer -> perception/dynamics models -> GLB`, all nine
prior milestones, V0-V9.1) and a real external dataset conversion (Physion),
not from reading the spec and speculating. Every gap has a concrete code
pointer to the workaround this project actually shipped, plus a measurement
or JSON exhibit where one exists.

## Executive summary

**What we built.** A complete, working world-model pipeline that uses glTF
2.0 GLB files as the transport at *every* hop: MuJoCo-simulated rigid-body
episodes are serialized as GLB (pose animation + draft
`KHR_physics_rigid_bodies`/`KHR_implicit_shapes` + a custom
`RWM_state_series` extension for velocity/action/uncertainty/joint-state
time series); a vendored, patched renderer turns those GLBs into RGB/depth/
segmentation frames; a perception model (frame -> scene state) and a
dynamics model (state[t] -> state[t+1]) train on that data; and inference
re-emits real, independently loadable, validator-clean GLB at every hop,
closing the loop back through the renderer. Nine substantive milestones
(V1-V9) shipped real numbers: dynamics beats a ballistic baseline by
42-176x at long horizons; perception is real but data-limited (existence F1
0.87, matched position error 0.18m, both short of the 0.05m/0.95 F1 target);
closed-loop rollout is 34x better than ballistic despite imperfect
perception, with a genuine, measured finding about correlated (not i.i.d.)
detector noise; a Physion external anchor shows a strong 92% state-based
oracle ceiling but a zero-shot dynamics-transfer collapse to chance —
reported honestly rather than hidden; articulated objects (hinged doors,
sliding drawers) round-trip through the transport with a trained
joint-state estimator clearing all four acceptance bars.

**What glTF could do.** Core glTF's accessor/bufferView/animation machinery
— designed for arbitrary typed, indexed, time-sampled numeric arrays — turns
out to be genuinely reusable as a general time-series transport, not just a
mesh-and-camera format: this project's custom `RWM_state_series` extension
carries velocity, action, pose-uncertainty, and per-joint-state channels
using nothing but ordinary SCALAR/VEC2/VEC3/VEC4 accessors and the same
shared time accessor pose animation already uses. `extensionsUsed`/
`extensionsRequired` let this ride alongside standard content: an ordinary,
off-the-shelf glTF viewer (no knowledge of physics or state-series
extensions at all) can still load and play back every episode this project
produced. Every GLB this project emits — 150 Physion conversions, 1,500
articulated episodes, every training/eval artifact — passes the independent,
pinned Khronos glTF-Validator with zero errors.

**What glTF could not do**, without either draft extensions or fully custom
ones: express rigid-body physics at all (mass/friction/restitution/
colliders — `KHR_physics_rigid_bodies`/`KHR_implicit_shapes`, both still
DRAFT), express *any* non-pose per-frame state (velocity, applied action,
uncertainty, joint angle — `RWM_state_series`, wholly custom, no Khronos
equivalent exists even in draft), express joints with viscous damping or
armature, express a bounded-duration push (only a persistent spring-to-
target `drive`), express a collider offset from its owning node's origin
(forcing an extra-node workaround for every articulated joint this project
generated), or express robotics-style semantics/affordances (a from-scratch
`extras.rwm.semantics` vocabulary). Fourteen further concrete impedance
mismatches surfaced converting a real external dataset (Physion) into this
transport. And — the most load-bearing uncertainty finding in this
report — even where a variance-like channel *is* representable
(`pose_variance`), the natural per-frame-independent noise model it invites
is empirically the *wrong* model for real perception-model error, which is
strongly frame-correlated (lag-1 autocorrelation 0.55-0.82, V7).

**Methodology in one sentence**: every gap below is backed by a working
system that hit it, a file that works around it, and (where applicable) a
number that shows the size of the gap — not a reading of the spec text
alone.

## Methodology

- **Evidence-driven, not spec-driven.** Every numbered finding traces to
  code in this repository (`src/gltfworld/...`) that had to work around the
  gap, plus (where one exists) a measurement, test, or JSON artifact
  produced by a real run — not a hypothetical.
- **Sources consolidated**: [DESIGN.md](../DESIGN.md) (all milestone
  sections V0-V9.1), [docs/PHYSION.md](PHYSION.md) (14 numbered conversion
  findings + format reconnaissance), [docs/RESULTS.md](RESULTS.md) (V5-V9
  measured results), [docs/RWM_EXTENSIONS.md](RWM_EXTENSIONS.md) (the custom
  extension's own field-by-field reference), [docs/VERIFICATION.md](VERIFICATION.md)
  (every milestone's independent-verifier checkpoints), and
  [docs/PRETRAINING_GATE.md](PRETRAINING_GATE.md).
- **No pre-existing gap skeleton.** DESIGN.md's milestones each recorded
  their own "honest gaps" as they were found (V9-prep: 4 gaps; V9: 5 gaps;
  V7/V8: the correlated-noise and Physion findings respectively); no
  earlier numbered G1-G13 catalog exists anywhere in this repo's history —
  this document is the first place these are consolidated and numbered.
  The one forward-reference that does exist in-repo
  (`docs/RESULTS.md`'s V7 section: "relevant to the gap-report's G6 on
  uncertainty models") is honored here: **G6 is the uncertainty-model
  finding**, as that reference anticipated.
- **Severity scale**: **Blocking** (this project could not have produced a
  correct pipeline without a custom workaround), **Significant** (a real,
  measured fidelity/interop cost, worked around but imperfectly), **Minor**
  (a real gap with negligible practical impact at this project's scale).
- **Prior art comparisons**: `UsdPhysics 1.0` (Pixar/OpenUSD's mature,
  shipped physics schema — the closest existing "physics on a scene graph"
  effort to what a finished KHR physics spec would need to be), `URDF`/
  `MJCF` (the two dominant robotics/sim description formats, both with
  first-class joints and no rendering-format ambition), and the two draft
  KHR physics extensions this project itself implements against
  (`KHR_physics_rigid_bodies`, `KHR_implicit_shapes`, pinned commit
  `9dc61cb3474ff9a51f58d3592f79d5c9e572056a`, see DESIGN.md's "Pinned
  specs").

---

## Part A — Core glTF: no concept of dynamic world state

### G1. No non-pose time-varying state at all in core glTF (foundational gap)

**What glTF lacks**: core glTF's `animation` object interpolates a node's
translation/rotation/scale/weights over time — nothing else. There is no
concept of "this scalar/vector also varies per frame" for anything that
isn't a TRS component or morph-target weight. A physics engine's per-frame
state (linear/angular velocity, an applied external action, a
joint's generalized position, a per-object uncertainty estimate) has
*no home anywhere in the base spec*, draft or shipped.

**Why it matters for world models/robotics**: a world model's entire job is
predicting/consuming state that isn't merely pose — velocity for physical
plausibility, action for causal attribution, uncertainty for downstream
planning. Without a channel for these, a transport can serialize *what a
scene looked like* but not *what is driving how it changes*.

**Severity**: Blocking.

**Workaround implemented here**: `RWM_state_series` (`gltfworld.ext.rwm`,
`src/gltfworld/ext/rwm.py`), a wholly custom root-level glTF extension that
reuses the *existing* accessor/bufferView machinery: one shared time
accessor (the same one pose animation's samplers use), and one channel per
(target, kind) pair, each backed by an ordinary SCALAR/VEC2/VEC3/VEC4
accessor. See [docs/RWM_EXTENSIONS.md](RWM_EXTENSIONS.md) for the full
channel reference (`linear_velocity`, `angular_velocity`, `pose_variance`,
`action`, `joint_position`).

**What a proper extension would need**: a Khronos-ratified (not
project-custom) "auxiliary per-node/per-joint/per-scene time series"
extension — effectively standardizing what `RWM_state_series` improvises:
an arbitrary named channel, tied to a shared time accessor, with a declared
physical unit/semantic tag, targetable at a node, a physics joint, or the
whole scene.

**Prior art**: neither URDF nor MJCF face this problem the same way — both
are *simulator input* formats with no native per-frame recording concept of
their own (simulators emit separate log formats, e.g. MuJoCo's own binary
state dumps). USD's `UsdGeom.PointBased` and `UsdPhysics`'s velocity
attributes are per-prim *authored* (typically constant or keyframed by an
external DCC/animation system) rather than a general "arbitrary simulator
output channel" concept either — this is a genuinely unaddressed space, not
just glTF lagging behind an existing solution.

### G2. Khronos's own physics extensions cover initial conditions, not time series

**What glTF lacks**: `KHR_physics_rigid_bodies`/`KHR_implicit_shapes` (both
still DRAFT) describe a rigid body's *simulation-time* properties — mass,
friction, restitution, collider shape, and (via `physicsJoints`) a joint's
limits/drive — but every one of those fields is a single, static value
attached to a node. There is no provision anywhere in the pinned spec for
"this body's velocity/joint-position *at frame t*" — that's exactly what
`RWM_state_series`'s `linear_velocity`/`angular_velocity`/`joint_position`
channels supply. This is a **cross-cutting finding across the whole
project**: even once the currently-draft physics extensions are ratified
and shipped, they still would not obsolete `RWM_state_series` — the two
solve genuinely disjoint problems (initial/static physics parameters vs.
recorded per-frame trajectory).

**Why it matters**: a reader might reasonably assume "once glTF has real
physics extensions, world-model transport is solved" — this project's own
experience says otherwise. Initial-condition physics parameters and
recorded time-series state are both necessary and neither substitutes for
the other.

**Severity**: Blocking (this is *why* `RWM_state_series` had to exist
alongside, not instead of, the KHR physics codec).

**Workaround implemented here**: `gltfworld.ext.khr_physics`
(`src/gltfworld/ext/khr_physics.py`) encodes the static physics parameters;
`gltfworld.ext.rwm` (`src/gltfworld/ext/rwm.py`) encodes everything
time-varying; `gltfworld.scene.convert.episode_to_gltf` (`src/gltfworld/
scene/convert.py`) is the one place that ties both to a single shared time
accessor so they can never drift out of sync (see DESIGN.md's "Transport
encoding" section, `tests/test_consistency.py`).

**What a proper extension would need**: an explicit Khronos position on
this distinction — a "recorded physics state" companion extension to
`KHR_physics_rigid_bodies`, scoped from the start as *reusing* the
initial-condition extension's node/joint indices as its own channel targets
(exactly the `{"node": i}`/`{"joint": j}` target scheme
`RWM_state_series` already uses, see docs/RWM_EXTENSIONS.md).

**Prior art**: `UsdPhysics 1.0` has the identical split in spirit — a
`UsdPhysicsRigidBodyAPI` describes simulation-ready static state, and
*recorded* per-frame simulation output is a separate concern USD addresses
via ordinary time-sampled attributes on the same prims (USD's time-sampling
mechanism is more general-purpose than glTF's animation-only one, so USD
does not need an equivalent of `RWM_state_series` to bolt on afterward —
this is arguably USD's biggest structural advantage over glTF for this use
case, see G23).

### G3. Accessor's 4-component width cap forces manual channel-splitting

**What glTF lacks**: the widest glTF accessor type is VEC4. Any per-frame
feature vector wider than 4 (this project's `pose_variance`, 7-wide:
3 position + 4 quaternion variance; and `action`, arbitrary task-defined
width) has no single accessor type that fits it.

**Why it matters**: any future standardized time-series-state extension
inherits this same ceiling — it's a structural property of glTF's type
system, not a defect specific to this project's own channels.

**Severity**: Minor (worked around cleanly, at the cost of some encode/
decode bookkeeping).

**Workaround implemented here**: `RWM_state_series` splits any channel
wider than 4 components into `ceil(width/4)` channels of the same `kind`,
each carrying up to 4 contiguous feature dimensions, tagged with a 0-based
`component` chunk index (`gltfworld.ext.rwm._chunks`,
`src/gltfworld/ext/rwm.py`; schema-validated by
`docs/schemas/rwm/RWM_state_series.schema.json`; exercised by
`tests/test_khr_schema.py`'s 5-dim-action/7-dim-pose_variance cases).

**What a proper extension would need**: either a new, wider accessor type
(a real glTF-core change, unlikely to happen for a narrow use case) or a
standardized chunk-and-reassemble convention baked into whatever
Khronos-track time-series extension eventually exists, so every consumer
doesn't reinvent this.

**Prior art**: none — this is a glTF-specific type-system constraint that
doesn't map onto USD (arbitrary-width `VtArray`-backed attributes) or
MJCF/URDF (plain-text numeric lists, unconstrained width) at all.

### G4. No discrete collision/contact-event concept

**What glTF lacks**: no concept of a discrete, non-continuous event record
(e.g. "these two bodies started/stopped touching at frame t, at this
contact point, with this relative velocity"). Every glTF/RWM channel is a
continuous per-frame sample.

**Why it matters**: real physics engines produce genuinely richer
event-level output than per-frame pose/velocity alone — useful for contact-
based reward signals, causal event detection, or any task (like Physion's
own Object Contact Prediction) that is fundamentally about discrete events,
not continuous trajectories.

**Severity**: Significant for any application needing event-level ground
truth; this project's own OCP evaluation (V8) worked around it by deriving
contact from continuous state (nearest-vertex proximity) rather than
consuming a native event channel.

**Workaround implemented here**: none attempted — a real, acknowledged gap.
Physion's HDF5 `frames/*/collisions/*`/`env_collisions/*` (contact points,
per-event relative velocity, discrete enter/stay/exit strings) is simply
**not carried into the GLB conversion at all**
(`gltfworld.physion.convert`, `src/gltfworld/physion/convert.py`; see
docs/PHYSION.md finding 8) — the one finding in this project's whole
conversion experience where it is *gltfworld's* schema, not the source
format's, that has nothing to receive richer available information.

**What a proper extension would need**: a discrete-event channel type
(irregularly-timed, not tied to the shared per-frame time accessor at all)
— structurally a different shape from every other `RWM_state_series`
channel, which assumes one sample per recorded frame.

**Prior art**: MuJoCo's own `mjContact` struct and TDW's HDF5 event records
both do this natively; no Khronos draft addresses it.

### G5. No video/frame-sequence-as-scene-content concept

**What glTF lacks**: glTF has textures (static images) and animated
materials via extensions, but no first-class "this is a recorded video
observation of the scene" concept — pixels simply have no home in a glTF
file as *simulation output* (as opposed to an authored texture).

**Why it matters**: a world-model pipeline's rendered RGB/depth/
segmentation frames are exactly this kind of content, and this project
deliberately keeps them *outside* the GLB (as sibling `.npy` files, see
`gltfworld.data.dataset.PerceptionDataset`) rather than forcing them into
the transport — a design choice, but one glTF's own feature set left no
real alternative to.

**Severity**: Significant, generically (this is the single largest reason
Physion's own team needed a second, non-glTF container tier — see G21).

**Workaround implemented here**: not attempted inside glTF at all — frame
data lives in per-episode sibling files
(`rgb.npy`/`seg.npy`/`depth.npy`, `gltfworld.render.renderer
.EpisodeRenderer`, `src/gltfworld/render/renderer.py`), memory-mapped by
`PerceptionDataset` rather than duplicated into the packed dataset (see
DESIGN.md's V4 section).

**What a proper extension would need**: this is arguably out of scope for
glTF as a *format* at all (video codecs, streaming, memory-mapping are a
different problem domain from a scene-description format) — the more
realistic ask is a standardized *sidecar convention* (a `KHR_video_frames`-
style pointer extension recording which external file/frame range
corresponds to which time-accessor range), not video-in-glTF itself.

**Prior art**: Physion's own `mp4s`/`mp4s-redyellow` are exactly this kind
of sidecar, informally — see G21.

### G6. No standard uncertainty/confidence channel — and i.i.d. noise is the wrong model anyway (flagship finding)

**What glTF lacks**: no Khronos extension, draft or shipped, has any
concept of per-object state uncertainty. This project's own
`RWM_state_series.pose_variance` channel (3 position + 4 quaternion
variance components, diagonal only, no cross-time correlation term) is a
wholly custom, bespoke solution with no standard counterpart anywhere in
the glTF/KHR ecosystem to converge toward.

**Why it matters, and the measured twist**: it would be tempting to assume
the fix is simply "standardize a per-frame variance channel" and move on.
This project's own V7 closed-loop attribution experiment shows that's not
enough — it would encode the *wrong statistical model*. Three arms were
compared: (A) oracle ground truth, (B) oracle poses perturbed by
i.i.d.-per-frame Gaussian noise calibrated to the real perception model's
*measured* error magnitude (via an exact chi(3) inversion, not an RMS
approximation — `gltfworld.eval.closed_loop.noise_params_from_metrics`),
and (C) the real, visual closed loop (render -> `PerceptionDETR` -> match
-> roll forward). **Arm B diverged 17x faster than Arm C at h=99** (27.6m
vs. 1.62m median position error) — an i.i.d.-Gaussian noise model,
calibrated to the *correct* per-frame error magnitude, still massively
overestimates real closed-loop degradation, because it's the wrong
*temporal* model: measured lag-1 autocorrelation of real detector error is
**0.55-0.82** across position/rotation (docs/RESULTS.md's V7 section) — a
detector's error on frame t is strongly predictive of its error on frame
t+1, so finite-differenced velocity noise *partially cancels* rather than
compounding via the `sqrt(2)` amplification an i.i.d. model assumes.

**Why this matters for a future spec**: any standardized uncertainty
channel that only carries a per-frame variance (the natural, minimal thing
to standardize) invites exactly this wrong i.i.d. assumption downstream.
A channel that's actually useful for closed-loop reasoning needs either an
explicit temporal-correlation parameter (e.g. an AR(1) coefficient
alongside the per-frame variance) or — more realistically — an explicit
disclaimer in the spec text that per-frame variance alone does not license
an i.i.d. assumption for anything that consumes multiple frames.

**Severity**: Blocking for any serious closed-loop/planning use of a
perception-model's uncertainty output; the gap isn't "no channel exists" so
much as "the only channel *shape* anyone would naturally standardize
encodes a demonstrably wrong noise model."

**Workaround implemented here**: `pose_variance` channel
(`gltfworld.ext.rwm`, docs/RWM_EXTENSIONS.md) exists and is exercised, but
this project does **not** claim it's sufficient — the V7 finding above is
reported specifically so a future extension author doesn't repeat the
same implicit i.i.d. assumption. `gltfworld.eval.closed_loop`
(`src/gltfworld/eval/closed_loop.py`) is the artifact that surfaced this;
`tests/test_closed_loop.py`'s noise-injection-statistics tests confirm the
i.i.d. injection itself is implemented correctly (matches its requested
sigma empirically) — the finding is about the *model choice*, not a bug in
this project's own noise injection.

**What a proper extension would need**: at minimum, explicit spec language
that a per-frame variance channel is not, by itself, sufficient basis for
i.i.d. rollout-noise modeling; ideally a richer channel shape (a
lag-1-autocorrelation coefficient, or a short covariance window) informed
by exactly this kind of empirical measurement across real detectors.

**Prior art**: none in the glTF/KHR ecosystem. Robotics uncertainty
representations (e.g. covariance matrices on `nav_msgs/Odometry` in ROS) are
per-message/per-frame only, with no standard temporal-correlation
convention either — this appears to be a genuinely open problem beyond
just glTF.

---

## Part B — Draft KHR physics extensions: real gaps found implementing against them

### G7. No collider local offset/center field on `KHR_implicit_shapes`

**What glTF lacks**: in the pinned commit, a `KHR_implicit_shapes` shape
(box/sphere/cylinder/capsule) is only ever defined centered on its owning
node's origin — no local offset/transform property exists to place a
collider anywhere else relative to that node.

**Why it matters**: this is not a hypothetical — it is *the exact reason*
this project's articulated-object design needed an extra node per joint
attachment point (V9-prep) rather than simply relocating an object's own
node origin to the physical hinge/slide point (which would have
desynchronized the collider from the visual mesh, both still centered on
the relocated origin but no longer at the mesh's true geometric center).
It surfaced *again*, independently, converting Physion: TDW's own
primitives are base-pivoted (local Y in `[0, height]`, not
`[-height/2, height/2]`), so the visual mesh (placed at the raw pivot,
correctly) and the KHR-physics collider approximation (implicitly centered
on that same pivot, per the schema's own assumption) are *displaced from
each other* along the local pivot-to-center vector for every converted
Physion object — a real, surfaced-twice fidelity cost, not a one-off.

**Severity**: Significant (worked around cleanly for articulation via an
extra node; left as an acknowledged, unfixed fidelity gap for Physion's
collider placement).

**Workaround implemented here**:
- Articulation: a motion-less, geometry-less "joint pivot" child node per
  attachment point (`gltfworld.scene.convert`, `src/gltfworld/scene/
  convert.py`; see DESIGN.md's "Attachment frames, concretely").
- Physion: no workaround attempted — the true center is separately
  available (`classify_bounding_shape`'s returned `center`,
  `gltfworld.physion.convert`) and used for stage-3's own contact-geometry
  math, but the *KHR-physics encoding itself* has nowhere to put it, so the
  collider approximation stays displaced from the true mesh center
  (docs/PHYSION.md finding 2).

**What a proper extension would need**: an optional local offset
(translation, and ideally full local transform) property on each
`KHR_implicit_shapes` shape type, matching what almost every physics
engine's own collider-authoring API already provides (MuJoCo's geom `pos`/
`quat` relative to its body, PhysX's `PxShape` local pose, Unity's collider
`center`).

**Prior art**: MJCF (`<geom pos="..." quat="...">` inside a `<body>`),
URDF (`<collision><origin xyz="..." rpy="..."/>`), UsdPhysics (a
`UsdPhysicsCollisionAPI` collider is itself a full prim with its own
xformable transform, so this isn't even a special case there) all support
this as a basic feature; glTF's implicit-shapes draft not supporting it is
a genuine gap relative to every comparable format.

### G8. `joint.limit`'s stiffness/damping are soft-stop parameters, not viscous joint damping

**What glTF lacks**: `KHR_physics_rigid_bodies.physicsJoints[].limit`'s
`stiffness`/`damping` describe the *restorative* force applied once a
limit is exceeded (an optional soft spring instead of a hard stop) — by
default the limit is infinitely stiff. There is no property anywhere in
this pinned commit's joint schema for continuous, in-range viscous drag
across a joint's whole free range (MJCF/URDF's per-joint `damping`), nor
for `armature` (MJCF's added rotational/reflected inertia term for
actuator/joint numerical stability — not physical damping at all, but
likewise unrepresentable).

**Why it matters**: this project's own generated door/drawer episodes
(`wm-articulated-v1`) *use* real MJCF joint damping to get physically
plausible settling behavior after a scripted push — that parameter simply
has no home in the KHR encoding. A downstream KHR-only consumer doing
fresh forward simulation from the `.glb` alone would see an undamped (or
only soft-limit-damped) joint, not the one MuJoCo actually simulated —
a real, silent fidelity loss for anyone re-simulating from the transport.

**Severity**: Significant (affects re-simulation fidelity, not playback —
this project's own use of the GLB, replay + state-series consumption,
never re-simulates from the KHR joint dict alone, so the gap is latent
rather than actively broken in this project's own pipeline).

**Workaround implemented here**: none — an acknowledged, unfixed gap.
Documented in DESIGN.md's "Honest gaps (feeding the full V9 gap report)"
subsection (under "Articulated objects (V9-prep)").

**What a proper extension would need**: a `damping` field on the joint
itself (not just its limit), applied continuously across the joint's whole
free range, plus an optional `armature`-equivalent for numerical-stability
tuning — both already first-class in MJCF/URDF.

**Prior art**: MJCF `<joint damping="..." armature="...">`; URDF
`<joint><dynamics damping="..." friction="..."/>`.

### G9. `joint.drive` models a persistent spring-to-target, not a one-shot push

**What glTF lacks**: `KHR_physics_rigid_bodies.physicsJoints[].drive`'s
force model is `stiffness * (positionTarget - positionCurrent) +
damping * (velocityTarget - velocityCurrent)` — an always-on motor chasing
a target, structurally incapable of expressing a finite-duration external
impulse that is then released.

**Why it matters**: this project's own door/drawer episodes are actuated
by exactly this kind of scripted, bounded-duration push (`gltfworld.datagen
.articulated`, `src/gltfworld/datagen/articulated.py`) — a real, common
actuation pattern (a person shoves a door, then lets go) that the KHR joint
schema has no way to represent without misrepresenting it as a permanently-
active motor holding some target forever.

**Severity**: Significant (a real, common actuation pattern with no
faithful encoding).

**Workaround implemented here**: not encoded as a KHR `drive` at all — the
driving force only ever existed inside the MuJoCo simulation that produced
the recorded `poses`/`joint_pos` trajectory; the KHR joint dict for these
episodes carries `limits` only, no `drives` (documented in DESIGN.md's
"Honest gaps" subsection under V9-prep).

**What a proper extension would need**: either a genuinely time-bounded
drive (a start/end time or duration alongside the existing
stiffness/damping/target fields) or an explicit "one-shot impulse" force
primitive distinct from the persistent-spring `drive` model.

**Prior art**: MJCF's `<general>`/`<motor>` actuators can be driven by an
external, time-varying control signal (including a scripted, bounded pulse)
rather than only a fixed spring-to-target; ROS's `effort_controllers` and
most robotics motion-planning stacks assume bounded-duration commanded
forces/torques as the default actuation model, not a persistent target.

### G10. No weld-joint/rigid-constraint primitive for cosmetic rigidly-attached parts

**What glTF lacks**: no way to express "this part is rigidly welded to
that part" as an explicit physics constraint — only the two joint types
this project implements (revolute, prismatic), both of which model a
genuine, one-degree-of-freedom relative motion.

**Why it matters**: this project's articulated scenes include a handle
rigidly attached to a door/drawer — purely cosmetic, no real degree of
freedom relative to the part it's attached to. Doing this properly (a
second, anchor-aligned pivot-node pair purely to describe a rigid weld,
mirroring the main joint's own construction) was judged not worth the
added node/joint count for a purely cosmetic part in this project's own
scope.

**Severity**: Minor for this project's own playback/training use case
(the handle's animated pose track is always exactly, derivedly consistent
with rigidly following its parent part); Significant for any downstream
engine attempting fresh forward simulation from the `.glb` alone without
this project's own `extras.rwm.semantics` convention to lean on.

**Workaround implemented here**: the handle's motion is *derived*
(`handle_pose(t) = part_pose(t) ∘ (handle_local_offset, identity)`), not
independently simulated or KHR-joint-constrained
(`gltfworld.datagen.articulated`, `src/gltfworld/datagen/articulated.py`);
`extras.rwm.semantics` labels it (`{"labels": ["handle"], "affordances":
["pullable"]}`) so this project's own tooling can identify it, but nothing
in the KHR-standard part of the encoding says "this moves rigidly with its
parent."

**What a proper extension would need**: a fixed/weld joint type (zero DOF
on every axis) — a small, natural addition alongside the existing
revolute/prismatic limit-composition machinery this project already uses.

**Prior art**: MJCF's `<weld>` constraint and URDF's `"fixed"` joint type
both exist for exactly this case.

### G11. Two friction coefficients collapse into one

**What glTF lacks**: `KHR_physics_rigid_bodies`'s physics-material schema
(as this project encodes it) carries exactly one friction field. Real
Coulomb-friction physics — and Physion's own HDF5 source data
(`static_friction`/`dynamic_friction`, two independent per-object values)
— distinguishes static from dynamic (kinetic) friction.

**Why it matters**: this is a real, lossy information collapse whenever
converting from a source with genuine two-coefficient friction, not a bug
in either format — `gltfworld`'s own physics-material encoding was built
against `wm-scenes-v1`'s single-friction-value physics distribution and
never needed a second coefficient before Physion's data exposed the gap.

**Severity**: Minor to Significant depending on the downstream physics
regime (static-vs-kinetic friction divergence matters most for stick-slip
and near-threshold sliding scenarios, which this project's own OCP task
does not depend on precisely).

**Workaround implemented here**: `gltfworld.physion.convert`
(`src/gltfworld/physion/convert.py`) averages the two into one value at
conversion time (docs/PHYSION.md finding 3) — a documented, one-directional
lossy collapse, not a round-trippable encoding.

**What a proper extension would need**: a second, optional
`dynamicFriction` field alongside the existing `staticFriction` (or a
rename to make the existing single field's semantics explicit) in the
physics-material schema.

**Prior art**: MuJoCo's own contact model *also* only exposes a single
scalar per axis-pair by default (its `friction` triplet is
tangential/torsional/rolling, not static-vs-dynamic) — so this gap is not
unique to glTF; it echoes the general "two-coefficient friction is a
minority feature even among physics engines" state of the art. PhysX and
Unity's `PhysicMaterial` both do expose separate static/dynamic
coefficients, which is where the Physion/TDW data this project converted
originated from.

### G12. No generic mesh/convex-hull collider type

**What glTF lacks**: `KHR_implicit_shapes` offers only implicit primitives
(box, sphere, capsule, cylinder in the pinned commit) — no arbitrary mesh
or convex-hull collider type for real-world asset geometry that isn't well
approximated by any of those.

**Why it matters**: every non-primitive Physion asset (`cone`, `torus`,
`dumbbell`, real furniture meshes like `linbrazil_diz_armchair`) has to be
crudely approximated. This project's own shape vocabulary
(`gltfworld.scene.scene.SHAPES`, sphere/box/cylinder only) makes this
worse still — an AABB-isotropy classifier (`classify_bounding_shape`)
cannot even distinguish a genuine cylinder from a box by bounding-box shape
alone, so every non-sphere Physion asset converted in this project's own
pipeline was boxed regardless of true shape (docs/PHYSION.md finding 4;
confirmed directly: `cone`/`torus`/`dumbbell` all classified `"box"`).

**Severity**: Significant for any real-world-asset conversion (which this
project's own Physion work is direct evidence of); Minor for this project's
own synthetic `wm-scenes-v1`/`wm-articulated-v1` distributions, which were
authored with sphere/box/cylinder primitives from the start.

**Workaround implemented here**: `classify_bounding_shape`
(`gltfworld.physion.convert`) picks the nearest of sphere/box by AABB
isotropy — a documented, one-directional lossy approximation, shared
verbatim between the physics-encoding path and the dynamics-tensor mapping
so at least the approximation is applied consistently rather than twice,
differently (docs/PHYSION.md finding 4).

**What a proper extension would need**: an optional mesh-collider or
convex-hull-collider shape type referencing an existing glTF mesh accessor
set directly, the same way `KHR_implicit_shapes`' existing shapes reference
implicit parameters — letting the *visual* mesh double as (or generate) the
physics collider for irregular real-world geometry.

**Prior art**: MuJoCo (`<geom type="mesh">`), PhysX (convex mesh
cooking), and Unity (`MeshCollider`) all support this as a standard
collider type; UsdPhysics has `UsdPhysicsMeshCollisionAPI` for exactly this
purpose.

### G13. No root-level scene physics properties (gravity) in the pinned commit

**What glTF lacks**: the pinned `KHR_physics_rigid_bodies` commit has no
root-level (scene- or document-wide) property for gravity or any other
global physics parameter — every property lives on a node or joint.

**Why it matters**: gravity is a genuinely scene-global quantity in every
physics engine this project touches (MuJoCo, TDW/Unity) — forcing it into
a per-node encoding (or omitting it) loses the "one value governs the whole
scene" semantics a real physics-authoring tool would expect.

**Severity**: Minor (cleanly worked around with a documented, non-KHR
field).

**Workaround implemented here**: `extras.rwm` per scene carries `gravity`
alongside `seed`/`scene_version`/`dt` (docs/RWM_EXTENSIONS.md's "Per-scene"
section; see DESIGN.md's "Documented deviations from the milestone spec
text" for the same pattern applied to static-object mass).

**What a proper extension would need**: a root-level (or per-`scene`)
gravity vector property directly on `KHR_physics_rigid_bodies`'s own
document-level object, rather than requiring every consumer to invent its
own extras convention.

**Prior art**: MJCF's `<option gravity="0 0 -9.81"/>` is a single
document-root property; URDF has no native gravity concept at all (left to
the simulator loading the URDF) — glTF sits between the two, closer to
URDF's gap than MJCF's solution.

### G14. No chirality/handedness declaration

**What glTF lacks**: core glTF specifies a right-handed coordinate
convention in its own spec text, but neither core glTF nor the draft
physics extensions provide any mechanism to *declare* or *negotiate*
handedness for content authored in a left-handed engine — the assumption
is simply "everything is right-handed," full stop.

**Why it matters, concretely**: TDW (Physion's underlying engine, itself
built on Unity) is left-handed. Converting its HDF5 ground truth into this
project's right-handed transport surfaced this two independent ways: the
raw quaternion data reproduces TDW's own `forwards` vectors exactly under a
right-handed read (self-consistent within TDW's own frame), while the
camera's `camera_matrix` rotation block has **determinant -1** under that
same naive right-handed read (a genuine, confirmed — not numerical-noise —
improper/chirality-flipping transform: `R @ R.T == I` holds, ruling out
simple noise). This project deliberately applies **no** chirality-
correcting flip anywhere in the conversion (every metric stage 3 computes —
Euclidean distance, contact thresholds, velocity magnitude, the dynamics
model's own features — is invariant under a single global mirror of all
coordinates together, so it has zero effect on any reported number), but it
does mean a GLB from this conversion, rendered by an ordinary right-handed
glTF viewer, shows a left-right-mirrored scene relative to Physion's own
reference renders of the same trial.

**Severity**: Minor for this project's own state-based use (zero metric
impact, confirmed); Significant for any visual/rendering use of the same
converted data, or for any pipeline mixing left- and right-handed sources
without this project's own careful invariance analysis.

**Workaround implemented here**: none attempted — documented as a known,
deliberately-unaddressed limitation
(`gltfworld.physion.convert._decode_camera`, docs/PHYSION.md findings 5-6).
Camera orientation is separately *synthesized* (a look-at aimed at the
frame-0 object centroid) rather than decoded from the chirality-ambiguous
matrix at all.

**What a proper extension would need**: not really a glTF-side fix at
all — a documented, standard practice (or lightweight per-document
metadata flag) for declaring "this content was authored in a left-handed
engine and mirrored at export" so downstream consumers know to expect it
rather than silently mis-set. This is arguably an ecosystem/tooling
convention gap more than a spec gap.

**Prior art**: FBX (also left-handed by default, Autodesk convention) and
Unity/Unreal interchange has faced this exact problem for years; the
common industry practice (mirror one axis at export/import) is exactly
what this project chose *not* to do, deliberately, given the confirmed
zero-impact on its own metrics.

---

## Part C — Real-world evidence: nobody else has solved this either

### G15. No published project uses glTF as an ML world-model transport

**Finding**: a literature/prior-art scan turned up no published project
using glTF as the interchange between physics simulation, rendering, and
learned perception/dynamics models — the exact role this project's own
pipeline occupies. This is not proof that it's impossible (this project is
itself a counter-existence-proof that it's *possible*), but it is real
evidence that this specific use case sits outside glTF's actual, observed
adoption pattern (asset interchange, real-time rendering, static/kinematic
animation).

**Severity**: n/a (a comparative-landscape finding, not a technical gap).

**What this project demonstrates instead**: a real, working counter-
example — nine milestones of a functioning glTF-as-world-model-transport
pipeline, independently verified at every stage (docs/VERIFICATION.md).

### G16. Physion invented a bespoke HDF5+MP4+CSV container rather than reaching for any structured transport

**Finding**: Physion's own benchmark team, needing to carry rendered video
+ per-frame object/camera state + depth/flow/segmentation + a scalar OCP
label through a research pipeline, did not reach for glTF or any other
existing self-describing structured transport. They built a two-tier,
ad hoc combination instead: plain HDF5 for the rich per-frame state tier
(exhaustively documented in this project's own docs/PHYSION.md, "Physion
HDF5 schema" section), plain MP4 for the video tier models actually
consume, and a hand-rolled CSV for the one scalar label everything is
graded against — with the two tiers ("Core" vs. full HDF5) carrying
genuinely different information and **no single self-describing file tying
video frame t to its corresponding object/camera state.** The full HDF5
tier is **380 GiB** for the complete 8-scenario test set (verified by
direct HTTP HEAD request against the hosting bucket, docs/PHYSION.md);
the lightweight "Core" archive most researchers actually download (270.9
MiB) carries *no structured scene-state ground truth at all* — only
rendered pixels and the final binary label.

**Why it matters**: this is the single clearest piece of real-world
evidence for this whole report's thesis. A general-purpose, self-
describing transport (frames + object state + camera + labels, one file,
one loader) would have let this benchmark ship a single artifact instead
of a two-tier "video-only, 270MB" / "everything, but 380GB, mostly pixels"
split with no format in between.

**Severity**: n/a (real-world comparative evidence).

**This project's own conversion of exactly this data** (V8,
`gltfworld.physion.convert`) is the concrete demonstration that a
glTF-based single-file alternative is achievable: 150 real Collide trials,
each converted from raw HDF5 into a single, self-describing, validator-
clean GLB carrying real mesh geometry, per-frame pose/velocity, physics
metadata, and role semantics together (docs/PHYSION.md, docs/RESULTS.md's
V8 section) — at a small fraction of the 380GB the source format needed to
express the same physically-relevant content, because it never had to
carry the per-frame pixel/depth/flow/normals data every OCP-relevant metric
in this project's own evaluation never touches (docs/PHYSION.md finding
10).

### G17. ReplicaCAD ships GLB + a separate URDF sidecar because glTF alone can't express joints

**Finding**: ReplicaCAD (a widely-used articulated-scene dataset for
embodied-AI research) ships its articulated assets as GLB *plus* a separate
URDF sidecar file — direct, independent evidence that glTF's lack of a
native joint concept (before this project's own draft-`KHR_physics_
rigid_bodies`-based joint work, V9-prep) is a real, currently-unaddressed
gap forcing exactly the kind of two-file split this report's G1/G2 findings
describe in the abstract. This project's own V9-prep/V9 milestones show
that the draft KHR joint extension *can* close this specific gap (a single
GLB, no URDF sidecar, carrying hinge/slider joints, round-tripped and
independently verified) — but ReplicaCAD's own two-file convention predates
that draft extension's availability and remains the norm elsewhere.

**Severity**: n/a (real-world comparative evidence, directly supporting
G1/G2/G8/G9's severity assessments).

### G18. MuJoCo <-> KHR semantic mismatches beyond friction: soft-contact vs. scalar restitution

**Finding**: `KHR_physics_rigid_bodies`'s physics-material schema exposes a
single scalar `restitution` coefficient — the same simple, textbook
"bounciness" model most physics-authoring tools expose. MuJoCo's actual
contact model has no such single coefficient at all: contacts are resolved
via a soft-constraint (`solref`) formulation, `(timeconst, dampratio)`,
with no textbook restitution term to read off directly. This project's own
MuJoCo episode generator (`gltfworld.datagen.mujoco_env.scene_to_mjcf`,
`src/gltfworld/datagen/mujoco_env.py`) has to go the *other* direction —
mapping a KHR-style scalar restitution *into* MuJoCo's `solref` form
(`dampratio = 1 - restitution`, `timeconst` fixed at MuJoCo's recommended
minimum) — a documented, one-directional, lossy approximation (see
DESIGN.md's "MJCF construction" section). This is the reverse of the
friction-collapse direction in G11 (there, a real source has *two*
coefficients KHR only has one for; here, KHR's own model is simpler than
what the actual physics engine driving this project's own ground-truth
generation needs).

**Severity**: Minor at this project's own scale (`wm-scenes-v1` fixes
restitution at a low, fixed 0.1, where the approximation barely matters in
practice, per DESIGN.md) — but a structural mismatch that would bite harder
at higher, more varied restitution values.

**What a proper extension would need**: either accept that a single scalar
restitution is an inherently lossy simplification of any real soft-contact
physics engine's internal model (true for essentially every engine, not
just MuJoCo — this may be an acceptable, permanent abstraction boundary)
or expose engine-specific contact-model parameters as optional extras
alongside the portable scalar.

**Prior art**: URDF has no restitution concept at all (left entirely to
the simulator); MJCF exposes its native `solref`/`solimp` directly (no
portable "restitution" abstraction at all) — KHR's simpler scalar sits
between the two, trading fidelity for portability, a reasonable design
choice but a real, measured one-directional lossy conversion in both
directions this project actually exercises.

---

## Part D — Comparative yardstick and tooling maturity

### G19. UsdPhysics 1.0 as the mature comparison yardstick

**Finding**: Pixar/OpenUSD's `UsdPhysics` (shipped as of USD 1.0, not
draft) is the most directly comparable "physics schema layered on a scene
graph" effort to what a finished KHR physics spec would need to become.
Structurally, USD's advantages over glTF for this exact use case stem from
properties baked into USD from the start, not bolted on: (a) **arbitrary
time-sampled attributes on any prim** (USD's core time-sampling model is
general-purpose, unlike glTF's animation-only-on-TRS-and-morph-weights
design — see G1: this is *why* USD would never need an
`RWM_state_series`-equivalent bolt-on extension at all), (b) **colliders as
first-class, independently-transformable prims** (directly closing G7's
collider-offset gap, since a `UsdPhysicsCollisionAPI` prim has its own full
xformable transform rather than being implicitly centered on its owning
body), and (c) a broader, already-shipped collider vocabulary including
mesh/convex-hull colliders (closing G12). This project's own experience —
needing a wholly custom time-series extension (`RWM_state_series`) and an
extra-node workaround for collider offsets (G7) — is a direct, lived
illustration of exactly the structural gaps USD's design choices avoid.

**Severity**: n/a (comparative yardstick, not a technical gap in itself).

**What this implies for a Khronos-track roadmap**: `UsdPhysics 1.0`
demonstrates these problems are solvable within a scene-graph format; the
open question for glTF specifically is whether closing G1/G7/G12 requires
glTF-core changes (a genuinely time-sampled attribute system, unlikely
given glTF's deliberately minimal, rendering-focused design goals) or
whether draft/future KHR extensions can achieve most of the same value
within glTF's existing accessor/animation constraints (this project's own
`RWM_state_series` is itself evidence the latter is at least partially
achievable, with real friction, as documented throughout this report).

### G20. Reference-tooling maturity: pygltflib silently drops legitimate empty-string data

**Finding**: `pygltflib`'s JSON serialization path (`gltf_to_json` ->
`delete_empty_keys`) silently deletes any dict entry whose value has
`len() == 0` (empty string/list/dict) *anywhere in the document*, including
inside `extras.rwm.parts`. This project does not work around it (would mean
monkeypatching a third-party dependency) — the Hypothesis round-trip test
(`tests/test_roundtrip.py`) deliberately avoids generating empty-string
`parts` values as a result, and DESIGN.md records plainly that a real
caller putting an empty string into `ObjectSpec.parts` will not get it back
after a real GLB save/load round trip.

**Why it matters**: this is not a glTF *spec* gap — it's a maturity gap in
one widely-used reference implementation, found only because this project
actually exercised round-trip fidelity with property-based testing rather
than assuming a mature library is bug-free. Anyone building a serious
transport on top of glTF's Python tooling ecosystem should expect to find
similar edge cases.

**Severity**: Minor (narrow, documented, avoided rather than silently
eaten).

**Workaround implemented here**: documented as a known deviation
(DESIGN.md's "Documented deviations from the milestone spec text" — "
pygltflib empty-value pruning"); no monkeypatch, no silent workaround.

---

## Positive findings: what glTF got right for this use case

It would be dishonest to report only gaps. Several real design properties
of glTF made this project's transport-first architecture *work*, not just
"work around obstacles":

**P1. Off-the-shelf viewer interop actually works.** Because
`extensionsRequired` is deliberately kept empty for every custom/draft
extension this project writes (only `extensionsUsed` lists them, per
DESIGN.md's "Transport encoding" section), an ordinary, standard glTF
viewer with zero knowledge of physics or state-series extensions can drag-
and-drop-load any episode this project produces and correctly play its
pose animation (`docs/VERIFICATION.md`'s V3 "MANUAL — preview a generated
episode in a real glTF viewer" checkpoint: verified against
<https://gltf-viewer.donmccurdy.com>, no console errors). This is a real,
demonstrated interop property, not a theoretical one — the whole point of
building on core glTF rather than a wholly bespoke binary format.

**P2. Validator-clean with additive extensions, at real scale.** Every GLB
this project has ever produced — hand-built fixtures, 10,000-episode
`dynamics-v1`, 500-episode `perception-v1`, 150 real Physion conversions,
1,500 articulated episodes — passes the independent, pinned Khronos
glTF-Validator binary with **zero errors** (the only non-error messages are
expected `UNSUPPORTED_EXTENSION`/`UNUSED_OBJECT` info-level notes, since the
validator doesn't know about draft/custom extensions). This was verified,
not assumed, at every scale this project operated at, including a real
regression (V8.1, 12/150 zero-length-normal defects) that the validator
itself caught and this project fixed — direct evidence the validator
discipline this project followed actually catches real defects, not just a
rubber stamp.

**P3. Single-file episodes.** Mesh geometry, camera, lights, pose
animation, rigid-body physics parameters, joint constraints, and every
custom time-series channel (velocity, action, uncertainty, joint state) all
live in exactly one GLB per episode — no sidecar files, no separate label
CSV, no split "video tier" vs. "state tier" the way Physion's own two-tier
container needed (G16). This is the single most direct payoff of building
on glTF's binary-chunk-plus-JSON container model rather than a bespoke
multi-file convention.

**P4. Accessor machinery generalizes cleanly to arbitrary channels.** Core
glTF's accessor/bufferView system — designed for vertex positions, normals,
and animation keyframes — turned out to be flexible enough to carry
*every* custom time-series channel this project needed (velocity, action,
pose uncertainty, joint position) with zero new binary encoding: every
`RWM_state_series` channel is an ordinary SCALAR/VEC2/VEC3/VEC4 accessor,
sharing the same time accessor pose animation's samplers already use. This
is real evidence that glTF's core binary machinery, not just its JSON
extension mechanism, is reusable well beyond its original mesh/animation
design intent.

**P5. Independent-renderer cross-validation is possible and passes.** This
project's transport is checkable, not just internally self-consistent: an
independent renderer (MuJoCo's own `mujoco.Renderer`) reconstructs scene
geometry from the *same* GLB via a documented, tested coordinate conversion
(`gltfworld.datagen.mj_convert`) and agrees with this project's own
pyrender-based renderer to **0.9962 binary silhouette IoU** (measured,
`tests/test_crosscheck.py`) — real evidence the transport encodes enough
information, precisely enough, for two independent consumers to agree on
what it describes.

---

## Recommendations (ranked)

What Khronos-track extension work would most unblock glTF as a real
world-model/robotics transport, ranked by (impact this project measured) x
(how blocking the gap was):

1. **Ratify a real KHR time-series/state extension, scoped explicitly
   alongside `KHR_physics_rigid_bodies`, not as a replacement for it**
   (closes G1, G2, G3). This is the single highest-impact, most-validated
   recommendation in this report: every one of this project's nine
   milestones depended on `RWM_state_series` existing, and G2's
   cross-cutting finding shows that shipping the physics extensions alone
   (even to full ratification) would not obsolete this need. Concretely:
   standardize the `{node|joint|scene}`-target + shared-time-accessor +
   named-channel pattern this project's own custom extension already
   validates at scale (10,000+ episodes, 0 schema-validation failures).

2. **Add a local offset/transform field to `KHR_implicit_shapes` colliders**
   (closes G7). The narrowest-scoped, cheapest-to-ratify fix on this list
   with an outsized fidelity payoff — it eliminates an entire class of
   workaround (this project's own joint-pivot-node design exists *purely*
   because of this one missing field) and directly fixes the
   mesh-pivot-vs-collider-center displacement this project's own Physion
   conversion couldn't avoid.

3. **Add joint viscous damping/armature and a genuinely bounded-duration
   drive mode** (closes G8, G9). Both gaps are common, ordinary actuation
   patterns (a damped hinge; a scripted push) that every comparable format
   (MJCF, URDF) already expresses natively — closing them would let a KHR
   joint dict alone (no side-channel `extras.rwm` metadata) faithfully
   describe the exact articulated scenes this project already generates
   and verifies.

4. **Add a fixed/weld joint type** (closes G10). A small, natural addition
   given the limit-composition machinery this project already implements
   for revolute/prismatic joints — closes the one remaining "derived, not
   constrained" pose relationship in this project's own transport.

5. **Add an optional mesh/convex-hull collider shape** (closes G12).
   Highest-value for any real-world-asset conversion (this project's own
   Physion work is direct, measured evidence of the cost: every non-
   primitive real asset boxed regardless of true shape) but the most
   involved to spec and implement correctly (mesh cooking, convexity
   requirements) — ranked lower than the cheaper fixes above despite
   real impact.

6. **Publish explicit guidance that a per-frame uncertainty channel is not,
   by itself, license for an i.i.d.-noise rollout model** (closes G6, at
   least partially). This is not primarily a schema change — it's a
   documentation/best-practices gap with a real, measured cost (this
   project's own Arm B vs. Arm C finding: 17x overestimate at h=99 from
   exactly this wrong assumption). Ranked lower only because it's advisory
   rather than blocking any particular pipeline from functioning at all;
   ranked here rather than omitted because the cost of getting it wrong
   silently (a planner trusting an i.i.d.-calibrated uncertainty estimate)
   is genuinely severe.

7. **Add a second, optional friction coefficient and a root-level gravity
   property** (closes G11, G13). Both are narrow, mechanical schema
   additions with modest but real payoff — grouped last not because they
   don't matter, but because both already have clean, low-cost extras-based
   workarounds this project's own transport already demonstrates work fine
   in practice.

**Explicitly not recommended**: a glTF-native video-frame-sequence concept
(G5) or a glTF-core change to widen accessor types past VEC4 (G3) — both
are addressable more cheaply by convention (sidecar-file pointers; documented
channel-splitting, respectively) than by changing glTF's own scope or type
system, and this project's own workarounds for both are already adequate
in practice.
