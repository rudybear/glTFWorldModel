---
marp: true
theme: default
paginate: true
title: "glTF as a World-Model Transport: Findings from an End-to-End Pipeline"
style: |
  section { font-size: 24px; }
  section h2 { font-size: 38px; }
  table { font-size: 21px; }
---

# glTF as a World-Model Transport
## Evidence-backed findings from an end-to-end ML pipeline

**Khronos 3D Formats WG presentation**
rudybear · github.com/rudybear/glTFWorldModel · MIT

*Every finding in this deck has a code pointer and a measurement in the public repo.*

---

## Why this experiment

- World models & robotics need a **scene-state interchange**: simulation → training → inference → rendering
- Today's reality: everyone invents a container
  - Physion (NeurIPS benchmark): bespoke HDF5 + MP4 + CSV
  - Habitat/ReplicaCAD: **GLB for geometry + URDF sidecar** — because glTF lacks joints
  - OpenUSD: UsdPhysics 1.0 shipped in core
  - No published project uses glTF as an ML scene-state transport
- Question: **how far does glTF 2.0 + draft extensions get — what exactly is missing?**
- Method: **build the whole pipeline; record every impedance mismatch**

---

## What we built (all open source, MIT)

```
MuJoCo sim ──► GLB episodes ──► headless renderer ──► rgb/seg/depth
(ground truth)  pose animation      (vendored pyrender)      │
                + KHR_physics_rigid_bodies (draft)           ▼
                + KHR_implicit_shapes (draft)      perception & dynamics
                + RWM_state_series (custom)          models (PyTorch)
                + semantics in extras                        │
                        ▲                                    ▼
                        └──── inference re-emits valid GLB ──┘
```

- 10 milestones, every one gated by **independent adversarial verification**
- 12,000+ generated episodes, 150 real Physion trials converted
- **Every emitted GLB passes the pinned Khronos glTF-Validator with 0 errors**
- Stock viewers (three.js, Babylon sandbox) play every episode — extensions ride along additively

---

## What glTF got RIGHT for this use (5 positive findings)

1. **Accessor/bufferView machinery is a general typed time-series transport** — our custom state channels reuse it unchanged; zero schema failures across 10k+ episodes
2. **Additive extension model works**: `extensionsUsed` (never `Required`) keeps every file loadable by tools that know nothing about physics or state
3. **Single-file GLB episodes**: atomic, diffable, validator-checkable artifacts — GT and model predictions are directly comparable documents
4. **STEP animation sampling** honestly represents sampled simulator states
5. **Fixed units & coordinate conventions** (meters, seconds, Y-up RH) eliminated a whole class of ambiguity

---

## Gap Part A — Core glTF has no concept of dynamic state

| Gap | What's missing | Our workaround |
|---|---|---|
| G1 | Velocity, action, uncertainty, joint state — **any** non-pose per-frame quantity | `RWM_state_series` (custom): named channels over ordinary accessors sharing the animation's time accessor |
| G2 | Physics initial conditions (mass, friction, colliders) | draft KHR extensions (next slide) |
| G3 | Channels wider than VEC4 | documented chunking convention |
| G6 | Uncertainty representation **and semantics** | diagonal-variance channel + a measured warning (slide 8) |

**Key point:** ratifying the physics extensions does **not** close G1 — a time-series extension is a *separate, complementary* need.

---

## Gap Part B — Implementing against the draft KHR physics extensions

We implemented `KHR_physics_rigid_bodies` + `KHR_implicit_shapes` (pinned draft commit) for real scenes, including articulated doors/drawers via limit-composed hinge/slider joints. Real gaps hit:

| Gap | Finding | Cost we paid |
|---|---|---|
| G7 | **No collider local offset/center** in KHR_implicit_shapes | forced a pivot-child-node design; unavoidable mesh-pivot vs collider-center error in real-asset conversion |
| G8 | Limit damping is soft-stop only — **no viscous joint damping** | articulated scenes need side-channel metadata; MJCF/URDF have this natively |

---

## Gap Part B (continued)

| Gap | Finding | Cost we paid |
|---|---|---|
| G9 | Drives model persistent spring-to-target — **no bounded-duration push** | scripted actuation not encodable as a KHR drive at all |
| G10 | **No fixed/weld joint** | handle attachment is derived, not constrained |
| G11 | Single friction model vs static+dynamic pairs | measured information loss converting Physion (e.g. 1.0 vs 0.1 collapsed) |

---

## Gap Part C — Real-world conversion evidence (Physion)

Converted 150 trials of a real NeurIPS physics benchmark (ThreeDWorld HDF5) into the transport:

- 14 documented impedance mismatches: missing normals, mesh pivot conventions, camera matrix chirality, no ground-plane object concept, dropped collision events, friction collapse…
- **Result: 150/150 validator-clean GLBs; poses/velocities bit-exact round-trip**
- State-based label reconstruction: **92% agreement** with the benchmark's own labels — the transport carries enough state to reproduce a published benchmark's ground truth
- Honest negative: our small dynamics model does **not** transfer zero-shot (chance level) — the transport works; transfer learning is future work

---

## A measured warning about uncertainty channels (G6)

Closed-loop experiment, 3 arms: oracle state / oracle + i.i.d. noise matched to measured perception error / real perception in the loop.

| arm | pos error @ h=99 |
|---|---|
| oracle state | 0.36 m |
| oracle + **i.i.d.** noise | **27.6 m** |
| **real** perception loop | **1.62 m** |

Real detector errors are **frame-correlated** (lag-1 autocorrelation 0.55–0.82) and largely cancel in finite-difference velocity estimation. An i.i.d.-calibrated uncertainty channel **overestimates closed-loop degradation 17×** — any future uncertainty extension should carry or at least warn about temporal correlation.

---

## Recommendations (ranked by measured impact × blocking severity)

1. **Ratify a KHR time-series/state extension** alongside (not inside) the physics extensions — the `{node|joint|scene}`-target + shared-time-accessor + named-channel pattern is validated here at 10k-episode scale
2. **Collider local offset in KHR_implicit_shapes** — narrowest fix, outsized payoff
3. **Joint viscous damping + bounded drive mode** — parity with MJCF/URDF for ordinary actuation
4. **Fixed/weld joint type**

---

## Recommendations (continued)

5. **Optional mesh/convex-hull collider** — the cost of its absence is measured in our real-asset conversion
6. **Best-practice guidance: per-frame uncertainty ≠ i.i.d. noise license** (measured 17× cost)
7. Second friction coefficient; root-level gravity

Explicitly *not* recommended: video-frame sequences in glTF; widening accessors past VEC4 — conventions suffice.

---

## External validity: we tested ourselves

Our own verification protocol only proves we built what we said we built —
so we ran two experiments designed to check the claims *from the outside*:

- **Blind spec-only reimplementation.** Zero source access — only
  `RWM_EXTENSIONS.md` + schemas + GLBs — decoded a whole episode
  **bitwise-identically**. It had to guess 6 conventions our docs left
  implicit; one was initially *wrong* (silently-wrong shapes). All 6 are
  now normative.
- **Clean-room reproduction from the public clone.** A fresh `git clone` +
  documented setup reproduced our smoke-test pass/skip counts and split
  sizes **digit-for-digit**, and seeded dataset generation **bit-identical**
  across machines.

**This is exactly what a ratification process exists to surface** —
ambiguities an author blind to their own tacit assumptions can't see in
their own writing. Two cheap experiments found 6 normative gaps in a spec
we thought was complete. That's the argument for taking the
`RWM_state_series` pattern to a real KHR track rather than shipping it as
one repo's permanent custom extension.

---

## Everything is verifiable

- **Public repo**: github.com/rudybear/glTFWorldModel (MIT)
- `docs/GAP_REPORT.md` — all 20 gaps + 5 positives, each with code pointer, measurement, JSON exhibit, prior-art comparison (UsdPhysics / URDF / MJCF)
- `docs/VERIFICATION.md` — every claim re-runnable: exact commands + expected outputs
- Every milestone was gated by **independent adversarial verification** (the verifier famously caught our own overclaims — the corrections are visible in the docs, per project policy)
- Sample episodes: drag any generated GLB into a stock viewer — it plays

**Ask:** feedback on recommendation #1 (time-series extension scoping) and #2 (collider offset) — we'd contribute the RWM_state_series experience to a KHR-track effort.

---

# Thank you

**github.com/rudybear/glTFWorldModel**

Questions — or better: clone it and run `docs/VERIFICATION.md` against us.
