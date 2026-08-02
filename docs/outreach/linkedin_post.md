# LinkedIn post (short form)

---

Can glTF be the transport format for world models and robotics — the way USD is becoming for simulation?

We built the experiment to find out: a complete open-source pipeline where glTF 2.0 GLB is the interchange at *every* hop — MuJoCo episodes → GLB (pose animation + draft KHR physics extensions + a custom time-series extension) → renderer → trained perception & dynamics models → inference re-emitting valid GLB, closed loop.

What we learned, the honest version:

✅ glTF's accessor machinery is genuinely reusable as a time-series transport — 10,000+ episodes, every file validator-clean, playable in any stock viewer
✅ Learned dynamics over the transport: 42–176× better than ballistic at long horizons; closed visual loop 34× better
❌ glTF cannot express *any* non-pose dynamic state (velocity, actions, joint angles, uncertainty) — we had to invent an extension
❌ The draft KHR physics extensions are close but miss viscous damping, bounded drives, weld joints, and collider offsets
📏 A finding for anyone doing perception-in-the-loop: detector errors are frame-correlated (autocorr 0.55–0.82) — i.i.d. noise models overestimate closed-loop degradation 17×

Full gap report — 20 evidence-backed findings, 5 things glTF got right, and ranked extension recommendations, every claim traceable to code and measurements:

🔗 github.com/rudybear/glTFWorldModel

Feedback welcome, especially from the Khronos 3D Formats and robotics simulation communities.

#glTF #WorldModels #Robotics #Khronos #3D #SimulationAI #OpenSource

---

*(Character count ~1,600 — within LinkedIn's optimal range. Emojis optional per your taste.)*
