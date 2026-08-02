# LinkedIn post (short form)

---

Could glTF — the format your 3D viewer already speaks — carry a full world-model pipeline? We put it to the test, and honestly: glTF surprised us with how far it goes. 🚀

We built a complete open-source pipeline with glTF 2.0 GLB as the interchange at *every* hop: MuJoCo physics episodes → GLB → renderer → trained perception & dynamics models → inference re-emitting valid GLB, closed loop. Delivered end-to-end, from empty repo to working models, in a matter of days.

The best part: where core glTF didn't yet have a vocabulary for dynamic world state, its extension mechanism was flexible enough that we simply added one — an experimental `RWM_state_series` extension carrying velocities, actions, joint states and uncertainty over glTF's own accessor machinery. 10,000+ episodes later: every file validator-clean, every episode still playable in any stock glTF viewer, extensions riding along additively. The draft Khronos physics extensions (rigid bodies, collision shapes, joints) slotted right in for mass, friction, colliders — including articulated doors and drawers.

Some numbers from the closed loop: learned dynamics 42–176× better than a ballistic baseline at long horizons; the full perceive→predict→re-render loop 34× better — plus a fun measured insight: real detector errors are frame-correlated, so naive i.i.d. noise models overestimate closed-loop degradation 17×.

We documented everything — including where extensions had to fill in — as an evidence-backed gap report with ranked recommendations for the glTF ecosystem.

Next step: generalizing the experimental RWM extensions beyond our pipeline — toward a reusable time-series/state vocabulary for world models, robotics, and digital twins on glTF.

🔗 github.com/rudybear/glTFWorldModel (MIT)

Feedback very welcome — especially from the Khronos 3D Formats and robotics simulation communities.

#glTF #WorldModels #Robotics #Khronos #3D #SimulationAI #OpenSource #DigitalTwin

---

*(~1,900 characters. Tone: glTF-positive — flexibility and interop as the headline, gaps framed as extension opportunities with a constructive next step.)*
