# Recorded Results

## V5 — Dynamics model (2026-07-28)

### Training summary

Two models trained on `data/dynamics-v1` (10,000 episodes, 90/5/5 train/val/test split):

1. **InteractionTransformer**: 4,815,113 parameters
   - Training: two-phase (40k teacher-forced steps + 10k rollout-finetune with K annealing 2→8)
   - Walltime: 9.6 minutes on RTX PRO 6000 Blackwell
   - Best validation loss: 0.015773 at step 23,000

2. **NoInteractionMLP**: 75,529 parameters
   - Training: same schedule, two-phase
   - Best validation loss: 0.016686 at step 37,000

Evaluation performed on test split (476 episodes).

### Position error (m)

| model | h=1 | h=5 | h=10 | h=30 | h=99 |
| --- | --- | --- | --- | --- | --- |
| model(transformer) | 0.0049 [0.0046, 0.0052] | 0.0215 [0.0177, 0.0243] | 0.0388 [0.0281, 0.0492] | 0.1063 [0.0573, 0.2206] | 0.3135 [0.1774, 0.5769] |
| ballistic | 0.0053 [0.0053, 0.0053] | 0.0267 [0.0267, 0.0267] | 0.0534 [0.0534, 0.1181] | 4.4796 [4.1577, 4.8743] | 55.3315 [54.3696, 56.3102] |
| mlp(mlp) | 0.0052 [0.0049, 0.0055] | 0.0249 [0.0211, 0.0284] | 0.0471 [0.0353, 0.0586] | 0.0908 [0.0504, 0.2034] | 0.2928 [0.1288, 0.5350] |

### Rotation geodesic error (rad)

| model | h=1 | h=5 | h=10 | h=30 | h=99 |
| --- | --- | --- | --- | --- | --- |
| model(transformer) | 0.0087 [0.0062, 0.0117] | 0.0428 [0.0292, 0.0576] | 0.0978 [0.0646, 0.1408] | 0.6156 [0.2919, 1.3949] | 1.3799 [0.6334, 1.9821] |
| ballistic | 0.0495 [0.0252, 0.0772] | 0.2477 [0.1256, 0.3863] | 0.5255 [0.2799, 0.7821] | 1.4250 [0.8618, 2.1181] | 1.4191 [0.8378, 2.1918] |
| mlp(mlp) | 0.0039 [0.0027, 0.0054] | 0.0232 [0.0159, 0.0321] | 0.0657 [0.0443, 0.1028] | 0.5372 [0.2815, 1.2189] | 1.4973 [0.5965, 2.1696] |

### Velocity error (m/s)

| model | h=1 | h=5 | h=10 | h=30 | h=99 |
| --- | --- | --- | --- | --- | --- |
| model(transformer) | 0.0152 [0.0096, 0.0234] | 0.0655 [0.0431, 0.0985] | 0.1868 [0.1152, 0.4012] | 0.1179 [0.0626, 0.2802] | 0.1042 [0.0578, 0.2555] |
| ballistic | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 3.1260] | 10.0151 [9.7438, 10.3007] | 33.0191 [32.7289, 33.3005] |
| mlp(mlp) | 0.0118 [0.0074, 0.0181] | 0.0518 [0.0346, 0.0860] | 0.1399 [0.0804, 0.3927] | 0.1047 [0.0369, 0.2480] | 0.0718 [0.0116, 0.2545] |

### Key results

**Acceptance bar met**: transformer beats ballistic at every horizon — 1.4× at h=10, 42× at h=30, 176× at h=99.

**Honest finding**: the no-interaction MLP is competitive. Transformer wins at h=1/5/10, but MLP is slightly better at h=30/99 medians (position: 0.0908 vs 0.1063 at h=30; 0.2928 vs 0.3135 at h=99), with overlapping IQRs across all horizons. Interpretation: in `wm-scenes-v1`, most objects fall and settle independently, so cross-object interaction modeling has limited long-horizon payoff at this scene density. A denser-interaction scene distribution is the natural follow-up to verify that transformer's extra modeling capacity pays off when interactions truly matter.

### Artifacts

Eval artifacts (metrics.json, divergence_curve.png, pred/gt GLBs, videos) live in `runs/dynamics-v1/eval/` (git-ignored). To regenerate, see the training and eval commands in [docs/VERIFICATION.md](docs/VERIFICATION.md) V5 section ("full training run command" and "eval CLI demo").

## V6 — Perception (final, 2026-07-30)

The V6 milestone encountered a fundamental memorization failure on the initial 500-episode `perception-v1` dataset: training loss fell to 0.19-0.21 while validation loss climbed to 8.87 and matched position error flatlined at 0.6-0.67m (mean-predictor level), indicating the model had learned scene identity rather than image-to-geometry mapping. Root cause: 45,800 train frames at 128 batch size over 25k steps = 69.9× epoch-equivalent, catastrophically over-training an 8.2M-parameter model. The recovery arc was: regenerate `perception-v1` at 4,000 episodes (363k train frames, reducing to 8.9× epoch-equivalent), add a dataset-scale training guard (refusing to start if configured steps exceed 15× epoch-equivalent), identify and filter out-of-box GT objects that fell outside the finite camera frustum and workspace bounds (4.5% of train/val/test GT positions), and swap the patch-based ViT encoder for a stride-2 CNN encoder (6.5M parameters) that provides convolutional inductive bias — a textbook fix for small-data-regime transformer under-generalization. The CNN encoder's `--smoke-val` at 5k steps (1.8 epoch-equivalents) showed 54.8% relative improvement vs. ViT's 7.7%, with final position error 0.2858m vs. ViT's 0.5958m, on the *identical* dataset, schedule, and loss, strongly supporting the data-hunger diagnosis.

The final 40k-step CNN encoder run reached convergence on the larger dataset and is independently verified below.

### Test-split results (perception-v4-cnn-40k)

| model | existence P | existence R | existence F1 | matched pos err (m) | class accuracy |
| --- | --- | --- | --- | --- | --- |
| PerceptionDETR (CNN) | 0.8930 | 0.8483 | 0.8701 | 0.1798 | 0.9496 |
| mean-state baseline | 0.8321 | 0.8060 | 0.8189 | 0.7732 | 0.4311 |

### PerceptionDETR rotation error (deg) by shape

| shape | median | mean | n |
| --- | --- | --- | --- |
| box | 14.0923 | 18.9388 | 23900 |
| cylinder | 21.8975 | 30.8924 | 5879 |

### PerceptionDETR per-N breakdown

| N | n_frames | existence F1 | matched pos err median (m) |
| --- | --- | --- | --- |
| 1 | 2704 | 0.6036 | 0.1030 |
| 2 | 2898 | 0.7524 | 0.1398 |
| 3 | 4250 | 0.8311 | 0.1621 |
| 4 | 3807 | 0.8934 | 0.1989 |
| 5 | 3173 | 0.9626 | 0.2294 |

### Key results & acceptance bar

**Acceptance bar NOT met**: existence F1 0.8701 < 0.95 bar (missed by 0.0799); median matched position error 0.1798 m >> 0.05 m bar (3.6× the target); class accuracy 0.9496 ≈ 0.95 bar (within rounding). What *was* achieved: the CNN encoder generalized where ViT memorized, position signal is real and grounded in image content (mean-state baseline at 0.7732m vs. model at 0.1798m, a 4.3× improvement), rotation signals are robust per-shape (box 14.1°/cylinder 21.9°), and per-object-count stratification shows performance scaling properly (F1 ranges 0.60 → 0.96, position error 0.103 → 0.229 m from N=1 → N=5). The plateaued validation curve at 0.155m around step 40k (confirmed by spot-checks) suggests the 4,000-episode dataset, while larger than the pathological 500-episode v1, remains insufficient for the 0.05m acceptance bar — a finding consistent with V6.1/V6.2's data-hunger diagnostic: perception in dense, visually low-dimensional scenes (fixed camera, 3 shapes, 8 colors) over a large continuous workspace requires more scene diversity than 4,000 episodes to achieve <5cm closed-loop accuracy, at least with a 6-12M parameter model and no domain-specific priors.

### Artifacts

Eval artifacts (metrics.json, metrics.md, pred_frames GLBs) live in `runs/perception-v4-cnn-40k/eval/` (git-ignored). Per-N, per-shape, and per-arm rotation analysis included.

## V7 — Closed loop (2026-07-30)

Closed-loop demo: perceive from frame 0 and 1 (rendering ground-truth RGB, running real `PerceptionDETR` CNN encoder), Hungarian-match detections across frames, finite-difference velocity, roll forward via `InteractionTransformer`, save every arm as real glTF, reload, and score. Three arms separate perception-induced error from dynamics-induced error: **(A) oracle** (exact GT state, no perception), **(B) oracle + measured noise** (exact GT poses perturbed by chi(3)-calibrated Gaussian noise from perception-v4-cnn-40k's measured error statistics, then finite-differenced), and **(C) visual** (real closed loop: rendered frames → PerceptionDETR → Hungarian matching across frames → velocity assembly → rollout).

### Per-arm median position error (m) by horizon

| arm | h=1 | h=5 | h=10 | h=30 | h=60 | h=99 |
| --- | --- | --- | --- | --- | --- | --- |
| A (oracle) | 0.0048 | 0.0199 | 0.0302 | 0.1003 | 0.2171 | 0.3597 |
| B (oracle + noise) | 0.1915 | 0.9371 | 1.7225 | 3.9312 | 7.6881 | 27.5660 |
| C (visual) | 0.4460 | 0.5615 | 0.8344 | 1.1175 | 1.4027 | 1.6232 |
| ballistic | 0.0053 | 0.0267 | 0.0534 | 4.6915 | 20.1479 | 55.5473 |

### Key findings

**(1) Visual closed loop is 34× better than ballistic at h=99**: C_visual 1.62m vs. ballistic 55.5m. The learned dynamics model keeps perceptually-imperfect detections physically plausible — accumulated trajectory divergence is bounded by the model's own learned dynamics constraints, not by the unbounded constant-gravity extrapolation ballistic uses.

**(2) i.i.d.-noise arm is 17× worse than the real visual arm at h=99**: B_oracle_noise 27.6m vs. C_visual 1.62m. Detector errors are frame-correlated, not independent: lag-1 autocorrelation measured 0.55–0.82 across position/rotation. When a detector makes a biased error on frame 0 (e.g., object center systematically shifted left), it makes a similar biased error on frame 1, causing the errors to *partially cancel* in finite-difference velocity rather than add via the `sqrt(2)` amplification an i.i.d. model assumes. This is a crucial, empirically-measured gap in uncertainty representation: naive per-frame-independent Gaussian noise models dramatically overestimate closed-loop degradation from correlated detector errors. Arm B's result (an upper bound on what uncorrelated measurement noise alone would cause) is not representative of real detector behavior, per this finding — relevant to the gap-report's G6 on uncertainty models.

### Artifacts

Artifacts (metrics.json, attribution.png, per-arm GLBs, videos) live in `runs/closed-loop-v1/` (git-ignored). Attribution curve shows median position error by horizon for all four arms + ballistic reference; `n=20` episodes on test split.

## V8 -- Physion external anchor (2026-07-30)

**Protocol**: option (b) from `docs/PHYSION.md` -- the state-based track. All 150 Collide test trials (the smallest rigid Physion scenario, `Collide_testing_HDF5s.tar.gz`) were converted HDF5 -> real glTF via `gltfworld.physion.convert` (real per-object mesh geometry, real per-frame poses/velocities, `KHR_physics_rigid_bodies`/`KHR_implicit_shapes` sphere/box approximation, `RWM_state_series`, `extras.rwm` role labels), then evaluated for Object Contact Prediction ("does the red/agent object ever touch the yellow/patient zone") three ways via `gltfworld.physion.ocp_eval`:

- **GT-contact oracle**: real per-frame mesh geometry (both objects), nearest-vertex proximity threshold, calibrated on a deterministic 50-trial held-out calibration split, evaluated on the remaining 100 (also reported on the full 150 for reference).
- **Our dynamics rollout**: the first 15 real GT frames mapped into gltfworld's own D=22 tensor contract (`gltfworld.scene.contract.episode_to_tensors`, no new conversion logic -- literally the same function `dynamics-v1`'s own eval uses), rolled forward with `InteractionTransformer` (`runs/dynamics-v1/best.safetensors`, V5's zero-shot-transferred checkpoint, no Physion-specific fine-tuning) to the trial's last recorded frame, contact predicted via the *same* calibrated threshold against a coarse bounding-sphere approximation (mean AABB half-extent).
- **Ballistic control**: identical protocol, `BallisticBaseline` in place of the learned model.

Calibrated threshold: **0.1 m** (grid search over 17 values, maximizing calibration-set oracle accuracy). Sanity check: the HDF5's own per-frame `target_contacting_zone` label (`any()` across the trial) agrees with the independently-packaged PhysionTest-Core archive's `labels.csv` on **150/150** trials (100.0%) -- the two ground-truth sources are consistent, so the whole exercise isn't bounded by a labeling discrepancy before any modeling starts.

### Accuracy (95% Wilson CI)

| track | calibration (n=50) | held-out (n=100) | full Collide test set (n=150) |
| --- | --- | --- | --- |
| GT-contact oracle | 0.940 [0.838, 0.979] | **0.920** [0.850, 0.959] | 0.927 [0.873, 0.959] |
| our dynamics (InteractionTransformer, zero-shot) | -- (not calibrated on) | **0.490** [0.394, 0.587] | 0.527 [0.447, 0.605] |
| ballistic control | -- (not calibrated on) | **0.490** [0.394, 0.587] | 0.527 [0.447, 0.605] |
| chance (label balance) | -- | 0.500 (75 True / 75 False, exactly balanced) | 0.500 |

### Divergence diagnostic (median final-recorded-frame agent-patient distance, full 150)

| track | median | p25 | p75 |
| --- | --- | --- | --- |
| our dynamics | **2.32 m** | 1.77 m | 4.16 m |
| ballistic | 102.54 m | 102.49 m | 102.54 m |

### Key findings

**(1) The GT-contact oracle establishes a strong, non-trivial ceiling: 92.0% held-out.** Full real trajectories + real mesh geometry + a single calibrated proximity threshold reproduces the simulator's own contact label 92% of the time -- confirming the OCP task is well-posed from state alone (not an artifact of a lucky calibration: the calibration-set number, 94.0%, and the full-150 number, 92.7%, both land in the same band). The ~7-8% of trials this ceiling misses are presumably where nearest-*vertex* distance (not true surface-to-surface distance, see `docs/PHYSION.md`'s findings) diverges from the simulator's own internal collision detection, or ambiguous grazing contacts.

**(2) Our dynamics model's zero-shot transfer to Physion collapses to chance -- exactly the outcome DESIGN.md/PHYSION.md's honest feasibility notes anticipated.** 49.0% held-out, indistinguishable from the 50% chance rate given the label's exact 50/50 balance (CI [0.394, 0.587] comfortably straddles 0.5). This is a real, reportable finding, not a bug: `InteractionTransformer` was trained exclusively on `wm-scenes-v1`'s tightly-bounded distribution (1-5 objects, size 0.05-0.25m, |v|<=1.5 m/s, density 300-3000 kg/m^3, DESIGN.md) and Physion's Collide trials sit far outside it (masses up to 500 in the units this milestone's D=22 contract expects, launch speeds ~7.6 m/s, no shared "ground" concept for the model's learned ground-token to attend to at all -- the true floor isn't even a tracked rigid body in Physion's HDF5, see `docs/PHYSION.md` finding 7). A failed transfer with a working, independently-verified transport conversion (stage 2's round-trip tests) is still a successful gap experiment, per this milestone's own framing.

**(3) The ballistic control is numerically identical to our dynamics model in binary accuracy, but the *continuous* divergence tells a different, more informative story.** Both tracks predict "contact" in only 3-4/150 trials (i.e. both are practically a trivial "always predict no-contact" classifier at the calibrated 0.1m threshold, which is why both land within noise of the 50% chance rate). But the underlying rollout trajectories are not remotely similar: our dynamics model's median final-frame agent-patient distance is **2.32m**, ballistic's is **102.54m** -- a ~44x difference, reminiscent of V5's own `wm-scenes-v1` result (ballistic diverges catastrophically past first contact; the learned model stays bounded). Ballistic's distribution is also strikingly *tight* (p25-p75 spans barely 0.05m around 102.5m) -- consistent with unconstrained free-fall diverging to roughly the same large distance regardless of trial specifics, since nothing stops it. The learned model, even completely out of its training distribution, still imposes *some* learned physical structure that keeps rollouts an order of magnitude closer to plausible -- **but not close enough to matter at the ~0.1m scale this particular task's threshold operates at.** This is the clearest evidence in this milestone that raw position-error improvement (the V5 acceptance bar) doesn't automatically transfer into task accuracy on a fine-grained, threshold-sensitive downstream metric under severe domain shift.

**(4) Honest comparison against the published table.** Humans ~71%, particle-based GNS ~71%, DPI ~70%, pixel-based models 55-65% (all cross-scenario averages, vision-based readouts, per `docs/PHYSION.md`). This milestone's numbers are **not directly comparable**: they cover one scenario (Collide, not the 8-scenario average the published table reports), and this track is **state-based** (real mesh geometry + real pose/velocity), not vision-based -- a fundamentally easier input modality than any of the published numbers assume. The GT-contact oracle's 92% (state-based, one scenario) sits *above* every published number precisely because it's not doing the same job -- it's an upper-bound sanity check on the label, not a competing model. The one number that *is* in the same spirit as the published pixel-model row is our-dynamics' 49% -- and it lands well *below* even the weakest published pixel baseline (55%), because it is tested zero-shot on a model that never saw Physion's object/scale/mass distribution during training, whereas every published number in the table (including the pixel-based ones) was trained or fine-tuned on Physion data. This is the honest, load-bearing caveat: the comparison is contextual, not head-to-head.

**(5) Conversion findings pointer.** Fourteen concrete impedance-mismatch findings (missing normals, mesh pivot-vs-center with no KHR offset field, two-friction-coefficients collapse, no cylinder/convex collider, left- vs right-handed chirality, synthesized camera orientation, no ground-plane object, dropped collision-event records, truncated per-object physics metadata for decorative objects, empty meshes for some real-world asset kinds, and a tensor-contract category-vocabulary mismatch) are recorded in `docs/PHYSION.md`'s "Physion conversion findings" section -- the primary gap-report evidence this milestone produces for V9.

### Artifacts

`runs/physion-ocp-v1/metrics.json` (git-ignored) has the full breakdown (per-track accuracy at all three splits, divergence diagnostics, calibration trial ids, the Core-labels.csv sanity check). Converted GLBs live in `data/external/physion/glb/Collide/` (git-ignored, 150 files, regenerated by `gltfworld.physion.ocp_eval.convert_all_trials`, idempotent/resumable). See `docs/VERIFICATION.md`'s V8 section for the exact re-run command.
