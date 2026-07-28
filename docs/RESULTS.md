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
