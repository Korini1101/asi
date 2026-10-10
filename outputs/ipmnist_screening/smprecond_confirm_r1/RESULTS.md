# smprecond_confirm_r1 — 200-task confirmation results (2026-09-02)

Development confirmation; permanently nonpromoting. Measurement commit
`03a574bb` (source bytes identical to the screen's `0bf003c8`; the
intervening commit is outputs-only). 40 shards (2 arms x seeds 0-19,
200 tasks x 5000 steps, OMP_NUM_THREADS=1), merged at the same commit.
Summary: `summary.json`.

## Null check — PASSED

Remeasured `rls_head_resid_identmap50_r` 20-seed mean 0.9165689 vs the
standing 0.91657 (identmap_star_confirm_r1): |diff| = 1.1e-6 <= 0.001.

## Verdict — FAILED CONFIRMATION (valid rejection at the frozen bar)

| arm | 20-seed mean | paired mean vs control | seeds positive | bar |
|---|---|---|---|---|
| `rls_head_resid_sm3e4_i50r` | 0.9176932 | **+0.0011243** (se 0.000109) | 20/20 (min +0.000489, max +0.002063) | MISSES +0.002 |

- The whole-life development difference is positive on every seed and too small:
  the preregistered CONFIRMED-WIN bar required paired mean >= +0.002.
- 60t -> 200t decay factor: 0.005236 / 0.0011243 = **4.66x** — steeper
  than negative result #21's 3.5x. The preregistered prediction that a
  within-task-convergence mechanism should decay LESS than an early-life
  transient mechanism is refuted; the gain concentrates in early life
  here too (consistent with the body's features maturing over life, after
  which per-weight step adaptation has little left to speed up).
- Within the same 200-task records, the paired mean is +0.00466583
  over tasks 0-59, then -0.00039343 over tasks 60-199 (only 5/20 seeds
  positive in the latter window). The whole-life positive average therefore
  hides a late-life reversal. Candidate late-window slope is -1.25e-4 versus
  control -1.03e-4; these data do not support a no-degradation claim.
- These observations reject this candidate at the preregistered +0.002
  development bar. They do not establish that other step-geometry methods
  cannot improve late-life performance. The summary's generic 0.005
  `confirmation_threshold` is not the preregistered decision rule here.


## Disposition (preregistered)

No new standing best — 0.91657 (`rls_head_resid_identmap50_r`) stands.
Both smprecond arms are deregistered; the mechanism implementation and
its bit-exact reduction pins are retained (lane precedent, negative
results #19-#21). Ledger entry #23 records the closure. No retuning: a
smaller step size or a boundary-reset variant is not licensed as a rescue
of this round.

## Historical validation limit

The candidate arms were deregistered after this failed development gate.
The current strict shard loader therefore rejects their names; it must not be
weakened to accept them. The paired arithmetic above was recomputed directly
from all 40 retained task curves. Strict historical revalidation requires the
recorded generating source and runtime, which this arithmetic check does not
replace. The recorded source digests are consistency metadata, not execution
attestation.
