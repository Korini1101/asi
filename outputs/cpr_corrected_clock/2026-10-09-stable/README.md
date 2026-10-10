# Corrected-clock CPR result

All 25 fresh-process shards completed, and strict shard loading, source/runtime/
dataset matching, aggregation and CPR resource receipts passed the independent
audit. The fixed primary gate is **inconclusive**, and CPR is **not selected**: it
beats off while losing to both stronger partial-reset controls.

| Arm | Mean online accuracy | CPR minus arm |
| --- | ---: | ---: |
| cpr_hard_reset | 0.844040 | +0.011815 |
| cpr_ipmnist | 0.855855 | +0.000000 |
| cpr_l2_init | 0.860235 | -0.004380 |
| cpr_off | 0.851100 | +0.004755 |
| cpr_utility_free | 0.860695 | -0.004840 |

This measures five development seeds and eight 5000-example tasks. It does not
establish a paper result, scientific evidence or a whole-agent improvement.
The utility-weighting mechanism adds no demonstrated benefit over the simpler
matched controls at this configuration. The hard-reset arm remains descriptive.
`audit.json` retains every seed, paired difference and standard error. No metric,
hyperparameter or decision threshold was changed after observing results.

Each arm records 40000 observations/updates and 80000 logical model queries;
the complete panel uses one million updates and two million model queries.
All controls carry the registered state envelope; persistent numeric bytes and
wall time are retained in the original shards/derived receipts. Timing is only
telemetry. “Without resets” in the plan means no task-boundary reinitialization:
the registered periodic hard/partial-reset mechanisms execute as specified.

The source commit is `dac2bceb`, with complete source, runtime and canonical
OpenML dataset bindings inside every shard. `PLAN.md` and `run.py` were committed before dispatch. The earlier invalid roster and its logs remain in
the sibling directory. That attempt ran two 40000-update learners, saved only
one control shard, and failed source-integrity checks; it is excluded here.
No consumed root from that invalid reserved roster was reused.

`merged.json` is the existing strict merger's output. The original auditor
incorrectly passed a validated mapping where the receipt API requires a typed
`ScreeningRunResult`; its failure is retained in `audit-first.log`. `audit_v2.py`
constructs that type explicitly and recomputes the existing merge without
rewriting it. This was an audit-adapter correction, not a learner rerun or
change to raw measurements. `verify.py` repeats the audit read-only and compares
the recomputed result to the retained audit.
