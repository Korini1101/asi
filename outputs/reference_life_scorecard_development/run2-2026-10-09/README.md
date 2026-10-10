# Hosted reference-life development scorecard, 2026-10-09

Run: [38023231911](https://github.com/SlopDotCash/asi/actions/runs/38023231911).
Launch source: `30c6f016d864ba4017ecd9b43e83fb5476098463`.
Plan: `4c4396848da3abebc53fb61c7b241866566ad2b44fa83ee3395a5a389a869b86`.

This directory retains the original hosted plan, all 144 fresh-process shards,
aggregate and run receipt unchanged. The literal-frozen development roster is
12 previously consumed seeds (70000–70011), two environments and six arms.
SwitchingTwoState has 4,000 events per life; RiverSwim has 20,000. A complete
run therefore accounts for 1,728,000 accepted environment events. Every shard
uses the locked Python 3.12.12 runtime and one matching source identity.

The frozen result is **`valid_baseline_failure`**. All 144 lives completed,
with zero execution failures and zero parameter-change violations. RiverSwim
has no qualifying SARSA-family control: its discounted-SARSA normalized paired
lower bound is 0.0805083 and differential-SARSA lower bound is 0.00170768,
both below the required strict 0.1 threshold. Candidate selection is therefore
`not_evaluated_baseline_failure`; no configuration is selected or promoted.

Descriptive mean lifetime rewards, each over the same 12 development seeds:

| Arm | SwitchingTwoState (4,000 events) | RiverSwim (20,000 events) |
| --- | ---: | ---: |
| `prototype` | 3006.916667 | 5463.068749 |
| `prototype_frozen` | 1996.666667 | 1080.111249 |
| `random` | 2006.916667 | 60.482916 |
| `privileged_oracle` | 4000.000000 | 17094.666667 |
| `differential_sarsa` | 3277.083333 | 96.605831 |
| `sarsa` | 2932.750000 | 6304.942499 |

These are separate environment-specific returns, not a pooled score. The
SwitchingTwoState controls qualify, but Prototype misses the frozen
noninferiority margin against differential SARSA (paired lower bound
-0.158223 versus the -0.05 margin). RiverSwim Prototype also misses the
Prototype improvement-over-frozen threshold (0.0467919 versus 0.05). These are
development heuristics on consumed seeds, not scientific inference.

The next useful question is a separately declared, bounded control-calibration
investigation and a matched candidate/control bottleneck study. Do not weaken
this run’s thresholds, reinterpret the privileged arm as a candidate, or
promote this result. Resource Pareto selection remains unevaluated because
latency is telemetry only.

## Independent validation

`verify.py` checks the exact launch commit, Git tree, workflow blob, lockfile,
complete shard roster, and all 146 receipt inventory entries (size and SHA256).
It strictly validates the original aggregate against the current source/runtime,
then independently reconstructs it from the original 144 shard files and
requires exact equality. `independent-validation.json` records the result.

Run from the repository root with its locked Python 3.12.12 project environment:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD" .venv/bin/python \
  outputs/reference_life_scorecard_development/run2-2026-10-09/verify.py \
  outputs/reference_life_scorecard_development/run2-2026-10-09
```

Leave `JAX_PLATFORMS` unset to match the bound runtime identity. Validation is
read-only and performs no new benchmark or seed consumption. Current-source
validation intentionally fails after covered source/runtime/dependency drift;
that does not authorize rewriting these historical records.

## Scope

This is permanently nonpromoting development selection on consumed development
seeds. It does not populate `reference-dev`, qualify timing, claim robotics
readiness, or establish performance/scientific evidence. The privileged
finite-horizon dynamic-programming arm normalizes utility and is excluded from
candidate Pareto comparisons. Timing is telemetry only; numeric payload counts
are not hardware peak-memory measurements. Receipt and artifact hashes establish
consistency, not authenticated execution attestation. The prior `run1` result and
all pinned evidence remain unchanged.
