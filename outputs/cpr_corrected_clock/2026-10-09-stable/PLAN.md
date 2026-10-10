# Corrected-clock CPR development comparison

This prospective screen asks whether utility-weighted partial resets add value
beyond matched off, utility-free pull, L2-to-initialization and hard-reset controls.
The older local eight-task record uses the superseded post-update reset clock;
its numbers are not the baseline for this experiment.

Five new development roots 15639100–15639104, five registered cpr_* arms,
eight tasks, 5000 examples per task, and the existing 784–300–150–10 network:
25 fresh-process shards, one million training observations and updates total.
The canonical OpenML MNIST first 60000 training rows and existing step schedule
are identical within each seed. Registered hyperparameters remain unchanged.
All learners see current examples and labels, without task IDs or resets.

Hypothesis: gradient-utility weighting protects useful parameters during a
partial reset, improving pre-update online accuracy. Additional utility state,
queries, persistent numeric bytes, and wall-time telemetry are recorded in each
existing strict CPR receipt. Timing is not a selection metric.

The primary question is incremental utility-weighting benefit: CPR must exceed
each of off, utility-free and L2 by a positive paired mean, with all five paired
differences positive against each control. Otherwise the mechanism is not
selected: all means nonpositive rejects; mixed outcomes are inconclusive.
Hard reset is descriptive. Report all arm means, individual paired differences,
and standard errors. No tuning or rescue by selecting another metric.

The scripts run through the existing CLI, with source/runtime/dataset receipts,
new create-only output files, explicit CPU backend, four CPU cores, and one
worker. Each shard has a 300-second wall limit; the total dispatch limit is
7500 seconds. This is a bounded development screen, not scientific evidence,
paper parity, authenticated execution proof, or a reference-dev update.
All errors/logs and negative outcomes remain in this new namespace. A failed
shard is not retried; existing shard/log names prevent accidental overwrite.
No prior consumed campaign roster is reused. A read-only search of available
repository records found no occurrences of these five roots before this plan.

This new roster follows the retained invalid source-drift attempt in the sibling
2026-10-09 directory. All helper Python files are committed before dispatch;
no checkout, source or index changes occur until the panel has finished.
