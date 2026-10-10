# Native suite and three-layer diagnostic qualification

Before dispatch: use the existing four frozen development roots 15780–15783
for all four native protocols, 64 examples per task, replay capacity 16, and all
four registered arms. These are consumed development roots, never scientific
seeds. Compare each protocol separately; no cross-suite winner is meaningful.
Keep every negative result. A replay candidate needs positive mean accuracy
minus matched online SGD; this is descriptive development selection only.
Each seed runs in a fresh process and has a second fresh-process replay whose
non-timing result fields must match exactly. The bounded native transforms and
linear learners are not Avalanche parity or full IPMNIST 200x5000 results.

Also run the current v2 three-hidden-layer plasticity diagnostic, with its
registered bounded-development profile and four consumed roots 15830–15833.
These runs qualify the corrected topology and resource receipts on real MNIST;
they cannot reproduce an 800-task paper or establish a plasticity benefit.
SGD and its mechanism-off reduction must remain identical. Labels stay fixed;
this is input-permuted MNIST, not random-label MNIST. ImageNet and RL expansion
remain outside this bounded diagnostic qualification.

Read only the existing local canonical training archives. Verify MNIST SHA-256
against tensorflow/datasets@24211da0508a86f0a9efd82c69bab9d25cda0a20 and CIFAR-100
binary MD5 against the official Toronto page; bind SHA-256 independently.
Parse the original IDX/binary bytes, check headers, counts and labels, then
normalize float32 pixels by float32(255). No test split, imported model or
pretraining is used. Archive identities and derived array hashes are retained.

One process at a time, four CPU cores, 300 seconds per invocation. Native
training observations are 4 seeds x (5+5+20+200) tasks x 64 examples x 4 arms x
2 executions = 471040; the diagnostic adds 4 x 8 x 64 x 3 = 6144 updates.
Actual model queries, updates and numeric persistent/replay storage are reported
by each runner. Timing is telemetry only. Every result/error is retained at a
new path. Run failures are not retried. No scientific promotion or reference
configuration change is allowed; hashes establish consistency, not attestation.
