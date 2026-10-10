# Native suite and three-layer plasticity qualification

All four native seed panels completed and independently replayed with exactly
matching non-timing result fields. All four current-v2 plasticity records passed
the typed validator, including the three-layer replacement accounting and exact
SGD/mechanism-off reduction. Every training dataset was reconstructed directly
from its checksum-verified original training archive; no test split was used.

| Native protocol | Replay minus online SGD | Development disposition |
| --- | ---: | --- |
| ipmnist | -0.01214844 | development_rejection |
| rotated_mnist | -0.01875000 | development_rejection |
| split_cifar100 | +0.00390625 | development_candidate |
| split_mnist | -0.02656250 | development_rejection |

These are separate protocol-level outcomes. The positive Split CIFAR-100 mean
is descriptive development selection, not a statistically established benefit,
full-benchmark result, cross-suite winner, or reference-agent improvement.
The panel uses native linear models, 64 examples per task, and replay capacity
16. Rotations, class order, transforms, task geometry and CVDF normalization are
explicit native choices, not external Avalanche parity.

The diagnostic has three hidden layers of width 64, eight tasks and 64 examples
per task. Its matched arms consume fixed-label input-permuted MNIST. These
current-v2 records replace no historical v1 record and do not reproduce the
800-task paper, random-label ImageNet, or the paper's RL experiments. Those
costly extensions are not justified by this qualification and are not claimed
complete. This closes the useful bounded MNIST implementation/diagnostic slice.

`PLAN.md` predates dispatch; `run.py` contains the complete bounded driver,
archive parsing/checks and create-only publication. `dispatch.json` records all
12 process exits. `summary.json` binds the result files, retains every per-seed
accuracy and paired delta, and records exact replay acceptance. Each result
contains source/runtime/dependency identities, data and schedule hashes, model
queries, updates, numeric persistent/replay bytes, and telemetry-only timing.
`verify.py` recomputes the typed validation, replay checks and summary read-only.
The original `audit.py` produces a new summary and refuses to overwrite one.

Dataset authorities checked on 2026-10-09:

- [Pinned TFDS MNIST checksums](https://raw.githubusercontent.com/tensorflow/datasets/24211da0508a86f0a9efd82c69bab9d25cda0a20/tensorflow_datasets/url_checksums/mnist.txt).
- [Official CIFAR download and binary format](https://cave.cs.toronto.edu/kriz/cifar.html). The local 168513733-byte CIFAR archive matches the published MD5 and independently bound SHA-256.

Data arrays remain outside Git; the driver records and verifies the exact
original archive hashes and checks dimensions, class counts and label ranges.
Timing and consistency hashes do not authenticate execution or establish
scientific evidence. No imported pretraining, model promotion or reference-dev
change occurred.
