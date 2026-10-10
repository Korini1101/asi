# Retained COOM runtime qualification

The reviewed COOM image built successfully and executed all eight official CO8
scenarios for two fixed-action steps each. Its strict in-image validator accepted
`smoke.json`. The trace SHA-256 is
`c74968494ccebaaeac4bc1e0c0f1db7546ac5091b831c05a4c0c727266da696f`, matching the
previously documented golden and the separately retained overlay invocation.

`manifest.json` binds the exact source commit, runtime files, built image ID,
receipts, build/run/validation logs, and caller resource limits. Both executions
used network isolation, a read-only root, UID/GID 65532, no capabilities,
no-new-privileges, two CPUs, 2 GiB memory, 64 PIDs, a 16 MiB no-execute tmpfs,
and a 120-second host timeout. The output was a new directory outside the tree.

The first invocation reused the locally available historical image and mounted
the current `smoke.py` and `qualification-manifest.json` read-only over their
original paths. Its Dockerfile, dependency lock and import patch were checked
byte-for-byte against this tree first. The second invocation used the newly
built image without source overlays. All receipts are retained unchanged.
The in-image verifier checks installed distributions, upstream source, license,
33 assets, sandbox process properties, task order and complete trace schema.

Reproduction follows `external_runtimes/coom/README.md`, launching the exact
image ID from the manifest, with a fresh output path. The build command was:

```bash
docker build --tag asi-coom-qualification:issue-1582-20261009 external_runtimes/coom
```

This completes the fixed-action environment/runtime qualification requested in
#1582. It does not implement a SAC learning experiment, evaluate forgetting or
forward transfer, establish COOM paper parity, or demonstrate an ASI performance
improvement. There are 16 environment steps and zero learner updates per smoke.
Timing is telemetry only; hashes bind consistency rather than authenticated
execution. Subsequent learning comparisons need their own matched controls,
resource plan and new output namespace.
