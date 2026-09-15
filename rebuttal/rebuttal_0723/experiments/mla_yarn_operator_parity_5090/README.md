# MLA shared-index YaRN operator parity

Status: `OFFLINE_CODE_COMPLETE / REMOTE_PREFLIGHT_PENDING`

This package is the implementation companion to
`../MLA_YARN_OPERATOR_PARITY_5090_PLAN.md`.

Implemented and locally checked:

- frozen four-arm/three-seed protocol and thresholds;
- CPU-only fresh-anchor generation;
- hard-link reuse of the verified training and validation tensors;
- proof that every new 32K window is disjoint from all previously observed
  selection/test windows;
- source/tensor/anchor identity manifest;
- shared-index YaRN helper with native formula parity and common masks;
- two-level READY receipts and RTX 5090 discarded probes;
- four-arm seed-42 training gate;
- `raw + shared_index_full` at 200M and all six operators at 300M;
- 8/4/2/1 evaluation micro-batches at 4K/8K/16K/32K, with independent
  per-window NLL and a constant 32K-token upper bound per forward;
- fail-closed test/seed access before a PASS gate;
- three-seed test summary with seed-level intervals;
- proof-backed checkpoint deletion and terminal compile-cache cleanup.
- compact artifact-only status snapshots, five-minute sleep-based monitoring,
  and automatic PASS/STOP Markdown reports with receipt hashes.

CPU-only preparation, after the completed scarcity data manifest is available:

```bash
PYTHONPATH=. python \
  rebuttal/rebuttal_0723/experiments/mla_yarn_operator_parity_5090/prepare.py \
  --source-manifest /path/to/completed-mla-scarcity/data_manifest.json \
  --output-dir /path/to/mla-yarn-operator-parity/data
```

The authoritative launcher is:

```bash
bash rebuttal/rebuttal_0723/experiments/mla_yarn_operator_parity_5090/run_5090.sh preflight
bash rebuttal/rebuttal_0723/experiments/mla_yarn_operator_parity_5090/run_5090.sh gate
# Run only when operator_parity_gate.json says PASS:
bash rebuttal/rebuttal_0723/experiments/mla_yarn_operator_parity_5090/run_5090.sh confirm
```

`preflight` is CPU-only and creates both the underlying training READY receipt
and the operator-parity READY receipt. `gate` trains only four seed-42 arms and
stops after approximately 49 training minutes if any registered criterion
fails. `confirm` is inaccessible until PASS and then trains seeds 43/88.

No target-server data manifest or READY receipt has been generated yet. Do not
open a paid GPU until `preflight` passes on the actual data disk and its receipt
hashes, free-space budget and exact launch command have been inspected.

Low-output monitoring:

```bash
# One compact artifact snapshot:
bash rebuttal/rebuttal_0723/experiments/mla_yarn_operator_parity_5090/run_5090.sh status

# One snapshot every 300 seconds; exits at STOP or final summary:
MONITOR_INTERVAL_SECONDS=300 \
  bash rebuttal/rebuttal_0723/experiments/mla_yarn_operator_parity_5090/run_5090.sh monitor
```

The monitor deliberately does not stream the training log. It reports the last
recorded step/loss/throughput/ETA and sleeps between checks. Artifact status
does not prove a live PID or GPU state; those are checked separately once after
launch. At terminal STOP or completion, the launcher writes
`MLA_YARN_OPERATOR_PARITY_REPORT.md` plus a hash receipt automatically.
