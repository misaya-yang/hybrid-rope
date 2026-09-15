# MLA scarce active-frequency experiment

This package tests whether EVQ's internal allocation-shape contribution grows
when the number of **active rotary-frequency pairs** is scarce.

## Terminal result (2026-07-24)

The seed-42 six-arm gate and disjoint test evaluation completed. The registered
shape-only gate returned `PASS`, but the practical native-baseline gate failed:
at K=8/8K the range control collapsed to 6.2195 NLL, EVQ recovered to 4.6437,
and native remained much better at 3.9915. K=32 EVQ was also 0.0382 NLL worse
than native at 8K. Seeds 43/88 were therefore not run.

The aggregate is `../mla_scarcity_seed42_result_20260724.json`; full analysis
and the operator-qualified YaRN diagnostic are in section 8 of
`../EXPERIMENT_REPORT_20260724.md`. Do not rerun `confirm` from the historical
formal `PASS` gate: the later decision receipt supersedes expansion.

## Frozen comparison

- 50.1M-parameter MLA model, `L_train=4096`, base 500K, no passkey mixture.
- Fixed architecture for every run: `d_rope=64`, `d_nope=0`, 32-pair rotary
  capacity, identical parameter count and initialization.
- Active frequency budgets: `K=8` and `K=32`. Inactive K=8 pairs have
  `inv_freq=0`, so their rotation is exactly the identity.
- Schedules: native endpoint Geo, EVQ-range-matched uniform, and EVQ-Cosh.
- 300M tokens; diagnostic snapshots at approximately 100M/200M/300M.
- Seed 42 uses selection anchors as a six-run gate. Seeds 43 and 88 run only
  after a PASS gate. Confirmatory reporting uses disjoint test anchors.
- Primary metric: paired tail NLL at 4K/8K/16K/32K. Windows are repeated
  measurements; seed-level paired contrasts are the inferential unit.

The fixed 64-dimensional architecture is deliberate. Varying `d_rope` in the
legacy MLA implementation also changes which key dimensions use the latent
projection and changes parameter count. Zero-frequency inactive pairs isolate
active spectral budget without that architectural confound.

## Off-GPU preparation

The launcher reuses the existing FineWeb-Edu train/validation tensors through
hard links when possible. It creates new disjoint selection/test anchors and
performs full tensor, prefix, initialization, frequency, and code checks.
It also generates a model-free cosine-kernel diagnostic. That diagnostic
checks the range/shape intervention and records its SHA-256 in READY; it is
explicitly excluded from the success metric and cannot replace GPU evaluation.

```bash
bash rebuttal/rebuttal_0723/experiments/mla_scarcity_5090/run_5090.sh preflight
```

Only after `ready_receipt.json` is `READY` should a paid GPU be enabled.
The first GPU action validates that the PyTorch build contains the running
CUDA architecture, enables BF16 and `torch.compile`, and forces Flash SDPA with
math, memory-efficient, and cuDNN SDPA fallbacks disabled. It then compiles one
discarded step and times five steady-state steps for K=8 and K=32. If the
Flash-only contract is unsupported, the suite stops before training.

The diagnostic currently predicts a favorable K=8 scarcity interaction only
in the registered 1x-to-2x region, then changes sign at 2x-to-4x. This is a
predeclared non-monotonicity risk: 8K remains the primary endpoint, while
16K/32K must still be reported without selection.

## GPU stages

```bash
# Six seed-42 runs; stops and deletes checkpoints if the registered gate fails.
bash rebuttal/rebuttal_0723/experiments/mla_scarcity_5090/run_5090.sh gate

# Only after a PASS gate: seeds 43/88, test evaluation, YaRN diagnostic, summary.
bash rebuttal/rebuttal_0723/experiments/mla_scarcity_5090/run_5090.sh confirm
```

The confirmation stage evaluates and verifies each run before deleting its
checkpoints. It retains raw JSON/JSONL rows, hashes, schedule sidecars, and the
shared TorchInductor cache. Check disk status at any time with:

```bash
bash rebuttal/rebuttal_0723/experiments/mla_scarcity_5090/run_5090.sh disk
```

Training refuses to start below 8 GiB free. The READY receipt also records a
conservative checkpoint peak: at most fifteen simultaneous checkpoints (the
twelve retained seed-42 200M/300M proofs plus one three-snapshot transient
run), under 4 GiB for this 50.1M-parameter model. The compile cache is retained
across runs and can be deleted only after a terminal STOP gate or confirmatory
summary:

```bash
bash rebuttal/rebuttal_0723/experiments/mla_scarcity_5090/run_5090.sh cleanup-cache
```

Do not describe the EVQ virtual-coordinate YaRN transform as official YaRN.
Only the active native-endpoint Geo arm uses the official YaRN equations; their
cross-arm result is a labeled deployment diagnostic.
