# Latest steering — core result first

The user explicitly withdrew six-arm-first sequencing. Preserve prepared code and
data, but do not automatically launch six or eighteen long trainings. First choose
and demonstrate the strongest central experiment for modern positional modeling;
broader comparisons come afterwards. See
`docs/research/POSITION_AFTER_SPARSE_RESEARCH_20260908.md`.

Real-data probe `probe_mb8_02` completed: 151,898,880 parameters, Flash-only BF16
forward/backward and finite training loss passed; micro8/accum4, compiled default.
Steady throughput ~74,650 tokens/s; ten discarded updates, total loop 38.0 seconds
including initial compile. The old 18-run plan would require about 33.49 GPU-hours
of pure training at this short-probe speed, excluding evaluation/IO. It is **not
launched**. GPU process 17018 is terminal, data preparation 16258 completed.
The reconstructed train prefix matches the old semantic int64 SHA exactly;
512 distinct eligible validation documents are frozen. New validation array SHA:
`f30d913567e54d8a00d4d25221eacf66af076f2eb1e19e6f1f807b2faaed7857`.

First probe 16847 failed before training due to an overstrict old native-sm89
check. The installed sm86 cubin is Ada-compatible per NVIDIA. New runtime checks
compatible cubin plus actual Flash forward/backward; it keeps all slow SDPA
fallbacks disabled. No training loss from that failed process is claimed.

The chronological preparation notes below are retained; their six-arm-first
sequence has been superseded by this latest user instruction.

# Rotary budget experiment — active owner

Status: implementation and resource qualification; no new LM results yet.
The user replaced the zero-training MrRoPE objective with this study on 2026-09-08.
The KLD v2 follow-up is separate in `experiments/kld_v2/`.

## Frozen scientific design

`DESIGN_SOURCE.md` is the user-supplied source, copied without alteration.
`protocol.yaml` instantiates all **six** arms (the source's isolated “5 arms”
phrase is a typo relative to its explicit six-arm specification).
The 18 runs are six arms times seeds 42/137/256, with seed42 completed first.
No old 256-length checkpoint is a 2K G32 baseline.

`build_tables.py` writes all 32-slot float32 tables. Inactive pairs are zero
frequency; no reshaping or repacking changes their identity. U16 uses G32
indices 0,2,...,30, assigned to the first 16 fixed pairs as specified in §8.
`eval_inputs.py` uses 8193-token anchors to supply 8192 inputs plus the final
shifted label. The source's minimum 8192 is strengthened by one token to make
its requested 8192-input evaluation and causal shift precise.

## Verified implementation evidence

- CPU tests: endpoint/reference construction, K=2 and tau=0 limits, six-arm
  identical initial trainable tensors, fixed inactive pairs, norm and relative
  translation, identical target IDs and position-preserving remote replacement.
- `theory_checks.json`: 32 finite-grid configurations; full spectra, T1/T2,
  three declared eigenvalue thresholds. This is geometry, not LM evidence.
- Supplied `theory_check.py`, `softmax_probe.py` were not present beside the
  source document. The local theory checker is an explicitly new implementation;
  the supplied document's softmax numbers are not claimed as reproduced.

## Reuse and remaining work

Live server: RTX 4080 SUPER, 32 GiB; idle at initial inspection, 19 GiB data disk free.
Recovered original GPT architecture from
`/root/autodl-tmp/hybrid-rope/experiments/native_rope_evq_150m/model.py`.
Original trainer/protocol snapshots are in `reference/` for comparison, not new
trainers. The original data manifest is at
`/root/autodl-tmp/iclr_exact_range_multiseed/data/data_manifest.json`.
Its train token array is recorded as 499,974,144 tokens in 256-token rows;
reshape the identical stream to 2048-token rows and regenerate one paired order.
Verify actual existence/hashes before use. Held-out long-document evaluation must
be rebuilt from shard004; the old packed validation anchors are unsuitable.

Remaining: actual-data throughput and finite-loss qualification; resource estimate
and unified budget freeze; resumable training and periodic validation; 512-document
freeze; seed42 six arms; registered decision; confirming seeds when warranted;
paired NLL/context contrasts and budget-quality plots; integrate actual results.
A reduced budget below roughly 268M tokens/arm must be called a pilot, not a stable
frontier. No result-dependent cancellation of a seed42 arm, tau change, or baseline
removal is allowed.
