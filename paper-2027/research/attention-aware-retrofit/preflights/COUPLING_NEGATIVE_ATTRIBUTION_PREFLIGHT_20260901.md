# Frozen coupling: negative attribution and static baseline resolver

## Decision and fixed scope

The next round keeps the existing frozen two-parameter law and uses the
already available checkpoints. It does not fit a new curve, boundary, gain,
or effective context length. The purpose is to identify why K32 has a
Native/long Pareto crossing and why the K128 screen collapses at 16K before
constructing a general method. An offline selector is not substituted for
the user's objective of a competitive, transferable static construction.

Existing evidence belongs to
[`FROZEN_2D_COUPLING_TRANSPORT_RESULT_20260901`](../results/FROZEN_2D_COUPLING_TRANSPORT_RESULT_20260901.md).
Its Native-versus-external canaries close only the loader-branch difference:
both branches share the custom attention backend and handwritten generation.
They do not establish stock-HF or phase-arithmetic correctness.

## A. Runtime attribution, first GPU stage

Code: [`audit_rope_runtime_parity.py`](../../../../scripts/eval/audit_rope_runtime_parity.py).

- Exact existing Gemma-1.1 instruction artifact and existing paired data.
- Native table, original checkpoint config, no model or frequency update.
- First row of `niah_single_1` and `niah_multikey_2` at 4096/8192/16384:
  six diagnostic rows, not a capability confirmation.
- Compare stock-HF SDPA plus `generate` with the existing custom Flash
  backend plus handwritten greedy decoding on identical input IDs and
  explicitly matched EOS and generation budgets.
- Record prefill and one teacher-forced cached-step logits, stock cached
  versus uncached continuation, raw generated IDs, scores, runtime source
  hashes, Native tensor identity, timing and peak memory.
- Flash-only execution, no quadratic full-length eager fallback.
- Freeze the script/data/weight/config identities in the raw run manifest.

Any discrepancy is diagnosed before a new capability matrix. Agreement
excludes the tested implementation differences only; both paths still share
the installed Gemma implementation and PyTorch Flash kernel. If phase
arithmetic is suspect, compare realized phases with integer-position/FP64
reference, and test IEEE-FP32 arithmetic as an implementation control, not
as a new RoPE candidate. A small Q/K slice can supply an eager reference
without materializing a full long-context attention matrix.

## B. Deterministic static baseline resolver

After the relevant runtime controls pass, add only the two missing static
baseline arms on Gemma-1.1: official-equation YaRN2 and static NTK2. Keep
`L_ref=8192`; the existing RULER scores must not be used to redefine it.

- YaRN uses the installed, source-hash-bound HF initializer with its fixed
  published ramp constants and amplitude `1+0.1 log(2)`; verify equation
  parity on CPU, distinguish tolerance parity from exact tensor identity.
- Static NTK uses `b'=b*2^(d/(d-2))`, `d=2K`, and amplitude one. It is not
  a dynamic, sequence-length-dependent NTK implementation.
- Load each exported frequency table once before prefill, retain it for the
  entire request and every tested length, using standard attention and KV
  cache. No per-request winner selection.
- First complete the same 16K core-four cells, 20 rows per task per arm:
  160 new generations. Reuse verified completed Native/physical/index rows
  only if their runtime identity remains valid after stage A.
- If a baseline resolves nonzero long behavior, the frozen law has a
  method-specific deficit in this protocol; this is not a causal K effect.
- If all arms remain at floor, report a tested-panel long negative and an
  unresolved coordinate comparison, not a universal impossibility.
- Supplement short and Native lengths to report the actual operating curve
  before claiming a competitive static method. A nonzero resolver is only
  a measurement entrance, not practical success or SOTA.

The generic old YaRN loader checks an already-scaled tensor as if it were
Native, and the OLMo convenience path hard-codes factor four. This new panel
therefore uses explicit static tables and the existing hash-bound external
loader for all models. It does not weaken identity checks or silently reuse
the factor-four path for a factor-two comparison.

## C. CPU and parallel research while GPU runs

Separate adjacent-frequency resolution from the absolute phase change of
each learned slot. Compare geometric quantities with finite, signed
checkpoint Q/K responses; geometry alone must not rank RULER outcomes.
Any subsequent Q/K collection or same-weight coarse-grid intervention gets
its own frozen inputs and estimand before launch. Cell-average/P3 remains
closed by its existing failed entrance check.

K, model family, head dimension, learned coefficients and training history
are confounded in the current checkpoints. Same-family Qwen K32/K64 reduces
some confounds but cannot identify the causal effect of K or training
exposure. No new model download is currently necessary, and the protected
1.485B checkpoint must remain intact. No automatic shutdown is scheduled
for this newly authorized session.

## Evidence ceiling

This is a preregistration and implementation plan, not a completed GPU
result. Frozen `G` and a scale-consistent formula do not by themselves prove
behavioral universality, Native compatibility or SOTA. No third rescue
candidate, parameter sweep, long-label fit, or hierarchical table is opened.
