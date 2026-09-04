# Constrained single-table frontier and 2x/4x LoRA transfer preflight

- **Date:** 2026-09-04
- **Status:** factor-frontier code prepared; LoRA implementation prototype
  prepared but its route-explicit data/causal diagnostic gate remains open; no
  new GPU result; the author shut the server down
- **Questions:** (Z) how far can one static frozen table extrapolate while its
  1x NLL and downstream damage remain about 0.12; (F) can small matched LoRA
  exposure at physical 2x/4x transfer to unseen 8x/16x/32x without exceeding
  the same 1x damage budget?
- **Estimands:** largest feasible factor on the declared log-p2 factor/gain
  family; adapted-minus-frozen and candidate-minus-Native capability at held-out
  physical lengths
- **Claim boundary:** this program estimates a checkpoint- and family-specific
  frontier. It cannot prove a global RoPE upper bound or identify a unique
  optimal table from finite arms.

## Material Passport

- **Objective:** separate short-context compatibility, long-context geometry,
  and learned capability conversion under one deployable table/path.
- **Object:** released OLMo-2-0425-1B-Instruct, its exact Native inverse-frequency
  tensor, and the exact retained legacy-u p2 log-s4 tensor.
- **Decomposition dimensions:** table factor, fixed scalar gain, frozen versus
  QKVO-LoRA weights, training exposure length, and evaluation length.
- **Evaluation:** paired PG-19 tail NLL, five natural generation tasks, RULER
  core-4, exact training receipts, and held-out physical 8x/16x/32x RULER.
- **Evidence required:** checkpoint/config/table/data/code hashes, raw per-row
  generations or NLLs, Flash-only runtime receipt, adapter bytes, failures, and
  exclusions.
- **Environment:** canonical work machine, BF16 CUDA, Flash SDPA enabled with
  math and memory-efficient fallbacks disabled; 32 GiB GPU. CPU/RAM display is
  not treated as a resource contract.
- **Output:** machine-local raw bundles plus a compact repository result owner;
  no private path, row, prediction, adapter, or server identity is committed.

## 1. Correction to the research target

The objective is not exact Native equivalence and not universal domination.
There is one global static table and one fixed attention gain installed before
prefill, no routing, boundary switch, dual table, cache handoff, or head-specific
clock. A method is acceptable when both measured 1x retentions are near the
author's 0.12-damage budget and longer capability is maximized.

The repository's historical threshold `retention >= 0.875` means damage at most
0.125 and is kept for comparability. Every result must also report whether the
stricter literal `retention >= 0.88` threshold passes; values between the two are
labelled **marginal**, not silently rounded to 0.12.

## 2. Existing observations and unresolved causes

The retained log-p2 s4/c=.074 arm has PG-19 PPL retention `0.875302` and
five-task retention `0.915103`, then strongly improves 2x/4x NLL and capability.
It is marginal under the literal 0.88 NLL threshold and passes the historical
0.875 gate. The old direct s8 log arm used the different analytic gain
coefficient `.10`; its core-4 row improved over arithmetic s8 but the full-13
run stopped after single-key-3 collapse. These facts do not identify a zero-
training upper bound.

A descriptive quadratic through the already measured Native, s2, and s4 PG-19
points predicts the p2/c=.074 NLL boundary near factor `3.91` for retention
0.88 and `4.01` for retention 0.875. This is a post-outcome local fit, not a
theorem. It makes a blind factor sweep unattractive and motivates the gain/table
discriminator below.

The LoRA conversion problem predates log-p2. Mature EVQ-Cosh adaptation improved
long NLL and causal source use without reliable top-1 generation; the recent
same-substrate log-p2 Q/K run again improved PG-19 but not generated capability.
The new experiment therefore changes the learning signal and QKVO capacity,
not merely rank, alpha, gain, or step count.

## 3. Stage Z: locate the zero-training bottleneck

The exact p2 movement encoded by the Native and retained log-s4 tensors is
recovered as

```text
m_k = -log(omega_s4_k / omega_native_k) / log(4)
omega_k(s) = omega_native_k * s ** (-m_k)
gain(s,c) = 1 + c log(s)
```

The table exporter freezes factors `4,5,6,7,8` and gain coefficients
`.05,.074,.10` before any new LM outcome. The reference s4 tensor is copied
byte-for-byte rather than reconstructed.

Execution is adaptive:

1. score all fifteen arms on a small PG-19 1x calibration subset;
2. retain only the lowest-NLL gain at each factor;
3. evaluate those five arms on the full 20-row PG-19 gate, five 1x natural
   tasks, and 1x core-4;
4. select the largest factor passing both NLL/PPL and downstream retention at
   0.875, while separately marking strict-0.88 status;
5. open 2x/4x/8x core-4 only for that factor. Do not evaluate factor 8 long
   merely because it is the requested ceiling.

This factorial distinguishes three useful failure classes:

- no gain passes 1x PG-19: the p2 table displacement itself exceeds the short
  likelihood budget at that factor;
- PG-19 passes but natural/core capability fails: frozen readout compatibility,
  not likelihood alone, sets the observed boundary;
- both 1x gates pass but 8x fails: the static geometry/gain does not convert to
  long generated capability.

It still estimates only the p2-family boundary. Calling it the global zero-
parameter limit would be an overclaim.

## 4. Stage F: physical 2x/4x LoRA to unseen 8x/16x/32x

The training substrate remains one static table and fixed gain. Training uses
the already hash-bound identifiable natural pair views at physical 4K, 8K, and
16K. Each correct/deranged variant changes both the source-owned answer span and
its matching output label, so the two CE targets are not contradictory.

The current prototype loss applies, for each variant, answer-plus-EOS CE plus a same-target
correct-source over corrupted-source log-probability margin. The corrupted view
keeps the teacher-forced output tokens fixed and changes the remote source
content. Unlike the failed recent Q/K screen, it updates standard PEFT Q/K/V/O
LoRA (rank 64, alpha 128). The 300-step schedule is
`2x,4x,2x,4x,1x replay`; no 8x/16x/32x row is visible to training or selection.
Before a scientific run, add a route-explicit target (context-random source
nonce, context-random answer nonce, immediate EOS) or otherwise accept that
retrieval cannot be scored separately from answer generation. The minimum
projection screen is frozen/QK/QKVO: QK runs first; QKVO opens only if causal
routing succeeds but oracle-routed answer/EOS remains weak. Match actual
trainable parameter counts, not nominal rank. The prepared runner currently
launches the QKVO prototype and must not be treated as this completed screen.

The first table uses retained s4/c=.074 as a resolving positive-control
substrate. Only after its one-step Flash smoke and finite run may the s8 table
with its Stage-Z-selected gain run. A matched Native-table adapter is required
if a non-Native candidate passes and causal table attribution is needed.

Evaluation order is mandatory:

1. 1x PG-19 plus five natural tasks and core-4; stop the candidate if either
   retention is below 0.875 and report strict-0.88 status;
2. held-out physical RULER core-4 at 8x, 16x, and 32x, 20 rows per cell;
3. expand to RULER-13 and held-out natural QA only if core-4 is non-floor at 8x
   and retains a nontrivial signal at 16x. A 32x floor does not erase a valid
   8x/16x result.

Training-loss reduction, answer-token NLL, source margin, attention mass, and
first-token rank are diagnostics. Only greedy generated-task endpoints establish
capability transfer.

## 5. Stop conditions and implementation

Stop immediately on tensor/data/hash drift, train/eval length leakage, missing
counterfactual target differences, non-finite loss or gradient, OOM, less than
1 GiB measured headroom, any non-Flash attention fallback, adapter reload drift,
or author stop. Preserve partial outputs and record exclusions.

Prepared code:

- `scripts/analysis/export_log_p2_factor_frontier.py`
- `scripts/train/train_log_p2_phase_transfer_lora.py`
- `scripts/eval/run_log_p2_frontier_and_transfer_4080.sh`
- extended 16x/32x support in `scripts/eval/target_free_ruler_smoke.py`

The runner has separate build, preflight, retention, smoke, train, adapted-
retention, and far-evaluation actions. No stage launches the next automatically.
