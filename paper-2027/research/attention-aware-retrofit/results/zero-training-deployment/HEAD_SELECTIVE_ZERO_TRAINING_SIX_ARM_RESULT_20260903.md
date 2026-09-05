# Head-selective mid-band compression does not pass the joint Native--long objective

- **Date:** 2026-09-03
- **Status:** `COMPLETE / VALID_EXECUTION / EXACT_CANDIDATE_NEGATIVE_ON_REUSED_DEVELOPMENT_PANEL`
- **Evidence label:** **Negative result**, limited to the exact candidate and protocol below
- **Question:** With OLMo-2-0425-1B-Instruct weights frozen, does the
  calibration-derived `Selective-31` mask outperform layer-matched fixed random
  and reverse masks when exactly the same RoPE slots are divided by four?
- **Primary estimand:** task-equal paired arm differences on realized input
  lengths `8192..16384`; the full 50-row mixed-length panel is secondary
- **Receipt:**
  [`HEAD_SELECTIVE_ZERO_TRAINING_SIX_ARM_RECEIPT_20260903.json`](../../evidence/HEAD_SELECTIVE_ZERO_TRAINING_SIX_ARM_RECEIPT_20260903.json)

## Verdict

The exact candidate fails its proposed joint objective on this development
panel.

1. On the 25 realized `>=8192` rows, Native, Selective, Reverse, and all three
   fixed random masks score `0` normalized EM and `0` normalized gold
   substring; none terminates with EOS within 32 generated tokens. The proposed
   `Selective > mean(Random) > Reverse` long-generation order therefore does
   not occur.
2. This is not merely an incapable assay. Exact matched historical controls on
   the same tokenized prompts score `7.14%` task-macro EM/substring for global
   `log_s4` and `22.02%` for official YaRN-4, with `85.12%` and `100%` EOS
   termination. The controls are adaptive matched-baseline completion, not a
   new untouched comparison.
3. The short-retention ordering reverses the hypothesis. Selective has higher
   forward KL than the three-mask random mean (`.10310` versus `.04191`) and
   lower Top-1 agreement (`85.85%` versus `90.26%`). Its first-reference
   answer-token NLL is also worse (`9.8055` versus `9.3299`).

This closes only `OLMo-2-0425-1B-Instruct / Selective-31 / s=4 / slots
[16,48) / current sensitivity-derived mask / current panel`. It does not close
head-selective RoPE, attention-aware selectors, other bands/scales/head counts,
or GQA models as classes.

## Protocol and identities

- Model: released `OLMo-2-0425-1B-Instruct`; 16 layers, 16 Q heads, 16 K
  heads, head dimension 128, Native context 4096, MHA only.
- Transformer weights: fully frozen; no learned parameter, adapter, routing,
  gain, or task-conditioned branch.
- Six primary arms: Native, `Selective-31`, layer-matched Reverse-31, and
  layer-matched random masks with seeds 42, 123, and 999.
- Intervention: for the 31 selected layer/head identities only, divide Native
  inverse frequencies at pair slots `[16,48)` by `4`; every other frequency
  remains bitwise Native.
- Calibration rows: HotpotQA tokenized-file rows `0..11`, first 128 tokens.
  Evaluation rows: tokenized-file rows `50..74` for HotpotQA and 2WikiMQA.
  They are calibration-disjoint, but not globally untouched: the earlier
  adaptive headwise development evaluated the original 200-row task panels.
- Realized input lengths: `1,234..16,309`, comprising 3 rows `<=4096`, 22 rows
  `4097..8191`, and 25 rows `8192..16384` (21 HotpotQA, 4 2WikiMQA).
- Generation: batch 1, greedy, maximum 32 new tokens. Primary generation
  endpoints are normalized exact match and normalized whole-word gold
  substring.
- Auxiliary endpoint: teacher-forced NLL of the first reference answer tokens;
  it is not context NLL, generated-answer likelihood, or QA accuracy.
- Short retention: each registered prompt's first at most 4096 tokens; full
  vocabulary forward KL and Top-1 agreement on its last 100 logit positions.
- Runtime: BF16, PyTorch SDPA with Flash enabled and math,
  memory-efficient, and cuDNN fallbacks disabled.

The executed runner produced `300/300` unique finite cells. A standalone
16K-shape Flash probe passed. Stock Transformers and the custom all-Native path
have bitwise-zero prefill and one-step decode logit differences both at 32
tokens and on the longest 16,309-token registered row.

A CPU-only provenance recovery reproduced the complete `12x256` sensitivity
matrix, all 66 stability pairs, and the frozen mask bytes exactly without
modifying the original asset. Mean/minimum cross-row Spearman are
`.973638/.955643`; mean/minimum Low-31 Jaccard are `.704072/.589744`. These
statistics describe only the fixed 12 prefixes and do not identify a model or
task population.

## Pre-outcome analysis amendment

The supplied runbook called all 50 rows `8K..16K`, although half are shorter.
Before any arm outcome value or ordering was inspected, an analysis amendment
froze:

- `8192..16384` as the primary target-long population;
- the full 50 rows and task-by-length cells as secondary;
- EM and substring as co-primary generation endpoints;
- per-row averaging of the three fixed random masks;
- paired row differences, task-equal macro averaging, and 50,000
  task-stratified bootstrap resamples with seed `20260903`.

The three random masks are fixed controls, not statistical repetitions or a
sample that identifies the random-mask population.

## Results

### Target-long population: realized `8192..16384`

| Arm | EM | Gold substring | EOS | Answer-token NLL | Short KL | Short Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | `.0000` | `.0000` | `.0000` | `9.4328` | `.00000` | `1.0000` |
| Selective-31 | `.0000` | `.0000` | `.0000` | `9.8055` | `.10310` | `.8585` |
| Reverse-31 | `.0000` | `.0000` | `.0000` | `9.1233` | `.07854` | `.8595` |
| Random-31 seed 42 | `.0000` | `.0000` | `.0000` | `9.3684` | `.04208` | `.9004` |
| Random-31 seed 123 | `.0000` | `.0000` | `.0000` | `9.2822` | `.04988` | `.8965` |
| Random-31 seed 999 | `.0000` | `.0000` | `.0000` | `9.3392` | `.03378` | `.9108` |
| Three-random mean | `.0000` | `.0000` | `.0000` | `9.3299` | `.04191` | `.9026` |
| Global `log_s4` matched control | `.0714` | `.0714` | `.8512` | n/a | n/a | n/a |
| Official YaRN-4 matched control | `.2202` | `.2202` | `1.0000` | n/a | n/a | n/a |

Task-stratified paired estimates, with positive values defined as Selective
advantage:

| Contrast | Estimate | Two-sided 95% bootstrap interval |
| --- | ---: | ---: |
| Selective minus random mean, EM | `.0000` | `[.0000,.0000]` |
| Selective minus random mean, substring | `.0000` | `[.0000,.0000]` |
| Random-mean NLL minus Selective NLL | `-.4756` | `[-.7507,-.2450]` |
| Random-mean KL minus Selective KL | `-.06119` | `[-.07872,-.04693]` |
| Selective Top-1 minus random mean | `-.04405` | `[-.06345,-.02315]` |

Reverse is not the predicted worst-retention arm: random mean has lower KL and
higher Top-1 than Reverse, while Selective is approximately as low in Top-1 as
Reverse and has still higher KL.

### Full 50-row registered panel

Selective scores `4%` EM/substring, versus `6%` for the three-random mean and
`4%` for Reverse and Native. Selective minus random mean is `-2.0` percentage
points with paired 95% interval `[-7.33,+2.67]`; the sparse generated-success
contrast is unresolved, not evidence of equality.

The auxiliary directions are adverse and resolved:

- answer-token NLL: Selective `8.1953`, random mean `7.6458`; adverse difference
  `+.5494`, 95% interval `[+.1512,+1.0092]`;
- short KL: Selective `.10934`, random mean `.04746`; adverse difference
  `+.06188`, 95% interval `[+.04838,+.07777]`;
- short Top-1: Selective `84.86%`, random mean `89.33%`; difference `-4.47`
  percentage points, 95% interval `[-5.49,-3.42]`.

Global `log_s4` and official YaRN-4 score `20%` EM on the full panel; substring
is `20%` and `22%`, and EOS termination is `96%` and `100%`.

## Interpretation and limits

### Post-run metric and gauge audit

The recovered selector score is
`||A_s4-A_Native||_F/||A_Native||_F`, evaluated from Native layer inputs on
twelve fixed 128-token prefixes. It is an exact endpoint attention-map
displacement for that calibration pack, not a loss derivative, `chi_func`, or
task-free functional compatibility metric. Reverse is layer/count matched but
not matched on evidence utility, output coupling, or head-output norm.
Accordingly this experiment tests the fixed mask candidate; it does not
identify sensitivity as the causal mechanism of protection.

The current OLMo2 source and executed custom path both use split-half logical
pairs `(j,j+64)`. The custom all-Native identity checks plus source inspection
support the realized intervention, but no nontrivial joint
frequency--Q/K-relabel identity was executed. Such a relabel would be a gauge
control, not the present fixed-Q/K frequency intervention. The full derivation
and exact boundary are owned by
[`LOCAL_FUNCTIONAL_COMPATIBILITY_AND_GAUGE_AUDIT_20260903.md`](../../theory/LOCAL_FUNCTIONAL_COMPATIBILITY_AND_GAUGE_AUDIT_20260903.md).

### Supported at exact scope

- The six frozen masks were applied with equal per-layer counts and identical
  frequency intervention; mask identity is the arm-level difference.
- The specific sensitivity-derived mask does not improve long generation over
  the three fixed random controls or Reverse on this panel.
- Its short distributional change and answer-token NLL are worse than the
  three-random mean under the registered metrics.
- Global static controls resolve the same natural-QA prompts, so the six-arm
  long floor is candidate-specific rather than proof that every RoPE extension
  must score zero.

### Not supported

- an untouched confirmation or formal causal-chain closure;
- geometric sensitivity as the unique mediator: head identity also carries
  baseline importance, retrieval function, output coupling, and other
  correlated properties;
- calling the calibration score `chi_func`, a loss sensitivity, or a
  task-agnostic compatibility metric;
- a universal head-specialization law, cross-checkpoint transfer, GQA
  compatibility, SOTA, or a head-selective method-class negative;
- Native preservation: no non-inferiority threshold was registered;
- treating all 50 rows as `8K..16K`, treating answer-token NLL as generation
  accuracy, or treating the three random masks as independent row samples.

## Relation to prior owners and next action

This result is a zero-training frozen-mask intervention. The 9/2
[`HEADWISE_FACTORIZED...`](HEADWISE_FACTORIZED_Z_AND_SCALE_FLOW_RESULT_20260902.md)
report learned per-head fields on adaptively reused full panels. The estimands
are different; neither supersedes the other.

The pre-outcome derivative tree stops here: resolving global controls passed,
while the Selective arm had neither a long advantage nor the expected short
ordering. No head-count, scale, slot, or selector sweep is justified by this
result. A future different selector requires a new prospective owner and a
fresh confirmation source; it must not be presented as continuation of this
candidate.
