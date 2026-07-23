# LLaMA-3-8B LoRA Industrial Capability Conversion Design

## Status

Approved on 2026-07-14. This pass writes the scientific design and implementation
plan only. It must not start a GPU process, modify a checkpoint, or change a paper
metric.

## Decision

Use a gated three-stage design:

1. evaluate the existing seed-42 base, Geo+LoRA, and EVQ+LoRA artifacts without
   training or range scaling;
2. only if those artifacts lack source-dependent capability, continue the two
   matched LoRA parents for a short, strictly 8K task-directed segment;
3. only after a positive transfer signal, run a same-base, same-initialization
   native-Geo/midpoint-Geo/EVQ confirmatory comparison.

This is the lowest-cost design that can distinguish a real long-context capability
gain from language-model NLL stability, output-format failure, generic LoRA
fine-tuning, endpoint-versus-midpoint quantization, and EVQ-specific frequency
shape.

## Existing Evidence and Its Boundary

Authoritative inputs for this design are:

- `data/curated/lora_longalpaca_temporal_s42_20260712.json` for the raw-backed
  seed-42 temporal result;
- `rebuttal/pre_rebuttal/LORA_LONGALPACA_TEMPORAL_NLL_20260712.md` for its interpretation and
  schedule-quantizer boundary;
- `rebuttal/pre_rebuttal/FULL_PAPER_INTEGRITY_AUDIT_20260713.md` for rejected LoRA-rank and
  paper-claim theory;
- `rebuttal/pre_rebuttal/EVQ_8K_ONLY_CAPABILITY_TRANSFER_PLAN_20260713.md` for the detailed
  phase-observability audit;
- `Agent.md` for model/data identity and paid-GPU safety.

The completed LongAlpaca seed-42 experiment is a matched training-pipeline
comparison on LLaMA-3-8B-Instruct. On a 2026 temporal holdout, EVQ+LoRA relative
to Geo+LoRA changes domain-macro NLL by `+0.38986` at 8K, `-1.51008` at 16K,
and `-2.04786` at 32K. The 16K and 32K signs agree in all 24 disjoint packs.

This is real evidence that a short LoRA adaptation can improve long-position token
likelihood on an external, temporally separated corpus. It is not yet evidence of
long-range retrieval, question answering, generation, or source dependence:

- most scored tokens can be predicted from local context;
- the held-out packs concatenate shorter documents;
- the comparison changes native endpoint Geo to midpoint EVQ-Cosh, so it is not
  a pure density-shape contrast;
- the result is one training seed.

The old Base-versus-EVQ LoRA table, WikiText-only evaluation, LongAlign label,
and rank/channel phase-transition narrative are not evidence for this design.

## Scientific Question

Can EVQ-Cosh on a mature pretrained 8B checkpoint convert short-window LoRA
training into source-dependent capability beyond the 8K training boundary, and is
any positive effect attributable to frequency allocation rather than generic LoRA,
output formatting, range scaling, or endpoint/midpoint quantization?

The first decision is narrower: do the already completed LongAlpaca adapters
contain latent source-dependent capability that previous evaluators failed to
measure? New training is not justified until that question is answered.

## Theory-to-Experiment Contract

### Exact statements

For rotary pair `k`, a fixed token-pair contribution has the exact form

```text
a_mnk cos(omega_k Delta) + b_mnk sin(omega_k Delta).
```

For a fixed frequency table and an explicit distance prior, phase moments,
dormant counts, and phase-feature Gram matrices are well-defined schedule
diagnostics. With the midpoint grid and positive `tau`, EVQ-Cosh raises every
interior frequency relative to midpoint Geo, reducing the number of channels that
remain near identity throughout 8K.

### Conditional mechanism

The experiment tests this falsifiable chain:

```text
schedule phase coverage
    -> task-weighted q/k/v/o gradient and update coverage
    -> source-sensitive attention and value transport
    -> held-out 16K/32K capability
```

Model-free phase coverage does not establish any later arrow. Task labels,
content activations, softmax competition, attention sinks, value transport, and
layer routing all intervene. The experiment therefore treats schedule diagnostics
as predictions, not as substitutes for capability metrics.

### Rejected theory

LoRA rank is not a frequency-channel count. A rank-`r` update can affect every
output coordinate. This design must not use `1-r/K`, `r approximately K` as a
phase transition, “48 frozen channels,” or `tau=d/sqrt(L)` as a global optimum.
Rank 64 and `tau=1.414` are retained only as the already executed empirical
operating point.

## Hypotheses

### H1: Existing latent capability

The existing EVQ+LoRA parent already has stronger 16K/32K source dependence than
the Geo+LoRA parent; prior failures were dominated by prompt format, decoding, or
an evaluator that measured the wrong target.

Prediction: EVQ+LoRA improves paired answer NLL, original/swapped consistency,
and source-removal sensitivity without new training.

### H2: NLL stability without capability

EVQ+LoRA improves long-position language modeling but does not improve retrieval
or evidence use.

Prediction: temporal NLL remains positive evidence, but controlled source
dependence and unseen task accuracy do not beat Geo+LoRA.

### H3: Task-directed 8K conversion

Both parents can learn the same source-dependent task within 8K, but EVQ transfers
more of that learned circuit to 16K/32K because its training-window phase coverage
better prepares the adapted projections for long positions.

Prediction: both arms pass the 8K learning gate, while the EVQ improvement from
parent to post-continuation is larger at 16K/32K.

### H4: Mature-band disruption

Full EVQ replacement damages a pretrained frequency band or source-routing
circuit. Long-position NLL can recover while source dependence remains weak.

Prediction: cross-swaps expose frequency-specific co-adaptation, and EVQ shows
weaker task-weighted pair energy or source-removal effects despite better temporal
NLL. A later method should then preserve or partially warp the trained band rather
than extend the present run.

## Global Experimental Invariants

- Base model and tokenizer: the same manifested LLaMA-3-8B-Instruct bytes used by
  the completed LongAlpaca seed-42 pair.
- Parent adapters: completed step-300 Geo+LoRA seed 42 and EVQ+LoRA seed 42.
- Parent data identity: the same frozen LongAlpaca tensor and manifest.
- LoRA: `q_proj,k_proj,v_proj,o_proj`, rank 64, alpha 128, dropout 0.05.
- Precision: BF16; no QLoRA, FP8, or full-parameter updates.
- Stage 0 and all primary evaluations use raw schedules. They do not use YaRN,
  NTK, PI, PoSE, LongRoPE, or any other range scaler.
- Training and evaluation examples use the complete LLaMA-3 chat template.
- Every compared arm sees identical prompt IDs, canonical target IDs, row order,
  and scorer definitions.
- A pure EVQ-shape claim requires midpoint Geo and midpoint EVQ; native endpoint
  Geo is a separate standard-checkpoint control.
- Compile mode, GPU model, checkpointing, and attention backend are execution
  metadata, not scientific factors, but each matched run uses one locked choice.
- Null and negative outcomes are retained. No arm, length, task, or seed may be
  removed after results are observed.

## Data Design

### Primary controlled source-dependence suite

Use three task families:

1. nonce key-value retrieval;
2. last-write-wins with an obsolete and a current value;
3. two-hop variable tracing.

Each semantic group produces three exactly equal-length records:

- `original`: source and target agree;
- `swapped`: source and target are changed together;
- `source_removed`: the source span is replaced with equal-length nonce-free
  filler while the original target is scored.

The triplet shares template, document, query position, answer position, and token
length outside the registered source/target substitutions. Train, validation, and
test use disjoint nonces, filler documents, templates, and semantic rules. The
query is near the end; source-query distances are explicitly stored.

Primary lengths are 4K, 8K, 16K, and 32K. They provide an in-range control, the
training boundary, and clean 2x/4x extrapolation. Lengths 12K and 24K are reserved
for breakpoint localization after a positive core result and cannot change the
primary decision.

### External capability suite

Use a pinned LLaMA-3-adapted RULER subset containing the existing registered NIAH,
variable-tracking, common-word-extraction, and frequent-word-extraction families.
The official prompt string and references remain unchanged inside one LLaMA-3
user message. Report this as a chat-adapted RULER subset, not bit-for-bit official
prompt parity.

LongBench NarrativeQA and Qasper, NoLiMa-Hard, and short MCQA remain secondary
guardrails. They are run only after the controlled source-dependence gate. MCQA
scores the requested answer labels, not full option text.

### Canonical answer contract

Teacher-forced NLL always uses a fixed canonical target selected at data-freeze
time. It never chooses a different reference per model arm. Alternate references
are used only by generation metrics, or by a preregistered fixed-set log-sum-exp
metric that is identical across arms. Every output retains raw NLL sum and target
token count.

## Stage 0: Existing-Artifact Evaluation

### Seven raw arms

| Arm | Adapter | Runtime frequency | Role |
| --- | --- | --- | --- |
| `base_native` | none | native endpoint Geo | original checkpoint |
| `base_midpoint` | none | midpoint Geo | quantizer-only injection |
| `base_evq` | none | midpoint EVQ, tau 1.414 | direct EVQ injection |
| `geo_lora_native` | Geo parent | native endpoint Geo | matched Geo pipeline |
| `geo_lora_evq_cross` | Geo parent | midpoint EVQ | co-adaptation diagnostic |
| `evq_lora_native_cross` | EVQ parent | native endpoint Geo | generic-SFT diagnostic |
| `evq_lora_evq` | EVQ parent | midpoint EVQ | current positive-NLL arm |

Cross-swaps are mechanism diagnostics, not fair performance arms. The primary
performance comparison is `evq_lora_evq` versus `geo_lora_native`, with
`base_native` as the pretrained reference.

### Cost-gated execution

1. CPU validation freezes data, hashes, arm contracts, and the exact row subset.
2. A three-arm 4K/8K canary runs `base_native`, `geo_lora_native`, and
   `evq_lora_evq` on four semantic groups per task/length.
3. If the canary passes evaluator validity, the same three arms run the full
   preregistered 4K/8K/16K/32K core.
4. Only if a source-dependent EVQ/Geo difference exists are the four diagnostic
   arms run on the frozen mechanism subset.
5. External tasks and full generation run only after the controlled gate.

The model is loaded once, both adapters are attached once, and arm activation
switches adapter state and the verified frequency tensor. Answer NLL is primary
and memory-bounded; generation is greedy and uses the normal cache-capable
attention path.

### Stage 0 validity gate

Stop before any new training if:

- `base_native` and `geo_lora_native` both fail to reach 0.60 pair consistency at
  8K;
- source removal does not raise canonical answer NLL in at least 60% of those
  in-range groups;
- prompt IDs, target IDs, or group order differ across arms;
- failures are dominated by EOS, answer extraction, or chat-format errors;
- any adapter, model, training-data, or frequency identity check fails.

Stage 0 is mechanism-positive only when the 16K/32K EVQ-versus-Geo endpoint
difference has a group-clustered paired-bootstrap 95% lower bound above zero,
averages at least 5 percentage points, and is positive in at least two controlled
task families. Answer NLL and source-removal sensitivity must agree with the
direction. This remains single-seed supporting evidence.

## Stage 1: Matched 8K-Only Continuation

Stage 1 is permitted only when Stage 0 is valid but existing parents lack a clear
source-dependent advantage.

### Parent and optimization contract

- Start from the two completed LongAlpaca step-300 parents.
- Keep each parent's own final frequency tensor fixed.
- Reset AdamW and cosine scheduler to identical zero states for both arms.
- Keep all base parameters and the frequency tensor frozen.
- Continue the existing q/k/v/o rank-64 adapters only.
- Physical sequence length is exactly 8,192 and every position ID is at most
  8,191; fail closed otherwise.
- Primary segment: 32 optimizer steps.
- Effective batch: four sequences; candidate microbatch/accumulation layouts
  `(1,4)`, `(2,2)`, and `(4,1)` may be benchmarked before results, then one layout
  is locked for both arms.
- Physical tokens: 32,768 per step and 1,048,576 per arm per segment.
- Supervised answer tokens are recorded separately; they are never described as
  one million task labels.
- Learning rate `2e-5`, warmup 4 steps, cosine decay, weight decay `0.01`, max
  gradient norm `1.0`, BF16.
- A second 32-step segment is allowed only when both arms pass the same 8K
  validation-improvement rule defined before 16K/32K evaluation. Test outcomes
  cannot trigger continuation.

The extension rule is exact. If both arms already reach 0.80 pair consistency,
evaluate transfer at step 32 and do not extend. Otherwise, both arms must have
pair consistency in `[0.50,0.80)`, improve pair consistency by at least 0.10 and
canonical answer NLL by at least 0.15 nats/token relative to their own parent,
reach source-removal positive fraction at least 0.60, and pass the 8K temporal
guardrail. If either arm misses any condition, stop rather than spending another
segment.

### Training objective

Use assistant-answer-only causal cross entropy. Prompt, filler, source, and query
labels are `-100`. The training mix is fixed before launch:

- 35% nonce key-value;
- 25% last-write-wins;
- 25% two-hop variable tracing;
- 15% natural-evidence extractive QA, with a deterministic 12-token extractive
  span plus EOS so every training example has the same 13 supervised tokens.

Half of the algorithmic examples use high distractor density. Distances allocate
20% to 256-2,048, 30% to 2,048-5,120, and 50% to 5,120-7,680. Official benchmark
test templates never enter training.

### In-range learning gate

Both arms must independently satisfy:

- 8K validation pair consistency at least 0.80;
- source-removal positive fraction at least 0.75;
- non-empty and finite results for every training task family;
- 8K temporal NLL degradation relative to its own parent no greater than 0.10;
- short-core accuracy degradation no greater than 2 percentage points.

If only one arm learns the 8K task, report an in-range trainability difference and
stop the extrapolation claim. Do not call it equal-capability transfer.

### Transfer estimand

For higher-is-better metric `M` at length `L`, report both endpoint difference and

```text
I_L = (M_EVQ_post,L - M_EVQ_parent,L)
    - (M_Geo_post,L - M_Geo_parent,L).
```

Bootstrap semantic groups, not triplet rows or repeated depths. Use 10,000
deterministic resamples with seed 20260714. Stage 1 is mechanism-positive only
when endpoint and difference-in-differences lower bounds exceed zero, the mean
effect is at least 5 points over 16K/32K, and at least two unseen task families
agree. The macro gives equal weight to each task-family/length cell before
bootstrapping groups within cells. Ten points across both lengths is strong
supporting evidence, not a primary or universal claim.

## Stage 2: Same-Base Shape Attribution

Stage 2 is permitted only after a Stage 1 mechanism-positive result. Initialize
three adapters from the same base checkpoint and byte-identical zero-output LoRA
state:

1. `native_geo`: native endpoint Geo;
2. `midpoint_geo`: midpoint Geo, tau 0;
3. `midpoint_evq`: midpoint EVQ-Cosh, tau 1.414.

All three use the same 8K curriculum, row order, optimizer, scheduler, precision,
steps, and evaluator. They begin directly at their final schedule; no homotopy or
range scaling is introduced. `midpoint_evq` versus `midpoint_geo` isolates cosh
shape, while `native_geo` preserves relevance to the original checkpoint.

The primary Stage 2 budget is 32 steps for every arm. If all three arms miss the
0.80 learning gate but independently satisfy the exact Stage 1 extension rule,
all three may receive the same second 32-step segment. Never extend only the best
or worst arm.

Seed 42 is the only rebuttal-default confirmatory run. Additional paired seeds are
not automatic; they require a real reviewer question or an explicit post-rebuttal
decision. A single-seed success remains supporting evidence.

## Metrics and Statistical Unit

Primary metrics:

- original/swapped pair consistency;
- source-removal positive fraction and paired delta NLL;
- canonical answer-token NLL;
- extracted and strict autoregressive exact match.

Secondary metrics:

- chat-adapted RULER task accuracy;
- LongBench/NoLiMa metrics;
- temporal holdout NLL by length and position bucket;
- short-context MCQA label accuracy;
- EOS rate and generated-token count;
- per-layer/head/rotary-pair q/k activation, gradient, and effective LoRA-update
  energy.

The resampling unit is the semantic group. Original, swapped, source-removed,
depth variants, and template rephrasings from one group remain in the same
bootstrap cluster. Report task and length cells before any macro average.

Gradient coverage must not be defined as `energy > 0`. Report normalized energy
mass, entropy effective rank, Gini concentration, and top-quartile mass by
layer/head/pair. These diagnostics can support or reject the mechanism but cannot
replace capability outcomes.

## Failure Interpretation

| Outcome | Interpretation | Action |
| --- | --- | --- |
| Invalid 8K canary | evaluator/task-format failure | fix CPU/evaluator; do not train |
| Temporal NLL positive, Stage 0/1 task null | LM stability without demonstrated capability | retain narrow NLL claim; stop retrofit claim |
| Geo and EVQ fail 8K learning | task/data/optimizer failure | stop; no long evaluation or seed expansion |
| Both learn 8K, both fail long | strict 8K exposure does not transfer | report negative; PoSE/long-position exposure is a different future question |
| EVQ loses source dependence during direct replacement | mature-band disruption | investigate partial/band-preserving EVQ later |
| EVQ wins Stage 1 but not midpoint Stage 2 | parent co-adaptation or quantizer effect | do not claim cosh-shape causality |
| EVQ wins midpoint Stage 2 | shape-specific single-seed support | use only within stated supporting boundary |

## Efficiency and Paid-GPU Safety

- Download, tokenize, freeze examples, hash artifacts, and run all unit/dry tests
  before CUDA model loading.
- Stage 0 loads one 8B backbone and two adapters in one process.
- Run memory-bounded answer scoring for all rows; generation is gated and uses
  small task-specific limits.
- Use Flash SDPA only after a representative parity canary. Do not apply the
  packed-free training attention patch to cached generation.
- For 8K training, benchmark only the three fixed effective-batch-equivalent
  layouts on 2-4 non-claim steps. Select before observing task results.
- Enable `torch.compile(dynamic=False)` only after eager/compiled loss and update
  parity on the same microbatch. Reuse a persistent Inductor cache across matched
  arms.
- Checkpoint only at segment boundaries. Do not save duplicate base weights.
- Every launcher command is explicit and non-advancing: a passing gate reports
  the next authorized command but does not start it automatically.

## Artifact Contract

Every stage writes immutable, non-overwriting outputs containing:

- model, tokenizer, parent-adapter, training-data, evaluation-data, and frequency
  hashes;
- exact arm, seed, schedule, quantizer, task split, lengths, row order, labels,
  optimizer, batch layout, precision, backend, and compile metadata;
- raw per-example prediction, canonical target, NLL sum, target-token count,
  metric inputs, group ID, variant, task, length, and distance;
- raw per-checkpoint training log and adapter/frequency receipts;
- deterministic summary and bootstrap configuration;
- explicit status `valid`, `invalid`, `negative`, or `positive` without deleting
  failed arms.

Code hashes are execution provenance, not a requirement that historical and new
arms share identical implementation bytes. Scientific validity comes from the
model/data/objective/arm contract and verified output semantics.

## Claim Ladder

1. Existing temporal result only: EVQ+LoRA improves external long-position
   language-model NLL on a mature 8B checkpoint.
2. Positive Stage 0: the existing end-to-end EVQ+LoRA pipeline also shows
   source-dependent long-context capability relative to Geo+LoRA.
3. Positive Stage 1: matched 8K-only task adaptation transfers farther under the
   EVQ pipeline than the Geo pipeline.
4. Positive Stage 2: under the tested seed and protocol, midpoint EVQ outperforms
   midpoint Geo, isolating a cosh-shape contribution.

None of these levels establishes universal long-context SOTA, production
deployment readiness, a global tau optimum, or a LoRA-rank theorem.

## Non-Goals

- No YaRN or other inference-time scaling in the primary path.
- No 16K/32K training in this design.
- No teacher/student representation distillation.
- No rank, tau, base, task-mixture, or decoding sweep after seeing results.
- No paper-number replacement or silent relabeling of the historical table.
- No automatic extra seeds or supporting benchmark zoo.
- No GPU launch as part of design or implementation validation.
