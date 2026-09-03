# Seed-42 LoRA 16K-max QA three-arm result

Date: 2026-07-15
Evidence tier: supporting / mechanistic, single seed
Paper claim: false

## Decision

The registered gate is **negative**: on the full 303-example LongBench QA set,
EVQ-LoRA does not outperform the matched native-geometric LoRA control. Its
task-macro F1 is `0.1126`, versus `0.2110` for Native-LoRA, for a paired delta
of `-0.0984` with bootstrap 95% CI `[-0.1297, -0.0697]`.

This does **not** isolate a specifically long-range EVQ failure. The aggregate
loss is driven by prompts at or below the 8K training length. Above 8K, all
three arms are already near the QA floor and the EVQ-versus-Native difference
is inconclusive. The experiment therefore rejects the current practical claim
that this EVQ-LoRA adapter is a better 16K QA model, while leaving the narrower
long-range substrate mechanism unresolved.

## What the three arms mean

| Arm | Runtime frequency | Adapter |
|---|---|---|
| Base-Native | exact original Llama geometric RoPE | disabled |
| Native-LoRA | exact original Llama geometric RoPE | matched 300-step LongAlpaca LoRA |
| EVQ-LoRA | EVQ-Cosh, `tau=1.414` | matched 300-step LongAlpaca LoRA |

`native_geo` is the repository name for the exact original Llama geometric
frequency grid; it is not a third positional method. Base-Native and
Native-LoRA therefore differ only by whether the Native/Geo LoRA is enabled.

The two LoRA adapters share seed 42, the frozen training rows, 300 steps,
8,192-token training length, rank 64, alpha 128, and q/k/v/o targets. The
registered variable is the training/runtime frequency substrate.

## Protocol

- Model: Meta-Llama-3-8B-Instruct.
- Benchmark: LongBench v1 Qasper (200) and NarrativeQA (103), pinned source
  revision.
- Prompts: 303 total, 2,022--16,384 tokens; 252 complete prompts and 51
  document-only truncations. There are 194 prompts at or below 8K and 109 above
  8K.
- Inference: dense attention, raw extrapolation, no YaRN, greedy generation,
  identical prompt token IDs across arms.
- Headline metric: equal-weight task-macro QA F1. The paired bootstrap resamples
  within each task, then averages the two task means; 10,000 trials, seed 42.
- This is generation-only evaluation. It does not use teacher-forced NLL as a
  proxy for task accuracy.

## Registered full-set result

| Arm | Task-macro F1 | Pooled F1 | NarrativeQA F1 | Qasper F1 | Strict exact |
|---|---:|---:|---:|---:|---:|
| Base-Native | **0.2309** | **0.2895** | 0.0479 | **0.4140** | **10.89%** |
| Native-LoRA | 0.2110 | 0.2600 | **0.0579** | 0.3641 | 8.25% |
| EVQ-LoRA | 0.1126 | 0.1413 | 0.0227 | 0.2024 | 1.32% |

| Paired comparison | Task-macro delta F1 | Bootstrap 95% CI | Gate |
|---|---:|---:|---|
| EVQ-LoRA minus Native-LoRA | **-0.0984** | **[-0.1297, -0.0697]** | negative |
| EVQ-LoRA minus Base-Native | -0.1184 | [-0.1486, -0.0903] | negative |
| Native-LoRA minus Base-Native | -0.0199 | [-0.0416, 0.0015] | inconclusive |

No arm produced empty outputs. EVQ's deficit is therefore not an empty-string
or crashed-generation artifact.

## Length-stratified diagnostic

This split was computed after the registered headline result and is diagnostic,
not a new primary endpoint.

| Prompt range | n | Base-Native macro F1 | Native-LoRA | EVQ-LoRA | EVQ minus Native, 95% CI |
|---|---:|---:|---:|---:|---:|
| <=8K | 194 | 0.4946 | **0.5088** | 0.1749 | -0.3339 `[-0.4560, -0.1940]` |
| >8K | 109 | **0.0523** | 0.0299 | 0.0481 | +0.0182 `[-0.0336, 0.0711]` |
| 8K--12K | 47 | 0.0688 | 0.0306 | **0.0700** | +0.0394 `[-0.0309, 0.1099]` |
| 12K--16K | 62 | 0.0141 | **0.0220** | 0.0054 | -0.0166 `[-0.0406, -0.0009]` |
| exactly 16K | 43 | 0.0023 | **0.0078** | 0.0019 | -0.0059 `[-0.0143, 0.0007]` |

All three arms have zero strict-exact answers above 8K. The >8K subset is also
task-skewed: it contains 95 NarrativeQA examples but only 14 Qasper examples;
the <=8K subset contains only 8 NarrativeQA examples and 186 Qasper examples.
Consequently, the stratified task-macro values are useful for locating the
failure but should not be promoted to a balanced long-context benchmark.

The decisive pattern is:

1. EVQ-LoRA is already much weaker inside the training-length regime. On the
   186 <=8K Qasper examples, F1 is `0.2114`, versus `0.3877` for Native-LoRA and
   `0.4377` for Base-Native.
2. Beyond 8K, the benchmark reaches a common floor. EVQ has no statistically
   resolved advantage over Native-LoRA, but neither does it show the large
   aggregate deficit seen at <=8K.
3. The full-set negative gate is therefore mainly a readout/instruction-quality
   problem, not evidence that EVQ uniquely destroys retrieval after 8K.

## Reconciliation with the earlier PPL result

The same frozen adapters previously gave the following temporal-holdout PPL:

| Arm | 8K PPL | 16K PPL | 32K PPL |
|---|---:|---:|---:|
| Base-Native | 7.948 | 150.545 | 1492.915 |
| Native-LoRA | **6.817** | 108.958 | 991.475 |
| EVQ-LoRA | 10.068 | **24.068** | **127.911** |

There is no metric contradiction. EVQ substantially stabilizes average
next-token prediction under extrapolation, but this adapter is worse at ordinary
QA within 8K and none of the arms retains usable exact QA beyond 8K. Low
long-context PPL is therefore insufficient evidence of answer binding, answer
selection, or instruction readout.

The strongest evidence-backed explanation is a conversion bottleneck:

- Llama pretraining and instruction tuning were learned on the native frequency
  grid. EVQ is injected only during a short, rank-64 LoRA adaptation, creating a
  representation/readout shift that 300 LongAlpaca steps do not fully absorb.
- The EVQ arm's final LoRA training loss (`3.1090`) and 8K PPL (`10.068`) are
  both worse than Native-LoRA (`2.4817`, `6.817`), consistent with the large
  <=8K QA deficit.
- At long lengths, better local language-model likelihood can coexist with a
  failed discrete retrieval-and-generation chain. The present data show that
  coexistence; they do not identify which internal stage fails.

These are mechanistic interpretations, not causal proof.

## Recommended next experiment

Do not spend another large inference window scanning sparse-attention or YaRN
hyperparameters on these adapters. The cheapest decisive decomposition is an
inference-only `2 adapter states x 2 runtime frequency states` cross:

1. Base-Native and Base-EVQ (adapter disabled).
2. Native-LoRA under Native and EVQ runtime frequencies.
3. EVQ-LoRA under EVQ and Native runtime frequencies.

A fixed, paired subset of the same <=8K and 8K--12K QA examples is enough for
this diagnosis. If the Base-EVQ arm already collapses, the pretrained readout is
frequency-grid sensitive; if only EVQ-LoRA collapses, the learned adapter is the
main problem. Only after this decomposition should a matched short retrieval/QA
calibration run be considered. Any such training remains a new, separately
approved supporting experiment.

## Validity boundary and artifacts

- Single seed, two tasks, supporting/mechanistic only; no paper table or claim
  is changed.
- The suite is **16K-max**, not uniformly 16K. Exact-16K performance is at
  floor for every arm.
- LongBench answer quality is measured by the existing QA F1 scorer and strict
  exact diagnostic; this run does not claim official LongBench leaderboard
  comparability.
- Raw JSON, per-arm logs, the frozen data manifest, readiness receipt, and
  launcher log are stored under
  `results/qa16k_three_arm_s42_20260715/` (ignored by default).
- Total evaluated input: 7,195,329 tokens across three arms. Runtime was 29.04
  minutes of model execution on one RTX 5090; peak allocated CUDA memory was
  20.39 GB per arm.
