# ICLR 2027 strong-evidence wrappers

This directory implements the thin execution layer specified by
[STRONG_EXPERIMENT_PLAN_20260915.md](../../docs/research/next_stage_20260912/STRONG_EXPERIMENT_PLAN_20260915.md).
It reuses the existing table, RULER, generation and report kernels. Code or data
being ready is not a completed model experiment.

## Current status routing (2026-09-16)

Current completed extreme/natural results are in the
[Pro6000 result owner](../../docs/research/next_stage_20260912/PRO6000_EXTREME_NATURAL_QA_RESULTS_20260916.md).
GLM is downloaded and its S4 queue has started; Qwen official static YaRN has a
complete generation receipt. These do not yet supply an uninspected paired conclusion.
See the [timestamped execution snapshot](../iclr2027_three_track_sprint_20260915/SERVER_TASK_LAYERS.md).

The older implementation table and128K package below preserve their original
specification and dated states, not today's queue. In particular, Llama S16 is
completed, NCP is completed, and "YaRN parked" does not describe Qwen's new arm.
Llama clean32K X8 remains a separate unexecuted comparison.
[Next revision preparation](../../docs/research/next_stage_20260912/PAPER_NEXT_REVISION_PREPARATION_20260916.md)
contains research priorities without adding GPU jobs.

## Completed report snapshot

[Portable reports](reports/README.md) preserve the completed OLMo clean and natural-QA,
Llama matched-dose/native/LongBench-v2, and Native exploration results.
[Paper-value analysis](../../paper-2027/research/COMPLETED_EXPERIMENTS_PAPER_VALUE_20260915.md)
records the applied manuscript changes, including the main-text NCP subsection.

## Original entry points and implementation snapshot

| Entry | Plan items | State | GPU behavior |
|---|---|---|---|
| `prepare_clean_transfer.py` | X1, X2, X5 | implemented and tested | CPU-only; Full-13 source-order, unpadded panels |
| `run_clean_matrix.py` | X1, X2, X5 | implemented and tested | plan-only by default; `--execute` required |
| `run_clean_c.py` | X4 | implemented and tested | plan-only by default; only generates C |
| `prepare_natural_long.py` | X3, X7 | implemented and tested | CPU-only; never downloads data |
| `run_natural_long.py` | X3, X7 | implemented and tested | plan-only by default; `--execute` required |
| `summarize_matrix.py` | all completed matrix cells | implemented and tested | report-only; refuses mixed contracts |
| `four_model_yarn_full13.py` | four-model Full-13×10 T/P/static-YaRN | strict reuse/resume/report orchestrator; plan-only by default | reuses Llama/OLMo, resumes Qwen T/P, adds only GLM back5, then runs YaRN |
| `prepare_olmo_naturalqa631.py` | OLMo QA transfer | implemented; server assets frozen | CPU-only; preserves the historical 631-row source pool |
| `run_olmo_naturalqa631.sh` | OLMo QA transfer | implemented and dry-run guarded | `--execute` required; TailSpline then canonical MrPro |
| `run_olmo_qa_then_ruler200.sh` | requested OLMo order | implemented and dry-run guarded | QA first; clean 16K RULER-13×200 last |
| `prepare_pro6000_128k_queue.sh` | X2, X6 | implemented; GPU assets not yet prepared | CPU-only Qwen asset/table freeze plus both-model validation |
| `validate_pro6000_queue_on_4080.sh` | X2, X6 engineering | implemented; pending 4080 execution | disposable integration canaries; never writes formal generations |
| `run_pro6000_128k_queue.sh` | X6 then X2 | implemented; pending Pro 6000 | requires sm_120 and >=80,000 MiB; formal 128K queue |

The existing 48GB+ X6 entry remains
[`run_llama_s16_128k_gate_48gb.sh`](../iclr2027_three_track_sprint_20260915/run_llama_s16_128k_gate_48gb.sh).
YaRN/X8 remains parked and has no entry in this directory.

## Historical Pro6000 128K package specification

The expensive-machine queue contains only two unconditional experiments:

1. Llama-3-8B, S=16, 128K: frozen Full-13 x 10 plus ten 128K LM documents,
   TailSpline and MrPro.
2. Qwen2.5-3B, S=4: clean Full-13 x 50 at 128K first and 64K second,
   TailSpline and MrPro. The 64K cell identifies the 2L response; the 128K
   cell is the primary 4L endpoint.

Run the CPU asset freeze once on the cloned data disk, then use the cheaper-GPU
canary before cloning it again:

```bash
bash experiments/iclr2027_strong_evidence_20260915/prepare_pro6000_128k_queue.sh
bash experiments/iclr2027_strong_evidence_20260915/validate_pro6000_queue_on_4080.sh
```

The validation entry targets the existing 32GB 4080 vGPU and refuses a
standard 16GB physical RTX 4080; it is not a claim that every 4080 can host the
Llama BF16 32K canary.

On the RTX PRO 6000 Blackwell machine, the only formal command is:

```bash
bash experiments/iclr2027_strong_evidence_20260915/run_pro6000_128k_queue.sh
```

The destination preflight requires PyTorch >=2.7, CUDA >=12.8, `sm_120`, BF16,
a forced Flash-SDPA smoke, at least 80,000 MiB, exact asset hashes and both
analytic table receipts. Direct, 64K-chunk and 32K-chunk prefill compete on the
actual GPU; generation and LM choose independently. A 96GB card may peak near
90GB (6% free-memory floor), but selection is by end-to-end time rather than by
allocated bytes. Hardware utilization, memory, clocks, power and temperature
are sampled every two seconds. When direct prefill wins, a second canary compares
sequential batch-1 generations with unpadded, exactly equal-length groups:
Llama tests batches 2/4 and Qwen tests 2/4/8, each on its own longest eligible
128K shape. A larger batch is used only if generated token IDs are identical,
speedup is at least 5%, and the same memory floor holds; chunked prefill remains
batch 1. Missing groups and OOMs fall back safely rather than stopping the queue.

NVFP4/MXFP8, quantized KV, TensorRT-LLM, FlexAttention/FA4 rewrites and
multi-prompt padding are not part of this BF16 comparison: they change numerical
or input/runtime identity and have no completed parity receipt for the custom
static RoPE tables. Conditional 128K expansion and InfiniteBench remain stopped
until the frozen Llama gate has a report; the queue never launches them itself.

## OLMo immediate package

The OLMo QA asset owner is
`/root/autodl-tmp/today_rope_plan_20260914/tailspline_olmo_s4_naturalqa631`.
It contains 631 rows from five Natural-QA tasks, all beyond the 4K Native length.
Repeated rendered prompts are retained and registered; uncertainty is clustered
by source context.

The clean 16K×200 owner is
`/root/autodl-tmp/today_rope_plan_20260914/tailspline_olmo_s4_16k_ruler200_clean`.
Its CPU assets are generated with the frozen source-order seed `20261101` and
QA offset `5600`. The earlier proposed offset `7000` is invalid because the
current SQuAD source contains 5,928 rows; this correction occurs before any GPU
output and is recorded in the plan.

To inspect the order without running a model:

```bash
bash experiments/iclr2027_strong_evidence_20260915/run_olmo_qa_then_ruler200.sh
```

The `--execute` form refuses to begin before the current NIAH Full20 completion
marker exists. It then runs QA before RULER-200. No other follow-on task is
embedded.

## Contract rules

- Full-13 clean panels use exact model tokenization, source order and no content
  padding. TailSpline and MrPro share prompt IDs and decoding.
- Runners take a non-blocking single-GPU lock and validate completed prefixes,
  table receipts and runtime contracts before reuse.
- LongBench-v2, InfiniteBench dialogue QA and InfiniteBench book QA retain their
  own official scoring adapters; RULER substring scores are not reused.
- Matrix summaries keep clean/classic, S2/S4/S16, benchmark families and metrics
  separate. Cross-model aggregation is descriptive only.
- Reports and manifests must not contain personal workstation paths. Repository
  artifacts use relative paths; remote raw evidence may use its stable server path.

## Four-model static-YaRN Full-13 comparison

`run_four_model_yarn_full13.sh` now delegates to `four_model_yarn_full13.py`.
The existing `run_yarn_full_after_quick.sh` chain remains unchanged: it waits for
the quick-YaRN completion marker and then invokes the strict full runner.

The frozen model set is Llama-3-8B, Qwen2.5-3B, OLMo-2-1B and GLM-4-9B;
Qwen2.5-1.5B is explicitly excluded. Llama/OLMo reuse the first ten source-order
rows per task from their completed 200-per-task runs. Qwen uses only
`tailspline_qwen25_s4_128k_ruler10_clean`, resuming its partial TailSpline arm and
running its missing MrPro arm. GLM first proves that current assets5 are exactly
rows 0--4 of assets10, then runs only rows 5--9 (65 generations per T/P arm) in
separate supplement directories. Existing GLM main outputs are read-only.

Every model receives a canonical 130-row merge in assets10 order and a three-arm
paired report. The report refuses mismatched row IDs, prompt hashes, task/length,
table FP32 hash or gain. Static YaRN is a frozen inference table comparison, not
evidence for a checkpoint trained with YaRN.

## Qwen response analysis

[Task decomposition and same-scale geometry](../../docs/research/reviews/QWEN_ALLOCATION_RESPONSE_ANALYSIS_20260916.md)
explain the current Qwen observations and distinguish measured behavior from
mechanistic hypotheses. The portable CPU analysis is in `qwen_diagnosis/`.
It does not change tables, launch GPU work, or alter the manuscript.

## Fixed S16 confirmation blocks

[`llama_s16_confirmation.py`](llama_s16_confirmation.py) prepares four independent
Full13 × 10 blocks at Llama 8K→128K (S16). Default invocation only prints the plan;
`--prepare` freezes CPU assets, while `--execute` is a separate GPU opt-in after
the existing InfiniteBench and Qwen append queue. It does not modify that queue.
QA source offsets are 5810/5820/5830/5840, disjoint from the gate's 5800–5809;
each block has a distinct source-seed range. All four blocks are fixed before
evaluation; no score-dependent stopping or candidate selection is implemented.
The runner reuses the gate's installed tables and measured batch/chunk runtime,
resumes individual arms, and does not rerun PPL or a preflight benchmark.

The primary report is independent **confirm40** (520 rows/arm). A separate
**gate10 + confirm40 cumulative50** summary includes the already-observed gate
and must not be described as an independent 50/task confirmation. Each paired
block is 260 generations; the approximate two-hour target is an estimate, not
a timeout that discards slow rows. Existing gate PPL10 remains separate.
