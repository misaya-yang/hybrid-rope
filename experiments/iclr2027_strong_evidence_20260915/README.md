# ICLR 2027 strong-evidence wrappers

This directory implements the thin execution layer specified by
[STRONG_EXPERIMENT_PLAN_20260915.md](../../docs/research/next_stage_20260912/STRONG_EXPERIMENT_PLAN_20260915.md).
It reuses the existing table, RULER, generation and report kernels. Code or data
being ready is not a completed model experiment.

## Entry points

| Entry | Plan items | State | GPU behavior |
|---|---|---|---|
| `prepare_clean_transfer.py` | X1, X2, X5 | implemented and tested | CPU-only; Full-13 source-order, unpadded panels |
| `run_clean_matrix.py` | X1, X2, X5 | implemented and tested | plan-only by default; `--execute` required |
| `run_clean_c.py` | X4 | implemented and tested | plan-only by default; only generates C |
| `prepare_natural_long.py` | X3, X7 | implemented and tested | CPU-only; never downloads data |
| `run_natural_long.py` | X3, X7 | implemented and tested | plan-only by default; `--execute` required |
| `summarize_matrix.py` | all completed matrix cells | implemented and tested | report-only; refuses mixed contracts |
| `prepare_olmo_naturalqa631.py` | OLMo QA transfer | implemented; server assets frozen | CPU-only; preserves the historical 631-row source pool |
| `run_olmo_naturalqa631.sh` | OLMo QA transfer | implemented and dry-run guarded | `--execute` required; TailSpline then canonical MrPro |
| `run_olmo_qa_then_ruler200.sh` | requested OLMo order | implemented and dry-run guarded | QA first; clean 16K RULER-13×200 last |

The existing 48GB+ X6 entry remains
[`run_llama_s16_128k_gate_48gb.sh`](../iclr2027_three_track_sprint_20260915/run_llama_s16_128k_gate_48gb.sh).
YaRN/X8 remains parked and has no entry in this directory.

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
