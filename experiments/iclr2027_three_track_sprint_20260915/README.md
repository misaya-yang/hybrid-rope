# ICLR 2027 three-track sprint execution

This directory turns the 2026-09-15 sprint plan into two non-overlapping GPU
queues and CPU-only frozen assets. It does not add a curve search or a model.

## Queue ownership

- Original GPU: finish clean TailSpline/MrPro, then Natural-QA631 and Native-Z5.
- Cloned GPU: 39-row classic TailSpline batch sensitivity, a one-arm Native 8K
  task reference, clean 32K YaRN on the exact 2,600 prompts, then classic YaRN
  at batch 1.
- BM is excluded by the author's 2026-09-15 decision.

The clean RULER runtime is batch 1 with exact unpadded prompt IDs. This is the
actual working runtime after masked left-padding failed in the installed
Flash-SDPA stack; the sprint proposal's older batch-2 description is not used as
present-state evidence.

## E0--E5 coverage

| Sprint item | Execution owner |
|---|---|
| E0 classic runtime identity | Complete: 39-cell TailSpline batch-2 replay has zero score drift versus batch 1; six texts differ, so this is not bitwise equivalence |
| E1 matched-displacement shape | Completed raw plus `e1_experimental_audit.py`; E0 remains the runtime-sensitivity qualifier |
| E2 clean RULER-200 | TailSpline/MrPro complete: `0.682660/0.565436`, delta `+0.117224`, CI95 `[+0.103231,+0.131148]`; clean YaRN remains optional and unscheduled |
| E3 Natural-QA631 | Complete: T/P 41.0791/40.8834% F1; +0.1957pp, cluster CI [−1.5311,+1.8864]pp; all 631 questions and both native strata reported |
| E4 strong static baseline | YaRN only, on both clean and batch-1 classic contracts; BM is excluded |
| E5 Native reference | Native PPL summary and Native-8K RULER complete; task macro `0.918846`, Native-minus-TailSpline CI95 `[-0.019231,+0.061410]` |

Native-Z5 is an additional checkpoint-calibrated question requested after the
sprint proposal; it follows Natural-QA on the original GPU and does not replace
E5's unchanged-Native reference.

## CPU preparation

```bash
experiments/iclr2027_three_track_sprint_20260915/run_cpu_reports.sh
```

CPU preparation writes the 39-row fixed probe, a path/hash/status ledger, the
exact sprint-math receipt, the 15-part theory-deepening operator receipt, the
completed E1 audit, and the matched Native-8K PPL summary. It does not load a
checkpoint or touch CUDA.

The interpretation owner for the theory receipt is
[THEORY_DEEPENING_CPU_VERIFICATION_20260915.md](../../docs/research/next_stage_20260912/THEORY_DEEPENING_CPU_VERIFICATION_20260915.md).

Compact completed reports are stored in [reports](reports/). Large generation
streams remain on the experiment server and are identified by SHA256 in the
Llama result owner; score-only changes reuse those streams.

## GPU entry points

After the data disk has been cloned and each server endpoint is known, launch
exactly one script on each server:

```bash
experiments/iclr2027_three_track_sprint_20260915/run_original_gpu_queue.sh
experiments/iclr2027_three_track_sprint_20260915/run_clone_gpu_queue.sh
```

Both scripts are restartable at completed-arm boundaries and preserve existing
raw generations. Do not run both scripts on the same GPU.

## Field-gap follow-ups

[Current arrangement](../../paper-2027/research/COMPARATIVE_GAP_AND_DECISION_MAP_20260915.md) preserves queue ownership; the frozen Natural-QA T/P comparison is complete and registered. [A1 YaRN launcher](run_naturalqa_yarn.sh) is prepared separately, prints its action by default and is not queued. E1 V2 retains `QUALIFIED_ONLY` after E0; the 39-row probe cannot establish full T/C runtime equivalence. [Discrete CPU examples](verify_discrete_kernel_equivalence.py) verify mathematical boundaries only.
