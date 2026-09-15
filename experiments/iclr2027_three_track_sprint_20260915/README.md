# ICLR 2027 three-track sprint execution

This directory contains the completed sprint evidence, CPU-prepared assets,
and the active or parked follow-up launchers.

Current operations no longer follow the original two-queue order. Use
[SERVER_TASK_LAYERS.md](SERVER_TASK_LAYERS.md) for the live/parked/high-memory
handoff and [reports/README.md](reports/README.md) for completed portable reports.
The old queue scripts are retained for provenance and must not be launched as the
current queue.

## Current operations

- Running: Llama NIAH Full20, TailSpline then MrPro, with no automatic successor.
- Prepared for a GPU with at least 45,000 MiB: Llama S16 128K gate.
- Parked: YaRN follow-ups. BM is excluded from this sprint by the author's decision.
- Completed: clean TailSpline/MrPro, Natural-QA631, Native references, Native-Z5
  follow-ups and the 39-row classic runtime probe; use their result owners.

These are the dated handoff states; [SERVER_TASK_LAYERS.md](SERVER_TASK_LAYERS.md)
defines the completion checks and authoritative execution routes. New experiment
specifications are linked from the [current research index](../../docs/research/next_stage_20260912/index.md).

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
| E4 strong static baseline | Parked: YaRN follow-ups are not scheduled; BM is excluded from this sprint |
| E5 Native reference | Native PPL summary and Native-8K RULER complete; task macro `0.918846`, Native-minus-TailSpline CI95 `[-0.019231,+0.061410]` |

Native-Z5 is a completed, separately identified checkpoint-calibrated exploration.
Its V1, consensus and all-50 refit outcomes are in the
[Native-Z5 result owner](../../docs/research/next_stage_20260912/NATIVE_Z5_EXPLORATION_RESULT_20260915.md).
It does not replace E5's unchanged-Native reference.

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

Compact completed reports are indexed in [reports/README.md](reports/README.md). Large generation
streams remain on the experiment server and are identified by SHA256 in the
Llama result owner; score-only changes reuse those streams.

## GPU entry points and historical wrappers

- Full20: [run_mrrope_niah_heatmap_full20.sh](run_mrrope_niah_heatmap_full20.sh).
- S16 128K: [run_llama_s16_128k_gate_48gb.sh](run_llama_s16_128k_gate_48gb.sh).
- Parked YaRN: [run_naturalqa_yarn.sh](run_naturalqa_yarn.sh), dry-run by default.

Use the matching A--C layer in [SERVER_TASK_LAYERS.md](SERVER_TASK_LAYERS.md)
before any launch. `run_original_gpu_queue.sh` and `run_clone_gpu_queue.sh`
encode the retired two-GPU order and are retained solely for provenance.

## Field-gap follow-ups

[Current arrangement](../../paper-2027/research/COMPARATIVE_GAP_AND_DECISION_MAP_20260915.md) preserves queue ownership; the frozen Natural-QA T/P comparison is complete and registered. [A1 YaRN launcher](run_naturalqa_yarn.sh) is prepared separately, prints its action by default and is not queued. E1 V2 retains `QUALIFIED_ONLY` after E0; the 39-row probe cannot establish full T/C runtime equivalence. [Discrete CPU examples](verify_discrete_kernel_equivalence.py) verify mathematical boundaries only.
