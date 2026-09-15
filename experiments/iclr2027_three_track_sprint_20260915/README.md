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

## CPU preparation

```bash
experiments/iclr2027_three_track_sprint_20260915/run_cpu_reports.sh
```

CPU preparation writes the 39-row fixed probe, a path/hash/status ledger, the
exact sprint-math receipt, the completed E1 audit, and the matched Native-8K PPL
summary. It does not load a checkpoint or touch CUDA.

## GPU entry points

After the data disk has been cloned and each server endpoint is known, launch
exactly one script on each server:

```bash
experiments/iclr2027_three_track_sprint_20260915/run_original_gpu_queue.sh
experiments/iclr2027_three_track_sprint_20260915/run_clone_gpu_queue.sh
```

Both scripts are restartable at completed-arm boundaries and preserve existing
raw generations. Do not run both scripts on the same GPU.
