# Server task layers

This file is the execution handoff for the current experiment server. It routes
operators to runnable work; it is not another experiment plan or result
narrative.

## Current observed snapshot: 2026-09-16T08:02:47Z

This snapshot was read without changing processes, queues, files or GPU work.
The active PRO6000 endpoint is `ssh -p 51638 root@connect.westd.seetacloud.com`.
The old32GB endpoint and its shutdown instructions below are historical.

| Work | Observed state | Interpretation |
|---|---|---|
| GLM model download | `DOWNLOAD_COMPLETE_VERIFIED` at `/root/models/GLM-4-9B-0414/DOWNLOAD_RECEIPT.json` | Download complete; not a model evaluation result |
| Qwen S4 official static YaRN | `official_yarn/run/status.json`: COMPLETE,40 generation rows,5 LM rows | Generation complete; paired result must come from its report |
| GLM S4/128K | `run_glm4_9b_s4_128k_queue.sh` active; no queue/QA completion report observed | In execution, no result claimed |
| Llama S16 gate and natural tasks; Qwen S8 health and S4 QA | Completed portable reports | [Result owner](../../docs/research/next_stage_20260912/PRO6000_EXTREME_NATURAL_QA_RESULTS_20260916.md) |
| Extra10/task RULER block | User reports assignment to another experiment agent | Completion and exact block identity remain with that owner |

The current GLM script uses Full13×5 per arm, PPL5 and En.QA. It preserves
partial RoPE handling; code/runtime readiness is distinct from task success.
This documentation update does not authorize a new launch or reorder existing work.
[Next manuscript preparation](../../docs/research/next_stage_20260912/PAPER_NEXT_REVISION_PREPARATION_20260916.md)
contains proposed follow-ups only. Read fresh completion receipts before treating
this timestamped snapshot as current execution state.

## Historical32GB queue override: 2026-09-15 (superseded)

Everything below records earlier locations, contracts and queue states. Its
"running now", "ready only" and shutdown instructions are not the current queue.


The currently authorized 32 GB sequence is X4 matched-dose control, then X5
Llama clean 8K Native/TailSpline/MrPro, then the OLMo Native Contrastive
Proximal (NCP) development run. NCP reuses the completed 780-row Native output
and generates only 780 new NCP rows. Its tested entry point is
`experiments/native_contrastive_proximal_20260915/run_ruler_gate.sh --execute`;
its output root is
`/root/autodl-tmp/today_rope_plan_20260914/olmo_native_contrastive_proximal`.
After NCP has a complete paired report, save outputs, verify that no other jobs
are active, and shut down this host. This live override supersedes older queue
snapshots below but does not authorize the parked YaRN or 48 GB tasks.

The implementation files remain at their stable paths because the active runner
and existing receipts reference them. The A--E layers below are the control
plane; physically moving live scripts or evidence directories would add breakage
risk without improving the scientific separation.

## Fixed locations

| Item | Location |
|---|---|
| Current host | `ssh -p 37849 root@connect.westc.seetacloud.com` |
| Server repository | `/root/autodl-tmp/hybrid-rope` |
| Experiment data root | `/root/autodl-tmp/today_rope_plan_20260914` |
| Sprint receipts root | `/root/autodl-tmp/iclr2027_three_track_sprint_20260915` |
| Llama checkpoint | `/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct` |
| Runner Python | `/root/miniconda3/bin/python` (explicit path for non-login SSH commands) |

Live state was inspected read-only at `2026-09-15T08:17:36Z`. The durable state tests
below take precedence over that snapshot.

The server repository is a copied code snapshot without `.git`. Do not use
`git pull` or a remote `git status` as a synchronization check; deploy a known
local commit/snapshot explicitly before launching newly added code. A cloned
data disk should preserve the fixed `/root/autodl-tmp` paths above.

## A. Running now

### Llama S4 TailSpline versus MrPro, NIAH Full20

- Purpose: confirm the 3-repeat NIAH pilot with 20 fresh examples per
  length/depth cell, retaining the pilot as separate completed evidence.
- Contract: `4 lengths x 9 depths x 20 = 720` prompts per arm; lengths are
  8K/16K/24K/32K; batch size 1; TailSpline runs before MrPro.
- Runner:
  `experiments/iclr2027_three_track_sprint_20260915/run_mrrope_niah_heatmap_full20.sh`
- Remote owner:
  `/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_mrrope_niah_heatmap_full20`
- Snapshot: TailSpline was at `665/720`, MrPro at `0/720`; the evaluator for
  this exact root was alive. GPU utilization was 100% with 23,839 MiB used on
  the 32,760 MiB GPU. The completion marker and final report were absent.

Determine state in this order:

1. Complete only if `complete.txt` exists, both arm `status.json` files equal
   `{"status":"COMPLETE","rows":720,"lm_rows":0}`, and the report exists at
   `reports/tailspline_vs_mrpro_niah_heatmap_full20.json`.
2. Running if `pgrep -af 'run_mrrope_niah_heatmap_full20|recovery_v2_eval'`
   shows this exact root and `nvidia-smi` shows active work. `queue.pid` alone is
   not proof because old experiment roots contain stale PID files.
3. Failed if neither condition holds. Inspect `logs/queue.log`, the current arm
   log, and `runs/<arm>/live.json`. Restart only through the same runner and
   `launch.lock`; never start a second evaluator beside a live one.

There is deliberately no automatic follow-on GPU job after Full20. Do not use
an old supervisor to fill the gap.

The read-only [status helper](server_task_status.sh) reports file presence, row
counts and PID liveness. Its `PRESENT`, `completion_marker_present` and
`pid_alive_unverified` labels are observations; final completion and running
state require the checks above.

## B. Ready on a 32 GB GPU, but parked

There is no automatically authorized follow-on after Full20. The strong-evidence
wrappers and OLMo assets below are prepared, but their default commands are dry
runs; code readiness does not authorize GPU execution.

| Task | Entry point | Output root | State / launch rule |
|---|---|---|---|
| OLMo S4 Natural-QA631, TailSpline/MrPro | `experiments/iclr2027_strong_evidence_20260915/run_olmo_naturalqa631.sh --execute` | `/root/autodl-tmp/today_rope_plan_20260914/tailspline_olmo_s4_naturalqa631` | CPU panel complete: 631 rows, 524 source clusters. A four-row canary selects exact batch4 left-padding or falls back to batch1. GPU execution not started. |
| OLMo QA then clean16K Full13×200 | `experiments/iclr2027_strong_evidence_20260915/run_olmo_qa_then_ruler200.sh --execute` | QA root above, then `/root/autodl-tmp/today_rope_plan_20260914/tailspline_olmo_s4_16k_ruler200_clean` | CPU RULER asset complete: 2,600 source-order unpadded rows. Wrapper enforces QA first and RULER-200 last; default invocation only prints the plan. |
| One YaRN arm on the frozen Natural-QA631 inputs | `run_naturalqa_yarn.sh --execute` | `/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_naturalqa631_yarn_a1` | Code-ready; not launched; default invocation is a dry run. Launch only after an explicit YaRN decision. Reuses completed TailSpline/MrPro generations. |

Do not label a proposed Native-Z successor as ready. The existing Native-Z5,
consensus, and all-50 refit are finished evidence; no new frozen Native-Z
launcher/contract is present in this sprint directory.

## C. Requires a cloned 48 GB or larger GPU

### Llama S16 128K gate

- Question: at 128K, compare frozen TailSpline and MrPro using Full-13 RULER
  (`10/task`, 130 prompts per arm) plus 10-document 128K PPL.
- CPU assets are complete at
  `/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s16_128k_gate`.
- Durable ready marker: `assets/ready.json` with status
  `TAILSPLINE_LLAMA_S16_128K_ASSETS_READY_V1`, `ruler_rows=130`,
  `ppl_documents=10`, and `length=131072`.
- GPU entry point:
  `experiments/iclr2027_three_track_sprint_20260915/run_llama_s16_128k_gate_48gb.sh`
- Launch from `/root/autodl-tmp/hybrid-rope` after the data disk is cloned to a
  machine exposing at least 45,000 MiB VRAM. The script refuses smaller GPUs
  with exit code 75.
- The runner first benchmarks direct/chunked prefill on the actual GPU, selects
  generation and LM strategies separately, then runs TailSpline and MrPro.
  Do not hard-code a chunk size from the 32 GB benchmark and do not bypass the
  canary.
- Complete only if `complete.txt` exists, each arm has exactly
  `{"status":"COMPLETE","rows":130,"lm_rows":10}`, and
  `reports/tailspline_vs_mrpro_s16_128k_gate.json` exists.

The 32 GB server contains no 128K GPU output under `runs/`; this is expected,
not a failed run. Asset preparation can be reproduced CPU-only with
`prepare_llama_s16_128k_assets.sh`, but it should not be repeated on a normal
handoff because `assets/ready.json` already validates the frozen assets.

## D. CPU and report-only utilities

| Owner | Entry / artifact | Current use |
|---|---|---|
| Sprint math, theory checks, E1 audit, Native PPL summary | `run_cpu_reports.sh` | Complete under `/root/autodl-tmp/iclr2027_three_track_sprint_20260915`; no checkpoint or CUDA. Re-run only if an input report changes. |
| 32-document ProofPile parity analysis | `proofpile_ppl_parity.py`; remote report `/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_s4_mrrope_niah_heatmap/reports/proofpile32_ppl_curve.json` | Completed score-only reanalysis of existing raw LM rows; no GPU rerun needed. |
| Full20 report | `mrrope_niah_heatmap_report.py` | Called by the Full20 runner after both arms finish; do not run against a partial arm. |
| 128K report | `llama_s16_128k_report.py` | Called by the 48 GB runner after both arms finish. |
| Prefill selector | `benchmark_prefill_chunks.py` | GPU canary utility embedded in the 128K runner, not an independent experiment and not CPU-only despite living beside report scripts. |
| Copied proposals | `/root/autodl-tmp/today_rope_plan_20260914/plans/` | Reference material only; file presence here never establishes queue priority or authorization. |

## E. Finished evidence and archived routes

### Canonical completed owners

These roots are evidence inputs. Read their compact report first; do not rerun
them to reconstruct a number.

| Evidence | Remote owner / report |
|---|---|
| Llama clean Full-13 32K, 200/task | `tailspline_llama_s4_32k_ruler200_clean/reports/tailspline_vs_mrpro_full13_32k_200_per_task_clean.json` |
| Llama clean Full-13 16K, 50/task | `tailspline_llama_s4_16k_ruler50_clean/reports/tailspline_vs_mrpro_full13_16k_50_per_task_clean.json` |
| Llama classic 8K/16K/32K plus PPL | `tailspline_llama_s4_classic/reports/tailspline_vs_mrpro_classic.json` |
| Llama Natural-QA631 | `tailspline_llama_s4_naturalqa631/reports/tailspline_vs_mrpro_naturalqa631.json` |
| Original 3-repeat NIAH heatmap and ProofPile32 reanalysis | `tailspline_llama_s4_mrrope_niah_heatmap/reports/` (completed pilot; separate Full20 confirmation pending, retain both) |
| OLMo classic transfer | `tailspline_olmo_s4_classic/reports/tailspline_vs_mrpro_classic.json` |
| Qwen 32K/64K report | `tailspline_qwen25_s2_32k64k/reports/tailspline_vs_mrpro_32k64k.json` |
| Native-Z5 existence test and follow-ups | `olmo_native_z5_enhancement/reports/`; all three completion markers exist, but the all-50 refit stopped before fresh task confirmation |
| Fixed-u decisive test | `olmo_fixed_u_decisive/reports/fixed_u_vs_fixed_m_existing6.json` |
| 32K prefill/runtime canary | `prefill_chunk_benchmarks/llama32k_32gb_v2.json` |
| Matched-dose control | `tailspline_llama_s4_matched_dose_c/reports/tailspline_vs_dose_control_c_classic.json` |

### Do not resume these legacy or superseded roots

- `tailspline_llama_s4_first`, `tailspline_llama_s4_complete108`: early small
  panels, superseded by clean Full-13 owners.
- `tailspline_llama_s4_32k_ruler200`: pre-clean diagnostic shards, superseded
  by `tailspline_llama_s4_32k_ruler200_clean`.
- `tailspline_llama_s4_32k_full500`, `tailspline_llama_s4_full50`: partial
  expansion/assets, not the accepted evaluation contract.
- `tailspline_qwen25_s2_unified_full324`: explicitly archived after one arm;
  do not invent the missing comparison by resuming it.
- `qwen3_native32k_batch_canary`, `qwen3_native32k_full108`: completed Native
  throughput/baseline probes, not TailSpline-versus-MrPro evidence and not a
  pending queue.
- `qwen25_s2_full324`, `qwen25_s4_full`,
  `tailspline_qwen25_s4_unified_core6`: partial/candidate-era routes, not current
  paper evidence owners.
- `qwen_then_llama`, `qwen_core6_then_llama32_full500`,
  `llama_s4_factorial_gap48`, `native_ppl_controls`: legacy supervisors or
  abandoned queue shells. Their PID files are not active-work evidence.
- `tailspline_llama_s4_classic_strong_baselines`: table storage only; it is not
  a queued experiment.

Never launch `run_original_gpu_queue.sh` or `run_clone_gpu_queue.sh` as a current
handoff. They encode an older two-GPU sprint order whose component jobs are now
completed, superseded, or deliberately parked; running them can reopen YaRN or
Native-Z work that is not in the current queue.

## Handoff decision rule

1. If layer A is healthy, leave it alone.
2. If layer A completes, register/analyse its report before selecting new GPU
   work; do not infer that a stale PID means a queue continues.
3. On this 32 GB host, layer B remains parked unless the author explicitly
   promotes YaRN.
4. On a cloned 48/96 GB host, layer C is the only prepared large-memory launch.
5. Treat layer E as immutable evidence. New scoring may reuse its raw rows;
   generation should not be repeated unless the evaluation contract itself
   changes.
