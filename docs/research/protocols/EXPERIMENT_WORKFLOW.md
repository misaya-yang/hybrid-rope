# Experiment workflow

This is the reusable execution contract. Priorities and current authorization
come from the [research index](../next_stage_20260912/index.md) and the author's
current task. Source and implementation owners are in the
[experiment index](../../../experiments/index.md).

## Select a scientific question

Choose whether the task tests practical deployment quality, pure allocation,
residual shape at equal displacement, learned coordinate use, or a proposed
mechanism. State the target endpoint and the smallest comparison that can
change the conclusion. Use an existing completed result when it answers the
question. Preserve the full family of outcomes rather than selecting only wins.

| Comparison | What it can establish | Essential controls |
|---|---|---|
| TailSpline vs MrPro with matched bands | Effect of the complete internal allocation | Weights, reference grid, endpoints, outer bands, gain, input and decoder |
| TailSpline vs equal-displacement C | Effect beyond total log displacement | All preceding controls plus the matched displacement |
| TailSpline vs official runtime YaRN | Quality of the two deployment recipes | Checkpoint, input, precision, decoder and metric; retain each declared recipe |
| Native-window intervention | Change within the original supported length | Original Native comparator, actual lengths, gain and prediction/task identity |
| Training/adaptation pair | Benefit of learning with the allocation | Initialization or parent, data/order, budget, optimizer and evaluation as actually matched |

LM quality, synthetic retrieval and natural downstream tasks remain distinct
endpoints. Native Natural QA is the preferred real-task target; a RULER/PPL
improvement does not replace its outcome.

## Reuse the existing path

Read the chosen owner's README/index and the specific runner before invoking
it. The fixed-table family supplies constructors and resident evaluation;
strong-evidence wrappers supply clean Full-13, natural tasks and paired reports;
Kanana has its own official-runtime recipe; native and learning families have
separate contracts. Archived queues are not a current execution order.

Before new model execution, record the necessary comparison identity in the
experiment's own plan: checkpoint and tokenizer, original RoPE reference,
constructor parameters and table, gain, lengths, input set and overlap with
prior panels, precision, generation cap/stop rule, endpoint and aggregation.
Use existing manifests rather than duplicating them. Constructor selection is
weight-independent; checkpoint inspection is not a calibration shortcut.

## Prepare and run only what is authorized

- A plan, downloaded model, prepared input or CPU check is not a model result.
  Respect an entrypoint's `--execute` boundary. Some older shell wrappers run
  immediately: inspect their behavior instead of assuming all scripts dry-run.
- On an authorized host, inspect current jobs and available resources, then
  select the relevant existing root. A stale port or PID file is not live status.
  Do not stop another experiment or restart a completed queue.
- Validate concrete changed risks: input indexing, runtime table replacement,
  scoring, precision or prefill behavior. Do not impose redundant hash/smoke
  rituals on unchanged, author-confirmed assets.
- Reuse matching arms and resume missing records through the established
  mechanism. Distinguish reused subsets from independent confirmations.
- CPU-only numerical/report work must not load CUDA accidentally. Follow the
  owner scripts' device/thread controls; CPU affinity IDs are host-specific.

## Report the actual endpoint

Completion requires the specified arms, expected input coverage and paired
report. Reconcile missing/duplicate IDs or changed contracts before aggregating.
When comparing Full-13, retain task-equal scoring; NIAH is a subset, not another
independent replicate. For PPL, retain the actual token/document aggregation
and NLL-before-exponentiation convention of that protocol.

For QA, state the question/source weighting and resampling unit. In particular,
Native-QA99's question-within-task mean and its source-equal sensitivity are
different estimands. Do not interchange their point estimates or intervals.
Natural book QA retains actual input lengths and book/source identity.

Keep a result owner with the question, exact comparison, complete results,
uncertainty where computed, interpretation and raw/report paths. Record point
superiority separately from an interval excluding zero. Preserve null and
adverse results without treating every diagnostic as a veto over all endpoints.

## Promote evidence and finish

Update the relevant result owner and nearest index. If used in the paper, update
the [asset registry](../../../paper-2027/research/evidence/asset_registry.json)
and claim map; add portable inputs to the existing generator when needed.
Report-backed verification and raw-row rescoring are different evidence levels.

Keep checkpoints, large raw and tokenizer caches outside Git. Keep the code,
compact report, protocol identity and required dependencies portable. Run
`python3 scripts/check_repository_docs.py` after navigation/source updates.
Do not modify an active runner to make a manuscript source archive match:
`paper-2027/package_source.py` uses its recorded runtime snapshot and additions.

An execution handoff states what is complete, what is running (with observation
time), what remains, and the exact next authorized action. A failed local lookup
does not justify rerunning an experiment or inventing a manuscript limitation.
