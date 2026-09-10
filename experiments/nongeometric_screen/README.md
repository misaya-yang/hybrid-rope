# Task-directed, frozen-weight RoPE experiments

The experiment question, ten candidates, ranking, and NLL/passkey/mixed-RULER
decision rules are in the [research plan](../../docs/research/NONGEOMETRIC_TEN_CANDIDATE_PLAN_20260909.md).
This implementation uses the pinned Qwen2.5-3B-Instruct checkpoint and reuses
the historical static-S4 MrRoPE-Pro inputs and scores. No model weights are updated.

Run modules from the repository root, not individual files from this directory:

```bash
python -m experiments.nongeometric_screen.worker --root RUN_DIRECTORY --history HISTORICAL_BM_DIRECTORY
```

`worker.py` keeps one model resident, executes explicit JSON jobs from `queue/`,
and writes per-input results immediately. Completed jobs are recorded in `done/`;
an explicit larger panel appends only missing rows under the same method contract.
`STOP` terminates the queue at an input boundary. Exceptions are preserved in
`failure.json`. The worker does not choose another model or launch a training job.

The initial static E3 candidates can run while the remaining construction code
is prepared. `capture.py` collects real complete-prefix Q/K/V states and exact
full-row normalizers. `select.py` changes selected-key contributions while holding
the omitted contribution fixed; this is an explicitly approximate proposal filter.
Full-model generation from prefill, ordinary NLL, and mixed RULER determine the
method's actual outcome. `project.py` constructs E7 using signed local output
responses, with the same local support in its matrix and finite control.

`operators.py` implements layer allocation, GQA-group allocation, and the E10
dual-frequency kernel. Its GQA assignments preserve the original two KV heads.
E10 averages logits before a single softmax, caches K160/V128, and temporarily
pads V for fused SDPA; it is not a free or full-model ensemble. The equivalent
factorization places the exact 1/2 weight on Q rather than rounding two sqrt(1/2)
factors. `distance_operator.py` implements E9 with compiled FlexAttention, two
block masks, and joint logsumexp normalization; it stores raw K and original V.
`checks.py`, `precision_check.py`, and `distance_checks.py` cover independent dense
formulas and the changed model/cache integration paths.

`prepare_long_sources.py` and `prepare_long_tokens.py` prepare true contiguous
PG19 test-book and Proof-Pile arXiv test-document prefixes. Source and tokenizer
identities are saved. `long_eval.py` measures 64K/128K tail-token NLL, computing
each missing MrPro reference once for all candidates. These results must not be
called full-document perplexity.

`smooth_budget.py` builds the fixed-MrPro-budget smooth allocation with a
primal/dual KKT certificate and a BM recovery check. `prepare_gap_probe.py`
builds the signed local gap controls. These constructions are not performance
claims. `origin_shift.py`, `binding_swap.py`, and `numerical_controls.py` test
specific numerical and causal alternatives using archived matched cases.

`long_bridge.py` constructs equal and opposite absolute frequency shifts of
the MrPro clocks with periods between the native and target windows.
`scale_taper.py` constructs one BM allocation that returns to MrPro as its
reference periods approach the native window. Both accept `--root` and
`--tables` and create static-table job contracts plus construction receipts.
Their distance-scale rationale is a hypothesis, not a whole-model guarantee.
`gap_budget_transfer.py` moves one native log-gap budget from the common
high band to either a target-distance band or a matched intermediate band;
both share donor frequencies, gain, and endpoints. This is an EVQ-inspired
resource-allocation test, not the literal Cosh family.

`holdout_eval.py` evaluates frozen tables on separately prepared RULER inputs
and records the exact row set for each job. Once a job is complete,
`paired_summary.py --root RUN_DIRECTORY --job JOB_JSON` reports task/length
paired differences and descriptive bootstrap intervals. Run it as a module,
as with the worker. These intervals do not correct candidate selection or
establish generalization to other templates and sources.

The active run is `/root/autodl-tmp/nongeometric_screen_20260909` on the user-supplied
server. Historical generation and NLL inputs are under
`/root/autodl-tmp/bm_transfer_20260908`. Large activations stay on that server;
compact receipts are copied into the ignored local `results/` directory.
The design report is not an experimental success claim; completed results will
be summarized with their sample size, task breakdown, and limitations.

Latest user-directed protocol (2026-09-10): focus on 128K. Reuse valid passkey
results and screen target-length PPL before releasing a small long downstream
test. Do not automatically complete 32K panels or release the old large holdout
jobs. The hand-built gap/pair/taper branch and the earlier expansion queue are
retained under `deferred_queue/`, not active. P2 passes the initial 128K screen;
only a four-row fresh 128K QA pilot has been released. Inspect its outcome
before adding work. See `planned_controls/long_first_protocol.json` for the
decision and exact jobs, and the mechanism report for the research rationale.
