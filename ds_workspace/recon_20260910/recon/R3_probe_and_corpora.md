# R3 — Frozen-Model Probe Stack, Attention Measurement, and Long-Context Corpora

Branch `09_09`, repo `/Users/yang/projects/hybrid-rope`. Read-only survey; nothing outside this file was written.
No GPU on this machine. Every path/line below was read directly unless marked
`[search-pass]` (found by a repo-wide search agent, not re-read by me) or `not recorded` / `not found`.
Arithmetic derived from recorded numbers is labelled `[derived]` with its inputs named.

---

## (a) Executive summary

1. **The FrozenRoPE probe API is complete and clean.** `experiments/curvature_20260910/model.py` exposes a frozen causal LM whose 64 RoPE frequencies plus a scalar `attention_scaling` are settable in one call — `install(values, gain, track_grad=False)` (`model.py:111`) — and every readout the new experiment needs already exists: `logits(ids, keep)` (`:147`), `nll` (`:156`), `nll_per_token` (`:162`), `log_probs` (`:167`), `grad_wrt_freq` (`:171`).
2. **The long-context memory problem is already solved in that class**, by `logits_to_keep=keep` at `model.py:153`, which slices hidden states *before* the vocab head. This is the single most reusable piece of engineering in the repo: it is why a 128K forward is affordable at all.
3. **The recorded 32K/128K cost numbers are projections, not measurements by this package.** `README.md:117-118` says 32K = 4.1 s median, 128K = 33.9 s median (max 100.7 s), peak 21–27 GiB, and `RUNBOOK.md:38` states in bold: "**Nothing has been run on a GPU.** Every number in README.md is a projection from the host's own archived timings, not a measurement." The only locally verifiable frozen-Qwen3B peak I found is **8.29 GiB at ≤32K** (`docs/research/ROPE_QWEN3_BM_NLL_RESULT_20260908.json`, `cost.peak_allocated_bytes = 8898835456`, `cost.elapsed_seconds = 297.47`, 48 rows at 8192/16384/32768).
4. **`joint_kkt_20260910` is a second, independent probe stack built on top of the first**: a 65-dimensional design (64 freqs + 1 gain), one autograd backward for all 65 (`joint_grad.py`), a QCQP step solver (`qcqp.py`), a teacher-forced error bound (`bound.py`), and a native-retention certificate. Written across 10:50–11:12 today; **actively being edited during this recon** (`loop.py` appeared 11:12, `selftest.py` 11:10 — neither existed in the file listing at 11:00).
5. **The teacher-forced bound `b_e` is exactly what a margin-resolution experiment needs**: `b_e(θ) = max_{t, v≠y_t}[z_{etv} − z_{et,y_t}]` with `1{greedy ≠ reference} ≤ softplus(b_e)/log 2` (`bound.py:8-11`), and its smooth surrogate `b̃ = τ logsumexp((wrong−right)/τ)`, `τ = η/log(N_e)`, sandwiches it exactly: `b ≤ b̃ ≤ b + η` (`bound.py:24-26`). This is a **continuous, per-row, gold-addressed** quantity — eval type (3).
6. **No attention-routing instrument exists for the depth × distance question.** Attention weights *are* extracted in this repo, but only by two narrow instruments: `experiments/native_sparse_position/activation_audit.py` (3 sampled layers × **the last 8 queries only**, 2 predetermined rows at 16K) and `experiments/pm_keep/replay_weighted_probe.py` (per-query-head probabilities over the full prefix, replayed from a saved trace). The distance-histogram primitive `scripts/lib/rope/attn_hist.py` measures exactly "attention mass by distance" but is only exercised by one unit test.
7. **`attn_implementation="eager"` is not an option at these lengths.** `[derived]` Qwen2.5-3B is 16 Q heads / 2 KV heads / head_dim 128 (anchor: `analysis/unify_20260910/digests/digest_thread-0909-pm.md:99`); HF eager attention repeats KV to 16 heads, so one layer's bf16 score tensor at 32K is 16 × 32768² × 2 B = **32.0 GiB**, at 128K = **512.0 GiB**, at 8K = 2.0 GiB — and `output_attentions=True` retains all 36 layers. Attention must be **recomputed from hooked Q/K for a sampled set of (layer, query) pairs**, which is the `activation_audit.py` pattern.
8. **The corpora are gold-addressable, and one of them records the needle position explicitly.** The RULER rows carry `prompt_ids` + `references` + `length_cap` + `task`, so a teacher-forced answer NLL needs no prompt re-tokenisation. `scripts/experiments/niah_retention_canary.py` goes further and stores `source_block: [position, len(needle)]` — the exact needle token span — plus `depth` and `length_cap`, which is the depth × distance design variable already materialised.
9. **Continuous-margin resolution barely exists today.** Of the frozen-checkpoint results, only per-document paired NLL deltas and per-row token-F1 are genuinely continuous; every RULER number is a binary/substring hit rate. The **only** place per-token logits were ever persisted is the 128K cross-cache diagnostic (`top_ids`/`top_logits`/`eos_margin`, 14 rows). The one line in the corpus that admits this is `ROPE_OLMO_BM_EXTRA_RULER_RESULT`'s scope: "Official recall and complete-string/EOS are distinct."
10. **Six standing vetoes constrain any small-panel margin optimisation, and one of them is aimed directly at attention-mass reasoning.** V-E2 forbids restarting the 18-sample/64-dof margin-gradient route; V-A7 forbids reading "attention top-1 / attention mass ≥ 0.5 ⇒ correct generation"; V-A6 forbids extrapolating a frozen-state selector score to whole-model task improvement. `joint_kkt_20260910` is admissible only because it is *not* that route — its own `risk.py:36-56` argues the distinction in the receipt rather than in a commit message.

---

## (b) Findings by question

### 1. `experiments/curvature_20260910/` — module map

Run everything from the repository root as `python -m experiments.curvature_20260910.X` (`README.md:93`).

| file | lines | what it is | public entry points |
|---|---|---|---|
| `model.py` | 322 | Frozen-model plumbing shared by every probe. Isolates transformers 4.x/5.x differences (`rope_scaling` vs `rope_parameters`, `inv_freq` as buffer). | `K=64`, `pick_device()`, `load_frozen(path, dtype, device, attn=None)`, `FrozenRoPE`, `output_kl`, `pair_nll`, `fisher_diagonal`, `fisher_cross`, `fisher_scaling`, `fisher_all_at_once` |
| `tables.py` | 296 | Pure-numpy table algebra in the compression coordinate `m_j = ln(ω_j/ν_j)/ln S`. All named families live here. | `native_inv_freq`, `m_to_inv_freq`, `inv_freq_to_m`, `uniform_phi`, `m_native/interp/mrpro/yarn/evq_shift/evq_deployed/power_shift/flat_gap_donor`, `CONSTRUCTIONS`, `build`, `from_eps`, `verify`, `describe` |
| `preflight.py` | 211 | Stage 0, no GPU. Fails the run on a wrong algebra, a drifted env, a probe corpus that is also an evaluation corpus, or an unrepresentable table. | `check`, `check_table_resolution`, `main()` |
| `local_probe.py` | 225 | The constraint side: output KL and the Fisher it implies. Three estimators, cheapest first (diag / cross / mc). | `load_ids(args, model)`, `mc_fisher(model, ids, keep, base_table, n_samples, seed=20260910, batch_report=16)`, `main()` |
| `arms.py` | 239 | Free falsification test: does `D_N` order the arms the way the panel's own 32K column orders them? Plus the `E1_s28_less` veto test. No forwards. | `load_fisher`, `panel_of`, `eps_of`, `main()` |
| `long_grad.py` | 199 | The objective side: `g_L = d(long loss)/d(log freq)`, forward-only central differences. Two objectives: `nll` (tail-token) and `gold` (teacher-forced answer NLL). | `BRIDGE`, `HIGH`, `LOW`, `SLOT_SETS`, `perturb`, `load_nll_ids`, `load_bind_rows(args, model)`, `mean_answer_nll(model, row)`, `main()` |
| `solve_kkt.py` | 339 | The closed-form trust-region step and the EXPLAINS/BEATS judge. CPU, seconds. | `load_fisher`, `load_grad`, `constraint_matrix(pin_ends=True, fix_sum=False, extra_rows=())`, `price_diagonal`, `projected_gradient`, `projected_step`, `monotone_report`, `lambda_spectrum`, `build_receipt`, `main()` |
| `forward_check.py` | 330 | Real forwards on the step, three budget-matched controls, the pre-registered gate. | `load_step`, `budget_matched`, `controls`, `fit_exponent`, `main()` |
| `panel_jobs.py` | 266 | Emits the two job contracts the existing harness dispatches on. Refuses to write them out of order. | `load_tables`, `reference_table`, `candidate_from_receipt`, `check_monotone`, `spec_of`, `emit_evaluate`, `emit_long`, `main()` |
| `driver.sh` | 354 | Staged runner; every stage writes one file under `runs/` and skips if it exists. | `s0 s1 s1b s2 s2c s3 s4 s5 s6 status all` |

#### 1.1 `FrozenRoPE` — exact API

`class FrozenRoPE` (`model.py:63`). Constructor (`model.py:66`):

```python
def __init__(self, path, dtype="bf16", device=None, attn=None, log_softmax_dtype=torch.float32)
```

Note `attn` is threaded straight into `AutoModelForCausalLM.from_pretrained(..., attn_implementation=attn)` via `load_frozen` (`model.py:46`), so `attn="eager"` is *accepted* — it is just not viable (see §3.4).

Attributes set in `__init__`: `.device`, `.dtype`, `.path`, `.model`, `.rotary` (`= model.model.rotary_emb`, `:83`), `.native_inv_freq` (float32 numpy copy, `:89`), `.head_dim` (`= n*2` = 128, `:90`), `._grad_patched`.

Guards at construction (`model.py:84-88`): raises if `rotary.rope_type != "default"` ("cannot stack dynamic scaling on a frozen reference") and if `inv_freq.numel() != 64`.

Methods, in full:

| signature | line | behaviour |
|---|---|---|
| `tokenizer` (property) | 93 | Lazy, `local_files_only=True`. Raises `RuntimeError` if unavailable, with the message that only the gold-answer objective needs one. |
| `install(self, values, gain, track_grad=False)` | 111 | The single write point. Validates `values.shape == (64)`, finite, non-negative; validates `gain` finite and `> 0`. Sets `rotary.inv_freq`, `rotary.original_inv_freq`, `rotary.attention_scaling = float(gain)`. Returns the new tensor. |
| `install_table(self, table)` | 123 | `install(table["values_float32"], table["gain"], track_grad=table.get("track_grad", False))` — the dict `tables.build()` produces. |
| `_patch_grad_rotary(self)` | 127 | Swaps `rotary.forward` for a grad-capable twin with identical arithmetic (`@torch.no_grad()` on the stock forward is why this exists). |
| `enable_grad_path(self)` | 143 | Idempotent wrapper for the above. Called by `logits()` unconditionally. |
| `logits(self, ids, keep, want_grad=False)` | 147 | `out = self.model(ids, use_cache=False, logits_to_keep=keep)` → `out.logits[0].float()`. Raises if `keep > ids.shape[1]`. |
| `nll(self, ids, keep=512)` | 156 | Mean next-token CE over the last `keep` targets, forward only. |
| `nll_per_token(self, ids, keep=512)` | 162 | Same, `reduction="none"` → 512 floats. |
| `log_probs(self, ids, keep, want_grad=False)` | 167 | `log_softmax` in float32. |
| `grad_wrt_freq(self, ids, keep)` | 171 | Exact `d(mean NLL)/d(inv_freq)` for all 64 slots in one backward. Returns `(loss, grad64)`. |

There is **no** `forward`, no `generate`, no `output_attentions` plumbing, and no attention hook anywhere in this class.

#### 1.2 How a frequency table is installed

Direct attribute assignment on the HF rotary module — no wrapper, no module replacement (`model.py:117-121`):

```python
v = v.clone().requires_grad_(track_grad)
self.rotary.inv_freq = v
self.rotary.original_inv_freq = v.detach().clone()
self.rotary.attention_scaling = float(gain)
```

Because `install()` is the only write point, a probe cannot half-install a table. `tables.build(name, gain=GAIN_YARN, cfg=QWEN25_3B)` returns the dict it consumes: `dict(name, m, gain, theta, values_float32)` (`tables.py:209-210`). `tables.from_eps(d_eps, base_name="mrpro_n17", gain, cfg, k)` applies a solver step `ν_j → ν_j·exp(−d_j)` (`tables.py:213-230`).

The gain is a **float** (`attention_scaling`) on this path, and `model.py:136-137` shows why the design package squares it: `cos = emb.cos() * attention_scaling`, and q·k therefore picks up `g²`.

#### 1.3 Logits and the memory optimisation

Yes — `logits_to_keep`. `model.py:15-18` states the reason verbatim:

> The vocabulary projection dominates memory at long length (131072 x 151936 x 2 bytes = 38 GiB if materialised). Every long forward passes `logits_to_keep`, which slices hidden states BEFORE the head, so a 128K forward costs activations and nothing else.

`logits()` also calls `enable_grad_path()` on every invocation (`model.py:151`), so the grad-capable rotary twin is the standard path; the forward-only probes wrap themselves in `torch.no_grad()`.

**Recorded cost.** `README.md:115-118`:

> Measured on the same checkpoint and the same inputs: a 32K forward is 4.1 s median, a 128K forward 33.9 s median (max 100.7 s), peak allocation 21–27 GiB.

Two provenance caveats that must travel with those numbers:

* `README.md:115` heads the section "Cost, from this host's own **640 archived rows**" — i.e. they are read off the pre-existing `bm_transfer` harness rows, not produced by this package.
* `RUNBOOK.md:38`: "**Nothing has been run on a GPU.** Every number in `README.md` is a projection from the host's own archived timings, not a measurement."

The single locally checkable frozen-Qwen3B memory receipt is at ≤32K, not 128K: `docs/research/ROPE_QWEN3_BM_NLL_RESULT_20260908.json` → `cost.peak_allocated_bytes = 8898835456` (8.29 GiB), `cost.elapsed_seconds = 297.47` for 48 rows. **A 128K peak allocation for the 3B checkpoint is `not recorded` locally** (the `long_eval` 64K/128K outputs are server-side only).

#### 1.4 The native-output-KL metric, exactly

`output_kl` (`model.py:187-210`):

```python
@torch.no_grad()
def output_kl(model, ids, keep, base_logp, table):
    model.install_table(table)
    lp = model.log_probs(ids, keep).double()
    b = base_logp.double()
    p = b.exp()
    return float((p * (b - lp)).sum(-1).mean())
```

Definition, quoted from the docstring (`model.py:189-204`):

> `E_u[ KL( p_base(.|u) || p_table(.|u) ) ]` over the last `keep` positions.
>
> This is the native-preservation metric: it has no first-order term, its quadratic form is the Fisher information `F_N = E[J^T(diag p - pp^T)J] >= 0`, it measures behaviour drift rather than weight drift, and it needs no assumption about which frequencies matter. A plain cross-entropy difference would have a non-zero first-order term and an indefinite second-order form; that is why it is not used.

Evaluated as `sum_v p (log p_base − log p_table)` with both log-probability tensors taken directly, in **float64**. The docstring records a real failure: an earlier `p_base.exp().log()` round trip returned **negative Fisher diagonals** (~1e-7 absolute on each log-probability, "the same size as the whole KL for a small delta").

Derived from it: `fisher_diagonal` (`model.py:219`) gives `F_jj ≈ 2·D_N(δ·e_j)/δ²` from one forward per slot; `fisher_cross` (`:236`) isolates `F_jk` from one forward per pair; `fisher_scaling` (`:256`) checks the log-log exponent is ≈2; `fisher_all_at_once` (`:300`) is explicitly labelled "a bounded warning, not a diagonality test".

---

### 2. `experiments/joint_kkt_20260910/` — public interfaces

**State at recon time (mtimes):** `qcqp.py` 10:54, `bound.py` 10:55, `risk.py` 10:58, `design.py` 11:10, `joint_grad.py` 11:11, `accept.py` 11:08, **`selftest.py` 11:10 and `loop.py` 11:12 are new** — they did not exist when this task's file list was taken. `__init__.py` is **0 bytes** (empty; no re-exports).

| file | what it computes | public interface |
|---|---|---|
| `design.py` | The 65-dim design variable `y = (x, a)`, `x_j = −log ν_j`, `a = log(g²)`; its feasible set; the gap-price structure of §5. | `K=64`, `N_DESIGN=65`, `A_INDEX=64`, `GAIN_BOX=(0.5,2.5)`, `DELTA_MIN_FRAC=0.1`, `x_of_nu`, `nu_of_x`, `a_of_gain`, `gain_of_a`, `eps_of_x`, `native_x`, `pack`, `unpack`, `to_table`, `from_table`, `a_box`, `delta_min`, `ordering_rows`, `endpoint_rows`, `span_row`, `gain_box_rows`, `feasible_set`, `gap_transfer`, `gap_prices`, `violations`, `describe` |
| `qcqp.py` | §6 gain elimination (Schur complement), §8 closed-form step, §8 epigraph QCQP. | `build_quadratic`, `psd_floor`, `damp`, `eliminate_gain`, `analytic_step`, `predicted`, `implied_box`, `project_tr`, `repair_step`, `solve_epigraph` |
| `joint_grad.py` | Real model derivatives for all 65 variables in **one** backward; grad-capable objective twins; the finite-difference gate. | `nll_tensor`, `nll_per_token_tensor`, `logp_tensor`, `nll_value`, `class JointDesign`, `finite_difference_check` |
| `bound.py` | The §3 complete-output error bound and the §4 native-retention certificate. | `ETA_DEFAULT=0.02`, `n_terms`, `tau_for`, `raw_bound`, `smoothed_bound`, `bound_loss`, `bound_is_informative`, `gamma_from_native`, `d_keep`, `bound_summary` |
| `risk.py` | The panel → objective skeleton: rows, groups, weights, LOO, and the span guard. Measures nothing about θ. | `REQUIRED_KEYS`, `PREPARED_TASKS`, `DEFAULT_PREFIX`, `SECONDS_PER_ROW`, `load_rows`, `group_key`, `build_groups`, `reference_ids`, `teacher_forced_record`, `answer_mask`, `mask_positions`, `span_guard`, `projector`, `null_fraction`, `guard`, `group_weights`, `loo_folds`, `budget_seconds`, `panel_receipt`, `verify_against_frozen`, `main()` |
| `accept.py` | §8 acceptance predicate and the retry ladder. No model calls in the predicate — unit-testable on CPU. | `RHO_BAND=(0.5,1.5)`, `EPS_DEFAULT=1e-3`, `model_values`, `check_acceptance`, `escalation_for`, `run_ladder`, `summarize_ladder` |
| `loop.py` | The sequential convex driver, its receipts, and Phase-I feasibility recovery. Objective is **injected** — the driver never imports torch. | `loop.py` head (read): reports `null_fraction` and `gain_unpriced` per accepted step |
| `selftest.py` | The CPU gate. 36 test functions covering coordinates, ordering, gain box, QCQP 1-D/2-D, trust, solver, repair, elimination, prices, bound + sandwich, D_keep, the five acceptance refusals, the ladder, span guard, panel guard, groups, loader, and the loop (Phase-I, known answer, null fraction, no-step, Pareto, stationarity, gain flag). | `t_coords` … `t_loop_gain_flag` |

#### 2.1 The teacher-forced `b_e` bound and its smoothing

`bound.py:8-11`, verbatim:

```
b_e(theta) = max_{t, v != y_t} [ z_{etv}(theta) - z_{et, y_t}(theta) ]

1{greedy output != reference}  <=  softplus(b_e) / log 2                (*)
```

Validity argument (`bound.py:12-18`): "an induction on the reference prefix … before the first divergence the model's greedy prefix IS the reference prefix, so the first disagreement is decided by exactly the logits (*) maximises over". Explicitly **not** "a margin normalised between two answers, nor a single head's margin" (`bound.py:17-18`).

Smoothing (`bound.py:21-30`):

```
b_tilde = tau_e * logsumexp((wrong - right) / tau_e),   tau_e = eta / log(N_e)
```

with `N_e = T_e(V−1)` (`bound.py:54-58`). It "satisfies `b <= b_tilde <= b + eta` exactly, because logsumexp exceeds the max by at most `tau*log(N)`. So eta is a declared numerical smoothing allowance … Choosing eta after seeing a result would silently convert the certificate into a fit; sec.3 fixes the default at 0.02 logits and `selftest.py` checks the sandwich numerically."

**The bound is allowed to be vacuous, and the package says so** (`bound.py:32-40`):

> softplus(b)/log 2 exceeds 1 for b > 0, so on a row where some wrong token outranks the reference token the bound says nothing beyond the trivial "<= 1". For the bound to be informative at the 0.1 level it needs b < -2.63 nats on EVERY answer position. Whether a frozen instruct model clears that on a 7-digit needle is an empirical question, and it is the FIRST thing worth measuring because if it does not, the sec.3 objective carries no signal and the solve must fall back to the sec.4 native-NLL formulation.

`bound_summary(bs, etas=None)` returns `vacuum_rate`, `informative_0p1`, `threshold_for_0p1`, `bound_median`.

#### 2.2 `D_keep` — the native-retention certificate

`d_keep(b_theta, b_native, weights=None)` (`bound.py:170`):

```
D_keep(theta) = mean over certified rows of [1 + b_e(theta)/gamma_e]_+
```

with `gamma_e = −b_e(θ_native)`, defined **only where the baseline is correct** (`gamma_from_native`, `bound.py:155-167`): "The plan requires that rows with `b_e(native) >= 0` are NOT included with a floored denominator: a baseline that does not clear its own reference has no margin to certify, and dividing by a floor would produce a certificate that looks the same as a real one." `D_keep(native) = 0` by construction; the return carries `note="bounds the native accuracy drop on THIS sample; not a population guarantee"` (`bound.py:193-194`).

#### 2.3 The RULER row loader and its corpus schema

Read from `ruler_prepared_01/rows.jsonl` (`risk.py:58`). Contract quoted verbatim (`risk.py:58-64`):

```
ROW SCHEMA.  Read from `ruler_prepared_01/rows.jsonl`; every key is validated, so
a schema drift names the missing field instead of producing a shorter panel:

    row_id, task, length_cap, ids (prompt token ids), references (gold strings),
    input_tokens, budget (max_new_tokens), prompt_sha256, upstream_index

`references` is a LIST and is matched by substring against the decoded output
(`ruler_bench.score`).  A teacher-forced bound needs ONE trajectory, so one
reference is chosen and the choice is recorded; rows with several are flagged
rather than silently averaged.
```

Machine-checked in code: `REQUIRED_KEYS = ("row_id","task","length_cap","ids","references","input_tokens","budget","prompt_sha256")` (`risk.py:77-78`). `PREPARED_TASKS = ("niah_single_1","niah_multikey_3","vt")` (`risk.py:83`). `DEFAULT_PREFIX = " "` (`risk.py:88`) — "a single space is the separator that makes `tokenize(prefix + ref)` the correct continuation for that convention."

The loader is **`risk.load_rows(path)`**, not `long_grad.load_bind_rows`: the latter (`long_grad.py:71-92`) is the older, looser reader that consumes any `screen.jsonl` and requires `length_cap`, `references`, `prompt_ids`, `row_id`, `task`. Both exist; `risk.load_rows` is the validated one.

Grouping is `(length_cap, task)` — 3 tasks × 2 lengths = **6 groups** (`risk.py:130-134`), and `build_groups` raises on an incomplete grid.

#### 2.4 The span guard — how many directions a 30-row panel can constrain

`span_guard(row_grads)` (`risk.py:263`):

* input `row_grads` is `(n_rows, d_design)`;
* SVD (`risk.py:287`), tolerance `tol = s[0] * max(G.shape) * eps` (`:288`), `rank = (s > tol).sum()`;
* returns `rank`, `null_dim = d − rank`, `tol`, `live_rows`, `singular_values`, and `basis` = the leading **right** singular vectors, shape `(rank, d)`.

`null_fraction(step, guard)` (`risk.py:320`) returns `||(I−P)d|| / ||d||`. The docstring explains it is computed by subtracting the projected **vector** rather than differencing squared norms, because the latter "floors at ~1e-8 instead of 0".

The governing arithmetic is stated at the top of the file (`risk.py:10-12`):

```
THE FACT THAT LIMITS IT, STATED FIRST BECAUSE IT IS EASY TO BURY.
    n_rows = 30,  d_design = 65.
```

> So the panel constrains θ along at most 30 of its 65 dimensions. The other >= 35 are null directions of this objective: the solver may move along them at zero measured long-range cost, and the panel cannot tell whether that helps or hurts. (`risk.py:19-22`)

`guard(n_rows, d_design, n_groups, ...)` (`risk.py:349`) **raises** on two configurations: `n_groups >= n_rows` ("per-row objective … the vetoed 18-sample/64-dof route", `:369-374`) and `n_groups >= d_design` (`:375-379`). It stamps `development_only=True` and returns `vex2_ratio` + a `vex2_note` in the receipt.

#### 2.5 Which modules lack a doc/test file

* **No `README.md`, no `RUNBOOK.md`, no `driver.sh`, no `lo`-level doc of any kind** exists in `experiments/joint_kkt_20260910/` — the directory holds only `.py` files and an empty `__init__.py`.
* The docstrings reference **`FINAL_PLAN.md`** three times (`design.py:3`, `joint_grad.py:3`, `selftest.py:3`) and cite its section numbers as the authority for the variable, the backward requirement, and the self-test list. **`FINAL_PLAN.md` does not exist anywhere in this working tree** (`find . -name "FINAL_PLAN*"` excluding `.git` → no hits; all three hits are the code's own citations). The same is true of `UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md` (`driver.sh:44`) and `NEXT_DERIVATION_KKT_PROBLEM.md` — the latter **does** exist at `analysis/unify_20260910/NEXT_DERIVATION_KKT_PROBLEM.md` (146 lines) and is the nearest thing to a governing document for this package.
* **Tests:** `selftest.py` is the only one. Nothing under `tests/` references either package (`grep -rl "curvature_20260910\|joint_kkt_20260910" tests/` → no hits), so neither package is in the pytest suite.
* **`curvature_20260910` by contrast has `README.md` (366 lines), `RUNBOOK.md` (389 lines), `driver.sh`, and a populated `__init__.py`.**

---

### 3. ATTENTION MEASUREMENT — the critical gap

**Verdict up front: attention weights *are* extractable in this repo, but no instrument measures attention routing as a function of evidence depth and query–evidence distance. Two narrow instruments exist and one distance-histogram primitive exists; the depth × distance sweep must be built.**

#### 3.1 Instruments that genuinely extract or recompute attention

| file:line | quote | what it measures | runnable as-is? |
|---|---|---|---|
| `experiments/native_sparse_position/activation_audit.py:99-129` | `ALL_ATTENTION_FUNCTIONS.register('sdpa',interface)`; inside: `all_scores=torch.einsum('hqd,hnd->hqn',q,kr)` (`:29`), `fullp=all_scores.softmax(-1)` (`:32`), `mass=fullp[:,:,:cut].reshape(H,Q,blocks,block).sum(-1)` (`:33`) | True fp32 softmax over **all visible keys**; per-head retained mass, per-block mass, value-relative L2. Bitwise parity with the real rotated q/k is asserted (`:104`, `raise RuntimeError('Raw capture / rotary reconstruction parity failed')`). | **Yes** — `--model --inputs --output`. Model-generic: `layers={m.layer_idx:m for m in model.modules() if m.__class__.__name__ in ('Qwen3_5Attention','Qwen2Attention')}` (`:84`) |
| same, sampling limits | `chosen=[indices[0],indices[len(indices)//2],indices[-1]]` (`:85`); `tail=8` (`:105`); `selected=[x for x in rows if x.get('length_budget')==16384 and x['family'] in (0,1) and not x['reverse'] and x['task']=='marker_M7']` with `if len(selected)!=2: raise` (`:131-132`) | **3 layers × last 8 queries × 2 predetermined rows @16K.** Not a depth sweep. | — |
| `experiments/pm_keep/replay_weighted_probe.py:~41-46` | `conditional = (torch.einsum("hgd,htd->hgt", q, k) * attention_scale).softmax(-1)`; `"reconstructed_prefix_mass_per_query_head"` | Per-query-head attention probabilities over the **full prefix**, replayed from saved post-RoPE Q + cached K. Writes per-head mass and mass×‖V‖. | Yes, from a saved trace; no model load |
| `experiments/pm_keep/fast_scores.py:71-73` | `logits = torch.bmm(grouped[...], key_transpose); logits.mul_(scale); probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)` | Batched fp32 full-prefix attention probabilities; aggregated (mean), not stored per head | Yes |
| `experiments/native_sparse_position/causal_probe.py:49-63` | `logits = torch.einsum("hgd,htd->hgt", q, context.k.float())`; `probability = logits.softmax(-1)` | Exact per-head attention mass at **one** query position | Yes (nosa harness + GPU) |
| `scripts/lib/rope/attn_hist.py:13-56` | `accumulate_distance_histogram(q, k, query_positions, max_distance, hist, block_q=128)`; `hist.scatter_add_(0, dist_flat, weight_flat)` | **Online attention-mass-by-distance histogram accumulation** + `fit_power_law` + `bootstrap_alpha_ci` | Yes — but only exercised by `tests/test_rope_core.py` |
| `experiments/pm_keep/test_adapter.py:137-153` | `attention.register_forward_hook(lambda m,a,output: actual_weights.append(output[1].clone()))` … `expected = torch.softmax(q @ repeated.transpose(-1, -2) * attention.scaling, ...)` | Real eager attention weights from a frozen Qwen2Attention (returned as `output[1]`), asserted against a manual softmax | Test-only |
| `scripts/m4_max_36gb/test3_attention_prior.py:110` | `outputs = model(input_ids, output_attentions=True)` | Per-head distance histograms from a **frozen GPT-2** | Yes, but GPT-2 |
| `scripts/train.py:380` | `model_inputs["output_attentions"] = True` | Attention weights during **LoRA training** (Llama-3-8B), feeding a collision penalty | Training-only |

#### 3.2 Anti-extraction, and dead code

* `experiments/nosa_position/qwen_transfer.py:113-114`: `if kwargs.get("output_attentions", False): raise ValueError("attention matrices are not materialized by this reference")` — a custom sparse-attention reference that **deliberately refuses** to materialise weights.
* `scripts/analysis/exp_tau_diagnostic.py:519`: `from scripts.analysis._minimal_gpt import MinimalGPT, MinimalGPTConfig` — **the module does not exist**; `MinimalGPT` is defined later in the same file. This file contains complete per-head entropy/IPR-over-distance analysis machinery and is currently **dead code**.
* Everything matching `attention_mask` elsewhere is ordinary causal/padding plumbing; no weight extraction.

#### 3.3 `output_attentions` / `attn_implementation` in the probe stacks

* **Neither `curvature_20260910` nor `joint_kkt_20260910` sets `output_attentions` anywhere.** `FrozenRoPE` passes `attn` through to `from_pretrained` and otherwise never touches attention.
* `experiments/nosa_position/runtime.py:250-266`, `experiments/rope_operator_family/model.py:151`, `experiments/rotary_budget/runtime.py:14`, `experiments/native_rope_evq_150m/model.py:101` all call `F.scaled_dot_product_attention(...)` with no weight output.
* All `attn_implementation="sdpa"` selections on frozen models (e.g. `experiments/pm_keep/run.py:196`, `experiments/nongeometric_screen/worker.py:89`) are performance choices, not extraction.

#### 3.4 Can the frozen model run with `attn_implementation="eager"` at these lengths?

**No — categorically, on any single card.** `[derived]`

Inputs: Qwen2.5-3B-Instruct is 36 layers, **16 Q heads / 2 KV heads**, head_dim 128 → 64 rotary slots. Anchor: `analysis/unify_20260910/digests/digest_thread-0909-pm.md:99` ("36 层、16Q/2KV heads、head 128→64 slots、W=32768、b=10⁶、S=4"), repeated at `analysis/unify_20260910/digests/digest_paper-state.md:90`.

HF's eager path repeats KV to `num_attention_heads`, so the score tensor is `(16, L, L)`:

| L | elements | bf16 tensor | fp32 tensor |
|---|---|---|---|
| 8192 | 1.074e9 | **2.0 GiB** | 4.0 GiB |
| 32768 | 1.718e10 | **32.0 GiB** | 64.0 GiB |
| 131072 | 2.749e11 | **512.0 GiB** | 1024.0 GiB |

These are **per layer, for the score tensor alone** — the softmax output is the same size again, and `output_attentions=True` retains one such map per layer (×36). The recorded whole-model peak for a *non-eager* 32K forward is 21–27 GiB (`README.md:118`) or 8.29 GiB in the one local receipt. Eager at 32K is therefore roughly the whole card for a single layer; at 128K it is 512 GiB.

This is precisely the constraint the existing code was built around: `model.py:15-18` exists because the vocab projection alone would be 38 GiB if materialised, and the whole probe design is "no activation of size L×L or L×V at any point". `local_probe.py`'s `mc_fisher` takes the **only** backward pass at any length and pins `--dtype fp32` by default with the note that "bf16 rounds the logits at ~4e-3 relative, which is the size of the whole perturbation response for a 2e-2 step on some slots".

**Consequence for the new experiment:** attention must be obtained by (a) hooking `q_proj`/`k_proj` (or `q_norm`/`k_norm`), (b) intercepting the real attention call through `ALL_ATTENTION_FUNCTIONS` to prove the captured tensors are the ones actually used, and (c) recomputing a **fp32, causal, blockwise** softmax for a *chosen set* of (layer, query-position) pairs against the visible prefix — i.e. exactly the `activation_audit.py` pattern, generalised from 3 layers × 8 queries × 2 rows to a full (layer × query-position) grid.

---

### 4. RULER and long-context corpora

#### 4.1 RULER rows — the Qwen2.5-3B development panel (the load-bearing one)

| | |
|---|---|
| **Path (server)** | `/root/autodl-tmp/bm_transfer_20260908/prepared_qwen3_01/` — `screen.jsonl`, `prompts.jsonl`, `tables.json`, `queue.json`, `generation_config.json`, `qualification.jsonl`, `manifest.json` (`experiments/nongeometric_screen/worker.py:56-59`) |
| **Baseline outputs** | `.../run_qwen3_01/MrPro.jsonl`; **local mirror exists and was verified**: `results/bm_transfer_20260908/run_qwen3_01/MrPro.jsonl` — **36 rows**, keys `correct, elapsed_seconds, ended_eos, family, generated_ids, input_tokens, length_cap, max_new_tokens, output_text, prompt_sha256, references, row_id, task` |
| **Rows × tasks** | 36 = 6 tasks × (2 @32768 + 4 @131072). Verified `Counter({'niah_single_2':6,'niah_multikey_2':6,'niah_multiquery':6,'vt':6,'fwe':6,'qa_1':6})`, and 24 rows with `input_tokens > 32768` |
| **Tasks** | `TASKS = ('niah_single_2','niah_multikey_2','niah_multiquery','vt','fwe','qa_1')` (`scripts/experiments/olmo_fast_screen/ruler_bench.py:5`); `FAMILIES = ('retrieval','retrieval','retrieval','tracking','aggregation','qa')` (`:6`) |
| **Producer schema** | `scripts/experiments/olmo_fast_screen/prepare_ruler.py:135-138`, verbatim: `row = dict(row_id=f'{task}_{cap}_{index}', task=task, family=families[task], upstream_index=raw['index'], length_cap=cap, prompt_ids=ids, prompt_sha256=digest(ids), input_tokens=len(ids), references=refs, max_new_tokens=budget)` |
| **Gold addressable?** | **Yes.** `prompt_ids` = the full token-id prompt (no re-tokenisation of the prompt needed) and `references` = gold answer strings. The existing implementation is `long_grad.load_bind_rows` + `mean_answer_nll` (`long_grad.py:84-101`): `ans = model.tokenizer("\n" + str(refs[0]), add_special_tokens=False).input_ids` then CE over the answer span only. |

Verbatim example row (from the run output, which does **not** carry `prompt_ids` — only `prompt_sha256`):

```json
{"row_id": "niah_single_2_32768_0", "family": "retrieval", "length_cap": 32768, "prompt_sha256": "5a9d9b7a188e869c8c45b06e64f021b5d68e706b629ca3ef01a0d4a9b0867f1b", "input_tokens": 32030, "task": "niah_single_2", "references": ["2949526"], "max_new_tokens": 128, "correct": 1.0, "output_text": "  2949526.", "generated_ids": [220, 220, 17, 24, 19, 24, 20, 17, 21, 13, 151645], "ended_eos": true, "elapsed_seconds": 4.607084859162569}
```

**A local mirror of `prepared_qwen3_01/screen.jsonl` (the input side, with `prompt_ids`) does not exist** — `not found`. The input rows for the Qwen panel live on the server only; the local `results/` mirror holds generated outputs (which keep only `prompt_sha256`).

#### 4.2 Other RULER row sets

| corpus | path | rows | lengths | gold-addressable | evidence |
|---|---|---|---|---|---|
| OLMo "new tasks" panel | `results/olmo_fast_screen_20260908/prepared_ruler_newtasks_02/screen.jsonl` (**local mirror verified**) | **350** = 7 tasks × 50 (run jsonls verified: `run_ruler_newtasks_01/{MrPro,MrProBM}.jsonl` = 350 rows each) | 16384 | Yes — same producer schema | manifest `screen_rows=350`, `samples_per_task_by_cap={'16384': 50}` |
| RULER 13-task shards | `scripts/experiments/scale_transport/ruler_full_prepare.py` → `--out/<task>/rows.jsonl` | `--count` default 50/shard | `--length` default 131072 | Yes — `ids` + `references` | `ruler_full_prepare.py:18-20` lists all 13 tasks; record at `:77-81` |
| **`ruler_prepared_01/rows.jsonl`** (the one `joint_kkt` reads) | producer chain: `ruler_full_prepare.py` shards → `experiments/evq_recovery/prepare_ruler.py:45-49` | evq grid: 3 tasks × 4 lengths × 24 = 288/split; `joint_kkt` uses the 3 tasks × **2** caps = 6 groups × 5 rows = **30 rows** (`risk.py:11-12`) | 2 caps | Yes | `risk.py:58-64`, `evq_recovery/prepare_ruler.py:14` (`TASKS=('niah_single_1','niah_multikey_3','vt')`) |
| RULER stratified QA holdout | `experiments/nongeometric_screen/prepare_diverse.py` → new prepared dir | 36 old rows minus `qa_1`, plus 32 fresh `qa_1` per cap × 2 caps = **64 new** | 32768, 131072 | Yes; `holdout_eval.py:19-21` re-checks the prompt hash and asserts no overlap with the dev panel | `prepare_diverse.py:65-84` |
| nosa public RULER subset | `experiments/nosa_position/prepare.py` | dev 8 + test 32 per task per length | 2048/8192/16384 | Yes — `prompt_ids` + `references` | `prepare.py:24, 163-174` |
| shared-48 DEV panel | `experiments/nosa_position/prepare_ten_panel.py` → `rows.jsonl` | exactly 48 (12 × 4 families, asserted) | 8192/16384 | Yes — `references` / `expected` | `prepare_ten_panel.py:48-55` |

#### 4.3 The needle-position corpus — the single most relevant artefact for a depth × distance design

`scripts/experiments/niah_retention_canary.py`. Its record, verbatim (`niah_retention_canary.py:63-66`):

```python
record={'semantic_id':f'passkey_{i:02d}','world':world,'length_cap':length,'depth':depth,
        'layout':'compact' if depth is None else 'needle','prompt_ids':ids,'generation_budget':32,
        'answer':answers[world],'answers_by_world':answers,'source_block':[position,len(needle)],
        'background_source_ids':sources}
```

* **Rows:** 8 semantic groups × 2 worlds × (1 compact @2048 + 3 lengths × 3 depths) = **160**.
* **Lengths/depths:** `(2048, None)` then `[(length, depth) for length in lengths for depth in (.1,.5,.9)]` with `--lengths` choices `(4096,16384,32768)` (`:55`).
* **What makes it uniquely useful:** `source_block = [position, len(needle)]` is the **exact token span of the needle**, and `position = round(depth*(size-len(needle)))` (`:60`) is computed, not inferred. The query is at a fixed place (the chat-template suffix), so **query-to-evidence distance = query_position − source_block[0]** is available directly, with no text parsing. `prompt_ids` and `answer` are both stored → teacher-forced margin is computable without generation.
* Docstring self-limits (`:1-6`): "Small paired passkey canary for retrieval retention, not general forgetting … **This is a declared NIAH variant, not an exact reproduction of RULER.**"

#### 4.4 Natural-text and other long corpora

| corpus | path | content | gold? |
|---|---|---|---|
| `prepared_nll_01` | `/root/autodl-tmp/bm_transfer_20260908/prepared_nll_01/doc_00..15.npy` + `manifest.json`; local mirror of the *runs* at `results/bm_transfer_20260908/run_nll_01/` | **16 docs × 32769 tokens**; scored at 8192/16384/32768, tail 512. Local mirror verified: 48 rows, keys `doc, input_sha256, length, method, nll, target_ids, token_nll`, `token_nll` length 512 | **No answers** — pure NLL |
| `long_inputs` | `$HARNESS/long_inputs/` (`/root/autodl-tmp/nongeometric_screen_20260909/long_inputs/`), e.g. `pg19_test_37702.npy` | **131073 tokens/doc**, one contiguous real document prefix, no filler. 32 proof-pile + 16 pg19 candidates → first 8 per source kept. Panel takes `docs_per_dataset=2`: proofpile 001364/001901, pg19 28988/30312; probe uses pg19 37702 (held out) | **No answers** |
| LongBench natural QA | `prepared_natural_extra_01/screen.jsonl` (local, **211 rows**, narrativeqa 61 + multifieldqa_en 150) | cap 16384; metrics via `scripts/eval/longbench_metrics.py:10-17` `TASK_METRIC_MAP` (token-F1 / Rouge-L) | Yes — `prompt_ids` + `references` |
| OLMo constructed bench | `scripts/experiments/olmo_fast_screen/{bench,prepare}.py` → `screen.jsonl` | 24 screen + 8 qualification; caps 4096/8192/16384; families `('lookup','linked_lookup','latest_update','attribute_binding')` | Yes |
| `single_table_generation.py` native pool | `scripts/experiments/single_table_generation.py:152-159` | — | **Yes, strongest form: `answer_ids` (gold answer *token ids*) and `answer` are stored directly** — a teacher-forced answer NLL needs zero re-tokenisation |
| Training/PT corpora | FineWeb-Edu 3×1B, SlimPajama-mixed 500M, OpenWebText+Pile, proof-pile-2, PG19 nested anchors | token arrays / `.pt` | No answers (NLL only) |

Two harness facts that a new experiment must respect (`panel_jobs.py:11-30`, `README.md:159-181`):

* `action: evaluate` reads `prepared_nll_01` (32769-token docs) and can only reach 8192/16384/32768; **a 128K request on that path truncates silently**.
* `action: module long_eval` reads `long_inputs/` (131073-token docs) at 65536/131072 and recovers each method's spec from `results/<method>/contract.json`, so the registration job must run first. The worker takes `sorted(queue/*.json)[0]`.

---

### 5. Existing eval numbers for frozen checkpoints — with receipts and type

Type key: **(1)** scalar PPL/NLL/loss · **(2)** binary task accuracy · **(3)** continuous margin · **(4)** other.

| receipt file | model | lengths | metric | headline (verbatim) | type |
|---|---|---|---|---|---|
| `results/bm_transfer_20260908/run_qwen3_01/MrPro.jsonl` (+ `MrProBM.jsonl`), 36+36 rows **verified** | Qwen2.5-3B-I | 32768/131072 | six-task RULER macro, per-row `correct` ∈ [0,1] | `docs/research/ROPE_BM_TRANSFER_RESULT_20260908.json`: `macro_delta_by_length = {32768:+0.04444, 131072:-0.07292}`, `paired_wins=6/paired_losses=4`, status `NO_LONG_GAIN` | **2** |
| `results/bm_transfer_20260908/run_nll_01/{Native,MrPro,MrProBM}.jsonl`, 48 rows each **verified**; summary `docs/research/ROPE_QWEN3_BM_NLL_RESULT_20260908.json` | Qwen2.5-3B-I | 8192/16384/32768 | mean NLL over final 512 tokens + per-doc `delta` | `runtime.scoring = "FP32 cross-entropy on final 512 next tokens"`; `cost.peak_allocated_bytes = 8898835456`, `cost.elapsed_seconds = 297.47`; per-length means e.g. 8192: Native `2.276074` / MrPro `2.323329` / BM `2.325812` | **1** (+ **3** per-doc deltas) |
| `results/bm_transfer_qwen7b_20260908/run_nll_01/*.jsonl` (48 each) + `run_screen_02/*.jsonl` (18 each) **verified** | Qwen2.5-7B-I | same | NLL + RULER | `ROPE_QWEN7_BM_RESULT_20260908.json`: `macro_delta_by_length={32768:-0.03333, 131072:-0.13333}`, "0胜、3负、15平" | **1**, **2** |
| `results/bm_transfer_20260908/gap_capped_run_01/GapCapped.jsonl` | Qwen2.5-3B-I | 32768/131072 | RULER macro | `ROPE_GAP_CAPPED_RESULT_20260908.json`: `score_sum=25.05`, `paired_wins=0/paired_losses=7`, `NO_LONG_GAIN` | **2** |
| `results/olmo_fast_screen_20260908/run_ruler_newtasks_01/{MrPro,MrProBM}.jsonl` — **350 + 350 rows verified** | OLMo-2-0425-1B-I (1.485B) | 16384 | 7-task RULER recall | `ROPE_OLMO_BM_EXTRA_RULER_RESULT_20260908.json/.md`: "七项等权41.67%对7.09%，+34.59pp"; paired wins 156 / losses 9 | **2** |
| `results/olmo_fast_screen_20260908/run_nll_01/*.jsonl` (MrPro 48, BM 48, Native 16) **verified** | OLMo-2-1B-I | 4096/8192/16384 | mean NLL | `ROPE_OLMO_BM_NLL_RESULT_20260908.json`: BM−MrPro means −0.2556 / −0.3017 / −0.8259; 16/16 docs lower at 16K | **1** (+ **3**) |
| `docs/research/ROPE_OLMO_BM_NATURAL_RESULT_20260908.json` | OLMo-2-1B-I | >4K inputs | token-F1 macro, 567 rows | `task_equal_macro.extended`: MrPro `0.21373` → BM `0.26008` (Δ `+0.04636`, CI `[+0.01316,+0.06292]`) | **3** |
| `docs/research/ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.json` | OLMo-2-1B-I | long inputs | 5-task token-F1, 778 rows | "BM25.44%对Mr21.62%，+3.82pp" | **3** |
| `docs/research/ROPE_QWEN15_FULL_LAG_P2_RESULT_20260907.json` | Qwen2.5-1.5B-I (+ 3B×24 rows) | 65536/131072 | 3-task recall | 64K MK2 `37.5` vs `12.5`; 128K MK2 `0/0`; FWE `70.83` vs `45.83` | **2** |
| `results/bm_transfer_20260908/cross_cache_run_01|02/results.json` | Qwen2.5-3B-I | 128K | per-token `top_ids`/`top_logits`/`eos_margin`, 14 rows | `[search-pass]` — **the only place per-token logits were ever persisted** | **4** (logits) / **3** (`eos_margin`) |
| `docs/research/NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md:38-44` | Qwen2.5-3B-I | 131072, 4 docs | mean NLL / pooled PPL | `1.705390 / 5.503529` (official MrPro), `1.680924 / 5.370516`, `1.678507 / 5.357549` — **prose only, no receipt** (`analysis/kkt_20260910/mine/R8_receipts.md:125` says "本机零凭据") | **1** |
| `paper-2027/research/attention-aware-retrofit/evidence/K32_*_RECEIPT_20260901.json` | Qwen2.5-0.5B-I | 32768/65536 | 4-task macro, RULER-13 delta | `native_32k 0.6475`; 64K index−YaRN `+0.064375` CI `[+0.0275,+0.102516]`; RULER-13 `+0.060897` CI `[+0.027627,+0.095835]` | **2** |
| `.../zero-training-deployment/HEAD_SELECTIVE_ZERO_TRAINING_SIX_ARM_RESULT_20260903.md` | OLMo-2-1B-I | 8192–16384 | EM / forward KL / teacher-forced answer NLL | 25 rows ≥8192: all six arms "0 normalized EM"; controls `7.14%` / official YaRN-4 `22.02%`; KL `.10310` vs `.04191`; first-reference answer-token NLL `9.8055` vs `9.3299` | **2**, **4**, **1** |
| `.../zero-training-deployment/NATIVE_ISOTONIC_PROFILE_RESULT_20260903.md` | OLMo-2-1B-I | 2×/4× PG19 + core-4 | PPL retention / tail-NLL / RULER delta | `0.885660` vs `0.875302`; 4K `−0.1450` CI `[−0.2350,−0.0550]`, 8K `−0.1750` | **1**, **2** |
| `results/legacy/llama_shape_theta_min/summary.md`; `.../llama_theta_matched_shape_control/summary.md` | Llama-3-8B | 2048/16384 | WikiText PPL | `geo_10k 518.013 → 11214.916 (22.026x)`; `geo_100k 9.591 → 3547.624` | **1** |
| `analysis/unify_20260910/tables/ground_truth_tables.json` | Qwen2.5-3B-I | 32K/128K | 26 methods' six-task macro, **rebuilt from another machine's `contract.json` paths, no local receipts** | e.g. `E1_s29_more` `95.5556/77.9167` | **2** — evidence grade `[部分证据-单源重建]` (`analysis/kkt_20260910/mine/R8_receipts.md:52-74`) |

#### 5.1 How much (1) + (3) resolution actually exists

* **Scalar (1) resolution: abundant but narrow.** Per-document mean NLL exists for three models at 8192/16384/32768 (48 rows each) and OLMo at 4096/8192/16384; the underlying `token_nll` is a **512-float array per row**, so per-token NLL on the scored tail is recoverable from disk today. Nothing longer than 32K (Qwen) / 16K (OLMo) has a local NLL receipt.
* **Continuous margin (3) resolution: essentially absent.** The genuinely continuous per-row quantities on disk are (i) per-document paired NLL `delta` rows (JSONL, all three models) and (ii) per-row token-F1 for the OLMo natural-QA panels (raw JSONL; the aggregated JSONs keep only task macros). Neither is a gold-vs-distractor logit margin.
* RULER is scored as an official substring hit-rate and stored as a per-row fraction (`ruler_bench.py:9-16`, `score()`); **no gold-vs-distractor logprob margin is recorded anywhere.**
* The harness's own limits are stated in its receipts: `ROPE_OLMO_BM_NLL_RESULT_20260908.json` scope → "16 frozen natural prefixes, final 512 next-token NLL per length. Document-paired exploratory intervals; not full corpus, whole-string generation or long retrieval capability."; `ROPE_OLMO_BM_RESULT_20260908.json` `inference_limits` → "Prompts, not individual reference strings, are the units."
* `analysis/kkt_20260910/mine/R8_receipts.md:116`: "本机不存在任何 64K 或 128K 的 NLL receipt。"; `:125`: "任何'面板方法在 PG19/ProofPile 128K 上有 NLL 数字'的说法，本机零凭据。"
* **The one teacher-forced answer-NLL implementation that exists** is `long_grad.mean_answer_nll` (`long_grad.py:95-101`) + `load_bind_rows` — wired to `--long-loss gold`, used by `forward_check.py` as an alternative long objective. It has never been run (no receipts).

**Summary: (1) exists up to 32K with per-token arrays; (3) must be generated fresh.** The machinery to generate it (`mean_answer_nll`, `bound.raw_bound`/`smoothed_bound`) is written and self-tested, but has no executed receipt.

---

### 6. "Do not do this again" — the vetoes that bind a small-panel margin optimisation

#### 6.1 V-E2, verbatim

`analysis/unify_20260910/digests/digest_failure-records.md:201`:

```
| V-E2 | 重启 18 样本/64 自由度的 margin-gradient 能力优化路线 | 已失败，"不能据此重新启动"；共享频率响应工具只作局部诊断 | REVIEW-0907 行 178 |
```

Source, `docs/research/ROPE_RESEARCH_FAILURE_REVIEW_20260907.md:178`, verbatim:

> 这避免逐频率调用模型，也不需要完整L×L attention。输入可复用已经保存的选定query与所有可见keys。**该量只描述冻结hidden-state框架中的局部块响应；它不是整网LM风险的梯度，不能据此重新启动历史上已失败的18样本/64自由度margin-gradient路线。**

The same review, `docs/research/ROPE_RESEARCH_FAILURE_REVIEW_20260907.md:236`:

> 完整sin/cos有限重放，而不把Jacobian接成另一个能力优化器。

Also at `REVIEW-0907:178` region, the finite-extrapolation bound: "Mr处单点Jacobian预测整张旧p2的有限输出变化，相对误差为 `71.15–468.12`" — the measured reason the local derivative cannot be extrapolated.

Restated in the master integration doc, `analysis/unify_20260910/INTEGRATION_20260910.md:88`:

> 直接优化两条死路（64 维行为梯度未开 holdout 即败；direct-z 定支撑 pilot 挂门——**优化器必须在可辩护统计对象下游**）；**attention≠generation**（cross-cache：BM 读 MrPro 前缀能对，MrPro 读 BM 前缀仍错；record coverage 29.75→67.1 而散文精确答 7/8→6/8）

#### 6.2 V-E3 and the trust-box requirement

`digest_failure-records.md:202`:

```
| V-E3 | 把正确局部 Jacobian/Fisher 接成"下一个能力优化器" | 局部 Taylor/Fisher 只在正则性与 trust region 内保留；不作全局退休也不作全局优化目标 | REVIEW-0907 行 236；SYNTHESIS §6 行 206-209 |
```

`INTEGRATION_20260910.md:93` (the negative-data list):

> **负数据清单（禁止项已并入红线 R1–R5 与 §5）**：无符号项、对角 Σ、纯 pairwise SNR、冻结问题上的密度参数化、**无界局部步（必须 trust box ½hᵀF_Nh≤ε＋精确三角重演认证）**、μ_r≡0 时唯一合法输出是 "not identified"。

#### 6.3 Vetoes aimed directly at attention-mass reasoning

These are the ones a new attention-routing experiment must answer before it starts.

`digest_failure-records.md:143`:

```
| V-A7 | attention top-1 / attention mass ≥0.5 ⇒ 生成正确；用它解释长度断崖 | top-1 只需目标 logit 最大（权重 .4/.3/.3 即反例）；attention top-1 ≠ 生成正确；未测得通用 S_dilution | SYNTHESIS §6 行 155 |
```

`digest_failure-records.md:142`:

```
| V-A6 | 冻结态选择器分数/局部导数外推 ⇒ 全模型任务改善 | E8 强冻结态选择分数对应 128K −13.889pp；Mr 处单点 Jacobian 预测旧 p2 有限变化相对误差 71.15–468.12。局部导数只能在明确 trust region 内用，不得接成能力优化器 | AUDIT-0910 Verified 表；REVIEW-0907 行 232-236 |
```

`digest_failure-records.md:153`:

```
| V-B3 | 一个 routing/oracle 控制无效 ⇒ attention 不是瓶颈 / 关闭所有注意力处理 | 原代码仍保留其他块并用普通 QK softmax——该控制的无效性不外推 | SYNTHESIS §2 行 47 |
```

`digest_failure-records.md:276` (the author's own standing instruction): "研究迭代只能产出：有依据的具体解；可区分于剩余解释的决定性预测；或者非可识别性的证明及明确可直接测量的缺失量。**不能把'更稳定/更平滑/局部改善'当完成目标**，也不能在没有可区分预测时继续科学GPU工作。"

And `digest_failure-records.md:274`: "**不得把代理指标说成能力结果**；不得把未测写成否证；每个结论标注证据等级".

#### 6.4 How `joint_kkt_20260910` argues it is *not* the vetoed route

`risk.py:36-56` states the distinction structurally, and it is the template a new attention experiment should copy:

> **WHY THIS IS NOT THE FAILED 18-SAMPLE / 64-DOF ROUTE.** V-E2 vetoed "18 samples / 64 degrees of freedom margin-gradient" as a capability-optimisation route -- a free 64-dim search fitted to 18 behavioural rows. This runs 30 rows against 65 dof, which is the SAME n < d regime, and the honest response is not to point at the task count. …
> * the step is NOT free. It is the analytic trust-region step of sec.8 against the native output-KL metric, accepted only if the real forward honours the model's own prediction (`accept.py`). The 18-sample route had no constraint side, which is why it could fit anything.
> * the objective is a 6-group mean, not 30 free per-row weights, so the effective flexibility is 6 and not 30 -- and 6 < 30 < 65.
> * `loo_folds` is mandatory … and a group mean that does not survive LOO is reported as not surviving rather than as a small effect.
> * every receipt from this panel is stamped `development_only` …
>
> None of that makes n=30 into n>65. It makes the null space visible instead of silent, which is the only thing available at this sample size.

Enforced in code, not prose: `guard()` raises on a per-row objective (`risk.py:369-374`), `span_guard` computes the null dimension, `loop.py` is required to report `null_fraction` per accepted step.

---

## (c) GAPS — to measure attention routing as a function of depth and distance on a frozen model, what must be built from scratch?

**1. The measurement primitive — MODERATE, pattern exists.**
An attention extractor for Qwen2.5-3B that returns, for a chosen set of (layer, query position) pairs, the fp32 causal softmax over the visible prefix, **plus** an assertion that the captured Q/K are bitwise the ones actually used. Build it by generalising `experiments/native_sparse_position/activation_audit.py` (hook `q_proj`/`k_proj` or the norms, register an `ALL_ATTENTION_FUNCTIONS["sdpa"]` interface, `torch.equal` parity check at `activation_audit.py:104`) and reusing `scripts/lib/rope/attn_hist.py`'s blockwise `einsum` + `scatter_add_` accumulation. Both exist and both work on Qwen2 — the Qwen2.5 variant has already produced a receipt (`results/position_observability_20260908/psr_qwen25_activation_01/result.json`, layers 0/18/35, 2 rows, 16301 tokens, 46.6 s, 7.88 GB peak). **What is missing is the sweep, not the primitive:** `activation_audit.py` is hard-wired to 3 layers, the last 8 queries, and exactly 2 rows at one length (`:85`, `:105`, `:131-132`). A depth × distance grid needs every layer (or a declared stride) × many query positions × many rows × several lengths, which is a new driver, not a config change.

**2. The depth/distance coordinate system — MUST BE BUILT.**
Nothing in the repo computes "distance from this query position to this evidence span" for a RULER row. `niah_retention_canary.py:63-66` is the only corpus that records the needle span (`source_block=[position, len(needle)]`) and it is a 160-row canary, not RULER. For RULER rows the needle position is **not stored anywhere** — `references` holds the gold answer *string*, and the needle must be located in the prompt by re-tokenising and searching. That is a real, non-trivial build item: a span locator that maps a gold string back to token indices inside `prompt_ids`, and that reports failure rather than guessing. The cleanest route is to *prefer the canary corpus* (exact span, recorded depth, recorded length) and treat RULER as a secondary, span-inferred corpus.

**3. The three-way separation the experiment asks for — PARTIALLY EXISTS, as one component.**
* *"read correctly but not propagated"* → **already implemented**: `bound.raw_bound` / `smoothed_bound` / `bound_loss` give a teacher-forced, per-row, continuous margin with a proven bound to the greedy-output event, and `bound_summary` gives the pre-registered gate on whether the bound is informative at all (`b < −2.63` nats per answer position). Reuse it verbatim; run `bound_summary` first, because if the vacuum rate is ~1 the sec.3 objective carries no signal.
* *"attended but read wrong"* → needs the attention primitive above **plus** a defined attribution: attention mass on the evidence span at the query position (or the decoded answer positions), per layer, per head. `activation_audit.py`'s `per_head_retained_mass` is the right shape but is computed against *blocks*, not against a labelled evidence span.
* *"never attended/routed"* → needs the same mass, thresholded, plus the span locator. **No existing code computes evidence-span attention mass for any corpus.** This is the genuinely new part.
* The joining of the three into a per-row (depth, distance) → (mass, margin) table with the three outcomes separated **does not exist**.

**4. The memory/attention-mode constraint is a hard architectural limit, not a tuning choice.**
`attn_implementation="eager"` cannot be used at 32K or 128K (`[derived]` §3.4: 32 GiB and 512 GiB per layer for the score tensor alone). `output_attentions=True` is likewise unusable. Any design that assumes "get the attention matrix, then analyse it offline" is dead on arrival. The instrument must be **selective in layers and query positions** and must recompute inside the forward pass. That also means the experiment's cost model is `n_layers_sampled × n_queries × n_rows` forwards-or-cached-K scans, not one forward per row — and at ~33.9 s per 128K forward (recorded, not measured by this package) the query-position budget is the binding constraint.

**5. Corpus availability.**
The RULER input rows with `prompt_ids` for the Qwen panel are **server-side only** (`prepared_qwen3_01/screen.jsonl`; no local mirror — verified). Locally present and usable today: the OLMo 350-row/211-row `screen.jsonl` mirrors, and the canary script (which regenerates its own rows from a FineWeb parquet + native pool). Any no-GPU design work must be written against the OLMo mirrors or the canary, or must fetch the Qwen rows first.

**6. Governance — one item, and it is not optional.**
V-A7 forbids the inference "attention mass ≥ 0.5 ⇒ correct generation" and V-A6 forbids extrapolating a frozen-state selector score to whole-model task improvement. An attention-routing experiment therefore has to state, **before** measuring, what it claims and what it does not: attention mass is a *descriptive routing* quantity, and any link from it to the answer margin must be carried by the separately-measured `b_e`, not asserted. `joint_kkt_20260910`'s `risk.py:36-56` + `guard()` + `span_guard` is the existing pattern for making that argument checkable in a receipt rather than in a paragraph; a new experiment should adopt the same shape (pre-registered grouping, an explicit null-space report, `development_only` stamping, mandatory holdout).

**7. What must NOT be rebuilt.**
`FrozenRoPE` (frozen loading + `logits_to_keep` + `install_table`), `tables.py`, `mean_answer_nll`, `bound.py` in full, and `selftest.py`'s CPU-gate discipline are all reusable as-is. The gap is a **span-labelled, per-(layer, query) attention readout joined to a teacher-forced margin** — nothing less, and (for once) not much more.

---

*End of R3. Sibling recon: `analysis/route_read_20260910/recon/R1_ordering_and_tables.md`, `R2_sparse_and_training.md`.*
