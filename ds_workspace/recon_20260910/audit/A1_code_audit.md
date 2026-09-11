# A1 — Code audit: existing experiment code vs. `rope_sparse_attention_kkt_research.md`

Scope: `experiments/joint_kkt_20260910/` (8 modules), `experiments/curvature_20260910/` (README,
model.py, tables.py, arms.py, solve_kkt.py, driver.sh), plus a repo-wide search for the two
"missing derivatives" of the new document's §6.4. Every path and line number below was read in
this session. Where a thing does not exist I say **not found** rather than paraphrasing a
neighbour.

The new document is cited as **DOC** with line numbers into
`/Users/yang/Downloads/rope_sparse_attention_kkt_research.md` (612 lines).

---

## (a) VERDICT

1. **NO** for the question the code currently answers — nothing in either package is wrong about
   what it claims; the CPU gate (`selftest.py`) is complete and the algebra is formulation-internal.
2. **YES** for the question DOC asks — the design variable, the objective functional and the
   parameterization at the zero-frequency end all have to change.
3. DOC's variable (DOC:357-360, :611) is "rotation **and non-rotation** resources, per-**branch/head**
   frequency and gain, and position's role in **selection, key, value and output**". The code's is
   65 numbers: 64 shared frequencies + 1 shared gain (`design.py:54-56`).
4. The migration is **not a rewrite**. By module, ~60% of the code is formulation-agnostic:
   `qcqp.py` (all of it), `accept.py` (all but three arguments), `bound.py` (all of it),
   `risk.py`'s corpus + guard + LOO layer, `curvature_20260910/model.py`'s forward/measurement layer.
5. Exactly three things are formulation-specific: the **parameterization** (`design.py`), the
   **leaf plumbing** (`joint_grad.py`), and the **objective functional** (`loop.py:157-175`
   `build_entries` + `risk.py:396-421` group means).
6. Branch-conditioning is, almost everywhere, an **argument-level** change: a length-64 vector
   becomes an R×64 matrix, one `gain_index` becomes a slice, one `self._inv` leaf becomes 2R leaves.
7. The one piece that is **not** argument-level is cross-branch coupling in the native metric:
   `F_N` becomes an R·K × R·K object with cross blocks, and nothing in the repo has ever measured
   a cross-branch Fisher block.
8. There is **one live arithmetic hazard**: `x_j = -log nu_j` cannot reach `nu_j = 0`, and the
   chain-rule factor at `joint_grad.py:203` is `-nu`, so the gradient of the NoPE decision
   vanishes exactly where DOC's §5.2 puts it.
9. That hazard is **already recorded in this project** — `docs/research/ROPE_RESEARCH_FAILURE_REVIEW_20260907.md:180`
   states the fix (additive `nu = nu_ref + omega_native·delta`) and `scripts/analysis/shared_frequency_response.py:17`
   implements it. The new packages did not adopt it.
10. DOC's §6.2 warning ("不能把当前点的 Fourier 梯度函数冻结下来", DOC:507) is aimed at the
    frozen-Fourier-histogram route. `joint_grad.py` is **not** that route — it differentiates the
    real forward. Implementing §6.2 literally here would be a **fidelity regression**.
11. DOC's two "missing derivatives" are **both absent from the KKT packages** and **both already
    exist as forward-only operators elsewhere in the repo**
    (`experiments/deepseek_mini_position/runtime.py:145`, `scripts/analysis/check_shared_kv_position_identity.py:37-41`).
12. Naively implemented, DOC §6/§7 **re-enter V-E2** with strictly more variables than the vetoed
    route had; the migration must carry `risk.guard` + the `selftest.py:1013` gate forward.

### Per-module keep / modify / rebuild

| module | verdict | what changes |
|---|---|---|
| `experiments/joint_kkt_20260910/design.py` | **REBUILD** (variable layer only) | `K/N_DESIGN/A_INDEX` → per-branch blocks; ordering/gain/endpoint rows block-diagonal; `x=-log nu` → additive coordinate; new NoPE-share column. `gap_prices`/`gap_transfer`/`violations`/`describe` are shape-level. |
| `experiments/joint_kkt_20260910/joint_grad.py` | **MODIFY** | 2 leaves → 2R (`:147-151`, `:197-199`, `:250-256`); chain rule `:203-204` per branch. The grad-path discipline (`:14-23`) is unchanged. |
| `experiments/joint_kkt_20260910/qcqp.py` | **KEEP** | Every function takes Q, r, A, b, G, h, D, Delta and no semantics. `eliminate_gain` already takes `gain_index` (`:147`). |
| `experiments/joint_kkt_20260910/accept.py` | **KEEP** | Only `gain_index` (`:108`, `:203`, `:224`) and `model_covers_gain` (`:108`, `:204`, `:225`, `:237`) are formulation-specific. The ladder (`:276-368`) is a predicate on measured numbers. |
| `experiments/joint_kkt_20260910/bound.py` | **KEEP** | Entire file is about logits and a reference string. `bound_summary` (`:197-215`) is the pre-registered signal gate the new objective also needs. |
| `experiments/joint_kkt_20260910/risk.py` | **KEEP + ADD** | Corpus assembly (`:99-257`), guard (`:263-393`), LOO (`:424-445`) carry over. What DOC §6.1 needs — a worst case over evidence positions — is a **new** objective layer, not an edit to `group_weights`. |
| `experiments/joint_kkt_20260910/loop.py` | **MODIFY** | `n = D.N_DESIGN` (`:194`), `gain_index=D.A_INDEX` (`:224`), `:263-264`, `_clip_gain` (`:365`), and the once-measured span guard (`:463`) which must become per-iteration if the attention graph can move. |
| `experiments/joint_kkt_20260910/selftest.py` | **MODIFY** | Add the new variable's known-answer checks; change **nothing** existing. `t_panel_guard` (`:1008-1026`, gate at `:1013`) must keep firing. |
| `experiments/curvature_20260910/model.py` | **MODIFY** | One rotary object (`:83`) and one `install(values, gain)` (`:111-121`) → per-branch. The patch (`:127-141`) closes over the module it patches, so it generalizes per-branch. |
| `experiments/curvature_20260910/tables.py` | **KEEP** (one guard to add) | The m-coordinate algebra and `verify` (`:233-266`) are the migration's safety net. `inv_freq_to_m` (`:50-52`) has no zero guard — see Q2. |
| `experiments/curvature_20260910/solve_kkt.py`, `arms.py` | **KEEP** | Frozen-weight diagnostic path; DOC does not touch the frozen-MrRoPE question. |
| `experiments/curvature_20260910/driver.sh`, `preflight.py`, `RUNBOOK.md` | **KEEP** | Staged/gated structure + the "probe document is not one the panel scores" gate transfer unchanged. |

---

## (b) The six questions

### Q1 — VARIABLE MISMATCH: everywhere branch-conditioning lands

DOC's variable (DOC:360) is
`s_ij^(r) = beta_r [ <q_i^c,k_j^c> + kappa_r <q_i^p, R_{nu^(r)}(Delta_ij^(r)) k_j^p> ] + b_r(R_ij)`.
The code has one `nu[64]` and one `g` for the whole model, and `design.py:8-12` says so out loud:
"65 numbers for the whole model: 64 shared rotary frequencies and one shared attention gain.
Every layer uses the same table … That is the 'shared table' model the panel was scored under,
and it is why the design dimension is 65 and not 65*36."

#### A. Argument-level (a vector becomes a matrix, an assumption becomes an argument)

| # | site | today | branch-conditioned |
|---|---|---|---|
| A1 | `curvature_20260910/model.py:83` `self.rotary = self.model.model.rotary_emb` | one rotary module for all 36 layers | one module per branch; requires per-layer attention-type dispatch — recon R2:89 records that this **does not exist anywhere in runnable repo code** |
| A2 | `model.py:86-88` `n = int(self.rotary.inv_freq.numel()); if n != K: raise` | K=64 | still 64 per branch; unchanged |
| A3 | `model.py:111-121` `install(values, gain, track_grad)` | one vector, one float gain | `install(dict branch → (values, gain))` |
| A4 | `model.py:127-141` `_patch_grad_rotary` — `forward(self, x, position_ids)` | closure over `self.inv_freq` / `self.attention_scaling` on the single module; **no branch identity in the signature** | one closure per branch module (works), or an explicit branch id. Wrong-branch table assignment is silent here |
| A5 | `joint_grad.py:147-151` `self._inv`, `self._gain` installed on `self.f.rotary` | 2 leaves | 2R leaves |
| A6 | `joint_grad.py:197-199` `autograd.grad(total, [self._inv, self._gain])`; `:203-204` chain rule; `:250-256` per-group twin | scalar pair | vectorize over R — mechanical |
| A7 | `design.py:54-56` `K`, `N_DESIGN = K + 1`, `A_INDEX = K` | 65 | `R*(K+1)` + slices. **Every other function in the file reads these**, so this is the one edit that forces the file |
| A8 | `design.py:150-176` `ordering_rows` | one band of K−1 rows | block-diagonal, one band per branch |
| A9 | `design.py:179-192` `endpoint_rows` / `span_row` (`A[0, :k] = 1.0`) | one bank over K slots | per-branch blocks **plus a modelling decision**: is the total span shared across branches? DOC §4 (DOC:255-262) says the four operations need different distance ranges, so the bank should *not* be shared — that is a choice, not a reshape |
| A10 | `design.py:195-230` `gain_box_rows` | one column `A_INDEX`; the docstring at `:198-223` records two shipped bugs in this function | R columns, one box each |
| A11 | `design.py:233-279` `feasible_set`; shape guard at `:258-263` | hard N_DESIGN | same generalization |
| A12 | `design.py:365-419` `violations`; `freq_order_ok` at `:416-418` | one vector unpacked | per-branch |
| A13 | `accept.py:203-204`, `:225`, `:236-237` `gain_index`, `model_covers_gain` | one scalar gain | slice / per-branch dict |
| A14 | `loop.py:194` `n = D.N_DESIGN`; `:224` `gain_index=D.A_INDEX`; `:263-264`; `:365` `_clip_gain` | — | mechanical |
| A15 | `qcqp.py:147-181` `eliminate_gain(Q, r, gain_index, interior=True)` | already generic in `gain_index`; refuses a non-interior gain at `:161-163` | apply R times in sequence (block elimination) |

#### B. Genuinely new mathematics

**B1. Cross-branch coupling in the native metric.** The constraint `N(theta) <= N_0 + eps` is one
number for the whole model. With R tables it is a joint constraint over R·K coordinates, i.e.
`F` becomes an `RK × RK` object with cross blocks `F_{rs}`. `qcqp.solve_epigraph` is already
joint (`qcqp.py:350-476` never factorizes Q), so **the algebra is agnostic**; the **measurement**
is not. `curvature_20260910/model.py:219-233` `fisher_diagonal` costs one forward per slot and
`:236-253` `fisher_cross` one forward per named pair. A block-truncated Fisher is a modelling
choice whose error is unproven at this size, and `solve_kkt.py:236-243`'s `step_diag_share` is
the only existing instrument that would show the truncation was wrong.

**B2. Worst case over evidence positions.** DOC §6.1 (DOC:443-444) requires
`E_{c,u}^in(xi) <= t` **for every task c and every evidence position u**. The existing min-max is
over 6 group **means** of 5 rows each: `risk.py:396-421` `group_weights`, `loop.py:157-161`
(one risk entry per group with the group-mean gradient), `qcqp.py:404-406` `objectives` takes the
max over those entries. Averaging over positions is not maximizing over them. Making the panel a
worst case over positions turns 6 numbers into 30 and makes the `n < d` ratio worse — see V-E2.

**B3. The `O(xi)` ordering constraint.** DOC:447 requires an ordering-sensitivity constraint
"避免通过抹掉所有位置来改善无序检索". Nothing in the repo is that. `design.py:150-176`
`ordering_rows` is a **frequency-monotonicity** regularity condition on the design
(`x_{j+1} - x_j >= delta_min`), not a sequence-order capability constraint, and it lives on the
wrong side of the problem (it constrains the table's own shape, not the model's behaviour).
New measurement, new constraint.

**B4. The NoPE share.** See Q2.

---

### Q2 — THE ZERO-FREQUENCY ATOM

DOC:399 states it exactly:

> 有限 log-frequency 区间上的连续密度，不能表示真正的零频原子，因为 `-log 0 = infinity`。

**Can the current parameterization express a NoPE channel? No.**

- `design.py:70-71` `nu_of_x(x) = np.exp(-x)` is strictly positive for every finite `x`, and
  `design.py:62-67` `x_of_nu` raises `"frequencies must be positive"` for `nu <= 0`. A true zero
  sits at `x = +inf`, off the axis.
- `tables.py:46-47` `m_to_inv_freq(m, theta) = native * 4**(-m)` reaches 0 only as `m -> inf`;
  `tables.py:50-52` `inv_freq_to_m` would need `log(0)`.
- `m_evq_deployed` (`tables.py:147-155`) anchors endpoints at `m_0 = 0`, `m_63 = 1`; the far end
  is `nu = omega/S`, never 0.

**Guards that exist.** `design.py:65-66` (positive frequencies), `tables.py:226-228`
(`from_eps` raises `"step leaves the positive-frequency cone"`), `qcqp.py:238-241`
(`implied_box` refuses a singular D).

**Where a zero frequency is accepted without a guard:** `curvature_20260910/model.py:113`
rejects `(v < 0).any()` but **accepts `v = 0`** — which is correct for the forward (cos=1, sin=0
is a NoPE channel times gain) but means the metric layer will silently carry it. And
`tables.py:50-52` `inv_freq_to_m` has no guard at all: `np.log(nu/native)` at `nu = 0` gives
`-inf`, which then poisons `m_sum`/`m_min`/`gap_min` in `tables.py:269-277` `describe` and in
`design.py:422-429` `describe` — the two functions every receipt uses. Not silent (numpy warns on
`log(0)`), but not a named error either.

**The one place that would produce a wrong answer rather than an error — and it is exactly where
the new decision lives.** `joint_grad.py:203`:

```python
grad[:K] = -nu * g_inv.detach().double().cpu().numpy()
```

with the docstring at `:29-31` asserting "both factors are strictly positive, so no direction is
annihilated." That claim is **false at the NoPE end**. As `nu_j -> 0`, the factor `-nu_j -> 0`, so
`dL/dx_j -> 0` while `dL/dnu_j` is generically **nonzero** (at `nu = 0`, `freqs = 0`, `cos = 1`,
`sin = 0`, and `d(sin)/d(nu) = pos·cos = pos`). The coordinate that would carry DOC's
`alpha_{0,r}` is therefore a **dead direction exactly where the decision lives** — the same
disease as the recorded softplus dead zone, in a new place.

**This project already derived the fix, in a different file, and the new packages did not adopt
it.** `docs/research/ROPE_RESEARCH_FAILURE_REVIEW_20260907.md:180`, verbatim:

> 参数使用 ν=ν_ref+ω_native·δ，而不是对含零频率的表做乘法更新。原因是ν_ref=0时，ν_ref exp(α)对α的导数恒为0，无法重新引入旋转；实际∂y/∂ν却可以非零。相对Native的加法坐标保留该自由度，不规定它应当增加或减少。

and `:182` records that a zero-frequency slot was part of the CPU check that verified this
("一个零频率槽，中心有限差分最大绝对误差8.44e−10"). The additive coordinate is **implemented**
at `scripts/analysis/shared_frequency_response.py:17`: "Coordinates satisfy
`nu = frequencies + parameter_scale * delta`. Raw frequency derivatives use scale=1; Native
frequencies provide relative-to-Native coordinates that remain usable when a deployment frequency
is zero."

**Minimal honest extension**, in DOC's own two-layer form (DOC:404): (i) a discrete per-branch
NoPE share `alpha_{0,r}` as its own design column, holding `alpha_{0,r}·K` channels at `nu ≡ 0`
(identity rotation); (ii) the additive coordinate `nu = nu_ref + omega_native·delta` for the
**rotary remainder only**, because (iii) the multiplicative `x = -log nu` has an identically zero
gradient in the `alpha_{0,r}` direction. (i)+(ii) are an argument-level extension of
`design.pack`/`unpack` plus one box row; the *reason* (iii) is the arithmetic that has to be
written into the receipt.

**Is the trap reachable today?** No. `design.py:89-90` `native_x` with `tables.py:42-44` gives
`x_j = j·log(theta)/K + m_j·log S`, i.e. `x in ~[0.11, 15.0]` at Qwen `theta = 1e6`, `K = 64`
(`tables.py:28`). float64 `exp(-x)` underflows to 0.0 at `x ≳ 745`, and the reachable movement is
bounded by `cfg.delta0 = 0.05` over `cfg.max_steps = 8` (`loop.py:72-73`). So the structure is
wrong but the operating point is far from it — which is why it has not bitten.

---

### Q3 — THE OBJECTIVE: same object or a different one?

**Answer: a different *functional*, the same *differentiation technique* — and DOC's warning is
aimed at the wrong target for this repo.**

**What `joint_grad.py` actually computes.** `:197-199`:

```python
g_inv, g_gain = torch.autograd.grad(total, [self._inv, self._gain],
                                    retain_graph=retain_graph, allow_unused=False)
```

This is the **exact derivative of the whole-network loss through the real forward**, at the current
weights *and* the current frequencies. `:14-23` states the three constraints that make that true
(one backward for all 65; weights frozen but the forward **not** under `no_grad`; the frequency
cache not detached), and `:288-324` `finite_difference_check` is the end-to-end gate against a
central difference of the real forward.

Therefore it **does** capture, by construction rather than by projection:

- **the content phase** — the `phi_ijm` / `C_m` of DOC:482-498 is the inner product with the
  actual hidden states, which is inside the graph;
- **attention competition** — the softmax is inside the graph, and DOC's
  `dL/ds_ij = a_ij g_i^T (v_j - o_i)` (DOC:476) is the analytic statement of exactly what that
  backward implements;
- **the value's contribution to the loss** — the same expression;
- **accumulation over all layers and heads that share the frequency** — DOC:505 requires it and
  `joint_grad.py:149-151` installs one buffer used everywhere, so the sum is in the graph.

**What the code does *not* capture is the objective.** `loop.py:127-175` `build_entries` assembles
`L = sum_i w_i q_i` where the `q_i` are (a) `bound.py`'s per-instance bound on
`1{greedy != reference}` (`bound.py:136-143`, valid by the induction at `:1-18`), (b) the native
output-KL, and (c) per-**group-mean** risk (`loop.py:157-161`). Against DOC §6.1 that is:

| DOC §6.1 term | in the code? |
|---|---|
| `E^{in}` per task | **partly** — `risk.py:130-154` groups by (length, task); `bound.py` is a per-instance capability objective, which is much closer to `E^{in}` than a PPL would be |
| worst case over **evidence positions u** | **NO** — group means over 5 rows (`risk.py:396-421`); `risk.py:369-374` explicitly *refuses* a per-row objective |
| `N(xi) <= N_0 + eps_N` | **YES** — native output-KL, `accept.py:79-82`, `:201-202`, measured not modelled |
| `O(xi)` ordering-sensitivity | **NO** — see B3 |
| `C_compute`, `C_KV` | not modelled; at frozen weights these are fixed, so this is vacuous today |
| `min t` epigraph form | **YES, structurally** — `qcqp.solve_epigraph` (`:350-476`) minimizes `t = max_i q_i` under log-sum-exp smoothing, so the *solver* is already the min-max solver DOC §6.1 asks for. The difference is what the max ranges over |

**DOC's §6.2 warning does not apply here.** DOC:507:

> `C_m` 随模型状态和频率变化，不能把当前点的 Fourier 梯度函数冻结下来，声称已经解出了全局真实损失。

The frozen-Fourier route is `scripts/analysis/shared_frequency_response.py`, whose own docstring
(`:1-5`) says: "It describes a local attention block, not a downstream loss gradient or a method
selector." The project recorded it as such at `docs/research/ROPE_RESEARCH_FAILURE_REVIEW_20260907.md:167-169`
("正确的共享频率响应，不把它冒充能力优化器"). **`joint_grad.py` is not that route and never was.**
Implementing DOC §6.2 as an explicit `C_m(Delta)` estimator over this codebase would replace an
exact derivative with a projected one — a fidelity **regression**, not an upgrade. The correct
reading of §6.2 here is: keep `joint_grad.py`, keep the trust region, and do **not** add a
histogram layer.

---

### Q4 — THE TWO MISSING DERIVATIVES

**(a) The value / output position path.** DOC:551-555: "只使用 QK 的 LeRoPE 公式，会漏掉
DeepSeek-V4 这类结构的一部分真实代价."

- **In the KKT packages: not found.** The model is Qwen2.5-3B (`tables.py:28`), whose attention
  has no value rotation; `curvature_20260910/model.py:147-154` `logits()` is the stock forward.
- **The framework would capture it for free** if the model were swapped: `joint_grad.py:172-210`
  differentiates whatever forward is installed. The derivative is missing from the **model set**,
  not from the differentiation machinery.
- **The operator itself is already written twice in this repo**, forward-only:
  `experiments/deepseek_mini_position/runtime.py:145`
  `out = module._apply_output_rope(out, layer_cos, layer_sin, positions)` on a downloaded
  DeepSeek-V4 Mini (with `_apply_output_rope` living in the external
  `results/reference_position_20260909/deepseek_mini_source/modeling_deepseek_v4.py`); and
  `scripts/analysis/check_shared_kv_position_identity.py:37-41` `absolute_forward()`, which is
  literally DOC:333's `R(-p_i) sum_j a_ij R(p_j) v_j`. Its docstring (`:3-4`): "This checks an
  operator identity; it is not a frequency optimizer or benchmark."
- **What has to be built:** a frequency-differentiable value-rotating attention module, plus a
  trained checkpoint on which a frequency optimum means anything. The second is the blocker — the
  Mini workbench is random-init and externally downloaded.

**(b) The selection-set change.** DOC:556-569: top-k is piecewise smooth; `Delta_route > 2 eta`
suffices for the set to be unchanged.

- **In the KKT packages: not found.** There is no top-k anywhere — the graph is the causal mask
  over a full prefill. The whole package is single-graph by construction.
- **Elsewhere in the repo**, three of the four pieces exist:
  - hard top-k with a separate indexer: `experiments/deepseek_mini_position/runtime.py:136`
    `selected = scores.topk(min(module.config.index_topk, compressed.shape[1]), -1)`;
  - the DSA-style indexer supervision DOC:567 cites: `runtime.py:64-76` `index_kl()`, wired at
    `:126` `module.index_aux_loss = index_kl(...)` and collected at `:165-169` `auxiliary_loss()`
    ("The auxiliary objective trains the indexer, not the teacher backbone."). The routing scores
    at `:123` `scores = (torch.einsum('blhd,bsd->blhs', iq, ik).relu() * iw[..., None]).sum(2)`
    already depend on the frequencies through the rotations at `:120-121`;
  - an actual **selection margin**: `experiments/nosa_position/analyze_cascade_failures.py:32`
    computes `'cutoff_margin_min': float((sortedscore[:, free-1] - sortedscore[:, free]).min())`
    — that is DOC's `Delta_route`, measured post hoc.
- **What has to be built**, quantified: (i) carry the graph from the routing scores back to the
  frequency leaves instead of treating selection as an index — small, and the scores are already
  a differentiable function of the rotations; (ii) a **per-step** `Delta_route` diagnostic with
  `eta` estimated as `max|∂s/∂nu| · Delta` from the trust radius — new, and DOC does not say how to
  bound `eta` (see "where DOC is unactionable"); (iii) a smooth relaxation or the `index_kl`
  surrogate for the `Delta_route <= 2 eta` branch — a modelling decision with unproven error.

**Does `accept.py` / `loop.py` assume a FIXED attention graph? Mostly no — in one place, explicitly
yes.**

- **No, in the predicate.** `accept.check_acceptance` (`accept.py:106-244`) compares *measured*
  real changes against model predictions; a graph switch shows up as a `(*)` violation
  (`:148-152`), and the ladder's remedy — shrink Delta, raise mu (`escalation_for`, `:247-268`;
  monotone by construction, `:64-67`) — is exactly the remedy that keeps the step inside the smooth
  region. `run_ladder` re-measures at **both** points on every attempt (`:276-308`), so a stale
  value can never be reused. `phase_one` (`loop.py:283-352`) verifies its own step against the real
  function (`:337-341`). **The machinery is sound under a moving graph.**
- **Yes, here:** `loop.py:463`
  `guard = RK.span_guard(row_grads) if row_grads is not None else None`, with the docstring at
  `loop.py:456-458`: *"It is measured ONCE and reused: the guard is a statement about which
  directions the panel can price, which is a property of the panel and the objective, not of the
  current step."* If the selection set can change, the per-row Jacobians `∇f_e` change, and
  `null_fraction` (`loop.py:257`, `risk.py:320-346`) becomes a statement about a graph that no
  longer exists. **That is the exact entry point.** The fix is to re-measure per iteration — one
  extra `n_rows`-backward batch per step.
- **A missing diagnostic, not a missing guarantee:** `accept.py` has no field that distinguishes
  "the curvature was underestimated" from "the selection set moved". Both surface as the same
  `(*)` violation, and the ladder responds identically. Given DOC:569 ("固定图内的解析／二阶更新，
  与图改变后的真实能力验证必须区分"), that distinction has to be added as a field.

---

### Q5 — WHAT SURVIVES UNCHANGED

Being explicit, because this is the migration's cost.

1. **`experiments/joint_kkt_20260910/qcqp.py` — the whole file.** `build_quadratic` (`:105-125`),
   `psd_floor` (`:128-135`), `damp` (`:138-141`), `eliminate_gain` (`:147-181`, already generic in
   `gain_index`), `analytic_step` (`:187-208`), `predicted` (`:211-216`), `implied_box` (`:222-243`),
   `project_tr` (`:246-254`), `repair_step` (`:257-335`), `solve_epigraph` (`:350-476`). Every
   function's arguments are Q, r, A, b, G, h, D, Delta. **Zero semantic content.** DOC §6.1's
   `min t` is already this solver's structure.
2. **`experiments/joint_kkt_20260910/bound.py` — the whole file.** `raw_bound` (`:71-100`),
   `smoothed_bound` (`:103-133`) with the `b <= b_tilde <= b + eta` sandwich, `bound_loss`
   (`:136-143`), `gamma_from_native` (`:155-167`), `d_keep` (`:170-194`), `bound_summary`
   (`:197-215`). A statement about logits and a reference string; indifferent to what
   parameterizes the logits. **`bound_summary` is the pre-registered gate that decides whether the
   objective carries signal at all** and the new objective needs it unchanged.
3. **`risk.py`'s corpus and panel layer:** `load_rows` (`:99-127`), `group_key` (`:130-134`),
   `build_groups` (`:137-154`), `reference_ids` (`:160-192`), `teacher_forced_record` (`:195-228`),
   `answer_mask`/`mask_positions` (`:231-257`), `budget_seconds` (`:448-461`), `panel_receipt`
   (`:465-482`). None of them knows what a frequency is.
4. **`risk.span_guard` (`:263-301`), `projector` (`:304-317`), `null_fraction` (`:320-346`).**
   Pure rank/projection on `(n_rows, d_design)`. Only `d_design` changes, and it is an argument.
5. **`risk.guard` (`:349-393`).** Its *arguments* change (d grows); its *logic* does not — and the
   new formulation makes `n < d` worse, so this gate becomes **more** load-bearing.
6. **`risk.group_weights` (`:396-421`) and `loo_folds` (`:424-445`).** The weighting and
   leave-one-out machinery is a statement about a panel. DOC §7's "位置搬移＋顺序对照" is a new
   panel, not a new estimator.
7. **`accept.py`'s discipline:** the deltas-not-values convention (`:106-130`), the two-sided rho
   band (`:39-48`, `:187-199`), the monotone ladder (`:64-67`, `:247-368`), and the "a refused step
   is a result" branch (`:363-368`). Only `gain_index` and `model_covers_gain` are
   formulation-specific.
8. **`loop.py`'s machinery:** `isotropic_curvature` (`:90-124`) — the `beta <= 2/3` arithmetic at
   `:99-119` is about a trust-region step in *any* parameterization; `phase_one` (`:283-352`);
   `stationarity_diagnostic` (`:372-421`); `gap_transfer_table` (`:424-442`); `pareto` (`:558-615`)
   and its `dE*/deps = -lambda` check.
9. **`curvature_20260910/model.py`'s measurement layer:** `load_frozen` (`:35-60`), the grad-path
   discipline (`:127-144`), `logits` (`:147-154`), `log_probs` (`:167-169`), `grad_wrt_freq`
   (`:171-179`), and `output_kl` (`:187-210`) — the KL-not-cross-entropy argument at `:187-205`
   ("A plain cross-entropy difference would have a non-zero first-order term and an indefinite
   second-order form") holds for any metric that is a Fisher.
10. **`curvature_20260910/tables.py`'s algebra:** `native_inv_freq`/`m_to_inv_freq`/
    `inv_freq_to_m` (`:42-52`), the m-coordinate, and **`verify` (`:233-266`)**. That gate is what
    makes a migration checkable at all — it pins the coordinate against the deployed tables before
    any card is touched.
11. **`curvature_20260910/preflight.py`, `driver.sh`'s staged/gated structure, `RUNBOOK.md`'s
    pre-registration.** Procedural, and the migration needs all of it.

**Honest cost statement:** the numerical core (items 1, 2, 8, 9, 10) is essentially 100% reusable.
What is formulation-specific is three layers — the parameterization (`design.py`), the leaves
(`joint_grad.py`), the objective (`loop.py:157-175` + the group-mean layer of `risk.py`).

---

### Q6 — CONTRADICTIONS AND VETOES

Repo-wide search for the four named rules and V-E2. Exact wording, with the primary source.

#### V-E2, primary source `docs/research/ROPE_RESEARCH_FAILURE_REVIEW_20260907.md:178`

> 该量只描述冻结hidden-state框架中的**局部块响应**；**它不是整网LM风险的梯度**，不能据此重新启动历史上已失败的18样本/64自由度margin-gradient路线。

The digest label (`analysis/unify_20260910/digests/digest_failure-records.md:201`, quoted at
`analysis/route_read_20260910/recon/R1_ordering_and_tables.md:305`) reads:
"V-E2 | 重启 18 样本/64 自由度的 **margin-gradient 能力优化路线** | 已失败，'不能据此重新启动'；
共享频率响应工具只作局部诊断 | REVIEW-0907 行 178". The "18 samples" origin is
`paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md:385-386`
(via R1:312-318): "64 DOF on 18 samples — so the measurement that would have supplied margins has
already empirically collapsed."

Both packages carry a pre-emptive defence:
`experiments/curvature_20260910/README.md:262-311` (quote at `:267-268`, distinction table at
`:278-283`, failure list at `:303-311`), and `experiments/joint_kkt_20260910/risk.py:36-56`
("WHY THIS IS NOT THE FAILED 18-SAMPLE / 64-DOF ROUTE"). The strongest existing pattern is that
the acknowledgement is a **machine-checked gate**, not prose: `selftest.py:1013`
`raise Fail("the V-E2 acknowledgement is missing from the receipt")`.

#### The four rules, verbatim, `analysis/unify_20260910/INTEGRATION_20260910.md:26-30`

> **红线**（全部两条线共同确认）：
> - R1 根（Σcos 第一零点）= 诊断量，**禁入 F**；但注意其定理外衣已被缩窄（§7 FLAG-1）。
> - R2 静态几何代理（碰撞能/覆盖/平滑/有效秩/能量/MAE/轨道计数）**不得作为选择子**——Smooth_MrBudget 几何全赢、far 端 68.3 vs MrPro 78.13（near 打平），是决定性反代理。[已验证-面板]
> - R3 端点 m=0/m=1 与 gain=1+0.1·lnS 是设计面（face），不是被证定律。
> - R4 Σm 质心是**自由决策变量**，水床守恒只在指定坐标里成立（零和频移 ΔΣm≠0，joint_mode 实测 −0.01826…+0.00185）——**任何守恒论证必须先点名坐标**。

Reinforced at `analysis/unify_20260910/NEXT_DERIVATION_KKT_PROBLEM.md:109-116`:
"1. 任何静态几何/代理量（根、碰撞核、effrank、Gram、曲率、平滑度）不得作为 F 的分项或选择器——只可作窗口内诊断"
and `:113` "5. 线性读出/固定态代理算子（J_r、校准-KL、E7 160×）只配假设生成，不入 F 主链；gain×相位不正交（6Pro 第 6 点），**F 不含 gain 自由度**。"

Note the exact wording difference between R1 and R2 as the task states them: R1 is "the first zero
of Σcos is a **diagnostic** and must not enter F"; R2 is "static geometric proxies must not be used
as **selectors**". Both are in `INTEGRATION_20260910.md:27-28` above. The task's phrasing of R2
("must not be used as selectors") is a subset of the repo's — the repo also forbids them as an
**F term**. Recorded here so the difference is not lost.

#### Would DOC §6/§7, naively implemented, re-enter a vetoed route?

**V-E2 — YES, and this is the most serious finding of this audit.** Four separate reasons:

1. **More variables, same panel shape.** DOC §5.1 gives each branch `beta_r`, `kappa_r`,
   `nu^(r)` and `b_r(R_ij)`. `b_r` is an **undefined free function** — it is not a function of
   anything named in the document. A free function per branch over an unbounded domain against a
   behavioural panel is V-E2 with more freedom than the vetoed route had, not less.
2. **A worst case over evidence positions is a more flexible objective than a 6-group mean.**
   `risk.py:369-374` already *refuses* a per-row objective by raising:
   *"a free per-row fit against a {d_design}-dimensional design is the vetoed 18-sample/64-dof
   route."* DOC §6.1's `E_{c,u}` maximises over positions rather than averaging them.
3. **DOC does not carry the constraints forward.** The existing defence is three specific things —
   the step is not free (trust region + `accept.py`), the objective is a 6-group **mean**, LOO is
   mandatory — and DOC §6/§7 mentions none of them.
4. **The existing acknowledgement gate would silently stop applying.** `selftest.py:1013` checks
   the V-E2 note is in the receipt; if the design vector grows, `risk.guard(30, 65, 6)` at
   `selftest.py:1009` no longer describes the run and must be re-parameterized, or the gate reports
   on a problem nobody is solving.

**Required, therefore:** the new objective must inherit `risk.guard` (`risk.py:349-393`), LOO
(`risk.py:424-445`), the `development_only` stamp (`risk.py:366`), and the `selftest.py:1013`
gate — **before** any new variable is added. Also relevant: **V-E3** (`REVIEW-0907:236`, cited in
the curvature README's defence at `:288-293`) — the local Taylor/Fisher is retained "只在正则性与
trust region 内". DOC §6.2's gradient is a local gradient; the same discipline applies.

**R3 — YES.** DOC §5.1 makes `beta_r` and `kappa_r` free design variables. R3 says the gain is a
**design face, not a proven law**, and `NEXT_DERIVATION:113` adds "**F 不含 gain 自由度**" with the
stronger reason "gain×相位不正交". The existing code encounters this from the numerical side only:
`accept.py:20-37` explains that the frozen `F_N` "carries 64 frequency rows only", so a gain move is
flagged `gain_unpriced` (`accept.py:225`, `loop.py:264`) as a **warning, not a refusal**. That is a
weaker position than the red line states: R3's objection is not "the Fisher lacks the coordinate"
but "pricing a gain with a phase-only Fisher is structurally wrong". `kappa_r` is a **new**
coordinate not covered by R3 — but the migration must not price `kappa_r` with an F that has no
`kappa_r` coordinate, which is the identical trap `accept.py:204` already names.

**R4 — no direct conflict, one trap.** DOC makes no conservation claim; §6.3's interval-price rule
is preserved **within** a branch (DOC:526). But if cross-branch allocation is later described as a
"budget transfer" with a conserved budget, R4 fires: "**任何守恒论证必须先点名坐标**". The existing
code is consistent with the rule: `design.py:39-42` calls `pin_ends` and `fix_span`
"off-by-default-recalled design choices … a matched-support experimental convention", not identities.

**R2 — no conflict; DOC is on the correct side, with one exposure.** DOC never proposes a static
geometric proxy as a selector; §4.4 (DOC:328-338), §6.2 (DOC:499-503) and §6.4 (DOC:545-569) all
push toward signed, dynamic, checkpoint-specific quantities. DOC's §3.2 (DOC:218-224,
"提高中间位置的 attention mass ≠ 提高中间信息的实际利用") is a restatement of R2/V-A7. **The exposure
is DOC §7's first row** ("位置搬移＋顺序对照", DOC:578-580): if it is operationalized with an
attention-mass readout, it hits **V-A7** — recon R3:21 records V-A7 as forbidding "attention top-1 /
attention mass ≥ 0.5 ⇒ correct generation".

**R1 — no conflict.** DOC:18 explicitly says "cosine sum 的零点不是任意真实模型的有效上下文上限".
DOC cites it as a RoPE-Bound fact rather than as this project's red line, but the content agrees.

**V-E4 — branch-conditioning is on the *right* side of it.** Recon R1:305 records V-E4 as
"用频率置换重新发现同一个 multiset 限制 → 限制已确立；保留槽位身份，不重复置换"
(SYNTHESIS §5 line 101). Splitting the table by branch **holds the multiset fixed and changes the
arrangement**; it does not re-discover the multiset limit by permuting. That makes
branch-conditioning one of the few directions V-E4 does not close — worth stating in the receipt,
since R1:352-353 records that no machine-checked V-E4 gate exists yet anywhere.

---

## (c) MIGRATION COST

Ranked by effort, with the argument-level / new-mathematics split made explicit.

### Tier 0 — argument-level, < 1 day each

| # | change | sites |
|---|---|---|
| 0.1 | Change `model_covers_gain` / `gain_index` from scalar to slice-or-dict | `accept.py:108,125,203-204,225,236-237,279,336-337`; `loop.py:224,263-264,365`; `qcqp.py:147` (already generic) |
| 0.2 | Re-measure the span guard per iteration instead of once | `loop.py:463` (+ `risk.span_guard`, unchanged) — cost: one extra `n_rows`-backward batch per step |
| 0.3 | Add a `graph_moved` field to the acceptance verdict | `accept.py:215-244`; consumed by `escalation_for` (`:247-268`) |
| 0.4 | Guard `inv_freq_to_m` against `nu <= 0` | `tables.py:50-52` — currently produces `-inf` and poisons `describe` (`:269-277`, `design.py:422-429`) |

### Tier 1 — argument-level, 1-3 days each

| # | change | sites | note |
|---|---|---|---|
| 1.1 | 2 leaves → 2R leaves, chain rule per branch | `joint_grad.py:147-151,197-199,203-204,250-256` | mechanical; `finite_difference_check` (`:288-324`) extends to R per-branch slots and is the gate |
| 1.2 | One rotary → per-branch rotaries; per-branch install | `curvature_20260910/model.py:83,111-121,127-141` | the patch already closes over the module it patches, so one closure per branch is natural. **Blocker:** per-layer attention-type dispatch does not exist in runnable code (recon R2:89) |
| 1.3 | `K/N_DESIGN/A_INDEX` → per-branch blocks | `design.py:54-56,96-119,150-230,233-302,365-419` | one edit to `:54-56` forces the file; `gap_prices`/`gap_transfer` generalize by slicing per branch |
| 1.4 | Cross-branch bank decision (shared span vs per-branch) | `design.py:179-192` `span_row`, `:233-279` `feasible_set` | a modelling decision with an arithmetic consequence: a shared `span_row` couples the branches and changes what `gap_transfer` means |

### Tier 2 — new mathematics, 3-10 days each

| # | change | why it is new | anchor |
|---|---|---|---|
| 2.1 | **NoPE-share variable + additive frequency coordinate** | the multiplicative `x = -log nu` has an identically zero gradient in the `alpha_0` direction (`joint_grad.py:203`, `design.py:70-71`). The additive coordinate is already derived in this repo and must be re-derived into `design.py`'s feasible set | `REVIEW-0907:180`; `scripts/analysis/shared_frequency_response.py:17`; new column in `design.pack`/`unpack`; new box row alongside `gain_box_rows` (`design.py:195-230`) |
| 2.2 | **Cross-branch native metric** | `F` becomes `RK × RK` with cross blocks. The solver is already joint (`qcqp.py:350-476`); what is new is the **measurement** — one forward per slot / per pair today (`model.py:219-253`), so a truncation scheme is needed and its error is unproven. `solve_kkt.py:236-243` `step_diag_share` is the only existing instrument that would catch a bad truncation | new measurement module alongside `model.py`; trust box unchanged (`qcqp.py:257-335`) |
| 2.3 | **Worst case over evidence positions** | averaging is not maximizing. Converts 6 group means into a per-position max, which makes `n < d` worse — must ship with `risk.guard` re-parameterized and a fresh V-E2 receipt | `risk.py:396-421` (new estimator) + `risk.py:349-393` (gate) + `selftest.py:1008-1026` |
| 2.4 | **`O(xi)` ordering-sensitivity constraint** | does not exist anywhere. `design.py:150-176` is frequency-monotonicity on the design, not a capability constraint | new measurement + new constraint row; DOC gives no construction |

### Tier 3 — new mathematics AND new hardware/architecture, weeks

| # | change | blocker |
|---|---|---|
| 3.1 | Value / output position path derivative | the machinery is free (`joint_grad.py:172-210` differentiates any forward); the operator exists forward-only (`experiments/deepseek_mini_position/runtime.py:145`; `scripts/analysis/check_shared_kv_position_identity.py:37-41`). **Blocker: no trained checkpoint with value rotation.** The Mini workbench is random-init and externally downloaded |
| 3.2 | Selection-set change / `Delta_route` | the routing scores already depend on the frequencies (`runtime.py:120-123`); the missing pieces are (i) keeping the graph, (ii) a per-step `Delta_route` with an `eta` bound, (iii) a relaxation for `Delta_route <= 2 eta`. All three are new; none is a config change |
| 3.3 | Per-layer local/global branch dispatch | recon R2:89 — **not found** in runnable repo code. Days-to-weeks on its own, independent of this package |

**One-line summary of cost:** Tier 0 + Tier 1 (~1-2 weeks) buys the branch-conditioned version of the
*existing* machinery. Tier 2 (~3-4 weeks) buys DOC's *new* question. Tier 3 is a different project.

---

## (d) VETO CHECK

| veto | source | DOC §6/§7 naive implementation | required mitigation |
|---|---|---|---|
| **V-E2** "18 样本/64 自由度 margin-gradient 能力优化路线" — 已失败，"不能据此重新启动" | `REVIEW-0907:178` (verbatim above) | **RE-ENTERS.** More variables (§5.1 adds `kappa_r` and an undefined `b_r`), and §6.1's min-max over evidence positions is *more* flexible than the vetoed route's objective. DOC does not mention the existing defences | carry `risk.guard` (`risk.py:349-393`), LOO (`:424-445`), the `development_only` stamp (`risk.py:366`), and re-parameterize the `selftest.py:1013` gate **before** adding variables |
| **V-E3** 局部 Jacobian/Fisher 只在正则性与 trust region 内保留 | `REVIEW-0907:236` | **At risk** if §6.2's gradient is used outside a trust region | keep the trust box `½ dᵀ F_N d <= eps` (`accept.py:79-82`) and the replay gate (`curvature_20260910/forward_check.py`) |
| **V-E4** 用频率置换重新发现同一个 multiset 限制 | recon R1:305 (SYNTHESIS §5 L101) | **NOT re-entered** — splitting by branch holds the multiset fixed. This is a point in the migration's favour | state it in the receipt; no machine-checked V-E4 gate exists yet (R1:352-353) |
| **R1** Σcos 第一零点 = 诊断量，**禁入 F** | `INTEGRATION:27` | **NOT re-entered.** DOC:18 reaffirms it | — |
| **R2** 静态几何代理不得作为选择子（and, per the repo, not as an F term either） | `INTEGRATION:28`; `NEXT_DERIVATION:109` | **NOT re-entered in the formulation**, but **exposed in the experiment design**: DOC §7's first row could be read as an attention-mass readout, which is **V-A7** | pre-register the readout as a *task* quantity (RULER-style), not an attention-mass quantity |
| **R3** 端点 m=0/m=1 与 gain=1+0.1·lnS 是设计面（face），不是被证定律 | `INTEGRATION:29`; `NEXT_DERIVATION:113` "F 不含 gain 自由度" + "gain×相位不正交" | **RE-ENTERS.** §5.1 makes `beta_r` a free design variable. The existing code handles this only numerically (`accept.py:20-37`, `gain_unpriced` at `:225`) and explicitly downgraded it to a *warning* | decide deliberately: either `beta_r` is a design face (R3) and only `kappa_r` is free, or the F must gain a `beta_r` coordinate — the current one cannot price it (`accept.py:204`) |
| **R4** Σm 是自由决策变量；任何守恒论证必须先点名坐标 | `INTEGRATION:30` | **No direct conflict.** Trap: describing cross-branch allocation as a conserved "budget transfer" | name the coordinate in every cross-branch statement; the within-branch rule (DOC:526) is unaffected |
| **V-A7** attention top-1 / attention mass ≥ 0.5 ⇒ correct generation, forbidden | recon R3:21 | **Exposed via DOC §7 row 1** | use task outcomes, not attention mass |
| **V-A6** frozen-state selector score ≠ whole-model task improvement | recon R3:21 | Exposed via DOC §6.4(b)'s `Delta_route`: a routing-score margin is a frozen-state quantity | keep the "fixed-graph update vs. graph-changed capability validation" split DOC:569 demands, as a receipt field |

---

## (e) Where I think DOC is wrong or unactionable

1. **§6.2 aims at the wrong target for this codebase.** The warning about a frozen Fourier gradient
   function does not describe `joint_grad.py`, which differentiates the real forward. Implementing
   §6.2 literally would replace an exact derivative with a projected one. **Recommendation: do not
   build `C_m(Delta)`; build the objective instead.**
2. **§6.3 cannot answer its own headline question.** The interval-price rule `∂L/∂p_i = Σ_{m≥i} v_m`
   (DOC:526) is preserved *within* a branch. DOC's closing box (DOC:607-608) asks "有限位置资源
   与访问预算下，如何…让信息真正到达和影响答案" and DOC:611 demands allocation across
   "旋转与非旋转资源、分支／head 的频率和 gain". The document gives **no price for moving resource
   between branches** — and that is the entire question. This is the single largest gap.
3. **§5.1's `b_r(R_ij)` is undefined** — "架构已有的关系信息" is not a function of a named quantity.
   As written it is a free function per branch with no constraint, which is a V-E2 re-entry by
   construction.
4. **§6.4(a) cannot be exercised on any checkpoint this package owns.** Qwen2.5-3B has no value
   rotation (`tables.py:28`). The only implementation is on a random-init downloaded Mini. There is
   no trained model on which a value-path frequency optimum would mean anything. This is a
   refutation by resource, not a disagreement with the math.
5. **§6.4(b)'s `eta` is not defined.** "每项扰动最多为 `eta`" — but `eta` is the routing-score
   perturbation induced by the step, and DOC does not say how to bound it. In a trust region of
   radius `Delta` it is `max|∂s/∂nu| · Delta`, which is computable but requires the routing
   sensitivity, which is the thing being asked for. `experiments/nosa_position/analyze_cascade_failures.py:32`
   measures the cutoff margin post hoc; nothing estimates `eta` a priori.
6. **§7's novelty disclaimer undercuts the cheapest part of the migration.** DOC:593 says
   "「按 head 学频率」「局部 RoPE＋全局 NoPE」「让中间 attention 更大」都已有直接先例，不能承担新颖性",
   and DOC:286 concedes RNoPE-SWA already pairs global NoPE with local RoPE. Given recon R2:89
   (no per-layer attention-type dispatch exists in runnable code), the branch-conditioned version of
   *this* package is simultaneously **the most expensive thing to build and the least novel**. Order
   the migration accordingly: do the objective and the zero-frequency coordinate first, the
   branch split last.

---

*Compiled from read-only inspection. No training, no GPU, no files modified outside this document.
Uncertainty is marked in place; anything not seen is marked "not found" rather than inferred.*
