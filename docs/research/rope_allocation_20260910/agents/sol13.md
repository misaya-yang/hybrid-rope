# Sol13 — Failure ledger and a constructive allocation principle

## Scope and evidential stance

This report audits the assigned dialogue shard and project records as failure evidence. Historical agent verdicts are not active instructions and their universal claims are not inherited. In particular, the old conclusion that *every* low-dimensional allocation law is non-identifiable is too strong: the cited counterexamples refute specific unsigned or permutation-invariant summaries, while a labeled, signed computation-level objective is a different object. No GPU job, new model run, or runtime/paper edit was performed.

## Non-redundant failure ledger

| Failure | What was proposed | What was actually implemented/tested | What the evidence establishes | Constructive constraint |
|---|---|---|---|---|
| Proxy-to-capability leap | Choose tables from coverage, smoothness, roughness, collision, response energy, or curve fit. | C2 achieved movement MAE `0.001223` yet missed OLMo and Qwen native-retention gates; exact-orbit count changed `6→64` under an ULP perturbation with indistinguishable model behavior; carrier objectives improved while retrieval collapsed. | These particular unsigned geometry summaries do not order downstream behavior. They remain diagnostics or feasibility terms. | The objective must contain the signed desired-source versus hard-distractor contrast, with slot labels preserved. |
| Slot exchangeability error | Treat the frequency multiset or density as the physical object. | A same-multiset slot permutation changed OLMo NLL `3.10423→6.86493` and Qwen 64K core-4 `0.70→0`. | Frozen Q/K coordinates are labeled and co-adapted; sorting or permuting a proposed table can destroy the intervention. | Optimize labeled `ν_j`; never repair monotonicity by sorting after optimization. |
| Local derivative promoted to finite intervention | Use a Jacobian/Taylor score to choose a full S=4 table. | Finite changes accumulated many phase turns; recorded linear predictions had 71–468% relative error, while second-order examples can be orders of magnitude wrong. | A derivative is trustworthy only inside a certified finite neighborhood. | Every local step needs an explicit exact-phase remainder bound and exact trigonometric reevaluation before acceptance. |
| Independent-channel additivity | Score each rotary pair independently and sum benefits. | The shared-frequency derivative contains cross-key signed coherence and shared-head/W_O cancellation; identical per-key nonnegative energies can yield total derivatives 0 versus 4. | Per-slot energies discard the terms that decide the actual shared-table response. | Retain the joint coefficient mean and full covariance/Gram structure for the deployed shared table. |
| Smooth transition as mechanism | BM/Smooth curves reduce boundary jumps or multiple distortion measures. | BM won strongly on OLMo but lost at Qwen 128K; the newer Smooth_MrBudget improves several audited distortions while worsening long-task outcomes. | Smoothness can be a numerical constraint, not a performance principle. Conditional success also rules out declaring the whole family useless. | A bridge may be sharp if signed margins require it; smoothness is used only for solver stability or an independently justified engineering limit. |
| Exact collision minimization transferred to deployment | Treat the finite-window EVQ kernel minimizer as the best frozen table. | Astra02 proves the exact finite-window measure optimum is a unique finite atomic equilibrium, not a positive Cosh density; its lower kernel energy has no demonstrated task-ordering consequence. | Exact EVQ and Cosh solve different problems. Atomic collision optimality does not choose a frozen checkpoint table. | Use exact-kernel equilibria only as scratch-design lower bounds/initializers under a declared generative model, never as a frozen-task certificate. |
| MrRoPE first-zero story overextended | Infer its progressive quadratic middle schedule from long-range coherent cosine positivity. | The paper assumes arithmetic radix increments and selects boundaries empirically; the first-zero story does not derive the exact interior profile. | MrPro is a strong structured baseline and feasible face, not a universal optimizer. | Preserve its evidenced Qwen face initially (`j≤23` native, `j≥40` `/4`) while letting the 16 interior labeled displacements be decided by useful computations. |
| P2 residual shape mistaken for portable cause | Project FullLagP2−MrPro onto an antisymmetric cubic or copy the dominant symmetric component. | Even/odd audit found 84.26% symmetric energy and only 13.18% explained by the cubic; P2 itself wins conditionally and reverses across checkpoints/tasks. | Shape energy says what differs, not which difference caused gains. | P2 and E1 are empirical directional controls, not targets to regress toward. Compare a derived direction with them after derivation. |
| Native retention inferred from long mean | Aggregate long-task gain was treated as sufficient. | Short/native QA and retrieval can regress even when macro score rises; task rows can be at floor/ceiling and EOS failures alter interpretation. | Long and native roles are separate constraints, and a positive pairwise margin can still be diluted by many distractors. | Carry explicit native-role thresholds plus a distractor-count-dependent softmax margin requirement. |
| Oracle/late masking overclaim | Use a recovered answer under oracle pruning as proof of the failure mechanism. | Removing background while retaining positions restored some cases, but late KV pruning did not fully restore them; the intervention changes competition and/or upstream states. | It narrows hypotheses but does not isolate PE, prefill, reader, or value/readout causally. | Qualify operations by source intervention and preserve the full prefix when estimating deployment envelopes. |
| Training and frozen deployment conflated | Apply one geometry law to scratch training, LoRA, and cold table replacement. | Weight×table crossings and permutation collapse show frozen compatibility; scratch/support-retargeted rankings can reverse; causal source use can improve without successful readout. | Co-adaptation changes the statistical object. | Use separate objective semantics: population allocation under exchangeable/co-adaptive channels for scratch; labeled finite transport for frozen weights. |
| Workflow substituted for research | Arbitrary timeouts, repeated hashes, broad grids, tiny milestones, and handoffs were added or treated as completion. | The dialogue contains repeated user corrections: five minutes was an iteration estimate, not a kill switch; ten candidates was a ceiling, not a quota; preparation/reporting was not the requested outcome. | These are process failures, not scientific evidence. | Run the least costly discriminating comparison, reuse baselines, and let results change the next decision. |

The strongest source-level formulation of the scientific error is already explicit: statistics-to-intervention lacked a mechanism, endpoints failed to distinguish explanations, and changing candidate names did not reduce the unknown (`docs/research/ROPE_RESEARCH_FAILURE_REVIEW_20260907.md:32-46`). The same record derives the exact labeled log-gap relation and warns that sorting changes slot identity (`:79-98`), then exhibits the missing coherent cross terms (`:127-146`). Those are the durable constraints; the archived bans on all new tables are not.

## Strongest constructive principle

**Allocate finite positional channels to maximize the worst role-conditioned useful-source margin relative to coherent interference, while preserving certified native margins; treat channel coefficients as co-adaptable in scratch training and as labeled, uncertain objects in frozen deployment.**

For rotary pair `j`, signed lag `d`, and pre-RoPE vectors, write

\[
q_j^T R(\nu_jd)k_j=A_j\cos(\nu_jd)+B_j\sin(\nu_jd).
\]

For operation cell `r` (query, correct source `+`, matched hard distractor `−`, role, layer/head, and their actual lags), stack coefficients `c_r` and exact phase features `f_r(x)` with `ν_j=ω_j e^{-x_j}` so the contrast is

\[
Z_r(x)=c_r^T f_r(x).
\]

Under an explicitly estimated transport envelope

\[
\|\mathbb E c_r-\mu_r\|\le\epsilon_r,\qquad
\operatorname{Cov}(c_r)\preceq L_rL_r^T,
\]

define

\[
z_r(x)=\frac{\mu_r^Tf_r(x)-\epsilon_r\|f_r(x)\|-\eta_r}
{\|L_r^Tf_r(x)\|},
\qquad
\eta_r=\log\frac{(M_r-1)(1-a_r)}{a_r}.
\]

Here `η_r` is the pairwise margin sufficient for source attention mass at least `1-a_r` among `M_r` keys. The finite frozen allocation rule is

\[
\boxed{\max_{x\in\mathcal F}\min_{r\in R_L}z_r(x)
\quad\text{s.t.}\quad z_r(x)\ge \bar z_r\;(r\in R_N).}
\]

`F` preserves slot labels, fixed endpoints/declared bounds, and frequency ordering only if ordering is part of the chosen deployment face. For the first Qwen 32K→128K calculation, the best evidenced face is the MrPro face with sixteen interior variables, not because 23/40 is universal but because it is the smallest complete departure from a strong baseline.

This is not a generic gradient QP. The optimization target is a semantic, signed source contrast; covariance retains cross-slot/cross-head coherent interference; native behavior is a hard role constraint; and finite steps are certified. At reference `x0`, use `f≈f0+Jv` only within a box and bound each pair's second derivative by `sqrt(u^2+u^4)`. The conic inner constraint from Astra03 then lower-bounds the exact mean and upper-bounds the exact standard deviation, so bisection in target reliability produces a finite candidate with an incumbent guarantee. Recompute exact sine/cosine margins afterward. If no positive feasible step exists, the correct output is incompatibility or a missing envelope, not another curve.

### Counterexample checks

1. **EVQ versus useful content.** At lag `2π`, frequencies `(1,1/2,1/4)` have zero aggregate cosine and win the exact point-lag EVQ square, while `(1,.9,1/4)` has aggregate `1.809`; for a matched remote key the latter has lower Gaussian pairwise error, whereas a zero-lag positional discrimination favors the former. Therefore neither EVQ energy nor MrRoPE coherent positivity is universally correct; the role label changes the desired sign.
2. **Actual Smooth/Mr slot.** Astra03's Qwen slot-28 construction at native lag `26590` and target lag `106360` gives expected target response `+0.99946` for MrPro and `−0.99954` for Smooth despite Smooth's better unsigned metrics. The proposed rule preserves this reversal because it keeps the coefficient phase and sign.
3. **Isotropic-noise degeneracy.** If coefficient means vanish, reduced variance cannot certify a positive margin. If independent isotropic noise is the whole nuisance model, changing frequency does not change projected variance at fixed lag. Any claimed collision benefit then needs coherent cross-slot or lag-conditioned covariance.
4. **Many distractors.** A positive pairwise mean is insufficient as `M` grows; `η_r` increases logarithmically with distractor count. This directly addresses the observed background-length failures without declaring background dilution the proven sole cause.
5. **Upstream drift.** If the new table changes `c_r` outside its envelope, the guarantee is void. Detached retiming is only a conditional diagnostic; actual long forwards or a validated drift radius are required.

## Scratch versus frozen interpretation

For scratch training, EVQ-Cosh can be recovered under the additional generative assumption that channels are exchangeable/co-adaptive, useful normalized signal is allocation-independent, and nuisance covariance equals an independent loading term plus a nested slow-tail term. Maximizing signal-to-noise then minimizes

\[
\alpha\int\rho^2+\beta\int S_\rho(t)^2dt,
\]

yielding Cosh. This is a conditional covariance model, not a consequence of the exact finite-window kernel. Astra02's atomic-minimizer theorem is the corrective caveat: without an explicit finite-resolution or density penalty, the exact kernel's optimum is finite atomic, and the continuous `αI` Hilbert–Schmidt projection is not literal in infinite-dimensional `L2`.

For frozen deployment, useful signal is `μ_r^T f_r(x)`, coefficients are labeled, and covariance/transport drift must be estimated. MrRoPE supplies a successful structured feasible family, while the role-conditioned max-min rule decides which middle-band budget placement is supported by the checkpoint's computations. This is the concrete unification: the same reliability ratio under different assumptions, not one universal curve.

## Decision-sufficient next calculation

Before generating a new table, take the same qualified source-dependent rows and compute signed mean margin and margin variance under the already existing MrPro, Smooth, P2, and E1 tables. Qualification requires a correct compact answer plus a source deletion/content fork; distractors must be content-confusable, and cells retain layer/head/relation/lag labels. Use held-out source groups for validation.

- If the signed statistic ranks MrPro above Smooth on Smooth's failed rows and preserves the known P2/E1 conditional positives, fit the stated finite conic step on the MrPro face and freeze exactly one table for parent validation.
- If it still ranks Smooth above MrPro, the theory has failed at that operation level. Investigate value/readout, prefix-induced upstream drift, or incorrect source labels; do not add another unsigned penalty.
- If native and long constraints are infeasible, report the incompatible roles. That is a scientifically useful obstruction and directly explains why a single cold table may require routing or adaptation.

## Claim boundary

The derivation guarantees only role-conditioned attention ranking/mass under the declared coefficient envelope. With Gaussian margins it gives exact pairwise error `Φ(-z)`; with a sub-Gaussian envelope it gives `exp(-z²/2)`, and with moments alone Cantelli gives `1/(1+z²)`. Full-generation correctness additionally requires the routed value and downstream decoder to preserve the intended output across all required prefix events. Existing source-deletion evidence shows why that extra premise cannot be assumed. No task-success theorem, Qwen winner, or universal Cosh/MrPro optimality claim is made.
