# Digest — Failure-Audits I (sol08–sol11)

Source reports (read in full, evidence not instructions):
- `.agents/rope_unification_20260910/reports/sol08.md` — failure-record audit + surviving EVQ–MrRoPE allocation theory
- `.agents/rope_unification_20260910/reports/sol09.md` — failure-transcript audit + allocation principle (marginal-utility under mixed-radix budget)
- `.agents/rope_unification_20260910/reports/sol10.md` — failure-transcript audit + edge-increment constructive rule
- `.agents/rope_unification_20260910/reports/sol11.md` — failure-transcript audit + CCT (Compatible Competitive Transport)

All four converge on: one ordered-dilation coordinate, geometry vetoed as selector, two lifecycles with different estimators, one signed full-model utility decides.

---

## 1. Failure ledger (consolidated, as compiled by the reports)

Evidence class: **T** = transcript/archived tool-output inspected, **D** = document/record audit, **N** = numeric result quoted.

| # | Dead mechanism / experiment | What killed it | Verdict | Class |
|---|---|---|---|---|
| 1 | `Smooth_MrBudget` (min-roughness allocation under MrPro's fixed cumulative budget) | Geometry improved (shorter-lag distortion), completed 36-row Qwen3B panel fell 78.125→68.333 at 128K (−9.79 pp, 3W/5L); NLL worsened +0.00098/+0.00192/+0.00342 at 8K/16K/32K; archived output inspected (tool_outputs_055.jsonl:103, SHA 62570a5b…) | Smoother field can damage evidence competition; roughness is load-bearing | N,T (sol10,sol11); sol08 notes it was outside its own corpus ("consistent, not independently verifiable") |
| 2 | HighGapToLong / "arbitrary EVQ borrowing" (move high-gap budget into long band) | Lost −17.08 pp @32K, −10.76 pp @128K, 0/36 improvements; middle-recipient control also lost (70.14/67.36 vs 87.22/78.13, 0W/7L, tool_outputs_056.jsonl:130) | "Borrow from high, give to low" is not a rule without donor/recipient justified by checkpoint's signed computation | N,T |
| 3 | LongBridge (group phase shift of four long-period slots) | Slower: −6.67 pp @32K, +1.94 pp @128K (VT-driven; = panel LBS 80.6/80.1); Faster not symmetric | Tests phase origin/direction, NOT intra-group resolution distribution; group shift ≠ allocation | N,T (tool_outputs_056.jsonl:30) |
| 4 | P2 / FullLagP2 as universal rule | Qwen-3B: 32K 72.92 vs matched-gain 98.33; 128K 81.67 (+3.54 vs official MrPro) but multikey fell; MK2 floor at 128K, mixed transfer signs; old p2's spikes were 2,048-rounded lag-centroid numerical alias | Conditional Pareto crossing on reused dev panel; direction hypothesis only; scale non-stationary (2× result ≠ 4×/8× law) | N,T,D |
| 5 | E1 single-slot additivity (pair28_29) | s28_less +5.21 pp dev (2W/0L; reverse-matched perturbation flat), but two-slot combination −4.17 pp @128K; continuous cross-cell margins close to additive → threshold crossing suffices | Per-slot scores must not be summed; compose only through jointly evaluated candidate; optimize continuous margins, validate accuracy | N,T |
| 6 | Cosine-only collision / full-subspace rank/logdet / any monotone unsigned geometry statistic | Frozen 50M crossing: static rank rises 4.57→12.54 while PPL collapses (7.14→76.20 Geo/EVQ cell); C2 reconstructed movement MAE 0.001223 yet failed Native operating point | Static rank/geometry selection directly rejected; geometry = regularizer/diagnostic, never utility | N,D |
| 7 | Attention-measure `kappa` selector | First-order and finite-swap rankings disagree; preregistered Branch C triggered | Vetoed as selector | D |
| 8 | LeRoPE oracle `rho ∝ w^(1/3)` | Lies beyond EVQ away from LeRoPE (alpha=−0.957) under structural softmax curvature | Requires signed trajectory-aware risk; dead as profile law | D |
| 9 | Universal density from assumed distance prior | Additive per-frequency utility collapses all channels to same frequency unless interference added; rankings swap with kernel choice | No kernel-free universal density | D |
| 10 | Arcsine claim (scratch note) | Kernel-to-potential argument invalid; numerical shape not U-shaped | Dead | D,N |
| 11 | Exact universal short no-harm from any changed static table | Only the Native frequency multiset (up to aliases) permits it; approximate no-harm is a risk constraint | Narrow-bridge permission only; endpoint invariants as constraints | D |
| 12 | Emulation rank bounds → re-adaptation inference | Corrected notes: needs loss-local model + joint Q/K budget | Dead inference | D |
| 13 | Slot exchangeability / unordered spectrum | Same multiset permuted across learned slots: OLMo NLL 3.10423→6.86493, Qwen core-4 0.70→0; OLMo log-s4 interior permutation +3.760692 PG-19 NLL | Ordered slot identity is part of the model; density-only theories falsified | N,D |
| 14 | Local Taylor / quadratic-Fisher at long horizon | Extreme Taylor errors slot 19; bounded sinusoidal operator invalidates unconstrained quadratic growth | Use exact finite-path replay / endpoints | T,D |
| 15 | Unsigned score proxy | Layer-0 raw score MSE 1.305e−5 but row-centered 0.48446, attention KL large | Softmax ignores row constants; signed competition only | N,T |
| 16 | NLL / reconstruction as capability endpoint | Operator family: KD NLL 5.30→3.94, score fit →9.99; both retrieval answers wrong; layer20 answer mass 0.50075→0.20339 (attn/val NMSE 0.18309/0.23950), layer26 0.04830→0.0000485; LoRA 16K low PPL + failed answer binding | NLL valid only as likelihood endpoint; strict generation + EOS required | N,T |
| 17 | Operator-family premature closure | First comparison's "release the GPU" reversed: attention/value damage, calibration mismatch; native answer deterministic and correct | Failed objective ≠ failed method class (negative-data discipline) | T |
| 18 | Calibration-distribution mismatch | 2K natural-text rephased states replayed for 8K retrieval-among-hundreds target | Rephasing does not create long-context hidden states/distractor competition | T |
| 19 | E8 answer-mass proxy | Larger predicted local attention gain than E1 yet −13.89 pp @128K | Moving mass to answer position ≠ content computation improvement | N,T |
| 20 | Frozen/scratch conflation; curve transplant | Weights×table crossing 3.400/3.251 NLL (two seeds); fixed-s4 overlays favor Cosh, target-s8 reverse; BM wins OLMo 16K, loses Qwen3B/7B (Qwen7B BM 71.11 vs MrPro 84.44; matched-gain +4.44 @32K, −7.29 @128K) | Two regimes = two optimizations; utility is checkpoint- and scale-conditional | N,T,D |
| 21 | Support ignored | Same three seeds favor Cosh at fixed support, reverse on retargeted support (+0.060/+0.227/+0.460) | (a,R) and z interact; fixed-support ordering does not transfer | N,D |
| 22 | Gain bundled with table | 4-cell: MrPro 87.22/78.13; BM 91.67/70.83; MrPro-g074 98.33/75.35; BM-g074 100/70; OLMo freq-only 0.400/0.115 vs joint 0.5825/0.400 @8K/16K | Gain is a separate operator coordinate; attribution requires the gain cross | N,T |
| 23 | Assay floors / weak baselines | QuALITY n=200 gains vanished at n=2,086; Quest32/strong baselines erased apparent gains; carrier lowered background objective, degraded generation; phase-isotropy screen UNRESOLVED | Floor means assay cannot rank; closest strong baseline first | T,D |
| 24 | "Unseen arc" GLM theory (slots 36–39 enter new regime at long context) | Those slots already exceed a full turn inside the Native window; FWE "evidence distance" inferred from repeated answer tokens | Derive phase regimes from actual native table + source positions | T |
| 25 | Governance/ops failures (invented 30.66/90 GiB storage gate; scope inflation; GPU-utilization-as-objective; eval data errors: Proof-Pile filename, header newline 4/8→0/8, seed not shuffling QA order, steam-engine sample clustering; partial 59/128 LoRA run promoted; estimand substitution incl. dense-KV-for-sparse) | Corrected in-record by user/audit | Process constraints: separate proposed/implemented/launched/tested; inspect source/prompt/scoring at first discriminating sample | T,D |

## 2. Causal explanations that survive cross-checking

- **Smooth 68.3-far vs s28_less 83.3-far (same MrPro budget simplex):** the far outcome depends on *placement* of budget at the checkpoint's learned-sensitivity knee (slots ~28/29), not on any global field property. Smooth's global roughness-minimization necessarily flattens the structure s28_less exploits — i.e., transition-band "roughness" is functional. This matches sol10's conditional statement ("if utilities have a knee near slots 28/29, budget concentrates there") and the EVQ ridge note (narrow changes charged LESS — mathematical permission for narrow bridges; s28_less is the empirical instance: −0.02 near cost for +5.2 far gain).
- **Near/far Pareto frontier is real and pay-able in both directions:** P2 pays −14.3 near for +3.5 far; LBS (LongBridge Slower) pays −6.7 near for +1.9 far; s28_less pays ~0 near for +5.2 far (dominant on this panel); HighGapToLong pays on both axes (dead). A surviving allocation must sit on the frontier's favorable side; no mechanism found so far escapes the trade-off except s28_less.
- **Co-adaptation is the mathematical reason for lifecycle bifurcation** (weights×table crossing, 50M rank/PPL crossing): scratch allocation can be learned (hypergradient's weight-response term indispensable), frozen allocation must pay a compatibility cost to already-slot-assigned weights.
- **Signed competition structure survives:** distractor replication multiplies the exponential sum and drops margin G by exactly log q (conditional identity); row-constant invariance of softmax kills all unsigned score proxies; V-transport and downstream-layer damage appear even when the QK path looks fine.
- **Slot identity + ordering + endpoint conventions are design constraints, not tunables** (three independent permutation-collapse numerics).

## 3. Surviving EVQ–MrRoPE allocation theory (exact statements + tags)

1. Common physical object: ordered log-frequency table `x_k = −log ω_k = a + R·z_k`, equivalently dilation field `ν_j = ω_j s^{−m_j}`, `0≤m_j≤1`, with fixed slot labels, gain `g`, checkpoint weights, and lifecycle. — **[已验证]** (matched-training causality of z; permutation and crossing collapses as verified counterexamples to alternatives)
2. EVQ (Cosh/EVQ density) and MrRoPE-Pro are *alternative structural priors/demand measures* over the same finite ordered allocation; their published rules do NOT form a closed optimal behavioral family; no assigned source proves MrRoPE mixed-radix optimality. — **[部分证据]** (family table classification + negative coverage statements; promotion explicitly gated on 7 counterexample checks)
3. Cosh density uniquely minimizes its declared convex surrogate; full-RoPE static identity r₂(R)=2K/(1+(K−1)c̄) and low-band collapse toward span{1,Δ} are exact. Neither orders LM behavior; Fourier-comb, length-reversal, and aliasing counterexamples are explicit. Root non-ranking vetoes geometry ranking. — **[已验证]** as math; **[vetoed]** as behavioral selector.
4. Scratch selection = bilevel trained risk with hypergradient `R_x − R_W H_WW⁻¹ H_Wx`; frozen selection = compatibility-constrained signed long-risk movement (`½hᵀF_N h ≤ ε` trust region; closed form `h* = −√(2ε/(g_LᵀF_N⁻¹g_L))·F_N⁻¹g_L` when unconstrained by order bounds). — **[已验证]** equations (conditional KKT/hypergradient exact); the programs' predictive success **[假设]**.
5. Signed finite-path marginal utility `U_i = −∫ ∂J/∂b_i dt` (or edge form `∂L/∂ε_r = Σ_{j≥r} ∂L/∂m_j`) as the selector is the proposed constructive unification. — **[部分证据]** (E1/s28_less dev-panel + flat reverse-matched control; P2 Pareto signal) / **[假设]** as universal predictor; not yet validated on independent confirmation rows.
6. Conserved-budget identity `B = Σ_q m_q = N − Σ_i i·p_i` (p_i = edge increments, simplex): budget total and center-of-mass are coupled — Smooth controlled B and still failed ⇒ placement weighting by learned computation is the open term. — **[已验证]** identity; **[vetoed]** budget-conservation-plus-smoothness as sufficient rule.
7. η-mixture prior `q_η=(1−η)q_E+ηq_M` + midpoint-quantile initialization `z_k⁽⁰⁾=F⁻¹((k+½)/K)`. — **[假设]** (initialization only; η must be predeclared, never tuned on the final benchmark).

## 4. Constructive next rules (recipe form), per report

- **sol08 (convex local program):** start from verified Native `x^N`, `h=x−x^N`; estimate on frozen predeclared rows: signed long-risk gradient `g_L`, full-model Native-divergence curvature `F_N`, optional PSD `H_L`; solve `min g_Lᵀh + ½hᵀH_Lh + prior prox terms s.t. ½hᵀF_Nh ≤ ε, ordering`; order constraints → cone-KKT projection; acceptance = final nonlinear 32K/128K endpoints.
- **sol09 (conservative marginal transfer):** start at exact MrPro table at fixed declared gain; per admissible transfer `δb = η(e_p−e_q)` on the log-radix simplex, score paired correct-vs-competitor margin minus registered short-context penalty via finite full-model evaluation (not Taylor); move budget only when the same signed direction holds across task strata AND Native retention holds; stop at no positive lower-confidence utility; freeze once; next single candidate = best-supported transfer (slot-28/E1 clue must be rechecked on the exact Qwen-3B margin panel).
- **sol10 (edge water-filling):** parameterize transition by `ε_r≥0, Σε=1` with fast/slow anchors fixed (`m=0`/`m=1`); full-model loss `L_F = L_answer + α·L_bind − margin target-vs-distractor + λ·L_preserve(short)`; estimate signed edge utility `u_r` + local `H` from finite differences on FULL prefill (one-query gradient may propose, cannot certify); solve simplex projection; diagonal-H KKT = water-filling `ε_r* = max{0, q_r+(u_r−ν)/h_r}`; reference q = MrPro; report continuous margins AND exact outcomes; opposite-direction magnitude-matched redistribution is the attribution control.
- **sol11 (CCT):** keep ordering/K/standard RoPE; separate declared gain `g`; basis of exactly three coordinates: {endpoint-matched EVQ−Mr direction `h_E`; FullLagP2−Mr direction `h_P2`; constant-gain coordinate}; estimate `D_N(h,g)` (next-token KL + strict short generation retention), `G_T(h,g)` (128K relevant-vs-distractor LSE margin on source-located real prefixes), `R_T`; solve `max Ĝ_T − λ_E‖h−h_E‖² − λ_M‖h‖² s.t. D̂_N ≤ ε, 0≤m_M+h≤1, p(m_M+h)≥0`; materialize ONE table+gain; if no labeled relevant sets: paired 128K prefixes with independently sampled distractor blocks + Native-teacher distillation on answer tokens (QK-only KL cache insufficient). Scratch rule: init from EVQ prior, joint/bilevel training; adaptation rule: fix table, train QKVO/readout with Native replay + answer+EOS loss, compare vs matched Native-table adapter.
- All four: do NOT launch another curve grid; one frozen low-dimensional candidate + predeclared falsifiers (sign mirror, permutation, gain cross, lifecycle cross, support-retarget, endpoint dissociation, locality radius).

## 5. Numerics extracted

- Cosh-minus-uniform NLL at 256/512/1K/2K (151.9M 3-seed, fixed support): +0.026 / −0.281 / −0.176 / −0.146; retarget reversal +0.060 / +0.227 / +0.460.
- Frozen 50M PPL cells Geo/Geo, Geo/EVQ, EVQ/Geo, EVQ/EVQ: 7.14 / 76.20 / 23.05 / 7.16; static rank 4.57→12.54 (worst cell).
- Mature OLMo same-support: 0.56% → 60.47% (ordered-z sensitivity); coarse ramp 61.04% (profile uniqueness dead).
- Weights×table crossing NLL 3.400 / 3.251 (two seeds).
- Panel-consistent Qwen-3B 32K/128K: MrPro 87.22/78.13; Smooth 87.2/68.333 (−9.79 pp, NLL +0.00098/+0.00192/+0.00342 @8K/16K/32K); s28_less +5.21 pp dev (=83.3); pair28_29 −4.17 pp; P2 72.92/81.67 (+3.54 vs official MrPro; 72.92 vs 98.33 matched-gain); LongBridge-Slower −6.67/+1.94 (=80.55/80.07); HighGapToLong −17.08/−10.76 (0/36), middle-recipient 70.14/67.36 (0W/7L).
- Gain 4-cell: MrPro 87.22/78.13; BM 91.67/70.83; MrPro-g074 98.33/75.35; BM-g074 100/70; BM independent transfer +4.44 @32K / −7.29 @128K matched gain.
- Permutation collapses: OLMo NLL 3.10423→6.86493, Qwen core-4 0.70→0; OLMo log-s4 interior multiset +3.760692 PG-19 NLL.
- Proxies: C2 movement MAE 0.001223 (still failed Native op point); score MSE 1.305e−5 raw vs 0.48446 row-centered; operator NLLs init 5.30 / KD 3.94 / score-fit 9.99, answer mass layer20 0.50075→0.20339, layer26 0.04830→0.0000485, attn/val NMSE 0.18309/0.23950; E8 −13.89 pp.
- 454M scratch: passkey 8K EVQ 100±0% vs Geo 61±3%; 16K PPL 107.5 vs 157.7 (shared fixed-index scaler only).
- QuALITY: apparent gains at n=200 shrank at n=2,086. Qwen7B dev-subset BM 71.11 vs MrPro 84.44.

## 6. Conflicts with established panel facts

**No numeric conflicts found.** All report numerics reconcile with the panel: E3 98.3/75.3 ≡ sol11's "MrPro-g074" (gain-matched cell); LBS 80.6/80.1 ≡ LongBridge-Slower (87.22−6.67, 78.13+1.94); Smooth/P2/s28_less/HighGapToLong deltas reproduce exactly. Flags:
- **MrUni 64.6 is absent** from all four audits — no explanation offered; the digest must not treat it as covered. (The middle-recipient control 70.14/67.36 is a HighGapToLong control, NOT MrUni.)
- sol08's hedge ("Smooth not independently verifiable from my corpus") vs sol10/sol11's direct archive inspection — resolution: treat Smooth −9.79 pp as T+N class (two independent inspections), sol08 simply lacked coverage.
- Reports consistently downgrade E1/s28_less to "development signal, not law" and demand recheck on the margin panel — do not carry 83.3 as a confirmation, only as the best current direction hypothesis.
- sol11 notes the long-recipient 0/36 output was NOT separately recoverable by name (dialogue-level retention only) — evidence-class caveat on the HighGapToLong number.

## 7. KKT relevance — negative data for candidate terms of F

Ruled OUT from the objective/selector (must not appear as a term F is optimized over or ranked by):
- collision / cos-only kernel and any root-based ranking (**[vetoed]**, root non-ranking); effective rank / logdet / subspace-span statistics (rank-PPL crossing); roughness or smoothness of m or x (Smooth −9.79); transport / movement MAE / short-lag distortion (C2); phase-isotropy profiles; unsigned density entropy; kappa attention measure; LeRoPE ρ∝w^(1/3); arcsine; any universal per-frequency utility from a distance prior without interference; emulation-rank bounds; raw (non-row-centered) score MSE, QK norms, Fisher magnitudes, unsigned Gram statistics (sol08: band derivative contains the signed task adjoint — norms cannot replace it); answer-mass/local attention gain (E8); teacher-forced NLL/KL as capability term.
- Structural exclusions: unordered-spectrum terms (any F not indexed by slot); pooled-regime objectives (scratch+frozen in one F); gain entangled with table (g must be its own coordinate/axis in the cross); Taylor/quadratic terms valid beyond phase radius; per-slot additive composition (sum of single-slot scores ≠ joint candidate value — pair28_29).

Surviving as CONSTRAINTS of the KKT program (positive structure the failures evidence):
- Simplex: `p_i ≥ 0`, `Σp_i = 1` (⇒ water-filling KKT, one-sided inequalities at bounds); box `0 ≤ m_j ≤ 1`; ordered positive deployed frequencies (cone-KKT projection, sol08).
- Endpoint/support anchoring: `m=0` fast band, `m=1` slow band (YaRN/MrRoPE scaffold — three-band rationale "sound but limited", exact thresholds empirical not theorem).
- Native-compatibility trust region: `½hᵀF_N h ≤ ε` (the bundling and co-adaptation failures are precisely why retention must be a constraint, not a penalty that can be traded away).
- Signedness: only adjoint-weighted (task-signed) gradients enter the linear term.
- The budget identity B = N − Σ i·p_i shows conservation is necessary but insufficient — F's value must weight p-placement by checkpoint-learned sensitivity (the knee).

---

## Lineage verdict (5 lines) — mechanisms ALREADY DEAD; next derivation must not re-propose them

1. **Geometry-as-selector, all variants:** cos-only collision / root ranking (vetoed), effective rank / logdet, smoothness/roughness minimum (Smooth_MrBudget, −9.79 far), movement MAE / short-lag distortion, phase isotropy, spectral entropy — dead as selectors, admissible only as regularizers.
2. **Unsigned / unlocalized statistics:** raw score MSE, QK norms, Fisher magnitude, Grams, kappa rule, LeRoPE ρ∝w^(1/3), arcsine, distance-prior universal densities, answer-mass proxy (E8), emulation-rank bounds.
3. **Curve transplants across the wrong axis:** unordered-spectrum or slot-exchangeable tables (permutation collapses), scratch→frozen or frozen→scratch single-objective, cross-model/cross-gain table transplants (BM), fixed-support→retargeted-support extrapolation, 2×→4×/8× scale stationarity, any "borrow high-give-low" rule (HighGapToLong) or group phase shift sold as allocation (LongBridge).
4. **Proxy endpoints:** teacher-forced NLL / KL / reconstruction as capability, local Taylor/quadratic beyond phase radius, per-slot score summation (pair28_29), gain-bundled attribution, weak-baseline or floored-assay wins.
5. **The surviving frame is fixed by these deaths:** one ordered simplex `p_i≥0, Σp=1` (equivalently `m_j`, endpoints pinned), slot-labeled, gain as separate coordinate, two lifecycles with different estimators, selection by one signed full-model target-vs-distractor margin movement from the MrPro prior under a Native-retention trust region — narrow, knee-directed (s28/P2 direction hypotheses), validated only at complete 128K generation + retention endpoints.
