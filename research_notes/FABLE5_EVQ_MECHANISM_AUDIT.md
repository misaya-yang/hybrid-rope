# FABLE5 EVQ Mechanism Audit

Date: 2026-08-06
Auditor scope: does the repository support the hypothesized mechanism chain

> frequency allocation → target/distractor QK competition → attention-weighted V written into the residual stream → propagation/correction across later layers → final answer-token logits and generation

Audit ground rules: primary code, raw JSON/tensor artifacts, manifests, and git history outrank narrative summaries. Every repository-specific claim below carries a file path; tracked-vs-local status is stated where it matters. Repo state at audit: branch `main_0726`, HEAD `515eb36`. Nothing in the repository was modified except the creation of this report.

**Overall verdict (one paragraph).** The repository supports a *two-stage* mechanism, not the single-stage chain as hypothesized. Stage 1 (supported, narrowly): training-time co-adaptation to the EVQ frequency grid produces long-range target addressing in pre-softmax QK scores and *causally used* remote V-information that measurably reaches the final answer logit (verified at 16K on matched Llama-3-8B LoRA arms, single seed, 10 cases). Stage 2 (contradicted as automatic): that upstream advantage does **not** by itself produce generation — exact match stays 0% at 16K while the gold token sits at rank ≈2,000, and forcing oracle attention onto the gold block changes almost nothing. Generation capability appears only when training exploits the substrate (OLMo-2-1B matched routing: Native 0/100 vs EVQ 69/100 and 67/100 at 8K; 750M matched continuation: 0% vs 77.5% AR at 8K), and it collapses at 4× (32K) everywhere. Two repository-internal results additionally undercut the "pure geometry → separability" reading: the QK/NLL advantage travels with the *trained adapter*, not the runtime frequency tensor, and the repo's own channel-ablation diagnostics point to a complementary mechanism — reduction of *destructive out-of-distribution phase interference* — rather than target/distractor separability per se.

---

## 1. Canonical implementation and trustworthy experiments

### 1.1 Canonical EVQ implementation

| Item | Location | Notes |
| --- | --- | --- |
| Schedule construction | `scripts/lib/rope/schedules.py` — `evq_cosh_phi` (lines 94–121), `evq_cosh_inv_freq` (124–140), `build_inv_freq` (147–253) | `phi_k(tau) = 1 − (1/τ)·asinh((1−u_k)·sinh τ)`, midpoint grid `u_k=(k+0.5)/K`, `inv_freq = base^(−phi)`. τ=0 recovers the midpoint-discretized geometric grid (which is *not* bit-identical to standard endpoint RoPE — see §3, confounder C7). Last touched in commit `2734d12` (2026-04-29). |
| Frequency identity contract | `rebuttal/rebuttal_0723/theory_results/FREQUENCY_DEFINITION_MANIFEST.json`, `rebuttal/rebuttal_0723/experiments/geo_rope_contract.py` | Hash discipline for realized tensors. |
| Realized-tensor hashes in key runs | e.g. `rebuttal/rebuttal_0723/theory_results/OLMO2_1B_4K_ONLY_ROUTING_CONVERSION_20260726.md` (Native `inv_freq` `dde15c31…`, EVQ `917a5242…`) | The intended sole scientific difference between matched arms. |

Directionality (verified numerically against `evq_cosh_phi`, K=32, head_dim=64, base=5×10⁵, τ=1.414): EVQ **compresses the ultra-long-period tail and densifies in-range frequencies**. Channels with rotation period ≤2048 tokens: 14 (geometric) → 18 (EVQ); ≤8192: 17 → 21. Channel k=16 moves from period ≈5,454 to ≈1,340. This matters for mechanism interpretation: EVQ buys in-window phase resolution by spending channels that geometric RoPE parks at wavelengths far beyond any usable context.

The paper's own mechanism claim stops at the *content-free kernel level*: `paper/sections/03_theory.tex` (lines 15, 32) validates the surrogate \(\mathcal{C}_{\mathrm{app}}\) functionally via 24–92% exact-kernel collision-score reduction, and `paper/appendix/a1_proofs.tex` (`\label{sec:mechanism-isolation}`, line 421) runs a collision-only oracle over 36 configurations with no language model. The paper never claims the QK→V→logits chain; that chain is entirely a post-submission investigation.

### 1.2 Trustworthy experiments (mechanism-relevant, in decreasing evidential strength)

**T1 — OLMo-2-1B matched routing conversion (strongest capability-level attribution).**
Owner: `rebuttal/rebuttal_0723/theory_results/OLMO2_1B_4K_ONLY_ROUTING_CONVERSION_20260726.md` (tracked, commit `fec0157`, 2026-07-26; status `RAW_BACKED_DUAL_COPY_FROZEN`, adapter/data/inventory SHA-256s recorded).
Protocol: `allenai/OLMo-2-0425-1B-Instruct`, identical checkpoint/rows/order/budget/rank-64 QKVO LoRA/optimizer for both arms; only `inv_freq` differs; physical training length 4K only; LM head frozen.
Result: strict first-number NIAH exact at 8K — Native **0/100**, EVQ **69/100** (seed 20260725) and **67/100** (independent EVQ training seed 20260726); McNemar p = 3.39×10⁻²¹. Beyond-training-gap subset 21/50 and 19/50; fresh all-long-gap set 49/100 and 48/100; 16K only 1/20. Boundaries in the same owner: EVQ 4K NLL is *worse* (2.548 vs 2.235); held-out UUID retrieval and variable tracking are **0%** for every EVQ arm (untouched Native: 55%/25%); EVQ injection with *no* adapter also 0%.

**T2 — Llama-3-8B seed-42 sparse-conversion + causal decomposition (the only completed link-by-link probe).**
Owners: `docs/exp/2026-07-14_lora_retrieval_conversion_probe.md` (tracked; commits `737cb31`, `d356977`, both 2026-07-14), code `experiments/lora_evq_v2/eval_sparse_conversion.py`; local raw (untracked, SHA-manifested): `results/lora_sparse_conversion_s42_20260714/{phase0_summary.json,phase0_evq.json,phase0_geo.json,rtx5090_causal_manifest.json,server_*/}`.
Setup: Meta-Llama-3-8B-Instruct + matched 300-step Geo/EVQ LoRA (QKVO, r64/α128; EVQ τ=1.414 midpoint, head_dim 128), 10 frozen true-16K passkey cases, 32 retrieval heads frozen by pooled Geo+EVQ reciprocal gold-block rank at 8K.
Key numbers (verified in raw JSON and tracked report):

| Measurement | Geo | EVQ |
| --- | ---: | ---: |
| Median target-block hit@16 over 32 heads, 16K | 18.75% | 64.06% (EVQ wins 10/10 cases) |
| Median sparse/dense answer-mass ratio @16, 16K | 0.000 | 1.033 |
| Same at 32K | hit deltas shrink to +0.03…+0.16; both mass gains 0.0 — preregistered 32K gate **failed**, `status: stop` |
| Dense answer NLL, 16K | 15.509 | 9.063 |
| Remove gold block, all heads (ΔNLL) | −0.010 (≈0) | **+1.506** (×4.51 gold-token probability) |
| Remove gold block, 32 selected heads (ΔNLL) | −0.000 | +0.527 |
| Forced-gold (oracle) inclusion vs score-sparse (ΔNLL) | −0.005 | −0.034 (near-null) |
| First-answer-token median rank | 33,774.5 | 2,043 |
| Exact match, 16K | 0% | 0% |
| Counterfactual swap: Geo adapter + EVQ runtime freq | NLL 14.134, source binding 0% | — |
| Counterfactual swap: EVQ adapter + native runtime freq | — | NLL 9.246, source binding 60% (vs 9.073 / 80% fully matched) |

Probe validity checks recorded in the owner: full-budget custom attention == dense logits (`max_abs=0`), custom KV decode == `model.generate` exactly, chat-boundary diagnostic did not change the conclusion. The probe measures genuine pre-softmax post-RoPE scores: `_probe_one` recomputes q/k projections, applies rotary, applies `module.scaling`, per layer per head, last-prompt-token query (`query_contract: last_prompt_token_predicting_first_answer_token` in `phase0_evq.json`).

**T3 — 750M matched continuation (submitted supporting; the scratch-family capability contrast).**
Owners: `paper/appendix/a2_experiment_details.tex` line 68 + `paper/tables/table6_750m_continue_supporting`; local raw (untracked): `results/core_text/phase15/phase15_geo_seed42_result.json`, `phase15_evq_r0_seed42_result.json`, `phase15_continue_summary.json`.
Both arms start from the *same* Geo-750M-2K checkpoint (`step_15258.pt`, remote path) and continue 500M tokens at 4K with 3% passkey mix; only the frequency schedule differs (EVQ τ=1.5 full replacement vs Geo). At 8K/d=0.5: Geo-continue AR exact **0.0**, EVQ-continue **0.775** (global 0.667 vs 0.925); PPL@16K 45.1 vs 24.4. Single seed 42; teacher-forced retrieval is 1.0 for *both* arms at 8K — the TF metric saturates where AR separates.

**T4 — Submitted primaries (context: what EVQ improves at the NLL/TF level).**
`data/curated/primary1_evq_yarn_10pct_raw.json`, `table2_evq_yarn_454m_passkey_10pct.json` (454M, seeds 42/123/7, PPL + TF NLL-gap passkey; the curated claim boundary itself notes raw `ar_exact_match_rate` @8K is 0.18/0.58/0.98 by seed, *not* the 100% TF number), `fig3_extreme_128.json` (125M L=128, PPL), `table18_mla_3seed_aggregate.json` (MLA, PPL). Metric definition verified in `scripts/supporting_eval/eval_passkey_scratch.py::eval_passkey_nll_gap` (gap = NLL_wrong − NLL_correct; "retrieved" = gap > 0; AR exact is a separate auxiliary).

**T5 — Channel-ablation / band diagnostics (the repo's alternative mechanism evidence).**
Owner: `rebuttal/rebuttal_0723/theory_results/EXPERIMENT_REPORT_20260724.md` §§10–11 (tracked, commit `fec0157`) + local raw `results/frequency_band_usage_s42_20260724/REPORT.md` (`DOES_NOT_SUPPORT_BAND_BRIDGE`). 151.9M scratch checkpoints, seed 42, checkpoint-only interventions:
- Pre-RoPE Q/K pair norm does **not** predict per-pair causal deletion effect (median Spearman −0.167; phase-covariance −0.045).
- Deleting the trained high-norm band *improves* 1K–8K NLL for Geo and EVQ (bootstrap CIs exclude zero).
- Jointly removing independently selected "interference" pairs improves held-out NLL: Geo −0.280 @8K (16 pairs), EVQ −0.108 @8K (18 pairs), FMRoPE-retargeted ≈0 (1 pair).
- Band swaps preserving the exact frequency multiset change NLL ≥0.05 in 22/24 cells — channel semantics are bound to specific frequencies.
Recorded conclusion (§11.2): "features learned under one frequency assignment become destructive after their phases move out of distribution. EVQ changes which channels suffer that interference, while target-aware FMR retargeting reduces it much more effectively."

**T6 — Dose-response NLL basins (exploratory, multi-seed, local weights).**
`results/weekend_sweep/` (untracked; 50M TinyStories, 7 τ × 3 seeds × L∈{256,512,1024}, weights on disk): PPL at 2× extrapolation decreases monotonically in τ up to the grid right edge r=1.7 (`analysis/summary.md` explicitly flags the grid-edge limitation). `results/m4_max_36gb/exp4_progressive_350m/REPORT.md`: EVQ-D wins extrapolation PPL at stage 1–2 (−26.5% @4K from L=512) but *loses to Geo at stage 3* at all lengths (V1/V4 FAILED), and passkey is at chance for all arms (~900 passkey samples is too few). Only 4 stage checkpoints + pilot retain `model.pt`.

**T7 — OLMo Q/K-only LoRA (NLL vs capability dissociation at the projection level).**
Owner: `rebuttal/rebuttal_0723/theory_results/OLMO2_1B_FRESH_GENERAL_QK_EVQ_YARN_20260727.md` (commit `2f9706b`, 2026-07-27 — note: "QK evaluation" here means Q/K *projection LoRA*, not QK-logit analysis). EVQ Q/K-only adaptation flattens the long-range NLL curve (3.38 vs native 5.11 @16K) but RULER macro collapses (6.1%/4.0%/2.4%) — worse than native's 58.5% at 4K.

### 1.3 Explicitly *not* trustworthy for mechanism conclusions

- `rebuttal/rebuttal_0723/theory_results/EVQ_ATTENTION_RESTORATION_SPEC.md`, `evq_attention_restoration.py`, `native_protected_evq.py`, `far_only_evq_residual.py`, spectral-frame 50M (`3908aae`, 2026-08-03): **design-only / never executed** (`DESIGN_ONLY_RUNTIME_UNVERIFIED`, `CPU_READY_GPU_RUNTIME_UNVERIFIED_NOT_TRAINED`). Their existence is not evidence.
- Weekend-sweep AR passkey outputs: generation is degenerate (`"ar_generated": "0-4-4-4-4>>"` for every trial in e.g. `results/weekend_sweep/L512/50m_tau1.41_seed42/passkey_nll.json`); the 50M TinyStories models lack the basic task. TF NLL-gaps there are noise-level (±0.1).
- Historical hybrid runs flagged `I-HYBRID-ALIAS` (realized-hash mismatch) and the pre-repair query-gap arm (`G-OLMO-QUERY-GAP-PRE-REPAIR`) — see `INTERNAL_NEGATIVE_AND_DIAGNOSTIC_LEDGER_20260726.md` lines 35, 42.
- Any 16K/32K capability claim sourced from rebuttal prose without one of the owners above.

---

## 2. Verdict per mechanism link

Grading: **verified** = raw/hash-backed causal or matched evidence; **plausible** = consistent evidence, missing controls or scope; **unsupported** = no completed measurement; **contradicted** = completed evidence against.

### Link 1: frequency allocation (realized, distinct, kernel-level effect) — **verified**

Closed-form implementation with hash-verified realized tensors (§1.1); content-free collision reduction verified in the submitted appendix (`sec:mechanism-isolation`, 36 configs). Boundary: this is a statement about the position code itself, upstream of any content-bearing computation.

### Link 2: frequency allocation → target/distractor QK competition — **verified in one narrow setting; plausible in general; the attribution is co-adaptation, not runtime geometry**

- Verified (T2, 16K, adapted 8B, seed 42, 10 cases): per-head pre-softmax gold-block hit@16 64.06% vs 18.75%, EVQ wins 10/10 paired cases; per-layer/per-head raw exists for 8K/16K/32K in `phase0_evq.json`/`phase0_geo.json` (30 case-entries per arm).
- Materially bounded at 4×: at 32K the addressing advantage shrinks to hit deltas of +0.03…+0.16 with zero sparse mass gain; the run's own preregistered gate failed (`phase0_summary.json`, `"32k_evq_mass_gain_at_least_1p5": false`, `status: stop`).
- Attribution caveat (T2 swap table): Geo-adapter + EVQ runtime frequencies improves NLL (15.53→14.13) but creates **0%** source binding; EVQ-adapter + native runtime frequencies retains most of the advantage (9.25 vs 9.07, 60% vs 80% binding). The QK competition advantage is predominantly a property of *weights trained under EVQ*, not of the rotation geometry at inference.
- Zero-shot injection control (T1): EVQ frequencies without adaptation drive held-out retrieval from 55% to 0% — runtime geometry alone is *destructive* on a native-trained model.
- No QK-competition measurement exists on any scratch-trained model or on the OLMo arms where capability conversion was proven. That is the single most important unmeasured cell (§4).

### Link 3: QK competition → attention-weighted V written into the residual stream (and causally used) — **verified at 16K in the adapted-8B setting (via ablation), single seed**

The gold-drop interventions (T2) delete the gold block from answer-side decode attention only, holding prefill and rotary indices fixed: EVQ +1.506 NLL all-heads (+0.527 on 32 heads), damage in 10/10 cases at every depth; Geo ≈0 in both. This is deletion-based causal evidence that attention-weighted V from the gold block enters the residual stream and matters for the output — and equally strong evidence that the **Geo arm makes no measurable use of the remote source at all at 16K** (the contrast is binary, not gradual). Boundaries: no direct per-layer A@V vector decomposition exists (the prepared `normalized_context_mse` in `evq_attention_restoration.py` was never run); effect is distributed beyond the 32 best-ranked heads, so the "retrieval-head" framing under-captures the pathway.

### Link 4: propagation/correction across later layers — **existence verified; structure unsupported (unmeasured)**

Existence: the +1.506 NLL output-level effect proves gold-block V-information written at attention layers survives to the final logit in the EVQ arm. Structure: never analyzed. The paired per-layer logit-lens tensors needed for this were collected and then abandoned: `results/readout_conversion_s42_20260715/raw/causal_{evq_cosh,native_geo}/records/*.pt` (bf16 `[3 answer positions × 32 layers × 128,256 vocab]`, dense vs `gold_drop_all`, 5 EVQ + 6 Geo records, 5 prompt-SHA-matched pairs at 16K, depths 10/10/25/25/50; tensor contract in each `manifest.json`; code `run_readout_trace`, commit `b42407b`, 2026-07-15). No analysis artifact exists anywhere in the repo. Nothing contradicts propagation; what is established (oracle near-null, rank ≈2,000 at output) suggests the loss of magnitude happens *downstream of attention routing*, but where in depth it happens is unknown. This is the gap §4 closes for free.

### Link 5: → final answer-token logits and generation — **split: answer-logit improvement verified; "logit improvement ⇒ generation" contradicted; conversion achievable only with exploit-training, and only at ≤2×**

- Answer-logit level, verified (T2): dense answer NLL 9.06 vs 15.51; first-token median rank 2,043 vs 33,774.5; top-1000 fraction 60% vs 0%. Consistent teacher-forced/NLL gains across scales (T4; QuALITY gold-NLL −30% in `data/curated/quality_454m_full_eval.json` with accuracy at chance).
- Generation as automatic consequence, contradicted (T2): exact match 0% for both arms at 16K; matched-budget *oracle* gold inclusion improves EVQ by only 0.034 NLL; score sparsity does not help EVQ (DiD favors Geo); chat-format and decode-parity artifacts ruled out. The recorded stop decision: "the remaining bottleneck is the learned task/readout path".
- Repeated NLL↔capability dissociations: fresh 8B counterfactual arm is **57.7% better PPL at 32K yet 0/20 strict exact** (`LLAMA8B_FRESH_COUNTERFACTUAL_RESULT_20260726.md`); OLMo Q/K-only LoRA (T7); `G-LLAMA-READOUT`, `G-LLAMA-32K` ledger entries.
- Conversion with exploit-training, verified at 2×: T1 (0/100 vs 69/100 & 67/100, two EVQ seeds) and T3 (0% vs 77.5% AR). Both have matched controls; both are single-model/single-protocol; T1 is task-family-adapted with held-out-task transfer **0%**.
- At 4× (32K): contradicted everywhere tested (T1 16K 1/20 is already marginal at 2× of its 8K design point; 8B 32K RULER zero for EVQ and Native-LoRA; 32K QK addressing decays, T2).

### Chain as a whole

The naive reading — "EVQ separates positions pre-softmax, therefore long-context behavior improves" — is **contradicted as a sufficient explanation** twice over: (i) the runtime tensor alone neither creates source binding (swap) nor preserves capability (injection); (ii) perfect attention routing onto the gold block (oracle) leaves generation at 0%. What the evidence supports: EVQ is a *training substrate* that (a) reduces wasted/OOD spectral budget in-window, (b) lets optimization learn long-range addressing + causal source use that geometric RoPE demonstrably does not learn under identical budgets (binary contrast in T1/T2/T3), while (c) the final readout/decoding competence is a separate, trainable, and currently distance-bounded bottleneck.

---

## 3. Strongest alternative explanations and confounders

Ranked by evidential weight in this repository:

**C1 — Training-time co-adaptation, not inference-time geometry (strongest).** Evidence: T2 swap table (advantage travels with the adapter); T1 zero-shot injection 55%→0%; `D-TRANSPLANT` (static LoRA cannot conjugate one frequency generator into another; `OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md`). Consequence: any mechanism story must be about *what becomes learnable under the EVQ grid*, not about geometric separability of a fixed computation.

**C2 — OOD-phase destructive interference, not target/distractor margin (the repo's own best alternative).** Evidence: T5 — held-out NLL *improves* when selected trained channels are deleted (Geo −0.28, EVQ −0.11 @8K); high-norm band deletion helps at ≥1K; norm/phase proxies fail to predict causal importance; FMRoPE retargeting suppresses interference far better than EVQ reallocation. Under this reading, part of EVQ's NLL gain is "fewer channels whose learned features go destructive beyond L_train" — a content-independent effect that needs no attention-competition story at all. The two mechanisms are compatible, but the repository has never separated them.

**C3 — Spectral-budget reallocation is a trade, not a free win.** EVQ densifies in-range frequencies (§1.1 numeric check) and consistently pays in-range: OLMo 4K NLL 2.548 vs 2.235 (T1); exp4 stage-3 Geo wins at *all* lengths (T6); submitted Table 2 protocol keeps the train-length cost adjacent. Any "long-context improvement" framing that omits the in-range cost misstates the mechanism as unidirectional.

**C4 — Task-family adaptation masquerading as general capability.** T1 trained on RULER-family routing data and transfers 0% to held-out UUID/VT tasks; clean LongAlign+Tulu SFT fails full RULER (9.74%/4.01%/2.05%, `G-OLMO-CLEAN-RULER`); 7 non-RULER objectives failed screening (`G-OLMO-NON-RULER`). The capability-level positives are within-task-family conversions.

**C5 — Metric artifacts: teacher-forced measures saturate/inflate.** TF retrieval is 1.0 for *both* 750M arms at 8K where AR is 77.5% vs 0% (T3); Primary I TF-passkey 100% coexists with per-seed AR 0.18/0.58/0.98 (curated claim boundary, T4); "retrieved" in `eval_passkey_scratch.py` is merely `nll_gap > 0`, a coin-flip-adjacent threshold for weak models (weekend sweep). Conclusions built on TF/NLL endpoints do not transfer to generation.

**C6 — QK-norm/temperature shifts unmeasured in the probe.** The phase0 probe compares raw post-RoPE scaled scores across arms whose LoRA training could have changed Q/K norms or effective attention temperature (sharpness) rather than gold-vs-distractor *selectivity*. No per-arm Q/K norm or attention-entropy comparison exists for the 8B pair. Related repo evidence says norm proxies are unreliable importance predictors (T5, Spearman −0.167), so a norm-based confounder can be neither assumed nor excluded. The hit@16/mass metrics are rank-based (less temperature-sensitive than margins), which partially mitigates this — but the raw `margin` fields in `phase0_*.json` should not be compared across arms without a norm control.

**C7 — Grid-convention nuisance.** `tau=0, midpoint=true` is not original RoPE: the half-cell shift alone improved 16K diagnostic NLL from 14.87 to 13.77 in the base-model diagnostic (`docs/exp/2026-07-14_lora_retrieval_conversion_probe.md`, "Geo identity"). Any experiment that labels midpoint-τ0 as "geometric baseline" imports a free confound; T2 used the true native tensor for the Geo arm (buffer error 1.8×10⁻⁸), but new experiments must re-check this.

**C8 — Selection and power.** T2: 10 cases, 32 heads selected at 8K by pooled block rank, single seed, single model; T1/T3: single model each, EVQ training-seed replication only in T1. Distance structure matters: within-training-gap 48/50 vs beyond-gap 21/50 (odds ratios 33–39, `OLMO2_N100_GAP_STRUCTURE_AUDIT_20260726.md`). None of the mechanism-level results have multi-seed or multi-model support.

---

## 4. Most informative experiment with existing checkpoints and no new training

### Recommendation: layer-resolved causal-readout decomposition from the already-collected paired traces, cross-referenced with the per-layer QK probe raw

**Why this one.** Link 4 is the only chain link with zero completed measurement, the data to measure it already exists on this machine (no GPU, no checkpoint loading, no downloads), and its outcome discriminates between the competing explanations for the one result that currently blocks the whole story — the readout bottleneck. Every alternative candidate is dominated: local scratch checkpoints (weekend 50M, exp4 350M) fail the basic-task-capability floor for retrieval mechanisms (§1.3, T6); the 8B/OLMo checkpoints for new forward-pass probes are remote-only; the association-swap and A@V-decomposition designs require model access.

**Data (all local, all frozen):**
- `results/readout_conversion_s42_20260715/raw/causal_evq_cosh/records/*.pt` (5) and `.../causal_native_geo/records/*.pt` (6): bf16 logit-lens tensors `[3 answer positions, 32 layers, 128256 vocab]` for dense (`full_logits`) and `gold_drop_all` (`ablated_logits`) attention, 5 prompt-SHA-matched 16K pairs (depths 10/10/25/25/50), gold token ids included per record; contract and SHA-256s in each arm's `manifest.json`.
- `results/lora_sparse_conversion_s42_20260714/phase0_evq.json` / `phase0_geo.json`: per-layer, per-head pre-softmax gold-block metrics (block rank, hit@{8,16,32}, margin, dense/sparse answer mass) for 10 cases × {8K, 16K, 32K} per arm.

**Metrics (chosen here; deliberately *not* assuming the margin formulation):**
1. **Per-layer causal evidence curve.** For each paired case, answer position t, layer l: `δ_l = z_l^full(gold_t) − z_l^ablated(gold_t)`, recentered per layer by the median over vocabulary to remove global shift: `δ̃_l = δ_l − median_v[z_l^full(v) − z_l^ablated(v)]`. Complement with the basis-robust rank statistic already defined in the trace contract: gold rank under `Δz_l = z_l^full − z_l^ablated` (`causal_delta_rank` semantics, `manifest.json` "rank" rule).
2. **Entry layer and retention.** `l* = min{l : δ̃_l exceeds 3×MAD of layers 0..l*−1}` and `R = δ̃_31 / max_l δ̃_l`. These two numbers classify the failure mode.
3. **Final-layer competition decomposition.** At l=31 (and post-final-norm), list top-32 tokens of `z^full`; classify gold / distractor-digit / local-copy tokens; report the gold-vs-best-competitor *output* gap under dense and ablated. This tests whether the readout failure is competition from locally-supported tokens (attention-independent) vs insufficient gold evidence.
4. **QK↔residual alignment.** From `phase0_*.json`, per-layer EVQ−Geo gold-block advantage (mean hit@16, gold mass over heads) vs the layer profile of `δ̃_l`: does causal evidence enter at the layers where the QK advantage lives (the frozen-head contract concentrates in layers 11–31)? Include the 32K phase0 slices to show the decay side.
5. **Built-in negative control.** Geo's curve should satisfy `δ̃_l ≈ 0` at every layer (its output-level gold-drop effect is −0.01). If it does not, stop and audit the traces before interpreting anything.

**Decision rules (pre-specified):**
- **H-late-entry** (l* ≥ 28): remote evidence reaches the residual stream too late for MLP/attention post-processing → mechanism story: addressing works, integration depth is the bottleneck; ICLR framing = mechanism clarification; conversion training should target late layers/readout.
- **H-decay** (early l*, R < 0.5): later layers actively attenuate the evidence → "correction across layers" fails; motivates activation-patching follow-up.
- **H-competition** (R ≥ 0.5 but final gap dominated by locally-supported competitors): evidence survives; the answer loses a prior fight, consistent with the oracle-null; motivates decoding/readout-calibration analysis rather than more attention work.
- Any outcome with Geo control violated → implementation audit, no interpretation.

**Cost and environment.** CPU-only, ≈520MB of tensors, minutes of compute. Note: the system `python3` on this machine lacks `torch` (verified); use a project environment per `requirements.txt` (CPU torch suffices). Output goes to `results/readout_decomposition_s42_<date>/` (untracked) + a `docs/exp/` report, per `INDEX.md` §5.

**Declared boundaries.** 5 paired cases, single seed, adapted-8B setting; both trace manifests carry `measurement_label: "oracle-diagnostic"` and `single_seed_supporting: true` — the analysis inherits those labels and cannot be promoted to a paper-level claim without expansion. Logit-lens basis sensitivity is mitigated (not eliminated) by the rank statistics and the paired-difference design.

**What this experiment deliberately does *not* re-measure (per the checklist in the task):**
- *Target-vs-distractor pre-softmax logits*: already measured, per-head/per-layer, at 3 lengths (phase0 raw); reused here, not re-run.
- *Head/layer specificity*: covered by metric 4 (and phase0 raw is per-head; no all-head averaging is used anywhere in this design).
- *Direct A@V vector into residual*: impossible without model forwards; the causal deletion surrogate (gold-drop) is already stronger for "used" vs "present"; the prepared `normalized_context_mse` path exists for a later authorized GPU session.
- *Downstream causal influence on the answer logit*: `δ̃_31` *is* that quantity, layer-resolved.
- *QK norm/temperature*: not recoverable from saved traces; carried as open confounder C6. First authorized follow-up on remote checkpoints (inference-only, no training): per-arm Q/K norm and per-head attention-entropy comparison on the same 10 cases, plus `run_association_swap_trace` (code exists, never run).
- *Remote-evidence removal / activation patching*: removal is already the ablation inside the traces (`gold_drop_all`); true layer-patching (copy layer-l residual from dense into ablated run) needs model access — specified as the H-decay follow-up, runnable on existing checkpoints remotely with zero training.

---

## 5. Repository files and functions needed to instrument §4

**Read-only inputs:**

| Artifact | Role |
| --- | --- |
| `results/readout_conversion_s42_20260715/raw/causal_evq_cosh/{manifest.json,records/*.pt}` | EVQ paired traces (5 records) |
| `results/readout_conversion_s42_20260715/raw/causal_native_geo/{manifest.json,records/*.pt}` | Geo paired traces (6 records; use the 5 SHA-matched ones) |
| `results/lora_sparse_conversion_s42_20260714/phase0_evq.json`, `phase0_geo.json` | Per-layer/per-head QK probe raw, 8K/16K/32K |
| `results/lora_sparse_conversion_s42_20260714/phase0_summary.json` | Frozen 32-head contract (layer/head list) + gate status |
| `results/lora_sparse_conversion_s42_20260714/rtx5090_causal_manifest.json` | SHA-256 anchors for the causal runs |
| `docs/exp/2026-07-14_lora_retrieval_conversion_probe.md`, `docs/exp/2026-07-15_lora_readout_conversion_plan.md`, `docs/exp/2026-07-15_readout_conversion_impl_plan.md` | Protocol definitions and stop history |

**Reference implementations (do not modify; import or mirror semantics):**

| Function | Location | Use |
| --- | --- | --- |
| `run_readout_trace` | `experiments/lora_evq_v2/eval_sparse_conversion.py:2371` | Defines the tensor contract the analysis must honor (shared dense prefill; answer-side-only ablation) |
| `per_layer_logit_lens`, `per_layer_logit_lens_delta` | same file, lines 1814, 1826 | Canonical lens semantics (final norm + lm_head) — traces already store lensed logits, so the analysis only consumes them |
| `causal_delta_rank` | same file, line 1791 | The registered rank-under-delta rule (metric 1) |
| `_probe_metrics`, `_probe_one` | same file, lines 1363, 1443 | Field meanings for `phase0_*.json` (block layout, hit@k, margin, mass) |
| `exact_block_attention_forward`, `_answer_nll` | same file, lines 1217, 1842 | Definitions of `gold_drop_all` and the NLL endpoints being decomposed |

**New code (one new file, no tracked-source edits):** `scripts/analysis/readout_trace_decomposition.py` — loads both manifests, verifies record SHA-256s, pairs by `prompt_sha256`, computes metrics M1–M5, writes `results/readout_decomposition_s42_<date>/{summary.json,per_case/*.json,figures/*.pdf}` and a `docs/exp/<date>_readout_trace_decomposition.md` report. Tests belong in `tests/` following `tests/test_olmo2_evq_attention_restoration.py` conventions.

**For the later authorized-GPU follow-ups only** (not part of §4): `run_association_swap_trace` (`eval_sparse_conversion.py:2485`, never executed), `relation_capture_sdpa` / `normalized_context_mse` (`rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/evq_attention_restoration.py`, design-only), and the frozen OLMo adapters whose SHA-256s are recorded in `OLMO2_1B_4K_ONLY_ROUTING_CONVERSION_20260726.md` (a QK phase0-style probe on those arms would test link 2 in the setting where capability conversion is proven — the most valuable missing cell overall, inference-only but remote).

---

## 6. ICLR recommendation

**Recommendation: mechanism clarification** — reframe the contribution as "training-time frequency allocation is a *substrate* that changes what long-range circuitry is learnable", built on the causal instrument set that already exists (matched-arm routing conversion, gold-drop deletion, frequency/adapter swaps, oracle inclusion), with a bounded theory component. Not a theory upgrade as the headline; not a new method; decisively not a no-go.

Reasoning against the alternatives:

- **No-go: rejected.** The repository contains a twice-replicated, hash-frozen, matched-control capability conversion with a binary effect (0/100 vs 69/100 & 67/100, McNemar 3.39×10⁻²¹; independently echoed at 750M scratch scale, 0% vs 77.5%). Phenomena of this effect size with clean controls are publishable; abandoning them would waste real evidence.
- **Theory upgrade as headline: capped by the repo's own negatives.** Cosh is not schedule-optimal in trained models (`G-COSH-NOT-UNIVERSAL`); the τ rule is a basin selector whose empirical support is grid-edge-limited (`G-TAU-FALLIBLE`; weekend-sweep summary); \(\mathcal{C}_{\mathrm{app}}\) is content-free and cannot speak to the co-adaptation and readout phenomena that dominate the capability results. A theory section should absorb the OOD-phase-interference finding (T5) — that is a genuine theoretical target the current surrogate does not model — but it cannot carry the paper alone.
- **New method: premature.** Every "fix the readout / protect native channels / restore attention" direction is design-only with zero completed results (§1.3), and the one executed conversion attempt (sparse conversion + 50-step micro-tune) hit its registered stop conditions. Proposing a method before the mechanism paper would repeat the current paper's weakness at higher cost.

**What the mechanism-clarification paper needs (gap list, in priority order):**
1. The free link-4 analysis (§4) — closes the last unmeasured link on existing data.
2. QK phase0-style probe + Q/K-norm/entropy control on the frozen OLMo matched arms (inference-only, remote checkpoints, no training) — moves link 2 from "adapted 8B only" to the setting where capability conversion is proven, and retires confounder C6.
3. Case/seed expansion of the 8B causal suite (n≥100 cases; ≥2 LoRA seeds) — currently everything mechanism-level is single-seed, 10 cases.
4. Honest boundary panel carried from this repo: 32K reversal (QK decay + RULER zero), held-out-task zeros, in-range NLL cost, gap-structure sensitivity, and the interference-pruning result. These are load-bearing for credibility, not concessions.

**Framing guardrails** (consistent with `AGENTS.md` §4): keep "operating prior/basin selector" language for τ; keep TF/NLL, AR exact, and task-family adaptation visibly separate; never present the runtime tensor as the causal agent — the swap and injection controls are the paper's own strongest argument that the story is co-adaptation.

---

## Appendix: verification log for this audit

- Read directly: `scripts/lib/rope/schedules.py`; `eval_sparse_conversion.py` (probe, causal, trace functions); `phase0_summary.json`; `rtx5090_causal_manifest.json`; both readout `manifest.json`s; `docs/exp/2026-07-14_lora_retrieval_conversion_probe.md`; `OLMO2_1B_4K_ONLY_ROUTING_CONVERSION_20260726.md` (full); `LLAMA8B_FRESH_COUNTERFACTUAL_RESULT_20260726.md`; `EXPERIMENT_REPORT_20260724.md` §§10–12 excerpts; `INTERNAL_NEGATIVE_AND_DIAGNOSTIC_LEDGER_20260726.md` entry table; `results/frequency_band_usage_s42_20260724/REPORT.md`; `results/weekend_sweep/analysis/summary.md` + sample `passkey_nll.json`; `results/m4_max_36gb/exp4_progressive_350m/REPORT.md`; `phase15_{geo,evq_r0}_seed42_result.json` + `phase15_continue_summary.json`; `paper/sections/03_theory.tex` and `paper/appendix/{a1_proofs,a2_experiment_details}.tex` excerpts (read-only); `eval_passkey_scratch.py` metric definitions.
- Computed independently: EVQ vs geometric channel-period tables at (K=32, head_dim=64, base=5×10⁵, τ=1.414) from a pure-math reimplementation of `evq_cosh_phi`.
- Git: `git ls-files` confirms `results/` is untracked except `llama8b_longbench/`, `video_dit/`, `qwen_longbench_21task/`, `README.md`; commits cited: `2734d12`, `737cb31`, `d356977`, `b42407b`, `fec0157`, `2f9706b`, `4384d69`, `3908aae`, HEAD `515eb36`.
- Environment limitation: system `python3` has no `torch`; all numeric checks above were pure-Python; no model was loaded, no GPU used, no training or downloads performed.
- Not independently verified (trusted at owner level): remote-hash-only artifacts (OLMo adapter archives, fresh-CF 8B artifacts), since the binaries are intentionally not in the repository.
