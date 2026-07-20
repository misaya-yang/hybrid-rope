# Pre-deadline experiment and theory review (2026-07-20)

Date: 2026-07-20 (rebuttal deadline 2026-07-22; real reviews **not yet received**)
Status: `canonical_pre_deadline_review` — consolidates all experiments closed
2026-07-13 → 2026-07-15 and an independent re-verification of the theory audit.
Mode unchanged: `triage-only / response-only`. No paper number is modified.

Authority: this document extends, and does not override,
`FULL_PAPER_INTEGRITY_AUDIT_20260713.md` (method identity / theory / protocol
facts) and `rebuttal_playbook.md` (operational entry). Where this review adds
new experiment dispositions (07-14/07-15 runs) and new minor theory findings,
it is the latest authority for those items only.

---

## 1. Independent theory re-verification (2026-07-20)

All checks below were re-run from scratch (fresh derivations + numerics),
not copied from the 07-11/07-13 audits. Scripts: local scratch; collision and
expansion checks reproduce with pure NumPy in <1 min.

### 1.1 Confirmed sound (safe to defend)

| Claim | Check performed | Result |
| --- | --- | --- |
| Cosh minimizer theorem (Thm `thm:ode`) | Full re-derivation: first variation, ODE, BCs `ρ'(0)=-τ²`, `ρ'(1)=0`, closed form, normalization, positivity, β=0 limit | **Holds** for the stated surrogate; interior stationarity self-consistent at all φ |
| `min`-kernel PSD identity | `∬ff·min = ∫(∫_s^1 f)² ds` re-derived via indicator decomposition | Holds; convexity/uniqueness sound |
| Self-consistency identity `τ²T₂+T₁ = τcothτ` | Green-identity re-derivation + numeric (τ=0.5/1/2) | Holds to numeric precision; `T₂` corollary closed form correct |
| `S_χ²` closed form and `τ⁴/45` leading term | Symbolic expansion + numeric integral (τ=0.3/0.5) | Both correct (`∫a²=1/45` re-derived) |
| Waterbed expansions | `∫M²=2τ⁴/945` re-derived analytically | Correct; χ²=∫1/ρ−1 and Rényi corollary steps check out |
| `Q₁(L,b)` and `c_pred=√(45Q₁)` | Independent numeric integration, b=500K, L∈{256,512,1024} | `Q₁∈[0.0314,0.0319]`, `c_pred≈1.19` — this *surrogate-side* prediction is real |
| Floor expansion leading term `4√(N/K)` | Taylor re-derivation of `arcsinh(sinh τ/2)/τ` | Leading `−τ²/16` confirmed |
| Midpoint-Geo global factor `b^{-1/(2K)}` | Direct computation | 0.6636 / 0.8146 / 0.9026 for K=16/32/64 — matches full audit §2.3 |

### 1.2 Confirmed flaws (previously identified; independently re-verified)

1. **Ordinary-KL order error (T-02) — confirmed.** Numeric check: for
   `z_θ = z₀+θg`, `KL(p₀‖p_θ)/θ² → ½gᵀJ_sm g` (constant ≈0.586 in the test
   instance), i.e. KL is second order in `θ=τ²`, so ordinary KL starts at
   `O(τ⁴)` with zero first variation. The submitted `O(τ²)` "post-softmax KL
   gain" naming in §`sec:tau-star`/Prop. `softmax-transport` is an
   order/identity error. The `τ²Q₁` term is the first-order change of the
   *linear phase-variance allocation score* `∫ρ_τ(φ)q(Lb^{-φ})dφ` — a
   transport proxy, not any KL.
2. **`c_coll=1.171` calibration — confirmed invalid, with a sharper
   counterexample.** `scripts/analysis/verify_c_coll.py` reads the paper
   table's `tau_coll` as *input* (`c_obs = tau_coll/(d/√L)`) and never
   optimizes the collision score. Independent optimization at the
   representative config (d=64, L=512, b=500K, midpoint EVQ family):
   `C(τ=3.3093)=28.183` (paper's point), `C(τ=13.05)=0.042`, coarse-grid
   argmin `τ≈13.3` (`c≈4.7`). The paper's point is ~670× above the family's
   collision minimum, so `c_coll≈1.17`, the "2% agreement", `CV 0.28%`, and
   `λ_∞=0.96` are all unusable. **What survives:** `c_pred=√(45Q₁)≈1.19` is a
   genuine surrogate-side computation, and the *fixed-allocation directional
   diagnostic* (deployed EVQ reduces the collision statistic vs Midpoint-Geo,
   24–92% across the 12 listed configs) still stands.
3. **Phase16 "27 configurations / all <1%"** — not re-run here; the 99-run /
   9-config / pilot+selected-confirmation structure and the 7/9-win, 2/9-loss
   common-metric result from the full audit remain the only safe description.
4. **LoRA rank / SFT recovery section (T-05) — confirmed inconsistent.**
   `R(x)=1/√(1+Λ₀e^{-x})`: 70%@x=5 implies `Λ₀≈154.5`; 95%@x=10 implies
   `Λ₀≈2380`. At fixed `Λ₀` the 70%→95% gap is `Δx≈2.265`, not 5. Combined
   with the rank-≠-frequency-channel error, the whole subsection stays
   `REMOVE_FROM_DEFENSE`.
5. **MLA `d_eff`** — unchanged: actual `head_dim=64`, `d_rope=32`;
   `τ=1.414` equals `128/√8192` only under the ad-hoc doubled convention;
   never a theorem.

### 1.3 New minor findings (not in the 07-11/07-13 audits)

1. **`T₁` closed-form typo (Appendix `a1_proofs.tex`, after the `T₂`
   corollary).** Printed: `T₁(τ)=τ(1+sinh2τ/2τ)/(2sinh²τ)`. This is off by an
   overall factor τ in one term — it evaluates to 2.0026 at τ=0.5 and 0.5947
   at τ=2 against true values 1.0013 / 1.1894, and is coincidentally correct
   only at τ=1. Correct closed form: `T₁(τ)=τ(τ+sinh(2τ)/2)/(2sinh²τ)`
   (equivalently `τ²/(2sinh²τ)+(τ/2)cothτ`; verified numerically). The
   self-consistency identity and the `T₂` corollary are unaffected. Low
   stakes, but if a reviewer checks the algebra it should be acknowledged as
   a typo, not defended.
2. **Useful reframing of the collision counterexample.** The collision-only
   optimum (`τ≈13`, `c≈4.7`) is far outside the trained-PPL basin (`c≈1`,
   Phase16). This is not only a flaw to concede — it is direct evidence that
   the collision diagnostic and the trained objective *diverge at large τ*,
   which the corrected three-layer story already predicts (in-range χ²/
   waterbed cost grows as τ⁴ while the collision score keeps falling). If a
   reviewer asks "if collision is the mechanism, why not deploy the
   collision-optimal τ?", the honest answer is: collision is a redundancy
   diagnostic, not the objective; the deployed point balances it against
   in-range fidelity, and that balance is validated only empirically. This
   turns a concession into a coherent mechanism statement without restoring
   any withdrawn claim.

---

## 2. Review of experiments conducted 2026-07-13 → 2026-07-15

Evidence tiers per playbook: all runs below are **single-seed, supporting /
mechanistic, deploy-on-trigger**. None modifies a paper number; none may be
upgraded to primary. Raw dirs under `results/` are local-only (gitignored);
tracked evidence = `docs/exp/` reports + curated JSONs where noted.

### 2.1 Closed runs and dispositions

| # | Experiment | Result (verified vs raw) | Disposition |
| --- | --- | --- | --- |
| E1 | **151.9M / 500M six-cell** (07-13; `data/curated/native_rope_evq_150m_s42_500m_20260713.json`) | Raw endpoint-EVQ beats native RoPE: PPL −8.6/−14.3/−16.2% at 4/8/16K; official/derived YaRN pulls both arms to near-parity; repo fixed-ramp shows strong negative interaction (−0.329/−0.413 NLL at 8/16K, EVQ-favoring) | Supports **withdrawal** of official-YaRN complementarity in abundant-channel MHA; fixed-ramp complementarity is repo-local only |
| E2 | **MHA four-operator decomposition** (07-14) | Gap collapse is caused by the **frequency correction** (gap → 0.0055/0.0124/0.0330), not mscale (gap intact: 0.091/0.177/0.206) | Mechanism attribution for E1; use only under P1 official-YaRN trigger |
| E3 | **MLA K=16 four-arm ablation** (07-14; `results/mla_yarn_short_s42_20260714/evaluation/analysis.json`, verified) | At scale 8/16 the substrate gap **survives** full official/derived YaRN: freq_only gap +0.0492/+0.2389; full +0.0259/+0.1776; EVQ+full 71.55 vs Native+full 85.46 PPL at 8K (≈16%); 8K DiD −0.106 (EVQ-favoring). Refuted: EVQ+mscale standalone (519.1 PPL), EVQ-raw-beats-Native-full (412.0 vs 85.5) | Single-seed mechanistic support for scarce-channel direction; **not** complementarity restoration; τ=1.414 not re-derived for L=512 (τ=5.66 hardening run optional) |
| E4 | **Repo fixed-ramp mechanism probe** (07-14) | Broad `scale^r` smoothstep carries nearly all the interaction (16K: −0.392 of −0.413); official *linear shared-index* control **reverses** the ordering (gap −0.1415 at 16K); repo_full+official mscale strengthens interaction to −0.507 but full official/derived YaRN stays best absolute (Native 33.16 / EVQ 32.94) | Honest boundary: the complementarity is specific to the broad gradual ramp; no novelty claim vs other by-parts scalers; no further tuning (8-offset overfit risk) |
| E5 | **Seed-42 8B LoRA capability eval** (07-13; `07 - rebuttal/seed42_lora_eval_20260713/REPORT.md`) | EVQ+LoRA temporal PPL 10.068/24.068/127.911 vs Geo+LoRA 6.817/108.958/991.475 at 8/16/32K; but S-NIAH top-1 56.67%→3.33%→0%, passkey exact 100%→0%→0%; official-YaRN x2/x4 pilot restored nothing | PPL stabilization ≠ retrieval capability; both facts must travel together |
| E6 | **Retrieval-conversion probe + sparse pilot + 5090 causal decomposition** (07-14; `docs/exp/2026-07-14_lora_retrieval_conversion_probe.md`) | 50-step micro-tune: no conversion (0% 16K before/after both arms). Attention gate: EVQ hit@16 median 64.06% vs Geo 18.75% (10/10 paired wins). Sparse pilot: **negative** (score-mode DiD −0.666 favors Geo). Causal: gold-drop-all-heads +1.5055 NLL for EVQ (×4.51 likelihood) vs ≈0 for Geo; first-token rank 33,775 (Geo) vs 2,043 (EVQ); forced-gold inclusion nearly null (−0.034); chat-wrap and decode-parity nulls | **Strongest new mechanistic asset** (see §3) plus a clean negative on zero/low-cost conversion. Legacy KV generator bug found: nominal-16K prompts were 6,827 tokens — old Geo "95% KV@16K" is invalid and must never be cited |
| E7 | **QA16K three-arm** (07-15; `results/qa16k_three_arm_s42_20260715/summary.json`, verified) | **Registered gate NEGATIVE**: task-macro F1 EVQ-LoRA 0.1126 vs Native-LoRA 0.2110 vs Base 0.2309; EVQ−Native = −0.0984, bootstrap 95% CI [−0.1297, −0.0697]. Stratified: deficit concentrated ≤8K (−0.334); >8K all arms at floor (EVQ−Native +0.018, CI spans 0; task-skewed subset) | Mandatory disclosure whenever LoRA/downstream is discussed. Kills "this EVQ-LoRA adapter is a better 16K QA model"; leaves long-range substrate mechanism unresolved |
| E8 | **Residual-RoPE pilot v6** (07-15; `results/residual_rope_pilot_s42_20260715_v6/summary.json`, verified) | **NEGATIVE / inert**: after 10-step residual-branch training (layers 28–31, branch_dim 8), all three arms produce byte-identical Qasper F1 (0.5534) and identical passkey results (8K first-value 100%, strict 0%; 16K 0%); zero-gate logit parity max_abs 0.0 | The residual-branch frequency route produced no signal at this budget; do not cite as evidence for or against EVQ — the branch never differentiated |

### 2.2 In progress — not citable

**Readout-conversion Z0** (07-15; `results/readout_conversion_s42_20260715/`):
only raw first-step logit records exist (5 prompts/arm, tensors `3×32×128256`,
`oracle-diagnostic` label, manifests hashed). The Z0/Z1 decision measurements
(first-token vs suffix causal decomposition; association swap) from
`docs/exp/2026-07-15_lora_readout_conversion_plan.md` have **no analysis
output yet**. Until the analysis lands with its registered kill conditions,
nothing from this line may enter any rebuttal component.

### 2.3 Cross-experiment validity notes

- **Controls are sound where claimed.** E6/E7 share frozen matched step-300
  adapters (same data manifest, steps, LoRA config, seed; only the frequency
  substrate differs); E1–E4 are checkpoint-only operator swaps with exact
  reproduction gates (max per-offset NLL diff 0.0). The E7 arms share prompt
  token IDs. No protocol-identity violations found.
- **Known confound retained:** the 8B pair's substrates are native-endpoint
  Geo vs midpoint-quantized EVQ (not same-quantizer), and Llama's pretraining
  is native-grid — so all 8B results measure *conversion of a pretrained
  native model*, not clean shape contrast. Already documented; keep it in
  every 8B-facing sentence.
- **Provenance gap (action item):** none of the 07-14/07-15 artifacts
  (`qa16k_three_arm`, `lora_sparse_conversion`, `mla_yarn_short`,
  `residual_rope_pilot`, fixed-ramp/mscale probes) is registered in
  `docs/overview/RESULT_PROVENANCE_MANIFEST.md` yet. Register curated,
  SHA-256-hashed summaries **before** any reviewer-facing use; raw dirs stay
  local-only.

---

## 3. What the new evidence changes for the rebuttal

### 3.1 A three-gate mechanism story (new, honest, and coherent)

The 07-14/07-15 causal work upgrades the P0-D "what remains" answer from a
defensive list to a mechanism chain with explicit pass/fail at each gate:

1. **Signal preservation — PASS (causal).** Under EVQ, deleting the gold
   block costs +1.5055 NLL (×4.51 likelihood) at 16K; under Geo the same
   deletion does nothing. EVQ demonstrably keeps remote content causally
   alive at 2× the training length.
2. **Addressing — PASS (directional).** Median target-block hit@16 64% vs
   19%; first-token gold rank 2,043 vs 33,775.
3. **Readout/task conversion — FAIL (registered negatives).** Rank ~2,000 is
   not top-1: 0% exact retrieval; QA F1 *worse* than the matched native
   control (−0.098, CI excludes 0), driven by ≤8K readout/instruction
   quality; sparse selection, forced-gold inclusion, temperature-class
   decoding, and a 50-step micro-tune all fail to convert.

Safe formulation (component, English, deploy-on-trigger):

> Frequency reallocation preserves causally usable long-range signal and
> improves attention addressing at 2× the training length, but in a short
> LoRA adaptation of a natively-pretrained model it does not by itself yield
> usable long-context task capability; our registered QA gate on the same
> adapters is negative, and the bottleneck localizes to the readout path.

This is strictly stronger than "PPL improves but capability is unknown" —
and it is falsification-disciplined (registered gates, disclosed negatives),
which is the correct posture for the P0-A/P0-C trust axis.

### 3.2 Mandatory-disclosure updates

If any of the following is used, its paired negative travels with it:

| If we cite… | We must also state… |
| --- | --- |
| EVQ+LoRA 16K/32K PPL advantage (24.1 vs 109.0; 127.9 vs 991.5) | 8K PPL is worse (10.07 vs 6.82); QA task-macro F1 is **worse** than the matched native control (−0.098, 95% CI [−0.130, −0.070]); 16K exact retrieval ≈0 for all arms |
| hit@16 / rank / causal source-use | 0% exact match; forced-gold near-null; sparse conversion negative |
| MLA scarce-channel survival under official YaRN (71.6 vs 85.5) | single seed; L_train=512; τ not re-derived (1.414, not 5.66); PPL-only (passkey saturated/near-zero); modest-scale extrapolation is still equalized |
| Fixed-ramp complementarity | repo-local scaler; official linear ramp reverses the ordering; full official YaRN is better in absolute PPL |
| Any legacy KV-retrieval @16K number | the pre-fix generator produced ~6.8K-token prompts; corrected 16K KV cells are 0% for both substrates |

### 3.3 Risk-register deltas (feeds playbook §4/§11)

- **P1 "LoRA / downstream" hardens.** A reviewer asking "does EVQ help a
  real pretrained model?" now has a documented negative QA answer. Strategy
  stays `OUT_OF_SCOPE` unless triggered — but if triggered, lead with the
  negative + three-gate story; never lead with the 16K PPL number.
- **P1 "Official YaRN / native endpoint" is now two-sided.** Withdrawal
  confirmed with mechanism (E1/E2) *and* a bounded scarce-channel survival
  signal (E3). Both directions must be presented together.
- **New P1-level risk: "PPL is your only currency."** E5–E7 make this the
  single most likely sharp follow-up if any capability wording survives.
  Prepared answer = §3.1 kernel + Primary-tier PK relabel (teacher-forced
  NLL-gap) + the AR-exact numbers already in the playbook P1 metric row.
- **P0-B unchanged** but add the §1.3(2) collision-divergence reframing as
  an optional second paragraph when the reviewer pushes on "what does the
  collision diagnostic actually buy you".

### 3.4 What remains genuinely open (do not paper over)

1. No same-quantizer native-endpoint EVQ vs native Geo training pair at any
   scale (Stage B of full audit §6 remains undone).
2. Primary II single-seed and Primary III heterogeneous-batch limits are
   unchanged by any of the new runs.
3. The readout bottleneck's mechanism (1a positional-OOD content vs 1b copy
   strength) is unresolved pending Z0/Z1.
4. Nothing new bears on Phase16 reporting or the checklist/compute conflicts;
   those corrections stand as in the full audit.

---

## 4. Verification log (2026-07-20)

- Re-derived: cosh theorem end-to-end; PSD identity; self-consistency
  identity; `∫a²=1/45`; `∫M²=2τ⁴/945`; floor leading term; χ² identities.
- Numerics (fresh NumPy, local): KL/θ² → const ⇒ KL=O(τ⁴) ✓; collision
  C(3.3093)=28.183 vs C(13.05)=0.042, argmin τ≈13.3 (c≈4.7) ✓; T₁ printed
  form wrong at τ=0.5/2.0, corrected form matches numeric ✓; `S_χ²` closed
  form matches numeric ✓; Q₁(256/512/1024, b=500K) = 0.03145/0.03192/0.03159
  ⇒ c_pred≈1.19 ✓; SFT Λ₀ inconsistency (154.5 vs 2379.6; fixed-Λ₀ gap
  2.265) ✓; midpoint factors 0.6636/0.8146/0.9026 ✓.
- Raw-artifact cross-checks: `qa16k_three_arm` summary.json deltas/CIs match
  the 07-15 report ✓; `mla_yarn_short` analysis.json gaps (+0.2266 raw @1K,
  +0.2389 freq_only @8K, +0.1776 full @8K; EVQ 71.55 / Native 85.46 PPL;
  DiD −0.106) match the ablation doc ✓; residual pilot summary (gate
  negative, parity 0.0, identical F1 0.5534) ✓; `verify_c_coll.py` confirmed
  to read `tau_coll` as input (no optimization) ✓.
- Not re-verified here (accepted from prior audits with their own evidence):
  Phase16 manifest recount; Primary II 151.9M parameter recount; video/750M
  provenance items.

## 5. Action items before 2026-07-22

1. Author sign-off on the §3.1 capability kernel and §3.2 disclosure pairs
   (playbook `AUTHOR_INPUT_NEEDED` items remain open).
2. Register curated summaries + SHA-256 for the 07-14/07-15 artifacts in the
   provenance manifest (E3, E6, E7, E8 at minimum).
3. Freeze the three short kernels from playbook §6 unchanged; add the §3.1
   kernel as a fourth frozen component.
4. When reviews arrive: verbatim mapping first (`REVIEWER_TRIAGE_PLAYBOOK`),
   3–5 score-driving concerns, then cut kernels to fit 10,000 characters.
5. Do not start Z0/Z1 analysis, τ=5.66 MLA rerun, or any GPU work unless a
   real reviewer question makes it score-changing (playbook §8 gate).
