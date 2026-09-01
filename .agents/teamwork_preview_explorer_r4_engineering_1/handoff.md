# Handoff Report: R4 Axis B Engineering Boundaries & Native-Support Pure-$z$ Paradigm

- **Agent:** `explorer_r4_engineering_1`
- **Working Directory:** `/Users/misaya.yanghejazfs.com.au/paper_project/hybrid-rope/.agents/teamwork_preview_explorer_r4_engineering_1`
- **Date:** 2026-09-01
- **Handoff Type:** Hard (Task Complete)

---

## 1. Observation

1. **Table-Shock in 50M Co-Adaptation Probe:**
   - *Source:* `paper-2027/research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` and `INDEX.md` §3.1.
   - *Verbatim Values:* Geo weights on Geo table PPL = `7.1413`; Geo weights on EVQ table PPL = `76.1955`; EVQ weights on Geo table PPL = `23.0524`; EVQ weights on EVQ table PPL = `7.1597`.
2. **Table-Shock in 151.9M OLMo Crossing:**
   - *Source:* `paper-2027/research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md` §4.
   - *Verbatim Values:* At 1K context, FMRoPE-trained weights prefer their FMRoPE-derived table (`3.426` vs `5.776` tail NLL), while anchored-Cosh-trained weights prefer their Cosh-derived table (`3.479` vs `4.455`).
3. **Exact Post-Hoc Linear Transplant Obstruction:**
   - *Source:* `rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md`.
   - *Verbatim Theorem:* For unequal frequency multisets $\Omega \neq \Omega'$, there is no fixed invertible linear map $M_Q, M_K$ on Q/K projections preserving all relative-position logits across all displacements $\Delta$.
4. **Analytic Zero-Parameter Single-Table Gate Failure:**
   - *Source:* `paper-2027/research/attention-aware-retrofit/results/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md`.
   - *Verbatim Values:* Anchored EVQ-Cosh at $1\times$ tail NLL degrades by `+3.9780` ($2.9712 \to 6.9492$), while $2\times$ tail NLL improves by `-0.2533` ($7.1029 \to 6.8496$). Protected-band Cosh at $1\times$ tail NLL degrades by `+0.6922` ($2.9712 \to 3.6633$), while $2\times$ improves by `-0.1845`.
5. **Continuous-Boundary-Slope Operator Failure:**
   - *Source:* `paper-2027/research/attention-aware-retrofit/analysis/ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md` §2 and `INDEX.md` §3.4 Item 12.
   - *Verbatim Values:* Stateless continuous-boundary-slope operator scored `0.0000` core-4 RULER macro at both 8K and 16K, vs `0.7175 / 0.4075` for the routed frozen-table policy.
6. **Inversion of $D^*$ and RULER:**
   - *Source:* `paper-2027/research/attention-aware-retrofit/analysis/RETROFIT_AXIS_FALSIFICATION_20260822.md` §2.
   - *Verbatim Values:* Spearman rank correlation between RULER and $-D^*$ was `-0.550`; `one_turn_floor_s2` achieved near-optimal $D^* = 0.0192$ but scored `0.0000` RULER.
7. **Failure of LeRoPE $w^{1/3}$ Structural Curvature Oracle:**
   - *Source:* `paper-2027/research/audits/LEROPE_PROFILE_ORACLE_AUDIT_20260820.md` §4.
   - *Verbatim Values:* Support-aware log-wavelength RMSE to LeRoPE was `6.244` for the $w^{1/3}$ oracle vs `2.623` for Geo and `3.050` for EVQ-Cosh; projection onto the EVQ-to-LeRoPE direction was $\alpha = -0.957$ (lying in the opposite direction).
8. **Position-Dependent Non-Monotonicity:**
   - *Source:* `paper-2027/research/attention-aware-retrofit/analysis/ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md` §5.
   - *Verbatim Values:* 8K forward pass decomposes into `+0.001` NLL (positions 0–4095), `+0.092` NLL (positions 4096–8192 transition band), and `-0.039` NLL (final-512 tail).

---

## 2. Logic Chain

1. **Premise 1 (Observation 1, 2, 3):** Pretrained weights $(W_q, W_k)$ are intimately co-adapted to their native rotary spectrum $\Omega^{\text{native}}$ through learned content phases $\psi_m$ and amplitudes $A_m$. Replacing $\Omega$ under frozen weights destroys constructive interference and cannot be linearly compensated.
2. **Premise 2 (Observation 4, 6):** Under frozen weights, attempting to achieve $4\times$ extrapolation by shifting frequencies inevitably causes severe $1\times$ in-window degradation ($+3.98$ NLL) or fails completely ($0.0000$ RULER for `one_turn_floor_s2`). Zero-training static retrofits cannot bypass this ceiling.
3. **Premise 3 (Observation 5):** Attempting to avoid a target length via piecewise continuous-boundary slope breaks shift invariance $\theta(q) - \theta(k) = f(q-k)$ across the boundary, causing total retrieval collapse ($0.0000$ RULER).
4. **Premise 4 (Observation 6, 7):** All decoupled 1D scalar table selectors ($D^*$, $\kappa_{\text{att}}$, $w^{1/3}$, coverage, phase risk) fail to correlate positively with downstream performance because they ignore weight co-adaptation.
5. **Deduction / Synthesis:** Therefore, the only scientifically sound and practically deployable paradigm is the **Native-support pure-$z$ adaptation paradigm** (`INDEX.md` §6.2). In this framework:
   - $b_{\text{native}}, e_0, R$ are strictly fixed.
   - A theoretically derived $z_{\text{new}}$ is frozen prior to training.
   - Matched low-rank adaptation (LoRA) updates $(W_q, W_k)$ to co-adapt with $z_{\text{new}}$.
   - The matched $2\times 2$ matrix ($z \times \text{adaptation}$) isolates the causal effect of $z_{\text{new}}$ from generic LoRA capacity.

---

## 3. Caveats

1. **Model Scope:** Empirical evidence in the repository is primarily derived from OLMo-2-1.485B, Qwen2.5-1.5B, and 50M/151.9M research transformers. While architectural principles apply generally, exact numerical thresholds may vary on larger models (e.g., 70B+).
2. **Downstream Task Granularity:** Single-key needle-in-a-haystack tasks are saturated ($1.00$) at tested lengths, so evaluation must rely on multi-key and variable tracking tasks to measure real headroom.
3. **Preflight State:** As recorded in `INDEX.md` §0, the current project state is `RQ_LOCKED / PROTOCOL_NOT_FROZEN / NO_GPU_AUTHORIZATION`. No new training runs have been executed in this read-only phase.

---

## 4. Conclusion

1. **Zero-Training Frozen Retrofit is Closed:** Static table swapping under frozen weights exhibits an insurmountable table-shock ceiling and cannot simultaneously achieve $1\times$ retention and $4\times$ extrapolation without weight co-adaptation.
2. **Native-Support Pure-$z$ Paradigm is Authoritative:** Length extrapolation must be framed as interior allocation $z_k \in [0, 1]$ within the inherited native support $[e_0, e_0 + R]$, combined with frozen-$z$ matched weight adaptation.
3. **Serving Invariants are Strict:** Operational systems require a single static table and a single model across all context lengths. Routing hacks, dynamic gain adjustments, and piecewise non-stationary coordinate systems are strictly rejected.
4. **12 Closed Routes Codified:** All 12 falsified routes in `INDEX.md` §3.4 are traced to the fundamental fallacy of using decoupled static scalar selectors.

---

## 5. Verification Method

To independently verify the observations, derivations, and conclusions in this report:

1. **Verify Co-Adaptation & Table Shock:**
   - Inspect probe results in `paper-2027/research/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` §2.3.
   - Inspect crossing in `paper-2027/research/attention-aware-retrofit/results/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`.
2. **Verify Single-Table Failure & Boundary Slope Breakdown:**
   - View `paper-2027/research/attention-aware-retrofit/results/ZERO_PARAMETER_SINGLE_TABLE_RESULT_20260824.md`.
   - View `paper-2027/research/attention-aware-retrofit/analysis/ZERO_TRAINING_MECHANISM_AND_CEILING_20260826.md` §2.
3. **Verify 12 Falsified Routes:**
   - Check `INDEX.md` §3.4 and respective audit reports (`KAPPA_ATTENTION_MEASURE_AUDIT_20260820.md`, `LEROPE_PROFILE_ORACLE_AUDIT_20260820.md`, `RETROFIT_AXIS_FALSIFICATION_20260822.md`).
4. **Verify Navigation & Repository Invariants:**
   - Run the repository navigation regression tests:
     ```bash
     python -m pytest tests/test_repository_navigation.py
     ```
