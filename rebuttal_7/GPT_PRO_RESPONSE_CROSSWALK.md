# GPT Pro reviewer crosswalk

**Mode:** evidence audit and response preparation. **No new experiments** are represented as completed. Each item below identifies the strongest answer supported by tracked assets and the boundary that must remain explicit.

This crosswalk complements the Fable5 18-question ledger. It is the tactical index for the 17 questions in `REVIEW_COMMITTEE_FULL_V2.md`; the formal prose for Original Questions 1, 8, 11, and 14 remains in `rebuttal/REBUTTAL_RESPONSE_DRAFT.md`.

### GPT-Q1 — Shape and scale come from different models

**Disposition:** concede and clarify, not withdraw. The cosh shape is the exact minimizer of the stated broadband surrogate; the deployed `tau` scale is a separate stiffness--utility operating rule. The response should lower the theoretical claim to a conditional shape derivation plus an empirically selected basin, then emphasize the practical payoff: a closed-form inverse CDF, no learned parameters, and no inference-time operation. This aligns with F5-Q13 and the revised theory section.

### GPT-Q2 — Why this collision surrogate?

**Disposition:** defend conditionally. The surrogate is not a pointwise model of the oscillatory RoPE kernel and is not the full trained-transformer objective. Its role is a tractable allocation design objective; the exact-kernel checks report 24--92% collision-score reductions across 12 configurations. This functional validation supports the design choice but does not establish that realistic attention priors leave the same optimum unchanged. This is the F5-Q12 boundary.

### GPT-Q3 — Trained-attention curvature and effective length

**Disposition:** open limitation. No reviewer-grade measurement of trained `L_eff^J`, attention curvature, or the diffuse-attention approximation was recovered. The rebuttal should identify the operating law as a falsifiable approximation and avoid implying that Phase 16 measured this latent quantity. This maps to F5-Q8.

### GPT-Q4 — Sensitivity to the stiffness functional

**Disposition:** narrow the conclusion to basin selection. The reported scaling structure and the 99-run rank audit support a robust neighborhood, not a uniquely identified exponent, functional, or prefactor. The 0.465 sensitivity result and the deployed 0.500 exponent should be presented as close operating rules whose direct trained comparison remains open. This maps to F5-Q13.

### GPT-Q5 — Fisher forcing at large tau

**Disposition:** separate implementation from approximation. Training uses the exact deployed cosh allocation; it does not numerically truncate a Taylor series or require the Fisher-forcing residual to vanish. The pure-tether branch is a simple design choice, while a measured forcing diagnostic remains absent. Defend the empirical results without claiming the homogeneous branch is a complete attention model.

### GPT-Q6 — MLA effective dimension convention

**Disposition:** calibrated convention, not theorem. `K=d_rope/2` controls the quantized frequency pairs, while `d_eff=d_head` was the operating convention in the reported 432M MLA stress test. The raw-backed three-seed result validates allocation sensitivity under that convention; it does not identify the optimal convention. A direct alternative-tau ablation remains open, as recorded in F5-Q7.

### GPT-Q7 — Other non-geometric allocation shapes

**Disposition:** do not claim shape-family dominance. The current trained controls establish cosh versus the tested geometric and learned/operator baselines, not superiority over every monotone or non-geometric density. The safe contribution is that training-time allocation shape is a usable design axis and that EVQ-Cosh is one closed-form, zero-parameter instantiation with positive results in the reported regimes.

### GPT-Q8 — Tuned geometric base

**Disposition:** acknowledge the missing full tuned grid and defend the stated regime. The paper already contains unfavorable small-base regimes, and the raw-backed 151.9M base 10K/500K pilot shows that the direction is not unique to base 500K. Neither is a complete tuned-base control. The mechanistic claim is strongest in modern high-base, finite-channel settings where low-frequency channels are effectively inactive; it is not universal dominance over geometric RoPE.

### GPT-Q9 — Separately tuned YaRN

**Disposition:** preserve the factorial question. Primary I compares Geo and EVQ under the same fixed YaRN scale to test substrate--range complementarity. It is not a best-tuned YaRN leaderboard, and the rebuttal must not imply otherwise. The seedwise matched-scale interaction remains the supported result; separate tuning is an unclosed extension.

### GPT-Q10 — Matched seeds for Primary II

**Disposition:** keep the printed contrast seed-scoped. The Geo/DAPE/fixed-EVQ comparison is the retained seed-42 protocol. Fixed EVQ tau=5 has recovered seeds 42/137/256, but matched Geo and DAPE seeds 137/256 were not found. The L=256 three-seed assets are supporting protocols and cannot be substituted for the L=128 comparison. This is F5-Q3.

### GPT-Q11 — Teacher-forced retrieval versus AR exact match

**Disposition:** resolved from the tracked Primary-I payload. At 8K, Geo+YaRN has 61.3% mean teacher-forced NLL-gap retrieval but 0.0% autoregressive exact match for seeds 42, 123, and 7. EVQ+YaRN has 100.0% teacher-forced retrieval and 58.0% mean AR exact match, with seedwise values 58.0%, 18.0%, and 98.0%. Report both endpoints and the wide seed range; do not relabel teacher-forced retrieval as generation accuracy. This closes F5-Q9 without a rerun.

### GPT-Q12 — MLA saturation and the 1B reversal

**Disposition:** do not use the single-seed 1B row as durability evidence. It changes training length, data/schedule, and budget relative to the three-seed 500M/8K primary anchor. It therefore shows schedule sensitivity rather than token-only saturation. The 500M three-seed result remains primary; a replicated matched continuation would be needed to upgrade the saturation claim. This maps to F5-Q11.

### GPT-Q13 — Isolating channel scarcity

**Disposition:** qualitative support only. The tracked 125M within-MLA pilot reports a larger direction at `d_rope=16` than at `d_rope=32`, but it is single-seed and includes a weak-baseline caveat. Use it to show that an isolation attempt exists, not as a quantitative law. The 432M three-seed MLA aggregate remains the anchor. This maps to F5-Q7.

### GPT-Q14 — Figure/table inconsistencies

**Disposition:** admit and correct both provenance errors. QuALITY now uses only the full `n=2086` Gold-NLL aggregate; the obsolete 200-sample accuracy pilot and its 32K point are excluded. The multiscale figure must use the 454M three-seed FineWeb-Edu value rather than the separate progressive-training value. These are presentation corrections; the rebuttal must distinguish corrected source-of-truth values from the originally submitted visual.

### GPT-Q15 — Contents of the 99-run sweep

**Disposition:** make the run accounting explicit. `data/curated/phase16_99run_manifest.csv` contains 99 sanitized run rows: 45 pilot and 54 confirm runs across the reported configurations, metrics, and inverse-frequency hashes. It supports the rank/basin audit. It is a run manifest, not a checkpoint archive and not evidence that every row has full historical logs.

### GPT-Q16 — Manuscript-level reproduction specification

**Disposition:** answer with concrete entrypoints. `docs/overview/REPRODUCE.md`, `docs/overview/DATA_PREPARATION.md`, the canonical schedule, primary/supporting runners, curated expected aggregates, and the supplement packager define the public rerun path. `paper_experiments/` now provides one manifest-driven code index. Historical checkpoints improve provenance but are not logically required to rerun from public data.

### GPT-Q17 — Practical long-context benefit

**Disposition:** lead with implementation and the generation result, not downstream SOTA. EVQ adds no learned parameters or inference-time operation, composes with matched-scale YaRN, and yields a material 8K AR exact-match separation in Primary I. QuALITY accuracy remains inconclusive and LoRA/video rows remain supporting. Claim a low-deployment-cost allocation intervention, not measured FLOP, latency, KV-cache, or universal task-accuracy gains.

## Final response discipline

- Lead with the four highest-value closures: AR exact match, corrected QuALITY source, explicit shape--scale separation, and high-base scope.
- Keep matched-scale, seed count, evidence tier, and protocol boundaries in the same sentence as each result.
- Do not promise new results inside the response window unless a completed artifact passes the provenance gates.
- Treat GPT-Q3, Q6--Q10, and Q12--Q13 as bounded limitations or follow-up controls, not as experiments already closed.
