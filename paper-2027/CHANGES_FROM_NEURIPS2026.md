# NeurIPS 2026 to the exponent-allocation manuscript

Internal comparison; excluded from the anonymous paper and source archive.
The historical manuscript in `../paper/` remains unchanged.

| Aspect | Earlier manuscript | Current manuscript |
| --- | --- | --- |
| Research question | EVQ-Cosh as a RoPE allocation method | How exponent distribution, sampled range, and learned weights jointly affect RoPE |
| Main evidence | Construction and model comparisons | Fixed-endpoint paired training, range-retarget reversal, and weights-by-table crossings |
| Theory | Convex Cosh criterion and operating rule | Full sine/cosine subspaces, canonical overlap, finite rank budget, slow-frequency limit; Cosh as an explicit construction |
| Mature-model design | Adaptation and broad method comparisons | Native-relative exponent displacements; fixed-support controls; static index placement; boundary-matched redistribution |
| Presentation | Method-first narrative | Controlled findings, mathematical characterization, then design cases |

Shared definitions, valid proof components, and earlier experiments remain where
they answer the current question. The Cosh solution is unique for its stated
criterion; the paper's empirical allocation finding does not depend on Cosh
being the uniquely best nonuniform family. The factorial comparison reports
matched exponential and strength variants, and the mature-model results retain
their in-window costs and model-dependent preferences.

Recent source checks recovered the historical 454M fixed-index smooth-ramp
operator and the 750M inclusive, endpoint-anchored Cosh table. These operators
are named by their actual formulas. The learned inverse-frequency comparator
is a learned table, not a relabeled DAPE result. LongRoPE's compact discussion
includes both dimension-wise scaling and its position threshold.

Current experiment identities and source hashes are in
`research/EXPONENT_CLAIM_EVIDENCE_MAP_20260909.md` and the accompanying source
index. Live review/build status is in `HANDOFF.md`; old reviewer scores and
research queues are not current scientific conclusions.
