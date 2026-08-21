# ICLR 2027 research index

This directory is the durable internal research layer for the active
`paper-2027/` manuscript. It is not submission prose. Raw experiments remain
with their canonical owners; this directory records the decisions, proofs,
audits, and evidence routing needed to write the paper without replaying the
entire NeurIPS rebuttal history.

## Read order

1. [`../../AGENTS.md`](../../AGENTS.md) — project rules, acceptance objective,
   evidence boundaries, and paper-work routing.
2. [`../HANDOFF.md`](../HANDOFF.md) — current manuscript/build/worktree state
   and the only active next-action queue.
3. [`ICLR2027_RESEARCH_SYNTHESIS_20260819.md`](ICLR2027_RESEARCH_SYNTHESIS_20260819.md)
   — implemented claim architecture and its decision record.
4. [`EXACT_RANGE_151M_3SEED_RESULT_20260820.md`](EXACT_RANGE_151M_3SEED_RESULT_20260820.md)
   — completed raw-hash-receipted three-training-seed fixed-support result;
   companion JSON owns the paper-facing aggregate values.
5. [`FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`](FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md)
   — canonical derivations, finite-K counterexamples, 50M 2x2 probe,
   base-only controls, and LeRoPE positioning.
6. [`audits/FULL_ROPE_CLAUDE_AUDIT_20260819.md`](audits/FULL_ROPE_CLAUDE_AUDIT_20260819.md)
   — verified contribution and known defects of the independent Claude
   full-RoPE audit.
7. [`audits/DEPENDENCY_SPECTRUM_CLAUDE_AUDIT_20260819.md`](audits/DEPENDENCY_SPECTRUM_CLAUDE_AUDIT_20260819.md)
   — verified contribution and limits of the independent dependency-spectrum
   audit.

## Current paper-facing decision

This architecture is implemented in the active manuscript. The paper's central
claim is no longer “EVQ-Cosh is the optimal RoPE table.” It is:

> Even at a fixed spectral range, the interior allocation of a finite RoPE
> table is a separately identifiable training-time design variable. It changes the full
> sin/cos subspace geometry and trained behaviour, while model weights co-adapt
> to the table they see during training.

Paper-facing notation: $x_k=-\log\omega_k=a+Rz_k$, where $(a,R)$ is sampled
support and $z$ is normalized interior allocation. Exact-range changes only
$z$; anchored and deployed Cosh share the same normalized $z$.

EVQ-Cosh remains a closed-form, zero-learned-parameter construction and a
controlled intervention for identifying the allocation axis. It is not the
universal solution of full-RoPE geometry, task loss, or extrapolation.

## Evidence routing

| Question | Canonical source | Outward role |
| --- | --- | --- |
| Full sin/cos geometry, canonical collision, stable-rank identity | `FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` | Main theory |
| Low-frequency collapse and softmax-centered limit | same report | Main theory |
| Exact post-hoc Q/K compensation obstruction | `../../rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md` | Main theory |
| Pure interior-allocation identification | `EXACT_RANGE_151M_3SEED_RESULT_20260820.md` + companion JSON + M4 owner | Main experiment; new three-seed result supersedes the unpromoted author-confirmed aggregate |
| Matched mature phase exposure | `../../rebuttal/rebuttal_0723/theory_results/OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md` | Main experiment |
| 50M weights-by-table interaction | canonical full-RoPE report and `../../scripts/analysis/attention_fisher_50m_probe.py` | Co-adaptation diagnostic |
| LeRoPE facts and Fixed-LeRoPE 63.6% | `../../rebuttal/rebuttal_0723/theory_results/LEROPE_CONCURRENT_WORK_NOTE_20260728.md` plus the primary paper | Related work |
| Multi-source RULER split | `../../nonuniform-alloc/RESEARCH_MEMO.md` and the RULER owner | Discussion/appendix diagnostic |
| Dependency-gradient spectrum pilot | dependency-spectrum audit record | Internal only unless promoted by a new owner |
| LeRoPE profile-oracle falsification | `LEROPE_PROFILE_ORACLE_AUDIT_20260820.md` | Internal only; $w^{1/3}$ does not predict the published profile |
| Attention-measure $\kappa_{\mathrm{att}}$ falsification | `KAPPA_ATTENTION_MEASURE_AUDIT_20260820.md` | Internal only; finite-swap Tier 1 fails the preregistered ordering gate |

## Maintenance rules

- Do not copy raw checkpoints, machine paths, ignored result trees, or private
  manifests into this directory.
- A research note may summarize a volatile source only when it records the
  exact source path, state, and hash.
- Do not make a second “canonical” synthesis. A new dated memo must explicitly
  supersede this one when an architectural decision changes.
- Every manuscript claim must still be checked against its raw/canonical owner;
  these notes are navigation and interpretation, not replacement evidence.
- Keep internal audits exhaustive. Write the paper for a busy human reviewer:
  result first, one central claim, and no audit/status vocabulary unless needed
  for truth.
