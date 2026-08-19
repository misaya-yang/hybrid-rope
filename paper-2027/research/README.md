# ICML 2027 research index

This directory is the durable internal research layer for the active
`paper-2027/` manuscript. It is not submission prose. Raw experiments remain
with their canonical owners; this directory records the decisions, proofs,
audits, and evidence routing needed to write the paper without replaying the
entire NeurIPS rebuttal history.

## Read order

1. [`../../AGENTS.md`](../../AGENTS.md) — project rules, acceptance objective,
   evidence boundaries, and paper-work routing.
2. [`ICML2027_RESEARCH_SYNTHESIS_20260819.md`](ICML2027_RESEARCH_SYNTHESIS_20260819.md)
   — current claim architecture and the decision record for the coming rewrite.
3. [`FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`](FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md)
   — canonical derivations, finite-K counterexamples, 50M 2x2 probe,
   base-only controls, and LeRoPE positioning.
4. [`audits/FULL_ROPE_CLAUDE_AUDIT_20260819.md`](audits/FULL_ROPE_CLAUDE_AUDIT_20260819.md)
   — verified contribution and known defects of the independent Claude
   full-RoPE audit.
5. [`audits/DEPENDENCY_SPECTRUM_CLAUDE_AUDIT_20260819.md`](audits/DEPENDENCY_SPECTRUM_CLAUDE_AUDIT_20260819.md)
   — verified contribution and limits of the independent dependency-spectrum
   audit.

## Current paper-facing decision

The general claim is no longer “EVQ-Cosh is the optimal RoPE table.” It is:

> A finite RoPE table is a training-time coordinate system. Its full sin/cos
> subspace geometry bounds positional identifiability, while model weights
> co-adapt to the table used during training.

EVQ-Cosh remains a closed-form, zero-learned-parameter construction and a
controlled intervention for identifying the allocation axis. It is not the
universal solution of full-RoPE geometry, task loss, or extrapolation.

## Evidence routing

| Question | Canonical source | Outward role |
| --- | --- | --- |
| Full sin/cos geometry, canonical collision, stable-rank identity | `FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md` | Main theory |
| Low-frequency collapse and softmax-centered limit | same report | Main theory |
| Exact post-hoc Q/K compensation obstruction | `../../rebuttal/rebuttal_0723/theory_results/OLMO2_POSTHOC_FREQUENCY_TRANSPLANT_OBSTRUCTION_20260726.md` | Main theory |
| Pure interior-allocation identification | exact-range owner + M4 owner, routed through the synthesis | Main experiment |
| Matched mature phase exposure | `../../rebuttal/rebuttal_0723/theory_results/OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md` | Main experiment |
| 50M weights-by-table interaction | canonical full-RoPE report and `../../scripts/analysis/attention_fisher_50m_probe.py` | Co-adaptation diagnostic |
| LeRoPE facts and Fixed-LeRoPE 63.6% | `../../rebuttal/rebuttal_0723/theory_results/LEROPE_CONCURRENT_WORK_NOTE_20260728.md` plus the primary paper | Related work |
| Multi-source RULER split | `../../nonuniform-alloc/RESEARCH_MEMO.md` and the RULER owner | Discussion/appendix diagnostic |
| Dependency-gradient spectrum pilot | dependency-spectrum audit record | Internal only unless promoted by a new owner |

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
