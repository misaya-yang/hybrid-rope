# Scope: Zero-Training RoPE Retrofit Multi-Role Research Audit

## Mission
Conduct a rigorous, auditable, strictly read-only multi-role joint research investigation and definitive final judgment on the zero-training RoPE retrofit problem for mature checkpoints:
Why is it impossible to simultaneously retain Native base capability and achieve Long extrapolation, and does a high-confidence solution direction exist?

## Architecture & Team Decomposition
1. **R1: Evidence Archivist** (`teamwork_preview_explorer`)
   - Directory: `.agents/teamwork_preview_explorer_r1_archivist_1/`
   - Scope: Complete read-only traversal of repository indices, evidence owners, and historical reports. Reconstruct timeline and causal chain. Produce Claim -> Evidence -> Evidence Strength table with exact file paths and numbers.
2. **R2: Mathematical Red Team** (`teamwork_preview_critic`)
   - Directory: `.agents/teamwork_preview_critic_r2_mathredteam_1/`
   - Scope: Adversarial audit of existing theorems, identities, bounds, compatibility arguments, Pareto claims, and phase interpretations. Distinguish algebraic identity, local approximation, bound, empirical regularity, and theorem. Construct minimal counterexamples.
3. **R3: Experimental Auditor** (`teamwork_preview_auditor`)
   - Directory: `.agents/teamwork_preview_auditor_r3_expertauditor_1/`
   - Scope: In-depth audit of experimental validity, confounds, S=2/4/8 failures, multiset ordering, head/layer specialization, gain tuning, and table-weight co-adaptation.
4. **R4: First-Principles Theorist** (`teamwork_preview_critic`)
   - Directory: `.agents/teamwork_preview_critic_r4_firstprinciples_1/`
   - Scope: Independent first-principles derivation starting from attention logit $s_{ij}(\Delta) = \sum_k \text{Re}[c_k e^{i \omega_k \Delta}]$. Determine the physical/geometric object that changing RoPE frequencies controls in frozen models.
5. **R5: Judge / Synthesizer** (`teamwork_preview_critic`)
   - Directory: `.agents/teamwork_preview_critic_r5_synthesizer_1/`
   - Scope: Cross-stream synthesis, resolving conflicts, executing 9 premature-stopping checks, and drafting answers to Six Core Questions A-F.
6. **Orchestrator Synthesis & Delivery** (`teamwork_preview_orchestrator`)
   - Scope: Verify findings, audit claims against AGENTS.md claim ceilings and locked nomenclature, produce final synthesis report, update handoff, and report to parent.

## Milestones & Status
| # | Milestone | Subagents | Dependencies | Status |
|---|-----------|-----------|--------------|--------|
| M1 | Initialization & Scoping | Orchestrator | none | DONE |
| M2 | Parallel Deep Investigation (R1-R4) | R1, R2, R3, R4 | M1 | IN_PROGRESS |
| M3 | Cross-Stream Synthesis & 9 Checks (R5) | R5 | M2 | PLANNED |
| M4 | Final Orchestrator Synthesis & Verification | Orchestrator | M3 | PLANNED |
| M5 | Final Deliverable & Handoff | Orchestrator | M4 | PLANNED |

## Rules & Guardrails
- Strictly READ-ONLY: No code, configuration, or paper modifications. Git status must remain clean.
- Zero GPU compute: CPU analysis only.
- Zero fabrication: All unbacked claims labeled `UNSUPPORTED BY REPOSITORY EVIDENCE`.
- Formatting contract: Assertions tagged with `[OBSERVED]`, `[DERIVED]`, `[HYPOTHESIS]`, `[UNKNOWN]`.
- All `[OBSERVED]` claims must cite file paths and numbers.
