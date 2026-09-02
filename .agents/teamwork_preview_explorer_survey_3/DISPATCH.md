# Dispatch Log

## 2026-09-02T19:29:30Z
Perform a deep architecture and specification analysis for R4 (Deterministic Evaluator & Mock Test Suite) and R5 (Fresh-Theorist Guide & Guardrails) for the RoPE Theory Falsification Benchmark in `/Users/yang/projects/hybrid-rope/falsification_benchmark`.

Investigate and specify:
1. Deterministic Evaluator Specification (`evaluator/`):
   - CLI invocation syntax: `python -m evaluator --predictions <pred_path> --answers <answers_path> [--output <output_path>]`
   - Mathematical formulas and scoring algorithms for all 4 dimensions:
     1) Directional correctness (binary / sign match, e.g. +1 / -1 / 0)
     2) Calibrated probability distribution (Brier score or negative log loss on discrete hypothesis/outcome probabilities)
     3) Effect magnitude accuracy (MAE, relative error, or bounded numerical tolerance on quantitative metrics)
     4) Qualitative pattern alignment (categorical or structured pattern matching with partial credit)
   - Composite aggregate score computation (e.g., weighted average or normalized index in [0, 1]).
   - Output format (detailed per-episode JSON report + clean CLI table summary).
2. Mock Unit Test Suite Specification:
   - Synthetic test fixtures: perfect predictions (1.0), inverted/worst predictions (penalty / ~0.0), uniform baseline / random prior, partial credit cases, malformed schema rejection.
   - Clean exit codes and CPU-only deterministic execution.
3. Fresh-Theorist Instructions & Guardrails (`fresh_theorist_guide.md`):
   - Submission protocol, prediction schema (JSON format expected from theorist), blind evaluation workflow.
   - Strict guardrails: CPU only, no GPU, no new theory/architecture generation, stop upon benchmark delivery.

Deliver your comprehensive report to:
`/Users/yang/projects/hybrid-rope/.agents/teamwork_preview_explorer_survey_3/survey_evaluator_and_guide.md`
and write your completion `handoff.md`.
Send a message back to parent when done.
