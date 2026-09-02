## 2026-09-02T19:29:30Z

Task:
Perform a deep architecture and specification analysis for R2 (Decoupled Visible Packets & Hidden Ground Truth) and R3 (Automated Zero-Leakage Audit) for the RoPE Theory Falsification Benchmark in `/Users/yang/projects/hybrid-rope/falsification_benchmark`.

Investigate and specify:
1. Schema & Structure for `experiment_registry.json` and `experiment_registry.md`.
2. Format & Schema for `visible_packets/` (e.g. `visible_packets/ep_01.json` or markdown):
   - What prior established facts, baseline observations, theoretical priors, and exact experimental protocols (model architecture, data split, prompt setup, metrics to measure) are strictly included.
   - Verification that no outcome hints, directional hints, or post-hoc timestamps exist.
3. Format & Schema for `hidden_answers/` (e.g. `hidden_answers/ep_01.json` or markdown):
   - Quantitative metrics schema (e.g. delta NLL, F1 scores, retention ratios, absolute values).
   - Qualitative pattern observations schema (failure/collapse patterns, headwise dynamics).
4. Automated Zero-Leakage Audit Specification (`leakage_audit/`):
   - Exact programmatic audit methodology: token scanning, n-gram matching, regex for numerical leakage, directional outcome terms (e.g. "improved", "degraded", "collapsed", "succeeded", "failed", "outperformed", etc.), post-hoc timestamp leakage, cross-referencing against hidden answers.
   - CLI design for the audit script (`audit.py` or `leakage_audit.py`).
   - Audit report format (`audit_report.md` / `audit_report.json`).

Deliver your comprehensive report to:
`/Users/yang/projects/hybrid-rope/.agents/teamwork_preview_explorer_survey_2/survey_leakage_and_packets.md`
and write your completion `handoff.md`.
Send a message back to parent when done.
