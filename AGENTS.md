# AGENTS.md

1. **Test the requested outcome.**
   Do not substitute proxy metrics or intermediate milestones for the requested
   result. Keep positive and negative conclusions within what was actually tested.

2. **Reason from mechanism and evidence.**
   Ground hypotheses in the computation, mathematics, implementation, and observed
   results. Challenge assumptions with counterexamples and competing explanations;
   a complete theory is not a prerequisite for a useful experiment.

3. **Run decision-sufficient experiments.**
   Use the least costly comparison that can distinguish explanations or determine
   the next decision. Reuse valid baselines; preserve diagnostic value and keep
   checks proportional to concrete risks without inventing workflow gates.

4. **Use prior results critically.**
   Check previous results, failures, and corrections before repeating a direction.
   Do not repeat failed assumptions or generalize failures beyond their evidence.
   Successful methods are evidence, not immutable constraints.

Project-specific hypotheses, baselines, parameters, experiment plans, results,
and failure records belong in project documents, not here.

## Navigation and documentation maintenance

- Start with the root `index.md`. For manuscript work, follow its current paper
  handoff and evidence index; for a specific experiment, follow the local index.
- Keep navigation in lowercase `index.md`, directory introductions in
  `README.md`, and dated findings/plans in their classified folders. Update the
  nearest index when adding or changing an evidence owner.
- Keep core documentation paths stable across machines. Use document-relative
  Markdown links and repository-root-relative paths in source registries; do
  not use machine-specific absolute paths for navigation. Mark ignored local
  artifacts separately from files distributed through Git.
- Treat dated GPU states, queues, budgets, old handoffs, and external guidance as
  historical context. They do not replace the current user's scope or authorize
  new execution. Check an experiment's actual result owner before claiming it
  completed.
- Preserve dirty work and source identities. Before moving evidence or code,
  check path and hash dependencies; record relocations and verify navigation.
  Keep report-backed results distinct from raw-row verification and plans.
- Put project-specific priorities and new scientific decisions in the current
  research plan linked from `index.md`, rather than expanding this file into an
  experiment ledger. Validate documentation with
  `python3 scripts/check_repository_docs.py` after navigation changes.
