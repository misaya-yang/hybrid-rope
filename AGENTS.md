# AGENTS.md

## Research execution

Complete the requested result, including relevant verification and repairs.
Existing authorization carries across related follow-ups; a plan alone does not
approve paid GPU runs, downloads, remote changes, or publication.

- Test the research claim itself. Distinguish mathematical/CPU checks, proxy
  diagnostics, and actual model/task results; keep conclusions within the evidence.
- Choose the least costly comparison that distinguishes the relevant explanations.
  Reuse valid baselines and prior findings; preserve conditional gains and failures.
  Incomplete theory need not block an informative experiment.
- Check concrete execution risks, not a fixed preflight ritual. Trust assets the
  user has confirmed; endpoint/path changes alone do not justify rebuilding,
  rehashing, or rerunning. Keep bookkeeping out of healthy experiment execution.

## Context and evidence

- For a fresh session, use `index.md` to select one task route. The repository
  contains portable skills under `.agents/skills/`; use those project-specific
  workflows for manuscript editing, PDF regression review and experiments.
  Do not assume another machine has personal skills or previous conversations.
- Current manuscript state and author corrections are in `paper-2027/HANDOFF.md`
  and `paper-2027/research/AUTHOR_WORKING_CONTRACT.md`. The current research
  index controls priorities; a historical plan is not an active queue.

- Use root `index.md` when the task location is unclear; otherwise read the relevant
  file or experiment directly. Catalogs, old plans, and full handoffs are for
  specific provenance questions, not startup reading.
- Current research decisions belong in the plan linked from `index.md`. Dated
  queues, memory, and external proposals do not establish current priorities,
  live execution status, or new authorization.
- Preserve dirty work and evidence identity. Inspect dependencies before moving
  code or sources; record relocations. Report-backed claims and raw-row verification
  are different evidence levels; missing local artifacts do not invalidate a report.
- Keep navigation in `index.md`, introductions in `README.md`, and findings in
  their existing classified folders. Use document-relative links and root-relative
  registry paths; distinguish ignored local artifacts from Git-distributed files.
  Update the nearest index when an evidence owner changes; update the claim map
  when the manuscript claim changes.
- After navigation changes, run `python3 scripts/check_repository_docs.py`.
  Inventory refresh and other maintenance procedures are in
  [maintenance](docs/maintenance/index.md), when needed.
