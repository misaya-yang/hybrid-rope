# ICLR 2027 active handoff

- **Updated:** 2026-09-02
- **Role:** volatile Git, PDF, machine, validation, and author-action state only
- **Active manuscript:** `paper-2027/`
- **Immutable baseline:** `paper/`

Start with [`../AGENTS.md`](../AGENTS.md), then
[`../INDEX.md`](../INDEX.md). The chronological record is
[`research/history/TIMELINE.md`](research/history/TIMELINE.md); mature-checkpoint
owners are under
[`research/attention-aware-retrofit/`](research/attention-aware-retrofit/).
Those files own rules, durable routing, and history respectively. This file
does not restate the research agenda.

## 1. Current outcome

- The repository navigation and historical evidence layout have been
  reorganized by chronology and artifact role.
- Historical experiment reports are grouped under `docs/exp/YYYY-MM/`.
- Active research now separates `foundations/`, `evidence/`, `audits/`,
  `history/`, and `archive/`; the mature intervention programme retains its
  protocol-specific directory.
- No scientific result, manuscript claim, experiment protocol, or model
  artifact was intentionally changed by this reorganization.
- No training, inference, paid compute, or GPU work was run.

## 2. Git state

- Branch / upstream before the reorganization:
  `main_0726` / `origin/main_0726`.
- Baseline local HEAD / tracking SHA:
  `b50d5eba179cb8f22de9ac8feb110ec0f3233f41`; ahead/behind `0/0`.
- The working tree is intentionally dirty with the documentation and path
  reorganization. Nothing is staged, committed, or pushed.
- No pull, fetch, rebase, switch, reset, stash, branch deletion, remote edit,
  commit, or push was performed.

Verify branch, upstream, divergence, and worktree scope live before any later
Git operation.

## 3. Manuscript and PDF state

- Active PDF: `paper-2027/main.pdf`
- Active PDF SHA-256:
  `37aa6402a65d68b21909b0b3479c4e8edd811079e3922c2c1be915ddeab167e4`
- Active PDF body / total: 9 / 31 pages
- Immutable [`../paper/main.pdf`](../paper/main.pdf) SHA-256:
  `fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772`
- `paper/` and the active manuscript sources have no intentional changes from
  this reorganization. Neither manuscript was compiled in this session.
- The curated supplement remains to be source-synchronized and rebuilt on the
  work machine before upload.

## 4. Machine and authorization state

- No GPU or paid-compute run is active or authorized.
- The work machine owns canonical `aidemo` validation and final packaging.
- The low-configuration personal PC remains a documentation/planning host;
  `aidemo` is not
  expected here and must not be installed or recreated for this change.
- The previous remote compute instance was already shut down. No remote host
  was contacted during this reorganization.

## 5. Evidence limitations still open

- The 2026-09-02 Qwen natural-NLL/QA/source-use statistics remain internal
  because their remote raw JSON/JSONL owners were not recovered before the
  prior shutdown. Recover a surviving copy if one exists; do not silently
  rerun or promote those numbers.
- Repository reports and manifests are evidence routers, not substitutes for
  their raw owners. Preserve the claim ceilings in `AGENTS.md`.
- The current submission has not been rewritten around the latest research
  chain. Manuscript inclusion remains an author decision.

## 6. Validation receipt

Passed on the low-configuration personal PC:

- 83 CPU/standard-library tests: 22 repository-navigation, 3 paper-workspace,
  29 rebuttal-evidence, 23 rebuttal-protocol, and 6 falsification-evaluator
  tests;
- theory-falsification leakage audit: 16 episodes, `PASS`, 0 violations, 0
  warnings, with regenerated artifact hashes after path migration;
- all affected curated/benchmark/workspace JSON parsed successfully;
- 70 modified or newly routed Markdown documents checked with 0 broken local
  links;
- affected Python entrypoints passed `py_compile`; `git diff --check` passed;
- paper-code workspace source/hash validation passed, including refreshed
  digests for the moved-path message and one pre-existing stale schedule entry;
- `paper/`, active manuscript TeX/sections/appendix/figures/tables, `.agents/`,
  and `internal/` have empty diffs. PDF hashes remain those in §3.

Skipped/unavailable here:

- canonical `aidemo`/PyTorch/pytest validation and supplement packaging belong
  to the work machine;
- manuscript compilation was unnecessary because manuscript sources were not
  changed;
- no model, inference, training, remote host, paid compute, or GPU was used.

## 7. Immediate author actions

1. Decide which completed mature-checkpoint results, if any, change the active
   manuscript claim set.
2. Recover the missing 2026-09-02 raw owners only from an existing surviving
   copy; absence does not authorize a rerun.
3. Rebuild and validate the curated supplement on the work machine before
   submission.
4. Complete the final owner-by-owner number review and freeze author metadata.
5. Treat any new experiment, GPU run, OpenReview upload, commit, or push as a
   separate authorization.

For durable research priorities and closed routes, use `INDEX.md`; do not turn
this volatile checklist into a second agenda.
