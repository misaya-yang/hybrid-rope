# AGENTS.md — Hybrid-RoPE

## Project conventions

- Keep process proportional to the task: retain only constraints that protect a
  concrete project requirement. Do not invent approval gates, handoff artifacts
  or extra validation without a decision-relevant purpose; simplify conflicting
  or redundant local rules instead of passing their burden to the user.
- Read `README.md` for project context and `paper-2027/HANDOFF.md` for live state.
  Use `INDEX.md` to locate only the files needed for the task.
- Keep core constraints here, detailed file routing in `INDEX.md`, and project
  details in `README.md`. Put research questions, numbers, equations, results
  and per-run plans in indexed documents. Do not create `Agent.md` or parallel
  root instructions/indexes; HANDOFF remains the sole live state record.
- `paper-2027/` is the active manuscript. Before restructuring it, read the
  current amendment in `paper-2027/REVISION_BRIEF.md` and the actual TeX/PDF.
  A supplied audit is an input to verify, not evidence or run authorization.
- Preserve the pre-slim archive on `main_0726`, including `paper/`: never edit,
  compile, regenerate or delete it. Read with `git show main_0726:<path>`.
- Inspect branch, upstream/divergence and dirty work before edits. Complete
  authorized code/docs work with a scoped Git commit after relevant verification,
  unless the user asks to leave it uncommitted. Stage explicit related paths and
  preserve unrelated work. Push when requested or covered by an established sync
  instruction; destructive history changes require explicit authorization.
  Record continuation context in tracked documents; do not invent a separate
  handoff archive or manual transfer workflow for ordinary Git-managed work.

## Research priorities

- Prioritize the project's own method toward SOTA and useful findings per GPU
  hour. Reuse published baselines and valid prior results; do not automatically
  reproduce papers, retrain opponents or fill complete comparison matrices.
- Add only checks/comparisons that can change the next research decision. Do not
  block promising work on complete theory or publication-level validation. Label
  cross-paper differences briefly rather than presenting them as matched runs.
- Reuse existing code, assets and receipts. Before a new scientific intervention,
  record a short bounded plan in its indexed protocol: question, discriminator,
  expected cost, outcome-to-action mapping and stop conditions. Keep it proportional
  to the experiment; a preparation milestone is not a research outcome.
- For a new method, derive a quantitative or directional prediction that differs
  from the relevant baseline, using the actual attention computation, fixed
  rotary-slot identity and complete prior deployment. A bounded mechanism argument
  is enough; universal optimality or a complete theory is not a prerequisite.
  Each iteration must produce a justified solution or a discriminator that changes
  a named research decision. An insufficiency proof must name its statistic and
  counterexample; it does not close all simple methods. Define any missing observable
  and what measuring it would change. Retire claims that cannot produce a decision.
- Use the smallest suitable model and frozen representative inputs to obtain that
  decision quickly. Reuse valid baselines and outputs; add controls, tasks, seeds
  or model sizes only when they can change the next decision. Measure end-to-end
  cost before a long run. Numerical qualification and completeness of a benchmark
  are separate from the evidence needed to decide whether a method is useful.
- Read author-supplied research collections with explicit source coverage and
  verify their assumptions against primary papers, code and artifacts. A source
  is input, not proof or execution authority. Preserve full model/table/gain/data/
  decoder identity when using a successful method as the starting point.
- Check proposed mechanisms against local failed runs and later corrections,
  including retained ignored receipts and archive owners. Deduplicate by actual
  intervention and run identity, not report title. Put a visible correction on
  superseded searchable claims; do not rebuild a rejected premise under new terms.
- Freeze the prediction and outcome-to-action rule before scientific GPU work.
  Do not silently relax either after seeing outcomes. Label any justified new
  decision prospectively and keep the original verdict. Missing information must
  be a named identifiable quantity, not a request for an unspecified larger sweep.
- Keep scientific predictions, practical acceptance thresholds and runtime checks
  separate. Distinguish a contradicted prediction from an untested condition, a
  gate miss from capability collapse, and retrospective explanation from prediction.
  Record effect sizes and mixed outcomes; retire only the contradicted claim.

## Own the experiment phase

- Interpret “开机”, “启动”, “继续” and similar instructions in the active research
  context. Within the authorized goal and budget, carry preparation, launch,
  monitoring, result checking and the next justified step forward. Do not stop at
  powering on a machine or wait for a second “start” when the intended experiment
  is already clear. Follow explicit requests to only change power state or pause.
- Authorization can cover a bounded research phase, not just one command or job.
  Reuse its goal, machine, total budget and stop conditions for necessary setup,
  downloads, qualified runs, ordinary fixes and analysis. Freeze each run before
  execution; do not require fresh approval for every script or routine choice.
  Ask only for a material unresolved decision, missing budget, expanded resource
  spend/scope or a serious blocker that available context cannot resolve.
- Prepare code, dependencies, model/data downloads and CPU checks before buying
  GPU time where possible. If the GPU is already on, promptly start ready useful
  work. Use no-GPU mode for substantial preparation delays; avoid stop/start churn
  between ready, in-budget jobs. When no justified authorized work is ready,
  release the idle GPU or switch to no-GPU mode within existing machine authority.
- While a GPU job runs, prepare the next justified inputs/commands and analyze
  completed outputs on CPU. Keep useful work ready without competing GPU processes
  or speculative sweeps. High utilization is not a reason to spend the budget on
  low-value experiments, nor to keep a finished phase's GPU idle.
- Fix ordinary software, dependency and execution faults autonomously within scope;
  verify the fix and resume the affected work while preserving failed receipts.
  Count all attempts against the same budget. Distinguish an engineering failure
  from a scientific stop: do not change frozen data, scoring, frequencies or
  thresholds after seeing results merely to obtain a favorable outcome.
  If a stale guard cannot express an authorized comparison, fix and verify the
  guard while preserving the old receipt; substituting another arm changes the
  scientific question and does not complete the requested comparison.
- A small experiment finishing is a decision point inside the research phase,
  not a handoff or automatic shutdown point. Analyze its result, select and
  prepare the next necessary step, and continue within the existing goal/resource/
  budget authorization without asking the user to say “continue” again. Do not
  redefine each job as a new phase to create extra approval checkpoints.
- Treat follow-up questions, corrections and status requests as updates to the
  existing goal unless the user explicitly pauses, cancels or replaces it. Resolve
  routine choices and apparent instruction conflicts from instruction priority,
  existing authorization and project facts before asking the user.
- Reports and status answers do not end the active work. Use bounded monitoring
  across waits. Return control only when the agreed phase objective is achieved,
  the budget is exhausted, the user explicitly pauses, or a serious unresolved
  blocker requires their decision. A partial result alone is not phase completion.
- HANDOFF owns the phase authorization, live process, cumulative budget/deadline
  and next action. Enforce timeouts and duplicate-launch protection in code, not
  only prose. Freeze model/data/code/config, actual arrays/gain, decoder/scorer and
  output identities; preserve raw rows, hashes, failures and exclusions. A new
  job, restart, checkpoint or session never silently resets the budget.
- Long jobs must survive a disconnected control session and expose an explicit
  stop path. Distinguish recovery snapshots from checkpoint selection: where
  implemented, save a clearly nonfinal snapshot at a completed-update boundary
  on interruption, while keeping the final evaluation checkpoint fixed. An
  operator stop is neither a scientific failure nor a completed training result.

## Evidence and manuscript changes

- Match each claim to its exact protocol and primary artifacts. Use the indexed
  research reference for metric, method and evidence-label definitions; neither
  proxy scores nor invalid/unresolved assays select or promote capabilities.
- Keep training/deployment tables, gain, model history, physical/phase length,
  scoring and uncertainty units distinct. Do not tune on exposed confirmation
  or sealed test outcomes, or generalize candidate failure into class closure.
- New/amended owners state status/date, question, assumptions/protocol, artifact
  identity, supported/unsupported claims and corrections. Update `INDEX.md` in
  the same change and leave a visible notice on superseded searchable claims.
- Validate proofs and prior-art assertions independently before manuscript use.
  The author decides claim promotion; a new narrative cannot manufacture evidence.

## Execution and verification

- Before scientific GPU work, review the actual experiment path: input construction,
  model/operator intervention, decoding/scoring, and launch/stop logic. Check it
  against the frozen question and an independent reference where needed. Resolve
  material implementation errors before launch; record the reviewed code state and
  remaining limits briefly in the existing protocol, not a new approval workflow.
- Review code first; smoke tests are not a substitute. Use one minimal runtime
  check for a new or changed execution path, then proceed to useful experiment rows.
  Reuse valid reviews and checks for unchanged paths. Repeat only for a relevant
  change, failure or unresolved concern; a checkpoint, arm or session change alone
  does not justify another preparation cycle.
- Record shared asset identities once in the existing manifest. Verify new,
  changed or transferred artifacts where mix-up/corruption matters; do not repeatedly
  hash unchanged models, data or receipts, or build parallel hash ledgers. Identity
  checks establish which bytes ran, not whether the code or hypothesis is correct.
- The personal PC supports code/docs, light CPU checks and local LaTeX. Canonical
  PyTorch/pytest, GPU, packaging and release validation belong on the work machine;
  do not recreate that environment here. Verify actual memory and attention
  backend; for Blackwell read `docs/overview/RTX5090_BLACKWELL_PROFILE.md`.
  Never silently fall back to quadratic math attention.
- Run only checks relevant to the change, from the repository root:

  Numerical solver checks must exercise a nontrivial improving case and compare
  against an independent calculation. A vacuous monotonicity check, convergence
  flag, hard-coded verdict or self-consistent formula is not evidence of the
  claimed optimum or a model mechanism. Once relevant checks pass, stop checking
  and continue the research decision; do not broaden testing without a concrete need.

| Task | Check |
| --- | --- |
| Documentation/routing | `git diff --check`; verify changed local links, anchors and archive paths |
| Work-machine code | `conda run --no-capture-output -n aidemo python -m pytest tests/<affected_test>.py -q` |
| Active manuscript | `bash paper-2027/compile.sh`; inspect rendered pages after layout changes |
| Authorized supplement | `python3 scripts/package_supplement.py --profile iclr2027`; verify the allowlist against this checkout before building |

- Keep identities, credentials, private/server paths, checkpoints, caches and
  ignored raw evidence out of reviewer-facing files; never archive the repo root.
- Report changed scope, verification limits, archive/Git state and required author
  actions. Distinguish local edits, tests, publication and acceptance.
