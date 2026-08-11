# AGENTS.md — EVQ-Cosh NeurIPS 2026

This is a paper-and-reproducibility repository for Submission 11628, not a
general software project. The priorities are reviewer decision leverage,
scientific truth, provenance, anonymous hygiene, reproducibility, and GPU-cost
discipline.

This file is the only project-level agent instruction. Do not create a second
root `Agent.md`. Keep volatile experiment status and rebuttal numbers in their
canonical owners, not here.

Active rebuttal branch: `main_0726`. `main` and
`backup/main-restored-paper-20260726` are preserved pre-cleanup baselines.

---

## 1. Rebuttal principles

### 1.1 Objective and the two document classes

**The objective is one thing: maximize the probability of acceptance.**

Exactly two constraints bind it. Nothing anywhere in this file may be read as a
third.

1. **No fabrication.** No invented data, experiments, provenance, statistics,
   completion state, or citations; no claim of a result that was not run; no
   sentence that is false as written. Evidence used must be completed and
   traceable to an owner.
2. **No junk experiments.** Do not run experiments to pad a narrative, make it
   symmetric, or accumulate benchmarks. See §1.7.

Conference format, length, and anonymity rules are hard limits because
violating them forfeits the objective, not because they are separate values.

Every other rule in this file — every boundary, caveat, disclosure convention,
tier, and hedge — is **subordinate** to the objective. Where a rule appears to
require weakening our own case without constraint 1 or 2 forcing it, the rule is
wrong and the objective wins. A future agent that finds such a rule still
written here should delete it rather than comply with it.

**Two document classes carry two different disclosure standards.** Collapsing
them is what cost Submission 11628 a 2/3/4 opening panel: every reviewer
condition was substantively met, and the replies volunteered enough unrequested
failures that no reviewer moved.

| Class | Examples | Disclosure standard |
| --- | --- | --- |
| **Internal audit** | evidence owners, `01_REBUTTAL_PLAYBOOK.md`, status records, this file | Exhaustive. Record every negative, reversal, boundary and failed arm. This is how we avoid deceiving ourselves. |
| **Outward-facing** | reviewer replies, AC comments, official comments, camera-ready prose | Answer what was asked. A boundary enters only by the §1.2 admission test. |

An internal record that a result has a limit constrains **what you may claim**.
It does not obligate you to **volunteer that limit** to a reviewer who did not
ask. Those are two different acts and only the first is mandatory. Never treat a
negative record as a writing input for an outward-facing document; it is a
filter applied to a draft, never a source of new sentences.

Write for a busy human decision-maker. Lead with the strongest result that
directly answers the concern, and stop.

### 1.2 Question-first contract

- Use only concerns and stable IDs retained in
  `rebuttal/rebuttal_0723/00_REVIEWER_SCORES_AND_AC_METAREVIEW.md`.
- The retained package is author-pasted official OpenReview text. Reviewer
  `27bE` has a retained payload hash; the other reviewer/AC texts do not have
  independent payload hashes in this workspace. Preserve that provenance
  distinction.
- Compress the panel into three to five score-critical questions. Allocate
  response space by decision leverage, not equally by reviewer or by file size.
- Answer only the question asked. Do not introduce simulated-review concerns,
  internal-audit questions, implementation history, abandoned experiments, or
  future-work ideas unless they are needed to make the exact answer accurate.
- Calmly resurface overlooked submitted evidence. Never say that a reviewer
  “missed,” “ignored,” or should be embarrassed by an experiment.
- Do not use reviewer psychology, social pressure, or accusations as strategy.
  The leverage must come from relevance, evidence, and clear closure.

**Boundary admission test.** A negative result, in-window cost, longer-range
reversal, metric disagreement, or protocol limit may enter an outward-facing
document only if at least one of these holds:

1. **this** reviewer or the AC asked about that specific thing — quote their
   sentence in the draft before writing the boundary;
2. omitting it would make a sentence you are actually writing **false** — not
   merely incomplete, false as written;
3. it is needed to defeat a **stronger** attack you would otherwise face, and
   you can name that attack.

Disqualified reasons, all of which were used in the 11628 replies: “a reader
might discover it later”, “another reviewer asked about it”, “our internal
record says this result has a boundary”, “it feels dishonest to omit it”.

If a boundary fails all three tests, do not write it — and do not write the
positive claim in a form that would need it. **The fix for an unqualifiable
claim is to narrow the claim, not to append a confession.**

### 1.3 Score-move contract

Reviewer-stated score-change conditions are public commitments made in front of
the AC. They outrank every other use of response space.

- Grep every review for `score would increase`, `would be helpful`,
  `at a minimum`, `my evaluation would improve`, `if the authors`. List them
  verbatim before drafting.
- Each condition gets **one** answer: the single strongest number that satisfies
  it, then stop. Do not attach a second result, a contrasting metric, an
  additional model, or a related experiment the reviewer did not request.
- Meet a low bar literally. “Even a small-scale comparison would help” means
  small scale is pre-approved; do not apologize for it and do not over-deliver
  into territory that invites new questions.
- Never disclaim a path the reviewer opened. If they wrote “advantage **or**
  complementarity”, delivering neither, or arguing against one, is a
  self-inflicted loss.
- Correct a reviewer's factual error about the submission in the **first
  sentence** of the relevant answer, not buried at the end of a long section.
- Before sending, re-read each condition and ask: *reading only my reply, would
  this reviewer conclude the condition is met?* If they must hunt for it,
  restructure.

### 1.4 Evidence contract

Every external claim must map to:

1. a concern ID;
2. a direct answer;
3. a metric and protocol;
4. an evidence tier;
5. an exact seed scope;
6. a standalone owner or submitted-paper source;
7. its nearest material limitation, **known to the author and recorded
   internally**. Item 7 governs how far the claim may be pushed; it is not an
   instruction to print the limitation in the outward-facing text. Whether it
   is printed is decided by the §1.2 admission test.

Use these evidence tiers without collapsing them:

- `SUBMITTED`;
- `POST_SUB_RAW_HASH_BACKED`;
- `AUTHOR_CONFIRMED_NOT_PROMOTED`;
- `CONDITIONAL_OR_PROVENANCE_CONFLICTED`;
- `DESIGN_ONLY_OR_PENDING`;
- `NEGATIVE`.

Hard rules:

- A plan, script, checkpoint inventory, launch log, or historical trace is not
  a completed result.
- Prefer raw artifacts and canonical reports over filenames, comments, or old
  narrative summaries.
- It is legitimate to use the strongest relevant result from a declared
  protocol. State single-seed versus multi-seed exactly; do not splice
  incompatible protocols or selectively average runs.
- Say “same” only for fields verified identical. Say “matched” only when the
  intended scientific contract is genuinely matched.
- Separate NLL/PPL, teacher-forced routing or gold-token probability, strict
  autoregressive exact match, NIAH/RULER, downstream QA, task-family adaptation,
  and unseen-task transfer.
- NLL/PPL improvement does not prove generation, retrieval, or downstream
  capability.
- Submitted and post-submission evidence must remain visibly distinct.
- Do not use significance language without the required seeds, uncertainty,
  and statistical owner.

Reviewer-facing evidence synchronization:

- a directly cited rebuttal experiment needs a standalone owner under
  `rebuttal/rebuttal_0723/theory_results/`;
- the matching row in `01_REBUTTAL_PLAYBOOK.md` must agree on status, numbers,
  protocol, seeds, and wording;
- raw/hash promotion gates must be closed before a conditional number is used.

### 1.5 Known identity traps

Data-identity facts, not disclosure requirements. These say which number is
which; they never obligate outward-facing text. Getting one wrong is a
correctness error, and several are unrecoverable from filenames alone.

| Trap | The fact |
| --- | --- |
| Historical hybrid frequency tensors | Runs before the in-place EVQ patch aliased the Native reference; labels do not match realized hashes. No success or failure conclusion may be drawn from those runs. |
| OLMo query-gap adapter generations | The pre-repair `+100` adapter gives `95/100` @8K and `51/100` @16K on first-number exact, but `0/100` on strict raw-token complete-string-plus-EOS. The final `+100+32` adapter is the one at `100/98/60`. Never apply pre-repair numbers to the final adapter. |
| OLMo strict-retrieval `69/67` vs `49/48` | `69/67` is the original 8K n=100 set, which mixes within- and beyond-training-gap rows. `49/48` is a fresh set where every gap exceeds training support. Use each only for its own claim. |
| LLaMA temporal tables | Two tables use different adapters and controls: a 300-step matched LongAlpaca Native-LoRA vs EVQ-LoRA, and a 516-step RULER-family EVQ-LoRA vs untouched Native. Do not combine the first's control with the second, or call the second matched Native-LoRA evidence. |
| “Counterfactual” wording | The OLMo 300-step routing stage and the fresh EVQ-only LLaMA arm use pairwise counterfactual loss. The OLMo 13-task continuation and the matched LLaMA natural-LM/RULER studies do not. Name the exact arm. |
| Exact-range three-seed aggregate | `AUTHOR_CONFIRMED` only. Local raw/per-seed values and confidence intervals are absent. Do not call it raw-backed or statistically significant. |
| OLMo scratch comparison | Same initialization and scientific recipe; the released checkpoint came from the upstream distributed trainer and the EVQ branch from a single-GPU loop. Never state it as bitwise paired. |

### 1.6 Response structure

For each concern, write in this order:

1. **Direct answer** — answer the literal question in the first sentence, and
   make that sentence carry the shape of the result. If the result improves
   out-of-window and costs in-window, the verdict sentence says so; a verdict
   that its own table contradicts is worse than no verdict.
2. **Evidence** — give the smallest decisive result set. Smallest is a
   requirement, not a preference: every extra number is a new attack surface.
3. **Interpretation** — state exactly what the result establishes.
4. **Boundary** — **only if the §1.2 admission test passes.** Otherwise this
   step is skipped and the claim in step 3 is narrowed instead.
5. **Closure** — ask whether the evidence resolves the named concern.

Use result-first openings for scale and evaluation questions. When a reviewer's
premise about the submission is factually wrong, lead the answer by correcting
it with the submitted evidence — that is the highest-leverage sentence
available, and it belongs first, not last.

Decision logic:

- Positive/borderline-accept reviewer: preserve credited strengths and close
  the one or two remaining gates that block a higher score.
- Borderline-reject/theory reviewer: answer requested ablations and separate
  theorem, modeling assumption, scaling structure, empirical calibration, and
  trained-model evidence.
- High-confidence reject: follow stated score-change conditions in order. Make
  one clean early concession on any error that is genuinely ours — it buys
  standing to be firm everywhere else — and concede nothing further. Give them
  a route to change position that does not require admitting they were wrong:
  a quote from the work *they* cited, applying *their* standard, is worth more
  than any argument of ours.
- AC: summarize each decision-critical condition as answered, partial, or
  unresolved. Do not substitute more results for an unresolved novelty or
  control question.

### 1.7 Rebuttal experiment triage

Before proposing any experiment, write:

1. reviewer or AC concern addressed;
2. existing evidence;
3. smallest missing evidence;
4. smallest executable plan;
5. stop condition.

Classify the proposal as `required`, `optional`, or `unnecessary`. Do not run an
experiment merely to make a narrative symmetric, accumulate benchmarks, or
turn rebuttal into a second paper. Prefer owner/provenance repair when the
missing piece is evidence promotion rather than a scientific result.

### 1.8 Send gate

Do not call a package sendable until all of the following hold:

- every number has an owner, protocol, endpoint, and seed scope;
- every matched-control claim is truly matched at the stated scientific level;
- every capability claim uses a capability endpoint;
- every material in-range cost or longer-range reversal is **recorded
  internally against the claim it bounds**, and any instance that appears in the
  outward-facing text has passed the §1.2 admission test;
- pending/design-only work is absent from completed-evidence prose;
- no unsupported significance, universal-optimum, or SOTA language remains;
- partial evidence is not described as closing a concern;
- internal paths, identities, machine details, and process notes are removed;
- the response fits the current format and character limit.

**Hostile-reviewer scan — run on every outward-facing draft, per recipient.**

1. Highlight every sentence that concedes a failure, cost, reversal, or metric
   disagreement. For each, name which of the three §1.2 tests it passes and
   quote the reviewer sentence that triggered it. No quote, no sentence.
2. Read the reply as the recipient, in isolation from the other replies. Ask
   what they will remember. If the memorable content is our concessions rather
   than our answers, the draft has failed regardless of its accuracy.
3. Check every verdict sentence against the table beneath it. `Included, and it
   improves.` above a table showing a 2× in-window regression is the failure
   mode to look for.
4. Check that same-named metrics from different runs, adapters, or training
   branches are never adjacent and unlabeled. Two numbers from two branches
   under one header read as catastrophic variance.
5. Grep the draft: `narrower`, `not solved`, `we did not run`, `cannot rule
   out`, `one limitation is`, `it is possible that`, `we would expect`. Every
   hit needs a §1.2 justification or gets cut.

**Length discipline.** Response space is not free; it dilutes. The 11628
package sent roughly 40k characters across five documents, and every
condition-satisfied signal sat next to unrequested negatives. When the panel's
conditions are met, the correct reply is short. Cut in this order: cross-section
pointers, restatements of the opening verdict, protocol recitations already
implied by “matched”, hedges. Never cut a number, a control description, or a
concession that passed §1.2.

If an unsupported optional claim can be removed, remove it. If it is
decision-critical, mark the response as needing author input.

---

## 2. GPU experiment principles

### 2.1 Authorization and scope

- GPU use is experiment-only. A reviewer request, plan, script, checkpoint, or
  available machine is not launch authorization.
- Start training, inference, evaluation, or a paid instance only when the user
  explicitly authorizes that run or queue.
- Every GPU job must name the reviewer/AC concern, smallest missing evidence,
  exact command, output owner, budget, and stop condition.
- If the experiment cannot change the answer to a retained concern, do not run
  it.

### 2.2 Offline-first READY gate

Before paid GPU time, prepare and validate off GPU:

- immutable code/config hashes and the exact entry command;
- model/checkpoint identity and load test;
- dataset/split identity, data hashes, reader/tokenizer contract, and sample
  counts;
- method identity, realized frequency tensor, and expected tensor hash;
- optimizer, LR semantics, effective token budget, global batch, objective, and
  evaluation metrics;
- output paths, raw-result schema, manifest fields, free-space budget, cleanup
  plan, and shutdown instruction;
- syntax, unit tests, CPU/dry-run checks, and a no-GPU preflight receipt.

For paid GPUs other than the local RTX 5090, including RTX Pro 6000, require a
completed READY receipt before launch. Missing modules, data, scripts, model
assets, or evaluator contracts are reasons to stop the GPU session and prepare
offline, not reasons to debug while paying.

RTX 5090 exception: a bounded diagnostic may skip a pre-existing READY document
only when the required code and assets already exist. It still requires a named
concern, exact command, budget, free-space check, stop condition, and
post-launch receipt. Missing inputs end the probe immediately.

### 2.3 Scientific contract versus runtime details

Matched arms must preserve the broad scientific contract:

- model/checkpoint and intervention;
- dataset, split, data order where required, and effective training budget;
- objective and optimizer/LR semantics;
- evaluation samples, metric definition, and decoding contract.

Micro-batch, accumulation, activation checkpointing, compile mode, kernel,
allocator, dataloader workers, and evaluation batching are execution details.
They may differ when numerically valid and recorded. Do not rerun a completed
arm merely for cosmetic runtime symmetry.

Trainer differences must be reported when they bound causal attribution, but
they do not automatically erase a completed raw/hash-backed result. Never call
different trainer implementations bitwise paired.

### 2.4 Blackwell cost-first execution

Read `docs/overview/RTX5090_BLACKWELL_PROFILE.md` before changing a Blackwell
run. Its receipts are shape-specific, not universal defaults.

Starting candidates, subject to an actual probe:

- BF16 autocast;
- Flash-only SDPA;
- fused AdamW;
- TF32 matmul where scientifically valid;
- `expandable_segments`;
- persistent TorchInductor cache;
- the fastest verified compile/checkpoint configuration for the active shape.

Mandatory rules:

- Confirm the active architecture is compiled, BF16 is supported, and Flash
  SDPA is eligible.
- Never silently fall back to quadratic math attention.
- Optimize estimated total GPU seconds and cost to completion, including
  compile and evaluation overhead. Do not optimize VRAM occupancy percentage.
- Reuse a completed receipt for the same or sufficiently similar shape.
- If no receipt exists, run only the shortest discarded probe needed to
  establish memory fit, finite loss, operator eligibility, compile cost, and
  useful throughput.
- Stop probing when one configuration is clearly sufficient; do not run a
  cosmetic fixed grid.
- Replace probe throughput with completed-run throughput when available.
- Do not interrupt a healthy registered run for marginal tuning or a prettier
  memory-utilization number.

Training saturation does not imply evaluation saturation. Probe evaluation
separately. Length-aware batching, merged same-shape counterfactual forwards,
and reduced allocator clearing are allowed only after fixed-sample metric
parity is verified.

### 2.5 Launch and monitoring receipt

Immediately after launch, verify and record:

- GPU name, compute capability, PyTorch/CUDA versions, and compiled
  architectures;
- PID/process group and the exact command;
- precision, SDPA backend state, compile mode/cache, micro/global batch, and
  accumulation;
- model parameter count and trainable parameter count;
- first optimizer step, finite loss, throughput, peak memory, utilization,
  power/limit reason, and ETA;
- output directory, free space, raw-metric creation, and checkpoint cadence.

Do not report a run as healthy from process existence alone. A healthy receipt
requires at least the first real optimizer step and finite loss.

### 2.6 Stop, evidence, and cleanup

Stop or do not expand when:

- the protocol no longer answers the named concern;
- required assets or provenance are missing;
- loss is non-finite, the realized intervention hash is wrong, or the evaluator
  contract changes;
- the model lacks basic task capability;
- the screening result has no stable direction under its registered rule;
- the gain disappears under the required matched control;
- disk or runtime budget approaches the declared floor;
- the work is evolving into a new method or next-paper campaign.

On completion:

- preserve raw metrics, per-seed rows, manifests, hashes, environment/runtime
  receipt, and negative endpoints needed to bound the claim;
- promote the result only after standalone-owner and playbook synchronization;
- distinguish completed, failed, stopped, skipped, and unverified checks;
- shut down a remote/paid machine only when the user requested shutdown for
  that run, and verify the terminal state rather than assuming it.

Never rerun a completed arm solely because its runtime settings differ. Rerun
only for a scientific-contract mismatch, evaluator change, material numerical
drift, corrupted output, or an explicitly authorized new question.

---

## 3. Repository hard rules

### 3.1 `paper/` is immutable

Do not modify, create, delete, move, rename, format, compile, or regenerate
anything under `paper/`, including source, tables, figures, bibliography,
build artifacts, and `paper/main.pdf`.

- `paper/` is the restored submitted-paper tree from `994f618`, identical to
  submission baseline `cb7d83e`.
- Repository cleanup, rebuttal work, and branch reorganization must preserve
  the complete directory byte-for-byte.
- Reading and auditing are allowed.
- A paper change requires the user to explicitly name the file or paper scope
  and explicitly override this prohibition.
- A request to review, align, optimize, or write rebuttal material is not a
  paper-edit override.

### 3.2 Preserve user work and private layers

- Uncommitted work belongs to the user unless proven otherwise.
- Do not modify `internal/`, `results/`, `audit_v3/`, `audit_v4/`, `.codex/`, or
  `.claude/` without an explicit request.
- Do not commit checkpoints, caches, generated bytecode, build directories,
  private machine paths, author identity, credentials, or server details.
- Do not use ignored local raw evidence as public/reviewer evidence. Promote
  only sanitized, narrow artifacts after explicit authorization.

---

## 4. Scientific claim guardrails

Core identity:

> RoPE is also a finite spectral budget. EVQ-Cosh is a closed-form,
> zero-learned-parameter training-time frequency-grid allocation axis,
> complementary to positional-operator design and inference-time range scaling.

Do not reframe EVQ-Cosh as universal long-context SOTA, a replacement for
YaRN/LongRoPE/FMRoPE/DAPE, or a learned-PE replacement.

These are **claim ceilings**: they bound what may be asserted. None of them is
an instruction to print a caveat. Whether any caveat appears in outward-facing
text is decided solely by the §1.2 admission test.

| Topic | Must not claim beyond |
| --- | --- |
| Submitted YaRN | Repository-defined fixed-index smooth-ramp scaler unless a specific artifact verifies another implementation |
| DAPE/tuning question | The verified tuning budget and executed control from its owner; do not infer method identity or pure shape attribution beyond the artifact |
| Cosh | Uniqueness holds for the stated convex surrogate \(C_{\mathrm{app}}\), not for trained task loss |
| Finite \(\tau\) | \(\tau=c(\Pi)d_{\mathrm{eff}}/\sqrt{L_{\mathrm{train}}}\): theory supplies the scaling structure; the finite \(O(1)\) coefficient is empirically calibrated |
| \(\tau\) rule | Operating prior/basin selector, not a global or near-optimal prescription |
| Passkey | Teacher-forced NLL-gap unless an owner explicitly reports strict autoregressive exact match |
| FMRoPE | Range control and allocation shape are distinct tested knobs; do not claim replacement or universal dominance |
| RULER/NIAH | Do not present task-family adaptation as clean unseen-task transfer |
| Scratch comparison | Same initialization/scientific recipe; never say bitwise paired |

Primary submitted evidence remains primary; LoRA-8B, DiT, progressive training,
750M continuation, and other supporting rows remain supporting unless their
current owner and playbook explicitly promote them.

---

## 5. Mandatory routing

Before rebuttal work, read in this order:

1. `AGENTS.md`;
2. `rebuttal/rebuttal_0723/00_REVIEWER_SCORES_AND_AC_METAREVIEW.md`;
3. `rebuttal/rebuttal_0723/01_REBUTTAL_PLAYBOOK.md`;
4. `rebuttal/rebuttal_0723/theory_results/EVQ_COSH_REBUTTAL_PRINCIPLES.md`;
5. `rebuttal/rebuttal_0723/README.md`;
6. the relevant positive/negative ledger, standalone report, JSON/raw owner,
   and submitted-paper source.

Before a GPU run, additionally read:

7. the registered experiment specification/manifest and launcher;
8. `docs/overview/RTX5090_BLACKWELL_PROFILE.md` for Blackwell execution.

Source map:

| Need | Source |
| --- | --- |
| Retained reviewer/AC wording and stable IDs | `rebuttal/rebuttal_0723/00_REVIEWER_SCORES_AND_AC_METAREVIEW.md` |
| Current send decision and response selection | `rebuttal/rebuttal_0723/01_REBUTTAL_PLAYBOOK.md` |
| Method/theory/experiment boundaries | `rebuttal/rebuttal_0723/theory_results/EVQ_COSH_REBUTTAL_PRINCIPLES.md` |
| Reviewer-usable evidence status | `rebuttal/rebuttal_0723/theory_results/REVIEWER_USABLE_EVIDENCE_LEDGER_20260726.md` |
| Data-identity traps (which number is which) | §1.5 of this file |
| Frequency identity | `rebuttal/rebuttal_0723/theory_results/FREQUENCY_DEFINITION_MANIFEST.json` and `rebuttal/rebuttal_0723/experiments/geo_rope_contract.py` |
| Submitted claim/code mapping | `docs/overview/PAPER_CLAIMS_MAP.md`, `docs/overview/RESULT_PROVENANCE_MANIFEST.md`, and `paper_experiments/MANIFEST.json` |
| Canonical schedule API | `scripts/lib/rope/schedules.py` |
| Data/reproduction contracts | `docs/overview/DATA_PREPARATION.md` and `docs/overview/REPRODUCE.md` |
| Historical pre-rebuttal material | `rebuttal/pre_rebuttal/`; never the current action queue |

Do not copy a transient readiness label, remote status, or fast-changing result
inventory into this file. Read the current owner at task time.

---

## 6. Implementation and verification hygiene

- Facts come from code, configs, logs, raw/curated artifacts, and hashes.
- Before an in-place frequency/schedule patch, clone every tensor used as a
  pre-patch reference and assert the realized tensor against an independently
  reconstructed expected hash.
- Separate reviewer request, repository fact, proposal, completed evidence,
  and interpretation.
- Prefer minimal, local edits. Use `apply_patch` for manual file changes.
- Preserve unrelated worktree changes.
- Verify at the same level as the claim. Tests or HTTP success alone do not
  establish experimental completion or agent behavior.
- Before completion, report passed, failed, skipped, and unverified checks.
- Do not compile the paper as a validation shortcut.

Useful code gates, when relevant and available:

```bash
python -m py_compile \
  scripts/lib/rope/schedules.py \
  scripts/core_text_phases/run_evq_sweep.py \
  scripts/supporting_eval/eval_passkey_scratch.py \
  scripts/package_supplement.py

python -m pytest tests/test_rope_core.py -q
python scripts/validate_rebuttal_evidence_bundle.py
```

If the active Python lacks the required packages or language version, report an
environment limitation rather than a pass.

---

## 7. Git and delivery

- Work on `main_0726` and synchronize `origin/main_0726` before serious edits.
- `main` and `backup/main-restored-paper-20260726` are preserved baselines, not
  the active rebuttal queue.
- Do not commit, push, stage, switch branches, reset, stash, or delete branches
  unless the user explicitly requests that action.
- Never reset or checkout away user work.
- After staging, run `git diff --cached --stat`, `git diff --cached --check`,
  and a leak scan.
- Never archive the repository root for reviewer delivery. Use
  `scripts/package_supplement.py`.
- Reconfirm that `paper/` is unchanged before any commit.
- Keep commit messages terse and factual.

Default final report: changed files, decision-relevant outcome, validation
passed/failed/skipped/unverified, paper status, Git status, and remaining
author input.
