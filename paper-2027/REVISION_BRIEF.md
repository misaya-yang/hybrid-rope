# ICLR 重构契约

- **更新 / 状态：** 2026-09-07；当前研究方向与论文重构边界。实际结果由 indexed owner 管理，实时状态与恢复入口见 [HANDOFF](HANDOFF.md)。
- **问题：** 结合 MrRoPE、CoPE 和本方工作，得到有实际价值的频率构造，再用结果决定论文贡献。
- **来源：** 作者持续纠正、指定 9/6 cross-audit 和 9/7 scale-transport 提案；外部报告是设计输入，不是科学证据或运行授权。
- **替代关系：** 下方旧 v5/v4 的阶段排序、对手微调默认、冻结日期和操作建议仅为历史，不是当前队列。另一实验的“v5”也不属于本项目当前运行规范。

## Current reconstruction contract

1. 优先本方方法与 GPU 决策价值；复用公开对手结果和已有有效证据，需要时才补最小比较。
2. 从有效 MrRoPE 分配继续改进，吸收 CoPE 深尾机制与本方分配经验；不返回 Cosh 曲线搜索，不要求先完成通用最优理论。
3. 长期覆盖冻结部署、轻量适配、从零训练；当前实际工作是 Qwen 冻结方法研究，不能把未做的阶段写成已有贡献。
4. 论文以实际有效构造、能力结果和可检验解释为中心。小面板、词面 F1、算子界或缓存改善不足以单独支持 SOTA/Oral 主张。
5. [当前研究主线](../docs/research/ROPE_FREQUENCY_UNIFIED_PLAN_20260907.md)定位方法关系；[实际协议与结果](../docs/research/ROPE_SCALE_TRANSPORT_PILOT_20260907.md)给出已完成范围与失败解释。由结果推进后续研究，不因一次小实验结束就重新交接。
6. 本次文档整理没有重写 TeX/PDF。决定论文主张前再对齐有效 owner 和真实稿件；不要让旧阶段清单自动触发训练或补实验。

## 原始设计历史

以下内容保留早期决策背景与原有锚点，仅供追溯；当前使用上面的契约及 HANDOFF。

<details>
<summary>展开历史 v5 / v4 规划，不作为执行指令</summary>

# REVISION BRIEF v5 — major reconstruction planning

> **Latest scope correction, 2026-09-07:** do not fine-tune MrRoPE/YaRN by
> default. They remain frozen references; adaptation is the project's own route.
> The earlier E2 three-arm full-adaptation proposal is withdrawn. See the
> [current experiment protocol](research/CROSS_AUDIT_EXPERIMENT_PROTOCOL_20260907.md).

> **Author priority amendment, 2026-09-07:** the current objective is one
> frequency-design method spanning from-scratch training, light adaptation and
> zero-training deployment, challenging the appropriate strong baselines in
> each regime. See the [unified plan](../docs/research/ROPE_FREQUENCY_UNIFIED_PLAN_20260907.md)
> and [E0/E1 ROI review](../docs/research/ROPE_FREQUENCY_LUNA_ROI_20260907.md).
> This replaces the earlier proposed stage priority, not completed evidence or
> frozen execution contracts. No new training or queue change is authorized by
> this amendment; Cosh is retained as historical evidence, not a new sweep target.

- **Status/date:** 2026-09-06; author-requested documentation and planning reset.
- **Question:** what reconstruction can establish useful finite RoPE allocation
  under explicit training/deployment budgets and credible Native/long evaluation?
- **Source:** [supplied cross-audit](research/external-reviews/ROPE_ICLR2027_CROSS_AUDIT_20260906.md),
  SHA-256 `eade4043ca4481a0f2f7a59da9ec1f5e8172808ee39d3890c745273ff688824a`; imported byte-for-byte. Its attachments and server
  receipts are not thereby verified.
- **Supported use:** revision framing, work ordering and protocol-gap identification.
  **Unsupported use:** new result claims, proof/novelty certification, or compute,
  restart, publication and submission authorization.
- **Supersession:** this amendment replaces v4's frozen-paper/default experiment
  priorities. It does not supersede completed result owners. The prior v4 text is
  retained below as history; its frozen design, automatic next steps and earlier
  two-route restriction no longer control this author-requested reconstruction.

## Reconstruction contract

The project overview belongs in [README.md](../README.md), file routing in
[INDEX.md](../INDEX.md), operational constraints in [AGENTS.md](../AGENTS.md),
and live permissions/state in [HANDOFF.md](HANDOFF.md).

The proposed paper should connect explicit pretraining/adaptation/deployment
tables to a matched strong-baseline comparison, finite-budget explanations and
actual generated-task/Native outcomes. Keep existing support/allocation controls
and bounded Cosh derivations at their valid scope; do not assume they establish
the revised novelty or system-level method advantage.

## Evidence reconciliation before implementation

| Input or issue | Available local lead | Required check / current boundary |
| --- | --- | --- |
| Supplied audit | Exact imported Markdown above | Recommendations only; audit P1/P2/P4 attachments are not byte-matched local owners |
| Recent execution | [Round12 report](claude_code_workspace/round12_20260906/REPORT_ROUND12_20260906.md), [Round10/11 reports](claude_code_workspace/reports/) | Reported outcomes and pauses; raw server artifacts/process state unverified this turn |
| YaRN identity | [Y2 builder](claude_code_workspace/round12_20260906/code/build_y2_canon.py), [pinned equation utility](../scripts/lib/rope/official_yarn.py), [parity tests](../tests/test_official_yarn_parity.py) | Different ramp/amplitude conventions are visible in code; qualify a specified implementation and compare executed tensors before claiming baseline fidelity |
| Z/M/Native identities | [Round12 table builder](claude_code_workspace/round12_20260906/code/rope_tables.py) | Manifest/array/gain readback, pair layout, config and checkpoint identity; a familiar label is insufficient |
| Existing scratch checkpoints | [fixed-support result and receipts](research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md) | Locate exact weights and code on the work machine; the result document is not checkpoint availability proof |
| Old versus strict scoring | [Round12 scorer](claude_code_workspace/round12_20260906/code/scoring.py), [generation contract](../scripts/lib/rope/generation_contract.py) | Compare saved outputs on identical prompts/tables/templates; keep official, semantic, full-answer/EOS and row/group endpoints separate |
| Full adaptation versus LoRA | [Round12 trainer](claude_code_workspace/round12_20260906/code/track_b_train_v2.py), [registered constrained trainer](../scripts/train/train_single_table_native_constrained.py) | Inspect actual trainable parameters, loss/replay normalization, optimizer, token and time budgets; no code-equivalence assumption |
| New CPU theory helpers | Audit §11.2 names `rope_codex_revision`, `verify_theory.py`, `select_allocation.py` | Not found in the searched checkout/adjacent Downloads names. Obtain the actual package and hash before reproducing its claimed checks |

The supplied file's P2 synthesis is not a verified alias for the local Round12
report. Record missing attachments and identity differences; do not fill them
with inferred experiment history. Existing source code is a candidate apparatus,
not an automatic valid assay.

## Proposed stages and decision value

These stages organize future exact protocols. None is a launch command or an
approved allowance. Preserve a complete matched comparison and confirmation
budget before broadening the matrix.

| Stage | Comparison / discriminator | Outcome-to-action mapping |
| --- | --- | --- |
| E0 | Reconcile baseline arrays/gain, templates, scoring, controls, data and complete-step memory/throughput | Identity or control mismatch: repair and freeze protocol. Qualified setup: cost a matched comparison. GPU probes require separate approval |
| E1 | Existing Geo/Cosh checkpoints under identity and specified strong transforms; mature frozen Native/credible YaRN/MrRoPE/current Z | Benefit survives: test learning/budget explanation. Benefit disappears or reverses: narrow the claim and choose an independent discriminator; do not assume new scratch training will rescue it |
| E2 | Proposed mature-model full-parameter paired comparison against credible opponents | Complete the preregistered budget/repeat design. Candidate loses: candidate-scoped negative. Long improves but Native fails: consider a separately frozen Native-constraint increment |
| E3 | Same-table/data/token-milestone LoRA bridge for the main pair | Compare achieved capability and cost; a failed low-rank recipe does not prove all-rank impossibility, and full adaptation is not guaranteed to succeed |
| E4 | Conditional scratch/support-range strengthening or an independent predicted boundary | Purchase only the comparison justified by E1/theory gaps; preserve tuned geometric and applicable learned-frequency baselines with fair selection cost |
| E5 | Independent exact model and task/document confirmation after selection freezes | Confirm only at the actual tested scope; exposed development instances cannot be relabelled as independent confirmation |

The external audit proposes a total 100 GPU-hour allocation. It is neither
measured runtime nor user authorization. Each launch needs actual machine
capabilities, throughput including replay/save/evaluation, complete paired
budgets, repetition reserve and stop/exit rules. Do not import a fixed large-token
run or restart the old continuation chain merely because its script exists.

The audit proposes a specific mature-model identity, physical training length,
full-parameter regime and low-rank bridge. These must survive E0 asset, protocol
and cost checks before becoming an exact run contract. No defaults are silently
substituted and no missing loss/data mixture is guessed.

## Theory and evaluation admission

- Check prior-art scope against the actual cited versions before claiming novelty
  over MrRoPE, LeRoPE, AdaRoPE, restoration distillation or data-scale work.
- Treat finite-feature learning/deployment risk, decision-KL bounds and rank
  repair as proposed analyses under their stated assumptions. They are not
  already validated explanations of a full Transformer or guaranteed selectors.
- New expensive curves need a prospective discriminator; existing strong-baseline
  comparisons can answer the method-value question without first predicting a winner.
- Define the final deployment table/gain and prefill/decode policy; run Native
  evaluation under that same declared policy. Record physical and phase exposure,
  evidence distance, distractors and model-native window separately.
- Preserve historical Native thresholds with their old protocols. Freeze the
  new confirmation margin before outcomes; do not inherit, loosen or replace it
  silently. Report Native text, task, format and termination strata separately.
- Keep complete-answer/EOS and official task metrics together; permit legitimate
  answer variation under a frozen semantic scorer. Preserve paired document/
  prompt/world groups and report training-seed variability separately.
- Count selection, training, checkpoint selection and evaluation cost. Repurpose
  neither selected maxima nor exposed confirmation pools as unbiased confirmation.

## Manuscript reconstruction and acceptance

The intended structure is: practical question and validated main comparison;
training/deployment objects and closest work; bounded theory and explicit
construction; matched experiments; analysis, failures and limitations.
Move supporting breadth and auxiliary algebra according to their contribution
to that argument. Rebuild the outline before rewriting sections.

The resulting story is conditional: static-table method value, learning/budget
value, or a narrower controlled/negative result if the strong comparisons do not
support the first two. Choose from validated evidence, not the desired outcome.

Before TeX changes, map each proposed claim/figure to an exact owner and identify
what is retained, rewritten, moved or withdrawn. Before scientific promotion,
resolve affected protocol/identity/uncertainty issues and obtain the author's
claim decision. Before release, rebuild and inspect the actual PDF, validate
anonymity/references and reconcile the curated package with archive-only inputs.
Official submission dates require a live venue check before external action.

---

## Historical v4 — superseded planning and narrative defaults

The text below preserves the 2026-09-04 brief and its original section anchors.
Use it only to interpret that prior scope; the v5 amendment above governs the
current reconstruction. Its result references retain their own evidence status.

## REVISION BRIEF v4 — September 2026 manuscript and research coordination

**Issued:** 2026-09-04
**Milestones:** internal abstract and author-metadata freeze on 2026-09-17;
official abstract deadline on 2026-09-18 at 11:59 PM AoE; full-paper deadline on
2026-09-25. Recheck the official pages immediately before each submission.

## Role and authority

This is the durable scope for the September manuscript iteration and its
coordination with active research. It defines the intended manuscript outcome,
research-to-manuscript admission boundary, and freeze gates. It does not itself
authorize manuscript edits, compute, Git operations, or upload. It intentionally does
**not** record current hashes, pass/fail state, worktree state, completed tasks,
or a live action queue.

- [`HANDOFF.md`](HANDOFF.md) is the only live state and action queue.
- [`../AGENTS.md`](../AGENTS.md) owns rules, claim ceilings, compute and Git
  discipline.
- [`../INDEX.md`](../INDEX.md) owns durable theory, evidence, code, closed-route,
  and research-agenda routing.
- [`NARRATIVE_GUIDE.md`](NARRATIVE_GUIDE.md) owns manuscript-level narrative
  discipline.
- Every fact, number, protocol identity, and uncertainty statement remains owned
  by the canonical source routed through the index.

This v4 supersedes v3 as the September scope reference. The August A/R ledger, panel
recommendations, simulated reviews, and model-review journal are historical
inputs only. No item from them is inherited automatically.

## 1. Outcome

Deliver the strongest truthful version of the **current paper**, not a replay of
an earlier review cycle:

1. freeze a submission-ready title, abstract, author roster, and author metadata
   internally on 2026-09-17;
2. submit the official abstract by 2026-09-18 at 11:59 PM AoE;
3. submit a fully verified paper and anonymous supplement by 2026-09-25.

The September pass starts from the current `paper-2027/` manuscript recorded in
the handoff. Proposed edits must identify a defect or missed opportunity in that
current source/PDF and explain how fixing it changes reviewer understanding,
technical credibility, or submission validity.

## 2. Current paper contract

The paper has one identity:

> A finite RoPE table decomposes as
> $x_k=-\log\omega_k=a+Rz_k$: sampled support $(a,R)$ and interior allocation
> $z$ are distinct design coordinates. Fixed-support intervention identifies
> $z$ as independently consequential, target-aware support retargeting identifies
> the second interacting coordinate, and exact full-sin/cos geometry exposes the
> finite spectral budget. EVQ-Cosh is one analytic construction on this object;
> frozen, adapted, and from-training results establish its behavioural reach.

Preserve the current evidence logic:

- **the decomposition leads:** support--allocation identification is the paper
  identity, not another name for changing frequencies or for EVQ-Cosh;
- **fully frozen zero-training remains the strongest practical consequence:**
  model-relative derived and coarse allocations own the mature no-update result;
  neither is EVQ-Cosh;
- **matched adaptation follows:** EVQ-Cosh under matched low-rank adaptation owns
  protocol-specific length transfer, probability, routing, and causal source-use
  results;
- **from-training/co-adapted evidence closes the loop:** anchored EVQ-Cosh owns
  the three-seed fixed-support identification, while MLA, 750M, the existing
  1.485B comparison, and video-DiT retain their separate persistence, scale, and
  modality roles;
- the fixed-support interventions identify `z` as a real causal variable, but
  the reviewer-facing claim concerns the tested structured schedules rather than
  arbitrary `z` perturbations;
- systems breadth and controlled identification are complementary, not competing
  narratives;
- static geometry diagnoses the positional basis but does not rank trained-model
  quality;
- EVQ-Cosh is unique only for its stated convex surrogate and is not the frozen
  derived or coarse allocation;
- support and allocation are distinct but interacting coordinates;
- FMRoPE is the support-versus-allocation causal control and may use `L_target`
  in its intended target-aware range-selection setting;
- LeRoPE appears in Discussion as attributed learned-allocation evidence that
  non-geometric allocation can improve in-window behaviour, not as a matched
  comparator or validation of EVQ-Cosh;
- the current manuscript's from-training evidence line stops at the completed
  1.485B comparison; later research requires a new owner and author promotion
  decision before it changes that claim.

## 3. September revision scope

### P0 — submission validity and scientific integrity

- Recheck every reviewer-facing number, endpoint, seed/unit label, method name,
  and protocol interpretation against its canonical owner.
- Resolve any source/PDF, caption/table, citation, anonymity, or supplement-route
  drift that could invalidate or misstate the submission.
- Recheck the live venue policy, author-profile requirements, reciprocal-review
  requirements, author limits, and dual-submission branch before the relevant
  freeze.

### P1 — reviewer path

- Preserve the 30-second path: support--allocation decomposition → fixed-support
  identification and target-aware retargeting → exact spectral-budget geometry
  → separate EVQ-Cosh construction → frozen/adaptation/from-training consequence.
- Improve the title, abstract, first-page framing, figures, and paragraph order
  only when the current version leaves a material ambiguity or buries decisive
  evidence.
- Lead with the scientific object and keep the strongest result visible with the
  nearest scope needed for truth. Do not turn the paper into a response ledger,
  method tournament, or limitations inventory.
- Keep zero-training prominent as a practical consequence, not as the paper
  identity. Do not present fully frozen, adapted, and from-training routes as
  three equally weighted headline methods.
- Replace lower-leverage material when space is needed; do not stack new prose or
  fill pages for their own sake.

### P2 — reproducibility and release clarity

- Keep protocol definitions, estimator units, task lists, recipe details, and
  supplementary evidence routes complete and mutually consistent.
- Preserve the curated supplement boundary and the immutable `paper/` baseline.
- Keep validation receipts out of this brief; record them only in the handoff.

### Out of scope

- No wholesale section reorganisation without a defect in the current reviewer
  path.
- No abstract, Figure 1, or contribution structure centred on a benchmark
  leaderboard, “we also change frequencies,” EVQ-Cosh as the paper identity,
  arbitrary `z`, or unrelated counterexamples.
- No relabelling of the frozen derived/coarse allocations as EVQ-Cosh, and no
  statement that makes EVQ-Cosh the owner of the frozen zero-training result.
- No portrayal of target-aware FMRoPE's use of `L_target` as an unfair baseline;
  fixed-support identification and target-aware deployment are different
  questions.
- No use of LeRoPE as mechanism validation or a matched comparison; its role is
  attributed learned-allocation evidence for in-window improvement.
- No revival of old comparison tables, evidence-hierarchy ledgers, defensive
  clauses, stale section locators, or panel wording merely because they appeared
  in an earlier plan.
- No change to a claim or number from a review memo without returning to the
  canonical owner.

## 4. Research and manuscript evidence boundary

The September manuscript currently uses completed evidence, but manuscript work
does not prohibit new training, GPU evaluation, method search, or paid compute.
Those actions require the exact user authorization, machine, budget, protocol,
owner, and stop conditions in `AGENTS.md` and the live handoff.

Running an experiment does not automatically make it submission evidence. A new
result may change the manuscript only after its protocol and raw artifacts are
validated, a durable owner is routed in `INDEX.md`, and the author explicitly
admits the claim. An integrity problem that needs new evidence must be surfaced;
it must not be hidden by weakening an unrelated claim.

The current manuscript's from-training ceiling remains the existing 1.485B
comparison until such a promotion occurs. This is an evidence statement, not a
ban on later research.

## 5. Active single-table research programme

The active method programme runs alongside manuscript work and is owned by
[`../INDEX.md`](../INDEX.md) §5. It has two targets:

1. **Zero training:** one global request-static table/gain, frozen weights,
   about `0.12` maximum damage separately on Native NLL and downstream tasks,
   and the largest useful extrapolation toward 8x. The log-p2 s4/c=.074 arm is
   the current OLMo incumbent; the 4-to-8 frontier remains open.
2. **Low-cost adaptation:** small physical-2x/4x data and few LoRA steps, the
   same Native limits, and untouched 8x/16x/32x generated capability. NLL or
   attention improvement alone is not success.

Routing, dual tables, cache switching, strict Native equivalence, a unique
checkpoint-derived `m_k`, and operator geometry as an LM selector are not the
programme. The canonical entrypoint is
[`research/attention-aware-retrofit/analysis/SINGLE_TABLE_ROPE_OPEN_PROBLEMS_HANDOFF_20260904.md`](research/attention-aware-retrofit/analysis/SINGLE_TABLE_ROPE_OPEN_PROBLEMS_HANDOFF_20260904.md).

Before any later experiment enters the research agenda, it must state:

1. the falsifiable hypothesis and the existing evidence it does not duplicate;
2. how it escapes the closed classes in `INDEX.md` §3.1;
3. the exact intervention, controls, data, metric, budget, owner, and stop rule;
4. the claim or decision the result could change;
5. the required compute authorization and shutdown plan.

Live progress for an authorised future run still belongs only in the handoff;
the protocol and completed result belong with their durable owners.

## 6. Freeze gates

### 2026-09-17 — internal abstract and author-metadata freeze

- Title and abstract are scientifically final and owner-audited.
- Author roster, order, profiles, submission limits, and reciprocal-review status
  have been checked against the current official policy.
- OpenReview metadata matches the frozen title and abstract.
- No unresolved central claim, citation, anonymity, or dual-submission issue is
  hidden behind the abstract freeze.

### 2026-09-18 — official abstract deadline

- Submit the frozen abstract and metadata by 11:59 PM AoE.
- Record the exact submitted state and any platform receipt in the handoff.
- After this deadline, do not change the author roster or other frozen metadata
  except where the official policy explicitly permits it.

### 2026-09-25 — full-paper deadline

- Complete the final owner-by-owner scientific audit and visual review.
- Satisfy format, anonymity, citation, AI-use, ethics, reproducibility, font,
  PDF, and supplement gates.
- Resolve the NeurIPS-decision citation/distinctness branch if it is triggered.
- Ensure the uploaded PDF, anonymous supplement, title, abstract, and permitted
  author metadata are mutually consistent; download and inspect the submitted
  artifacts.
- Record final receipts only in the handoff.

The stable gate definitions are in
[`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md). That checklist does not
record completion state.

## 7. How to admit a revision item

A proposed September edit must provide, in the working discussion or handoff:

| Field | Required content |
| --- | --- |
| Current defect | Exact current PDF/source passage, not an old review locator |
| Owner | Canonical fact, theorem, protocol, or policy source |
| Reviewer effect | Score, credibility, comprehension, or validity consequence |
| Smallest change | Exact manuscript or release-package delta |
| Verification | Claim-level, build, visual, or packaging check required |
| Freeze impact | Whether it must land before 9/17 or 9/25 |

The table is an admission test, not a second tracker. Accepted work and current
status are recorded only in the handoff.

## 8. Retired inputs

The following remain useful only as provenance or adversarial review history:

- `AUTHOR_VERDICTS_20260828.md`;
- `research/CODEX_CLAUDE_PAPER_REVIEW_LOG.md`;
- `research/external-reviews/` bundles;
- August simulated-review, optimisation-plan, and whole-paper-plan documents;
- the superseded v2 content preserved in Git history.

Read them only to investigate a concrete current-paper question. Never treat an
old open item, model verdict, or proposed experiment as current merely because
it was once accepted or labelled mandatory.

## 9. Outcome-dependent manuscript edits (2026-09-04)

**Status:** prospective edit design, not completed manuscript work or evidence.
The dossier integration below supersedes the initial QK/source-margin research
sequence; existing-evidence editorial edits and outcome branches remain valid.
The author requested this plan alongside a first-principles independent assay.
No new experiment is assumed successful. Use the
[first-principles owner](research/attention-aware-retrofit/theory/CONSTRAINED_GENERATION_FIRST_PRINCIPLES_20260904.md)
and [execution contract](research/attention-aware-retrofit/preflights/CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md)
for research; this section owns only proposed reader-facing edits.

### Immediate edits supported by existing evidence

| Current source | Specific issue/opportunity | Proposed smallest change | Evidence/check |
| --- | --- | --- | --- |
| `sections/00_abstract.tex`, final three sentences | Repeats the decomposition identity, contains no effect size | Keep the decomposition opening; replace the repeated closing identity with one quantitative frozen fixed-support consequence | `SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823` and JSON; 0.56% is SAME-SUPPORT GEOMETRIC, not Native; nine tasks, not RULER-13 |
| `sections/01_intro.tex`, route/contribution paragraphs | Several construction names can obscure one scientific question | Preserve Figure 1/order; shorten repeated explanations and retain explicit derived/coarse versus EVQ-Cosh names | Current pure-z and exact-range owners; no new benchmark claim |
| `sections/04_experiments.tex`, frozen and deployment subsections | Pure-z arithmetic table and bundled Native/long route differ from the current log-p2 incumbent | Keep existing numbers/formulas attached to their actual methods; add a separate single-table row only if admitted | No silent arithmetic-to-log replacement; no splicing old RULER panels |
| `sections/03_theory.tex` | New internal ceiling explorations are not needed to support the current scientific object | Keep exact budget/collapse/co-adaptation/surrogate story; do not add an unvalidated 8x upper bound | New first-principles owner labels conditional implications and unresolved steps |

Suggested quantitative abstract sentence, to replace redundant closing prose
rather than extend the abstract:

> At matched support and amplitude in a frozen OLMo checkpoint, changing only
> interior allocation raises 16K nine-task RULER from 0.56% to 60.47%; a coarse
> label-free allocation reaches 61.04%.

This sentence is supported by the current owner; its insertion still requires
source/PDF build and layout review. It does not describe log-p2, natural QA,
Native retention, or a new zero-training 8x result.

### New-result branches: add, replace, or keep out

| Validated outcome | Add/replace in manuscript | Keep out / claim ceiling |
| --- | --- | --- |
| Only independent assay diagnosis succeeds | Reproducibility appendix may document the generated-output contract if relevant | No new abstract result, method superiority, or claim that both research questions are solved |
| One table passes independent Native gates and natural/RULER 4x confirmation | Add a distinct static-deployment row and its retention values beside the routed policy; update Discussion's future-direction paragraph to measured scope | Do not say exact Native preservation; do not reuse routed-policy results for log-p2 |
| Same table additionally gives useful blind physical 8x and natural-task confirmation | Replace lower-value deployment detail with a compact 1x/4x/8x curve, two retention constraints, baseline, and no-routing specification | No global 8x optimum; maximum tested useful reach only |
| New objective improves 2x/4x training diagnostics but not held-out exact+EOS | Normally no main-text addition; at most scoped adaptation limitation if it bears directly on a manuscript claim | No claim of capability conversion from NLL, route output or source margin |
| The fixed Native-constrained adapter improves blind 8x, passes Native, but fails 16x/32x | Add exact 8x transfer at the tested task scope after natural/official benchmark confirmation; retain 16x/32x outcome in appendix | Do not discard a valid 8x result or call it general 32x transfer |
| Blind 8x/16x transfer survives matched table/adaptation baselines and natural tests | Upgrade the adaptation subsection from probability/source-use evidence to reliable generated-task transfer at those lengths | An all-linear win alone does not identify a unique MLP/V/O mechanism; one seed is not training-variance evidence |
| All new candidates fail with working controls | Keep the current paper's causal core and completed practical results; only amend wording if a current claim is contradicted | Do not rewrite the paper as a failure ledger or claim the static-table/adaptation class is impossible |

### Highest-ROI follow-up and resource order

**2026-09-05 amendment:** The historical order below is superseded for the current experimental increment by [protocol §10](research/attention-aware-retrofit/preflights/CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md#10-开机后的固定比较轮次--2026-09-05-准备版): N_compact, then matched Qwen Z/Y. Its outcome-to-edit table governs prospective claims; current TeX remains unchanged. Report semantic/format/EOS separately and retain Native stratum regressions. No automatic claim of FFN necessity or zero-training success.


1. Repair current truth/identity/abstract clarity using completed evidence.
2. Qualify the independent diagnostic on a tiny real-model screen. An oracle or
   format failure should cost minutes, not a full long-context matrix.
3. Confirm a Native-feasible single-table point; even strict 4x can replace an
   open deployment question with measured evidence.
4. After synthetic source/output success, test a natural same-support
   geometric/coarse/derived contrast. This most directly connects the existing
   pure-z causal result to application behavior.
5. Only then buy broader blind-length, seed or model confirmation. A matched fixed-table N/Z/Y
   comparison has information value; another old rank/gain/step
   sweep without a discriminator does not.

These are expected-value priorities, not numerical success probabilities or
acceptance odds. The paper remains centered on decomposition and controlled
identification. New single-table/adaptation evidence is a consequence of that
object, not a replacement identity. No universal optimality, operator-bound
selector, or post-outcome theory fit is needed for the submission.

### Dossier integration: changes to the research-dependent edits

The complete supplied dossier was reviewed and its provenance hash/decisions are
in the first-principles owner §8. Adopt fixed-witness confirmation, qualified
Native-compact natural worlds, actual-deployment Native constraints, and matched
N/Z/Y all-linear r16 training. Do not add a new frequency search or frame QKVO
as an untried fix: the seven-arm 2026-07-31 owner already tested QKVO and sparse KL.

The useful manuscript delta if the new protocol succeeds is a compact matched
N/Z/Y before/after table, final Native retention, full generated answers/EOS and
source-twin success. If all three adapt equally, report that the tested Z table
is not uniquely necessary for conversion; the existing pure-z causal claim is
separate and remains owned by its fixed-support studies. Report compact/near/far
absolute scores, not only gaps that can shrink because compact was damaged.

Do not automatically replace the current Figure 1, contribution structure or
full theory section with the dossier's proposed three-story layout: it inspected
an older uploaded PDF. Current TeX already foregrounds the controlled decomposition.
A joint-relabel figure needs actual model parity receipts; the known symmetry is
not new mathematical novelty. The pointwise KL-to-argmax radius and decision-
direction identity are good diagnostics/appendix explanations if actual logits
support them, not additional headline theorems on their own.

The dossier's .03-NLL/-2-point Native thresholds are stricter than the author's
current .88 ratio convention; do not silently replace the primary gate. Its 72
GPU-h plan and nine runs are not authorization. Preserve the existing deadlines,
causal owners and separate construction identities; optional breadth follows a
resolving, affordable first-seed result.

### FFN review and the smallest useful manuscript addition

The [reviewed mechanism](research/attention-aware-retrofit/theory/CONSTRAINED_GENERATION_FIRST_PRINCIPLES_20260904.md#9-ffn-review-learning-under-a-native-constraint)
is a hypothesis about adapting nonlinear feature composition after positional
change. Do not write that frozen FFNs cannot use evidence, that low-rank tuning
prevents forgetting, or that an all-linear win alone identifies the old failure.
The result-dependent addition is one table: N/Z/Y, step0/restoration-only32/
selected, Native NLL and full generation/EOS retention, compact/near/far scores,
and the resulting blind useful reach. Label the new stratified Native generation
macro separately from the historical five-task macro.

Only if this main result resolves should a small matched-budget attention-only
r46 for OLMo or r68 for Qwen versus all-linear r16 panel enter the appendix. A positive contrast supports
the tested allocation of adaptation capacity; it does not prove universal FFN
necessity. Gradient-conflict and KL/margin plots may explain observations when
backed by runtime logs, but never replace generated outputs or count as another
headline contribution. If Native preservation or held-out transfer fails, omit
the proposed mechanism claim and retain the current support/allocation core.


### Author scope amendment: three benchmark families, not universal coverage

On2026-09-04 the author asked for a focused increment: NIAH, one natural QA
benchmark and one standardized short retention benchmark, supported by two
checkpoint regimes where feasible. The existing small-format and synthetic
probes are diagnostics for design; they need not become a long main-paper
benchmark table. No exact endpoint, outcome or absent8B result is invented here.

The [YaRN comparison](research/attention-aware-retrofit/theory/CONSTRAINED_GENERATION_FIRST_PRINCIPLES_20260904.md#10-yarn-correspondence-supervision-density-and-a-smaller-next-claim)
changes the next hypothesis: dense language-model supervision and an adequate
label budget may matter alongside FFN placement. Preserve YaRN's full-parameter,
dataset and metric differences when discussing precedent. Today's NIAH strict
0→100% mostly corrects output form; never advertise that as retrieval recovery.
The natural QA gains mix form and content and need the source-only companion.
A measured transfer increment can enter after author review; a universal method,
FFN necessity or oral-level result is not a current conclusion.

</details>
