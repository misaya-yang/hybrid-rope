# Path B Paper-Only Rebuttal Brief

日期：2026-06-10

用途：在 **不依赖 Geo+LoRA 新实验** 的前提下，基于当前论文、已完成的 paper/source 修复、以及现有 rebuttal 材料，给出可执行的 rebuttal 主线。Geo+LoRA 是未来 Path A 升级项，不作为当前回复的前置条件。

## 0. Bottom Line

当前应采用 **Path B：paper-only + honest-scope rebuttal**。

这个版本的核心不是“我们补了一个杀死所有质疑的新实验”，而是：

1. 主 claim 收窄到 mechanism / design-axis。
2. 主证据回到论文已经有的三个 primary anchors。
3. 主动修复 trust 问题：Figure 8/Table 21、token/provenance、1B label、LoRA attribution wording。
4. 对 reviewer 的合理质疑不强辩，而是把它们变成 scope、limitation、revision edits。
5. 不做二次投稿，不承诺 broad new experiments。

一句话策略：

> We do not claim universal long-context SOTA; we defend EVQ-Cosh as a zero-learned-parameter training-time RoPE frequency-allocation mechanism, supported by matched-scale YaRN leverage, PE-dominant diagnostics, and scarce-channel MLA evidence, while explicitly scoping LoRA, 1B schedule sensitivity, PK metric, and downstream benchmarks.

## 1. What We Can Defend Without Geo+LoRA

### 1.1 Main claim: finite spectral budget / allocation axis

**Defendable.**

Paper evidence:

- `paper/sections/05_experiments.tex` now opens with “This is a positional-encoding mechanism study.”
- `paper/sections/06_limitations.tex` says it is not a universal long-context recipe.
- EVQ modifies only inverse-frequency initialization and adds no learned parameters.

Safe response:

> The paper studies training-time frequency allocation as a third PE design axis, complementary to operator design and inference-time scaling. We will revise wording that could be read as universal SOTA or deployment validation.

Do not say:

- EVQ replaces YaRN/LongRoPE.
- EVQ is a general long-context recipe.

### 1.2 Primary I: EVQ x YaRN matched-scale substrate test

**Defendable, but only as matched-scale.**

Paper evidence:

- 454M, 10% passkey mix, 3 seeds.
- EVQ+YaRN reaches 100% teacher-forced NLL-gap PK@8K vs Geo+YaRN 61%.
- PPL improves at 8K and 16K.
- Current text explicitly says this is increased matched-scale YaRN leverage, not dominance over every tuned-scale YaRN baseline.

Safe response:

> Table 2 is not a tuned-scaler leaderboard. It tests whether the same YaRN range-scaling operation has different leverage on a Geo-trained versus EVQ-trained frequency substrate. We will make the matched-scale scope explicit.

Concession:

- A tuned Geo+YaRN scale sweep would be useful, but is not required to defend the narrower substrate-complementarity claim.

### 1.3 Primary II: PE-dominant DAPE-style diagnostic

**Defendable only as seed-scoped diagnostic.**

Paper evidence:

- DAPE-style 128->8K protocol.
- Geo/DAPE/EVQ retained seed-42 contrast.
- Learnable-tau row is 3-seed, but not every row is 3-seed.
- Current text says “competitiveness in this tested protocol, not comprehensive dominance over DAPE.”

Safe response:

> We agree the DAPE-style row should be read as a PE-dominant diagnostic, not a broad learned-PE benchmark. We will report seed scope explicitly.

Do not say:

- EVQ broadly dominates DAPE or learned PE.
- Primary II is a full multi-seed benchmark for all methods.

### 1.4 Primary III: MLA scarce-channel stress test

**Defendable and likely the strongest systems-facing anchor.**

Paper evidence:

- 432M MLA, 500M tokens, L_train=8192, 3 seeds.
- EVQ reduces 2x extrapolation PPL from 138.8 to 95.6 (-31.1%) with +1.1% in-distribution cost.
- It exceeds matched Geo+YaRN at s=4 in that setup.
- Current text explicitly says d_rot determines quantized channels; d_eff is an operating convention, not theorem.

Safe response:

> MLA is not presented as production-identical DeepSeek; it is a scarce-rotary-channel stress test. The evidence supports the finite-budget view because allocation effects are larger when rotary channels are scarce.

Concession:

- Direct d_rope-based tau ablations and production-scale MLA variants remain future work.

### 1.5 Supporting 750M continuation and progressive training

**Useful, but not primary.**

Paper evidence:

- 750M continue@4K: 16K PPL improves 45.1 -> 24.4; 8K AR exact 0% -> 77.5%.
- Progressive training: EVQ+YaRN remains stable while Geo+YaRN degrades; single seed.

Safe response:

> These rows do not prove trillion-token durability, but they argue against the simplest “only undertrained tiny models” explanation and motivate the mechanism claim.

Do not say:

- Training longer can never remove EVQ.
- 750M single-seed proves scaling durability.

## 2. What We Should Concede

| Issue | Concession | Safe framing |
| --- | --- | --- |
| LoRA attribution | Base vs EVQ-LoRA cannot isolate EVQ from LoRA/LongAlign | supporting post-hoc observation only |
| 1B MLA reversal | not robustness to saturation | schedule-sensitivity limitation |
| YaRN tuning | Table 2 is fixed/matched scale | substrate test, not tuned leaderboard |
| PK metric | teacher-forced NLL-gap is not generation exact match | PE diagnostic endpoint |
| Primary II seeds | seed-42 for Geo/DAPE/EVQ | diagnostic stress test |
| Downstream QA | QuALITY accuracy near random | Gold-answer NLL as probability-space diagnostic |
| MLA tau convention | d_eff is operating convention | future tau ablations |
| Production validation | not from-scratch trillion-token industrial proof | mechanism study with production-relevant stress tests |

The important move is to concede these early and then explain why the remaining scoped claim is still valuable.

## 3. Existing Fixes That Matter

These are high-value because they repair reviewer trust without needing new experiments.

| Fix | Why it matters |
| --- | --- |
| Figure 8 now plots Gold-answer NLL | removes a real figure/table inconsistency |
| Appendix token/protocol table added | blocks “you hid training budget” attack |
| 1B row relabeled schedule-sensitivity | prevents reviewer quoting our own contradiction |
| LoRA wording scoped to post-hoc | avoids false attribution |
| Primary II text narrowed | prevents overclaim against DAPE |
| PK definition explicit | avoids TF metric inflation |
| MLA d_eff wording narrowed | avoids theorem/convention confusion |

These should appear in the first half of the response, before optional experiments.

## 4. Reviewer-Specific Strategy Without Geo+LoRA

### 4.1 R2 empirical reviewer

R2 is still the hardest reviewer. Without Geo+LoRA, do not try to “win” the LoRA point. Concede it and move.

Best order:

1. Figure/Table correction.
2. Token/provenance table.
3. LoRA attribution concession.
4. Training budget: progression + 750M + primary token clarity, not “overtraining.”
5. 1B reversal as schedule sensitivity.
6. YaRN as matched-scale scope.
7. PK as teacher-forced NLL-gap.

Safe paragraph:

> We agree that the LoRA row cannot isolate EVQ frequency injection from LoRA/LongAlign adaptation. We therefore do not use it as a primary attribution claim. The main evidence remains the from-scratch controlled comparisons where only the RoPE frequency allocation changes, especially the 3-seed EVQ x YaRN and MLA anchors.

### 4.2 R1 theory reviewer

R1 needs epistemic status, not more bravado.

Best order:

1. Cosh shape is derived under the stated surrogate.
2. Tau is an operating default / basin selector.
3. Learnable tau failure is not surprising because training loss is in-range.
4. NTK-aware reversal shows scaler-specific composition.

Safe paragraph:

> We will separate derivation status from empirical calibration: the cosh density is derived for the broadband surrogate; tau is not a global optimum for trained attention, but an operating point validated by sweeps and diagnostics.

### 4.3 R3 systems reviewer

R3 can still be positive if we do not oversell.

Best order:

1. Zero learned parameters.
2. MLA scarce-channel relevance.
3. Dead-channel audit and diagnostic utility.
4. Supporting LoRA as post-hoc observation, not proof.

Safe paragraph:

> The practical value is not a ready deployment recipe; it is a low-complexity intervention and diagnostic lens for RoPE frequency allocation, especially in compressed-RoPE designs where channels are scarce.

### 4.4 AC

The AC needs a clean reason not to accept R2’s harshest reading.

AC message:

> The authors narrowed the claim, corrected a real figure inconsistency, disclosed token/seed scope, relabeled the 1B reversal as a limitation, and kept the strongest evidence on the three primary anchors. The remaining weaknesses are real but now scoped rather than hidden.

## 5. Path B Response Skeleton

Use this order:

1. Scope opening:
   - mechanism study;
   - no universal SOTA;
   - no replacement for inference-time scaling.
2. Corrections:
   - Figure 8 NLL;
   - token/seed/protocol;
   - PK definition;
   - 1B relabel.
3. Main evidence:
   - Primary I matched-scale EVQ x YaRN;
   - Primary II seed-scoped PE diagnostic;
   - Primary III MLA scarce-channel 3-seed.
4. Concessions:
   - LoRA is supporting only;
   - tuned YaRN baseline not claimed;
   - AR exact separate from TF PK;
   - downstream accuracy not primary.
5. Theory:
   - shape vs scale separation;
   - tau operating default;
   - learned tau myopic-loss explanation.
6. Systems:
   - zero-parameter schedule;
   - MLA relevance;
   - dead-channel audit.
7. Close:
   - scoped contribution remains valuable;
   - production-scale validation future work.

## 6. Minimal Text To Reuse

This is the most compact paper-only rebuttal core:

> We agree that several rows should be scoped more carefully. EVQ-Cosh is a mechanism study of training-time RoPE frequency allocation, not a universal long-context SOTA claim or a replacement for inference-time scaling. We therefore revise the paper around three primary anchors: 3-seed EVQ x YaRN matched-scale substrate leverage, a seed-scoped DAPE-style PE-dominant diagnostic, and a 3-seed MLA scarce-channel stress test. We also correct a stale QuALITY figure to match the Gold-answer NLL table, add token/seed provenance, define PK as teacher-forced NLL-gap unless AR exact is explicitly marked, and relabel the 1B MLA row as schedule sensitivity rather than saturation robustness. The LoRA row is kept as supporting post-hoc adaptation evidence only; without a matched Geo+LoRA control it is not used for EVQ-specific attribution.

## 7. Experiments To Not Chase Right Now

Given the user instruction to ignore not-yet-done Geo+LoRA and base the response on current paper evidence:

| Do not chase | Reason |
| --- | --- |
| Geo+LoRA as prerequisite | not done; treating it as blocker distracts from usable Path B |
| broad LongBench/RULER | capacity-limited and not central to mechanism claim |
| 1B multi-seed rerun | too large for rebuttal; current response should scope the row |
| many PE baselines | turns rebuttal into second submission |
| VideoRoPE fairness argument | not a reviewer-answering move |
| new theorem | opens new attack surface |

Optional if already available, but not required:

- learned tau trajectory;
- AR exact from existing evaluator;
- Geo+YaRN scale sweep if checkpoint/eval is immediately ready.

## 8. Verdict

Without Geo+LoRA, the rebuttal is weaker on industrial-checkpoint attribution, but still viable if it is honest:

- Do not pretend LoRA closes the confound.
- Do not fight the 1B reversal; relabel and scope it.
- Do not sell benchmarks; sell diagnostics and mechanism.
- Put trust repairs first.
- Keep Primary I and MLA as the strongest anchors.

Current recommendation:

> Proceed with Path B as the default. Geo+LoRA should be treated as a future upgrade, not a blocker for a paper-only rebuttal based on current evidence.
