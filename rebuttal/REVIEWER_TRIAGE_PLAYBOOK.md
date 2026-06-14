# Reviewer Triage Playbook For July Rebuttal

Date: 2026-06-14

Purpose: use this as the first rebuttal document when real reviews arrive. It
is deliberately short because a human reviewer will spend hours, not days, on
the paper. Do not turn rebuttal into a second paper or an artifact audit unless
the reviewer explicitly asks for that evidence.

## One-Sentence Status

The submission is not in a severe state: the main mechanism claim is defensible,
but the response must stay narrow and avoid volunteering weak supporting rows.

## What We Are Defending

Defend this:

> EVQ-Cosh studies training-time RoPE frequency allocation as a finite-spectral
> budget design axis, complementary to inference-time range scaling.

Do not defend this:

- EVQ as universal long-context SOTA.
- EVQ as a YaRN, LongRoPE, DAPE, FIRE, or learned-PE replacement.
- Supporting rows as primary evidence.
- The compact anonymous supplement as a full checkpoint/result archive.

## The 4 Reviewer Questions To Expect

### 1. Is this more than a tuned RoPE/base/YaRN trick?

Fast answer:

- Yes, the paper is about training-time allocation shape, not a new inference
  scaler.
- The strongest evidence is matched-scale EVQ x YaRN: same fixed YaRN scale,
  different trained frequency substrate.
- We should concede that this is not a tuned Geo+YaRN or LongRoPE leaderboard.

Use:

- Matched-scale EVQ x YaRN result.
- The theory framing: operator, range scaling, and allocation are separate axes.

Avoid:

- "EVQ beats tuned YaRN."
- "EVQ replaces LongRoPE."

### 2. Are the experiments strong enough, or are they mostly single-seed?

Fast answer:

- The primary anchors are explicitly tiered.
- The matched-scale EVQ x YaRN result and MLA are 3-seed anchors.
- The PE-dominant DAPE-style result is a seed-scoped diagnostic, not broad
  learned-PE dominance.
- Video, LoRA, progressive, 750M, and QuALITY are supporting context unless a
  reviewer asks about that exact issue.

Use:

- Evidence-tier table.
- Matched-scale EVQ x YaRN for substrate/range complementarity.
- MLA 8K/500M 3-seed result as a scarce-channel stress test.
- PE-dominant result only with "seed-42 diagnostic" wording.

Avoid:

- Promoting LoRA/video/progressive into primary proof.
- Calling the 1B/4K MLA row "saturation robustness."

### 3. Is PK / downstream evidence overstated?

Fast answer:

- PK in the main tables is teacher-forced NLL-gap retrieval, not autoregressive
  exact match.
- AR exact is only used where explicitly labeled.
- Downstream accuracy is not the main claim; QuALITY is a Gold-NLL supporting
  check in a capacity-limited 454M setting.

Use:

- The metric definition in the experiments section.
- The 750M table only if the reviewer specifically asks about AR exact.

Avoid:

- "PK means exact retrieval."
- "QuALITY proves downstream task superiority."

### 4. Is the supplement/reproducibility package enough?

Fast answer:

- The anonymous supplement is a compact code archive, not a full result dump.
- It includes the core EVQ schedule implementation, public reproduction paths,
  figure scripts, and curated JSON for the strongest matched-scale and
  PE-dominant diagnostic values.
- It intentionally excludes checkpoints and large result directories.
- If asked, we should acknowledge that some traceability docs point beyond the
  compact archive and commit to releasing cleaned launch scripts/manifests with
  the full release.

Use:

- Core schedule code and unit tests.
- Curated matched-scale EVQ x YaRN JSON.
- Curated PE-dominant/Fig.3 panel-a JSON.
- Reproducibility guide as a compact-path guide, not a full archive manifest.

Avoid:

- Claiming the compact supplement already contains every historical script,
  checkpoint, and result log.
- Leading with missing artifacts unless the reviewer asks about reproduction.

## Fastest Response Workflow

1. Classify each real review into the 4 questions above.
2. Answer only the 2-3 questions that drive the score.
3. Use one paragraph of trust repair before defense if the reviewer flags a
   figure, metric, seed, or reproduction issue.
4. Add new experiments only if they answer a specific reviewer question and can
   be reported with exact numbers.
5. Delete any sentence that sounds like a broader paper claim than the submitted
   evidence supports.

## If There Is Time For Only One Concrete Update

Prepare a compact "supplement clarification" paragraph:

> The anonymous supplement is a compact code archive rather than a full
> checkpoint/result dump. It contains the EVQ-Cosh schedule implementation,
> evaluator code, figure scripts, and curated JSON for the matched-scale EVQ x
> YaRN and PE-dominant diagnostic values. We will release cleaned launch scripts,
> result manifests, and checkpoint hashes for the larger MLA/supporting runs in
> the full public release.

Use this only if a reviewer attacks reproducibility or missing scripts. Do not
volunteer it in the opening if no reviewer asks.

## Final Send Gate

Before sending the rebuttal, check the draft from two angles:

Reviewer angle:

- Does this answer the actual review, not an internal audit concern?
- Can the reviewer understand the point in one pass?
- Are we admitting real scope limits instead of arguing around them?

Evidence angle:

- Is every number already in the submitted paper, supplement, or a verified
  result table?
- Are supporting rows clearly marked as supporting?
- Did we avoid exact claims about experiments whose results are not in hand?

If either angle fails, shorten and scope the sentence.
