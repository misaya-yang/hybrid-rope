# Kanana 64K RULER and 128K complete-context QA

## Verdict

On `kakaocorp/kanana-1.5-8b-instruct-2505` at a physical length of 65,536 tokens,
paired RULER Full-13x10 ranked the three methods as **TailSpline 72.65% >
MrRoPE-Pro 70.33% > official runtime YaRN 65.64%**. TailSpline led MrRoPE-Pro
by **+2.32 percentage points** and YaRN by **+7.01 points**.

The two-task pilot pointed in the opposite direction (-10.25 pp). Completing all
13 tasks therefore changed the conclusion rather than merely reducing uncertainty.

## Frozen comparison

- Model: `kakaocorp/kanana-1.5-8b-instruct-2505`
- Target length: 65,536 tokens (native window: 32,768)
- Inputs: the same 130 prompts, 10 per RULER task, in the same order for all arms
- Panel SHA256: `e3b7cf1f75ff9672872eda295b7a334203ccdab10d171f255ce28994d870d1b1`
- Decoder, precision, scorer and output cap were shared
- Official runtime YaRN: factor 4.4, original window 32,768, beta-fast 64,
  beta-slow 2
- TailSpline: exact S=2 table, canonical transition band `[20,35]`
- MrRoPE-Pro: canonical S=2 table, the same band `[20,35]`, gain and endpoints
  as TailSpline
- No model training or checkpoint-weight changes were performed

## Three-arm pilot: 2 tasks x 10

| Method | Multiquery | VT | Task-equal score |
|---|---:|---:|---:|
| Official runtime YaRN | 25.0% | 82.0% | 53.50% |
| MrRoPE-Pro S=2 | 32.5% | 70.0% | 51.25% |
| TailSpline S=2 | 22.5% | 64.0% | 43.25% |

Pilot differences: MrRoPE-Pro - YaRN `-2.25 pp`; TailSpline - MrRoPE-Pro
`-8.00 pp`; TailSpline - YaRN `-10.25 pp`. All three arms contain 20 rows and no
empty outputs. The subsequent Full-13 result reverses this pilot ordering.

## Paired RULER Full-13x10

| Task | Official YaRN | MrRoPE-Pro | TailSpline | TS - MrPro | TS - YaRN |
|---|---:|---:|---:|---:|---:|
| NIAH single 1 | 100.0% | 100.0% | 100.0% | +0.0 pp | +0.0 pp |
| NIAH single 2 | 100.0% | 100.0% | 100.0% | +0.0 pp | +0.0 pp |
| NIAH single 3 | 100.0% | 100.0% | 100.0% | +0.0 pp | +0.0 pp |
| NIAH multikey 1 | 80.0% | 80.0% | 80.0% | +0.0 pp | +0.0 pp |
| NIAH multikey 2 | 50.0% | 50.0% | 60.0% | +10.0 pp | +10.0 pp |
| NIAH multikey 3 | 40.0% | 50.0% | 70.0% | +20.0 pp | +30.0 pp |
| NIAH multivalue | 60.0% | 57.5% | 65.0% | +7.5 pp | +5.0 pp |
| NIAH multiquery | 25.0% | 32.5% | 22.5% | -10.0 pp | -2.5 pp |
| Variable tracking | 82.0% | 70.0% | 64.0% | -6.0 pp | -18.0 pp |
| Common-word extraction | 13.0% | 51.0% | 43.0% | -8.0 pp | +30.0 pp |
| Frequent-word extraction | 93.33% | 93.33% | 90.0% | -3.33 pp | -3.33 pp |
| QA 1 | 70.0% | 70.0% | 80.0% | +10.0 pp | +10.0 pp |
| QA 2 | 40.0% | 60.0% | 70.0% | +10.0 pp | +30.0 pp |
| **Task-equal macro** | **65.64%** | **70.33%** | **72.65%** | **+2.32 pp** | **+7.01 pp** |

Against MrRoPE-Pro, TailSpline won 5 tasks, tied 4 and lost 4. Against official
YaRN it won 6, tied 4 and lost 3. Row-level TailSpline/MrRoPE outcomes were
16 wins, 94 ties and 20 losses; the official benchmark score remains task-equal
and preserves the magnitude of partial-credit differences.

## Output health

| Method | Rows | Empty | EOS rate | Cap rate |
|---|---:|---:|---:|---:|
| Official runtime YaRN | 130 | 0 | 65.38% | 34.62% |
| MrRoPE-Pro | 130 | 0 | 69.23% | 30.77% |
| TailSpline | 130 | 0 | 70.77% | 29.23% |

Every row ID and prompt SHA256 matched across all three Full-13 arms. All arms
completed 130 unique rows over all 13 tasks. Independent replay of the documented
official RULER scorer produced zero mismatches in all 390 outputs. Manual output
inspection confirmed that large task differences correspond to substantive
answer differences rather than formatting-only credit. Removing any one task
leaves TailSpline ahead of both baselines.

As a fixed-13-task sensitivity analysis, paired within-task row resampling gave
`[-3.13,+8.04] pp` for TailSpline - MrRoPE and `[+0.21,+13.81] pp` for
TailSpline - YaRN. These intervals describe the 10-row-per-task sampling budget;
they do not replace the official paired benchmark scores above.

## Evidence boundary

This is a 64K comparison between three zero-training runtime frequency
allocations on one instruction checkpoint. TailSpline versus MrRoPE-Pro is the
controlled allocation comparison: both use S=2, the same transition band, gain
and endpoints. Official runtime YaRN instead uses Kanana's documented factor-4.4
deployment configuration, so that contrast is a complete-method comparison, not
a pure-allocation ablation. The experiment does **not** establish that Kanana's
checkpoint underwent YaRN long-context continued training.

The sample budget is 10 examples per task, so the result is broad cross-task
evidence rather than a high-sample per-task estimate. Complete InfiniteBench
English-QA prompts begin at 69,680 Kanana tokens; no untruncated row fits this
64K condition. Natural-QA therefore remains a separate 128K large-memory run and
was not replaced by a truncated proxy.

Compact reports are in `reports/pilot2_three_arm.json`, `reports/full13x10.json`
and `reports/full13x10_three_arm.json`. Large generation rows remain on the
experiment server and are intentionally not committed to Git.

## 128K complete-context InfiniteBench English QA

At a physical target of 131,072 tokens, TailSpline S=4 scored **19.59%** and
official runtime YaRN factor 4.4 scored **17.85%** on the official English-QA
token-F1 metric, a TailSpline difference of **+1.74 percentage points**. The
source-context-cluster-equal scores were `19.63/18.04%`, preserving a
`+1.58 pp` TailSpline lead.

| Method | Rows | QA F1 | Cluster-equal F1 | Empty | EOS | Cap |
|---|---:|---:|---:|---:|---:|---:|
| TailSpline S=4 | 118 | 19.59% | 19.63% | 0 | 80 | 38 |
| Official runtime YaRN 4.4 | 118 | 17.85% | 18.04% | 0 | 76 | 42 |

The panel contains 118 paired prompts from 23 complete source contexts. Kanana
input lengths range from 69,680 to 130,852 tokens; no context was truncated.
TailSpline had 31 row wins, YaRN 30 and 57 ties. Ordered row IDs and prompt
SHA256 values match exactly. Independent rescoring from every stored output
reproduced both arm means exactly and matched every stored row score.

- Panel SHA256: `87dafb6f07bb5611e741bf14f6ba17fa7778ba20b5569592845f7dc19b779626`
- TailSpline raw SHA256: `4f8988e21ec0966619451da7195832a5945e0454f12a3149304f06a1aaab27e1`
- Official YaRN raw SHA256: `627034794dced6dfbad1a5437672d8ef8712e2b0f2931a6e5ffe72b766e1d36a`

This is a complete-method deployment comparison at approximately matched
physical extension scale: TailSpline uses exact S=4, whereas the checkpoint's
documented runtime YaRN uses factor 4.4. It establishes a positive fixed-panel
128K QA result for TailSpline over official runtime YaRN; it does not identify
training history or include a 128K MrRoPE arm, which was skipped before start.
The compact report is `reports/qa128k_two_arm.json`; large raw generations stay
on the experiment server.
