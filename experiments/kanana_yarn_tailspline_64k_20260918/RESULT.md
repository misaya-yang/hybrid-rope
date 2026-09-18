# Kanana 64K: TailSpline vs official runtime YaRN

## Verdict

On `kakaocorp/kanana-1.5-8b-instruct-2505` at a physical length of 65,536 tokens,
TailSpline scored **72.65%** on paired RULER Full-13x10, compared with **65.64%**
for Kanana's official runtime YaRN configuration: **+7.01 percentage points**.

The two-task pilot pointed in the opposite direction (-10.25 pp). Completing all
13 tasks therefore changed the conclusion rather than merely reducing uncertainty.

## Frozen comparison

- Model: `kakaocorp/kanana-1.5-8b-instruct-2505`
- Target length: 65,536 tokens (native window: 32,768)
- Inputs: the same 130 prompts, 10 per RULER task, in the same order for both arms
- Panel SHA256: `e3b7cf1f75ff9672872eda295b7a334203ccdab10d171f255ce28994d870d1b1`
- Decoder, precision, scorer and output cap were shared
- Official runtime YaRN: factor 4.4, original window 32,768, beta-fast 64,
  beta-slow 2
- TailSpline: exact S=2 table, canonical transition band `[20,35]`
- No model training or checkpoint-weight changes were performed

## Three-arm pilot: 2 tasks x 10

| Method | Multiquery | VT | Task-equal score |
|---|---:|---:|---:|
| Official runtime YaRN | 25.0% | 82.0% | 53.50% |
| MrRoPE-Pro S=2 | 32.5% | 70.0% | 51.25% |
| TailSpline S=2 | 22.5% | 64.0% | 43.25% |

Pilot differences: MrRoPE-Pro - YaRN `-2.25 pp`; TailSpline - MrRoPE-Pro
`-8.00 pp`; TailSpline - YaRN `-10.25 pp`. All three arms contain 20 rows and no
empty outputs. MrRoPE-Pro was a pilot-only diagnostic and was not expanded to
Full-13.

## Paired RULER Full-13x10

| Task | Official YaRN | TailSpline | TailSpline - YaRN |
|---|---:|---:|---:|
| NIAH single 1 | 100.0% | 100.0% | +0.0 pp |
| NIAH single 2 | 100.0% | 100.0% | +0.0 pp |
| NIAH single 3 | 100.0% | 100.0% | +0.0 pp |
| NIAH multikey 1 | 80.0% | 80.0% | +0.0 pp |
| NIAH multikey 2 | 50.0% | 60.0% | +10.0 pp |
| NIAH multikey 3 | 40.0% | 70.0% | +30.0 pp |
| NIAH multivalue | 60.0% | 65.0% | +5.0 pp |
| NIAH multiquery | 25.0% | 22.5% | -2.5 pp |
| Variable tracking | 82.0% | 64.0% | -18.0 pp |
| Common-word extraction | 13.0% | 43.0% | +30.0 pp |
| Frequent-word extraction | 93.33% | 90.0% | -3.33 pp |
| QA 1 | 70.0% | 80.0% | +10.0 pp |
| QA 2 | 40.0% | 70.0% | +30.0 pp |
| **Task-equal macro** | **65.64%** | **72.65%** | **+7.01 pp** |

TailSpline won 6 tasks, tied 4 and lost 3. Across the 130 paired rows it won 30,
lost 16 and tied 84.

## Output health

| Method | Rows | Empty | EOS rate | Cap rate |
|---|---:|---:|---:|---:|
| Official runtime YaRN | 130 | 0 | 65.38% | 34.62% |
| TailSpline | 130 | 0 | 70.77% | 29.23% |

Every row ID and prompt SHA256 matched across the two Full-13 arms. Both arms
completed 130 unique rows over all 13 tasks. The above rates are diagnostics;
the reported benchmark metric is RULER's official task-equal score.

## Evidence boundary

This is a clean 64K deployment comparison between two zero-training runtime
frequency allocations on one instruction checkpoint. Kanana documents runtime
YaRN beyond its 32K native window; this experiment does **not** establish that
the checkpoint itself underwent YaRN long-context continued training. The sample
budget is 10 examples per task, so the result is cross-task evidence rather than
a high-sample per-task estimate.

Compact reports are in `reports/pilot2_three_arm.json` and
`reports/full13x10.json`. Large generation rows remain on the experiment server
and are intentionally not committed to Git.
