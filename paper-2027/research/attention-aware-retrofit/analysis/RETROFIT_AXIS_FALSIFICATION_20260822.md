# Training-free retrofit: three candidate axes, all falsified

> **2026-09-08 更正：** 本文继承的“D*约束任意LoRA”“rank64等于每头rank4／
> 已跑适配仅达9%修复能力”撤回，见[原分析更正](TRANSPORT_RESIDUAL_ANALYSIS_20260822.md)。
> 原ALS还存在首轮虚假收敛错误，现已修复；历史表值未重跑。所列实际RULER
> 观察保留各自身份，不能用代理误差、相关性或该实现修复关闭／恢复整个方法类。

- **Date:** 2026-08-22
- **Status:** CPU analysis complete; GPU RULER points supplied by the parallel session
- **Evidence role:** internal. Frequency-table analysis under an isotropic content
  model, plus a first prospective test of it against measured RULER. Not
  manuscript evidence.
- **Code:** `scripts/analysis/rope_transport/` (`nullband.py`,
  `run_pareto_search.py`, `run_nullband_search.py`, 24 unit tests)
- **No GPU was used by this analysis. No checkpoint was loaded. No training.**

> **Post-handoff update:** the axis falsification below still stands, but its
> adapter-first next-step judgement is superseded by
> `../results/zero-training-deployment/LENGTH_CONDITIONED_BUDGETED_RETROFIT_RESULT_20260822.md`. A matched GPU
> crossing found that the frozen budgeted table plus deterministic length-only
> attention scaling reaches 0.5825 at 8K and 0.4000 at 16K, while Native plus
> the same scaling remains 0 at both lengths. The missing object was attention
> concentration after frequency transport, not another CPU table score.

## 1. What was being tested

A retrofit operator is a frequency table swapped into a mature checkpoint. The
question was whether a **CPU-computable** property of that table predicts
downstream capability, which would let operator design proceed without GPU.

Three candidate axes were built and pre-committed before the RULER numbers were
read:

| axis | meaning |
| --- | --- |
| `D*` | in-window logit energy no fixed Q/K content map can restore. Exactly the transplant obstruction's operator class, a strict superset of any Q/K LoRA |
| `phase risk` | mean unseen phase per channel at the deployed length, in turns |
| `coverage` | distance from the deployed operator trajectory to the trained one, requiring a **single** `D'` to match all channels at once |

## 2. The measurement

RULER-core4 on OLMo-2-1B-Instruct, one seed, from the parallel session.

| table | len | RULER macro | `D*` | coverage | phase risk |
| --- | ---: | ---: | ---: | ---: | ---: |
| official YaRN s=2 | 8192 | **0.5375** | 0.2485 | 0.5227 | 0.0000 |
| budgeted s2 p2 | 8192 | 0.4000 | 0.1848 | 0.4055 | 0.0000 |
| budgeted s2 p1 | 8192 | 0.3875 | 0.2629 | 0.4019 | 0.0000 |
| budgeted s2 p2 + LoRA 50 step | 8192 | 0.3875 | — | — | — |
| budgeted s4 p2 | 8192 | 0.2025 | 0.2239 | 0.4186 | 0.0000 |
| **budgeted s4 p2** | **16384** | **0.1150** | 0.2239 | 0.4292 | 0.0000 |
| official YaRN s=4 | 16384 | 0.0125 | 0.2900 | 0.5967 | 0.0000 |
| **one-turn floor s=2** | 8192 | **0.0000** | 0.0192 | 0.2410 | 0.0000 |
| native | 8192 | 0.0000 | 0.0000 | 0.2363 | 0.0531 |
| native | 16384 | 0.0000 | 0.0000 | 0.5811 | 0.1053 |

```
Spearman(RULER, -D*)        = -0.550
Spearman(RULER, -coverage)  = -0.250
Spearman(RULER, -phase risk) = +0.000
```

## 3. What is falsified

**(a) "A channel that wrapped once in training is safe to leave alone."** This
was the whole argument for a minimal-displacement retrofit. `one_turn_floor_s2`
is built to satisfy it exactly, has near-optimal `D*` (0.019) and the lowest
coverage residual of any moved table, and scores **0.0000** — identical to doing
nothing. One decisive counterexample, not a noise result.

**(b) `D*` does not rank retrofit operators.** The sign is wrong: within this
set, *more* unrepairable in-window damage goes with *better* RULER. `D*` remains
valid as what it was derived to be — an upper bound on what a Q/K adapter can
restore — but it is not a design objective.

**(c) `coverage` does not rescue it.** The joint-match reformulation explains
YaRN's `beta_fast` cleanly in principle and still fails the test.

**Caveat, stated plainly:** eight points, three tied at zero, one model, the
core-4 RULER subset, one seed. That is weak evidence for the two correlations.
It is *not* weak evidence for (a), which rests on a single table scoring zero
while being optimal on every axis proposed for it.

## 4. What survives

1. **`D*` as a bound, not a predictor.** The closed form, the two exactness
   tests (frequency permutation and sign alias are compensable and the solver
   drives both to `<1e-8`), and the multi-start guarantee that `D*` is an upper
   bound all stand.
2. **The LoRA capacity arithmetic.** Rank-`r` on `q_proj` gives
   `rank(M - I) <= r` shared across all 16 heads; the repository's rank-64 Q/K
   arm is **4 ranks per head**. The completed EVQ retrofit arms ran at ~9% of
   the repair their own operator class permits. Independent of any axis.
3. **One positive downstream point:** `budgeted_s4_p2` at 16K scores 0.1150
   against official YaRN s=4's 0.0125 — a 9x gap, and the only place anything
   derived here beats the incumbent. Single point, one seed, low absolute value.
4. **The incumbent is unbeaten where it matters.** Official YaRN s=2 at 8K
   (0.5375) is the best result on this checkpoint. Nothing derived in this line
   of work has beaten it.

## 5. Read for the next session

Three distinct table-space retrofit attempts have now been tested downstream on
a mature checkpoint — anchored EVQ-Cosh, uniqueness-budgeted transport, and the
one-turn floor — and official YaRN wins each time at the length that matters.
That is now downstream evidence, not a prediction from `D*`.

Judgement, not proof: **the "find a better frequency table for a mature model"
route looks close to exhausted, and further GPU on new table shapes has poor
expected value.** The two things in this file that did survive are both about
the *adapter*, not the table:

- rank must be allocated **per head** (a block-diagonal per-head map costs the
  identical 8.39M parameters as the current rank-64 arm and reaches ~0.89 of the
  achievable repair versus ~0.09);
- `budgeted_s2_p2 + LoRA 50 step` scored 0.3875 versus 0.4000 for the same table
  with no adapter at all — the adapter did **not** help. Whether that is the
  rank allocation or the step count is untested and is the one adapter question
  with a cheap decisive answer.

## 6. Operational notes

- `_init_worker` set `OMP_NUM_THREADS` *after* fork, which is too late: BLAS
  reads it at numpy import. Any relaunch must export the thread limits in the
  shell before `python`. This oversubscribed the box (load 1179 on 208 cores).
- The parallel session's `pareto_search` and `nullband_search` both died at the
  stage1 -> stage2 boundary with no traceback while the box was oversubscribed
  and while this session killed its own 192 workers. Cause not established;
  they need relaunching, with thread limits exported.
- Raw analysis outputs and the frozen target manifest remain on the original
  evaluation host; this owner records their identities without publishing the
  host's private filesystem layout.
