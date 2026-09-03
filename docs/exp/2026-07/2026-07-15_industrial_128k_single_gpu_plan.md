# 128K industrial EVQ-vs-YaRN LoRA on one 96 GB card (plan, 2026-07-15)

## Status

Forward-looking plan for a single RTX Pro 6000 (96 GB). Single-seed discovery
only; no paper number changes. It extends the archived 2026-07-14 industrial
capability design to a 128K-native model, and it deliberately BREAKS that
design's "raw schedule only, no scaler" invariant, because a 128K model's
long-context ability IS a
frequency scaler (YaRN). The matched baseline here is therefore YaRN, not raw
Geo. All feasibility and tau numbers below are produced by
`scripts/analysis/industrial_128k_feasibility.py`, whose coverage
diagnostics reproduce the registered 8B/8K anchors exactly
(native dormant=20 / erank=23.60; EVQ tau=1.414 dormant=15 / erank=36.22).

## Bottom line

Yes, you can train this well on one 96 GB card, provided you do NOT train at
full 128K -- which you should not want anyway. The EVQ thesis is train-short /
test-long, and the demonstrated regime is ~4x extrapolation. Memory and science
agree: train at 16-32K, evaluate to 128K.

- Real industrial pick: Qwen2.5-14B-Instruct, train 16-32K, eval 32/64/128K.
- Drop-in pilot: Llama-3.1-8B-Instruct (same arch as your existing 8B pipeline,
  base 5e5, native 8K; train 8-16K). It also lets you reuse tau=1.414 at 8K.
- Rule out: Qwen2.5/3-32B for training on one 96 GB card (below).

## Hardware verdict (BF16, rank 64, microbatch 1, full grad checkpointing)

No QLoRA/FP8: 4-bit weights would confound the frequency-shape claim, and it is
not needed here. Peak VRAM (GB), '!' = exceeds 96 GB at microbatch 1:

| model | 8K | 16K | 32K | 64K | 128K | eval NLL@128K | eval GEN@128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.1-8B | 23 | 25 | 30 | 40 | 61 | 23 | 37 |
| Qwen2.5-7B | 21 | 23 | 27 | 36 | 52 | 21 | 26 |
| Qwen2.5-14B | 40 | 44 | 53 | 72 | 108! | 38 | 60 |
| Qwen2.5-32B | 80 | 86 | 99! | 124! | 173! | 76 | 105! |

Verdict:

- 8B / 7B: train at any length up to 128K; 128K generation eval fits. Easiest,
  fastest, safe.
- 14B: train comfortably at <=64K (53 GB at 32K, 72 GB at 64K); 128K training
  OOMs (108 GB). 128K generation eval fits (60 GB). This is the intended regime.
- 32B: not viable on one card (32K train 99 GB, 128K gen eval 105 GB). Drop it.

Throughput: at 32K, microbatch 1 with gradient accumulation to effective batch
8-16, ~500 optimizer steps is a few hours on this card, not days. The KV cache
dominates 128K eval (~25 GB at 14B); teacher-forced answer-NLL (one forward,
answer-only tail logits) is much cheaper and is the primary metric.

## Design deltas vs the 8B raw-schedule design

1. Baseline arm = matched YaRN(native)-LoRA, not raw-Geo. The model's native
   long context IS YaRN; the fair contrast is EVQ frequency map vs YaRN
   frequency map, same data/seed/rank/steps.
2. Carry the attention temperature. YaRN bundles frequency reshaping with an
   mscale `= 0.1*ln(factor) + 1`. To isolate the cosh-shape effect, give the
   EVQ arm the same mscale (use its effective scale `s = test_L / native_L`),
   so the ONLY difference between arms is the inverse-frequency distribution --
   the same discipline as your midpoint-Geo vs midpoint-EVQ isolation. If you
   instead let LoRA absorb the temperature mismatch, log it explicitly.
3. Test region = the YaRN-stressed far zone. Below native length is an in-range
   control (expect EVQ neutral-to-slightly-negative, matching the +0.39 in-range
   cost seen at 8B/8K); the contest is at 2x-4x native (64K/128K for a 32K-native
   Qwen).
4. Retrieval-limited benchmark, not generic QA: RULER + NoLiMa (which shows most
   models drop below 50% of base score by 32K -> real room over YaRN) + one
   long-doc QA. Report intermediate metrics (answer NLL, source-removal delta,
   gold rank) so a null on pass rate is still interpretable.

## Hyperparameters

### tau -- do not reuse 1.414

tau sets the EVQ-Cosh warp. Reusing the 8B/8K value 1.414 on a different base or
train length is wrong: the tau that yields a given active-channel coverage
shifts with both. The principled, pre-registerable transfer is to reproduce the
coverage profile that WORKED at 8B/8K (dormant~=15, entropy-effective-rank~=36,
test-length wrap~=0.73), not to maximize entropy rank (which just runs tau up
and raises test aliasing). Numerically:

| model | base | train L | pick tau | dormant | erank | wrap(test) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.1-8B | 5e5 | 8K | 1.414 | 15 | 36.2 | 0.73 |
| Llama-3.1-8B | 5e5 | 16K | 0.9 | 15 | 35.6 | 0.73 |
| Qwen2.5-14B | 1e6 | 16K | 1.2 | 15 | 35.9 | 0.73 |
| Qwen2.5-14B | 1e6 | 32K | 0.7 | 15 | 36.4 | 0.73 |

Key facts: tau DECREASES as train length grows (a longer window naturally
activates more channels, so less warp is needed) and rises modestly with base.
Reusing 1.414 at 32K would over-warp (dormant 12, erank 48.6, higher aliasing).
The value 1.414 equals 128/sqrt(8192) only by coincidence at 8B/8K; the rejected
closed form tau=d/sqrt(L) is not used -- the coverage match gives 0.7 at 32K, not
128/sqrt(32768)=0.71 by luck here but 1.0 at 16K where the formula would say
128/sqrt(16384)=1.0 yet coverage wants 0.9-1.2 depending on base. Treat the pick
as a center and run a small locked dev grid `pick +/- 0.2` (3 values, chosen on a
dev set before any test eval = ZT-cal-style calibration), never a post-hoc sweep.

### rank -- keep 64, matched

rank 64, alpha 128, dropout 0.05, on q/k/v/o, identical across the EVQ and YaRN
arms. Reasons: matches your prior operating point; keeps adapter capacity modest
so an EVQ-vs-YaRN gap is not swamped by raw SFT capacity; and your design
explicitly rejects "rank = frequency-channel count", so rank is held fixed, never
swept as a frequency knob. Memory is not the constraint (rank-64 LoRA + optimizer
is <1 GB). Only if the readout-conversion diagnostics show readout capacity is
the bottleneck, add a rank-128 or O+MLP arm as a SECONDARY, matched arm.

### lr -- 1e-4, matched, fresh-adapter regime

This is a fresh LoRA on a new model, not a 32-step continuation of an existing
adapter, so the 8B recipe's 2e-5 is too low. Use lr 1e-4, linear warmup ~5% of
steps, cosine decay to ~10%, weight decay 0.0, max grad norm 1.0, BF16. Lock lr
from a small dev set {5e-5, 1e-4, 2e-4} and use the SAME lr for both arms. If the
arms differ sharply in lr sensitivity, that is itself a reportable finding, but
the primary comparison keeps lr matched.

### steps / batch / budget -- report in tokens

microbatch 1 (memory-bound at long L), gradient accumulation to effective batch
8-16; benchmark the accumulation layout on 2-4 non-claim steps, lock it, and use
the same layout for both arms. Target ~500 optimizer steps (~4k-8k sequence
passes), checkpoint every ~50 steps, log the EVQ-vs-YaRN first-token margin /
gold rank / EM trajectory, and early-stop on plateau. Report physical and
supervised tokens separately, never "N steps". Data: deterministic synthetic
retrieval (nonce KV, last-write-wins, two-hop) plus natural extractive QA,
answer-only CE, chat-templated, matched prompt/target IDs across arms, with
source-query distances scaled to the 16-32K training window.

## Gates and kill conditions

- In-range learning gate: both arms must learn the <=native-length task first
  (pair consistency above threshold). If only one learns, report a trainability
  difference, not equal-capability transfer.
- Primary contest: EVQ-minus-YaRN first-token margin and EM in the far region
  (64K/128K), clustered bootstrap over semantic groups, mean effect >=5 points,
  >=2 task families agree, CI lower bound >0.
- Negative is informative: if EVQ with matched coverage and matched mscale does
  not beat a tuned YaRN graft in the far region, that bounds the external-validity
  claim (the 8B train-short advantage does not automatically beat production
  YaRN). Report it; do not sweep sparse/tau post hoc.
- Single-seed is discovery only; confirm any positive with 3 new seeds before a
  claim.

## Reproduce

`python scripts/analysis/industrial_128k_feasibility.py` prints the
coverage-anchor validation, the tau study, and the VRAM tables. Before
launch, replace each MODELS entry with the exact values from the target model's
`config.json` (layers, d_model, n_kv, head_dim, d_ff, vocab, rope_theta,
original_max_position_embeddings) -- the memory and tau numbers are only as
correct as those fields.
