# Low-dimensional coupling-law GPU confirmation (2026-09-01)

## Material passport

- **Status:** `COMPLETE_MIXED / PRACTICAL_NATIVE_GATE_FAILED /
  QWEN_LONG_BEHAVIOR_PASSED`
- **Candidate:** the single pre-frozen `C2_clipped_affine` law from
  [`CPU_LOW_DIM_COUPLING_LAW_20260901.md`](CPU_LOW_DIM_COUPLING_LAW_20260901.md)
- **Training:** zero model updates, zero table updates, zero training tokens
- **Deployment identity:** one static `s=4` table per checkpoint at every
  evaluated length; no Native/long routing; fixed `c=0.074` gain
- **Compact receipt:**
  [`../evidence/LOW_DIM_COUPLING_GPU_RECEIPT_20260901.json`](../evidence/LOW_DIM_COUPLING_GPU_RECEIPT_20260901.json)

## 1. Decision

The two-parameter law preserves the existing 64-point transported long-context
behavior on Qwen at 64K and 128K, but it does **not** pass the registered
Native-retention operating point. It therefore supports a low-dimensional
description of the long-range slot--dilation coupling, not a deployable
`checkpoint + s -> one static table` method under the declared `0.875` Native
retention tolerance.

No `c`, boundary, width, table, or per-model parameter was changed after an LM
endpoint was read. The result does not authorize a rescue sweep.

## 2. Frozen construction and identity

The law is

\[
G_4(x)=\operatorname{clip}\!\left(
  \frac{x_H-x}{x_H-x_L},0,1
\right),\qquad
\omega_i'=\omega_i4^{-G_4(x_i)}.
\]

The OLMo-fitted parameters and Qwen zero-refit geometry were frozen by the CPU
owner before this GPU study. The exact float32 table identities were:

| Checkpoint | Tensor SHA-256 | `.npy` SHA-256 |
| --- | --- | --- |
| OLMo-2-0425-1B-Instruct | `113089f05b1e01e61bd9c78f06151ad96c1237b4602e0c380175c27d30562264` | `e9e16faad38091800ca09d457b6df848e5a16cc58ec4048dd73f05df002e3700` |
| Qwen2.5-1.5B-Instruct | `e78a0b6a04e2ae39a6004eedf1c11ef5f2250998b3adf2b8d8166274c2d4f14e` | `36c9c1ce69be3bfa2426902ed2ae288f5e45dce6004c92c499b4c3e3542e4700` |

Both tables were positive, strictly decreasing, and used attention scaling
`1.102585782722872`. They were installed before prefill and remained fixed for
the request's KV-cache lifetime.

## 3. OLMo registered Native double gate

### 3.1 PG-19 likelihood

| Static table | 1x PG-19 tail NLL | PPL retention vs Native |
| --- | ---: | ---: |
| Native | `2.971047461` | `1.000000` |
| original 64-point log-s4 | `3.104233813` | `0.875302` |
| two-parameter C2 | `3.109194076` | **`0.870971`** |

The registered maximum candidate NLL was `3.104578854`. C2 missed it by
`0.004615222` NLL and missed the original 64-point profile by `0.004960263`.
The 20 paired C2-minus-64D row differences had mean `0.004960`, median
`0.005222`, minimum `-0.002244`, maximum `0.013003`, and population standard
deviation `0.004514`. The miss is a small broadly distributed offset, not one
corrupt document.

### 3.2 Five-task downstream retention

| Static table | Five-task macro | Retention vs Native |
| --- | ---: | ---: |
| Native | `0.345134307` | `1.000000` |
| original 64-point log-s4 | `0.315833287` | `0.915103` |
| two-parameter C2 | `0.312322714` | **`0.904932`** |

C2 passed the downstream half of the double gate. Its task vector was
`0.409091 / 0.193862 / 0.228095 / 0.518911 / 0.211655` for
2WikiMQA / GovReport / HotpotQA / MultiFieldQA-en / Qasper. The conjunction
still failed because PG-19 retention was below `0.875`. Per the registered stop
rule, OLMo 2x/4x natural and RULER endpoints were not opened **as method-gate
evidence**. They were later run under explicit authorization as post-gate
mechanism diagnostics.

### 3.3 OLMo post-gate long diagnostic

The same frozen table remained near the original 64-point profile:

| Endpoint | original 64-point | C2 | C2 minus 64-point |
| --- | ---: | ---: | ---: |
| 2x PG-19 NLL | `3.083278` | `3.088093` | `+0.004815` |
| 2x six-task macro | `0.307614` | `0.299812` | `-0.007802` |
| 4x PG-19 NLL | `3.081946` | `3.086138` | `+0.004192` |
| 4x six-task macro | `0.260055` | `0.258640` | `-0.001415` |

On fresh RULER-13, C2 scored `0.668397/0.502436` at 8K/16K, versus
`0.667051/0.498590` for the original profile. Its core-four vectors were
`1.00/1.00/.30/.56` and `1.00/.20/.15/.00`, giving macros `.7150/.3375`.
Thus the strict Native miss coexists with near-parity natural long behavior and
full-task long retention on OLMo. These post-gate rows explain mechanism; they
do not convert the registered deployment verdict into a pass.

## 4. Qwen long-context post-gate diagnostic

The CPU preregistration would have stopped after the OLMo gate failure. To
separate that operational failure from the scientific question "does the
two-parameter compression preserve transported long behavior?", the already
frozen Qwen table was subsequently evaluated as a **post-gate diagnostic**.
These rows cannot turn the registered method verdict into a pass.

| Qwen endpoint | Existing 64-point transport | C2 | Diagnostic floor | Verdict |
| --- | ---: | ---: | ---: | --- |
| 64K core-4 | `0.6725` | **`0.6775`** | `0.6225` | pass |
| 128K core-4 | `0.5450` | **`0.5725`** | `0.5375` | pass |

C2's per-task vectors were:

- 64K: `1.00 / 0.65 / 0.15 / 0.91`;
- 128K: `1.00 / 0.35 / 0.10 / 0.84`.

At 128K, single-key, multikey-2, and multikey-3 exactly matched the 64-point
transport's `1.00 / 0.35 / 0.10`; VT improved from `0.73` to `0.84`. Thus the
current behavior does not require the isolated spikes in the original 64-slot
movement vector. This is evidence on one second `K=64` checkpoint, not a
universal law or a proof that `x` is the unique causal coordinate.

## 5. Qwen 32K paired Native-window diagnostic

The long-context diagnostic passed before the 32K dataset was generated.
Consequently this section is a descriptive paired follow-up, not untouched
confirmation. No method parameter changed. The data used the same official
RULER commit, Qwen tokenizer, seed `20260822`, four core tasks, and 20 rows per
cell; its manifest SHA-256 is
`a5fd30cf5e53be7a9e447dc6306eff50138771f4e12698510f365322fc97c98f`.

| Qwen 32K table | single | multikey-2 | multikey-3 | VT | Macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native | `1.00` | `0.85` | `0.50` | `0.93` | `0.8200` |
| static C2 | `1.00` | `0.75` | `0.30` | `0.80` | `0.7125` |
| Qwen self 64-point | `1.00` | `0.75` | `0.35` | `0.86` | `0.7400` |
| OLMo-to-Qwen 64-point transport | `1.00` | `0.60` | `0.35` | `0.85` | `0.7000` |

The macro retention is `0.868902`, below the reused `0.875` tolerance; the
corresponding candidate floor was `0.7175`, missed by `0.0050`. Qwen therefore
repeats the OLMo pattern: strong long-context preservation with a small but
gate-changing Native-window cost.

The self profile passes the reused retention floor, whereas both C2 and the
transported OLMo 64-point profile fail. This prevents attributing the Native
gap merely to low dimensionality: it is checkpoint-specific. A parameter-free
monotone minorant of the Qwen self movement removes only isolated non-monotone
spikes and reproduces the self-profile 32K macro `0.7400`; it then scores
`0.6525/0.5925` at 64K/128K. That is a useful engineering static profile, not a
literal two-parameter universal law.

## 6. What this changes

### Supported

- The current 64-dimensional movement profile is not behaviorally required in
  full detail for Qwen long-context capability: the frozen two-parameter law
  reaches parity or better at 64K and 128K without Qwen refitting.
- Low movement-vector reconstruction error was behaviorally meaningful for the
  tested long endpoints, rather than merely visually or geometrically close.
- OLMo post-gate natural and complete RULER-13 rows independently preserve the
  same low-dimensional long behavior.
- Ordered slot--dilation coupling remains the relevant object; this experiment
  does not reduce the result to an unordered frequency multiset.

### Rejected under the declared operating point

- C2 is not the final one-table Native-to-long deployment law. It fails OLMo
  1x PPL retention and Qwen 32K core-task retention under the same `0.875`
  tolerance.
- The failures must not be hidden with Native/long routing or repaired by a
  post-outcome search over `c`, `x_H`, `x_L`, another curve, or a third
  parameter.

### Still open

- Whether a theory-derived coupling can preserve this low-dimensional long
  behavior while recovering the small Native-window margin.
- How to derive checkpoint-specific Native compatibility without fitting an LM
  outcome. The subsequent K32 holdout shows that physical `x` still improves
  64K while failing Native 32K; see
  [`K32_FINITE_K_COUPLING_ANALYSIS_20260901.md`](K32_FINITE_K_COUPLING_ANALYSIS_20260901.md).
- Whether the fixed gain can be removed in a different construction. This
  study does not revisit the prior gain-free failure.

This result selected no rescue candidate. The subsequent exact finite-cell
projection also failed its CPU entrance condition and was not launched.
