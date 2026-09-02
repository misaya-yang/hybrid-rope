# Headwise factorization narrows the gap but does not solve the Native--long conflict

- **Date:** 2026-09-02
- **Status:** completed mixed result; **not SOTA and not a deployable method**
- **Model:** released `OLMo-2-0425-1B-Instruct`
- **Intervention:** frozen Transformer weights; 1--512 learned layer/head scalars over frozen
  frequency directions
- **Primary evidence:** local-only raw/state/script bundle, intentionally excluded from Git;
  canonical hashes are recorded below
- **Decision:** stop GPU development after this owner; preserve the positive headwise signal and
  the two decisive failure modes

## 1. Executive verdict

This round establishes a useful but bounded result.

1. The assumption that every layer and head must share one frequency table is a real source of
   lost capacity. A 256-scalar headwise clock improves official HotpotQA-200 F1 from `0.21169`
   for frozen `log_s4` to `0.24237`. Adding a second headwise range coordinate reaches `0.25223`.
2. The improvement is not universal. On 2WikiMQA the two-axis arm reaches `0.26651`, but on
   Qasper it falls to `0.18853` versus `0.20095` for frozen `log_s4`.
3. The two-axis arm is competitive with official YaRN-4 over the three evaluated natural tasks,
   but does not beat it: macro F1 is `0.23576` versus `0.23817`, with paired stratified-bootstrap
   difference `-0.00241 [-0.03372, 0.02885]`.
4. The competitive log-start arm fails Native retention. Its 4K PG-19 PPL retention is `0.77138`,
   below the registered `0.875` threshold.
5. Starting from exact Native produces a sparse field and improves 4K PG-19 NLL over Native,
   but fails long QA almost completely: HotpotQA F1 is `0.02560`, with only `8/200` EOS
   terminations.
6. Free headwise gain is not the missing solution under this protocol. It lowers teacher-forced
   training loss from `3.5612` to `2.3105`, yet reduces HotpotQA F1 from `0.24237` to `0.19439`
   and EOS termination from `178/200` to `108/200`.

Therefore the round supports **head-specific allocation/range specialization as a useful method
axis**, but it does not solve the joint Native/long objective and does not establish SOTA.

## 2. Registered model class

Let the frozen Native inverse-frequency table be `omega_N`, and let `m_k` be the frozen movement
profile that defines the existing `log_s4` table:

\[
\omega^{\log4}_k=\omega^N_k 4^{-m_k}.
\]

The first ladder varies only the amount of this same movement used by each layer/head:

\[
\omega_{\ell h k}=\omega^N_k4^{-\alpha_{\ell h}m_k}.
\]

The complexity ladder was:

| Scope | Learned scalars | Meaning |
| --- | ---: | --- |
| shared | 1 | one `alpha` for the whole model |
| layer | 16 | one `alpha` per layer |
| head | 256 | one `alpha` per layer/head |
| head + free gain | 512 | one `alpha` and one Q/K amplitude per layer/head |

The second axis uses the exact official Transformers YaRN-4 table as one additional frozen
direction:

\[
\log\omega_{\ell h}
=\log\omega^N-\alpha_{\ell h}m\log4
+\beta_{\ell h}\log\!\left(\frac{\omega^{\rm YaRN4}}{\omega^{\log4}}\right),
\qquad
g_{\ell h}=1.1386294361^{\beta_{\ell h}}.
\]

The YaRN gain is analytically tied to `beta`; it is not a separately learned amplitude. At
`alpha=1, beta=0` the model is exactly `log_s4/unit`. At `alpha=0, beta=0` the corrected
Native-start implementation is exactly Native.

This is still a static, prefill-time field. It does not route by token, task, request length, or
decode step, and it preserves the ordinary D128 KV cache.

## 3. Protocol and identity

- Model config SHA-256:
  `0d15ebb6cb8d998513b46ef337214176a6fd59fe5f16b30387c70d5f87795a9c`.
- Model weight SHA-256:
  `36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f`.
- Frozen `log_s4` float32 table SHA-256:
  `56ddfae2800d4bbf9e6bd2d20bae751edc9865dcbaf641c7c8c4f1d7f1c15e5b`.
- Official YaRN-4 float32 table SHA-256:
  `cc9da456982ffce5ca0558e9ea661abc4a880ec002179ce6b9149d45aa4a016c`.
- Training data manifests:
  `11433edaa494966ec1c08fc19f29ac48ccfe07a461928137d133c685d0f77849`
  at 4K and
  `375f4066058ef5ceddefeacf0eeddf236306d51939d0d95fea5b50be5943dec8`
  at 16K.
- Training: 96 steps, deterministic `LLLSS` schedule (60% physical 16K, 40% physical 4K),
  both paired variants per step, all Transformer weights frozen.
- Log-start learning rate: `3e-3`. Native-start learning rate: `3e-2`, fixed before the run to
  traverse a unit-sized `alpha` coordinate in the same 96-step budget.
- Official natural QA: original LongBench 200-row task files and prompts; middle truncation to
  16,384 input tokens; full greedy generation; task-native maximum generation of 32 tokens for
  HotpotQA/2WikiMQA and 128 for Qasper; official-style QA F1.
- 4K retention: 20 frozen PG-19 documents from formal token manifest
  `74022bf36d444a1735baab72bda0312b9867dd38c9f85ece376049b5f35f66f3`,
  scoring the same final 512 targets per arm.

The log-start and corrected Native-start implementations each passed a 6-cell, full-vocabulary,
bit-exact parity panel at short, 8K, and 16K positions. An earlier attempt to synthesize Native by
composing a delta rotation on top of `log_s4` failed parity and was not trained. The corrected
Native-start arm instead uses Native as its underlying table, so `alpha=beta=0` is a literal
identity path.

## 4. Training behavior is not task behavior

| Arm | Parameters | Last-20 mean loss | 4K last-20 | 16K last-20 | Learned field summary |
| --- | ---: | ---: | ---: | ---: | --- |
| shared `alpha` | 1 | 4.2004 | 3.3822 | 4.7458 | `alpha=1.0246` |
| per-layer `alpha` | 16 | 3.8554 | 3.1170 | 4.3476 | `alpha=0.8955--1.0962` |
| per-head `alpha` | 256 | 3.5612 | 2.8961 | 4.0046 | `alpha=0.8893--1.1085` |
| per-head `alpha+gain` | 512 | **2.3105** | **2.0181** | **2.5054** | `gain=0.8880--1.1346` |
| log-start two-axis | 512 | 3.3688 | 2.6872 | 3.8232 | `beta=0--0.1298`, mean `0.0379` |
| Native-start two-axis | 512 | 4.8825 | 2.0759 | 6.7536 | mean `alpha=0.0966`, mean `beta=0.1073` |

The free-gain arm is the strongest warning in the round: its training loss is by far the lowest,
but its complete generated answers are worse. The extra amplitude is a teacher-forced shortcut in
this protocol, not evidence of better long-context computation.

## 5. Complete natural-task results

### 5.1 Scope ladder on HotpotQA-200

| Arm | F1 | EM | EOS / 200 |
| --- | ---: | ---: | ---: |
| Native | 0.01942 | 0.015 | 8 |
| frozen `log_s4/unit` | 0.21169 | 0.130 | 178 |
| shared `alpha` | 0.22351 | 0.145 | 183 |
| per-layer `alpha` | 0.22437 | 0.135 | 171 |
| per-head `alpha` | 0.24237 | 0.140 | 178 |
| per-head `alpha+gain` | 0.19439 | 0.110 | 108 |
| log-start headwise two-axis | 0.25223 | 0.160 | 181 |
| official YaRN-4 | **0.27644** | **0.180** | **194** |
| Native-start headwise two-axis | 0.02560 | 0.015 | 8 |

The per-head clock signal is positive but not individually resolved: per-head minus frozen
`log_s4` is `+0.03068 [-0.00059, 0.06282]`; per-head minus shared is
`+0.01886 [-0.01723, 0.05480]`. The two-axis arm resolves the contrast against frozen
`log_s4`: `+0.04055 [0.00573, 0.07658]`. It does not resolve against YaRN:
`-0.02420 [-0.07896, 0.03064]`.

Free gain is a resolved negative relative to per-head frequency alone:
`-0.04798 [-0.09299, -0.00331]` F1. Its average generated length rises from `9.10` to
`18.50` tokens and `92/200` rows hit the 32-token limit.

### 5.2 Cross-task comparison

| Arm | HotpotQA F1 | 2WikiMQA F1 | Qasper F1 | Three-task macro |
| --- | ---: | ---: | ---: | ---: |
| frozen `log_s4/unit` | 0.21169 | 0.25210 | **0.20095** | 0.22158 |
| official YaRN-4 | **0.27644** | 0.26474 | 0.17333 | **0.23817** |
| log-start headwise two-axis | 0.25223 | **0.26651** | 0.18853 | 0.23576 |

Task-stratified paired bootstrap, 50,000 resamples with seed `20260902`, gives:

- two-axis minus `log_s4`: `+0.01418 [-0.00268, 0.03127]` macro F1;
- two-axis minus YaRN: `-0.00241 [-0.03372, 0.02885]` macro F1.

The two-axis arm therefore reaches the same observed three-task frontier as YaRN within sampling
uncertainty, but moves the trade-off: it is lower on HotpotQA, approximately tied on 2WikiMQA,
and higher on Qasper. This is not task-uniform superiority.

## 6. Native retention

| Arm | 4K PG-19 tail NLL | PPL | PPL retention vs Native |
| --- | ---: | ---: | ---: |
| Native | 2.97105 | 19.512 | 1.0000 |
| frozen `log_s4/unit` | 3.30362 | 27.211 | 0.7171 |
| official YaRN-4 | 3.38844 | 29.620 | 0.6588 |
| log-start headwise two-axis | 3.23062 | 25.295 | 0.7714 |
| Native-start headwise two-axis | **2.92585** | **18.650** | **1.0462** |

The log-start arm is better than `log_s4` on all `20/20` documents, but fails the registered
retention threshold. The Native-start arm is better than Native on all `20/20` documents; its
paired NLL difference is `-0.04520 [-0.06238, -0.02993]`.

The Native-start success is only short-side success. Its Hotpot result is essentially Native and
its 16K training loss remains high. This directly exposes the unsolved conflict:

> Starting near the long solution yields useful long computation but unacceptable Native cost;
> starting at exact Native preserves or improves Native likelihood but the current scalar-only
> objective cannot bootstrap the long computation.

## 7. What the result says about the shared-table hypothesis

The result supports a weaker and more useful statement than “per-head tables solve the problem.”

- A single shared movement coefficient is too restrictive: the headwise and two-axis point
  estimates improve complete natural QA.
- Layer-level factorization is not enough under this protocol; its Hotpot F1 is almost identical
  to the shared scalar.
- Different tasks prefer different allocation/range mixtures. A single global table/gain remains
  a compromise across task circuits.
- Nevertheless, headwise clocks alone do not remove the Native--long conflict. Initialization
  selects which side of the Pareto frontier the optimizer reaches.
- The current evidence does not justify arbitrary per-head, per-frequency tensors. The positive
  result uses only two scalars per layer/head and frozen global directions.

## 8. Scale-flow theory audit

The proposed scale-flow coordinate is useful:

\[
x_k=-\log\omega_k,\qquad \tau=\log S,\qquad \frac{dx}{d\tau}=v_\theta(x).
\]

It correctly identifies current `log_s4` as the constant-flow special case
`x(\tau)=x(0)+m\tau`. It is also a clean way to require one constructor to emit one static table
for any deployment horizon.

It is **not yet a derived theory or method**, for four reasons.

1. Any smooth `S -> table` path can be rewritten as a non-autonomous ODE. Along one non-self-
   intersecting Native orbit, it can also be embedded in an autonomous field. ODE notation alone
   does not reduce the search space.
2. A semigroup is behaviorally non-identifying when only the final static table is installed.
   The model depends on the endpoint tensor, not the integration path. Existing `log_s4` already
   satisfies exact composition, while straight continuation to `s8` still fails; composition is
   therefore not sufficient for LM quality.
3. Existing `s2` and `s4` evidence does not identify curvature. Reusing the same frozen `m`
   makes them exactly collinear in `(log-frequency, log-scale)` space. The separate successful
   `s2` owner uses a different C2 construction and cannot be spliced into a curvature estimate.
4. `M_theta(x)` is not yet a defined or validated functional metric. A Hessian/Fisher
   implementation would re-enter a class whose prior attention-Fisher ordering already failed,
   unless it first establishes a distinct finite-step prediction contract.

The strongest justified status is:

> Continuous scale-flow is a candidate low-dimensional, multi-scale constructor and a useful
> problem formulation. The current evidence does not establish a curved trajectory, a canonical
> vector field, or a solution to arbitrary-horizon extension.

If this project is resumed, flow work should begin with a CPU identifiability gate using matched
same-family anchors and an off-orbit composition test. It should not begin with another GPU table
or an unrestricted 64-dimensional vector field.

## 9. Claim table

### Established in this protocol

- Both corrected factorized implementations have exact zero-delta parity.
- Per-head frequency clocks have a positive Hotpot point estimate over shared/layer scopes.
- The headwise allocation+range arm significantly improves Hotpot F1 over frozen `log_s4`.
- Free headwise gain significantly harms complete Hotpot generation despite much lower
  teacher-forced loss.
- Log-start and Native-start reach opposite sides of the Native--long frontier.
- No tested arm simultaneously passes the 4K retention gate and produces useful Hotpot long QA.

### Supported but not established

- Learned task circuits may require different per-head dilation/range mixtures.
- The all-layer/all-head shared table is one important ceiling, but not the only ceiling.
- Cheap weight co-adaptation or a better structured training objective may be needed to connect
  the two observed basins.

### Not supported

- SOTA or uniform superiority to YaRN.
- A universal per-head law or cross-checkpoint transfer.
- A nonlinear scale-flow trajectory.
- A canonical `M_theta(x)` metric.
- Stable 8x/16x/32x extension.
- A zero-training result: the factorized fields in this report were learned.

## 10. Final decision

This round ends method development rather than opening another sweep.

- Preserve the log-start two-axis arm as the strongest long-capability candidate and the
  Native-start arm as the decisive retention/bootstrapping counterexample.
- Keep free headwise gain closed under the tested teacher-forced objective.
- Do not promote scale-flow from framing to method without the CPU identifiability gate.
- Do not claim SOTA. The best three-task macro is still the official YaRN-4 point estimate, and
  the closest headwise arm fails Native retention.
- No further GPU experiment is authorized by this report.

## 11. Evidence map

The raw bundle is stored locally outside the repository and must not be committed. It contains six
training state/log/receipt triplets, eight compact raw/result evaluation bundles, and the exact
executed scripts. The tracked report retains the result-defining hashes without recording a private
machine path.

Key identities:

- log-start two-axis state:
  `0b115d4497faefeeffd454dade2dcba93b2bc3c296017d67ac5f4f98989090ef`;
- Native-start state:
  `79ca8dbb130ecff110e2da13dbf929a2d9f6339f9108194620834de982314450`;
- Hotpot scope-ladder raw rows:
  `6fcf6e9904b67321126c686cefa1c22fec0fd50d4b6452ce6360a698f35bcba3`;
- Hotpot / 2Wiki / Qasper log-start two-axis raw rows:
  `523cd1ceb4472b2f17c4974a077d2fb490c3f836c4bd4c236a72e6a7aa57bf16`,
  `dd600cb5339945624e94f0d296b5a113807d570f34b2fd997d6fcc2adaf8910f`, and
  `0fa4f3edbf1ddfa64bf5cd3cf9349504c7caf0e59bcc8e4d0efaca9c45dcaa2a`;
- log-start and Native-start PG-19 raw rows:
  `3f23022bf71e61608debb31c8a4d52b255ff98a2edaba699dd5b0653c119d408` and
  `e8c6c64aba8180704b46448576a743eb0af35c128da60c251c58b3afa893ea75`;
- Native-start Hotpot raw rows:
  `f91e01997b94a49da5b5a66e4505e64327a45f769d7c826997d0f90a3d98a7b5`.

Local archive verification checked every training state against `state_sha256`, every training log
against `train_log_sha256`, every receipt against an archived script hash, and every evaluation raw
file against the hash in its result JSON: `42` files, all checks passed.
