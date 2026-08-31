# Native-or-frozen-s4 zero-training retrofit

- **Date:** 2026-08-23
- **Status:** complete internal owner; bounded policy endpoint routed to
  Appendix F.2
- **Decision:** retain the binary Native/s4 session policy; reject the
  stateless boundary-slope operator and the smallest-covering-profile router
- **Model:** released OLMo-2-0425-1B-Instruct, 1.485B parameters
- **Training:** zero learned parameters, zero training tokens

**Causal scope.** This owner evaluates the complete practical intervention:
one deterministic long frequency tensor, one fixed long attention amplitude,
and one session-static Native/long route. It does not own a pure
interior-allocation effect. That effect is isolated by
[`SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`](SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md),
while the frequency-by-gain interaction remains owned by
[`JOINT_MECHANISM_REPORT_20260822.md`](JOINT_MECHANISM_REPORT_20260822.md).
The global variable/stage grammar is
[`../../ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md`](../../ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md).

## 1. Conclusion

The practical deployment problem does not require the model to know an
experimenter's `L_target`. Freeze the specific long profile that passed the
registered deployment gate (`s=4` here), then choose once before prefill:

```text
required_tokens = prefill_tokens + max_new_tokens
required_tokens <= L_native  -> exact Native route
required_tokens >  L_native  -> frozen budgeted-s4 route
```

The selected route remains fixed for the lifetime of that request's KV cache.
The rule uses the model's own `L_native`, so a model native to 32K receives the
same relative decision boundary rather than an OLMo-specific 4K constant.  It
does not distinguish 8K from 16K and does not receive `L_target`; the only long
operator is the deployment's fixed maximum profile.

On the same RTX 4090, checkpoint, data rows, decoding code, and precision, this
binary method scored `0.7175/0.4075` on core-4 RULER at 8K/16K, versus
`0.2225/0.0125` for official Transformers YaRN configured once at factor four.
On the full 200-row 16K 2Wiki run it scored token F1 `0.2666`, versus `0.2569`
for YaRN factor four.  It preserves every formal 1x output exactly (`120/120`)
and matches the frozen s4 oracle on every 4x output (`138/138`).

The formal natural-context result is more nuanced but still positive.  The
binary method improves the six-task 2x macro from `0.2127` for YaRN factor four
to `0.2858`, and PG-19 tail NLL from `3.4404` to `3.1060`.  At 4x, its six-task
macro is `0.2497` versus YaRN's `0.2559`, while its PG-19 tail NLL is better
(`3.0977` versus `3.7955`).  The result is therefore not a claim of universal
task dominance.

## 2. Why the first target-free construction failed

The stateless boundary-slope construction preserved each frequency's phase at
the Native boundary and changed only its slope afterward.  It scored exactly
`0.0000` at both 8K and 16K on core-4 RULER.

The failure is consistent with a cross-boundary relative-phase mismatch.  For
a long query `q > L_native` and a Native-region key `k <= L_native`, a
piecewise phase has

```text
theta(q) - theta(k)
= omega L_native + omega' (q - L_native) - omega k.
```

The key coefficient remains `-omega`, whereas a stationary long table requires
`-omega'`.  Preserving absolute phase at one boundary does not preserve the
query-key relative phase used by attention.  This does not prove that every
nonlinear or custom relative-position operator must fail; it rejects this
Flash/KV-compatible absolute-phase construction.

## 3. Why smallest-covering-profile routing was also rejected

A second policy selected the smallest frozen profile covering the observed
`prefill_tokens + max_new_tokens`: Native, s2, or s4.  Its CPU contract matched
all `386/386` frozen natural-context rows without an `L_target`, and a one-row
per-cell GPU parity run matched the target-aware oracle on `20/20` cells.

On the full 200-row 16K 2Wiki evaluation, however, it scored only `0.2473`.
Its route counts were Native/s2/s4 = `24/121/55`.  On the same hardware and
inputs, fixed s4 scored `0.2774`.  Coverage is therefore a feasibility rule,
not a performance selector.  Choosing s2 merely because a request fits inside
8K discards useful s4 behavior.

## 4. Protocol

### 4.1 Shared identity

- Checkpoint weight SHA-256:
  `36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f`.
- Token manifest SHA-256:
  `74022bf36d444a1735baab72bda0312b9867dd38c9f85ece376049b5f35f66f3`.
- Frozen long-table tensor SHA-256:
  `a435d75441444bcea39b73d9cf530005249dc5afdc3cfb5a60fda10ef33312d3`.
- Long attention amplitude: `1 + 0.1 ln(4) = 1.1386294361`.
- Hardware/runtime: RTX 4090 48GB, PyTorch 2.8.0+cu128, BF16 autocast,
  Flash-only SDPA, greedy generation.

### 4.2 Formal natural-context matrix

The frozen token manifest contains Qasper, NarrativeQA, MultiFieldQA-en,
HotpotQA, 2WikiMQA, GovReport, and PG-19 at relative 1x/2x/4x buckets.  Each
available task-length cell uses at most 20 rows.  NarrativeQA 1x is unavailable,
NarrativeQA 2x has 8 rows, and Qasper 4x has 18; all other cells have 20.

Four arms use the same rows:

1. Native at every length;
2. official Transformers YaRN configured once at factor four;
3. the target-aware oracle using Native/s2/s4 by frozen bucket;
4. the deployable binary method using Native at 1x and frozen s4 at 2x/4x.

### 4.3 Full 2Wiki and RULER

The full 2Wiki comparison uses all 200 official rows, one 16K service window,
32 generated tokens, and only one truncated row.  The core-4 RULER check uses
20 rows per task at each length and official task-specific scoring.  RULER is
an engineering/retrieval assay, not unseen-task transfer.

The canonical raw-row audit rechecks token counts after chat-template
rendering. Every 2Wiki arm satisfies `input_tokens + 32 <= 16384`, and every
Qasper arm satisfies `input_tokens + 128 <= 16384`; no completed row exceeds
the physical budget. The method-selection ledger records Qasper as a
post-policy-freeze endpoint and the final 2Wiki policy comparison as
exploratory bundled-policy evidence.

## 5. Results

### 5.1 Formal task macro and PG-19

The task macro averages the available task-level scores; it does not pool
examples across tasks.  PG-19 is reported separately because its endpoint is
tail NLL, where lower is better.

| Method | 1x task macro (5 tasks) | 2x task macro (6 tasks) | 4x task macro (6 tasks) | PG-19 1x | PG-19 2x | PG-19 4x |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 0.3424 | 0.0661 | 0.0210 | 2.9712 | 7.1029 | 7.2050 |
| official YaRN factor 4 | 0.3173 | 0.2127 | **0.2559** | 3.3880 | 3.4404 | 3.7955 |
| target-aware Native/s2/s4 oracle | 0.3424 | 0.2631 | 0.2497 | 2.9712 | **2.9746** | **3.0977** |
| **binary Native/s4** | **0.3424** | **0.2858** | 0.2497 | **2.9712** | 3.1060 | **3.0977** |

The binary method's 2x heterogeneity is important:

| Task | Native | YaRN factor 4 | oracle s2 | binary s4 |
| --- | ---: | ---: | ---: | ---: |
| 2WikiMQA | 0.0904 | 0.3250 | 0.3916 | **0.4786** |
| GovReport | 0.1162 | 0.1657 | 0.1877 | **0.1920** |
| HotpotQA | 0.1047 | 0.3369 | 0.2577 | **0.3621** |
| MultiFieldQA-en | 0.0387 | 0.2690 | 0.3258 | **0.3265** |
| NarrativeQA | 0.0069 | 0.0312 | 0.2125 | **0.2500** |
| Qasper | 0.0396 | 0.1485 | **0.2031** | 0.1056 |

S4 is not uniformly better than s2 in this small matrix: the 20-row 2x Qasper
cell is the clearest counterexample.  It is not the definitive Qasper result;
the matched full-200 evaluation below reverses its direction relative to YaRN.
The claim is that one frozen s4 profile gives the stronger deployable aggregate
and removes target-length routing, not that it is optimal for every task.

### 5.2 Full 200-row 2Wiki, same hardware

| Operator | Token F1 | Normalized exact | Native/s2/s4 route counts |
| --- | ---: | ---: | ---: |
| smallest-covering Native/s2/s4 | 0.2473 | 0.175 | 24 / 121 / 55 |
| official YaRN factor 4 | 0.2569 | 0.200 | n/a |
| **binary Native/s4** | **0.2666** | 0.205 | 24 / 0 / 176 |
| fixed budgeted s4 everywhere | **0.2774** | **0.220** | 0 / 0 / 200 |

The binary policy pays `0.0108` F1 relative to applying s4 even to short rows,
but it retains the exact Native short path by construction and remains `+0.0097`
above YaRN factor four.

### 5.3 Full 200-row Qasper, same hardware

| Operator | Token F1 | Normalized exact | Native/s4 route counts |
| --- | ---: | ---: | ---: |
| official YaRN factor 4 | 0.1803 | 0.110 | n/a |
| **binary Native/s4** | **0.2457** | **0.115** | 70 / 130 |

All 200 input rows match between arms by source-row hash, input hash, input
token count, and truncation status; both arms truncate the same three rows and
use the same evaluator hash. Binary wins on both routing subsets separately:
`0.3454` versus `0.2517` on the 70 Native-routed rows and `0.1921` versus
`0.1418` on the 130 s4-routed rows. The paired row difference is `+0.0654`
token F1 with a 4,000-resample evaluation-row interval `[+0.0219,+0.1102]`.
This interval conditions on one checkpoint and one task; it is not model- or
task-population uncertainty.

Raw `results.json` SHA-256 values are
`e8101c8cdc00c9b3c3394d66736de8e87a4cac9bb4b9a9421a57e57ba3179240`
for YaRN and
`04ea9e2aa503378e2a8008f7e20b16bded1820d2c054c588d52eaa44ea43ccfe`
for binary. They were independently copied and rehashed before this owner was
updated.

### 5.4 Core-4 RULER, same hardware

| Operator | 8K | 16K |
| --- | ---: | ---: |
| stateless boundary-slope target-free | 0.0000 | 0.0000 |
| official YaRN factor 4 | 0.2225 | 0.0125 |
| **binary Native/s4** | **0.7175** | **0.4075** |

Binary-method per-task scores are:

| Length | single-key | multikey-2 | multikey-3 | variable tracking |
| --- | ---: | ---: | ---: | ---: |
| 8K | 1.00 | 0.95 | 0.30 | 0.62 |
| 16K | 1.00 | 0.60 | 0.00 | 0.03 |

### 5.5 Confirmation-only RULER-13 breadth

After freezing the method on core-4, a fresh complete 13-task dataset was
generated at the same seed and protocol. All eight regenerated core-4 cell
hashes match the selection data exactly. The nine remaining tasks were then
evaluated without changing the table, amplitude, routing rule, checkpoint, or
metric. They are confirmation-only and did not reopen method selection.

| Unseen task | Native 8K | YaRN-4 8K | Binary 8K | Native 16K | YaRN-4 16K | Binary 16K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NIAH single-2 | 0.000 | 0.250 | **1.000** | 0.000 | 0.050 | **0.950** |
| NIAH single-3 | 0.000 | 0.150 | **1.000** | 0.000 | 0.000 | **0.950** |
| NIAH multikey-1 | 0.000 | 0.350 | **0.650** | 0.000 | 0.100 | **0.750** |
| NIAH multivalue | 0.000 | 0.300 | **0.850** | 0.000 | 0.050 | **0.725** |
| NIAH multiquery | 0.000 | 0.325 | **0.863** | 0.000 | 0.038 | **0.638** |
| CWE | 0.000 | **0.015** | 0.005 | 0.000 | 0.010 | **0.030** |
| FWE | 0.000 | 0.367 | **0.567** | 0.000 | 0.117 | **0.550** |
| QA-1 | 0.000 | 0.250 | **0.600** | 0.000 | 0.200 | **0.500** |
| QA-2 | 0.000 | 0.200 | **0.400** | 0.000 | 0.150 | **0.350** |
| **Unseen-9 macro** | **0.0000** | **0.2452** | **0.6594** | **0.0000** | **0.0794** | **0.6047** |

Binary versus YaRN wins/ties/losses are `8/0/1` at 8K and `9/0/0` at
16K. Combining the untouched core-4 rows with the nine confirmation tasks
gives the complete task-macro result:

| Method | RULER-13 8K | RULER-13 16K |
| --- | ---: | ---: |
| Native | 0.0000 | 0.0000 |
| official YaRN factor 4 | 0.2382 | 0.0588 |
| **binary Native/s4** | **0.6772** | **0.5440** |

Across all 13 tasks, binary versus YaRN wins/ties/losses are `12/0/1` at
8K and `12/1/0` at 16K. This broadens the retrieval/task-family result; it
does not turn RULER into evidence of unseen natural-task transfer.

### 5.6 Static-profile and fresh-likelihood follow-up (2026-08-31)

A later zero-training development run measured the same frozen `s4` table and
gain as one request-static profile rather than routing 1x requests to Native.
The table and gain remain those in §4.1. The FineWeb-Edu development,
selection, and confirmation splits contain `64/64/128` documents and have
exact R0-prefix overlap `0/0/0` after the R0 token owner excluded one matched
source row.

| Split | s4−Native prefix | s4−Native dense | s4−Native tail | s4−YaRN prefix | s4−YaRN dense | s4−YaRN tail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D | +0.1180 | −2.9921 | −4.3159 | −0.0699 | −0.1700 | −0.4735 |
| S | +0.1137 | −3.0068 | −4.4311 | −0.0567 | −0.1838 | −0.4699 |
| T | +0.1138 | −3.0695 | −4.5280 | −0.0461 | −0.1609 | −0.4569 |

All T paired 95% intervals exclude zero: s4−YaRN is
`[-0.0558,-0.0379]` for Native-prefix, `[-0.1743,-0.1476]` for long-dense,
and `[-0.4891,-0.4247]` for far-tail NLL. Thus the fixed profile has a stable
in-window NLL cost against Native, while improving all three likelihood
endpoints against official YaRN-4 on three disjoint splits. This is a
`YARN_ANCHORED_PARETO`-shaped system result, not absolute no-harm and not a
pure-`z` attribution.

The core-four RULER 4K run used the static s4 profile for the complete request
and KV-cache lifetime:

| Profile | single-key | multikey-2 | multikey-3 | variable tracking | macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native (existing diagnostic) | 1.00 | 0.85 | 0.60 | 0.03 | 0.620 |
| **static s4, gain c=0.10** | **1.00** | **0.90** | 0.15 | **0.61** | **0.665** |

The higher macro does not mean uniform in-window improvement: multikey-3
falls sharply while variable tracking rises sharply. It does establish that a
single model-load-time table is an engineering-viable option rather than a
generic 4K capability collapse. Because the table is installed before prefill
and never changes, the run uses the standard stationary phase
`omega*(q-k)` and normal cached generation; it does not switch coordinates
under an existing KV cache. A deployment requiring exact Native short behavior
should instead route before prefill between fixed Native and s4 profiles.

A fresh 13-task, 20-row-per-cell matrix then evaluated one fixed profile at
every length. Task macros at 4K/8K/16K were:

| Profile | 4K | 8K | 16K | all-cell macro |
| --- | ---: | ---: | ---: | ---: |
| Native | **0.7131** | 0.0000 | 0.0038 | 0.2390 |
| official YaRN-4 | 0.4314 | 0.2431 | 0.1056 | 0.2600 |
| **static s4, gain c=0.10** | 0.7036 | **0.6662** | **0.5442** | **0.6380** |

This independent generated-row matrix confirms that the static profile remains
close to Native at 4K and broadly useful at 8K/16K. It still contains task-level
losses and is RULER-family adaptation/capability evidence, not unseen natural
task transfer.

A protected local-gap redistribution (`rev28`) produced a smaller, workload-
specific trade. On the targeted multikey-2/multikey-3/VT subset its macro moved
from `0.4833` to `0.5011` on one seed and from `0.5378` to `0.5411` on a second.
However, its full RULER-13 macro was `0.6339`, below the frozen s4 value
`0.6380`. On D/T likelihood it worsened prefix and dense NLL by roughly
`+0.004` while improving tail NLL by roughly `-0.004`. This closes the local-gap
table as a default replacement; it is only evidence of a narrow task exchange.

Two follow-up routes stopped:

- a Native-support segmented phase-chord `z` candidate kept its 1x NLL cost to
  `+0.0018` and improved far-tail NLL by `−0.1050`, but regressed long-dense
  NLL by `+0.0457`; its RULER single-key screen scored `0/20` at both 8K and
  16K, so the remaining cells were not opened;
- increasing the s4 gain coefficient from `0.10` to `0.12` reduced unseen-nine
  RULER macros from the frozen `0.6594/0.6047` to `0.6487/0.5846` at 8K/16K.
  Larger coefficients also worsened the D likelihood endpoints, so scalar-gain
  tuning is closed for this profile;
- an exploratory factor-eight `p=1` table improved 32K core-four macro over the
  arithmetic `p=2` table on two generated-row seeds (`0.3950` versus `0.2300`
  and `0.3175` versus `0.2250`), but its 4K RULER-13 macro was only `0.3593`
  versus Native `0.7131`. It is not an all-length static profile. The subsequent
  scale-law experiment supersedes this route and is owned by
  [`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`](SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md).

The W0 implementation did not serialize the feasibility-mode decision before
F1 started. The measured first W0 branch uniquely implies `ABSOLUTE`, and F1
has no feasible arm under either mode, so this receipt omission does not change
the candidate verdict; it remains an implementation/provenance limitation and
must not be backdated as a preregistered file.

## 6. Runtime and parity receipts

- Binary 1x versus Native: exact generated outputs/NLL for `120/120` rows.
- Binary 4x versus target-aware s4 oracle: exact for `138/138` rows.
- Smallest-covering session policy versus oracle, one row per available cell:
  exact for `20/20` rows.
- CPU observed-request routing: `386/386` frozen rows selected their expected
  profile; the binary policy selected Native for 120 rows and s4 for 266.
- Focused local tests: `24/24` passed for length-conditioned, target-free, and
  context-builder tests; the downstream helper suite plus policy tests passed
  `21/21` before the full 2Wiki run.
- No model parameter was added or trained.

## 7. Evidence boundary and manuscript routing

This is a strong internal result on one released 1.485B checkpoint.  It uses
deterministic zero-training operators, so there are no training seeds to pool,
but evaluation sampling is still limited: most formal cells contain 20 rows.
The 200-row 2Wiki and Qasper results are broader for two tasks only. No
statistical significance or cross-model universality is claimed.

The owner and manuscript were updated together on 2026-08-24. The outward
claim ceiling remains:

> A mature checkpoint can preserve its Native short-context path exactly while
> using one frozen, receipt-bound long-context profile selected only by whether
> the observed request exceeds the model's Native window. On the tested OLMo-2
> checkpoint this zero-training policy is stronger than a one-deployment YaRN
> factor-four control on core-4 RULER and full Qasper, while full 2Wiki and the
> smaller formal LongBench matrix show mixed task-level effects.

The second mature checkpoint with a different Native window is now complete on
the frozen Qwen RULER protocol and is owned by the same-support control report.
The next high-value generalization would be a frozen natural-context protocol
on that checkpoint. It is not currently authorized; another OLMo table/gain
sweep remains stopped.
