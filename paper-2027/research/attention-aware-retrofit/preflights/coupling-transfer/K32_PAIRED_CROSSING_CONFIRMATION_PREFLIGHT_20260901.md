# K32 independent paired crossing confirmation

## Evidence and question

The completed same-family baseline panel preserves a strong physical-versus-
Native short/long contrast, but the historical physical-minus-index differences
are only `-.0550` at 32K and `+.0675` at 64K; their paired intervals include
zero. The owner is
[`QWEN_K32_MATCHED_S2_BASELINE_RECEIPT_20260901`](../../evidence/QWEN_K32_MATCHED_S2_BASELINE_RECEIPT_20260901.json).
Do not explain this particular coordinate ordering as established causality.

The single confirmatory question is whether the **unchanged** physical/index
profiles reproduce opposite Native/long ordering on new paired inputs.
This experiment increases information, not the number of candidate methods.

## Frozen construction and inputs

- Existing Qwen2.5-0.5B-Instruct, K32, b=1e6, reference 32768 and target 65536.
- s=2, boundaries `.7382780681078285/.366403835112904`, c=.074 remain frozen.
- Only Native, the original physical table and original normalized-index table.
  The tensor hashes are respectively `6d1e10125bd0468a7cf91c6175a3af31c1bffca24592cf5630f0f8402a8746e3`,
  `b61a58f3e84429e00eaac69a0d9ab43abf89bc193987b2bcbcd3ab3bccd455fb`,
  and `8c19ab976f71d30c6409f78a661209a8535ef9f101e8bf42f5bfce6f7817dc5f`.
- Official RULER commit `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`,
  fixed core-four tasks, fresh seed **202609026**, 80 rows per task at both
  32768 and 65536. All three profiles evaluate identical inputs.
- No new YaRN/method arm, prompt/scorer change, table fit, reference selection,
  gain change or long-result-based profile choice. The previously completed
  Native/YaRN resolvers are controls for the checkpoint, not extra confirmation
  rows. Do not pool the old pilot into the new confirmatory estimate.

## Budget and decision

Exactly 3 profiles x 2 lengths x 4 tasks x 80 rows = **1920 generations**.
The fourfold increase from 20 rows/task reduces sampling standard error by
about half under the same row model; it is not a guarantee of significance.
All arms finish unless there is an execution/resource error or nonfinite loss.
No optional stopping or additional seed after reading a partial score.

Primary contrasts are physical-minus-index at each length. Use paired row
bootstrap within the four fixed task strata, 10000 replicates, seed 202609027.
For a confirmed crossing require the 97.5% marginal percentile interval at
32K to be wholly below zero and the corresponding 64K interval wholly above
zero (Bonferroni over the two prespecified endpoints). Otherwise retain an
unresolved or contradicted ordering, without a rescue profile.

Report all task vectors, Native comparisons and the existing .875 Native
macro-retention point gate separately. Native 1x zero would make the instrument
unresolved. This is independent-input confirmation at one checkpoint, not a
training-seed or K-causal study. It neither validates a Native KL selector nor
opens a new boundary correction. Only after this result is attributed may
the next Native-compatibility mechanism experiment be opened.

## Execution and ownership

Freeze generated input and runtime/table/checkpoint hashes before GPU entry;
preserve full prediction/token/EOS rows. Concurrent existing-model inference
may use the available GPU while staying below the user's 30 GB operating
limit. No model download or deletion, no training, and no shutdown is needed.
The protected 1.485B asset is untouched. Source preparation uses the existing
official-RULER builder, not a new benchmark generator.
