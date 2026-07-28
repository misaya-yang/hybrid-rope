# OLMo-2 1.485B Fresh General-Data Q/K Adaptation

## Status

`POST_SUB_RAW_HASH_BACKED / SELECTED MATRIX COMPLETE`

The matched training arms, exact-overlap gate, raw 2WikiMultiHopQA, complete
13-family RULER, factor-2 official Transformers YaRN, and factor-2 repository
fixed-ramp controls are complete. Each capability cell retains full
generations, a run manifest, and hashes.

The wider transform-by-factor-by-benchmark matrix registered at preflight was
stopped after the factor-2 results settled the operator question and the
generic EVQ arm failed the capability/retention decision rule. Native
official-YaRN factor 4 started but was stopped incomplete at `506/520` RULER
rows; it has no `results.json` and no aggregate is promoted. The other
factor-4 and transformed-2Wiki cells were not run. This report promotes only
the completed cells below.

Relevant retained concerns: `RDz6s.1`, `RDz6s.2`, `RzWsa.3`, `RzWsa.4`,
`R27bE.2`, `R27bE.5`, `AC.2`, and `AC.4`.

## Question and direct answer

This experiment asks whether EVQ can be adapted from the untouched mature
OLMo-2 Instruct checkpoint using only fresh Q/K LoRA and generic data, without
inheriting a Q/K/V/O adapter or training on 2Wiki/RULER task-family rows. It
also asks whether official YaRN or the repository fixed-index smooth-ramp
transform changes the Native-versus-EVQ result.

The answer is negative for broad task transfer from generic Q/K adaptation.
EVQ has lower 8K/16K natural-text NLL, but its raw 2Wiki and complete-family
RULER capability remains low and carries a severe 4K cost. The tested
factor-2 official YaRN transform strongly improves the Native-trained adapter
at 8K RULER but not the EVQ-trained composite. The repository fixed ramp
behaves differently from official YaRN and leaves both 8K scores low.

Thus, generic Q/K adaptation can improve long-position probability modeling
without creating broad autoregressive task capability. NLL/PPL and capability
remain separate endpoints throughout this owner.

## Registered matched protocol

| Field | Native arm | EVQ arm |
| --- | --- | --- |
| Base checkpoint | untouched OLMo-2-0425-1B-Instruct | same |
| Actual model size | 1.485B parameters | same |
| Adaptation | fresh Q/K-only LoRA | same |
| LoRA rank / alpha | 64 / 128 | same |
| Trainable parameters | 8,388,608 in 64 tensors | same |
| Trainable projections | all-layer `q_proj` and `k_proj`; no V/O/readout | same |
| Data schedule | LongAlign full-token, LongAlign full-token, Tulu assistant-only | same |
| Steps / global batch | 600 / 8 | same |
| LR / warmup | `1e-4` / 30 steps | same |
| Optimizer | fused AdamW, betas 0.9/0.95, no weight decay | same |
| Physical train length | at most 4,096 tokens | same |
| Position IDs | ordinary contiguous positions only | same |
| Seed | 20260727 | same |
| Active training frequency | Native RoPE | EVQ-Cosh |

The only registered method variable is the frequency table active during
otherwise matched fresh Q/K-only adaptation. The learned LoRA tensors
subsequently diverge as a consequence of that intervention, not as an
additional experimental variable.

Matched receipts:

- initial adapter tensor SHA-256:
  `b1280c2001a3bd1bf588f10838969a5d2225a8d042d466176d865369740280c0`;
- row-selection SHA-256:
  `23e41570f599055fe8d678473a4a3f7da32e038f793d3af2d6e4e1f4a5c2f2cd`;
- Native adapter SHA-256:
  `3b8735c9b0cd6b65238f256ce0d9395c6e835be9ba7583fca485bc1e675c4740`;
- EVQ adapter SHA-256:
  `26157b7b586449bcc1a3ad5ff574ae4428640a047bc3fdad90a40e71d2653c79`.

The schedule executes 400 LongAlign full-token steps and 200 Tulu
assistant-only steps. Both arms consume 19,656,000 dense tokens and
13,729,836 supervised tokens under the same selected rows and order.

## Data and overlap boundary

Training uses prepared views of:

- `zai-org/LongAlign-10k`, revision
  `12f17c4baff1001f0d44c4f8feab09ee2ee8c6dc`;
- `allenai/tulu-3-sft-olmo-2-mixture-0225`, revision
  `d91a0785ade02942520280fb484866fce41e448f`.

The registered exact-token audit found:

- `0/200` exact 2Wiki test-question occurrences in the prepared training
  views;
- `0/780` exact raw or chat-rendered RULER test-prompt occurrences in the
  prepared training views.

This supports “no direct benchmark-family supervision” and
“benchmark-unsupervised evaluation after generic adaptation.” It does not
prove document-, topic-, or upstream-corpus-level non-contamination and
therefore is not labeled clean unseen-task transfer.

## Natural-text NLL/PPL

Single-seed evaluation on 16 held-out natural-text rows per length:

| Length | Native NLL | Native PPL | EVQ NLL | EVQ PPL |
| ---: | ---: | ---: | ---: | ---: |
| 4K | 2.2889 | 9.8642 | 2.8803 | 17.8190 |
| 8K | 3.9317 | 50.9912 | 3.0876 | 21.9245 |
| 16K | 5.1079 | 165.3291 | 3.3819 | 29.4257 |

Interpretation: under the matched generic-data Q/K-only protocol, EVQ pays a
material in-window modeling cost but degrades much less at 2× and 4× context.
These numbers do not establish autoregressive task capability.

## Executed capability matrix

The preflight registered a wider matrix. The following table distinguishes
completed evidence from cells deliberately stopped by the decision gate.

| Benchmark | Component | Evaluation operator | Lengths | n per cell | Status |
| --- | --- | --- | --- | ---: | --- |
| 2Wiki | untouched Native, adapter off | Native | 4K | 200 | complete |
| 2Wiki | Native fresh Q/K | Native | 4K/8K/16K | 200 | complete |
| 2Wiki | EVQ fresh Q/K | EVQ | 4K/8K/16K | 200 | complete |
| RULER13 | untouched Native, adapter off | Native | 4K | 20/family | complete |
| RULER13 | Native fresh Q/K | Native | 4K/8K/16K | 20/family | complete |
| RULER13 | EVQ fresh Q/K | EVQ | 4K/8K/16K | 20/family | complete |
| RULER13 | Native fresh Q/K | official Transformers YaRN, factor 2 | 4K/8K | 20/family | complete |
| RULER13 | EVQ fresh Q/K | EVQ + official-YaRN per-index transform, factor 2 | 4K/8K | 20/family | complete |
| RULER13 | Native fresh Q/K | repository fixed ramp, factor 2 | 4K/8K | 20/family | complete |
| RULER13 | EVQ fresh Q/K | EVQ + repository fixed ramp, factor 2 | 4K/8K | 20/family | complete |
| Native official-YaRN factor 4 | registered wider matrix | official Transformers YaRN | 4K/16K | 20/family | stopped incomplete at 506/520; no aggregate |
| Other factor-4 and transformed-2Wiki cells | registered wider matrix | corresponding transform | 4K/8K or 4K/16K | — | not run |

### 2WikiMultiHopQA

Each cell contains 200 greedy autoregressive generations. Values are
token-F1 / normalized exact / terminal EOS; all values are percentages.

| Component | 4K | 8K | 16K |
| --- | ---: | ---: | ---: |
| Untouched Native, adapter off | 27.64 / 22.00 / 97.50 | not evaluated | not evaluated |
| Native fresh Q/K | 25.47 / 21.00 / 99.00 | 0.13 / 0 / 0 | 0.10 / 0 / 0 |
| EVQ fresh Q/K | 15.56 / 4.50 / 92.00 | 6.49 / 0.50 / 46.00 | 1.11 / 0 / 4.00 |

Deterministic answer-filtered distractor filling produces mean input lengths
of 4,063.96, 8,159.86, and 16,351.86 tokens. This is a controlled
task-family long-QA protocol, not the unmodified LongBench leaderboard
protocol.

EVQ retains more QA signal than Native at 8K, but the absolute result is weak:
only `1/200` rows is exact and terminal EOS is `46%`. At 16K exact match is
zero. The 4K cost is also material. This is not broad downstream capability.

### Complete 13-family RULER

The endpoint uses all 13 families, 20 held-out examples per family-length
cell, greedy autoregressive generation, and the official family-specific
string-match score.

| Component | 4K official macro | 8K official macro | 16K official macro |
| --- | ---: | ---: | ---: |
| Untouched Native, adapter off | 65.16% | not evaluated | not evaluated |
| Native fresh Q/K | 58.53% | 0% | 0% |
| EVQ fresh Q/K | 6.12% | 3.96% | 2.38% |

Generic-data EVQ adaptation produces small non-zero 8K/16K scores but loses
most inherited 4K RULER capability. It does not support broad task transfer
from generic Q/K adaptation.

All family-level official scores are shown below.

| Family | Base 4K | Native 4K | EVQ 4K | Native 8K | EVQ 8K | Native 16K | EVQ 16K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cwe | 0.50% | 1.00% | 0% | 0% | 1.50% | 0% | 1.00% |
| fwe | 48.33% | 41.67% | 23.33% | 0% | 20.00% | 0% | 30.00% |
| niah_multikey_1 | 85.00% | 85.00% | 15.00% | 0% | 10.00% | 0% | 0% |
| niah_multikey_2 | 95.00% | 90.00% | 0% | 0% | 0% | 0% | 0% |
| niah_multikey_3 | 60.00% | 15.00% | 0% | 0% | 0% | 0% | 0% |
| niah_multiquery | 70.00% | 76.25% | 2.50% | 0% | 2.50% | 0% | 0% |
| niah_multivalue | 46.25% | 60.00% | 3.75% | 0% | 2.50% | 0% | 0% |
| niah_single_1 | 100.00% | 100.00% | 5.00% | 0% | 0% | 0% | 0% |
| niah_single_2 | 95.00% | 100.00% | 0% | 0% | 0% | 0% | 0% |
| niah_single_3 | 100.00% | 95.00% | 0% | 0% | 0% | 0% | 0% |
| qa_1 | 70.00% | 60.00% | 10.00% | 0% | 5.00% | 0% | 0% |
| qa_2 | 50.00% | 30.00% | 20.00% | 0% | 10.00% | 0% | 0% |
| vt | 27.00% | 7.00% | 0% | 0% | 0% | 0% | 0% |

### Official YaRN versus repository fixed ramp

These are evaluation-only transforms applied to the completed generic
adapters on the same RULER rows.

| Training substrate + evaluation transform | 4K official macro | 8K official macro |
| --- | ---: | ---: |
| Native fresh Q/K + official Transformers YaRN | 61.06% | 52.19% |
| EVQ fresh Q/K + official-YaRN composite | 11.05% | 6.99% |
| Native fresh Q/K + repository fixed ramp | 60.62% | 1.03% |
| EVQ fresh Q/K + repository fixed-ramp composite | 7.56% | 3.81% |

The two operators are not interchangeable. The tested official YaRN factor-2
transform partially recovers the Native-trained adapter at 4K and yields
`52.19%` at 8K, whereas the repository fixed ramp leaves the Native 8K score
near zero. On the EVQ substrate, neither composite yields broad capability.
These results do not support additional EVQ substrate leverage under official
YaRN.

Operator identities:

| Report label | Exact identity | Realized frequency SHA-256 | Attention scale |
| --- | --- | --- | ---: |
| official Transformers YaRN | Installed Transformers YaRN initializer | `8accc312855e440d24c9a3542a1cbff45a64774460c7aa7fd8513dc26c333039` | `1.0693147181` |
| EVQ + official-YaRN composite | EVQ substrate multiplied by the exact realized official-YaRN per-index scaler | `45c4484495368a8dfc5b4fcc6f6efad446e004782a0625897457c25dad006d01` | `1.0693147181` |
| repository fixed ramp | Repository-defined 20%–90% fixed-index smoothstep scaler; not official YaRN | `1355e594f8e72953c5ee73ac78df5a7779c239c8b7f4273cd2ab05c7e35c6bbf` | `1.0` |
| EVQ + repository fixed-ramp composite | The same repository scaler applied to the EVQ substrate; not official YaRN | `d11ddab909667b882ef59c465ed1a70bb98e25fa278635f1f7067ba3ffa9ed0d` | `1.0` |

The official-YaRN cells ran with Transformers `4.57.6`; the executed
`transformers/modeling_rope_utils.py` SHA-256 is
`55cc0c8cb76f592ab178b4662adcbdf4ad5012bb0b4249c0a96603141f18e9ac`.
The repository fixed-ramp helper SHA-256 is
`814deec59fa7e3ee39becb3174bfb69d0d85fbbbac06b8b46484ef4b3e90d99c`.

## Additional-ablation decision

The registered matrix already covers the decision-critical controls:

1. an untouched-base 4K capability anchor;
2. matched fresh Native/EVQ Q/K-only training from the same base;
3. generic-training/test exact-overlap checks;
4. raw, official-YaRN, and repository fixed-ramp evaluation on the same
   adapters and test rows.

The post-hoc decision rule fails: EVQ does not show a reviewer-facing advantage
on both 2Wiki and complete-family RULER at one target length while retaining
most 4K capability. The preflight READY receipt registered the wider matrix and
identity/runtime stop conditions, but did not pre-register this capability
rule. No second seed, further factor-4 expansion, transformed-QA matrix,
Q-only/K-only, V/O, rank, learning-rate, loss, or data-ratio sweep is
scientifically justified by the completed results. The bounded single-seed
result is final.

## Claim boundary and send gate

- The completed selected cells are `POST_SUB_RAW_HASH_BACKED`; cells stopped
  by the decision rule are not evidence.
- The experiment is single-seed.
- Any 4K deficit must sit next to an 8K positive.
- Any weak or negative 16K result must sit next to the 8K result it limits.
- NLL/PPL cannot substitute for QA or RULER.
- “No direct benchmark-family supervision” is not clean unseen-task transfer.
- Fixed-factor official YaRN is not a fully tuned YaRN sweep.
- The comparison isolates the Native-versus-EVQ training frequency table at
  the method level; it does not decompose which internal Cosh coordinates
  cause the outcome.
- This is not a direct FMRoPE comparison and does not support universal SOTA,
  universal no-harm, or replacement of range-scaling methods.

## Artifact provenance

The selected `10/10` completed components pass the strengthened post-hoc
validator over checkpoint, adapter metadata, training length, independently
anchored realized frequency, recorded evaluator/helper or bound-code identity,
data manifest, cell coverage, unique per-example row identity, physical token
budget, aggregate recomputation, and results/examples/run-manifest hashes.

The EVQ 2Wiki process completed all 4K and 8K rows, then stopped on a CUDA OOM
caused by a competing process before the 16K cell. The registered evaluator
resumed fail-closed: it required an identical run manifest, verified every
completed row's source/QA identity and role, reused the completed rows, and
generated only the missing 16K rows. The final gate requires exactly 600
unique rows and recomputes every aggregate from raw generations.

Core identities:

| Artifact | SHA-256 |
| --- | --- |
| Base checkpoint composite | `36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f` |
| Native fresh-Q/K adapter | `3b8735c9b0cd6b65238f256ce0d9395c6e835be9ba7583fca485bc1e675c4740` |
| EVQ fresh-Q/K adapter | `26157b7b586449bcc1a3ad5ff574ae4428640a047bc3fdad90a40e71d2653c79` |
| Native frequency tensor | `dde15c31724177356ae954d6e11fb337e6fccef56e4520a905cac3f0d9885b34` |
| EVQ frequency tensor | `917a52426b4ac986545c8ec73b115daae3c6515d6b9047f09d30c972ea1a4607` |
| Downstream READY receipt | `b910e5053c188e5f40e663005eeb098ffc1daa4851754e853a98d0d7d75dc334` |
| Strict validation receipt | `5b5541efc9d67e0eec2dc3a4b88a3b3d11613221c553883d77638a73037e5774` |
| Completion receipt | `b425f0d3848a60d0b5c5efb13aff3b549c3bda119e1f5a83f713ee9dad802e0c` |
| Initial EVQ 2Wiki failure log | `8fc81f82a3353408350e0ab55b3c3ae0cf120568672b4217f10940a4c494e012` |
| Frozen EVQ 2Wiki resume log | `763167562ac722fc2a085ef2850aa63f78eb29ef811500aa279af620e4be80b8` |

The excluded Native official-YaRN factor-4 partial trace contains `506`
examples with examples/run-manifest SHA-256
`14e71766f1de8bc0d42adf38d49da065f30b835b05e14ba639408f14ac8da229`
and
`a4a56a4d34e7949b79f859711b0020dcf5a5e401fb8cdd0c37600f71796f1216`.
It has no aggregate owner and is not used as evidence.

Each row below is `results / examples / run-manifest` SHA-256:

| Evaluated component | Artifact SHA-256 values |
| --- | --- |
| 2Wiki untouched Native 4K | `3a836250d4d2f32cc5498f9a967e85bd159ec6d636d52579912cb85a6d100a2a` / `d17044cc81081f4471555c5a08c80e84cbb27233b0ac70b1a249dac6e009590b` / `e5c87ce5623b27d7c47c99f325774fcd76a7a654425bb61cc5a80c2c12da24fb` |
| 2Wiki Native fresh Q/K | `db6cddb98ad3f086eeaad75f6da280d66f50d605b4df80d2849e7b7c38a247ef` / `b2dddd0adc053b5cb42126d5d0eb77affda552890bd99db2f39e8e1344fffbd3` / `f8a785cbf98a90e3f855d6847eeae76a256c55bc815ef9d7eb2d8afba80bc6bd` |
| 2Wiki EVQ fresh Q/K | `130e72b02597529c5dd7ac7f847ef7102cdff476d1d643523c1064c8d12e2a51` / `2b9c56b611cacc1586b151261e39700b4da8f138772f07b9bc13362fe78adb7c` / `fae58f5cf1f7703a95add90c654d95b0f624ec17c4d8d29e8aaab16cd434fc7a` |
| RULER untouched Native 4K | `6880005fda82956e977ab65547fa06ffb7c77d09043ad7503a0c05d59e2d5d45` / `a46e92f0364e7b932f3b8a95e6aaac9c120277cf971693153433a4ab455a7acb` / `f48581dc8566fcefc4267d8d94fb443f0550c63cc7c9400650f3def7fc7fa794` |
| RULER Native raw | `805cdaa9e6548f3f5543da7620bd4599fadfcc77c362d3ce84077ee39f91459c` / `c1158bc395b2a4c1927df1392bf40201267bf2ddb55fba929ead2d84d3ccb4db` / `67b4390e22634570933618081055c8500d757142c384b37e8920c6a417ca0aab` |
| RULER EVQ raw | `4df05747bb0bf3f8d80c02ba5fc7de51f781382fd9e8f26120536606d40ceb3e` / `c170a856a9b0d9dbe0e6102ac8efbb168afe43fd1f30d0c4ba208656575870c7` / `d7c0132044c0a6b6958020158d966d2de9519fe119c7b7134490018f2f7da1e0` |
| RULER Native + official YaRN | `946151cd9900eac0918b938a9606ae2d53c6b4c730742da81552dad2b679b6d5` / `9d197e96bfdd8bc168c682f06e20e9720afb7ecf2edcb023b1281a5bb9b2c00d` / `a135d563a7e892114934436b6c147dc05f01da8fe6190562ce0733d72ed53466` |
| RULER EVQ + official-YaRN composite | `54b2b351ac2c470cc74988f49bcba92d20f94f5dd3380ce093659382ee2be707` / `792441d7a2fde8c28253e63986d99b02c25e53d6a5e9f443951ba1b2749a132e` / `280fbd305461cc67d9c52db068c84ed4779404bb97c5839e7242663d4d480b26` |
| RULER Native + repository ramp | `53b86b07a0263ddc4d0704b1cf52a8f2c9a16e668d8e6510684f00cff5eb9b88` / `525e2fb745de4b9ff8b7e94437cbc49f33db32d3e9ef81de73b7795e0a8e35f6` / `5de849b30d8c274b0db76cfd75e1fba3e28f0bc0e7f7b09061550973e8b94efc` |
| RULER EVQ + repository-ramp composite | `037f67b11d7353a722536dd9cf5fa6a863a9142c14b6c448fe96b6f1929e111d` / `5b3b8329d0180e74d7a1d4343a29c43a7ba0c898fb08f4572cb1d496b2f4cbfa` / `50bd97fdb6336287706ecdc8c81285bc9bea6aaca26f1a2cabe5f1f4c1736c05` |

## Reviewer-facing wording

> Starting from the untouched OLMo-2 1.485B checkpoint, we trained one matched
> seed of fresh Q/K-only LoRA adapters on generic LongAlign/Tulu data, with
> every physical training sequence at most 4K and no 2Wiki or RULER-family
> supervision. EVQ lowers natural-text NLL relative to Native at 8K/16K
> (3.088/3.382 versus 3.932/5.108), but this probability gain does not become
> broad task capability. Native/EVQ complete-family RULER macro is
> 58.53/6.12% at 4K, 0/3.96% at 8K, and 0/2.38% at 16K; Native/EVQ 2Wiki
> token-F1 is 25.47/15.56% at 4K, 0.13/6.49% at 8K, and 0.10/1.11% at 16K.
> The 2Wiki endpoint deterministically fills prompts with answer-filtered
> distractors and is not the unmodified LongBench leaderboard protocol. Under
> the tested factor-2 official Transformers YaRN transform, Native/EVQ RULER
> macro is 61.06/11.05% at 4K and 52.19/6.99% at 8K. We therefore treat
> generic Q/K adaptation as evidence that long-position NLL and autoregressive
> task capability must be separated, not as a general downstream-transfer
> result.
