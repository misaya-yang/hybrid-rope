# Verified state of the zero-training retrofit evidence

- **Date:** 2026-08-23
- **Method:** every scalar below was recomputed from raw `examples.jsonl` rows or
  rebuilt from first principles. Values stored in `results.json` were *compared*,
  never copied.
- **Receipt:** [`evidence/OPUS_RECOMPUTATION_20260823.json`](evidence/OPUS_RECOMPUTATION_20260823.json)
- **Outcome of the comparison:** **0 mismatches** across every aggregate checked
  (Qwen macros, 2Wiki/Qasper token F1 and exact, all RULER cells).

## 1. Identity — VERIFIED

Both checkpoints were checked against the **huggingface.co** API, not the mirror
the weights were downloaded from.

| Model | HF repo | `model.safetensors` SHA-256 | On-disk | HF API |
| --- | --- | --- | :---: | :---: |
| OLMo-2-0425-1B-Instruct | `allenai/OLMo-2-0425-1B-Instruct` | `36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f` | ✓ | ✓ |
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee` | ✓ | ✓ |

Every frequency table was rebuilt from the model config alone and matched the
run receipts bit-for-bit:

| Table | Rebuilt float32 SHA-256 | Matches receipt |
| --- | --- | :---: |
| OLMo native (d=128, θ=5e5) | `dde15c31724177356ae954d6e11fb337e6fccef56e4520a905cac3f0d9885b34` | ✓ |
| OLMo budgeted s2 | `f94a34381cfb3d05621db41bd3778779b16812246c392bd645013a1a06f80814` | ✓ |
| OLMo budgeted s4 | `a435d75441444bcea39b73d9cf530005249dc5afdc3cfb5a60fda10ef33312d3` | ✓ |
| OLMo official YaRN f4 (orig 4096) | `cc9da456982ffce5ca0558e9ea661abc4a880ec002179ce6b9149d45aa4a016c` | ✓ |
| Qwen native (d=128, θ=1e6) | `138c99b109d7affbfba059e435670918fe4531bce4709b6e86f3f22f7ef80f6e` | ✓ |
| Qwen budgeted s4 (frozen p=2, s=4, 2048 pts) | `15754e606a1c141fb370afb58760dd0eeb0cbab5ce4054629c477ed320954018` | ✓ |
| Qwen official YaRN f4 (orig 32768) | `427ed49dc1d6e18683800eca336467c70fc6817bfdae8a61166cbc7e01b05cb1` | ✓ |

The two YaRN tables were regenerated with **local transformers 4.57.6** while the
runs used **server transformers 5.15.1**, and matched exactly. The YaRN arm is
therefore genuinely `ROPE_INIT_FUNCTIONS["yarn"]`, correctly configured
(`factor=4.0`, `original_max_position_embeddings` = each model's true native
window, `rope_theta` = each model's true θ), and is not a library-version
artifact. Attention scaling is `1.138629436111989` = `1 + 0.1 ln 4` for the
official YaRN arm **and** the budgeted arm — the amplitude is YaRN's own published
mscale, not a tuned constant.

## 2. OLMo formal natural-context matrix (386 rows) — VERIFIED

Token manifest `74022bf36d444a1735baab72bda0312b9867dd38c9f85ece376049b5f35f66f3`
records `padding: false`, `synthetic_needles: false`, `unrelated_concatenation:
false`, `main_result_truncation: false`. PG-19 uses 20 books, one anchor per book,
right-aligned nested suffixes so all three multipliers share the same 512-token
NLL target span.

Four arms, **386 rows each, `row_sha256` sets IDENTICAL across all four, zero
duplicate rows, zero rows reused across cells.**

| Method | 1x macro (5 tasks) | 2x macro (6) | 4x macro (6) | PG-19 1x | PG-19 2x | PG-19 4x |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 0.3424 | 0.0661 | 0.0210 | 2.9712 | 7.1029 | 7.2050 |
| official YaRN f4 | 0.3173 | 0.2127 | 0.2559 | 3.3880 | 3.4404 | 3.7955 |
| target-aware oracle | 0.3424 | 0.2631 | 0.2497 | 2.9712 | 2.9746 | 3.0977 |
| **binary Native/s4** | **0.3424** | **0.2858** | 0.2497 | **2.9712** | 3.1060 | **3.0977** |

Cell counts: 20 per task-multiplier except NarrativeQA 1x (absent), NarrativeQA
2x (8), Qasper 4x (18); PG-19 20 per multiplier. Total 386. ✓

**Route labels and parity — VERIFIED bitwise, not read from a receipt:**

- binary at 1x selects `native` on all 120 rows; at 2x and 4x selects
  `budgeted_s4_p2` on all 128 and 138 rows.
- binary 1x vs Native: **120/120** identical on `prediction`,
  `generated_token_ids`, `nll`, and `score`.
- binary 4x vs target-aware s4 oracle: **138/138** identical.

## 3. OLMo full 200-row tasks — VERIFIED

Inputs are identical across arms at row level (`source_row_sha256`,
`input_sha256`, `input_tokens`, `truncated` all match), zero duplicate source
rows, and no row has `input_tokens + max_new_tokens > 16384`.

### 2WikiMQA, 16K window, 200 rows, 32 generated tokens

| Operator | token F1 | norm. exact | routes N/s2/s4 | truncated |
| --- | ---: | ---: | ---: | ---: |
| smallest-covering router | 0.247265 | 0.175 | 24/121/55 | 1 |
| official YaRN f4 | 0.256858 | 0.200 | — | 1 |
| **binary Native/s4** | **0.266596** | 0.205 | 24/0/176 | 1 |
| fixed budgeted s4 everywhere | **0.277433** | **0.220** | 0/0/200 | 1 |

Subset decomposition (computed here, not in any owner): on the **24 short rows
that binary routes to Native**, binary scores 0.3380 while fixed-s4 scores 0.4283
and YaRN scores 0.4333. On the 176 s4-routed rows binary and fixed-s4 are
**bitwise identical (176/176)**. The `−0.0108` F1 that binary pays versus fixed-s4
is therefore located entirely in the Native branch. **Exact Native preservation is
a construction requirement here, not a source of gain.**

### Qasper, 16K window, 200 rows, 128 generated tokens — **NOT IN ANY OWNER**

| Operator | token F1 | norm. exact | routes N/s2/s4 | truncated | raw `results.json` SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| official YaRN f4 | 0.180295 | 0.110 | — | 3 | `e8101c8cdc00c9b3c3394d66736de8e87a4cac9bb4b9a9421a57e57ba3179240` |
| **binary Native/s4** | **0.245733** | 0.115 | 70/0/130 | 3 | `04ea9e2aa503378e2a8008f7e20b16bded1820d2c054c588d52eaa44ea43ccfe` |

Both arms used the **same** evaluator (`67b615ed7659…`). Binary wins on both
subsets separately (Native-routed 0.3454 vs 0.2517; s4-routed 0.1921 vs 0.1418).
This is currently the strongest natural-document evidence in the whole line and it
appears in no report, receipt, or handoff. See `05` DEFECT-5.

## 4. OLMo RULER — VERIFIED

**Anti-leakage receipt, checked on disk:** the fresh 13-task v5 dataset
(`431ad942eead911da4693a4ab7428085fd04e63d6914fc8831c3d5b0cc88c239`) and the
core-4 selection dataset (`6f64aa6bb44821f52015c868b1f13d07a17a5f710b7645a9e0fb3fb86ac09d37`)
share **8 of 8 core-4 cells byte-identically**, same seed `20260822`, same
lm-eval commit `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`, same chat overhead 11,
same checkpoint. The nine confirmation tasks were evaluated with the table,
amplitude, routing rule, checkpoint, and metric all frozen.

| Method | core-4 8K | core-4 16K | unseen-9 8K | unseen-9 16K | full-13 8K | full-13 16K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| official YaRN f4 | 0.2225 | 0.0125 | 0.2452 | 0.0794 | 0.2382 | 0.0588 |
| **binary Native/s4** | **0.7175** | **0.4075** | **0.6594** | **0.6047** | **0.6772** | **0.5440** |
| stateless boundary-slope | 0.0000 | 0.0000 | — | — | — | — |

binary vs YaRN win/tie/loss: unseen-9 `8/0/1` and `9/0/0`; full-13 `12/0/1` and `12/1/0`.

**core-4 is the selection set.** The `p`/amplitude/`s` choices were made on it
(see `05` DEFECT-8). The honest OLMo confirmation numbers are the unseen-9 row.
The gap between selection and confirmation is modest (0.7175→0.6594 at 8K),
which is evidence of limited overfitting — but it is not zero and must be stated.

## 5. Qwen2.5-1.5B-Instruct — VERIFIED, with a live sample-asymmetry hole

Only `2 × L_native` (65536) and `4 × L_native` (131072) were ever evaluated: the
script gates `lengths ∈ {2·L_native, 4·L_native}`. **There is no in-window (1x)
cell on Qwen at all, and no natural-document or PG-19 evaluation on Qwen at all.**

### 64K = 2× native, n=20 per task (80 rows per arm), script `d2e0a518…`

| Method | single_1 | multikey_2 | multikey_3 | vt | macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native | 1.00 | 0.30 | 0.00 | 0.88 | **0.5450** |
| official YaRN f4 | 1.00 | 0.45 | 0.10 | 0.86 | **0.6025** |
| **binary s4** | 1.00 | 0.65 | 0.15 | 0.88 | **0.6700** |

**Native does not collapse on Qwen.** This is a qualitatively different regime
from OLMo, where Native is exactly 0.0000 at every length and task. The
OLMo-style "floor → 0.68" headline does not transfer; the Qwen margin over Native
is 0.125 and over YaRN is 0.0675. Predictions were inspected and are coherent
(Native retrieves the single needle verbatim at both 64K and 128K).

### 128K = 4× native — **currently sample-asymmetric, do not aggregate**

| Method | n per task | macro |
| --- | ---: | ---: |
| Native | 5 | 0.4500 |
| official YaRN f4 | 5 | 0.5200 |
| binary s4 | 5 | 0.6900 |
| binary s4 (completed during this review) | **20** | **0.6175** |

Going from n=5 to n=20 moved the binary macro by **−0.0725** (`niah_multikey_3`
0.60→0.30, `vt` 0.96→0.77). Determinism was confirmed: the first five rows of the
n=20 run reproduce the n=5 run **20/20** exactly, so this is sampling, not drift.
**The n=5 128K three-arm comparison is a probe and is now known to be biased.**
Native and YaRN have not been run at n=20. Cost to close: ≈17 min GPU per arm.

## 6. Uncertainty — DERIVED (paired cluster bootstrap, 4000 resamples)

Evaluation rows resampled within task/cell. These operators are deterministic and
zero-training, so there is no training-seed component; this is **evaluation-sampling
uncertainty on one checkpoint**, nothing more.

| Comparison | Δ | 95% CI | P(≤ reference) |
| --- | ---: | --- | ---: |
| OLMo RULER-13 8K, binary − YaRN | +0.4390 | [+0.393, +0.490] | 0.000 |
| OLMo RULER-13 16K, binary − YaRN | +0.4853 | [+0.441, +0.526] | 0.000 |
| OLMo PG-19 2x tail NLL, binary − YaRN | −0.3344 | [−0.389, −0.279] | 0.000 |
| OLMo PG-19 4x tail NLL, binary − YaRN | −0.6978 | [−0.800, −0.607] | 0.000 |
| OLMo Qasper full-200, binary − YaRN | +0.0654 | [+0.022, +0.110] | 0.002 |
| OLMo formal 2x macro, binary − YaRN | +0.0731 | [+0.008, +0.147] | 0.013 |
| Qwen core-4 64K, binary − YaRN | +0.0675 | [+0.008, +0.133] | 0.015 |
| Qwen core-4 64K, binary − Native | +0.1250 | [+0.030, +0.220] | 0.005 |
| **OLMo 2Wiki full-200, binary − YaRN** | **+0.0097** | **[−0.036, +0.057]** | **0.344** |
| **OLMo formal 4x macro, binary − YaRN** | **−0.0063** | **[−0.062, +0.048]** | **0.582** |
| **Qwen core-4 128K (n=5), binary − YaRN** | **+0.1700** | **[−0.020, +0.370]** | **0.055** |
| OLMo 2Wiki full-200, binary − fixed s4 | −0.0108 | [−0.028, +0.004] | 0.909 |

**Robust:** RULER (both lengths), PG-19 (both multipliers), Qasper.
**Marginal:** formal 2x macro, Qwen 64K.
**Indistinguishable from zero:** 2Wiki full-200, formal 4x macro, Qwen 128K n=5.

## 7. Fairness and implementation audit — VERIFIED

| Item | Finding |
| --- | --- |
| Flash-only, no math fallback | `configure_cuda()` sets `enable_math_sdp(False)`, `enable_mem_efficient_sdp(False)`, `enable_cudnn_sdp(False)`, `enable_flash_sdp(True)`. `ruler_flash_forward` raises on any mask, on non-prefill/non-single-token shapes, and on non-divisible GQA heads. A silent fallback is impossible — `probe_binary_64k_v3.log` shows the harness *crashing* with `RuntimeError: No available kernel` when `enable_gqa` was missing. All completed runs therefore used flash. |
| GQA | Qwen 12Q/2KV → `enable_gqa=True`; OLMo 16Q/16KV → no GQA. Correct on both. |
| Greedy decoding | Single shared `greedy_generate` (argmax + KV cache + bf16 autocast + EOS stop). `generation_config.json` sampling parameters are never used. |
| Row selection | `rows[:limit_per_cell]` — deterministic prefix, identical across arms. |
| Truncation | RULER raises rather than truncating (zero truncated rows). 2Wiki 1 row and Qasper 3 rows truncated, identical in every arm. |
| Data validation | `_validate_data` fails closed on manifest status, checkpoint path, tokenizer hash vs the checkpoint's own `tokenizer.json`, per-cell file hash, and row count. |
| Routing inputs | `select_observed_session_factor` uses only `prefill_tokens + max_new_tokens` and `L_native`. No task label, no reference answer, no external `L_target`. |
| Aggregation | macro = mean of cell means; examples are never pooled across tasks. Recomputed, zero deviation. |
| Receipt integrity | All **24/24** `raw_result_sha256` entries in the evidence receipt match the host files. |
| Local tests | 24 passed (`test_target_free_rope`, `test_length_conditioned_budgeted_rope`, `test_target_free_context_builder`). |

**Code-version seams found, and their resolution:**

| Seam | Resolution |
| --- | --- |
| Formal matrix: Native/YaRN/oracle on `db42afa0…`, binary on `bf3c29d8…` | **Proven inert**: binary 1x ≡ Native 1x on 120/120 rows and binary 4x ≡ oracle 4x on 138/138 rows, across the version boundary. |
| 2Wiki: binary evaluator `b8388721…`, other three arms `f1e24903…` | **Proven inert**: binary's 176 s4-routed rows are bitwise identical to the fixed-s4 arm. |
| Qwen 64K on `d2e0a518…`, 128K on `e489d4de…` | Only difference is an added `runtime` receipt block; method receipts and table hashes are identical. |
| RULER-13 Native row spliced from 2026-08-22 runs under a different evaluator family | Floor-valued (0.0000) under both implementations; low risk, but it is a splice. See `05` DEFECT-9. |

## 8. What is NOT established

- Any 128K conclusion on Qwen.
- Any Qwen in-window (≤32K) behaviour, including whether the Native branch is
  preserved there — the Qwen path installs the s4 table once at load time and
  never exercises the routing code (see `05` DEFECT-1).
- Any Qwen natural-document or natural-LM behaviour.
- That the binary policy beats YaRN on 2WikiMQA, or on the six-task 4x macro.
- Cross-model universality, or optimality of any kind.
- That the redundancy measure — rather than the resulting band split — is what
  produces the effect. This is the central open question; see `02` and `03`.
