# NTS2 Native-window result

**Status: paused.** Natural-QA is the primary Native-window endpoint; the four-model
result does not justify further scale, gain, band or curve tuning.

## Primary Natural-QA result

`native_tailspline_s2_midgain_v1` uses exact finite-grid TailSpline at `S=2`
and the fixed rotary gain `sqrt(1 + 0.1 ln 2) = 1.0340767466953285`. Model
weights, prompts, decoding and scorers are unchanged within each paired run.

The primary metric is task-equal LongBench token-F1 over 2WikiMQA, HotpotQA
and Qasper.

| Model / Native window | Rows per arm | Native | NTS2 | NTS2 − Native |
|---|---:|---:|---:|---:|
| OLMo-2-1B / 4K | 99 | 39.41% | **47.45%** | **+8.03pp** |
| Llama-3-8B / 8K | 189 | 46.47% | **46.99%** | **+0.52pp** |
| Qwen2.5-3B / 32K | 60 | **37.48%** | 37.29% | −0.19pp |
| GLM-4-9B / 32K | 60 | **50.36%** | 49.35% | −1.01pp |

| Model | 2WikiMQA delta | HotpotQA delta | Qasper delta |
|---|---:|---:|---:|
| OLMo-2-1B | +9.49pp | +13.33pp | +1.28pp |
| Llama-3-8B | +1.59pp | 0.00pp | −0.03pp |
| Qwen2.5-3B | +2.35pp | −2.35pp | −0.56pp |
| GLM-4-9B | −14.00pp | +12.68pp | −1.72pp |

NTS2 is a large Native-QA enhancement on OLMo and a small positive transfer on
the larger Llama confirmation. Qwen is effectively flat and GLM is mildly
negative overall with a large positive HotpotQA tradeoff. The evidence supports
a model-dependent Native effect rather than a universal QA improvement.

The first Llama screen used only 45 residual rows after excluding the historical
631-row pool and produced `−5.25pp`. The broader output-blind Native-8K census
contains 189 rows (`80/29/80`) and produces `+0.52pp`; it is the formal Llama
estimate. No scale, gain, band or task-dependent setting changed between them.

## Secondary task and LM evidence

- Llama Native-8K Full-13×10: `90.59% → 92.40%` (`+1.81pp`).
- Llama 8K whole NLL on 46 documents: `1.66315 → 1.65387`, approximately
  `0.924%` lower PPL.
- OLMo Native-4K Full-13×10: `73.53% → 76.14%` (`+2.62pp`).
- OLMo same-target full-context NLL: `2.91527 → 2.91984`, approximately
  `0.458%` higher PPL.

These are secondary to the Natural-QA table for the Native-window claim.

## Frozen table identities

| Model | Canonical band | FP32 table SHA256 |
|---|---|---|
| OLMo-2-1B | `[14,32]` | `0ccaecb736d94072579e76caa3ae12fa65ac565a2f1b7a4e7f10287c51938485` |
| Llama-3-8B | `[18,35]` | `7d6739a3dc98c909a09dd05662d961a2408d8cefbceaf8dac3a6fe1d0983ced0` |
| Qwen2.5-3B | `[23,40]` | `bb55a80244758dc071953c5e557e521490f1e1a1ee4c601c8bfb8bd814af19ca` |
| GLM-4-9B | `[17,30]` over 32 partial-RoPE pairs | `fac0cf96475697c8b7d978b62f0cafca449093955f59f444aa1eea400402ba26` |

## Server evidence

Raw generations, contracts, tables and summaries remain on the 4080 data disk;
they are intentionally not copied into Git.

| Model | Server root | Decision SHA256 |
|---|---|---|
| OLMo | `/root/autodl-tmp/today_rope_plan_20260914/native_tailspline_s2_midgain_20260917` | `6394d3162ae087477e465e16635f475080ee624f22a99c8ca91f3ce7a9a82cd6` |
| Llama | `/root/autodl-tmp/today_rope_plan_20260914/native_tailspline_s2_midgain_llama_qa_confirm_20260917` | `a9098311bb9d401eb796135ad03702bc2e492b9a93ea0b590072120b57df4b74` |
| Qwen | `/root/autodl-tmp/today_rope_plan_20260914/native_tailspline_s2_midgain_qwen3_qa_20260917` | `e61649b3ef3c65ba8dc12ada34f7f38815b4bd465760dc7d3185c49a3fc5778d` |
| GLM | `/root/autodl-tmp/today_rope_plan_20260914/native_tailspline_s2_midgain_glm_20260917` | `bd3d484c9dddb07060764701c1bf67957e5baecf2022b1be9a770d8c602a215e` |
