# Research compute and model assets — 2026-09-04 snapshot

- **Status/date:** Observation / 2026-09-04; dated asset and execution snapshot.
- **Question:** Which machines and model assets were verified, and which research
  questions can they support at the observed execution boundary?
- **Protocol:** CPU inspection of model configurations, safetensors headers and
  file hashes; GPU/runtime facts and completed-run states from the
  [execution owner](../../paper-2027/research/attention-aware-retrofit/results/SINGLE_TABLE_FFN_SERVER_EXECUTION_20260904.md).
- **Receipt:** `model_asset_inventory_20260904.json`, SHA-256
  `4c6e8cfdc13f9c6cf85727af45cac64718a8c59095cc4eff09236ff1c8ee8810`.
  The machine-readable receipt and asset locations remain in the private run
  bundle. This document contains public model labels and content identities only.
- **Supported:** asset presence, recorded architecture/parameter counts, file
  identities and the explicitly completed execution boundaries below.
- **Unsupported:** universal model rankings, unmeasured throughput, future asset
  availability, full Native retention, or successful blind extrapolation.
- **Correction:** “OLMo 1B” here contains about 1.485B parameters; Gemma's “2B”
  labels here contain about 2.506B. Marketing names are not measured counts.

This is not a live machine-status page, authorization ledger or action queue.
Read [HANDOFF](../../paper-2027/HANDOFF.md) for current state and the execution
owner for results; recheck volatile machine and asset facts before another run.

## Compute and runtime

| Resource | Evidence on this date | Scope |
| --- | --- | --- |
| RTX 4080 SUPER | Measured 32760 MiB; CUDA compute capability 8.9 | The supplied work machine used for this session's completed GPU runs |
| RTX 5090 32GB | Author-reported inventory; not measured in this session | Recheck memory, architecture, BF16, Flash and runtime before use |
| Conda runtime on the measured machine | PyTorch 2.8.0+cu128, Transformers 5.15.1, PEFT 0.20.0 | Existing runtime; non-interactive PATH lacking `python` did not mean Python was absent |
| BF16 and attention | Actual BF16 GPU execution; Flash-only SDPA, with math, memory-efficient and cuDNN fallbacks disabled | Qualification applies to executed shapes; new shapes still need the shortest adequate probe |

The [Blackwell performance profile](RTX5090_BLACKWELL_PROFILE.md) contains
historical, shape-specific 5090 measurements. They are not current 4080 throughput
estimates. Optimize time and cost to a decisive result: low memory use alone does
not show an idle GPU. Training, prefill and token-by-token generation have
different bottlenecks.

## Verified model architecture

Counts below are from the CPU asset inventory, not rounded model names. `K` is
the number of rotary frequency pairs per head. `Q/KV` counts attention heads;
head dimension is read from the recorded architecture, not always inferred by
dividing hidden width by the number of heads.

| Public model or family label | Actual parameters | Layers | Hidden / FFN width | Q/KV heads | Head dimension / K | Native window | RoPE theta |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: |
| `allenai/OLMo-2-0425-1B-Instruct` | 1,484,916,736 | 16 | 2048 / 8192 | 16 / 16 | 128 / 64 | 4096 | 500000 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1,543,714,304 | 28 | 1536 / 8960 | 12 / 2 | 128 / 64 | 32768 | 1000000 |
| Qwen2.5-0.5B family; upstream variant unverified | 494,032,768 | 24 | 896 / 4864 | 14 / 2 | 64 / 32 | 32768 | 1000000 |
| `google/gemma-1.1-2b-it` label | 2,506,172,416 | 18 | 2048 / 16384 | 8 / 1 | 256 / 128 | 8192 | 10000 |
| `google/gemma-2b-it` label | 2,506,172,416 | 18 | 2048 / 16384 | 8 / 1 | 256 / 128 | 8192 | 10000 |

OLMo has untied embeddings; the two Qwen configurations declare tied embeddings.
All five assets use one `model.safetensors` file. These are separate checkpoints,
not a controlled scaling family. The Qwen 1.5B weight hash was independently
matched to the official Instruct repository at revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. The Qwen 0.5B and two Gemma labels
were not matched to an official upstream revision in this session; their local
file hashes and architecture are verified, while upstream provenance remains
unresolved. No new upstream-identity claim is inferred from an asset label.

### File identities

Each weight hash covers the complete single safetensors file; each configuration
hash covers its `config.json`. Tokenizer files, realized frequency tensors,
gain and adapters require their separate run receipts before experiments are
compared. The asset inventory alone does not establish deployment parity.

| Asset | Weight SHA-256 | Config SHA-256 |
| --- | --- | --- |
| OLMo 1.485B Instruct | `36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f` | `0d15ebb6cb8d998513b46ef337214176a6fd59fe5f16b30387c70d5f87795a9c` |
| Qwen 1.5B Instruct | `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee` | `98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670` |
| Qwen 0.5B family | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` | `18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45` |
| Gemma 1.1-2B-IT label | `584d0f7d939d235ee14a4ba307b40dbc3f03d5483181b9381e9f10636b618933` | `a2724d74fedba414a172b6e66524814ea699b925586f3d7825062e539c56c3c0` |
| Gemma-2B-IT label | `8fdf067bdfd010c75d8c4c0508fe45f4567c6097f3193928501d63a05020e0e6` | `90fd9a9035ea6cadd4fb7b3d2d0762fae249e492aa43f68b78645474c5987e88` |

## Execution boundary at the snapshot

| Asset | Completed in this session | Limit of that evidence |
| --- | --- | --- |
| OLMo 1.485B Instruct | Native short pilot, runtime checks, frozen Z/Y Native comparisons, simple-task and compact / 4K NIAH diagnostics | Frozen Z and Y fail the registered separate Native gates; this does not close adapted methods. No new all-linear OLMo training completed in this session |
| Qwen 1.5B Instruct | Native short pilot; corrected N seed 42 all-linear 128-step training; paired natural validation; Native text/task/window guards; simple tasks and NIAH | Aggregate Native confirmation passes on the declared pool, while format/indexing regresses; no blind farther-generation or FFN-necessity claim. Z stopped during restoration because controls did not resolve; Y training did not complete |
| Qwen 0.5B family | Native short pilot | No adaptation or farther-context capability result from this session |
| Gemma 1.1-2B-IT label | Native short pilot | No adaptation or farther-context capability result from this session |
| Gemma-2B-IT label | CPU asset inventory | No GPU pilot or training result from this session |

The Qwen N128 checkpoint is a completed training artifact, not a terminal success
verdict. Its strict NIAH improvement at 16K/32K is primarily output-format
compliance: original outputs already contained the correct passkey. Complete
answer/EOS remains the endpoint; surface presence is only an error diagnostic.
The source-only companion and fixed Native confirmation subsequently completed; the latter has an aggregate pass with a format/indexing regression. See the
execution owner for exact rows, failures, exclusions and receipt identities.

## Physical lengths and decision value

| Model | Physical 16K | Physical 32K | Physical 64K |
| --- | --- | --- | --- |
| OLMo Native 4K | 4x Native | 8x Native | 16x Native |
| Qwen Native 32K | 0.5x Native | 1x Native | 2x Native |
| Gemma Native 8K | 2x Native | 4x Native | 8x Native |

These are length ratios, not measured useful reach. Current Qwen training caps
are 2K/8K/16K, so 64K is 4x its maximum training cap and 2x its Native window.
Qwen 16K must never be described as Native extrapolation. A good compact control
is required before a long failure can distinguish position sensitivity from an
unsolved local task; changing checkpoint alone does not isolate parameter count.

## What 32GB supports, and when an 8B pilot helps

**Observation:** the measured 32GB machine completed the small-model evaluations
and Qwen all-linear 128-step recipe above. This is a usable research platform;
it is not evidence that every model, sequence length or microbatch fits.

**Derived storage estimate:** 8 billion BF16 parameters occupy about 14.9 GiB before
activations, caches, adapters and optimizer state. Four-bit payload alone would
occupy about 3.73 GiB, with additional quantization metadata and non-quantized
tensors. Neither estimate is an actual runtime memory measurement. For a dense
BF16 KV cache, batch 1 memory is `4 * layers * tokens * KV_heads * head_dim`
bytes; model size alone does not determine long-generation memory.

**Decision principle:** first test a chosen 8B checkpoint with Native-only compact
and near-source controls on the same semantic groups. Better resolving controls
justify a bounded adaptation comparison; failed controls justify assay diagnosis.
If the selected BF16 training shape cannot fit or its measured cost to completion
is too high, extra memory can reduce recomputation or accommodate longer inputs.
An upgrade should answer that measured limitation, not compensate for an
unvalidated benchmark. No 8B asset, download or training run is established by
this inventory.

QLoRA adds quantization to the intervention. If used, preserve separate BF16
Native and quantized-Native baselines before attributing changes to adaptation;
do not silently compare a quantized student to BF16 results as if only its
adapter changed. Verify the chosen 8B model's actual Native window and RoPE
operator rather than borrowing the 4K or 32K denominator from another model.
The current runtime contract expects one weight file; sharded assets need a
verified manifest/loader extension before this execution path can admit them.
