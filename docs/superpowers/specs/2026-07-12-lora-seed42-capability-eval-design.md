# LLaMA-3-8B Seed-42 Three-Arm Capability Evaluation

## Goal

Evaluate exactly the same prepared examples with three model arms:

1. `geo_base`: the original LLaMA-3-8B-Instruct model with native geometric RoPE;
2. `geo_lora_s42`: the seed-42 LongAlpaca LoRA adapter with native geometric RoPE;
3. `evq_lora_s42`: the seed-42 LongAlpaca LoRA adapter with EVQ-Cosh (`tau=1.414`).

Checkpoint 200 is out of scope. Only the completed step-300 adapters are used.

## Evaluation Sets

### Long-range controlled tasks

- Official RULER v1 examples at 8K, 16K, and 32K: single-needle,
  multi-key, multi-value, multi-query, variable tracking, common-word
  extraction, and frequent-word extraction.
- A passkey grid at 8K, 16K, and 32K with depths 10%, 25%, 50%, 75%, and
  90%. Prepare 20 deterministic examples per length/depth cell.
- NoLiMa-Hard examples at 16K and 32K with multiple document depths.

### Real long-document tasks

- LongBench v1 NarrativeQA and Qasper examples whose complete LLaMA-3 chat
  prompt fits between 8K and 32K tokens.
- Do not keep an example if truncation would remove part of the source
  document, question, or answer prompt.

### Short-context retention tasks

- Deterministic subsets of MMLU, ARC-Challenge, HellaSwag, OpenBookQA, and
  WinoGrande. OpenBookQA replaces PIQA because the server's Datasets 5.0
  runtime no longer supports PIQA's dataset-loading script.
- Score choices by conditional answer likelihood; also report accuracy.

## Metrics

- Controlled retrieval: autoregressive exact match plus gold-answer NLL.
- Real QA: normalized exact match/F1 plus gold-answer NLL.
- Multiple choice: normalized option likelihood, accuracy, and the margin
  between the correct option and the best incorrect option.
- Report each task and length separately. Do not turn the suite into one
  headline score before inspecting the individual results.

## Cost Control

- CPU mode performs every download, conversion, token-length check, and data
  validation.
- Prepare 20 examples per controlled cell but provide a five-example pilot
  switch for the first GPU run.
- Load the 8B backbone once, attach both seed-42 adapters, and switch adapters
  and frequency schedules in one process.
- The GPU run must fail before loading weights if model, adapter, or prepared
  data paths are missing.

## Outputs

- CPU-prepared JSONL files and a compact dataset manifest on the server data
  disk.
- One raw result JSON containing per-example outputs and metrics for all three
  arms.
- One summary JSON grouped by arm, task, context length, and depth.

No paper claims or reported numbers change as part of preparing this suite.
