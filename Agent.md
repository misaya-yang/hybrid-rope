# Experiment Rules

Read this before any LoRA, rebuttal, or paid-GPU experiment.

## Never repeat

- Do not replace the paper-lineage LongAlpaca protocol with LongAlign or infer data identity from filenames.
- Do not use paid GPU time for downloads, tokenization, tensor preparation, protocol design, or ordinary debugging.
- Do not report “ready,” “running,” ETA, or completion without checking the command, process, log, and artifacts.
- Do not treat code SHA, GPU, runtime, compile, checkpointing, cache, or telemetry as scientific variables.
- Do not use WikiText-only gains as broad generalization evidence or overclaim a short LoRA run.

## Clean Geo/EVQ pair

- Match exact model/tokenizer, LongAlpaca tokens and order, objective/labels, LoRA, BF16, optimizer, steps, B2/GA4, and external evaluator.
- Within a seed, only the frequency schedule differs: native Geo versus EVQ-Cosh (`tau` only for EVQ).
- Retain seed-42 Geo/EVQ; run only EVQ seeds 43/44 when authorized, using Geo-42 as a labeled fixed reference rather than a paired control.
- LongAlign remains a separate experiment and cannot fill a LongAlpaca arm.

## Execution and GPU

- Record execution metadata per run, but never require global equality; FP8/FP4, quantization, packing, sample-order, label, or batch changes are new protocols.
- Before GPU startup, finish data/model/eval hashes, dry-run/tests, launch command, paths, compile cache, and automatic evaluation.
- Once GPU is on, launch within five minutes or recommend shutdown; then verify PID, first optimizer step, loss, speed, memory/utilization, and ETA.
- Reuse compile caches, use only short representative performance probes, and never launch another arm without permission.

## Valid result

A result counts when training/artifacts are valid, any claimed pair obeys the contract, external evaluation is complete, and raw per-seed NLL/PPL is retained. Null and negative results still count; never replace a missing arm with another dataset or protocol.
