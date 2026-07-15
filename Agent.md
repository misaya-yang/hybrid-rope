# Experiment Rules

Read this before any LoRA, rebuttal, or paid-GPU experiment.

## Never repeat

- Do not replace the paper-lineage LongAlpaca protocol with LongAlign or infer data identity from filenames.
- Do not use paid GPU time for downloads, tokenization, tensor preparation, protocol design, or ordinary debugging.
- Do not report “ready,” “running,” ETA, or completion without checking the command, process, log, and artifacts.
- Do not treat code SHA, GPU, runtime, compile, checkpointing, cache, or telemetry as scientific variables.
- Do not use WikiText-only gains as broad generalization evidence or overclaim a short LoRA run.
- Do not inherit an official method name from a class, comment, citation, or old report.
- Trace every named baseline from paper row to artifact, runner, forward path, and pinned official source.
- Require representative output parity with the official implementation; otherwise use a descriptive local label.
- Record native endpoint versus midpoint frequency grids; both are geometric but are not the same control.
- Re-derive leading orders and optimization claims independently before calling a proxy a theorem.

## Clean Geo/EVQ pair

- Match exact model/tokenizer, LongAlpaca tokens and order, objective/labels, LoRA, BF16, optimizer, steps, B2/GA4, and external evaluator.
- Within a seed, only the frequency schedule may differ; a pure shape claim also requires the same endpoint/midpoint quantizer.
- Retain seed-42 Geo/EVQ; run only EVQ seeds 43/44 when authorized, using Geo-42 as a labeled fixed reference rather than a paired control.
- LongAlign remains a separate experiment and cannot fill a LongAlpaca arm.

## Execution and GPU

- **GPU-on is experiment-only.** Never start a paid GPU until every required code file, model, adapter, dataset/tensor, manifest/hash, output path, environment, and launch command is present and validated in no-GPU mode. If anything is missing or mismatched after startup, shut the instance down immediately, finish preparation off-GPU, and restart only when the experiment can launch without debugging, data preparation, or artifact transfer.
- Run all SHA-256 checks, path discovery, downloads, copying, compilation, tests, dry-runs, token counting, case generation, and result-transfer preparation in no-GPU mode. A GPU session may only source a completed READY receipt and immediately execute its frozen experiment command.
- After durable experiment outputs exist, shut the GPU down before analysis, packaging, result transfer, figure generation, or report writing only when no further authorized GPU experiment remains and the user has not explicitly asked to keep the instance running. Explicit keep-running instructions take precedence between contiguous experiment stages.
- Record execution metadata per run, but never require global equality; FP8/FP4, quantization, packing, sample-order, label, or batch changes are new protocols.
- Before GPU startup, finish data/model/eval hashes, dry-run/tests, launch command, paths, compile cache, and automatic evaluation.
- Once GPU is on, launch within five minutes or recommend shutdown; then verify PID, first optimizer step, loss, speed, memory/utilization, and ETA.
- Reuse compile caches, use only short representative performance probes, and never launch another arm without permission.

## Valid result

A result counts when training/artifacts are valid, any claimed pair obeys the contract, external evaluation is complete, and raw per-seed NLL/PPL is retained. Null and negative results still count; never replace a missing arm with another dataset or protocol.
