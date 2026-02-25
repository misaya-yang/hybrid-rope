# 03 Next Execution Gates (Cost-Control + Acceptance Impact)

Date: 2026-02-25

## Gate A: LongBench parity must be green first

Mandatory before expensive reruns:

1. Run parity audit on a shared subset:
   - local evaluator vs official-style reference.
2. Threshold:
   - absolute task delta <= 1.0 (pct scale),
   - ranking consistency true.
3. If failed:
   - fix evaluator alignment first,
   - do not launch lb21 full reruns.

## Gate B: Protocol lock must be auditable

Each training/eval run must record:

- base model id/path
- tokenizer id/path
- adapter path/hash
- inv_freq hash
- decode settings
- manifest id/checksum
- parity knobs (`prompt_source/chat_template/truncate_mode/max_new_tokens_policy`)

Required output:

- `baseline_protocol_lock.json` in each run directory.

## Gate C: Compute profile selection (avoid wasting RTX PRO 6000 96GB)

Current observed run (`cross_model_fast_tuned_b1_gc`) is fast-validation profile:

- 4bit enabled
- batch size 1
- grad checkpointing on
- memory around 22GB

This is intentionally conservative and does not maximize memory usage.

For final expensive reruns (after A/B pass), use one unified high-util profile across compared methods:

- disable 4bit if stable for target model
- disable grad checkpointing if memory allows
- increase `per_device_train_batch_size` to 2 then test 4
- keep `gradient_accumulation_steps=1` unless fairness requires effective batch matching
- keep fairness-critical knobs identical across compared methods

Decision rule:

- if throughput improves and loss curve behavior remains stable, keep high-util profile;
- otherwise step back one notch (batch or precision) but keep protocol equality.

## Gate D: Statistics claim policy is mandatory

After lb21 reruns:

1. run upgraded `significance_test.py` with FDR output;
2. enforce wording policy:
   - significant only when adjusted p < 0.05;
   - otherwise directional statement with raw+adjusted p.

## Suggested run order (single GPU)

1. Parity subset audit.
2. lb21 full eval for baseline + anchored.
3. Add NTK-aware + LongRoPE under locked protocol.
4. Recompute significance (per-sample + FDR).
5. Run theory residual + Theorem3 fragility figures on finalized prior artifacts.

## Stop conditions (hard)

- Any INVALID protocol comparison appears in main table candidates.
- Any method row missing lock metadata/hash.
- Any claim sentence inconsistent with FDR policy.
