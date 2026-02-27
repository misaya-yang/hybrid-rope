# Protocol Lock (2026-02-26 Delta)

## Equality contract (non-negotiable)

Compared rows must share:
- base checkpoint
- tokenizer + chat template policy
- identical manifest/sample indices
- decode settings
- injection path semantics
- max context + truncation policy

## Adapter contract

Accepted adapter layouts:
- `root_adapter_only`
- `final_lora_only`
- `dual_root_and_final_lora`

Registry must expose:
- `adapter_layout`
- `adapter_resolved_path`

## Result schema contract (required)

Every eval output used for claims must include:
- `protocol_lock`
- `manifest_json`
- `per_sample_scores_raw`
- `inv_sha256`

Applied in:
- `scripts/eval_longbench.py`
- `scripts/eval_niah_recall.py`
- `scripts/eval_passkey_teacher_forcing.py`

## Claim policy

- `p_fdr_bh < 0.05`: significant improvement
- else: directional / non-significant wording only
