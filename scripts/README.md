# `scripts/` — Runnable Entrypoints (Paper-Facing)

This directory contains **repo-level runnable scripts** for training/evaluation/audit.
If you are new, start from `AI_HANDOFF.md` and `docs/README.md`.

## Core Entry Points

### Evaluation
- LongBench (supports lb6/lb21 + parity options): `scripts/eval_longbench.py`
- NIAH recall: `scripts/eval_niah_recall.py`
- Passkey (teacher-forcing scoring): `scripts/eval_passkey_teacher_forcing.py`
- Plan B orchestrated evaluator (LongBench + NIAH + Passkey): `scripts/plan_b_eval_longbench.py`
- Passkey sanity sweep (quick regression): `scripts/run_passkey_sanity_check.py`
- SOTA downstream eval runner (controlled protocol): `scripts/run_sota_downstream_eval.py`

### Training
- Llama-3 8B LoRA variants: `scripts/train_llama8b_lora_variant.py`
- Cross-model LoRA (e.g., Qwen/Mistral): `scripts/train_cross_model_lora.py`
- Cross-model LoRA (fast tuned entrypoint, cost-optimized): `scripts/train_cross_model_lora_fast_tuned.py`
- Plan B locked trainer (Llama-3-8B 8K contract): `scripts/plan_b_train_anchored_v2.py`
- Long instruction mix builder (jsonl -> train/valid/test + manifest): `scripts/prepare_long_instruction_mix.py`

### Theory Bridge / Audits
- Attention distance prior estimation (E3-lite): `scripts/run_attn_hist.py`
- LongBench score scaling audit: `scripts/import_2024/longbench_scale_audit.py`
- Significance tests (bootstrap + FDR): `scripts/import_2024/significance_test.py`
- Base/LoRA registry builder (ready-to-eval table): `scripts/build_model_registry.py`

## Conventions

- Outputs should go to `artifacts/` (small manifests/audits) and `paper_exports/` (paper-ready tables/figures).
- Do not commit large weights/traces; see `docs/TERMS_AND_PROTOCOLS.md` and `.gitignore`.

## Legacy / Compatibility

- Compatibility wrappers live in `scripts/compat/`.
- Old prototypes live in `scripts/legacy/` and should not be used for new claims.
