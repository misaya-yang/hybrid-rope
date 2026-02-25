# AI Handoff (One-Glance Ops Card)

Last updated: 2026-02-25 20:36 CST  
Local repo: `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope`  
Server repo: `/root/autodl-tmp/dfrope/hybrid-rope`

## 0.1) New dated handoff package (2026-02-25)

Use this folder as the current reviewer-remediation entrypoint:

- `handoff_2026-02-25/0_README.md`
- `handoff_2026-02-25/1_PROTOCOL_LOCK.md`
- `handoff_2026-02-25/2_ASSET_MAP.md`
- `handoff_2026-02-25/3_RUNBOOK.md`
- `handoff_2026-02-25/4_RECOVERY_AND_CLEANUP.md`
- `handoff_2026-02-25/README.md` (legacy)
- `handoff_2026-02-25/01_IMPLEMENTED_SCOPE.md`
- `handoff_2026-02-25/02_VALIDATION_SNAPSHOT_AIDEMO.md`
- `handoff_2026-02-25/03_NEXT_EXECUTION_GATES.md`

## 0) Must-Read TL;DR

- **Locked tuned params for next runs**: `anchor_factor=4`, `slope_raw=20`, `center_ratio=0.70`.
- **Main evidence chain focus (cost-sensitive)**: `Qwen2.5-7B-Instruct` baseline_gold + anchored(tuned) + modern anchor (NTK) on **full lb21** and **multi-seed**.
- **Current server training uses the tuned fast entrypoint**:
  - `scripts/train_cross_model_lora_fast_tuned.py --method anchored_sigmoid --anchor_factor 4 --slope_raw 20 --center_ratio 0.70`
- **Current run status (confirmed on 2026-02-25 20:36 CST)**:
  - done: `qwen2_5_7b_instruct_baseline_42` (`checkpoint-400` exists).
  - done: `qwen2_5_7b_instruct_anchored_sigmoid_42` (`checkpoint-400` exists).
  - running: `qwen2_5_7b_instruct_baseline_1337` (`checkpoint-200` exists; training PID shown below).
  - pending: `qwen2_5_7b_instruct_anchored_sigmoid_1337` (directory not created yet).

## 0.2) New “facts-first” indices (do not skip)

- Experiment inventory mirror: `docs/exp/EXPERIMENT_INVENTORY.md`
- Authoritative registry (single source of truth): `docs/EXPERIMENT_REGISTRY.md`
- Server cleanup manifest (quarantine-first): `docs/exp/SERVER_CLEANUP_MANIFEST.md`

## 1) Current live server status (for immediate decision)

As of `2026-02-25 20:36 CST`:
- GPU: ~`90277 MiB / 97887 MiB`, util ~`100%` (training running)
- Active process:
  - `scripts/train_cross_model_lora_fast_tuned.py --method baseline ... --seed 1337 --max_steps 400`
- Progress:
  - `checkpoint-200` exists under the baseline seed=1337 output dir (ETA depends on compile/save overhead)
- Completed artifacts:
  - `artifacts/cross_model_fast_tuned_b1_gc/qwen2_5_7b_instruct_baseline_42/checkpoint-400`
  - `artifacts/cross_model_fast_tuned_b1_gc/qwen2_5_7b_instruct_anchored_sigmoid_42/checkpoint-400`
- Running artifact:
  - `artifacts/cross_model_fast_tuned_b1_gc/qwen2_5_7b_instruct_baseline_1337/checkpoint-200`

## 3) Fast verification commands (do before any next launch)

```bash
# A) Check running process and progress
cd /root/autodl-tmp/dfrope/hybrid-rope
pgrep -af 'scripts/train_cross_model_lora_fast_tuned.py'
ls -1d artifacts/cross_model_fast_tuned_b1_gc/qwen2_5_7b_instruct_baseline_1337/checkpoint-* 2>/dev/null || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
```

```bash
# B) Verify schedule defaults in code (should show legacy values)
cd /root/autodl-tmp/dfrope/hybrid-rope
grep -nE 'center_ratio|slope =|anchor_factor|eff_anchor' rope/schedules.py scripts/train_cross_model_lora.py
```

## 4) Next-action checklist (operator handoff)

1. Let current `qwen baseline seed=1337` finish, then launch `qwen anchored seed=1337` to complete the 2-seed pair.
2. Before launching any new model, log `inv_freq_sha256` and keep protocol locked (data/model/steps/lora).
3. Keep fairness locked: same data, steps, LR, LoRA rank/alpha, tokenizer, eval manifest.
4. For every rerun, write one-line provenance:
   - method, seed, model, `anchor_factor/slope_raw/center_ratio`, `inv_freq_sha256`.

## 4.1) New speed-first launcher (for future runs)

Use:

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
bash scripts/cross_model_finetune_fast_tuned.sh
```

This launcher uses a separate train entrypoint (`scripts/train_cross_model_lora_fast_tuned.py`) and does not require modifying legacy scripts.

Defaults in this launcher:
- `MAX_STEPS=400`
- `PER_DEVICE_BATCH=2`
- `GRAD_ACCUM=1`
- `GRAD_CHECKPOINTING=0`
- tuned anchored schedule:
  - `ANCHOR_FACTOR_DEFAULT=4`
  - `ANCHORED_SLOPE_RAW=20`
  - `ANCHORED_CENTER_RATIO=0.70`

Why `MAX_STEPS=400`:
- baseline log on this server showed strong gains in 0-200, moderate 200-400, and very small 400-600 marginal gain.
- this keeps most quality gain while cutting runtime cost significantly.

Example override (`batch=4` smoke):

```bash
PER_DEVICE_BATCH=4 MAX_STEPS=50 bash scripts/cross_model_finetune_fast_tuned.sh
```

## 5) Single source of truth for tuned schedule

- Tuning evidence: `handoff_2026-02-23/local_tuning_proof_2026-02-24.md`
- Runbook: `handoff_2026-02-23/tomorrow_tuned_param_runbook_2026-02-25.md`
- Locked recommendation:
  - `anchor_factor=4`
  - `slope_raw=20`
  - `center_ratio=0.70`

## 6) Guardrails

- Do not compare numbers across mismatched manifests/settings.
- Do not claim SOTA; use theory-guided consistency framing.
- If a condition loses, keep it in final sign-test table (no cherry-pick).
- Cleanup is quarantine-first: move to `/root/autodl-tmp/dfrope/trash/hybrid-rope/<date>/` and record every move in `docs/exp/SERVER_CLEANUP_MANIFEST.md`.
  - Note: `/autodl-pub/data` is mounted read-only for `trash/` on this server.

## 7) New NeurIPS sprint assets (2026-02-25)

Implemented scripts for high-priority整改:

- LongBench parity + 21-task pipeline
  - `scripts/eval_longbench.py`
    - `--task_set {lb6,lb21}`
    - `--prompt_source {official,legacy}`
    - `--chat_template {auto,on,off}`
    - `--truncate_mode {tail,middle}`
    - `--max_new_tokens_policy {official,manual}`
    - `--strict_parity_check`
    - (new) `--batch_size N` (default 1) for batched greedy generation
  - `scripts/longbench_official_config/dataset2prompt.json`
  - `scripts/longbench_official_config/dataset2maxlen.json`
  - `scripts/import_2024/longbench_pipeline_audit.py`

- Protocol-trace runner updates
  - `scripts/run_eval.py`
    - adds longbench parity args passthrough
    - writes `baseline_protocol_lock.json` per run folder

- Statistical rigor updates
  - `scripts/import_2024/significance_test.py`
    - `--fdr_method {bh,by,both}`
    - outputs `p_raw`, `p_fdr_bh`, `p_fdr_by`, `claim_grade`
    - auto-generates `claim_policy_report.md`

- Theory strengthening scripts
  - `scripts/import_2024/functional_residual_real_prior.py`
  - `scripts/import_2024/theorem3_adversarial_bimodal.py`

Supporting notes:
- `handoff_2026-02-23/longbench_pipeline_parity.md`
- `handoff_2026-02-23/anchored_sigmoid_math_note.md`
