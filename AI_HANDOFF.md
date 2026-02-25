# AI Handoff (One-Glance Ops Card)

Last updated: 2026-02-25 09:25 CST  
Local repo: `/Users/misaya.yanghejazfs.com.au/neurIPS-2026/hybrid-rope`  
Server repo: `/root/autodl-tmp/dfrope/hybrid-rope`

## 0) Must-Read TL;DR

- **Locked tuned params for next runs**: `anchor_factor=4`, `slope_raw=20`, `center_ratio=0.70`.
- **Current cross-model training script on server is NOT using tuned params**.
  - It uses legacy default anchored-sigmoid (`center_ratio=0.47`, `slope=16.05/head_dim`, auto anchor at 16K ~= 5).
- **Current run status (confirmed)**:
  - `llama baseline seed=1337` is done (`checkpoint-600` exists).
  - `llama anchored_sigmoid seed=1337` is running now.

## 1) Current live server status (for immediate decision)

As of `2026-02-25 09:23 CST`:
- GPU: `50567 MiB / 97887 MiB`, util ~`100%`
- Active process:
  - `scripts/train_cross_model_lora.py --method anchored_sigmoid ... --seed 1337 --max_steps 600`
- Progress:
  - `artifacts/cross_model/monitor.log` shows `step=347/600`
- Completed artifact:
  - `artifacts/cross_model/llama_3_8b_instruct_baseline_1337/checkpoint-600`
- Running artifact:
  - `artifacts/cross_model/llama_3_8b_instruct_anchored_sigmoid_1337/checkpoint-200`

## 2) Why this matters

Current cross-model run is protocol-clean in budget, but **parameter-misaligned** with the tuned schedule selected for v6 remediation.  
If we continue full queue without correction, Mistral/Qwen may consume compute on non-target params.

## 3) Fast verification commands (do before any next launch)

```bash
# A) Check running process and progress
cd /root/autodl-tmp/dfrope/hybrid-rope
pgrep -af 'scripts/train_cross_model_lora.py'
tail -n 10 artifacts/cross_model/monitor.log
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
```

```bash
# B) Verify schedule defaults in code (should show legacy values)
cd /root/autodl-tmp/dfrope/hybrid-rope
grep -nE 'center_ratio|slope =|anchor_factor|eff_anchor' rope/schedules.py scripts/train_cross_model_lora.py
```

## 4) Next-action checklist (operator handoff)

1. Let current `llama anchored` finish, then decide whether to early-stop queued models.
2. Before launching Mistral/Qwen, switch to tuned schedule path and log `inv_freq_sha256`.
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
