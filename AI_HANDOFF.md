# AI Handoff (No-Chat Continuation)

Last updated: 2026-02-23 09:05 CST  
Repo: `e:\rope\hybrid-rope`  
Remote repo: `/root/autodl-tmp/dfrope/hybrid-rope`

## 1) Current live status

The fair 8B suite is running and healthy.

- Run id: `llama8b_fair_v2_longbench_stable_20260223_0150`
- Main process:
  - `bash scripts/run_fair_comparison.sh` (PID `72124`)
  - `/root/miniconda3/bin/python ... run_llama8b_fair_suite.py --method sigmoid` (PID `85858`)
- GPU real usage:
  - `~86802 MiB` by python (compute-apps visible)
  - power ~`580-610W`

Important: on this machine, `GPU-Util=100% with 0 MiB` can appear when no job is running.  
Always trust `nvidia-smi --query-compute-apps` first.

## 2) Progress inside 5-method fair suite

Methods order in this run:

1. baseline (done)
2. pi (done)
3. yarn (done)
4. sigmoid (running now)
5. anchored_sigmoid (pending)

Log markers in:
- `logs/llama8b_fair_v2_longbench_stable_20260223_0150.log`

## 3) Completed results so far (from `summary.json`)

Output root:
- `/root/autodl-tmp/dfrope/hybrid-rope/results/llama8b_fair_v2_longbench_stable_20260223_0150`

Completed:

| Method | train_loss | eval_loss | Tail-PPL@16K | Passkey@16K |
|---|---:|---:|---:|---:|
| baseline | 1.8807 | 2.5187 | 11.4764 | 0.80 |
| pi | 1.6833 | 2.4714 | 10.6616 | 1.00 |
| yarn | 1.6852 | 2.4605 | 10.5914 | 1.00 |

Files:
- `results/llama8b_fair_v2_longbench_stable_20260223_0150/baseline/summary.json`
- `results/llama8b_fair_v2_longbench_stable_20260223_0150/pi/summary.json`
- `results/llama8b_fair_v2_longbench_stable_20260223_0150/yarn/summary.json`

## 4) Running model snapshot (sigmoid)

From watchdog parse:

- latest step: `70/400`
- latest loss: `1.7155`
- timestamp: `2026-02-23 09:04:57 CST`

Watch log:
- `logs/llama8b_fair_v2_longbench_stable_20260223_0150.watch.log`

## 5) Commands to monitor from new PC

Use your existing plink endpoint:

```powershell
C:\Users\Admin\.ssh\plink.exe -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0 "pgrep -af 'run_fair_comparison.sh|run_llama8b_fair_suite.py'"
```

```powershell
C:\Users\Admin\.ssh\plink.exe -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0 "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader"
```

```powershell
C:\Users\Admin\.ssh\plink.exe -batch -ssh -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0 "tail -n 80 /root/autodl-tmp/dfrope/hybrid-rope/logs/llama8b_fair_v2_longbench_stable_20260223_0150.log"
```

## 6) If interrupted, resume without wasting finished work

Do not rerun full `run_fair_comparison.sh` from method 1 if 1-3 are already done.

Run only remaining methods:

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope

# Common args
BASE=/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct
OUT=./results/llama8b_fair_v2_longbench_stable_20260223_0150
PY=/root/miniconda3/bin/python
SCRIPT=2026-02-22/scripts/run_llama8b_fair_suite.py

# resume sigmoid only (if needed)
$PY $SCRIPT --method sigmoid --run_name sigmoid \
  --base_model_path $BASE --output_root $OUT \
  --max_seq_len 16384 --max_steps 400 --seed 42 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 --bf16 --lora_rank 64 --lora_alpha 128 \
  --logging_steps 10 --save_steps 200 --warmup_steps 20

# then anchored_sigmoid
$PY $SCRIPT --method anchored_sigmoid --run_name anchored_sigmoid \
  --base_model_path $BASE --output_root $OUT \
  --max_seq_len 16384 --max_steps 400 --seed 42 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 --bf16 --lora_rank 64 --lora_alpha 128 \
  --logging_steps 10 --save_steps 200 --warmup_steps 20
```

## 7) Data corpus used in this run

- Training text file: `/root/autodl-tmp/data/long_text.txt`
- Size: ~`343 MB`
- Source built from local LongBench mirror text extraction
- In summary provenance: `text_sha256=2ab984a109a45f9d...`

## 8) What to sync after run finishes

Minimum:

- `results/llama8b_fair_v2_longbench_stable_20260223_0150/*/summary.json`
- `logs/llama8b_fair_v2_longbench_stable_20260223_0150.log`
- `logs/llama8b_fair_v2_longbench_stable_20260223_0150.watch.log`

Optional (for paper plots):

- each method folder under:
  - `results/llama8b_fair_v2_longbench_stable_20260223_0150/`

## 9) Known pitfalls already fixed

1. Startup pitfall fixed: `python: command not found`
   - `scripts/run_fair_comparison.sh` now forces `PYTHON_BIN=/root/miniconda3/bin/python`.
2. GPU monitoring pitfall:
   - ignore bare `GPU-Util` when `compute-apps` is empty.
3. Chat template mismatch in passkey:
   - evaluation path already fixed in current script version.
