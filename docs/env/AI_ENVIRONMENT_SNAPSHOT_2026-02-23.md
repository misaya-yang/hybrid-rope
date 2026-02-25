# AI Environment Snapshot - 2026-02-23

Last updated: 2026-02-23 09:05 CST

## 1) Repository and machine map

- Local repo: `e:\rope\hybrid-rope`
- Remote repo: `/root/autodl-tmp/dfrope/hybrid-rope`
- Remote host: `connect.bjb1.seetacloud.com:42581`

## 2) Remote runtime environment

- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- VRAM: `97887 MiB` (about 96 GB)
- Python: `/root/miniconda3/bin/python`
- PyTorch: `2.8.0+cu128`
- Current GPU telemetry during run:
  - memory used: `~86811 MiB`
  - power: `~580-610W`

## 3) Active experiment (live)

- Script launcher: `scripts/run_fair_comparison.sh`
- Core train script: `scripts/run_llama8b_fair_suite.py`
- Output root:
  - `results/llama8b_fair_v2_longbench_stable_20260223_0150`
- Log files:
  - `logs/llama8b_fair_v2_longbench_stable_20260223_0150.log`
  - `logs/llama8b_fair_v2_longbench_stable_20260223_0150.watch.log`

Method progress:

1. baseline - completed
2. pi - completed
3. yarn - completed
4. sigmoid - running
5. anchored_sigmoid - pending

## 4) Completed method metrics

Extracted from:

- `results/llama8b_fair_v2_longbench_stable_20260223_0150/baseline/summary.json`
- `results/llama8b_fair_v2_longbench_stable_20260223_0150/pi/summary.json`
- `results/llama8b_fair_v2_longbench_stable_20260223_0150/yarn/summary.json`

| Method | train_loss | eval_loss | Tail-PPL@16K | Passkey@16K |
|---|---:|---:|---:|---:|
| baseline | 1.8807 | 2.5187 | 11.4764 | 0.80 |
| pi | 1.6833 | 2.4714 | 10.6616 | 1.00 |
| yarn | 1.6852 | 2.4605 | 10.5914 | 1.00 |

## 5) Current running method snapshot

From watchdog parse:

- method: `sigmoid`
- latest step: `70/400`
- latest loss: `1.7155`
- latest timestamp in logs: `2026-02-23 09:04:57 CST`

## 6) Dataset provenance for this run

- Main text file: `/root/autodl-tmp/data/long_text.txt`
- Size: about `343 MB`
- In summary provenance:
  - `source=local:/root/autodl-tmp/data/long_text.txt`
  - `text_sha256=2ab984a109a45f9d7788a4027afb2272fb6d195b3b316280ff2c57200f1f2263`
  - `split_overlap_count=0`

## 7) Monitoring commands (copy/paste)

```bash
pgrep -af 'run_fair_comparison.sh|run_llama8b_fair_suite.py'
```

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,pstate --format=csv,noheader
```

```bash
tail -n 120 logs/llama8b_fair_v2_longbench_stable_20260223_0150.log
tail -n 120 logs/llama8b_fair_v2_longbench_stable_20260223_0150.watch.log
```

## 8) Reliability note

On this GPU/driver combo, `GPU-Util=100%` with `0 MiB` can happen when no compute process is attached.
Use `compute-apps` as source of truth.

## 9) Startup safety

`scripts/run_fair_comparison.sh` has been patched to avoid PATH issues:

- prefers `PYTHON_BIN=/root/miniconda3/bin/python`
- falls back to `python`/`python3` only if needed

This prevents the previous silent failure:

- `python: command not found`
