# 明日新参数实验执行单（2026-02-25）

## 目标
在不改变公平协议的前提下，仅替换 `anchored_sigmoid` 的频率曲线参数，验证论文主结论是否进一步稳固。

## 锁定参数（理论校准后）
- `anchor_factor=4`
- `slope_raw=20`
- `center_ratio=0.70`
- 仅改 `custom_inv_freq`，其余保持不变（基座、LoRA、tokenizer、manifest、decode、seed）。

## Step 1: 生成 tuned `custom_inv_freq.pt`

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
CONDA_EXE=/root/miniconda3/bin/conda

OUT_DIR=artifacts/tuned_invfreq
mkdir -p "$OUT_DIR"

$CONDA_EXE run -n base python - <<'PY'
import json
from pathlib import Path
import torch

model_cfg = Path('/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct/config.json')
cfg = json.loads(model_cfg.read_text())
hidden_size = int(cfg.get('hidden_size', 4096))
num_heads = int(cfg.get('num_attention_heads', 32))
head_dim = hidden_size // num_heads
base = float(cfg.get('rope_theta', 500000.0))

anchor_factor = 4.0
slope_raw = 20.0
center_ratio = 0.70

k = head_dim // 2
idx = torch.arange(k, dtype=torch.float64)
base_inv = 1.0 / (base ** (2.0 * idx / float(head_dim)))
slope = slope_raw / float(head_dim)
center = center_ratio * float(k)
sig = 1.0 / (1.0 + torch.exp(-slope * (idx - center)))
scale_factor = 1.0 + (anchor_factor - 1.0) * sig
inv = (base_inv / scale_factor).float().cpu()

out = Path('artifacts/tuned_invfreq/custom_inv_freq_anchor4_slope20_center070.pt')
out.parent.mkdir(parents=True, exist_ok=True)
torch.save({'inv_freq': inv}, out)
print(f'written: {out}')
print(f'head_dim={head_dim}, base={base}, inv_shape={tuple(inv.shape)}')
PY
```

## Step 2: 快速烟雾验证（先跑 Hybrid 单条）

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
CONDA_EXE=/root/miniconda3/bin/conda

MODEL=/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct
CTX=32768
SEED=1337
SUITE_ROOT=results/llama8b_fair_v2_longbench_stable_20260223_0150
TUNED_INV=artifacts/tuned_invfreq/custom_inv_freq_anchor4_slope20_center070.pt

$CONDA_EXE run -n base python scripts/run_eval.py \
  --exp TEST \
  --model "$MODEL" \
  --method hybrid \
  --ctx "$CTX" \
  --seed "$SEED" \
  --suite ppl,longbench_full,needle \
  --suite_output_root "$SUITE_ROOT" \
  --adapter_override "$SUITE_ROOT/anchored_sigmoid/final_lora" \
  --custom_inv_freq_path "$TUNED_INV" \
  --notes tuned_anchor4_slope20_center070
```

## Step 3: 正式 E2 -> E1（固定协议）

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
CONDA_EXE=/root/miniconda3/bin/conda

MODEL=/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct
CTX=32768
SEED=1337
SUITE_ROOT=results/llama8b_fair_v2_longbench_stable_20260223_0150
TUNED_INV=artifacts/tuned_invfreq/custom_inv_freq_anchor4_slope20_center070.pt

# E2: 先看 shape 贡献
$CONDA_EXE run -n base python scripts/run_eval.py --exp E2 --model "$MODEL" --method yarn --ctx "$CTX" --seed "$SEED" --suite ppl,longbench_full --suite_output_root "$SUITE_ROOT" --notes tuned_anchor4_slope20_center070
$CONDA_EXE run -n base python scripts/run_eval.py --exp E2 --model "$MODEL" --method hybrid --ctx "$CTX" --seed "$SEED" --suite ppl,longbench_full --suite_output_root "$SUITE_ROOT" --adapter_override "$SUITE_ROOT/anchored_sigmoid/final_lora" --custom_inv_freq_path "$TUNED_INV" --notes tuned_anchor4_slope20_center070

# E1: 主表
$CONDA_EXE run -n base python scripts/run_eval.py --exp E1 --model "$MODEL" --method baseline_native --ctx "$CTX" --seed "$SEED" --suite ppl,longbench_full,needle --suite_output_root "$SUITE_ROOT" --notes tuned_anchor4_slope20_center070
$CONDA_EXE run -n base python scripts/run_eval.py --exp E1 --model "$MODEL" --method pi --ctx "$CTX" --seed "$SEED" --suite ppl,longbench_full,needle --suite_output_root "$SUITE_ROOT" --notes tuned_anchor4_slope20_center070
$CONDA_EXE run -n base python scripts/run_eval.py --exp E1 --model "$MODEL" --method yarn --ctx "$CTX" --seed "$SEED" --suite ppl,longbench_full,needle --suite_output_root "$SUITE_ROOT" --notes tuned_anchor4_slope20_center070
$CONDA_EXE run -n base python scripts/run_eval.py --exp E1 --model "$MODEL" --method hybrid --ctx "$CTX" --seed "$SEED" --suite ppl,longbench_full,needle --suite_output_root "$SUITE_ROOT" --adapter_override "$SUITE_ROOT/anchored_sigmoid/final_lora" --custom_inv_freq_path "$TUNED_INV" --notes tuned_anchor4_slope20_center070
```

## Step 4: 汇总与显著性

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
CONDA_EXE=/root/miniconda3/bin/conda

$CONDA_EXE run -n base python scripts/summarize.py --registry artifacts/registry.jsonl --out artifacts/tables
$CONDA_EXE run -n base python scripts/import_2024/significance_test.py --data_dir results/llama8b_fair_v2_longbench_stable_20260223_0150 --n_bootstrap 10000
```

## 通过标准（给导师/审稿人口径）
- 协议不变，仅改频率曲线参数。
- 至少给出 `Hybrid vs YaRN/PI/Baseline` 的 paired CI 和 p-value。
- 若显著性仍弱，结论写成：
  - “在锁定协议下，方向一致且机制更匹配理论；统计上不夸大显著性。”
