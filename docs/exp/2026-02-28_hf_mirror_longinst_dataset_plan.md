# 2026-02-28 HF Mirror 数据准备与训练执行说明

## 目标
在 Llama-3-8B + 单卡 96GB 上，构建可审计的 mixed-prior 数据集并执行 EVQ 对照训练：
- baseline: `evq_cosh, tau=0.0`
- method: `evq_cosh, tau=1.5`
- 双 seed: `42/1337`

## 为什么必须使用 HF 镜像（文本理由）
1. 当前机器位于中国大陆网络环境，直接访问 `huggingface.co` 存在高概率超时、断流和限速。
2. 训练前数据准备需要稳定批量拉取（HotpotQA/Alpaca），若下载中断会导致样本分布漂移，破坏实验可复现性。
3. 使用 `HF_ENDPOINT=https://hf-mirror.com` 可以在同一脚本下获得稳定下载路径，减少“同配置不同机器结果不一致”的非算法噪声。

## 新增脚本
- 脚本：`/Users/yang/projects/hybrid-rope/scripts/prepare_external_longinst_sources.py`
- 功能：
  - 从 HF 下载并转换为 longinst 可直接读取的 JSONL：
    - `hotpotqa_multihop.jsonl`（多跳推理源）
    - `alpaca_scaffold.jsonl`（格式对齐源）
  - 输出 `source_manifest.json`（样本数 + sha256）

## 服务器执行命令（恢复连接后）

### 1. 生成外部高质量源
```bash
cd /root/autodl-tmp/dfrope/hybrid-rope-main
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/.cache/huggingface
export HF_HOME=/root/autodl-tmp/.cache/huggingface

/root/miniconda3/bin/python scripts/prepare_external_longinst_sources.py \
  --output_dir /root/autodl-tmp/dfrope/datasets/longinst_external_20260228 \
  --hotpot_max_samples 30000 \
  --alpaca_max_samples 12000 \
  --hf_endpoint https://hf-mirror.com \
  --force
```

### 2. 重建 mixed-prior（2e8 token）
```bash
cd /root/autodl-tmp/dfrope/hybrid-rope-main

/root/miniconda3/bin/python scripts/prepare_mixed_prior_dataset_v1.py \
  --wikitext_path /root/autodl-tmp/wikitext_data/train.txt \
  --bimodal_jsonl_paths /root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/LongAlpaca-12k.min64.jsonl,/root/autodl-tmp/dfrope/datasets/longinst_external_20260228/hotpotqa_multihop.jsonl \
  --scaffold_jsonl_paths /root/autodl-tmp/dfrope/datasets/longinst_external_20260228/alpaca_scaffold.jsonl \
  --tokenizer_path /root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
  --target_total_tokens 200000000 \
  --powerlaw_ratio 0.50 \
  --bimodal_ratio 0.40 \
  --scaffold_ratio 0.10 \
  --max_seq_len 8192 \
  --min_supervised_tokens 64 \
  --ratio_tolerance 0.02 \
  --strict \
  --dataset_prefix mixed_prior_v2_hotpot_alpaca \
  --output_root artifacts/datasets
```

### 3. 训练前 smoke（强制）
```bash
cd /root/autodl-tmp/dfrope/hybrid-rope-main

/root/miniconda3/bin/python scripts/isolated/longinst/new_lora_longinst_train_v1.py \
  --base_model_path /root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct \
  --mixed_dataset_dir <LATEST_MIXED_DATASET_DIR> \
  --mixed_dataset_split train \
  --seed 42 \
  --rope_schedule evq_cosh \
  --evq_tau 1.5 \
  --max_steps 80 \
  --run_name smoke_tau1p5_s42 \
  --max_seq_len 8192 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --warmup_steps 20 \
  --save_steps 200 \
  --attn_implementation sdpa \
  --longbench_local_data_dir /root/autodl-tmp/dfrope/ms_datasets/LongBench/data \
  --qwen_seed42_json /root/autodl-tmp/dfrope/hybrid-rope/artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/downstream_eval_fast400/longbench/anchored_sigmoid.json \
  --qwen_seed1337_json /root/autodl-tmp/dfrope/hybrid-rope/artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed1337/downstream_eval_fast400/longbench/anchored_sigmoid.json \
  --morning_reference_json /root/autodl-tmp/dfrope/hybrid-rope/artifacts/reviewer_2026-02-25/h2_qwen_fast400_seed42/downstream_eval_fast400/longbench/anchored_sigmoid.json
```

### 4. 正式四运行（双 seed × baseline/tau=1.5）
使用 `run_llama8k_theory_v1.py`，保持 `A1/A2/B1/B2` 不改，`tau=0.0/1.5`。

## 成功标准
1. `mix_manifest.json` 与 `quality_report.md` 中 ratio 和 mask 检查全部 PASS。
2. smoke 训练无异常（无 NaN/无标签门禁失败）。
3. 再进入 4-run 正式训练。
