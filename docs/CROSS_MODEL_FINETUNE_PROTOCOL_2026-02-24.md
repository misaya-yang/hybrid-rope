# 跨模型公平 LoRA 微调协议（2026-02-24，2026-02-25增补）

本文件用于固定跨模型微调实验的训练协议、数据口径和复现入口，避免实验条件漂移。

## 0. 当前生效版本

- 历史冻结版本（600 steps）保留为对照，不删除。
- 从 2026-02-25 起，后续实验默认使用 fast-tuned 协议（400 steps + 高利用稳定档）。

生效入口：

- `scripts/cross_model_finetune_fast_tuned.sh`
- `scripts/train_cross_model_lora_fast_tuned.py`

## 1. 实验目标

在统一训练预算下，比较 `geometric baseline` 与 `anchored_sigmoid` 在不同基础模型上的迁移一致性。

## 2. 任务矩阵（默认 6 组）

1. Mistral-7B-Instruct-v0.3 + baseline, `seed=42`
2. Mistral-7B-Instruct-v0.3 + anchored_sigmoid, `seed=42`
3. Qwen2-7B-Instruct + baseline, `seed=42`
4. Qwen2-7B-Instruct + anchored_sigmoid, `seed=42`
5. Llama-3-8B-Instruct + baseline, `seed=1337`
6. Llama-3-8B-Instruct + anchored_sigmoid, `seed=1337`

## 3. 统一超参数（fast-tuned 默认）

- `max_steps=400`
- `learning_rate=2e-5`
- `warmup_steps=50`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=1`
- `max_seq_len=16384`
- `lora_rank=16`
- `lora_alpha=32`
- `lora_target_modules=q_proj,k_proj,v_proj,o_proj`
- `bf16=true`
- `lr_scheduler_type=cosine`
- `optim=paged_adamw_8bit`
- `load_in_4bit=true`（QLoRA 路径）
- `gradient_checkpointing=true`

调度参数（anchored）：

- `anchor_factor=4`
- `slope_raw=20`
- `center_ratio=0.70`

## 4. RoPE 注入规范

- baseline：保持几何 `inv_freq`（不改形状）
- anchored_sigmoid：使用 anchored-sigmoid schedule 生成 `inv_freq`
- 注入方式统一为原地覆盖：`inv_freq.copy_()`（见 `rope/inject.py`）

每次训练会保存：

- `<run_dir>/artifacts/custom_inv_freq.pt`
- `<run_dir>/artifacts/summary.json`
- `<run_dir>/final_lora/`

## 5. 训练数据口径

当前批次使用本地文本语料目录：

- `/root/autodl-tmp/wikitext_data/train.txt`
- `/root/autodl-tmp/wikitext_data/valid.txt`

说明：

- 训练脚本实际读取 `train.txt` 作为连续文本并做随机窗口切分（`seq_len=16384`）。
- `valid.txt` 作为同目录验证参考文件留存，本轮主流程不参与 early-stop。

## 6. 运行与复现

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
PYTHON_BIN=/root/miniconda3/bin/python bash scripts/cross_model_finetune_fast_tuned.sh
```

输出目录（默认）：

- `artifacts/cross_model_fast_tuned/`
- `artifacts/cross_model_fast_tuned/_logs/`

## 7. 资产登记（必须执行）

每批训练后必须执行：

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
python scripts/build_model_registry.py
```

产物目录：

- `artifacts/model_registry/base_models.tsv`
- `artifacts/model_registry/lora_weights.tsv`
- `artifacts/model_registry/latest_by_method.tsv`

## 8. 评测加载规范（base + LoRA + inv 三元组）

```bash
python scripts/run_eval.py \
  --exp cross_model_eval \
  --model <BASE_MODEL_PATH> \
  --method anchored_sigmoid \
  --ctx 16384 \
  --seed 42 \
  --adapter_override <RUN_DIR>/final_lora \
  --custom_inv_freq_path <RUN_DIR>/artifacts/custom_inv_freq.pt \
  --suite ppl,longbench_full,needle \
  --longbench_score_scale pct
```

要求：

- 必须显式给出 `--adapter_override` 与 `--custom_inv_freq_path`。
- 禁止仅凭方法名推断路径用于正式结果。
