# 跨模型公平 LoRA 微调协议（2026-02-24）

本文件用于固定本轮跨模型微调实验的训练协议、数据口径和复现入口，避免实验条件漂移。

## 1. 实验目标

在统一训练预算下，比较 `geometric baseline` 与 `anchored_sigmoid` 在不同基础模型上的迁移一致性。

## 2. 任务矩阵（固定 6 组）

1. Mistral-7B-Instruct-v0.3 + baseline, `seed=42`
2. Mistral-7B-Instruct-v0.3 + anchored_sigmoid, `seed=42`
3. Qwen2-7B-Instruct + baseline, `seed=42`
4. Qwen2-7B-Instruct + anchored_sigmoid, `seed=42`
5. Llama-3-8B-Instruct + baseline, `seed=1337`
6. Llama-3-8B-Instruct + anchored_sigmoid, `seed=1337`

执行脚本：`scripts/cross_model_finetune.sh`

## 3. 统一超参数（6 组任务严格一致）

- `max_steps=600`
- `learning_rate=2e-5`
- `warmup_steps=50`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=8`
- `max_seq_len=16384`
- `lora_rank=16`
- `lora_alpha=32`
- `lora_target_modules=q_proj,k_proj,v_proj,o_proj`
- `bf16=true`
- `lr_scheduler_type=cosine`
- `optim=paged_adamw_8bit`
- `load_in_4bit=true`（QLoRA 路径）

实现位置：
- `scripts/cross_model_finetune.sh`
- `scripts/train_cross_model_lora.py`

## 4. RoPE 注入规范

- baseline：保持几何 `inv_freq`（不改形状）
- anchored_sigmoid：使用现有 anchored-sigmoid schedule 生成 `inv_freq`
- 注入方式统一为原地覆盖：`inv_freq.copy_()`（见 `rope/inject.py`）

每次训练会保存：
- `artifacts/cross_model/<model>_<method>_<seed>/artifacts/custom_inv_freq.pt`
- `artifacts/cross_model/<model>_<method>_<seed>/artifacts/summary.json`

## 5. 训练数据口径

当前批次使用本地文本语料目录：
- `/root/autodl-tmp/wikitext_data/train.txt`
- `/root/autodl-tmp/wikitext_data/valid.txt`

说明：
- 训练脚本实际读取 `train.txt` 作为连续文本并做随机窗口切分（`seq_len=16384`）。
- `valid.txt` 当前作为同目录验证参考文件留存，本轮 6 组训练主流程不参与 early-stop。
- 语料内容形态与 WikiText-103 风格一致（如 `<unk>` 标记与百科段落格式）。

## 6. 运行与复现

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope
PYTHON_BIN=/root/miniconda3/bin/python bash scripts/cross_model_finetune.sh
```

输出目录：
- `artifacts/cross_model/`
- `artifacts/cross_model/_logs/`

## 7. 时间预估方法

单任务粗略耗时可用：

`wall_time ≈ step_time * 600 + model_load_overhead`

其中 `step_time` 可由 `artifacts/cross_model/monitor.log` 的 step 增速实时估计。全队列总耗时约为 6 个任务耗时之和。
