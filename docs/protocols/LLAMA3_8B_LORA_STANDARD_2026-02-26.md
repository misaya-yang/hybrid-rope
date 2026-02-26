# Llama-3-8B 最新 LoRA 标准协议（2026-02-26）

本协议用于后续主线实验，目标是：

- 复用已验证的 `~1h / 800 steps` 高性价比训练范式；
- 先保证主结论稳定，再做小范围参数搜索；
- 避免 attention 侧改动引入额外混杂因素。

## 1. 适用范围与红线

- 基础模型锁定：`Meta-Llama-3-8B-Instruct`（8K native）。
- RoPE 注入方式锁定：`inv_freq.copy_()`。
- 主线实验默认：`不启用 attention bias / gate / macro-micro KL`。
- 若需探索 attention 联合优化，必须走独立分支，不得与主线结果混用。
- hardest-task 主线（`qasper/musique`）**必须**使用 `scripts/isolated/attn/new_lora_longalpaca_attnbias_train.py`；
  不再使用随机窗口 LM 训练脚本（会破坏 instruction/answer 对齐，易造成 tradeoff）。

## 2. 当前推荐默认配置（主线）

训练配置（已验证可稳定跑完 800 steps）：

- `max_steps=800`
- `max_seq_len=8192`
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=1`
- `effective_batch=2`
- `learning_rate=2e-5`
- `warmup_steps=50`
- `lora_rank=32`
- `lora_alpha=64`
- `lora_target_modules=q_proj,k_proj,v_proj,o_proj`
- `attn_implementation=sdpa`
- `attn_bias_mode=off`
- `use_macro_micro_kl=false`
- LongAlpaca 输入建议：`/root/autodl-tmp/dfrope/datasets/LongAlpaca_msdownload/LongAlpaca-12k.min64.jsonl`
  （数据审计见 `docs/exp/2026-02-26_longalpaca_min64_data_audit.md`）

Anchored 调度参数（固定）：

- `anchor_factor=4`
- `slope_raw=20`
- `center_ratio=0.70`

## 3. Gate-first 执行策略（省算力）

任何新配方（新数据集 / 新 rank / 新 steps）必须先过 gate，再跑全量：

1. 只评测 `qasper,musique`（seed=42）。
2. 同时输出 `base_unfinetuned` 与 `hybrid_lora`。
3. 继续条件：
   - `qasper`: `LoRA >= base`
   - `musique`: `LoRA >= base - 1.0 pct`（容差）
4. 不满足则直接止损，不进入 `lb21 full`。

## 4. 当前最新 Gate 结论（2026-02-26）

基于：

- `/root/autodl-tmp/dfrope/hybrid-rope/artifacts/new_attnbias_v1/eval/gate_qasper_musique_s42/longbench_gate_qasper_musique_compare.json`

结果：

- `qasper`: `base=42.4927`, `LoRA=41.5714`, `delta=-0.9213`
- `musique`: `base=19.1116`, `LoRA=7.1415`, `delta=-11.9701`

判定：

- 本次 attention 参与配置不通过 gate；
- 后续主线默认回到“仅 RoPE + LoRA”。

## 5. 后续优化搜索空间（主线）

在“不动 attention”的前提下，优先搜索：

- 数据集配方：
  - `Long instruction` 主体（建议 >=70%）
  - `short instruction` 辅助（建议 <=30%）
  - 不再以 WikiText 作为主训练源
- `lora_rank`: `16 / 32 / 48`
- `max_steps`: `400 / 800 / 1200`

控制变量要求：

- 其余超参不变；
- 每个配置至少跑 `seed=42,1337`；
- 评测统一 `lb21` + `score_scale=pct`。

## 6. 产物合同（必须）

每次训练必须保留：

- `<run_dir>/run_config.json`
- `<run_dir>/artifacts/custom_inv_freq.pt`
- `<run_dir>/artifacts/custom_inv_freq.sha256`
- `<run_dir>/adapter/*`

每次评测必须保留：

- `longbench.json`（含 `base_unfinetuned` + `hybrid_lora`）
- `per_sample_scores_raw`
- `protocol_lock`
- `manifest_json`

## 7. 快速命令模板

训练（主线）：

```bash
python scripts/isolated/attn/new_lora_longalpaca_attnbias_train.py \
  --base_model_path <llama3_8b_instruct> \
  --run_name <run_name> \
  --max_steps 800 \
  --max_seq_len 8192 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --attn_implementation sdpa \
  --attn_bias_mode off \
  --no-use_macro_micro_kl \
  --response_only_loss \
  --require_assistant_header \
  --min_supervised_tokens 16
```

Gate 评测：

```bash
python scripts/eval_longbench.py \
  --base_model_path <llama3_8b_instruct> \
  --hybrid_adapter_path <run_dir>/adapter \
  --custom_inv_freq_path <run_dir>/artifacts/custom_inv_freq.pt \
  --variant custom \
  --tasks qasper,musique \
  --max_samples_per_task 0 \
  --max_input_tokens 8192 \
  --batch_size 8 \
  --max_batch_input_tokens 98304 \
  --score_scale pct \
  --output_json <gate_json>
```
