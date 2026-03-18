# Qwen Hybrid-LoRA 标准化对比协议（V1）

## 1. 目的
只回答一个问题：

`Hybrid 频率 + LoRA` 相比 `原始频率 +/或原始模型` 是否在长上下文更稳，且短上下文不过度退化。

## 2. 本轮必须固定的变量（不可改）
- 基座模型：`Qwen2.5-7B-Instruct`（同一路径）
- Tokenizer：与基座一致（同版本）
- 训练脚本：同一脚本、同一 commit
- 训练数据：`TinyStories train`（必须成功，不可 fallback）
- 训练 token 数：`40M`
- max_steps：`500`
- seq_len：`8192`
- LoRA 配置：`r=16, alpha=32, dropout=0.05, target=[q,k,v,o]`
- seed：固定（建议 `42`）
- 评测脚本：同一脚本
- 评测集：`TinyStories validation`（必须成功，不可 fallback）
- 评测长度：`[2048, 8192, 16384]`
- 每长度 chunk：`5`

## 3. 对比组（最小可发表）
- `base_orig`：原始模型，不训练
- `lora_hybridfreq_int4`：int4 训练 + hybrid 频率
- `lora_hybridfreq_fp16`：bf16/fp16 训练 + hybrid 频率

可选补充（第二阶段）：
- `lora_origfreq_int4`
- `lora_origfreq_fp16`

## 4. 强制失败即停（Fail-fast）
任一项失败，整轮作废，不得入主表：
1. 训练数据不是 TinyStories（例如回退到 wikitext）
2. 评测数据不是 TinyStories validation
3. Tokenizer 或模型路径不一致
4. 评测脚本版本不一致
5. 结果 JSON 缺少 manifest 字段

## 5. 每次 run 必须写入 manifest
每个实验目录必须保存 `manifest.json`，包含：
- model_path
- tokenizer_path
- train_dataset
- eval_dataset
- seq_len
- target_tokens
- max_steps
- seed
- freq_mode (`orig`/`hybrid`)
- quant_mode (`int4`/`fp16`)
- script_path
- git_commit (若可用)
- timestamp

## 6. 文件命名规范
统一输出根目录：
`/opt/dfrope/results/qwen_standard_v1/`

子目录：
- `base_orig_eval/`
- `lora_hybridfreq_int4/`
- `lora_hybridfreq_fp16/`
- `summary/`

最终汇总：
- `summary/results_main.json`
- `summary/results_main.md`

## 7. 执行顺序（严禁并发）
1. 跑 `lora_hybridfreq_int4` 训练
2. 跑 `lora_hybridfreq_fp16` 训练
3. 同一评测脚本依次评测 3 组
4. 生成统一主表（2048/8192/16384）

## 8. 主表模板
| Config | PPL@2048 | PPL@8192 | PPL@16384 | Eval Dataset |
|---|---:|---:|---:|---|
| base_orig |  |  |  | TinyStories/validation |
| lora_hybridfreq_int4 |  |  |  | TinyStories/validation |
| lora_hybridfreq_fp16 |  |  |  | TinyStories/validation |

## 9. 质量门槛（建议）
- 短程门槛：`PPL@2048` 相对 `base_orig` 不劣于 `+10%`
- 长程目标：`PPL@16384` 优于 `base_orig`

## 10. 本轮教训（必须记录）
- 若训练或评测出现数据源 fallback（TinyStories -> wikitext），该轮结果仅可做 debug，不可做结论。
