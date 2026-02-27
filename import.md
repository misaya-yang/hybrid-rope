# import

## Llama-3-8B + RTX PRO 6000 96GB（单卡）+ EVQ 论文主线：可发表级微调方案（2026-02-28）

本文是“明天可直接执行”的统一方案：结合你仓库现状、近期失败样例、以及外部一手最佳实践，给出数据集与训练的最优路径，目标是最小化算力浪费并最大化论文证据强度。

---

## 1. 先给结论（关键决策）

1. 不再接受 `WikiText-only` 或 continuation-dominant 训练。
2. 数据集必须使用 **token-level 混合先验**，不是 row-level 混合。
3. 训练采用 **QLoRA 标准栈**（NF4 + double quant + bf16 + paged optimizer）。
4. EVQ 只和 geometric 做公平对照，其他条件全部锁死。
5. 运行顺序固定：`数据审计 -> 120步smoke -> gate(qasper,musique) -> lb6/lb21 -> paired+FDR`。

---

## 2. 你当前项目的真实状态（来自仓库与历史结果）

### 2.1 已有优势（可直接复用）

1. 已有混合先验构建器：`/Users/yang/projects/hybrid-rope/scripts/prepare_mixed_prior_dataset_v1.py`
   - 输出 `train/valid/test + mixed_prior_finetune.jsonl + mix_manifest.json + quality_report.md`。
   - 已实现 token-ratio 审计、label-mask preview、source/task/lang 统计、sha256 审计。
2. longinst 训练主链已具备关键保护：
   - prebuilt mixed dataset 直读与比例硬门禁。
   - response-only mask 边界检查。
   - post-token ratio 漂移检查。
   - continuation-dominant 语料拒绝机制。
3. 调度器已切换到 EVQ 主路径（`rope_schedule=evq_cosh`）并支持 tau 记录。

### 2.2 已确认问题（必须规避）

1. 你们出现过强烈 task tradeoff：`qasper +1.85` 同时 `musique -8.03`。
2. 根因已在仓库文档证实：数据分布和监督信号偏移（continuation 过重、QA推理监督不足），不是单靠“多训几步”能解决。
3. 训练 loss 早降不代表 LongBench 提升；它经常只说明模板/格式先被学会。

---

## 3. 外部最佳实践与本项目的对应关系（联网主源）

1. **QLoRA**（NeurIPS 2023）证明单卡低成本高质量适配可行，关键是 NF4 + double quant + paged optimizer。  
   来源：[QLoRA](https://arxiv.org/abs/2305.14314)
2. **Llama-3-8B-Instruct** 官方卡给出 8K context，当前先在 8K 内建立强证据最稳。  
   来源：[Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
3. **LongLoRA / LongAlign** 指向共同事实：长上下文提升依赖“训练数据分布 + 训练工程”共同设计，不是只换位置编码曲线。  
   来源：[LongLoRA](https://arxiv.org/abs/2309.12307), [LongAlign](https://arxiv.org/abs/2401.18058)
4. **LongBench** 是主评测基线，但仅看 LongBench 不足以证明鲁棒性。  
   来源：[LongBench](https://arxiv.org/abs/2308.14508)
5. **RULER / NoLiMa / Lost-in-the-Middle** 提醒：必须补 harder stress 与中段位置鲁棒性，否则审稿人会质疑“只会简单检索”。  
   来源：[RULER](https://arxiv.org/abs/2404.06654), [NoLiMa](https://arxiv.org/abs/2502.05167), [Lost in the Middle](https://arxiv.org/abs/2307.03172)
6. **NeurIPS checklist** 强调可复现与风险披露，必须严格管理协议锁与统计口径。  
   来源：[NeurIPS Checklist Guide](https://nips.cc/public/guides/PaperChecklist)
7. **RTX PRO 6000 Blackwell 96GB** 的显存容量足够支撑你当前 8K QLoRA 双方法对照和严谨门禁。  
   来源：[NVIDIA RTX PRO 6000 Family](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/)

---

## 4. 数据集到底该长什么样（论文理论到工程落地）

你论文主张是“混合距离先验 D(Δ)”。因此数据必须显式覆盖三类先验，而不是单一 continuation。

### 4.1 配方（按 token 比例，不按样本条数）

1. `power_law_base`：50%
   - 自然长文本块（长程平滑语义）。
   - 目标：稳住长距离语义与低频区。
2. `bimodal_reasoning`：40%
   - 多跳QA、长文推理、结构化推理。
   - 目标：保护中频局部解析度，避免 musique 再次塌陷。
3. `uniform_scaffold`：10%
   - 助手格式对齐、短指令稳定。
   - 目标：保持评测 prompt 对齐，不让模型“听不懂题”。

### 4.2 数据条目必须满足的最小规范

1. 统一用 chat messages（最后一条必须 assistant 且非空）。
2. `bimodal_reasoning` 的 assistant 不允许 1-3 token 短答案，必须有可监督推理文本。
3. 样本长度分桶建议覆盖 2k/4k/8k/16k（至少让 8k 区间有足够占比）。
4. 训练集必须输出可审计元数据：`source_name/lang/task_type/count/token_ratio/sha256/filter_rule`。

### 4.3 数据硬门禁（建议锁死）

1. 构建时 `ratio_tolerance <= 0.02`。
2. `label_mask_preview.pass == true`，否则禁止开训。
3. `assistant_tokens_lt64_ratio <= 0.10`，否则禁止开训。
4. pre-token 与 post-token 两次比例都要通过。
5. 一旦 continuation-dominant（>0.70），默认直接失败。

### 4.4 推荐数据规模（结合你的算力和步数）

1. 仅 fast verify：60M-100M tokens。
2. 双方法 x 双seed + 重跑余量：150M-250M tokens。
3. 你当前 `target_total_tokens=200M` 是合理上限。

---

## 5. 到底怎么训练最好（单卡 96GB 最优性价比）

### 5.1 固定训练协议（不要漂移）

1. Base：`Meta-Llama-3-8B-Instruct`
2. Quant：`4bit NF4 + double quant`
3. dtype：`bf16`
4. LoRA target：`q_proj,k_proj,v_proj,o_proj`
5. seq：`8192`
6. lr/warmup：`2e-5 / 50`
7. attention：先 `sdpa` 稳定，再考虑 flash-attn

### 5.2 EVQ 对照协议（论文核心）

1. baseline：`geometric`（等价 `tau=0`）
2. method-A：`evq_cosh`（tau 为唯一宏观耦合参数）
3. 只改 RoPE 映射，其余参数全锁死。
4. tau 搜索建议用小网格：`0.4, 0.6, 0.8, 1.2, 1.5`，先 gate 再 full。

### 5.3 运行时成本与现象解释

1. 你看到 CPU 300%-500% 波动，典型是 flash-attn / CUDA 扩展编译，不是训练崩溃本身。
2. 早期 loss 快降（如几十步内明显下降）在 response-only SFT 常见，不能直接推断“已学好”。
3. 继续/停止训练必须由 gate 和下游任务分数决定，不由 loss 单指标决定。

---

## 6. 实验编排（避免再白烧 GPU）

### 6.1 标准执行顺序

1. `Data build + audit`（CPU阶段）
2. `Smoke`：120 steps（只看稳定性与数据监督质量）
3. `Gate`：qasper + musique
4. `LB6`
5. `LB21`
6. `Paired stats + FDR`

### 6.2 Gate 建议阈值

1. `qasper_lora >= qasper_base`
2. `musique_lora >= musique_base - 1.0`
3. 不满足则停止，不进 full。

### 6.3 统计与论文 claim 规则

1. 双 seed 配对（至少 2 run pairs）。
2. paired bootstrap + sign-flip/permutation（现有脚本支持）。
3. 多任务比较做 FDR（BH/BY）。
4. 仅当 `p_fdr_bh < 0.05` 才写 significant improvement。

---

## 7. 你明天直接可执行的命令模板

### 7.1 构建 mixed prior 数据集

```bash
cd /Users/yang/projects/hybrid-rope

.venv/bin/python scripts/prepare_mixed_prior_dataset_v1.py \
  --wikitext_path <WIKITEXT_TXT> \
  --bimodal_jsonl_paths <LONGALPACA_JSONL>,<LONGQA_JSONL> \
  --tokenizer_path <LLAMA3_MODEL_PATH> \
  --target_total_tokens 200000000 \
  --powerlaw_ratio 0.50 \
  --bimodal_ratio 0.40 \
  --scaffold_ratio 0.10 \
  --max_seq_len 16384 \
  --min_supervised_tokens 64 \
  --strict \
  --output_root artifacts/datasets
```

### 7.2 调度训练与评测

```bash
cd /Users/yang/projects/hybrid-rope

.venv/bin/python scripts/isolated/longinst/run_llama8k_theory_v1.py \
  --execute \
  --base_model_path <LLAMA3_MODEL_PATH> \
  --longalpaca_path <LONGALPACA_JSONL> \
  --longqa_path <LONGQA_JSONL> \
  --wikitext_train_path <WIKITEXT_TXT> \
  --mixed_dataset_dir <MIXED_DATASET_DIR> \
  --mixed_dataset_split train \
  --longbench_local_data_dir <LONGBENCH_DATA_DIR> \
  --qwen_seed42_json <QWEN_SEED42_JSON> \
  --qwen_seed1337_json <QWEN_SEED1337_JSON> \
  --morning_reference_json <MORNING_REF_JSON>
```

---

## 8. 风险清单（审稿人会盯的点）

1. 数据先验错配（最危险）：会导致“检索涨、推理崩”的伪进步。
2. 只看均值不看失败边界：会被质疑 cherry-pick。
3. seed 不足或协议漂移：统计显著性失效。
4. 把 loss 当主证据：高概率误判。
5. 不报告负向任务：会被审稿人直接打回。

---

## 9. 明天开工前 10 分钟检查单

1. `mix_manifest.json`、`quality_report.md`、`label_mask_preview.json` 是否齐全。
2. manifest 比例和训练后 post-token 比例是否都过阈值。
3. run_tag 是否唯一，避免覆盖旧产物。
4. gate 是否通过再进入 lb21。
5. paired stats 是否满足最小 run pairs。

---

## 10. 参考文献（本方案依据）

1. [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
2. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
3. [LongLoRA](https://arxiv.org/abs/2309.12307)
4. [LongAlign](https://arxiv.org/abs/2401.18058)
5. [LongBench](https://arxiv.org/abs/2308.14508)
6. [RULER](https://arxiv.org/abs/2404.06654)
7. [NoLiMa](https://arxiv.org/abs/2502.05167)
8. [Lost in the Middle](https://arxiv.org/abs/2307.03172)
9. [LongRoPE](https://arxiv.org/abs/2402.13753)
10. [LongRoPE2](https://arxiv.org/abs/2502.20082)
11. [Transformers bitsandbytes docs](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes)
12. [NeurIPS Paper Checklist Guide](https://nips.cc/public/guides/PaperChecklist)
13. [NVIDIA RTX PRO 6000 Blackwell Family](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/)

