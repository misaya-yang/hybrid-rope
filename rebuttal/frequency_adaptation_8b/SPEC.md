# LLaMA-3-8B RoPE Frequency-Adaptation Experiment

状态：`implementation-ready / no result yet`

目的：回答一个窄问题——**在一个已经用 15T+ token 训练、内部已经形成原生 Geo-RoPE 频率分工的 LLaMA-3-8B checkpoint 上，怎样提供足够且方向正确的梯度，使它真正适应 EVQ-Cosh 的频率重分配？**

本实验不是 rebuttal 的装饰性补充，也不预设 EVQ 会赢。任何训练启动、loss 下降、PPL 改善或频率注入成功都不等于“模型学会了频率分配变化”。成功必须由 held-out、counterfactual、距离分桶的任务能力证明。

它是一套新的 mechanism-adaptation protocol，**不替代** paper-lineage LongAlpaca Geo/EVQ clean pair，也不能修补 historical EVQ-LoRA row 的 dataset provenance。两条实验线的问题、数据和可升级 claim 必须分开。

## 1. Bottom line

旧 positional-distillation pilot 不值得继续：它在 8K 上一次性把原生 Geo 改成 EVQ，只训练 q/k LoRA，并要求 student 拟合 Geo teacher 的最终隐藏态。对多个相对距离同时存在时，一个 position-independent q/k adapter 一般不存在能够把两个不同 RoPE generator 全局共轭起来；隐藏态 MSE 还允许大量不依赖远端 source 的局部梯度。因此，失败既不能说明 EVQ 不可适应，也不能验证“梯度不足”。

新实验采用四个原则：

1. **任务梯度先于频率迁移。** 先在原生 Geo 下让同一套 q/k/v/o LoRA 学会明确的长距离随机检索。
2. **连续迁移，不做瞬时替换。** 从模型实际加载出的 native Geo frequency tensor 出发，在 log-frequency 空间连续走到 exact EVQ tensor。
3. **只监督答案。** filler、source 和 query 全部 mask；每个样本提供 12 个随机 value token 加 EOS 的监督，cross-entropy 只在这些 token 上归一化。
4. **能力与机制同时观测。** 用 source-swap/source-removal counterfactual 证明模型确实读取远端 source；用逐 rotary pair 的 q/k LoRA 梯度与更新能量判断频率通道是否收到学习信号。

## 2. RoPE 底层约束与实验含义

对第 \(k\) 个二维旋转平面，位置 \(m,n\) 的 attention contribution 可写为

\[
q_{m,k}^{\top}R_{\omega_k}(n-m)k_{n,k}.
\]

改变 \(\omega_k\) 不是简单改变一个“位置尺度”：它改变每个 rotary plane 对所有相对距离 \(\Delta=n-m\) 的相位响应。预训练后的 q/k projection 已经把不同内容特征分配到这些平面；v/o 又决定被选中的信息怎样被运输和读出。因此：

- 只改 frequency tensor 而不给 position-dependent task gradient，模型没有理由重新分配通道；
- q/k-only 可以改变寻址，却不能保证已有 value transport/readout 适合新形成的检索电路；
- 用固定 adapter 精确恢复旧 Geo hidden states，相当于要求它同时抵消所有 \(R_{\omega'_k}(\Delta)-R_{\omega_k}(\Delta)\)，一般不可实现；
- 真正需要学习的是：在新的 phase basis 下，哪些内容进入哪些 q/k rotary planes，以及 v/o 怎样传递被选中的随机 value。

频率路径定义为

\[
\log \omega_k(s)
=(1-h(s))\log \omega_k^{\mathrm{Geo}}
+h(s)\log \omega_k^{\mathrm{EVQ}},
\qquad h(s)=3s^2-2s^3.
\]

这里必须直接连接两个真实 tensor，不能令 \(\tau:0\to1.414\)：当前 EVQ 构造在 \(\tau=0\) 时是 midpoint-discretized Geo，而 LLaMA 原生 Geo 使用 endpoint \(k/K\)，两者并不相等。log-space 路径保持频率为正，起点严格等于 checkpoint 的 native tensor，终点严格等于 canonical EVQ-Cosh midpoint tensor。

## 3. 可证伪假设

### H1：旧实验的主要问题是任务相关梯度不足/方向错误

若 H1 成立：

- native-Geo warm-up 会快速获得 held-out nonce retrieval 和 source-swap 能力；
- q/k rotary pairs 的累计 LoRA-B gradient energy 非零且不是只集中在少数偶然通道；
- 频率平滑迁移时 EVQ arm 能保留该能力，优于一次性替换后的坍塌模式。

### H0：在可用 LoRA 容量和预算下，8B checkpoint 不能稳定重组已有 frequency-plane specialization

支持 H0 的结果包括：Geo warm-up 已经学会任务，但 EVQ arm 在连续迁移中仍稳定失去 source dependence；提高距离而非改变频率时 Geo arm 保持正常；梯度存在但无法形成有效的 q/k channel update。此时不能继续用更多 benchmark 掩盖失败，应把 retrofit 路径降级为负结果/限制。

### 不能由本实验推出

- EVQ 是 8B long-context SOTA；
- \(\tau=1.414\) 全局最优；
- surrogate 或 collision kernel 等价于 LM loss；
- LoRA rank 64 是理论充分条件；
- retrieval 改善自动意味着一般下游任务改善。

## 4. 最小 matched protocol

模型固定为同一份 LLaMA-3-8B-Instruct bytes；默认 BF16、SDPA、gradient checkpointing。LoRA 只放在 `q_proj,k_proj,v_proj,o_proj`，`r=64, alpha=128, dropout=0.05`。`tau=1.414` 只作为论文已有 operating point，不做 tau sweep。

所有 phase 都保持每个 optimizer step 恰好 32,768 个 physical sequence tokens：

| Phase | 起点/终点 | seq len | source-query distance | steps | effective batch | tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| W: shared Geo warm-up | Geo → Geo | 4,096 | 256–2,048 | 64 | 8 | 2,097,152 |
| H: frequency homotopy | Geo → arm target | 4,096 | 512–2,048 | 128 | 8 | 4,194,304 / arm |
| E8: exact endpoint | target → target | 8,192 | 2,048–6,144 | 128 | 4 | 4,194,304 / arm |
| E16: exact endpoint | target → target | 16,384 | 6,144–14,336 | 96 | 2 | 3,145,728 / arm |

W 只训练一次。之后从**同一个 W adapter byte snapshot**分叉：

- `geo`：H/E8/E16 始终使用 native Geo；
- `evq`：H 中平滑迁移，E8/E16 使用 exact EVQ-Cosh。

两臂使用同一 token tensor、相同 sample order、seed、LoRA、optimizer、steps 与 evaluator。分叉后的每臂是 11,534,336 tokens；seed-42 全部两臂加共享 warm-up 共 25,165,824 tokens。

每个 phase 边界都会重建 optimizer；Geo/EVQ 在同一 phase 使用相同的零状态 optimizer 和 scheduler。这样避免把 W 的单一 optimizer state 复制成两个难以核验的 mutable 分支，也使不同 sequence length 的阶段边界清楚。代价是本实验只比较相同 phase protocol 下的两臂，不把跨 phase loss 曲线解释为连续优化轨迹。

不在 v1 中加入 LongAlign replay。原因是当前首先要验证 position-dependent gradient 是否足够；混入一般 LM/instruction token 会再次稀释因果解释。若 retrieval 学会但 temporal NLL 或 instruction behavior 明显破坏，再单独触发一个低比例 assistant-only replay 试验。

## 5. 数据：让答案只能来自远端 source

复用 `prepare_positional_distill_data.py` 生成的 frozen FineWeb-Edu plain-text `train.pt` / `validation.pt`，但不复用旧 hidden-state objective。训练和评估 filler 保持 document-disjoint。

每个样本在 token level 构造，长度严格等于 phase seq len：

- user/assistant 边界从同一 tokenizer 的真实 chat template 提取，不用 plain-completion 另教一种输出格式；
- 75% single nonce key-value retrieval；
- 25% last-write-wins update，包含同 key 的旧 value 与远端最新 value；
- key 与 value 从 tokenizer 的普通单-token词表中确定性抽取；train/eval nonce pools 分离；
- value 固定 12 tokens，query 固定在末端；
- 主 distance 定义为 `answer_start - 1 - source_value_start`，即 RoPE 真正作用的首答案 predictor query 到 source value 的相对位置；由于 source 与 answer 都是同长的 12-token value，第 $j$ 个答案 predictor 到第 $j$ 个 source token 的距离保持同一个常数；另存 textual `query_key_distance`，两者不混用；
- 训练只保留 `input_ids + answer_start/end`，labels 在 dataset 中按需生成；prompt labels 全为 `-100`。

每个 held-out eval group 生成三条等长记录：

1. `original`：source 和 target 一致；
2. `swapped`：只替换 source value，target 同步替换；
3. `source_removed`：把整个 source span 换成等长、独立的 filler，但仍测原 target 的 NLL。

这三联能排除“记住模板/位置/答案先验”：可靠模型应在 swapped 条件下跟随新 source，在 source_removed 条件下显著降低原答案概率。

## 6. 训练信号与诊断

### 6.1 Primary loss

唯一训练 objective 是 answer-only causal CE。Prompt label 全为 `-100`，且训练器只保留“答案前一位置 + 12 个 value token + EOS”对应的 14 个 tail logits，再用前 13 个位置预测完整答案。因此 4K/16K filler 既不进入 loss 分母，也不会触发 128K-vocab 的全序列 LM-head 计算；长序列成本保留在真正需要的 attention/hidden-state 路径。若服务器 Transformers 不支持 `logits_to_keep`，代码直接停止，不静默退回昂贵的 full-sequence logits。

### 6.2 Frequency transition

H phase 在每个 optimizer step 开始时更新所有 rotary modules 的 `inv_freq`，清理 cos/sin cache，并在最后强制写入 exact target。每个输出 adapter 同时保存 `custom_inv_freq.pt` 和 immutable `run_protocol.json`；后续 phase 必须验证上一个 frequency artifact 和 adapter hash。

### 6.3 Gradient sufficiency diagnostic

LLaMA 的 `rotate_half` pairing 是每个 head 内 \((i,i+d_{head}/2)\)，不是相邻的 \((2i,2i+1)\)。训练代码对 q/k LoRA-B 注册 gradient hook，并按这一 half-split pairing 累积：

- per-pair q gradient energy；
- per-pair k gradient energy；
- nonzero pair coverage；
- phase 结束后的 effective LoRA update energy；
- 每个 pair 的 \(|\log\omega^{target}-\log\omega^{Geo}|\)。

这些量只诊断“是否收到以及如何分布梯度”，不能单独证明模型学会；最终判断仍由 counterfactual capability 给出。

## 7. Gate：先决定实验是否值得继续

### Gate W — 任务/梯度是否成立

Gate 解释前必须同时具有 base Geo 与 W adapter 的 held-out 4K triplets。为缩短上卡后的 time-to-signal，执行顺序默认先训练并评估 W；只有 W 出现任务信号时，再显式运行 `eval-base warmup` 补齐基线。满足以下条件才分叉：

- original 与 swapped 的 exact retrieval 都达到可用水平，且 W 不低于 base；
- 至少 75% group 的 source removal 使原答案 per-token NLL 上升；
- q/k gradient pair coverage 非零、无 NaN/Inf；
- loss 或 exact match 的改善不能只来自 update/KV 某一个模板。

若 r64 的 Geo warm-up 失败，只允许在 **同一个 W gate** 上尝试一次 `r=128, alpha=256`；不能看 EVQ 结果后再选 rank。r128 仍失败则停止，先修数据/训练，不跑 EVQ。

### Gate H — 是否真的适应到 exact EVQ

H 结束后必须验证 frequency tensor 精确到达 target，并在 4K held-out triplets 上检查 source dependence。EVQ 若在 smooth transition 中持续坍塌，而 Geo control 正常，则停止 E8/E16；这已经是高价值负信号。

### Gate E8/E16 — 是否获得距离泛化

仅当两臂都通过前一 gate 才增加距离。主比较是 matched Geo versus EVQ 的：

- exact token-sequence match；
- answer NLL；
- original+swapped pair consistency；
- source-removal paired NLL increase；
- KV 与 update 分项；
- distance buckets。

seed 42 只有在 EVQ 相对 Geo 出现与 source-dependent capability 一致的方向时才触发 seeds 43/44。只改善 PPL、没有 task signal，不增加 seed。32K 训练默认不运行；只有两臂 16K 能力成立但 32K eval 同时失败时，才考虑 48-step、batch-1 的 E32。

## 8. Secondary guardrails

不建立新 benchmark zoo。复用仓库已有 evaluator：

- temporal holdout NLL/PPL：检查短训练是否破坏一般语言建模；
- `eval_ruler.py` / `eval_ruler_logprob.py` 的小型预注册子集：只在 E16 gate 通过后运行；
- base / W / Geo / EVQ 四个 checkpoint 的频率 artifact 验证。

主能力端点失败时，不用 secondary PPL 包装成成功。

## 9. 明确不做

- 不再做 final-hidden-state MSE teacher distillation；
- 不做自定义 doubled-Q/K attention 或 logit homotopy；
- 不同时驻留 teacher 与 student；
- 不解冻 full attention / MLP，不加 LoRA 到 MLP；
- 不 sweep tau、base、YaRN scale 或 benchmark；rank 只保留 Gate W 预注册的一次 r128 retry，不看 EVQ 结果调参；
- 不把 training loss、frequency injection、adapter update 当作能力证据；
- 不在结果出现前修改论文数字或 claim tier。

## 10. 代码入口

- `curriculum.py`：phase contract、frequency homotopy、answer mask、rotary-pair diagnostics；
- `prepare_data.py`：从 frozen plain-text tensors 构造 matched train/eval bundles；
- `train.py`：单 phase LoRA 训练与 frequency callback；
- `evaluate.py`：counterfactual answer NLL / exact retrieval；
- `run_seed42.sh`：只编排 seed-42 gate，不自动启动额外 seed。

当前仓库没有本地 8B weights 或 frozen filler tensors。它们应在非 GPU 环境先准备：

```bash
python experiments/lora_evq_v2/prepare_legacy_model_manifest.py \
  --model_dir "$EVQ_FREQ_MODEL" \
  --output "$EVQ_FREQ_MODEL_MANIFEST"

python experiments/lora_evq_v2/prepare_positional_distill_data.py \
  --tokenizer "$EVQ_FREQ_MODEL" \
  --output_dir "$EVQ_FREQ_FILLER_DIR"

bash rebuttal/frequency_adaptation_8b/run_seed42.sh prepare
```

运行命令和环境变量见 `run_seed42.sh`; 每个训练命令完成后会自动评测**当前 checkpoint**，但下一 phase 和另一 arm 始终需要显式启动，脚本不会绕过 gate 串行扩张 GPU job。模型 full-byte manifest 必须在租用 GPU 前由现有 `prepare_legacy_model_manifest.py` 生成；训练和评测只检查该 manifest 的文件大小/mtime与已有 SHA-256，不在模型上卡后重新 hash 8B 权重。
