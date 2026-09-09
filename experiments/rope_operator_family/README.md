# 算子族压缩：方法定位代码

当前推进这一种方法；归因对照为同架构、同初始化、同优化预算的 attention 输出蒸馏。每次入口使用一份配置、一份输入，不自动运行预算网格或多任务 benchmark。

本轮的 [Core Experiment、各实验价值与失败处理](EXPERIMENT_VALUE.md) 已单独整理；核心是同KV预算下的远距离依赖修复，备选读数是同一问题的正确答案条件NLL。

方法已经实现为不同的 A/B、可学习旋转频率和内容/value latent 的联合拟合。FreqFold/PCA 仅作初始化，默认不搜索其配置。模型接入保存共享 `c,k_R` 紧凑缓存；首个目标是现有 Qwen2.5-1.5B-Instruct。

首次真实 GPU 运行已完成，见 [配对结果与原始证据](results/20260909_gpu/REPORT.md)：缓存减半；输出蒸馏的 NLL 为 3.94，score 拟合为 9.99，原模型为 2.65。两边均未恢复冻结检索题，当前不支持 score 目标的实际优势。

随后按用户要求完成 [三项后续与检索损伤分析](FOLLOWUP_RESULTS_20260909.md)：BKV+KD 4.99、attention KL+KD 3.75、逐层学生输入KD 4.57；三项均未恢复检索。原模型同题两次贪心输出完全一致。新增平衡、KL和逐层恢复代码通过全部18项远端检查。

## 已准备的入口

| 命令 | 这一种方法中研究/验证的内容 |
|---|---|
| `prepare` | 自然文本重编码、按来源文档分开校准与留出 |
| `capture` | 原模型真实 pre-RoPE Q/K/V；抽样 query，保留完整 keys |
| `fit` | 单条 A/B/频率/内容优化轨迹，逐层完成并保存 |
| `diagnose` | 静态内容、距离误差、generator leakage、margin、完整 softmax 输出；校准预测与留出观测 |
| `evaluate` | 接入全部层后的自然文本 NLL |
| `generate` | 一个用户指定 prompt 的真实逐 token 生成 |
| `profile` | 当前配置的实际缓存、prefill/decode 成本 |
| `report` | 汇总当前配置的原始结果，不自动判胜或选择方法 |
| `compare-objectives` | 核对一对拟合的初态、实际文档/位置轨迹与评测输入，输出算子目标相对输出蒸馏的配对差值 |

实现文件为 `operator.py`（方法）、`model.py`（实际模型/缓存）、`study.py`（拟合和机制观测）、`prepare.py`（数据）、`run.py`（入口）、`report.py`（结果）。`DESIGN.md` 保留较完整的研究背景；当前执行以这里的单方法流程为准。

## 环境

已在本机 `aidemo` 环境验证；新机器使用 `/root/miniconda3/bin/python`。代码依赖 PyTorch、Transformers 及其 tokenizer 依赖，不需要额外安装训练框架或复制一整份 TransMLA 环境。正式运行时设备由 `--device` 指定；小模型代码检查使用 CPU。

远端准备目录：`/root/autodl-tmp/operator_family_prepare_20260909`。

从包含 `experiments/` 的目录运行：

```bash
cd /root/autodl-tmp/operator_family_prepare_20260909
OPERATOR_PYTHON=/root/miniconda3/bin/python
OPERATOR_MODEL=/root/autodl-tmp/qwen25_1p5b_32k
```

模型代码支持标准 full-RoPE Qwen2/Llama 投影接口，目前验证的是 Qwen2。dynamic/scaled RoPE、Q/K normalization、sliding attention 需要相应的原始算子定义，不会被当成当前 Qwen 模型直接转换。

## 输入

当前远端输入已经准备好，见 `input_readiness.json`：`work/data` 含32篇校准与8篇独立留出文档，来自官方PG19的不同书籍连续前缀，并用当前Qwen tokenizer编码；`work/core/prompt.txt` 是一个8189-token、412条记录的关联检索输入，答案为 `f83a9144`（另存 `work/core/answer.txt`）。来源、文档选择、tokenizer和文件哈希保存在 `work/sources/manifest.json`、`work/data/manifest.json` 和 `work/core/manifest.json`。首次 GPU 运行已完成；实际回答与留出 NLL 见上方结果链接。保留以下入口用于复现，已有 capture/初始化/拟合应复用。

自然文本来源可以是 JSONL、单个 `.txt` 或包含多个 `.txt` 的目录。JSONL 每条为：

```json
{"source_id": "document-or-book-id", "text": "完整自然文本……"}
```

同一文档/书的多个片段共用 `source_id`；准备器每个来源只选择一条合格记录。校准、留出按来源分开。使用模型自身 tokenizer，不复用旧 GPT-NeoX token IDs。

以下命令是单方法各实验的入口示例，不会自行提交一张实验矩阵。`texts.jsonl` 应替换为实际原始文本文件，`prompt.txt` 是要实际检验的一条问题。

### 准备与捕获

```bash
$OPERATOR_PYTHON -m experiments.rope_operator_family.run prepare \
  --model "$OPERATOR_MODEL" --source texts.jsonl --out work/data \
  --calibration-documents 32 --validation-documents 8 \
  --calibration-length 2048 --evaluation-length 8192

$OPERATOR_PYTHON -m experiments.rope_operator_family.run capture \
  --model "$OPERATOR_MODEL" --data work/data --out work/capture \
  --device cuda --dtype bfloat16
```

`capture` 只做原模型前向一次，校准与诊断复用这些内容。每条只存约64个 query 行和全部 keys/values。默认40篇、2K窗口、28层的 Q/K/V payload 约2.6GB，避免存全 attention 矩阵。

### 拟合这一个配置

```bash
$OPERATOR_PYTHON -m experiments.rope_operator_family.run fit \
  --capture work/capture --out work/initialization \
  --content-rank 192 --rotary-dim 64 --initialize-only --device cuda

$OPERATOR_PYTHON -m experiments.rope_operator_family.run fit \
  --capture work/capture --init-from work/initialization --out work/operator \
  --content-rank 192 --rotary-dim 64 --steps 500 \
  --max-position-scale 8 --device cuda
```

默认拟合全部层。每层共用静态内容/value 分支和 rotary 分支，A/B 可以不同。目标是 normalized real-score MSE 加 value reconstruction；完整输出 loss 的权重默认0，可通过 `--output-weight` 在当前方法里研究。每隔一步使用真实位置，其他步在同一内容上采样较大的位置尺度；因果 mask 始终按原 token 顺序保留。

500步和上述数据量是运行起点，不是结论阈值。已经完成的层会按相同配置复用；中断后重复同一条命令即可继续剩余层。输出含每步损失、实际位置尺度、梯度和耗时。

定位一个具体问题时，可在**一次独立运行**中固定相关部分：

- `--freeze-projections --freeze-content`：A/B 和静态内容固定，只研究旋转更新；Δ=0 内容严格不变。
- `--freeze-frequency`：只研究投影/内容更新。
- `--freeze-content`：保持共同内容/value 分支，只研究 A/B 与频率。
- `--max-position-scale 1`：只用真实观测位置。

入口不会自动把这些选项展开成多个实验臂。

### 机制观测与实际模型验证

```bash
$OPERATOR_PYTHON -m experiments.rope_operator_family.run diagnose \
  --capture work/capture --factors work/operator --layer 15 \
  --position-scale 8 --out work/diagnostic.json --device cuda

$OPERATOR_PYTHON -m experiments.rope_operator_family.run evaluate \
  --model "$OPERATOR_MODEL" --data work/data --factors work/operator \
  --out work/nll.json --device cuda

$OPERATOR_PYTHON -m experiments.rope_operator_family.run generate \
  --model "$OPERATOR_MODEL" --factors work/operator --prompt work/core/prompt.txt --chat \
  --expected-answer f83a9144 --max-new-tokens 32 --out work/answer.json --device cuda
```

`diagnose` 在一份已拟合模型上测真实 score、静态误差、相对位置响应误差 `position_response_mse`、距离分桶、top-key margin、attention KL、log-mass 与 value output。位置响应误差直接比较 `(ŝ_Δ−ŝ_0)−(s_Δ−s_0)`。它同时报告校准估计和留出观测；相位回放与真正长文本 NLL 分开解释。

`evaluate` 在完整模型上预测每篇最后256个目标 token。`generate` 只生成指定的一条问题，保留原始回答和 token IDs，不自动铺开任务集。

提供 `generate --expected-answer ...` 后，还会报告严格答案匹配及条件答案NLL。关键归因对照的完整命令见 [EXPERIMENT_VALUE 第6节](EXPERIMENT_VALUE.md#6-可直接执行的core入口)：先用 `--initialize-only` 保存共同起点，两边用 `--init-from` 加载；本方法用 `--score-weight 1 --output-weight 0`，普通输出蒸馏用 `--score-weight 0 --output-weight 1`，其余条件一致。初始化本身不作为这一归因比较的对照。

### 实测时间与缓存

```bash
$OPERATOR_PYTHON -m experiments.rope_operator_family.run profile \
  --model "$OPERATOR_MODEL" --factors work/operator \
  --length 8192 --decode-tokens 64 --out work/profile.json --device cuda

$OPERATOR_PYTHON -m experiments.rope_operator_family.run report \
  --factors work/operator --diagnostic work/diagnostic.json \
  --evaluation work/nll.json --generation work/answer.json \
  --profile work/profile.json --out work/result.md
```

默认 backend 在总 attention width≤256 时用 SDPA，否则用分块精确 softmax reference，记录实际采用的路径。持久缓存仅有 `c,k_R`，不缓存展开 K/V。SDPA 的临时拼接和 padding 会影响峰值，profile 将其计入实际分配；`chunked` 是可运行的有界内存 reference，不代表优化 kernel 的性能。

此前按多组配置和整套多任务生成估计的“数天到数周”，不用于当前方法定位。当前配置的捕获、拟合、NLL与单次生成分别计时，以这些实测量估算下一次运行；没有用小模型测试耗时冒充1.5B运行时间。

## 验证

```bash
OMP_NUM_THREADS=2 HF_HUB_DISABLE_PROGRESS_BARS=1 \
  $OPERATOR_PYTHON -m unittest experiments.rope_operator_family.test_operator -v
```

检查涵盖：原生 GQA 的精确表示、同频混合、原模型/压缩因子的 affine 接入、padding、分段 prefill 和 decode、实际缓存字节、FP32/BF16、拟合梯度、静态内容保持，以及从准备输入到报告的整个 CLI 路径。

这些验证使用随机小型 Qwen 和合成文本，证明代码路径可运行；尚未作为真实1.5B方法效果或GPU性能结果。具体环境与通过记录见 `validation.json`。
