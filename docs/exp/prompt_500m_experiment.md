# Claude Code Prompt: 500M EVQ From-Scratch Training + Passkey Evaluation

> 直接复制下面分隔线内的内容给 Claude Code

---

## 任务概述

我需要你在现有的 `scripts/m4_evq_sweep/run_evq_sweep.py` 基础上，完成以下工作：

1. 新增 500M 模型配置
2. 实现适用于从零训练小模型的 Passkey Retrieval 评测（含训练数据混入）
3. 数据源从 TinyStories 升级为 FineWeb-Edu（多领域高质量网页文本）
4. 完整的校验机制

**重要背景**：这是一个 NeurIPS 2026 投稿项目，预算有限。我们已经有 50M 和 125M 的实验结果。500M 是 scaling law 证据链的关键一环（50M → 125M → 500M → 1B）。我的服务器是 RTX PRO 6000 Blackwell 96GB，6 元/小时。

**实验范围**：核心对比 **geometric baseline (τ=0)** vs **EVQ-Cosh (τ=最优值)**。训练只跑这两组。但在评测阶段，额外对 geometric 训出的模型做 inference-time 的 PI frequency scaling 作为零成本第三 baseline（不需要额外训练，只在 eval 时替换 inv_freq）。

---

## 1. 新增 500M Tier Config

在 `run_evq_sweep.py` 的 `TIER_CONFIGS` 中新增 `"500m"`。参数设计要求：

- `head_dim` 必须是 64（与 50M/125M/350M 一致，保证 RoPE 维度可比）
- `hidden_size` 和 `num_heads` 必须满足 `hidden_size = num_heads * head_dim`
- `num_heads` 选常见值（16 或 20），避免奇怪的 head 数
- 目标参数量 ~500M（允许 450M-550M 范围）
- `vocab_size`: 50304（与其他 tier 一致）
- `seq_len`: 2048（训练序列长度，与其他 tier 一致）
- `max_position_embeddings`: 2048
- `train_tokens`: 500_000_000（5 亿 token）
- `lr`: 参考 GPT-3 scaling law，500M 模型建议 1.5e-4
- `eval_lengths`: [2048, 4096, 8192]（8192 为标准外推测试；16384 标注为 **optional**，见下文 OOM 说明）
- `eval_chunks`: 10
- `batch_size`: 先写 4，后面会根据 GPU 自动调整（已有的 CUDA VRAM 检测逻辑）

请先算出具体的 hidden_size / num_layers / num_heads / intermediate_size，确认参数量在目标范围，然后写入配置。建议参考：
- hidden=1024, layers=28, heads=16, intermediate=4096 → ~520M
- 你自己算，选最接近 500M 的配置

**重要说明**：500M tokens 训练 500M 参数模型，远低于 Chinchilla 最优的 ~20x（即 ~10B tokens）。这是 **budgeted regime**（预算约束训练），我们对比的是**同 token 预算下不同频率分配方案的相对性能**，而非追求模型饱和性能。这个 framing 在论文中会明确说明。

**校验**：写完后用一个小脚本实例化模型，打印实际参数量，确认在 450M-550M 范围内。

---

## 2. 数据源升级：FineWeb-Edu

### 为什么换数据

之前用 TinyStories（合成儿童故事）一直被审稿人攻击：单一领域、文本极短、词汇简单，无法证明方法的泛化性。

### 使用 FineWeb-Edu 的 `sample-10BT` 子集

数据集：`HuggingFaceFW/fineweb-edu`，config=`sample-10BT`

**关键：不要用默认的全量 1.3T 数据集！** 官方提供了预采样子集：
- `sample-10BT`：~10B tokens，28.5GB（约 970 万条文档）— **用这个**
- `sample-100BT`：~100B tokens
- `sample-350BT`：~350B tokens
- 默认 config：全量 1.3T tokens（太大，不要用）

我们只需要 500M tokens 训练 + 5M tokens 验证，从 `sample-10BT` 的 10B tokens 中 streaming 取就绰绰有余了。

优势：
- **多领域覆盖**：科学、历史、法律、技术文档、百科等，审稿人无法攻击"单一领域"
- **文本长度分布自然**：远长于 TinyStories，对长上下文位置编码测试更有意义
- **学术界广泛认可**：2024 年发布后大量论文采用
- **streaming 极快**：sample-10BT 只有 ~13 个 parquet 分片，取 500M tokens 只需遍历前几个分片

**注意：FineWeb-Edu 本身已经是 score≥3 过滤后的子集，不需要再做 score filter。直接用即可。**

### 实现要求

修改 `load_data()` 和 `load_val()` 函数，添加 `--dataset` 参数：

```bash
--dataset fineweb-edu  # 默认，用于 500M 实验
--dataset tinystories  # 向后兼容已有的 50M/125M 实验
```

#### 数据加载（参考已有代码）

项目中已有一个可用的参考实现，在 `artifacts/a800_2026-02-13/run_llama3_hybrid_lora_v3.py` 中：

```python
# 已验证可用的加载方式：
from datasets import load_dataset

ds = load_dataset("HuggingFaceFW/fineweb-edu",
                  name="sample-10BT",       # 关键：指定子集！
                  split="train",
                  streaming=True)

# text 字段名是 "text"
for x in ds:
    ids.extend(tokenizer.encode(x["text"], add_special_tokens=False))
    if len(ids) >= target_tokens:
        break
```

#### 训练集/验证集切分

**不要用 token-level skip/take！** HF streaming 的 `skip()` 按样本条目计数，在大规模 streaming 上极慢。

正确做法：

```python
if dataset == "fineweb-edu":
    # 训练集：正常按顺序 streaming 取前 train_tokens 个 token
    train_ds = load_dataset("HuggingFaceFW/fineweb-edu",
                            name="sample-10BT",
                            split="train", streaming=True)

    # 验证集：用不同的 shuffle seed，取不同的样本子集
    val_ds = load_dataset("HuggingFaceFW/fineweb-edu",
                          name="sample-10BT",
                          split="train", streaming=True)
    val_ds = val_ds.shuffle(seed=99999, buffer_size=10000)
    # 从 shuffled stream 中取 5M tokens 作为验证集
    # 由于 seed 不同，与训练集重叠概率极低（500M/10B = 5%）

elif dataset == "tinystories":
    # 保持原有逻辑不变
```

**注意**：
- **必须设环境变量 `HF_ENDPOINT=https://hf-mirror.com`**（服务器在中国大陆）
- 用 streaming 模式，不要一次性下载
- tokenizer 继续用 `gpt-neox-20b`（与已有实验一致）
- 训练集和验证集用不同 shuffle seed 分开，在论文中声明是 sample-level split

### 下载失败容错

**必须实现自动 fallback。** 如果 FineWeb-Edu 下载失败（镜像问题/网络超时），自动切换到备选数据集：

```python
def load_streaming_dataset(dataset_name, tokenizer, target_tokens, seq_len):
    """带 fallback 的数据加载。"""

    # 候选数据集列表，按优先级排列
    candidates = [
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
        ("cerebras/SlimPajama-627B", None, "text"),
        ("roneneldan/TinyStories", None, "text"),  # 最终保底
    ]

    for ds_name, config, text_key in candidates:
        try:
            print(f"[data] Trying {ds_name} (config={config})...")
            kwargs = {"split": "train", "streaming": True}
            if config:
                ds = load_dataset(ds_name, name=config, **kwargs)
            else:
                ds = load_dataset(ds_name, **kwargs)

            # 尝试取前 1000 个 token 验证连接
            ids = []
            for x in ds:
                txt = x.get(text_key)
                if not txt:
                    continue
                ids.extend(tokenizer.encode(txt, add_special_tokens=False))
                if len(ids) >= 1000:
                    break

            if len(ids) < 1000:
                raise RuntimeError(f"Only got {len(ids)} tokens from {ds_name}")

            print(f"[data] SUCCESS: using {ds_name} (config={config})")
            # 重新开始 streaming 取全量数据
            # ... 完整加载逻辑 ...
            return data

        except Exception as e:
            print(f"[data] FAILED: {ds_name} — {e}")
            print(f"[data] Falling back to next candidate...")
            continue

    raise RuntimeError("All dataset candidates failed!")
```

**fallback 顺序**：
1. `HuggingFaceFW/fineweb-edu` (sample-10BT) — 首选
2. `cerebras/SlimPajama-627B` — 备选（同样多领域）
3. `roneneldan/TinyStories` — 最终保底（至少能跑起来）

---

## 3. 实现 Passkey Retrieval 评测（关键！）

### 背景

现有的 `scripts/eval_passkey_teacher_forcing.py` 和 `scripts/eval_niah_heatmap.py` 是给 Llama-3-8B Instruct 模型写的，依赖指令格式和 LoRA adapter，**完全不适用于从零训练的小模型**。

我需要一个全新的、**适用于纯语言模型（非指令微调）**的 passkey 评测。

### 核心设计决策：必须在训练数据中混入少量 passkey 样本

**为什么必须混入**：从零训练的小模型如果从没见过 `<<PASS:...>>` 这种模式，它会把触发标记当作无意义噪声，NLL gap 会退化到 ≈0，无法区分位置编码的好坏。这样会把"模型不会做任务"误判成"位置编码不行"。

**混入比例**：在训练数据中混入 **0.5%–1%** 的合成 passkey 样本。这足以让模型学会"看到 `<<PASS:` 就去回忆前面出现过的数字"这个 copy 模式，但不会显著影响语言建模质量。

**混入方法**：

```python
def make_passkey_training_sample(filler_tokens, tokenizer, seq_len=2048):
    """生成一条 passkey 训练样本。

    格式：[filler tokens...] <<PASS:XXXXX>> [filler tokens...] <<PASS:XXXXX>>
    - 在随机深度插入 passkey
    - 在序列末尾重复同一 passkey（作为 target）
    - 模型通过正常 next-token prediction loss 学会 copy
    """
    passkey = "".join([str(random.randint(0, 9)) for _ in range(5)])
    # ... 构造序列 ...
    return input_ids  # shape: (seq_len,)

class MixedDataset(torch.utils.data.Dataset):
    """混合语言建模数据 + passkey 样本。"""
    def __init__(self, lm_data, filler_tokens, tokenizer,
                 passkey_ratio=0.005, seq_len=2048):
        self.lm_data = lm_data          # shape: (N, seq_len)
        self.filler_tokens = filler_tokens
        self.tokenizer = tokenizer
        self.passkey_ratio = passkey_ratio
        self.seq_len = seq_len

    def __len__(self):
        return len(self.lm_data)

    def __getitem__(self, idx):
        if random.random() < self.passkey_ratio:
            return make_passkey_training_sample(
                self.filler_tokens, self.tokenizer, self.seq_len)
        return self.lm_data[idx]
```

**passkey 格式**：固定使用分隔符格式 `<<PASS:X-X-X-X-X>>`（每位数字用 `-` 分隔），不要用连续数字格式。原因：GPT-NeoX tokenizer 会把多位数字合并成单 token，用分隔符确保每位数字独立编码，结果可复现。

### 评测方法

```
方法 A: Teacher-Forcing NLL Gap（主要方法，更严谨）
1. 用验证集真实 token 填充，生成长度为 L 的序列
2. 在深度 d% 处插入: <<PASS:X-X-X-X-X>>
3. 在序列末尾追加触发: <<PASS:
4. Teacher forcing 计算模型对正确 5 位数字的 NLL
5. 同时计算一个错误 passkey（每位都不同）的 NLL
6. 指标 = NLL_wrong - NLL_correct（越大越好）
7. NLL_correct < NLL_wrong → 判定 "retrieved"

方法 B: 自回归生成（辅助，直观）
1. 同上构造序列，但截断到 <<PASS:
2. 模型自回归生成 5 个 token
3. 精确匹配 passkey → 判定 "retrieved"
```

### 实现要求

创建新文件 `scripts/m4_evq_sweep/eval_passkey_scratch.py`，包含：

```python
def make_passkey_training_sample(
    filler_tokens: torch.Tensor,
    tokenizer,
    seq_len: int = 2048,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """生成单条 passkey 训练样本。"""

def build_passkey_eval_sequence(
    filler_tokens: torch.Tensor,
    passkey: str,                  # 如 "7-4-2-9-1"
    tokenizer,
    total_length: int,
    depth_percent: float,
) -> Tuple[torch.Tensor, int, int]:
    """构造评测序列，返回 (input_ids, passkey_start, probe_start)。"""

def eval_passkey_nll_gap(
    model, tokenizer,
    filler_tokens: torch.Tensor,
    lengths: List[int],           # [2048, 4096, 8192]
    depths: List[float],          # [10, 25, 50, 75, 90]
    num_trials: int = 10,
    seed: int = 42,
) -> Dict:
    """返回 results + summary 字典。"""
```

### 集成方式

**训练阶段**：修改 `run_evq_sweep.py` 的训练数据准备，用 `MixedDataset` 包裹原始数据：

```python
# 在 run_single() 中，train_data 替换为:
from eval_passkey_scratch import MixedDataset
mixed_data = MixedDataset(
    lm_data=train_data,
    filler_tokens=val_data[:50000],  # 用 val 前 5w token 做填充
    tokenizer=tok,
    passkey_ratio=0.005,  # 0.5%
    seq_len=cfg["seq_len"],
)
```

**评测阶段**：在 PPL eval 之后加入 passkey eval：

```python
ppl = eval_model(model, val_data, cfg["eval_lengths"], cfg["eval_chunks"])

if not dry_run:
    from eval_passkey_scratch import eval_passkey_nll_gap
    passkey_results = eval_passkey_nll_gap(
        model, tok, val_data,
        lengths=[2048, 4096, 8192],
        depths=[10, 25, 50, 75, 90],
        num_trials=10,
    )
```

---

## 4. 零成本第三 Baseline：Inference-Time PI Scaling

训练只跑 2 组（geometric vs EVQ），但评测时额外加一个 PI baseline，**不需要任何额外训练**。

原理：PI (Position Interpolation) 只是在 eval 时把 inv_freq 除以一个 scale factor。对于已经训练好的 geometric 模型，可以在 eval 阶段用 PI-scaled 的 inv_freq 做推理。

实现：在 `run_single()` 的 eval 阶段之后，**只对 geometric baseline 的模型**额外做一轮 PI eval：

```python
if method == "geometric" and not dry_run:
    # PI eval: 零成本第三 baseline
    pi_scale = max(max(cfg["eval_lengths"]) / cfg["seq_len"], 1.0)
    pi_inv_freq = base_inv_freq / pi_scale
    # 替换模型的 inv_freq
    # ... 跑一轮 eval_model + eval_passkey ...
    # 恢复原始 inv_freq
```

这样论文里就有三条线：geometric (baseline), PI (inference-time baseline), EVQ-Cosh (ours)。PI 是审稿人最熟悉的方法，加这条线几乎不增加成本但大幅增强说服力。

已有的 `rope/schedules.py` 中 `build_inv_freq("pi", ...)` 已实现了 PI，直接 import 使用。

---

## 5. 实验范围

**训练**：2 个 run（geometric + EVQ-Cosh），全部条件控制一致。

**评测**：3 个方法对比。

| Run | Method | 训练 | 评测 | 备注 |
|-----|--------|------|------|------|
| A | geometric baseline | ✅ 从零训练 | PPL + Passkey | τ=0.0 |
| — | PI (inference-time) | ❌ 不训练 | PPL + Passkey | 用 Run A 的模型 + PI inv_freq |
| B | EVQ-Cosh | ✅ 从零训练 | PPL + Passkey | τ=最优值（先用 1.0） |

可选：如果预算允许，加 τ=0.5 和 τ=1.5 做 ablation。

**运行命令**：

```bash
# 最小实验（2 runs 训练，约 16-20 小时，约 100-120 元）
python scripts/m4_evq_sweep/run_evq_sweep.py \
    --tier 500m --taus 0.0,1.0 --seeds 42 \
    --base 500000.0 --dataset fineweb-edu \
    --work_dir ~/evq_500m_sweep
```

---

## 6. 校验机制

### 6.1 inv_freq 校验

在每个 run 开始时：
1. 打印 inv_freq 的 shape, min, max, mean, std
2. 计算并打印 inv_freq 的 SHA256 hash
3. 对 evq_cosh τ=0.0 的 inv_freq，assert 它与 geometric baseline 的 inv_freq 完全一致（元素级 allclose）

### 6.2 模型参数量校验

```python
n_params = sum(p.numel() for p in model.parameters())
assert 450_000_000 <= n_params <= 550_000_000, f"500M tier: got {n_params/1e6:.1f}M params"
```

### 6.3 Passkey 校验

运行前先做 sanity check：
- tokenizer 校验：`<<PASS:7-4-2-9-1>>` 的 token 化结果中，每个数字必须是独立 token（打印 token 列表确认）
- NLL 校验：用 L=2048, depth=50% 检查 NLL gap 是否为有限值（不是 NaN/Inf）
- 混入验证：训练前打印 MixedDataset 的前 1000 个 idx，统计 passkey 样本占比（应在 0.3%–1.5% 范围）

### 6.4 数据校验

- 打印实际加载的 token 数量
- 对每个 run，确保训练数据 hash 一致（所有 run 看到完全相同的 LM 数据）
- 打印 FineWeb-Edu 的前 3 条样本的前 100 个字符，确认数据质量

---

## 7. 16K Eval 的 OOM 处理

**不要假设 16384 长度一定能跑。** 正确的处理方式：

1. eval 阶段强制使用 `batch_size=1` + `torch.no_grad()` + flash attention / SDPA
2. `eval_lengths` 默认设为 `[2048, 4096, 8192]`，把 16384 标注为 optional
3. 添加 `--eval_16k` flag，用户显式启用才尝试 16384
4. 如果 16384 OOM，catch RuntimeError 后 gracefully skip（打印警告，不中断实验）
5. **不要预估 attention 矩阵大小来决定是否跑**——直接 try/except，让实际 OOM 决定

---

## 8. 文件结构

```
scripts/m4_evq_sweep/
├── run_evq_sweep.py          # 修改：新增 500m config, --dataset, MixedDataset 集成, PI eval
├── eval_passkey_scratch.py   # 新建：passkey 训练样本生成 + 评测（NLL gap + 生成）
├── evq_analysis.py           # 已有，无需修改
└── run_500m_minimal.sh       # 新建：最小实验启动脚本
```

### run_500m_minimal.sh 内容：

```bash
#!/bin/bash
# 500M Minimal Experiment: geometric vs EVQ-Cosh (+ PI inference-time baseline)
# 训练 2 runs，评测 3 methods
# 预计耗时: ~16-20 小时，成本 ~100-120 元 (6元/小时)

set -e

WORK_DIR="${HOME}/evq_500m_sweep"
TIER="500m"
SEED=42
BASE=500000.0
DATASET="fineweb-edu"

echo "=========================================="
echo "  500M Minimal Experiment"
echo "  Train: geometric + EVQ-Cosh"
echo "  Eval:  geometric + PI + EVQ-Cosh"
echo "  Work dir: ${WORK_DIR}"
echo "=========================================="

python scripts/m4_evq_sweep/run_evq_sweep.py \
    --tier ${TIER} --taus 0.0,1.0 \
    --seeds ${SEED} --base ${BASE} \
    --dataset ${DATASET} --work_dir ${WORK_DIR} --resume

python scripts/m4_evq_sweep/evq_analysis.py \
    --input ${WORK_DIR}/results_final.json

echo "Done! Results in ${WORK_DIR}/"
```

---

## 9. 注意事项

1. **不要改动现有的 50M/125M/350M 配置**，只添加 500m
2. **保持向后兼容**：`--dataset tinystories` 应该和之前的行为完全一致
3. **HF 镜像**：服务器在中国大陆，保持 `HF_ENDPOINT=https://hf-mirror.com`
4. **Gradient checkpointing**：500M 训练时如果显存紧张，添加 gradient checkpointing 支持
5. **所有 print 语句用中括号前缀**，如 `[500m] Loading data...`，方便日志 grep
6. **FineWeb-Edu 如果下载失败**：自动 fallback 到 `cerebras/SlimPajama-627B`，打印警告
7. **Passkey 混入对所有 run 一致**：geometric 和 EVQ 的训练数据必须包含相同的 passkey 样本（用相同的 seed 控制）

---

## 10. 完成后的自检清单

写完代码后，请逐项确认：

- [ ] `python run_evq_sweep.py --tier 500m --dry_run` 跑通，参数量在 450-550M
- [ ] evq_cosh τ=0.0 的 inv_freq 与 geometric 的 inv_freq 完全一致（allclose）
- [ ] passkey tokenizer 校验：`<<PASS:7-4-2-9-1>>` 中每个数字是独立 token
- [ ] MixedDataset 的 passkey 混入比例在 0.3%-1.5%（打印统计确认）
- [ ] `--dataset fineweb-edu` streaming 加载正常（打印前 3 条样本摘要）
- [ ] `--dataset tinystories` 向后兼容
- [ ] PI inference-time eval 逻辑：在 geometric 模型 eval 后自动运行
- [ ] 8192 eval 正常，16384 OOM 时 graceful skip
- [ ] `run_500m_minimal.sh --dry_run` 全流程跑通
- [ ] 所有新代码有 docstring 和类型注解
- [ ] FineWeb-Edu 下载失败时自动 fallback 到 SlimPajama

---
