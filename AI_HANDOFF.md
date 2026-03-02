# AI Handoff 总纲 (Master Guide)

> **最后更新**: 2026-03-02
> **论文**: NeurIPS 2026 — "RoPE Frequency Allocation as a Variational Inverse Problem"
> **当前版本**: V5 (EVQ / τ 单参数 / Waterbed rewrite)
> **本地路径**: 以本仓库根目录为准（用户可能在不同机器上工作，路径会变化）
> **🔴 核心理论参考**: `docs/paperdraft/CORE_THEORY.md`（v3，包含所有最新理论、实验数据、红线）

---

## ⚠️ AI 工作守则 (MANDATORY — 每个 AI 必读)

**你在完成任何工作后，必须执行以下两件事：**

1. **整理文件**: 临时文件归入 `tmp/` 或删除；产出物按下方约定放入正确文件夹；不得在根目录留下散落文件。
2. **更新文档**: 在 `docs/exp/` 下为你的工作写一份简短报告 (`YYYY-MM-DD_<topic>.md`)，并更新本文件的"当前状态"部分。

**如果你不遵守，后续 AI 将花费大量时间理解你留下的混乱——这是对用户时间和金钱的浪费。**

---

## 1. 项目一句话总结

本项目证明 RoPE 的频率分配是一个变分逆问题，其精确解 **EVQ (Exact Variational Quantization)** 由单参数 **τ = √(β/α)** 完全控制。τ 调节高频/低频资源的重分配比例，且满足 **Waterbed Inequality**：改善长上下文必然退化短上下文。

---

## 2. 理论框架速览

| 概念 | 公式/描述 | 位置 |
|------|-----------|------|
| **EVQ warp** | φ_k(τ) = 1 - (1/τ) arcsinh((1-u_k) sinh(τ)) | `rope/schedules.py` → `evq_cosh` |
| **τ 物理含义** | τ = √(β/α), interference coupling ratio | 论文 Section 3 |
| **Theorem 1** | Joint ODE exact solution (cosh tether + Fisher pulse) | 论文 Appendix A |
| **Theorem 2** | EVQ → geometric RoPE as τ→0 (asymptotic degradation) | 论文 Appendix A |
| **Waterbed** | ∫₀¹ ln E(φ)dφ ≥ ln b - ln c | 论文 Section 3.5 |
| **Phase Collision** | cos(θΔ) 在低频碰撞导致注意力局部性崩溃 | `knowledge_base/06_phase_collision_D_analysis.md` |

---

## 3. 论文当前状态

**→ 详见 `docs/PAPER_DRAFT_STATUS.md`**

简要：
- **V5 已完成**: 全文 EVQ/τ/Waterbed 改写，8/9 页（剩 1 页用于实验结果）
- **Figure 1**: EVQ warp curves (已插入)
- **缺失**: τ-sweep 实验数据表、longinst 8B 实验结果
- **截止**: NeurIPS 2026 DDL (约 9 周后)

---

## 4. 代码核心入口

| 文件 | 用途 | 备注 |
|------|------|------|
| `rope/schedules.py` | 所有频率 schedule 的构建（含 EVQ） | **核心实现** |
| `rope/inject.py` | inv_freq.copy_() 注入协议 | 2 行代码 |
| `scripts/m4_evq_sweep/run_evq_sweep.py` | τ-sweep 从零训练 + 评估 | 50M/125M/350M |
| `scripts/m4_evq_sweep/evq_analysis.py` | 分析出图（NeurIPS 级 PDF） | 4 图 + CSV |
| `scripts/isolated/longinst/` | 8B LoRA long-instruction 训练 | **关键实验** |
| `scripts/eval_longbench.py` | LongBench-21 评估 | 88KB, 全功能 |
| `train.py` | 主训练脚本 | 根目录 |

---

## 5. 实验全景 (Experiment Landscape)

### 5.1 论文中引用的实验 (V5)

| 实验 | 论文位置 | 数据来源 | 质量 | 备注 |
|------|----------|----------|------|------|
| 50M 3-seed scaling | Table 1 | `results/evidence_chain_50m_3cfg3seed/` | ⚠️ 旧框架 | 使用 anchored_sigmoid，非 EVQ |
| 100M scaling | Table 1 | `artifacts/a100_2026-02-13/data/100m_scaling/` | ⚠️ 旧框架 | 同上 |
| 350M scaling | Table 1 | `artifacts/a100_2026-02-13/data/350m_final/` | ⚠️ 旧框架 | 同上 |
| 50M YaRN compare | Table 2 | `results/50m_yarn_compare_v2/` | ⚠️ 旧框架 | 同上 |
| Phase collision | Fig/Text | `results/phase_collision_comparison_v2/` | ✅ 理论有效 | 通用分析 |
| EVQ warp curves | Figure 1 | `plot_evq_warp_v2.py` 生成 | ✅ 理论图 | 纯数学 |

> **关键问题**: 当前论文中所有从零训练实验都使用旧的 anchored_sigmoid 方法，而论文 V5 已改写为 EVQ/τ 框架。**急需用 EVQ τ-sweep 替换这些实验数据**。

### 5.2 进行中的实验（2026-03-02 更新）

| 实验 | 状态 | 位置 | 预期用途 |
|------|------|------|----------|
| **Phase 9A Group 2** (1.7B, L=2048, τ=1.5) | 🔄 R6000 下载数据中 | `scripts/phase9a_1b_2k.py` | 论文核心实验 |
| **Phase 9A Group 1** (1.7B, L=4096, τ=1.0) | ⏳ Group 2 后 | `scripts/phase9a_1b_4k.py` | Learnability 验证 |
| **5090 快速验证** (350M, L=2048, τ=1.5) | ✅ 单seed完成，等多seed | Codex 管理 | 快速信号确认 |

**5090 单seed结果（2026-03-02，FineWeb-Edu, 350M, seed=42）**：
- PPL@16K: -15.4% ✅ (远超10%噪音门槛)
- PPL@8K: -10.2% ✅
- PPL@2K: +1.9% (轻微短程退化，理论可解释——E_learn 主导)
- Retrieval: ~55%两边都低（模型太小）
- **结论**：信号明确，等多seed确认后 R6000 可放心跑

### 5.3 已完成关键实验

| 实验 | 结论 | 位置 |
|------|------|------|
| **Phase 8D** (τ* scaling law, 5 点) | τ*=d_head/√L 验证，L≥1024 吻合 | CORE_THEORY §7.2 |
| **Phase 8F** (350M, base=500K, 4-seed) | Hybrid 方差最低（8×），均值持平 | CORE_THEORY §8.1 |
| **Phase 8H** (350M, base=10K, τ sweep) | 全败，碰撞块死区确认 | CORE_THEORY §7.7.5 |
| **50M/125M EVQ τ-sweep** | τ=1.5 赢 Geo 10-19% (L=2048) | `results/paper_ready/evq_tau_sweep/` |
| **5090 350M 验证** (FineWeb-Edu) | PPL@16K -15.4%, PPL@2K +1.9% | CORE_THEORY §7.8 |

### 5.4 历史实验（存档参考）

大量旧实验保存在 `results/`, `artifacts/`, `archives/`, `sigmoid_rope_experiments/` 中。这些实验使用旧方法名（anchored_sigmoid, hybrid 等），**不可直接用于 V5 论文**，但可作为方法演化的参考。

---

## 6. 文件夹结构约定

```
hybrid-rope/
├── AI_HANDOFF.md            ← 你正在读的这个文件 (总纲)
├── README.md                ← 项目简介
├── train.py                 ← 主训练入口
│
├── rope/                    ← 核心库 (schedules, inject)
├── eval/                    ← 评估模块
│
├── docs/                    ← 📄 所有文档
│   ├── EXPERIMENT_REGISTRY.md  ← 实验权威注册表
│   ├── PAPER_DRAFT_STATUS.md   ← 论文进度追踪
│   ├── protocols/              ← 锁定的实验协议
│   ├── exp/                    ← 实验报告 (按日期)
│   ├── notes/                  ← 研究笔记
│   ├── env/                    ← 环境快照
│   └── review/                 ← 审稿意见
│
├── scripts/                 ← 🔧 所有可执行脚本
│   ├── m4_evq_sweep/          ← τ-sweep 实验
│   ├── isolated/              ← 独立实验管线
│   │   ├── longinst/          ← 8B long-instruction
│   │   └── attn/              ← 注意力实验
│   └── ...                    ← 其他工具脚本
│
├── knowledge_base/          ← 📚 研究知识库 (中文为主)
│   ├── 00-11 编号文档          ← 系统知识
│   └── ALL_IN_ONE.md          ← 合并版
│
├── results/                 ← 📊 整理后的实验结果
│   ├── paper_ready/           ← 论文级结果
│   └── ...                    ← 其他结果包
│
├── artifacts/               ← 🗄️ 服务器快照
│   ├── a100_2026-02-13/       ← A100 集群
│   ├── a800_2026-02-13/       ← A800 集群
│   └── reviewer_2026-02-*/    ← 审稿相关
│
├── paper_exports/           ← 📝 论文编译输出
│   ├── neurips_v5/            ← 当前版本
│   └── neurips_v5_fig/        ← 带图版本
│
├── handoff/                 ← 🔄 历史交接包 (按日期)
│   ├── 2026-02-23/
│   ├── 2026-02-25/
│   └── 2026-02-26/
│
├── archives/                ← 📦 历史存档
├── sigmoid_rope_experiments/← 旧 sigmoid 实验管线
├── tools/                   ← 工具脚本
├── data/                    ← 原始数据
├── outputs/                 ← 分析输出
└── experiments/             ← 探索性实验
```

---

## 7. 新 AI 快速上手路径

**如果你是第一次接手这个项目，按以下顺序阅读：**

1. **本文件** (`AI_HANDOFF.md`) — 你正在读
2. **`docs/PAPER_DRAFT_STATUS.md`** — 论文当前状态和 TODO
3. **`knowledge_base/02_论文故事线与主张.md`** — 论文叙事逻辑
4. **`knowledge_base/09_unified_theory_crlb.md`** — EVQ 理论推导
5. **`knowledge_base/11_waterbed_strict_proof.md`** — Waterbed 证明
6. **`docs/EXPERIMENT_REGISTRY.md`** — 实验权威表
7. **`docs/protocols/LLAMA3_8B_LORA_STANDARD_2026-02-26.md`** — 8B 实验协议
8. **`rope/schedules.py`** — 代码实现

---

## 8. 关键术语映射 (新旧对照)

论文已从 V4 (anchored-sigmoid) 全面改写为 V5 (EVQ)。以下是术语对照：

| V4 旧名 | V5 新名 | 备注 |
|---------|---------|------|
| anchored_sigmoid | EVQ (τ parameterized) | 论文主方法 |
| hybrid / shape_only | EVQ 的特例 | 代码中 alias 保留 |
| anchor_factor, slope, center_ratio | τ (single parameter) | 论文只用 τ |
| sigmoid schedule | EVQ-cosh warp | 理论化表述 |
| Phase 4 | Section 5 experiments | 论文结构改 |

**代码中 `anchored_sigmoid` 和 `evq_cosh` 都保留**，但论文中统一使用 EVQ 术语。

---

## 9. 当前紧急任务优先级（2026-03-02 更新）

1. **🔴 等 5090 多 seed 结果** — 单 seed 已出：PPL@16K -15.4% ✅，等多 seed 确认稳定性
2. **🔴 R6000 Phase 9 准备中** — 1.7B 模型, base=500K, 2B tokens, 多 seed 确认后开始训练
   - 脚本: `scripts/phase9a_1b_2k.py`（Group 2: L=2048, τ=1.5）, `scripts/phase9a_1b_4k.py`（Group 1: L=4096, τ=1.0）
3. **🟡 5090 多 seed 验证** → 单 seed 赢了后补 2-3 个 seed，论文投稿必需
4. **🟡 论文实验部分重写** → 用 Phase 9 数据替换旧数据
5. **🟢 Multi-needle 评测** → `scripts/eval_multi_needle.py` 已写好，Phase 9 4K 组包含

---

## 10. 🔴 关键发现与理论进展（2026-03-02，必读）

> **详细推导和公式见 `docs/paperdraft/CORE_THEORY.md`**

### 10.1 τ* Scaling Law（已验证）

$$\tau^* = \frac{d_{head}}{\sqrt{L_{train}}}$$

- L=2048: τ*=1.41（实测最优 τ=1.5，偏差 6%）— 50M/125M/350M 三个规模全赢 Geo 10-19%
- L=4096: τ*=1.0 — Phase 8D 5 点验证
- **所有早期赢的实验 L_train 都是 2048**，Phase 8F 用 L=4096 + τ=1.0 没赢但方差降低
- 双变量 τ*(L,b) 含 base 修正的公式已被否定，**论文只用单变量形式**

### 10.2 Base=10K 是"死区"（负面结果，理论确认）

350M, 50M tokens, base=10K 下 5 组实验全败（EVQ τ=0.7/1.1/1.2, Hybrid r=22）。原因：

**碰撞块（Collision Block）分析**：EVQ 净增益 ΔJ ∝ (1-c)/lnb，其中 c=lnL/lnb。
- Base=10K: c=0.903, 碰撞块仅 3/32 通道 → 优化空间极小
- Base=500K: c=0.634, 碰撞块 12/32 通道 → 充足空间
- **"低 base 放大 EVQ 增益" 叙事被推翻**，实际 base=500K 净增益是 10K 的 2.7 倍

### 10.3 Learnability 假说（待 Phase 9 验证）

E_total = E_alloc(τ) + E_learn(τ, N_params)。EVQ 信息论最优但学习难度更高，小模型可能学不到。Phase 9 的 1.7B 模型是验证这个假说的关键。

### 10.4 Phase 9 实验设计

| Group | L_train | τ | 模型 | Tokens | 方法 |
|-------|---------|---|------|--------|------|
| 2 (先跑) | 2048 | 1.5 | 1.7B | 2B | Geo / EVQ / Hybrid r=16 |
| 1 (后跑) | 4096 | 1.0 | 1.7B | 2B | Geo / EVQ / Hybrid r=16 |

- 1.7B 配置: hidden=2048, layers=24, heads=32, head_dim=64, intermediate=8192
- R6000 (96GB GDDR7): micro_bs=8, grad_accum=2, effective_bs=16
- **head_dim=64 不变 → τ* 和 r*=16 公式沿用**
- 评测: PPL@2K/4K/8K/16K + 单针 passkey + 多针 passkey（5 needles）

### 10.5 Anchored Sigmoid 历史战绩（重要参考）

Anchored sigmoid（EVQ 的前身，cruder approximation）在 from-scratch 预训练中：
- 50M: PPL@16K 赢 Geo 7-11%
- 125M: 赢 6-19%
- 350M (L=2048): 赢 13.7%
- 8B LoRA 微调: LongBench +14.5%, Passkey@16K 100% vs 80%
- **全部 L_train=2048**

### 10.6 Hybrid 理论

r* = (d_head/(2lnb)) · ln(L/2π)。Base=500K 下 r*≈16（恰好是 50/50 split）。
Waterbed 隔离：锁定高频 Geometric（保护 PPL），仅低频做 EVQ。
方差降低 8×（passkey std），论文叙事的核心卖点之一。

---

## 11. 服务器与环境

| 环境 | 用途 | 备注 |
|------|------|------|
| 本地 M4 Max 36GB | 小模型验证、论文编写 | conda `aidemo` |
| 5090 32GB (租用) | 350M 快速验证（Codex 管理） | 便宜，试错用 |
| RTX PRO 6000 96GB (租用) | 1.7B Phase 9 实验（Claude Code 管理） | Blackwell 架构, GDDR7, 1.8TB/s |
| A100/A800 (按需) | 备选大模型实验 | 价格更高 |

### ⚠️ 服务器环境初始化 (MANDATORY — 每次 SSH 登录后必须执行)

**每个 AI 在服务器上执行任何 Python 命令之前，必须先运行以下环境变量设置：**

```bash
export PATH="/root/miniconda3/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
```

说明：
- `PATH`: 服务器的 Python/conda 不在默认 PATH 中，不设置会找不到 python/pip
- `HF_ENDPOINT`: 中国大陆服务器无法直连 HuggingFace，必须使用镜像站
- `PYTHONUNBUFFERED`: 确保训练日志实时输出，不被缓冲

**建议写入脚本开头或 `~/.bashrc`**，避免每次手动设置。

---

## 12. ⚠️ 已知陷阱

1. **服务器环境未初始化**: AI 最常犯的错误！登录服务器后必须先 `export PATH="/root/miniconda3/bin:$PATH"` 和 `export HF_ENDPOINT=https://hf-mirror.com`，否则会报 python 找不到或 HF 连接超时。详见上方"服务器环境初始化"。
2. **MPS OOM**: M4 Max 跑 float32 attention 在 L≥16384 会 OOM。`run_evq_sweep.py` 已添加保护。
3. **旧实验数据不可直接引用**: results/ 中大量数据使用旧方法名，需确认是否与 EVQ 框架一致。
4. **anchored_sigmoid ≠ evq_cosh**: 代码中这是两个不同的 schedule！anchored_sigmoid 用 sigmoid 函数，evq_cosh 用 arcsinh。论文 V5 理论基于 evq_cosh。
5. **Zero-shot 频率替换无效**: 必须配合训练（至少微调）才能看到效果。
6. **已花费 >1000 RMB 在无效实验上**: 务必先在小模型验证再上大机器。

---

## 13. 文档更新日志

| 日期 | 更新内容 | 操作者 |
|------|----------|--------|
| 2026-03-02 | 新增 §10 关键发现：τ* scaling law, 碰撞块分析, learnability 假说, Phase 9 设计, anchored sigmoid 历史战绩; 更新 §9 任务优先级; 添加 CORE_THEORY.md 引用 | Claude (Cowork) |
| 2026-02-27 | 全面重写为 V5/EVQ 框架；新增总纲结构 | Claude (Cowork) |
| 2026-02-26 | 旧版（anchored-sigmoid 框架，已过时） | Claude Code |
