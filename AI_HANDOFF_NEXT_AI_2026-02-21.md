# Hybrid-RoPE / Sigmoid-RoPE 单文件交接包（给下一位 AI）

最后更新：2026-02-21（本轮 `run_phase4_corrected.py` 已跑完）

## 1. 当前任务与交接目的

用户认为当前结果中“PPL 很好但任务能力异常（尤其 Passkey 全 0）”，怀疑实验设计存在系统性问题。  
本文件用于让下一位 AI 在不依赖长上下文历史对话的情况下，直接接管后续实验。

核心目标从“继续盲跑”切换为：
1. 审计实验设计是否存在数据/评测偏差。
2. 重新建立可信证据链（LM 指标 + 检索/理解指标一致）。
3. 为 8B 级实验做最小可行、可复现实验闭环。

---

## 2. 运行环境（关键约束）

### 2.1 本地
- OS: Windows (PowerShell)
- 工作目录: `e:\rope\hybrid-rope`
- 本地显卡: RTX 4070 Super（适合纯计算实验/小模型短跑）

### 2.2 远端（本轮主实验）
- 连接方式：`plink/pscp`
- 远端工作目录：`/root/autodl-tmp/dfrope/hybrid-rope`
- 本轮实验目录：`/root/autodl-tmp/dfrope/hybrid-rope/sigmoid_rope_experiments`
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (96GB)

### 2.3 网络/数据约束
- 用户环境对 HuggingFace 外网访问受限。
- 实际训练数据在本轮由本地数据路径驱动（见 `run_phase4.py`）：
  - 优先读取 `/root/autodl-tmp/dfrope/ms_datasets/LongBench/data/*_e.jsonl`
  - 否则回退 synthetic 数据。

---

## 3. 本轮关键代码改动（已经存在）

### 3.1 新增/修改
- `sigmoid_rope_experiments/run_phase4_corrected.py`
  - 四模型串行训练：`standard` / `sigmoid` / `anchored20` / `anchored_alpha`
  - 统一 seed/初始化/数据顺序/超参
  - completion 模式 passkey 评测（含 debug 打印）
  - 统一导出训练曲线、PPL-vs-length、positional loss、passkey 图和 CSV
- `sigmoid_rope_experiments/src/rope.py`
  - 增加 `anchored_sigmoid(...)` 频率构造方法

### 3.2 相关核心脚本
- `sigmoid_rope_experiments/run_phase4.py`
  - 基础模型/数据加载逻辑（`load_training_tokens` 读取 LongBench-local 或 synthetic）
- `sigmoid_rope_experiments/run_passkey_debug.py`
  - RoPE 注入与 passkey 诊断脚本

---

## 4. 本轮已完成实验（Phase4 corrected）

已完成脚本：
- `python -u run_phase4_corrected.py --include_alpha_star`

### 4.1 训练最佳结果（来自最终日志）
- 模型规模：123.6M，head_dim=64，3000 steps
- 数据标识：`LongBench-local`

| 模型 | best_step | best_val_loss | best_val_ppl |
|---|---:|---:|---:|
| Standard | 2950 | 3.3088 | 27.352 |
| Sigmoid | 2900 | 3.1238 | 22.734 |
| Anchored-20 | 2900 | 4.0023 | 54.722 |
| Anchored-alpha* | 2900 | 3.5789 | 35.835 |

结论（仅就本轮 LM 验证损失）：`Sigmoid > Standard > Anchored-alpha* > Anchored-20`。

### 4.2 长度外推 PPL（日志汇总）

| 模型 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|---:|
| Anchored-20 | 71.32 | 51.76 | 55.02 | 44.14 | 43.93 | 124.57 |
| Anchored-alpha* | 36.92 | 41.87 | 56.95 | 26.16 | 25.56 | 62.43 |
| Sigmoid | 17.75 | 20.91 | 20.30 | 24.70 | 19.03 | 147.50 |
| Standard | 35.38 | 24.23 | 75.25 | 40.95 | 56.07 | 412.75 |

注意：该表噪声偏大（见第 6 节问题分析）。

### 4.3 Passkey（修复版）
四模型在 `L={1024,2048,4096,8192,16384}` 均为 `0.0%`。

日志中可见模型生成偏模板化文本，未输出目标数字，虽然评测协议已切到 completion 格式，但任务依然失败。

---

## 5. 产出文件位置（远端）

### 5.1 核心汇总
- `sigmoid_rope_experiments/data/phase4_corrected_summary.json`
- `sigmoid_rope_experiments/data/ppl_vs_length.csv`
- `sigmoid_rope_experiments/data/passkey_fixed_results.csv`
- `sigmoid_rope_experiments/data/positional_loss.csv`

### 5.2 图表
- `sigmoid_rope_experiments/results/training_curves_all.pdf`
- `sigmoid_rope_experiments/results/ppl_vs_length.pdf`
- `sigmoid_rope_experiments/results/passkey_fixed.pdf`
- `sigmoid_rope_experiments/results/positional_loss.pdf`
- `sigmoid_rope_experiments/results/freq_comparison_trained.pdf`

### 5.3 主日志
- `sigmoid_rope_experiments/run_phase4_corrected_live.log`

---

## 6. 当前最可能的实验设计问题（重点）

这部分是下一位 AI 必须优先排查的点。

1. 指标不一致：LM loss 很好，但 Passkey 全 0  
可能不是单纯 RoPE 形状问题，而是训练目标/数据分布与检索任务不对齐。

2. 训练/验证集合构造可能过于“同分布且相邻切片”  
`split_train_val` 是连续切分，同语料域内泄漏风险高，可能导致 val loss 过于乐观。

3. PPL-vs-length 估计方差偏高  
采样量有限（`base_samples` 机制），且长长度自动降采样，导致曲线波动较大。

4. Passkey prompt 与训练语料风格错配  
模型是从头训练小模型，不是 instruction 模型；即便 completion 协议修复，仍可能缺乏“复制数字”能力。

5. Anchored 参数并未在该训练设定下重新调优  
`alpha=20` 与 `alpha*` 直接套用，可能不适配 head_dim=64 / 数据分布。

---

## 7. 下一位 AI 的最小执行清单（建议顺序）

### Step A: 先做“实验可信性审计”（不训练或少训练）
1. 固定模型，重复跑 `ppl_vs_length` 多次并给置信区间。
2. 增加 out-of-domain 验证集（和训练语料解耦）再比较四模型 val loss。
3. 跑 `run_passkey_debug.py` 的 re-inject 对照：`Original` vs `Standard(re-inject)` 一致性检查。

### Step B: 重新定义 passkey 成功标准
1. 保留数字正则提取。
2. 增加 top-k 候选解析，避免仅字符串包含导致误判。
3. 保留 raw generation 样本文件做人工 spot check。

### Step C: 再决定是否继续大卡训练
仅当 A/B 证明评测链条可信，再做 8B Anchored 对照（与 YaRN/PI 同协议）。

---

## 8. 可直接复现命令（远端）

```bash
cd /root/autodl-tmp/dfrope/hybrid-rope/sigmoid_rope_experiments

# 本轮主实验（已跑完）
/root/miniconda3/bin/python -u run_phase4_corrected.py --include_alpha_star

# 建议先跑注入一致性/Passkey审计
/root/miniconda3/bin/python -u run_passkey_debug.py --help
```

---

## 9. 仓库当前本地未提交项（交接时请注意）

当前 `git status --short` 显示新增/未追踪（节选）：
- `scripts/run_theoretical_validation.py`
- `frequency_design_insight.py`
- `frequency_range_analysis.py`
- `optimal_base_search.py`
- `knowledge_base/07_frequency_design_theory.md`
- `knowledge_base/08_8b_experiment_analysis.md`
- `results/theoretical_validation/`
- `results/frequency_range_analysis/`
- `results/optimal_base_search/`
- `tmp_phase4_compare/`

建议下一位 AI 接管时先做一次：
1. 结果归档（保留可复现证据）
2. 无关临时目录清理（如 `tmp_phase4_compare`）
3. 再提交分层 commit（代码/数据索引/文档分开）

---

## 10. 给下一位 AI 的一句话

不要再以“单次 PPL 结果好”作为成功标准。  
先把评测协议和数据拆分做成可审计的闭环，再做 8B 主实验；否则会持续出现“PPL 好看但能力崩”的伪进展。

