# AI Environment Snapshot — 2026-02-22

> **用途**：新 AI 读完这一个文件就能立刻接手项目。
> 最后更新：2026-02-22 09:15（实验正在服务器上运行中）

---

## 1. 项目概述

研究 RoPE 频谱形状对长上下文外推的影响。核心贡献：提出 anchored_hybrid 方法（高频锚定 + 低频平滑混合），在 50M→350M 从零训练中一致改善 5-14%，目标投 NeurIPS。

详细文档见 `knowledge_base/` 目录（有 `README.md` 索引）。

---

## 2. 本地环境

- **本地 repo**：`e:\rope\hybrid-rope`（Windows）或 Mac 上 clone 后的对应路径
- **GitHub**：`https://github.com/misaya-yang/hybrid-rope` (main 分支)
- 本地无 GPU，所有训练/评测在远程服务器运行

---

## 3. 远程服务器

### 连接方式

```bash
# SSH（Windows plink）
plink -P 42581 root@connect.bjb1.seetacloud.com -pw htG0sD63/yG0

# Mac/Linux
ssh -p 42581 root@connect.bjb1.seetacloud.com
# 密码: htG0sD63/yG0

# 文件传输（Windows pscp / Mac scp）
pscp -P 42581 -pw htG0sD63/yG0 local_file root@connect.bjb1.seetacloud.com:/remote/path
scp -P 42581 local_file root@connect.bjb1.seetacloud.com:/remote/path
```

### 硬件与软件

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| VRAM | 97,887 MiB (~96 GB) |
| Python | 3.12.3 (`/root/miniconda3/bin/python`) |
| PyTorch | 2.8.0+cu128 |
| **transformers** | **5.1.0**（⚠️ 见第 7 节兼容性注意） |
| peft | 0.18.1 |
| datasets | 3.0.0 |
| OS | Linux 5.15.0 (Ubuntu) |

### 关键路径

| 内容 | 路径 |
|------|------|
| 代码仓库 | `/root/autodl-tmp/dfrope/hybrid-rope` |
| 模型镜像 | `/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct` |
| 数据集镜像 | `/root/autodl-tmp/dfrope/ms_datasets` |
| GPT-NeoX tokenizer | `/root/autodl-tmp/dfrope/ms_models/EleutherAI/gpt-neox-20b` |
| 从零训练 checkpoints | `/root/autodl-tmp/dfrope/hybrid-rope/sigmoid_rope_experiments/checkpoints` |
| 8B 实验结果（旧） | `/root/autodl-tmp/dfrope/hybrid-rope/results/llama8b_fair_lora_suite_20260214` |
| **当前实验结果** | `/root/autodl-tmp/dfrope/hybrid-rope/results/overnight_8h/` |

### 网络约束

- ⚠️ HuggingFace 外网可能受限
- 环境变量已设置：`HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`
- **必须使用本地模型和数据集镜像**

---

## 4. 当前正在运行的实验 🔄

**PID 44430**，启动于 2026-02-22 09:10

### 配置

| 参数 | 值 |
|------|-----|
| 脚本 | `archives/2026-02-22/scripts/run_overnight_8h.py` |
| 方法 | baseline, PI, YaRN, anchored_hybrid |
| 训练步数 | 300 |
| 上下文长度 | 16384 |
| LoRA rank/alpha | 64/128 |
| batch × grad_accum | 2 × 2 = 4 |
| lr | 2e-4 |
| attention | sdpa |
| 基础模型 | LLaMA-3-8B-Instruct |
| NIAH 评测长度 | [4K, 8K, 16K, 32K] |

### 流水线

1. **Gate 0** Calibration（✅ 4/4 通过）
2. **Gate 1** 训练 4 方法（🔄 baseline 进行中，loss=2.69 @ step 10）
3. **Gate 2** NIAH 评测（等待）
4. **Gate 3** 生成对比表和热力图（等待）

### 监控

```bash
# 实验进度
tail -f /root/autodl-tmp/dfrope/hybrid-rope/results/overnight_8h/experiment.log

# 训练 loss
cat results/overnight_8h/train_baseline/logs/train.log

# GPU 状态
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv

# 检查进程
ps aux | grep run_llama
```

### 预计完成时间

- 每步 ~48 秒，300 步/方法 ~4 小时
- 4 方法训练 ~16 小时 + NIAH ~1.5 小时
- **预计约 2026-02-23 凌晨完成**

---

## 5. 关键脚本

| 脚本 | 用途 | 位置 |
|------|------|------|
| `run_llama8b_fair_suite.py` | 单方法 8B LoRA 训练（公平协议） | `archives/2026-02-22/scripts/` |
| `run_overnight_8h.py` | 4 方法自动化流水线 | `archives/2026-02-22/scripts/` |
| `_launch.sh` | **正确的实验启动脚本** | `archives/2026-02-22/scripts/` |
| `run_passkey_sanity_check.py` | Teacher-forcing passkey 评测 | `scripts/` |

### RoPE 频率注入机制

**核心原则**（绝对不能违反）：
1. ❌ 不修改 `model.config.rope_scaling`
2. ❌ 不 monkey patch `forward()` 函数
3. ✅ 只通过 `inv_freq.copy_(custom_freq)` buffer 原地覆写
4. ✅ 覆写后必须清理 cos/sin 缓存

代码位置：`run_llama8b_fair_suite.py` → `overwrite_inv_freq_inplace()`

---

## 6. 已完成的实验证据

| 实验 | 核心结论 | 数据位置 |
|------|---------|---------|
| 50M 3cfg×3seed | Hybrid PPL@16K 改善 ~10% | `results/evidence_chain_50m_3cfg3seed/` |
| 100M | 改善 13.5% | 分散在 results 中 |
| 350M 10seed | 改善 13.7% | 分散在 results 中 |
| Phase4 124M Sigmoid | **16K PPL -66%, 32K PPL -64%** | `sigmoid_rope_experiments/data/` |
| Phase Collision 机理 | 崩溃比 22x → 1.08x | `results/llama_shape_theta_min/` |
| Sigmoid 公式拟合 | R²>0.99 | `sigmoid_rope_experiments/data/fitting_results.json` |
| Qwen 交叉验证 | 即插即用频谱替换效果有限 | `results/qwen_*/` |
| **8B 旧实验** | ⚠️ 不公平协议，已废弃 | `results/llama8b_fair_lora_suite_20260214/` |
| **8B 新实验** | 🔄 正在运行 | `results/overnight_8h/` |

---

## 7. ⚠️ 必读：已踩的坑

### 坑 1：transformers 5.1.0 Config 变化

```python
# transformers 4.x:  config.rope_theta = 500000.0
# transformers 5.1.0: config.rope_theta → AttributeError!
#   config.rope_scaling = {'rope_theta': 500000.0, 'rope_type': 'default', ...}
```

**已修复**：`infer_model_rope_base()` 和 `rope_scaling_is_effectively_default()` 均已兼容。

### 坑 2：Kill 实验时必须杀子进程

```bash
# ❌ 错误：只杀父进程，子训练进程变僵尸占满 GPU
kill <overnight_pid>

# ✅ 正确：同时杀所有相关进程
pkill -f run_overnight; pkill -f run_llama; sleep 3
```

### 坑 3：禁止用 Python Popen 启动实验

Python `import torch` 会分配 CUDA context (~81 GB)，导致子训练进程 OOM。
**必须用纯 shell `nohup` 或 `bash _launch.sh`。**

### 坑 4：训练速度参考

16K context + 8B model + LoRA + gradient_checkpointing ≈ **48 秒/步**。
用此数据计算合理的 `TRAIN_STEPS` 和 `timeout`。

详细踩坑日志：`archives/2026-02-22/SESSION_LOG.md`

---

## 8. knowledge_base 文档索引

| 编号 | 文件 | 内容 |
|------|------|------|
| 00 | `项目与结论总览.md` | 证据分层、可信结论 |
| 01 | `已完成实验核心数据.md` | 全部数字和表格 |
| 02 | `论文故事线与主张.md` | 论文叙事、贡献点、图表清单 |
| 03 | `负结果与风险复盘.md` | 失败模式、修复策略 |
| 05 | `执行计划.md` | 进度追踪 |
| 08 | `8b_experiment_analysis.md` | 8B 问题诊断（已修复）+ 新设计 |
| ALL | `ALL_IN_ONE.md` | NeurIPS 全景评估 |

---

## 9. 接手后立即要做的事

1. **检查实验是否完成**：`tail results/overnight_8h/experiment.log`
2. **如果完成**：下载 `results/overnight_8h/summary/` 的 CSV 和热力图
3. **如果失败**：读 `results/overnight_8h/console.log` 找错误，用 `bash archives/2026-02-22/scripts/_launch.sh` 重启
4. **分析结果**：对比 4 方法的 loss 曲线和 NIAH recall
5. **开始写论文**：参考 `knowledge_base/02_论文故事线与主张.md` 的图表清单
