# 复现实验（中文）

本仓库记录了我们在 TinyStories 上做的 RoPE 频率分布实验（从 DF-RoPE 方向 pivot 到“频谱设计”，保持 Translation Invariance：相位仍是 `ω_k * d` 形式，只改变 `ω_k` 的分布）。

## 1. 环境要求

已验证可跑环境（参考）：

- GPU：A100/A800 80GB（单卡）
- CUDA：13.0
- PyTorch：2.9.1+cu130
- Python：3.12.x
- dtype：BF16（脚本内固定）

依赖（conda/env 任意，只要能 import）：

- `torch`
- `numpy`
- `datasets`
- `transformers`

## 2. 数据与 Tokenizer（关键：必须一致）

- 数据集：`roneneldan/TinyStories`
  - 训练：split=`train`，streaming 模式，按 tokenizer 编码后截取前 N 个 tokens
  - 验证：split=`validation`，streaming 模式，按 tokenizer 编码后截取最多 5,000,000 tokens
- Tokenizer：`EleutherAI/gpt-neox-20b`
- 编码方式：`tokenizer.encode(text, add_special_tokens=False)`

注意：**即使代码相同**，streaming + 网络缓存也会带来一定噪声；我们通过“多 seed 统计”来消化这类不确定性。

## 3. 脚本与输出

### 3.1 `unified_search.py`（小规模候选扫描）

用途：在 50M 模型上，快速扫描一组候选频谱（geometric/sigmoid/anchored poly/hybrid）。

脚本：
- `a100/unified_search.py`

输出（远端机器上路径示例）：
- `/opt/dfrope/results/unified_search/results_A.json`
- `/opt/dfrope/results/unified_search/log_A.txt`

看进度：
- 先会出现 “Loading TinyStories ...” 的 tokenization 阶段（GPU 可能是 0%）
- 进入训练后 `nvidia-smi` 会出现显存占用和进程

### 3.2 `unified_search_3cfg_3seed.py`（3 配置 × 3 seed 稳健性验证）

用途：在 50M 模型上验证核心结论是否对 seed 稳健。

脚本：
- `a100/unified_search_3cfg_3seed.py`

配置：
1. `geo_500k`
2. `hybrid_a0.2_t100k`
3. `anchpoly_p3.9_omf0.3_t500k`

Seeds：`[42, 123, 7]`

输出：
- `/opt/dfrope/results/unified_search_3cfg_3seed/results.json`
- `/opt/dfrope/results/unified_search_3cfg_3seed/log.txt`

结果文件会包含：
- 每一轮（config, seed）的 PPL@2048 / PPL@16384
- 每个 config 的 mean±std 汇总表（脚本末尾也会打印）

### 3.3 `run_350m_final.py`（350M 终极验证，运行中）

用途：验证 “hybrid 用较小 theta 打败很大 theta 的 geometric” 是否随模型规模保持。

脚本：
- `a100/run_350m_final.py`

特点（和 50M 不同的地方）：
- 训练 tokens 更大（500M），不能把 tokens 全放内存里；脚本会把 streaming tokenization 写成磁盘 memmap cache，然后训练时按 chunk id 取数据。
- tokenization 阶段是 CPU+磁盘为主：**GPU 0% 不代表挂了**。

输出（远端）：
- `/opt/dfrope/results/350m_final/run.log`
- `/opt/dfrope/results/350m_final/results.json`（每个配置结束会写）

## 5. GitHub 同步（只同步非权重）

本仓库提供脚本用于从 A100 拉取 350M 的运行日志/结果到本地并推送到 GitHub（不含权重，不含 memmap cache）。

脚本：

- `scripts/pull_a100_350m_artifacts.sh`
- `scripts/commit_and_push.sh`

使用方式（仓库根目录）：

```bash
scripts/pull_a100_350m_artifacts.sh
scripts/commit_and_push.sh "Update 350M artifacts"
```

认证说明：

- 优先使用 SSH key/ssh-agent
- 若必须用密码，可临时设置环境变量 `A100_SSH_PASS`（脚本会用 `sshpass`，仓库不保存密码）

## 4. 常见问题（FAQ）

### Q1：`nvidia-smi` 显示 GPU 0%，是不是挂了？

如果脚本正处于 “streaming tokenization -> memmap cache” 阶段，GPU 0% 是正常的。应优先检查：

- 进程是否存在：`pgrep -af run_350m_final.py`
- 日志是否继续增长：`tail -n 50 /opt/dfrope/results/350m_final/run.log`
- cache 是否增长：`du -sh /opt/dfrope/results/350m_final/cache`

### Q2：为什么 A100 / A800 的数会差一点？

streaming 拉取、缓存、tokenize 的细微差异会导致样本流不完全一致；这是我们做 3-seed 汇总的原因。重要结论应以 mean±std 为准。
