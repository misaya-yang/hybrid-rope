# Mac M4 Max Local Experiments

> 设备：M4 Max, ~20GB 可用内存, MPS backend
> 环境：`conda activate aidemo`
> 定位：Phase 18 之后的补充实验，全部 125M 模型，单 run ~10-15min

## 实验清单（按优先级）

### EXP-1: Progressive Chain 512→1024→2048 (最高优先级)

**目的**：验证 "EVQ 越训越强 + YaRN 相变" 在 125M 上复现，并延伸到 2048。

Phase 17b (454M) 发现：
- 512→1024 后 EVQ raw 反超 EVQ+YaRN (PPL@16K: 11.2 vs 16.8)
- EVQ 优势从 34.6%→83.1%@16K

如果 125M 上也成立，说明这不是 454M 的特殊现象。
如果延伸到 2048 后趋势继续，趋势线就有 3 个点。

**设计**：
- Stage 0: 125M, L=512, 50M tokens, base=500K → checkpoint
- Stage 1: 从 Stage 0 checkpoint continue, L=1024, 25M tokens → checkpoint
- Stage 2: 从 Stage 1 checkpoint continue, L=2048, 25M tokens → checkpoint
- 每个 stage 评估: raw + YaRN, PPL@{512,1K,2K,4K,8K,16K,32K}
- 方法: Geo vs EVQ (τ*=d/√L per stage)
- Seeds: 42, 137, 256

**预计时间**：3 stages × 2 methods × 3 seeds × ~10min = ~3h
**脚本**：`exp1_progressive_chain.py`

---

### EXP-2: Multi-Seed Full Grid (第二优先级)

**目的**：给 Phase 17b "YaRN 相变" 发现提供 3-seed error bars。

**设计**：
- 125M, base=500K, d_head=64
- 训练配置: L=512 (50M tok) + L=1024 continue (25M tok)
- 评估: 8 configs = {geo, evq} × {512_only, 1024_cont} × {raw, yarn}
- 每 config 评估 PPL@{2K,4K,8K,16K,32K}
- Seeds: 42, 137, 256

关键验证点:
- [ ] evq_1024_cont raw < evq_1024_cont yarn? (YaRN 负优化)
- [ ] evq_512+yarn ≈ evq_1024_cont raw? (training-inference 等价)
- [ ] EVQ advantage 从 512→1024 放大?

**预计时间**：~2h (大部分训练和 EXP-1 共享 checkpoint)
**脚本**：`exp2_multiseed_grid.py`

---

### EXP-3: τ Robustness Landscape (第三优先级)

**目的**：在多个 base 下验证 τ* 附近 loss landscape 平坦。

**设计**：
- 125M, L=512, base ∈ {100K, 500K, 1M}
- 对每个 base: τ ∈ {0.5τ*, 0.75τ*, τ*, 1.25τ*, 1.5τ*, 2.0τ*}
- 评估: PPL@{512,2K,4K,8K}
- Seed: 42 (pilot), 如果有时间加 137

**预计时间**：3 base × 6 τ × ~10min = ~3h (pilot single-seed)
**脚本**：`exp3_tau_landscape.py`

---

## 执行方式

所有脚本设计为可断点续跑（检查 result.json 是否存在）。
可以让 Claude Code 随时执行：

```bash
cd ~/neurIPS-2026/hybrid-rope
conda activate aidemo
python scripts/mac_train/exp1_progressive_chain.py          # EXP-1
python scripts/mac_train/exp2_multiseed_grid.py             # EXP-2
python scripts/mac_train/exp3_tau_landscape.py --pilot      # EXP-3 pilot
```

## 依赖

所有脚本复用 `scripts/core_text_phases/run_evq_sweep.py` 的模型和数据加载。
