# M4 Max Local Experiments

> 设备：M4 Max, ~20GB 可用内存, MPS backend
> 环境：`conda activate aidemo`
> 定位：125M 规模的系统性验证，单 run ~10-15min，适合暴力扫参数空间

## 执行顺序

| 顺序 | 实验 | Phase | 脚本 | 预计时间 | 状态 |
|------|------|-------|------|---------|------|
| 1 | Base Generalization Sweep | Phase 18 | `core_text_phases/phase18_base_generalization_sweep.py` | ~8-11h | 🔄 |
| 2 | Progressive Chain 512→1024→2048 | **Phase 19** | `mac_train/exp1_progressive_chain.py` | ~3-4.5h | ⏳ |
| 3 | τ Robustness Landscape | 待定 | `mac_train/exp3_tau_landscape.py` (未写) | ~3h | ⏳ |

## Phase 19: Progressive Chain (最高优先级)

验证 Phase 17b (454M 单seed) 的三大发现在 125M × 3-seed 上是否复现：
1. YaRN 相变：progressive training 后 EVQ raw > EVQ+YaRN
2. Progressive 放大：EVQ 优势随 stage 单调递增
3. 延伸到 2048：第三个数据点确认趋势

详细计划：`team/plans/phase19_progressive_chain_125m.md`

```bash
cd ~/neurIPS-2026/hybrid-rope
conda activate aidemo
python scripts/mac_train/exp1_progressive_chain.py --pilot   # 验证
python scripts/mac_train/exp1_progressive_chain.py           # 全量
python scripts/mac_train/exp1_progressive_chain.py --summary # 汇总
```

## 设计原则

- 所有脚本支持 `--pilot`（单 seed 验证）和断点续跑
- 复用 `core_text_phases/run_evq_sweep.py` 的模型和数据加载
- 结果存放在 `results/mac_train/` 下，按实验名分目录
