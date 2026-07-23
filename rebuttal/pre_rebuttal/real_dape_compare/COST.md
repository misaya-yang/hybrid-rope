# 5090 GPU 成本估计（真实 DAPE-ish 对照）

单位：单卡 RTX 5090 假设（24GB+ 类；实际以 `nvidia-smi` 为准）。
**未开机实测前均为估算**；phase11b 历史 wall-clock 作锚。

---

## 历史锚点

| 协议 | 每 run 大约 wall-clock | 来源 |
|------|------------------------|------|
| L=128, 15M tok, plain（Geo/EVQ/free） | 整包 Phase1+2+3 ~**25 min**（多 run） | `docs/exp/2026-02-24_128tok_baseline_report.md` |
| L=256, 100M tok, plain 125M | ~**15 min** (900s) | phase11b curated |
| L=256, 100M tok, +DAPE manual attn | ~**27 min** (1600s) | phase11b curated |

缩放粗算（tokens 与 seq 主导）：

\[
T \propto \text{tokens} \times \text{attn 常数}
\]

L=128 / 15M 相对 L=256 / 100M ≈ \(15/100 \times\) 序列更短，plain 每 run 约 **2–6 min**（5090 通常比旧卡更快）。
DAPE manual \(O(L^2)\) attention ≈ plain 的 **1.6–2×**。

---

## 推荐最低成本矩阵

### Tier 0 — 不需要 GPU

| 工作 | 成本 |
|------|------|
| 诚实重标 Table 4 free_inv_freq | 0 |
| 引用 phase11b：EVQ 对 Kerple+MLP **不赢** | 0 |
| 本目录 `python -m pytest` / smoke | CPU 分钟级 |

**结论：若问题是「有没有赢过真实 DAPE-ish」——已经答完：没有。不必为「证明赢」开机。**

### Tier 1 — Pilot（建议上限）

| 项 | 值 |
|----|-----|
| 协议 | P1：L=128, 15M, 125M |
| seeds | **仅 42** |
| methods | 5：`geo, evq, free_inv_freq, kerple, dape_kerple_mlp` |
| 估计 wall-clock | **1.0–2.5 h** 单 5090 |
| 估计电费量级 | 可忽略相对租卡；租卡按 **2–3 h** 计 |
| 目的 | 在 Table 4 协议上补 **诚实对照表**；**预期不翻盘** |

### Tier 2 — 多 seed（仅当 pilot 出现意外）

| 项 | 值 |
|----|-----|
| seeds | 42, 137, 256 |
| methods | 同上 5 |
| 估计 | **3–8 h** 单 5090 |
| 触发条件 | pilot 显示 EVQ@8K **显著优于** `dape_kerple_mlp`（与 phase11b 矛盾）→ 必须多 seed 验证实现 bug vs 真信号 |

### Tier 3 — 官方 DAPE 复现（不推荐作最低成本）

| 项 | 值 |
|----|-----|
| 栈 | GPT-NeoX + Pile Arxiv/Books3 + 官方 config |
| L_train | 常 512+ |
| 成本 | **数十到数百 GPU-hour** 量级 + 数据准备 |
| 对本 rebuttal | 超预算；且 phase11b 已足够否定「EVQ 绝对碾压 DAPE」幻想 |

---

## 开机门禁（AGENT.md 对齐）

GPU 计费前必须完成：

1. [ ] `FINDINGS.md` 已读：接受「可能再次确认 EVQ 不赢」
2. [ ] FineWeb cache 已在本地，hash 写入 manifest
3. [ ] `python rebuttal/pre_rebuttal/real_dape_compare/run_dape_compare.py --smoke` 通过
4. [ ] `python rebuttal/pre_rebuttal/real_dape_compare/run_dape_compare.py --dry-run` 打印 5 个 run_id
5. [ ] 启动命令与 `WORK` 路径固定
6. [ ] 开机后 **5 分钟内** 看到 step>0 与 loss；否则关机查日志

**禁止**：GPU 上现下 FineWeb；未 smoke 就跑 full；把 free_inv_freq 结果写成 DAPE。

---

## 命令示例（Tier 1）

```bash
# 数据已缓存后：
export EVQ_DAPE_WORK=/path/to/work/real_dape_p1
python rebuttal/pre_rebuttal/real_dape_compare/run_dape_compare.py \
  --protocol p1 \
  --seeds 42 \
  --methods geo,evq,free_inv_freq,kerple,dape_kerple_mlp \
  --train-cache /path/to/train_fineweb_....pt \
  --val-cache /path/to/val_fineweb_....pt \
  --work "$EVQ_DAPE_WORK"
```

预计：5 个 run，约 1–2.5 小时。
