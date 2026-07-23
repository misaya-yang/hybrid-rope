# real_dape_compare — 诚实 DAPE 对照（rebuttal 包）

## 先读结论

详见 **`FINDINGS.md`**：

1. Table 4「DAPE」= **free inv_freq** → EVQ **赢**（但不是 Zheng DAPE）。
2. Phase11B **Kerple+MLP**（最接近真实 DAPE）→ EVQ **没有赢**（3-seed）。
3. 官方 GPT-NeoX DAPE：**没做过**。
4. **不要**为「证明 EVQ 打赢真实 DAPE」开机；现有证据不支持。

## 文件

| 文件 | 用途 |
|------|------|
| `FINDINGS.md` | 已有实验盘点（只读事实） |
| `SPEC.md` | P1 协议与身份标签 |
| `COST.md` | 5090 时间/门禁 |
| `run_dape_compare.py` | 可复现 runner |
| `tests/test_smoke.py` | 离线/ smoke 测试 |
| `expected/phase11b_dape_headline.json` | phase11b  headline 数字快照 |

## 快速检查（无 GPU）

```bash
cd /path/to/hybrid-rope
python -m pytest rebuttal/pre_rebuttal/real_dape_compare/tests/test_smoke.py -q
# 或
python rebuttal/pre_rebuttal/real_dape_compare/run_dape_compare.py --dry-run
python rebuttal/pre_rebuttal/real_dape_compare/run_dape_compare.py --smoke --methods geo,evq --work /tmp/dape_smoke
```

## 5090 最低成本（仅当需要 L=128 诚实表）

1. 读完 `FINDINGS.md` + `COST.md`，接受可能再次确认「不赢」。
2. 准备 FineWeb train/val `.pt` cache（**GPU 前完成**）。
3. Smoke + dry-run 通过。
4. Pilot seed 42，约 **1–2.5 h**：

```bash
export EVQ_DAPE_WORK=/data/real_dape_p1
python rebuttal/pre_rebuttal/real_dape_compare/run_dape_compare.py \
  --protocol p1 \
  --seeds 42 \
  --methods geo,evq,free_inv_freq,kerple,dape_kerple_mlp \
  --train-cache /path/to/train.pt \
  --val-cache /path/to/val.pt \
  --work "$EVQ_DAPE_WORK"
```

禁止使用 method 名 `DAPE`；脚本会拒绝。

## 身份强制规则

| method_id | 含义 |
|-----------|------|
| `free_inv_freq` | 旧 Table 4 误标行 |
| `dape_kerple_mlp` | Zheng-inspired Kerple+MLP |
| `evq` | 固定 τ EVQ-Cosh（P1 默认 τ=5.0） |

每个 `result.json` 含 `identity` / `identity_not` / 禁止项。

## 与 phase11b 关系

- **不必重跑** L=256 100M DAPE 兼容实验；见 `data/curated/phase11b_125m_l256_3seed.json`。
- 本包 P1 是 **Table 4 协议上的诚实补洞**，不是翻盘实验。
