# DAPE 对比实验：已有结果盘点（只读事实）

日期：2026-07-12
状态：`author-internal` — 不是 OpenReview 文案

---

## 一句话结论

| 问题 | 答案 |
|------|------|
| 是否做过「相对 Table 4 那行 DAPE」且 EVQ 赢？ | **是，但那行不是 Zheng DAPE**，是 **32 free `inv_freq`**。 |
| 是否做过更接近 Zheng 的 Kerple+MLP DAPE 且 EVQ 赢？ | **做过，EVQ 没有赢。** |
| 是否做过官方 GPT-NeoX + Pile 上的 Zheng DAPE 复现？ | **没有。** |
| 因此「对真实 DAPE 赢」能否支撑 rebuttal 主叙事？ | **不能。** 现有证据指向 **DAPE-ish 算子在绝对 PPL 上碾压 plain EVQ**；EVQ 的定位应是 **零额外参数频率分配**，不是「打赢 DAPE」。 |

---

## 实验 A：Table 4 / Primary II（论文打印行）

| 项 | 内容 |
|----|------|
| 协议 | 125M，`L_train=128`，FineWeb-Edu，**15M tokens**，base=500K |
| 行名 | 论文写 **DAPE**，extra params=**32** |
| 实际方法 | **32 维可学 `inv_freq`**（独立频率，非 Kerple、非 attention-MLP） |
| 证据 | `docs/exp/2026-02-24_128tok_baseline_report.md`；`data/curated/fig3_extreme_128.json` |
| Seed | Geo / 「DAPE」/ EVQ：**seed 42 only**（Learnable τ 为 3-seed） |

### 结果（PPL@8K，seed 42）

| Method | Extra params | PPL@128 | PPL@8K | vs Geo |
|--------|--------------|---------|--------|--------|
| Geo | 0 | 184.9 | 513.7 | — |
| Free inv_freq（误标 DAPE） | 32 | 183.6 | 455.3 | −11.4% |
| EVQ τ=5.0 | 0 | 182.0 | **333.7** | **−35.0%** |
| Learnable τ（3-seed mean） | 1 | 181.2±1.3 | 437.9±12.2 | −14.8% |

**相对 free-inv_freq：EVQ 赢。**
**相对 Zheng DAPE：此表无资格说话。**

---

## 实验 B：Phase 11B Kerple+MLP（仓库内最接近 DAPE 的实现）

| 项 | 内容 |
|----|------|
| 协议 | 125M，`L_train=256`，FineWeb-Edu，**100M tokens**，3 seeds (42/137/256) |
| 实现 | `scripts/core_text_phases/phase11b_125m_dape.py`：`KerpleBias` + `DAPERefine` MLP on pre-softmax scores |
| 身份边界 | **Zheng-inspired DAPE-ish**，非官方 GPT-NeoX 仓库逐行复现；仍是 attention-score adaptive PE，**不是 free inv_freq** |
| 证据 | `data/curated/phase11b_125m_l256_3seed.json`；`docs/exp/2026-03-05_phase11b_125m_results.md` |
| 训练耗时 | plain ~900s/run；+DAPE ~1600s/run（原服务器日志） |

### 结果（3-seed mean PPL）

**Plain（无 DAPE 模块）@8K：**

| Method | PPL@8K mean |
|--------|-------------|
| Geo | 352.7 |
| EVQ τ=4.0 | **254.7**（约 −27.8% vs Geo） |

**+ Kerple+MLP DAPE-ish @8K：**

| Method | PPL@8K mean | vs Geo+DAPE |
|--------|-------------|-------------|
| Geo+DAPE | **55.9** | — |
| EVQ4+DAPE | 56.8 | **+1.6%（更差/噪声内）** |

**关键读数：**

1. 在 **plain RoPE** 上，EVQ 相对 Geo **赢**（与机制叙事一致）。
2. 加上 **DAPE-ish 算子**后，两者都被拉到 ~56 PPL；**EVQ 不再提供额外优势**。
3. **绝对 PPL：Geo+DAPE ≪ plain EVQ**（55.9 vs 254.7）。若问「EVQ 是否打赢真实 DAPE-ish」，答案是 **否**。
4. 因此 **不能** 用 Table 4 free-inv_freq 行冒充「赢 DAPE」，也不能假装没跑过 adverse 结果。

---

## 实验 C：未做

| 缺口 | 说明 |
|------|------|
| `L_train=128` + Kerple+MLP | Table 4 协议上从未跑真实 DAPE-ish |
| 官方 DAPE (GPT-NeoX, Pile-Arxiv/Books3, L=512) | 未做；成本远高于 PE-dominant 诊断 |
| DAPE V2 (1×3 conv on attention map) | 未做 |
| FIRE/ALiBi + DAPE 全家 | 未做 |

---

## 对「还有一点点机会吗」的直接含义

- **若「机会」= 诚实写 free-inv_freq 对照下 EVQ 更好**：材料已有，**不必重跑**；改标签即可。
- **若「机会」= 证明 EVQ 在绝对指标上打赢 Zheng-style DAPE**：现有 3-seed Kerple+MLP **已经给出否定信号**。在 L=128 再跑一轮 **更可能复现「DAPE 碾压 plain、EVQ 无叠加」**，而不是翻盘。
- **若「机会」= 机制论文残核（零参数分配 vs 重算子）**：phase11b **支持**「DAPE 覆盖频率轴」叙事，与「第三轴互补」一致，但 **不是 win claim**。

**不建议**把 GPU 押在「跑出 EVQ 赢真实 DAPE」上。
**若仍要跑**：目的应是 **补全 L=128 协议上的诚实对照表**（errata 用），接受可能再确认 EVQ 不赢。

---

## 引用路径（复核用）

```
data/curated/fig3_extreme_128.json
data/curated/phase11b_125m_l256_3seed.json
docs/exp/2026-02-24_128tok_baseline_report.md
docs/exp/2026-03-05_phase11b_125m_results.md
scripts/core_text_phases/phase11b_125m_dape.py
README.md Claim 2 (explicit: not faithful DAPE)
```
