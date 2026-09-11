# ds_workspace

本工作区的分析与实验记录。原则见 [LESSONS.md](LESSONS.md)——**从失败中学习**，每一条规则
都带出处，且写明它在下一次实验里具体禁止什么。

代码在 `experiments/zerotrain_20260910/`（零训练筛选与转数窗口）与
`experiments/joint_kkt_20260910/`（冻结侧 KKT 工具链，CPU 自测 35/35）。

---

## 当前研究线

**目标**：解释 YaRN/MrRoPE 为什么强、它们的规则是不是最优、如果不是就直接找出最优，
并让理论与 KKT 同构。方法必须**跨模型族通用**。

**当前答案**（推导见 [TURN_WINDOW_THEORY](recon_20260910/TURN_WINDOW_THEORY_20260910.md)、
[BAND_FORCING](recon_20260910/BAND_FORCING_20260910.md)、[KKT_ANALYTIC](recon_20260910/KKT_ANALYTIC_20260910.md)）：

> **频率侧**：YaRN 家族的频带规则 = **训练窗内转数落在 [α, β] 圈**，α=1、β=32 是手调常数。
> 转数无量纲 ⇒ 同一个 (α, β) 自动落到任何 checkpoint 上（已复现 Qwen [23,40] 与
> OLMo [14,32]，且分别以机器精度复现 MrRoPE 与部署 BM）。
>
> **形状侧**：两张部署表是**同一个 Poisson 问题**的两个解——
> `L ε = f`（Dirichlet Laplacian，零边界），`f ≡ const` 给 BM（抛物线），
> `f = (2/n)δ_n` 给 MrRoPE（直线），两者均机器精度吻合。
> 零边界条件不是假设：它等于「m 在频带两侧贴住盒子 [0,1]」，
> **频带本身就是活跃集**。
>
> **但一阶 KKT 对形状轴无分辨力**（今晚实测）：长程梯度在频带内**恒为零**，
> `<e, BM−MrRoPE> = −0.178` 对 `|e| = 92.4`；窗内差 ≤ 5e−3 nats；
> 而端点实差 OLMo **48.5 点 / 0.83 nats**。
> ⟹ 这条轴是**阈值/分岔**，不是光滑最优点。见
> [CONTRACTION_KKT1](recon_20260910/CONTRACTION_KKT1_20260910.md)。

---

## 文件索引

| 文件 | 内容 |
|---|---|
| [LESSONS.md](LESSONS.md) | **工作规则**：12 条从失败中提炼的禁令，每条带出处 |
| [recon_20260910/TURN_WINDOW_THEORY_20260910.md](recon_20260910/TURN_WINDOW_THEORY_20260910.md) | 转数窗口理论：推导、复现、与 KKT 的关系、最优边界的求法 |
| [recon_20260910/BAND_FORCING_20260910.md](recon_20260910/BAND_FORCING_20260910.md) | 带内 profile 是 Poisson 问题的解；两张部署表的统一；forcing 形状预测 |
| [recon_20260910/CONTRACTION_KKT1_20260910.md](recon_20260910/CONTRACTION_KKT1_20260910.md) | **第一次解空间收缩**：一阶路线对形状轴无分辨力（实测） |
| [recon_20260910/KKT_ANALYTIC_20260910.md](recon_20260910/KKT_ANALYTIC_20260910.md) | 解析推导：Dirichlet **不是** Fisher 损伤型（已证）；混合族 `ε ∝ k(C−k)`；各臂隐含 forcing |
| [recon_20260910/EVQ_UNIFY_20260910.md](recon_20260910/EVQ_UNIFY_20260910.md) | EVQ vs 三段族：九张表数值对比、保持区 KKT 不等式、判决实验设计 |
| [recon_20260910/CONTRACTION_SUPPORT_20260910.md](recon_20260910/CONTRACTION_SUPPORT_20260910.md) | **第二次收缩**：窗内轴不为三段式辩护；逐长度 offset 校正；仪器在 32K 被归档验证 |
| [recon_20260910/CORRECTION_QWEN_20260911.md](recon_20260910/CORRECTION_QWEN_20260911.md) | **更正**：Qwen 上的「反转」配对检验不显著（4W/4L/16T，CI 含 0）；真实命题是「同一干预在一个模型上巨大、另一个测不到」 |
| [recon_20260910/DOC_SYNTHESIS_20260910.md](recon_20260910/DOC_SYNTHESIS_20260910.md) | **Pro 文档综合**：它的 kill 条件已被触发；后半是绕开 g_long 的转向；排序因果实验是唯一不受影响的高 ROI 项 |
| [recon_20260910/RESULT_B2_OLMO_20260910.md](recon_20260910/RESULT_B2_OLMO_20260910.md) | **首个非归档赢家**：OLMo 上 b=2 达 50.01% vs BM 41.67%（+8.34pp，7 任务中 6 个改善） |
| [recon_20260910/RESULT_SIGMA_SHAPE_20260910.md](recon_20260910/RESULT_SIGMA_SHAPE_20260910.md) | **三个方向的赢家全落在 Σm≈42**；Σm 主导（r=0.96）但**不充分**（同 Σm 差 12pp）；**β 外推 = 向保持区渗漏**，现任最优 53.84% |
| [recon_20260910/PHASE1_ROUND1_20260910.md](recon_20260910/PHASE1_ROUND1_20260910.md) | Round 1：窗内轴对中段形状不可辨识（29 臂 × 16 文档 × 3 长度） |
| [recon_20260910/K6_PREFLIGHT_20260910.md](recon_20260910/K6_PREFLIGHT_20260910.md) | 事前体检：规划字面的 K6 四格退化（只改 2 槽），及 η 交叉推导 |
| [codex_failures_20260910/](codex_failures_20260910/) | codex transcript 全量失败挖掘（3 片，31 MB，~70 个失败案例 + 全部 veto） |
| [recon_20260910/recon/](recon_20260910/recon/) | 仓库侦察：表构造/排序、稀疏与训练设施、探针与语料 |
| [recon_20260910/audit/](recon_20260910/audit/) | 代码审计：新 KKT 文档 vs 现有代码，迁移成本与 veto 再入风险 |
| [recon_20260910/design/](recon_20260910/design/) | 首轮实验设计（三实验排序与算力） |

---

## 已确立的事实（可复算）

| 事实 | 出处 |
|---|---|
| MrRoPE = `ε_k ∝ k`；部署 BM = `ε_k ∝ k(n+1−k)` | **从归档 `prepared/tables.json` 直读**，与精确有理数比，误差 1.1e−7 / 8.3e−8 |
| 二者同带、**同 gain（1.138629436111989）**、平台区**逐位相同**，只差带内 18 个增量的分布 | 同上（`changed_slots = [15,31]`） |
| 同一对表：OLMo 16K **51.32% vs 2.78%**（350 行，压倒性）；Qwen 128K **70.83% vs 78.13%**，配对检验 **4W/4L/16T，CI 含 0** ⇒ **不显著** | 归档 `run_ruler_newtasks_01/`；**更正见 [CORRECTION_QWEN](recon_20260910/CORRECTION_QWEN_20260911.md)** |
| 频带 = `1 ≤ W·ω/2π ≤ 32`，且这解释了 Qwen 与 OLMo 两个不同的带 | TURN_WINDOW 推导 + 数值复核 |
| `L⁻¹(1) ∝ 抛物线`（即 BM），`L⁻¹(δ_n) ∝ 直线`（即 MrRoPE），均机器精度 | BAND_FORCING + KKT_ANALYTIC，双方独立复算 |
| `R(BM)/R(MrRoPE) = 3/(n+2)`，**精确**；MrRoPE 粗糙度的 94.4% 是末端那一跳 `ε_n²` | KKT_ANALYTIC（精确 Fraction） |
| **Dirichlet 能量不是 Fisher 损伤型**——`K_W` 只依赖 `max(i,j)`，`L_D` 不是（已证） | KKT_ANALYTIC §1.2 |
| **长程梯度在频带内恒为零**（槽 24–63，\|∂L/∂m\| ≤ 0.05；槽 0–23 量级 10–70） | `kkt_residual.py` 首测（4 行，噪声底待补） |
| `<e, BM−MrRoPE> = −0.178`，`\|e\| = 92.4` ⇒ **一阶模型对该轴无分辨力** | 同上 |
| Round 1：中段形状在窗内只动 **≤5e−3 nats**，不可辨识 | Round 1 receipt（29 臂） |
| 三个几何旋钮**已被证明**耗尽，MrPro 取到下界 | transcript F4，精确算术 N=2..100 |
| **OLMo b 曲线单调穿过 BM**：7.09 / 14.47 / 23.20 / 41.67 / **50.01** | `olmo/beta_b*_summary.json`，350 行 × 5 臂 |
| **仪器可复现性**：两张**逐位相同**的表、**两次独立运行**，结果差 **0.00e+00** | `mixC_18` vs `beta_b1` 跨 run 对照 |

## 正在跑（GPU 100%，~20 GB，~300 W，三条链）

```
chain_v2  kkt(已完成) → olmo_beta → qwen_beta → qwen_turns → olmo_turns → gain → CPU 派生
chain_v3  等 v2 哨兵 → kkt 重测（16 行，带噪声底）×2 → support 轴 → rank ×2
chain_v4  等 v3 哨兵 → mixC 族（正确的插值路径）
chain_v5  等 v4 哨兵 → **排序因果实验**（OLMo，5 臂；progressive=MrRoPE 取自归档）
chain_v6b 等 v5 哨兵 → b=3/4/6/8 + **leak 长程判决** + **消共线 (a,b) 单元**
手动并发  support2（Qwen，18 臂：mixC + 扩展 b + 排序）
```

**两端点一律取自归档，不重跑**（MrRoPE 78.13% / BM 70.83%；OLMo 2.78% / 51.32%）。

## 解空间收缩记录

见 [CONTRACTION_KKT1](recon_20260910/CONTRACTION_KKT1_20260910.md)。已废弃的候选：
`ε = −(1/λ)L⁻¹g`（一阶）、`<e,m>` 与 `<e+λn,m>` 排序（λ̂=0）、
「压缩区便宜/保持区昂贵」对**带内分配**的论证。

## 已测出的关键对照

| 轴 | 状态 |
|---|---|
| 带内**形状** | 窗内不可辨识；长程梯度恒零 ⇒ **阈值型，局部量无法描述** |
| **支撑**（保持区是否可动） | 正在测：EVQ/companding 8 臂 + 漏槽 5 臂（预算精确守恒） |
| **带边界**（转数 α, β） | 已排队（Qwen + OLMo 各 4 臂） |
| b 族插值 | 已排队（b = 0.25, 0.5, 2.0，两模型） |
| `mixC` 混合族（正确插值） | 正在跑（support2 已出 mixC_18） |
| **Σm vs 形状**（b=2 赢在哪） | **已排队判决**：`beta_b2`(Σm 42.379) vs `leak_a0p01`(Σm 42.740) |
| **增量顺序**（同多重集，只换顺序） | **已排队（chain_v5）**——唯一不受零梯度影响的高 ROI 实验 |
