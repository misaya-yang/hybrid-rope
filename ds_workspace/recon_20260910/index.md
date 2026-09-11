# index — RoPE 频率分配 campaign（截至 2026-09-11）

零训练 · OLMo-2-0425-1B-Instruct（主仪器）+ Qwen2.5-1.5B（跨模型）
桌面汇总：`~/Desktop/RoPE_KKT_核心结论_20260911.md` · 公司执行：`RUNBOOK_OFFICE_20260911.md`

---

## ★ 先读三条

### 1. held-out 测试推翻了"超过部署 BM"这个头条

72 行、**6 个从未用于选择的任务**上，冠军 `b3_lo14` 与部署 BM **完全相同**
（0.5350 vs 0.5350，Δ=−0.00pp，t=−0.00），而它在**用于选择的** 350 行面板上是 **+14.20pp**。

> **⟹ 那 +12～14pp 主要是任务选择的产物**，集中在 `niah_single_3` 一个任务（它贡献 +42～+46pp）。
> **四个成员测完，排序被完全推翻**：held-out 上 `wide_b4` **+6.94pp** > `turns_a1_b64` **+4.40pp**
> > `b3_lo14` **−0.00pp**（= BM）；而选择面板的排名恰好是 `b3_lo14` > `wide_b4` > `a1_b64`。
> **合并检验（三成员）：+3.78pp，t=+1.22 —— 不显著。**
> **★ 而连续仪器（@16384 NLL）的排名与 held-out 一致**——它比 RULER 选择面板更会选。
> **仍成立且跨仪器复现的是：MrRoPE 显著差于部署 BM（0.0709 vs 0.4167）。**
> **第一期目标（零训练超过 MrRoPE）达成；"超过部署 BM"未被稳健证明。**
> → `HOLDOUT_VERDICT_20260911.md`

### 2. YaRN / MrRoPE / 部署 BM 是同一单纯形的三个顶点（机器精度验证）

增量坐标下 `Lε` 是设计变量：
- **部署 BM**：`Lε` **逐点常数**（`+0.00175`，min == max）
- **MrRoPE**：`Lε` = **慢端一个点源**（其余全零）

**⟹ "MrRoPE 的中频改进"在理论上就是：把 forcing 从常数改成慢端的一个点源。**
第四个顶点（快端）从未被试过 —— **实测判决：角是坏的，最优在内部**。
→ `FOUR_CORNERS_20260911.md`

### 3. KKT 条件退化了：带内重分配免费

带内逐槽窗内代价 `∂D/∂m_j` 符号交替、均值 +0.0015（1 SE 内）；
**代价的分界线是"是否作用到频带之外"**（带内 ≲0.009，全局 +0.067…+3.17，归档 Qwen 扫描验证）。

> **⟹ "最小化窗内损失 / 最大化外推能力"在带内不是权衡——只有一个目标在动。**
→ `CONSTRAINT_IS_SLACK_20260911.md`

---

## 结论清单（按可复现强度排序）

| # | 结论 | 强度 | 文档 |
|---|---|---|---|
| 1 | **MrRoPE 显著差于部署 BM**（0.0709 vs 0.4167，350 行；连续 3.69 vs 2.86） | ★★★ 跨仪器、效应巨大 | `HEADLINE` |
| 2 | **`Lε` forcing 框架**：BM 常数 / MrRoPE 末端点源 | ★★★ 机器精度 | `FOUR_CORNERS` |
| 3 | **`S = 31 + Σ(19−i)ε_i`**（带 [15,32]/n=18） | ★★★ 10 case 机器精度 | `FOUR_CORNERS` |
| 4 | **带内重分配免费，代价只在带外** | ★★★ 归档扫描 + 逐槽实测 | `CONSTRAINT_IS_SLACK` |
| 5 | **C42/C42V24 受控对在两台仪器上同向分离**（RULER +10.73pp t=+5.47；连续 −0.1109 t=−3.91） | ★★★ **本战役唯一跨仪器验证的机制** | `HEADLINE` |
| 6 | 四个角：两角都坏，最优在内部（同带 b=3 胜缺角 t=−3.98） | ★★ 16 篇配对 | `FOUR_CORNERS` |
| 7 | **`S` 不是操作变量**（同 S 差 10.73pp） | ★★ 受控对 | `HEADLINE` |
| 8 | **held-out：`wide_b4` +6.94pp > `a1_b64` +4.40pp > `b3_lo14` −0.00pp；选择面板排名被完全推翻；合并 +3.78pp t=+1.22 不显著** | ★★ 72 行配对 | `HOLDOUT_VERDICT` |
| 8b | **连续仪器 @16384 的排名与 held-out 一致，RULER 选择面板的不一致**（3 臂 1 次观察，待预注册检验） | ★ 待证 | `HOLDOUT_VERDICT` §三·终 |
| 9 | **Qwen 两台仪器都判不了**：4× 上八个压缩臂跨 **0.0077 nats**（SE 0.030），覆盖 S=29.3–35.0 的整个三段式家族；归档 RULER 面板自判 `NO_LONG_GAIN`（6胜4负） | ★★ | `CROSS_MODEL_VERDICT` |
| 10 | **12 个静态泛函全部出局** | ★★ 三类受控对 | `NO_STATIC_FUNCTIONAL` |

## 被证伪的（解空间收缩，成本已付）

| 被证伪 | 依据 | 文档 |
|---|---|---|
| `S`（预算/质心） | 同 S 差 10.73pp | `HEADLINE` §2.0b |
| 12 个静态泛函（`S_in`,`S_entry`,`C`,`N`,`dev`,`cv`,`ks`,`span`,`Σν`…） | 三类受控对 | `NO_STATIC_FUNCTIONAL` |
| **forcing 坐标下的形状** | 坏表与好表的 `Lε` 形状相关 0.899 | `NO_STATIC_FUNCTIONAL` §六 |
| **Fisher 二次代价** | **符号错**：`F_24=7842` 恒正，真实 ΔNLL 在 m=1 处 −0.0150 | `WHY_THE_FISHER_ROUTE_DIED` |
| 单槽梯度路线 | SNR<1（单槽扰动不移动 NLL） | `CONSTRAINT_IS_SLACK` §六 |
| **可读窗计数 N / 释放平台** | 六张表全部差于 BM（N 从 17 提到 22–26 无改善） | `RELEASE_AXIS` |
| 两条"建模后解最优"的路线 | `Tstar` 3.0372 与 `step_hi25` 3.0452 落到同一张崩掉的表 | `RELEASE_AXIS` §6.1 |

## ★ 正在进行：180 行 held-out（预注册已提交 `62cef88`）

72 行 held-out 说"不显著"（pooled +3.78pp, t=+1.22），但**最大臂 SE=4.36pp，
排除不了 ±10pp 以内的任何东西**。**"没测出来"不等于"是零"。**

**材料**：三个现有 held-out 面板的并集 = **180 行**（仍是那 6 个选择面板从未用过的任务，
每任务 30 行），与选择面板任务重叠 **0**，行 id 冲突 **0**。
按 1/√N 外推 **SE≈2.8pp，t=2 可判 ~5.5pp**。

**判据（写在读数之前，`HOLDOUT180_PREREG_20260911.md`）**：

| pooled Δ | 判定 |
|---|---|
| ≥ +5pp 且 t ≥ 2 | 效应真，72 行那次是**功效不足** |
| ≤ +1.5pp 且 SE ≤ 3pp | 效应在噪声内，**"调参不能泛化"升级为已确立结论** |
| 其它 | **仍未决，报告区间并停止**（不得为移动它而加行数） |

**4 臂（BM 端点 + 三个平台成员），约 30 分钟。不引入新表、不做形状搜索。**
读取脚本 `code/holdout180_read.py` 直接实现该判据。

**我的预判（也写进预注册）**：pooled Δ 落在 **+2～+5pp**，即弱正向但达不到判据。

## 六·补 本轮的流程性教训（都已在服务器上遇到并修复）

| # | 问题 | 修法 |
|---|---|---|
| 1 | **嵌套 heredoc 静默失败**（`ssh '... <<"EOF" ... EOF'`）：外层引号吃掉终止符，文件**没被创建**却无报错 | **一律写本地文件 + `scp`**；部署后立即 `ls -la` 核验 |
| 2 | **`$ROOT` 未定义** ⟹ `--root "$ROOT/holdout"` 展开成 `/holdout`，最终测试写到了文件系统根 | chain 脚本里每个变量显式定义；启动时 echo 出来核验 |
| 3 | **重复启动**：等待条件相同的两条链会同时醒来；且 `setsid` 后 ppid=1，来源不易追 | 串行化 + 部署后核对 `pgrep -af` |
| 4 | **`pkill -f` 自杀**（模式匹配到自己的命令行） | 用中括号 `chai[n]` 且按 PID kill |
| 5 | **runner 缺依赖时静默** | 每次构造表后跑一个**已知答案的自校验** |

> **共同点：都是「命令成功了但什么都没发生」或「发生了两次」这类静默失败。**
> **⟹ 纪律：部署后立即核验产物存在且内容正确，不信 exit code。**

## 未决 / 待跑

| 项 | 状态 | 命令 |
|---|---|---|
| `--pro-tables condEVQ,step42`（Pro §8/§9.3） | 代码**已接好并 dry-run 验证**，未跑 | `RUNBOOK` §3 |
| Pro §7 Qwen 传输表（S=33.47081679 已验证） | 表已构造，未接进 runner | `RUNBOOK` §5 |
| **投影 Fisher**（修 SNR 的正路） | 未做。Pro §4 建议，子代理 A 建议**改靶测 R 的 Hessian** | `_reports/PRO_PLAN_VERIFY_A` |
| Pro §6 的 `D_pattern` | 零 GPU，从现有 jsonl 可算 | `_reports/PRO_PLAN_VERIFY_B` |
| EVQ τ=0.5/1/2 首次长程 | `chain_serial` 跑中 | — |
| `--gain-tables` 2×2 | `chain_pro` 排队 | `RUNBOOK` §3 |
| `hold_a1b64` / `hold_b4wide` | `chain_holdout` 跑中 | `RUNBOOK` §2B |

## 三个子代理的报告（`_reports/`）

| 文件 | 内容 |
|---|---|
| `PRO_PLAN_VERIFY_A_20260911.md` | Pro 计划 §2/§4 逐条判决。**h=22 的真正对照是 `ctl_C42`**（同 S/S_in/N/可读集）；**h=22 的 forcing 是 `−1,+2,−1`（内部点源，全新对象）** |
| `MINIMAL_PARAMS_20260911.md` | 最小参数集 3 个 `{N, med_h, ex_needed}`；**N 已被证伪，该基需重算** |
| `EVQ_LIMIT_20260911.md` | 可读窗是宽 3 的斜条带；条带斜率 ≡ ν 退化斜率 ≡ `lnθ/(2K ln2)` |

## 目录结构

```
recon_20260910/
  index.md                     ← 本文件
  HEADLINE / RUNBOOK_OFFICE / HOLDOUT_VERDICT / FOUR_CORNERS /
  CONSTRAINT_IS_SLACK / WHY_THE_FISHER_ROUTE_DIED / COST_IS_NOT_QUADRATIC /
  NO_STATIC_FUNCTIONAL / PLATEAU / RELEASE_AXIS / CROSS_MODEL_VERDICT
  _reports/     子代理深度报告（7 份，证据保留）
  _archive_20260911/  已被后续结果覆盖的早期文档（23 份，见其 README）
  code/         可复算脚本（纯 numpy）
  audit/ design/ recon/ work/
```

> **本文档取代 `INDEX_20260911.md`**（已移入 `_archive_20260911/`）。
