# Llama-3-8B：20 方向 × 60 配置 —— 试验规范与 CPU 参考实现

本目录实现 `LLAMA3_ROPE_20_DIRECTIONS_60_CONFIGS_REVIEW_PLAN_20260911.md` §9 声明的那个包。

**状态：未跑任何 GPU，未产生任何任务成绩。** 这里只有算子编译、代数恒等式检验、
预算算术与统计工具。**fixture 不是结果，`test_tools.py` 里的数字一个都不能引用。**

---

## 0 · 落地前的三件事（实现本包时发现）

### 0.1 计划里的"本包"原本不存在

§9.1 列出 `operators.py` / `budget.py` / `directions.json` / `candidates.csv` /
`paired_report.py` / `upgrade_report.py` / `test_tools.py` / `static_identity_audit.json` /
`reference/`，§9.2 给出可直接运行的命令。但我在**本地全盘与服务器 `/root` 全盘**搜索后
确认：这些文件一个都不存在（唯一的 `operators.py` 命中是仓库里无关的
`experiments/nongeometric_screen/`，以及 `.venv` 里 torch/sympy 的同名模块）。
本目录是**从零实现**，不是"补齐"。§9.1 里 `paired_report.py` 被描述为"原包的配对工具"，
但不存在原包；该文件是按 §5.4/§6.6 的口径新写的。

### 0.2 §6.6 引用的四个数字是 **SE**，不是 95% MDE

§6.6 写 `SE=sqrt((q−δ²)/N)`，随后引"S 合并 N96 约 4.56pp、每长度 N48 约 6.45pp；
H 合并 N256 约 2.80pp、每长度 N128 约 3.95pp"。验算 `sqrt(0.2/96)=4.56pp` ——
**这四个数是标准误**。按 95% 判据用需要 ×1.96（4.56→8.94pp）。
本包把两者显式分开命名（`se` / `mde_95`），`test_tools.py` 对四个原值断言的是 `se`。

这不推翻 §0.2 的标准（那里写的是"配对 95%CI 下界 > 0"，是对的），但读 §6.6 的数字时
必须知道它是什么量——历史上正是这类混淆烧过钱。

### 0.3 §8.1 灵敏度表的**总分钟数**与其自身分项不自洽

分项模型精确复现 §8.1 主表（9 项全中，合计 599.880），也精确复现灵敏度表的
**候选数上限 60/49/41/35**；但灵敏度行报的总分钟数比同一分项口径低 0.2–0.5%
（如 8.0 min/候选时：分项 595.48 vs 计划 594.38）。差异在计划自身算术，**本包不为
对齐而改系数**，`budget.py --explain` 会把两套数并列打印。

### 0.4 另外两处

- §6.5 只定义 PASS / FAIL / UNDERPOWERED，**未覆盖"守门量根本没测"**这一情形。
  本包不把它硬塞进三个标签，而是新增 `INCOMPLETE_GUARDS` 并声明该标签属于本工具、
  不属于计划，且**不得当作 PASS 报告**。
- §1.1 的四条"伪新方法"里，第 5 条（两阶段相同 m 缩放）的正确识别是**两侧相等**
  （相等才说明它退化成一次总倍率），与另外三条的极性相反。本包按无歧义的方式实现：
  字段名 `identified_as_non_method`，四种都是"确认两侧是同一算子"才算识别成功。

---

## 1 · 已验证的锚点

本包的关键正确性证据不是自测通过，而是**与战役既有权威实现逐位一致**：

| 量 | 结果 |
|---|---|
| 频带边界 | `find_correction_range(32,1,128,5e5,8192) = (18,35)`，与 `tables.band_from_turns(1,32,…)` 相同 |
| `m_mrpro`（n=17, low=18） | 与 `experiments/curvature_20260910/tables.m_mrpro` **max\|diff\| = 0.000e+00**（位相同） |
| `nu_mrpro` | 同上，**0.000e+00** |
| gain | `1.138629436111989`，与 `tables.GAIN_YARN` 位相同 |
| §3 的官方 YaRN 公式 | 与官方索引 ramp 实现吻合到 **3.5e-18**，确认计划的写法正确 |

另：`operators.py selftest` 的 **108/108** 条代数恒等式全部通过（含 D02 的绕圈锚点
`exp(iAν′)=exp(iAω/s)`、D03 的无权核置换不变性、D04 的 cos 不变/sin 反号、
D10 在 `d=d0` 处退化为原生、D11 的互逆度量、D12 的相对距离包络、D18 的端点与瞬时斜率）。

---

## 2 · 用法（与计划 §9.2 一致）

```bash
PY=/path/to/python        # 需要 numpy；本仓库内可用 .venv/bin/python

# 60 项算子的代数恒等式检验
$PY operators.py selftest --out cpu_test_local.json

# 列出 60 项及其 scope 与优先级
$PY operators.py list
$PY operators.py list --scope position_phase

# 从**未 patch 的 checkpoint** 导出 stock inv_freq 后编译批准子集
$PY operators.py export \
    --native-npy stock_inv_freq_fp32.npy \
    --window 8192 --theta 500000 --scale 4 \
    --approved-scopes frequency,frequency_assignment \
    --out deployed_review_subset

# 不传 --approved-scopes：只生成审查件，execution_authorized 全部 false
$PY operators.py export --out . --approved-scopes ""

# 预算
$PY budget.py --t8 1.2 --t16 2.5 --t32 4.7 --n-candidates 60 \
    --s-long 96 --v-long 128 --h-long 256 --out budget_example.json
$PY budget.py --t8 <实测> --t16 <实测> --t32 <实测> --fit-candidates --out budget_locked.json

# 规范解析（防漂移：计划改了但 JSON 没跟着改会报错）
$PY parse_plan.py
$PY parse_plan.py --check

# 工具自检（拒绝测试）
$PY test_tools.py --json tool_tests.json
```

`--approved-scopes` **不传或传空**时，`export` 不写任何算子数组，只写 `reference/` 里的
原生频率与一份 60 项 catalogue（`execution_authorized=false`）。这是默认的安全态。

**未批准的 scope 绝不落盘。** 计划 §9.2 明确：若让 D07+ 退化成 MR 表，那一轮实验就是无效的。
`export` 因此把未批准项**从 `exported` 里移除**并记入 `not_exported_scope_not_approved`，
而不是生成一张静默回退的表。

---

## 3 · H 阶段判决

```bash
$PY upgrade_report.py \
    --mr results/H/MR.jsonl \
    --candidate results/H/LOCKED.jsonl \
    --yarn results/H/YARN_INDEX.jsonl \
    --expected-data data/H.jsonl \
    --out upgrade_H.json
```

三个 contrast（§0.2）：`G_new = A(M)−A(MR)`、`G_old = A(MR)−A(YARN)`、
`C_upgrade = A(M)−2A(MR)+A(YARN)`，同一套来源 cluster bootstrap 给**联合**下界
（max-|t| 临界值），空 cell 重抽样被拒并报告比例（§9.2）。

**工具会拒绝**的情况（不是警告，是 exit 2）：

- 任一臂与 `--expected-data` 的 `row_id` 集合不一致（缺行、多行、重复行）；
- 某臂整段长度缺失；
- 某臂与其他臂的 task/length 不一致。

理由：**部分面板报出来的数看起来就是结果**。计划 §5.4/§6.1 要的是身份锁定，
不是"能跑就行"。

未提供 `--native` / `--strict` / `--qa` 时，对应守门**不评估**，verdict 落到
`INCOMPLETE_GUARDS`，**不会**默认通过（见 §0.4）。

---

## 4 · 目录内容

| 文件 | 作用 |
|---|---|
| `operators.py` | 60 个配置的 CPU 参考编译器；`selftest` / `export` / `list` |
| `directions.json` | 从计划 §4 **机械解析**的 20 张卡（非手抄，避免转录漂移） |
| `candidates.csv` | 60 项简表，`execution_authorized` 默认 false |
| `parse_plan.py` | 解析器 + `--check` 防漂移 |
| `budget.py` | §8 的预算算术、`--fit-candidates`（只按计时，不读成绩） |
| `paired_report.py` | 配对 macro、来源 cluster bootstrap、§6.6 功率算术 |
| `upgrade_report.py` | §0.2 三 contrast 与 §6.5 判定 |
| `test_tools.py` | 51 条拒绝测试（人工 fixture） |
| `reference/` | 解析原生频率生成的数组 + 60 项 manifest |
| `cpu_test.json` | 最近一次 `selftest` 输出 |
| `tool_tests.json` | 最近一次 `test_tools.py` 输出 |

---

## 5 · 本包**没有**提供（计划 §9.1 同款声明）

GPU 模型加载器；RULER 数据下载/生成器；20 种 scope 的 GPU 集成；
实际吞吐与真实性能；全历史仓库同构排重；跨模型验证。

另外本包**尚未**生成 `static_identity_audit.json`（计划 §9.1 列为"新 18 静态表与旧归档
数组的逐位重复检查"）。它需要**新 18 静态表**——而"新 18 表"来自哪一轮配置，计划没有指明；
`candidates.csv` 里也没有这个集合。在这一点澄清之前不生成该文件，好过生成一个来源不明的
对照表。**这是待澄清项，不是已完成项。**

---

## 6 · 一处必须说清的**未验证**假设

D03c 的"中段"范围：计划写"中段整体循环 hop=round(log(s)/(logθ/K))=7"。hop 验算为 7 ✓，
但**"中段"的槽区间计划没有写死**。本实现取 `[19, 34]`（与 D03a 的"中段 19–34"保持一致），
16 槽、gcd(16,7)=1 构成单循环。若作者本意是别的区间（如 `[18,35]` 全带），
D03c 就是另一个算子。**这一条在执行前需要作者确认。**

同理 D03b 的"平台 35–37 与 38–40"按字面取 `[35,37]` 与 `[38,40]` 两块互换。

---

# 附：2026-09-11 作者修正之后的增补

原计划的实验纪律保留；**候选集与论文判据重构**。详见
`priority_correction_20260911.json`（逐方向判决）与 `execution_order.json`。

## A · 判据缺陷已修（F1 / F2）

**F1（严重）**：`G_new>0` 且 `C_upgrade>=0` **不蕴含** `A(M)>A(YARN)`。
精确反例：`A(YARN)=10, A(MR)=8, A(M)=9` ⟹ `G_new=+1>0`、`C_upgrade=9-16+10=+3>=0`，
但 M 连 YaRN 都没打过；加入 BM 后洞更宽。原计划只在 H 里要求"报告"BM。

⟹ binding gate 改为 `LCB[A(M) − max(A(MR),A(YARN),A(BM))] > 0`。
反例已固化为测试 `test_baseline_hole`（断言它必须判 FAIL、且 `hero_result=False`）。

**F2**：Hero 门槛 3pp → **5pp**。3pp 记为小改进（`thresholds.record`），
只有 ≥5pp（对最好 baseline）才值得消耗 V/H 预算。

## B · 候选集收缩

`plan_arms.py` 读入修正并产出 `candidates_corrected.csv`（60 行带 tier 与 RUN/HOLD/DROP）
与 `arms_llama3.csv`（5 控制 + 16 候选 = 21 臂）。

```
60 configs -> RUN 19 / HOLD 5 / DROP 36
Llama-3-8B: 5 controls + 16 candidates = 21 arms
```

## C · 接线时发现的三个"读起来对、跑起来挂"缺陷

这三条都会**静默**给出错误实验，已在本包内全部规避：

| # | 缺陷 | 证据 |
|---|---|---|
| 1 | `--m-file` 是普通 `store` 动作，`--m-file A --m-file B` **只保留 B** | argparse 实测 → `'B_1p30'`；分号写法 → `'A_1p13;B_1p30'`。**原就绪清单的 B 命令正是这个写法，会让 1.13 臂悄悄不跑** |
| 2 | `--dry-run` 在 `olmo_beta.py:243` 就 return，而 `--m-file` 块在 **314** 行 | dry-run **从不触及**注入路径 ⟹ 预注册里"dry-run 通过"对 m-file **无效**。`phase1.py preflight` 因此改为校验 argv 本身 |
| 3 | 文档里的 `PYTHONPATH` 不可导入 | `$D/repoharness` 不存在，且该根无 `scripts/` 包。可用根是 `.../olmo_fast_screen_20260908/code`（`scripts.*`）+ `.../nongeometric_screen_20260909/code`（`experiments.*`） |

**另外，本包自己的 preflight 也被查出同类缺陷**：`nvidia-smi -L \| wc -l` 与
`nvidia-smi --query-gpu=... \| grep -c .` 在无卡实例上**都返回 1**，因为
"No devices were found" 走的是 stdout（rc=6）——即把"没卡"读成"有卡"。
改用 `torch.cuda.device_count()`，因为那才是运行器真正会看到的答案。

## D · 当前真实阻塞（`phase1.py preflight` 输出）

```
REFUSING: 2 prerequisite(s) unmet
  - cuda devices (torch): got 0, want >=1              # 平台无卡模式
  - .../qwen25_1p5b_32k/model.safetensors: MISSING      # A 臂的模型
```

A 臂（现列第一优先级）需要 `Qwen/Qwen2.5-1.5B-Instruct` 的
`model.safetensors`，期望字节数 **3087467144**。该权重在 9/11 从归档恢复时静默丢失，
全盘无副本。上卡前须经镜像重下，落盘后**按字节数核验**（勿信返回码）。

## E · 新增文件

| 文件 | 作用 |
|---|---|
| `priority_correction_20260911.json` | 20 方向逐项判决 + 执行顺序 + 全阴叙事 |
| `plan_arms.py` | 应用修正，产出收缩后的臂集（带校验，修正与计划不一致即拒） |
| `candidates_corrected.csv` | 60 行：tier + RUN/HOLD/DROP |
| `arms_llama3.csv` | 5 控制 + 16 候选 |
| `execution_order.json` | phase 1 → phase 2 → phase 3 |
| `phase1.py` | A–E 臂注册表 + `commands` / `preflight` / `launch` |
| `llama_runner.py` | Llama-3-8B GPU 运行器（§9.1 自陈缺失的那部分） |

## F · Llama 运行器的两条硬约定

1. **配对布局是 `(i, i+K)`，不是相邻配对。** Llama 的 `rotate_half` 把 head 对半分再配对；
   配错会得到一个**仍然能跑、仍然输出通顺文本**的不同算子。`operators.apply_rotation`
   把它做成显式参数，运行器传 `"half"`，并有 CPU 测试证明两种布局确实不同算子。
2. **`sum_m` 对 D04/D05 无定义**（负频率、零频率使 `log(ω/ν)` 无定义）。
   运行器现在报 `"m_coordinate_defined": false` 并给出原因，而不是把 `nan`/`inf`
   当作一个数往下传。

**状态：运行器未在 GPU 上执行过。** 算子构造、cache 形状、配对布局与旋转代数由
CPU 测试覆盖（`test_tools.py::test_rotation`）；模型集成部分在拿到卡之前**未经测试**。
