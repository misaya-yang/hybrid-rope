# digest failure-records

任务：为统一理论（unify_20260910）整理项目内全部失败复盘文档中"已判死的推理跳接 / 永久撤回的强断言 / 不许重访的死区"，形成可机检否决清单。本文基于对 7 份失败复盘材料的全文直读（非转录提取），日期 2026-09-10。

铁律回显：不得把代理指标说成能力结果；不得把未测写成否证；每个结论标注证据等级 [已验证]/[部分证据]/[假设]；引用给出路径与行号/章节。

---

## 1. 来源清单

| 文件（绝对路径均在 `/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/paper_project/hybrid-rope/` 下） | 行数 | 字节 | 文件 mtime | 本轮读取方式 |
| --- | ---: | ---: | --- | --- |
| `docs/research/ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md`（下称 SYNTHESIS） | 248 | 24,928 | 9月7日 23:21 | 全文 Read |
| `docs/research/ROPE_RESEARCH_FAILURE_REVIEW_20260907.md`（下称 REVIEW-0907） | 273 | 25,347 | 9月7日 22:49 | 全文 Read |
| `docs/research/ROPE_OVERNIGHT_EXPERIMENT_REVIEW_20260908.md`（下称 OVERNIGHT） | 185 | 12,882 | 9月7日 21:16 | 全文 Read |
| `docs/research/ROPE_OVERNIGHT_EXPERIMENT_LEDGER_20260908.json`（下称 LEDGER） | 909 | 36,386 | 9月7日 21:16 | 全文 Read |
| `docs/research/COSH_REDESIGN_EVIDENCE_REVIEW.md`（下称 COSH-REVIEW） | 183 | 15,563 | 9月10日 07:50 | 全文 Read |
| `docs/research/PARALLEL_NONGEOMETRIC_20X10_AUDIT_20260910.md`（下称 AUDIT-0910） | 127 | 17,596 | 9月10日 07:50 | 全文 Read |
| `docs/research/ROPE_EXTRAPOLATION_FAILURE_AND_LIMITS_20260910.md`（下称 EXTRAP-0910） | 219 | 11,976 | 9月10日 07:50 | 全文 Read |

被上述文档引用、但本轮**未读**的关联材料（结论追溯时如需更细证据须另行读取）：
- `docs/research/ROPE_LOCAL_FAILURE_EVIDENCE_20260908.json`（逐源证据、行号、复算记录）
- `docs/research/ROPE_SOURCE_COVERAGE_20260908.json`（来源覆盖 SHA 与阅读级别）
- `docs/research/ROPE_FIXED_POSITION_VISIBILITY_PROTOCOL_20260908.md`（L/P 可见性协议）
- `docs/research/ROPE_SCALE_TRANSPORT_PILOT_20260907.md`、`ROPE_SCALE_TRANSPORT_REVIEW_20260907.md`、`ROPE_SCALE_TRANSPORT_ASSUMPTIONS_20260907.json`
- `docs/research/ROPE_CARRIER_REMOVAL_PILOT_20260907.md`、`ROPE_NATIVE_SECTOR_CARRIER_20260907.md`
- `docs/research/ROPE_QWEN15_FULL_LAG_P2_RESULT_20260907.json`、`ROPE_QWEN15_MINIMAL_MECHANISM_20260907.md`
- `docs/research/ROPE_FREQUENCY_UNIFIED_PLAN_20260907.md`、`ROPE_PRO_DECISION_REQUEST_20260907.md`
- `paper-2027/research/foundations/FULL_ROPE_SPECTRAL_BASIS_AND_COADAPTATION_REPORT_20260819.md`
- `paper-2027/research/attention-aware-retrofit/results/coupling-transfer/CPU_LOW_DIM_COUPLING_LAW_20260901.md`、`LOW_DIM_COUPLING_GPU_RESULT_20260901.md`、`SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`
- `docs/exp/2026-03/2026-03-11_test3_broadband_r2_validation.md`、`2026-03-09_phase16_formula_optimality_sweep_results.md`、`2026-03-03_phase9f_50pct_checkpoint_report.md`
- `paper-2027/appendix/a1_proofs.tex`、`a5_identification.tex`；`paper-2027/research/attention-aware-retrofit/results/PHASE_ALLOCATION_M4_EXTENDED_RESULT_20260824.md`
- `.agents/worker_remediation_1/report.md`（718 行整改报告，SHA256 `cf02670b...eaa1139`，SYNTHESIS §6 对其逐条裁决）

---

## 2. 任务时间线

### T1. 2026-03：早期理论期（Phase9F / Phase16 / broadband）
- **目标**：为 Cosh 形状建立"碰撞核近似 + 最优律"理论。
- **方案**：`p(Δ)=1/(Δ log L)` 先验上的 cosine-only collision kernel，`αδ+β min(φ,ψ)` 近似；变分解 `ρ''=(β/α)ρ` 得 Cosh；Phase16 跑 99 个训练配置验证 τ 公式 near-optimal。
- **结果**：**失败（后被重分析否定）**。
  - broadband 检查记录：先验被反向扫描挑选——24,000 个配置中找到 886 个 R²_mid>0.99（COSH-REVIEW §1.1，行 43）。
  - Phase16 本轮全 99 个 `result.json` 重算：公式−Geo 配置均值 7/9 胜、配对 18/27、平均 NLL 差 −0.01331075；公式−pilot 邻点（另 2 seed）3/9 胜、配对 8/18、+0.02210473（COSH-REVIEW §2，行 75-80）。"near-optimal law"排名混用 stage 复合评分与不同 seed 数，7月24日重分析已否定。
  - surrogate √(β/α)=6.244/5.704 vs 部署规则 d/√L=1.414/1.000，比值 4.42/5.70 —— τ 的解析缺口有直接反证（COSH-REVIEW §2，行 66-71）。
- Phase9F Hybrid：8K 完整生成 AR exact 双方均 0%（50%/62.5% 是 retrieval 口径）；无同条件纯 Cosh 臂（COSH-REVIEW §3.1，行 97-100）。

### T2. 2026-07：适配与生成评测期
- QuALITY 评分/长度 bug（选项首字符、长输入裁到 8K）；修后接近随机又被归因"容量上限"——两头都不成立（SYNTHESIS §2 行 41）。
- rank≈2000 的目标块仍全错；50-step 检索 micro-tune 把训练 loss 降得很低仍未修复真实 16K S-NIAH/KV exact（SYNTHESIS §2 行 46；COSH-REVIEW §4 行 123）。
- qa16k 三臂 303 题：Base-Native F1 23.09% / Native-LoRA 21.10% / EVQ-LoRA 11.26%；exact 33/25/4；同一批 adapter 的 PPL 却显示 EVQ 16K 24.068 vs Native 108.958 —— PPL 改善而能力受损的直接案例 [已验证]（COSH-REVIEW §4，行 114-121）。
- 7 月原始重分析否定 Phase16 排名（见 T1）。

### T3. 2026-08-31 至 09-01：旧 log-p2 与 C2 链
- 旧 Qwen2.5-1.5B log-p2：64K/128K core4 0.7000/0.5875；槽 1/18 有频率交叉尖峰（legacy 数值构造）；去除孤立非单调尖峰后为混合结果 [部分证据]（REVIEW-0907 行 156-160、189-198）。
- C2 拟合边界：OLMo 约 11.2865/7.7814 turns、Qwen 约 10.7764/7.4297 turns；两模型槽 19 的 C2 movement=0 [已验证，标量复算]（SYNTHESIS §6 行 150）。

### T4. 2026-09-07 白天：尺度传输提案（scale transport）
- **目标**：用响应统计给每槽位定压缩量。
- **方案**：分位匹配 + 合法投影，生成 64 槽新表，5 点 guard。
- **结果**：**失败（guard 拦截）**。只有 λ=0 通过；提案实际同时做了三件事：槽 24-39 全部比 MrPro 慢（0.28205–0.85724 倍）、槽 23→24 log 间距 0.22493→1.48037、投影产生 8 对相邻等频只剩 56 个不同频率；j=24 累计压缩指数 0.0065→0.912；短 QA F1 0.425 vs Native 0.498465（REVIEW-0907 行 20-25、121-125）。
- **复盘裁决**：不能把该提案失败理解为"适当修改中段已经失败"；整个 Pro 理论未被否定，guard 也不等于真实能力（行 22）。
- 同日独立相位能量算例：两算例能量同为 (4,1)，统一换频输出导数 0 vs 4（有限差分 3.99999999992）→ χ_i 不能当逐频率分配依据 [已验证，代数算例]（REVIEW-0907 行 142-144）。
- 工具修复：新 NumPy 入口 rotate_half 反号，独立绝对旋转修正后偏差 4.09798→1.44e−15；旧自检对同一错误函数做有限差分仍通过 —— 自洽测试不能替代独立参照 [已验证]（REVIEW-0907 行 210-217）。
- 真实缓存：12 个 32K 块 replay 相对误差 .000142–.000328；相干/独立能量比中段后 .918–2.900、尾段 1.111–8.728；Mr 处单点 Jacobian 外推旧 p2 相对误差 71.15–468.12 [已验证]（REVIEW-0907 行 222-236）。

### T5. 2026-09-07 夜至 09-08：整夜 GPU 实验（本 digest 的账本主体）
- 34 个监督器作业、总墙钟 24,924.96 秒（6.92 小时），86 个原始文件哈希入账（LEDGER `known_job_seconds`，行 416）。
- **P2Middle**（旧 1.5B p2 中段搬到 Qwen3B）：44 次生成；长 UUID 0 vs Mr 12.5；VT 65 vs 62.5；短能力退化。裁决：换模型+换 gain+改表组成后不能视为同一已验证方法；运行前没有解释 1.5B 有效分配为何应改善 3B——设计缺口被承认 [已验证，混合结果]（OVERNIGHT §1、§4.2；REVIEW-0907 行 240-244）。
- **原始 Carrier**：20 条开发比较长 UUID/VT 均 0；50 条数字检索 32、EOS 14/50 结束。裁决：背景二次目标改善≠能力改善；后续矩阵停止 [已验证，开发面板]（OVERNIGHT §1）。
- **Native-sector Carrier**：9 任务×50 条部分矩阵：single1=100、multikey2=30、multikey3=6、VT=60.8、CWE=7.2、FWE=70、single2=96、single3=98、multikey1=92；450 行均值 62.22。裁决：**不能**与 MrPro 论文完整 13 项 53.2 判胜；扩展消耗可核对作业时间 63.8%，是本夜最大资源失误 [部分证据，不可比]（OVERNIGHT §1 行 33-36）。
- **FullLagP2**（旧 p2 的数值修复：2048 取整质心→全部 32768 lag；SHA `ecd0c280...d461096b`；与旧 p2 中段最大相对差 0.01243%；本方 gain `1+.074 ln4` vs 官方 Mr `1+.1 ln4`）：
  - 1.5B 64K（每任务 8 条）：MK 37.50/12.50/25.00（P2/Mr/同gainMr），VT 87.50/82.50/77.50，FWE 70.83/45.83/45.83；对官方 Mr 9胜13平2负，聚合 5胜3平0负 [已验证，开发面板]。
  - 1.5B 128K：MK 双方 0（地板）；FWE 2胜4平2负；VT 4胜3平1负但存在单行 5/5→1/5 [已验证，混合]。
  - 3B 64K（每任务 4 条）：1胜9平2负；VT 本方 0/4 EOS；无同 gain 控制 [已验证，样本小]。
  - 裁决：局部收益真实存在，无跨长度/跨 checkpoint 全面优势；数值修复不自动晋级。
- **LoRA**（1.5B 64K，r16/α16 七类模块，128 步计划、8388608 token 计划）：smoke 2 步通过（每步≈32s、峰值 20.7GB）；正式训练 59/128 步、3,866,624 token 后操作员 SIGTERM（`exit=-15`）；只保存最终 adapter 的设计导致无最终 adapter。裁决：**不能判 LoRA 有效或无效** [未执行→不可判]（OVERNIGHT §3；LEDGER 行 307-318、877-898）。
- **工程失败**：CARRIER_NATIVE_MEANS_01 输入 hash 类型错误（模型前向之前），已保留并修复（LEDGER 行 30-41）；1.5B 64K 参照臂因远端 SSH 关闭未自动接续，控制器改为脱离 SSH 生命周期（OVERNIGHT §6）。

### T6. 2026-09-08：三份复盘 + 整改报告逐条裁决
- REVIEW-0907 定调：循环根因是"从局部量到能力的跳接"反复复发且失效结论不退出；作者要求研究迭代只能产出三类东西（见 §6 引用）。
- SYNTHESIS：10 类复发跳接谱系表（§2）+ 实质修复（ALS `inf<=inf` 虚假收敛、LoRA 容量错误界撤回、`install_signed` 频率降序 guard 放宽）+ 对 worker_remediation_1 报告 15 行逐条裁决（§6）+ 非可识别性只保留可证明版本（三个扩大撤回）。
- OVERNIGHT：本夜数字与六条"为什么再次进入局部循环"。

### T7. 2026-09-09/10：Cosh 改版复核
- COSH-REVIEW（基于 09_09 分支、起始 HEAD `fe10272`）：CPU 重算 Phase16 99 文件 + 三臂 303 题 QA + NumPy 理论脚本；结论——当前已有结果证明"分配"这个变量值得研究，但尚未建立能可靠选出更优分配的理论或算法；Cosh 的四个命题（闭式可解/碰撞减小/PPL 改善/任务能力改善）没有被连成因果链；M4 中 min-eigenvalue 5.67675 < Cosh 5.71740 < Geo 5.78930 是单 seed 单格线索，不升格 [部分证据]；deformation-matched exponential 与参考 Cosh 差 +0.00074 NLL 跨零；LeRoPE 构成实质近邻 prior art。

### T8. 2026-09-09/10：nongeometric 平行 20×10 筛选与审计
- 远端 `/root/autodl-tmp/nongeometric_screen_20260909`：
  - E1 s28_less（槽28少压缩）：历史 36 行全量，32K 不变，128K 78.125%→83.333%，两行改善无回归 [部分证据，开发面板]。
  - E1 s29_more：32K +8.333pp 完全来自一行 QA；128K −0.208pp（VT 改善与 multiquery 回归几乎抵消）[部分证据]。
  - E3 BM gain0.074：32K +12.778pp / 128K −8.125pp——表×gain 打包，非 gain 因果 [部分证据]。
  - E7 local projection：32K +2.778 / 128K −9.514 [部分证据，判负]。
  - E8 zero slot51：前 12 行 128K −13.889pp，multiquery/FWE 回归 [部分证据，判负]。
- AUDIT-0910 核心纠正：E7 的 161 倍差异在 BF16 实现（NMSE 线性预测 7.9366844e-8 ≈ ideal 7.9364028e-8 ≈ FP32 7.9325473e-8，BF16 1.2807392e-5），不是局部线性化失败；gain 只乘 logit（gain²）改变温度，不恢复相位弧；弦距离 `D²(Δ;ν)=4Σw_j sin²(Δν_j/2)` 对压缩非单调，s28 成功改动在 5 个测试距离中 3 个**降低**分离度；multikey 成功 margin 序列 (−2.125, −1.125, −1.000, +0.250) 近似可加（prefix +1.125、reading +1.000、交互仅 +0.250），"隐藏态强交互"叙事被削弱 [已验证，同前缀可检].

### T9. 2026-09-10：外推失败机制与极限文档（EXTRAP-0910）
- 为统一理论准备的机制文档：正交性⇒外推不内在地造成范数无界；整表全位移平移在精确算术下不变（BF16 可破坏，留作数值诊断）；三频带不是定理；低频覆盖圆不够；`G_W(δ)` 联合有限窗相关量（sinc 型）；softmax 竞争的对数稀释恒等式；精确局部保持钉死静态算子（ν=ω mod 2π）；鸽笼相位码界在 K=64 极松；E1 跨模型迁移混合：Qwen7B 32K tie / 128K −1.667pp，OLMo 4K −3.333 / 16K +3.819pp [部分证据]。
- FullLagP2 与 E1 在槽 28/29 的 m 值一致表（28：MrPro .098039 / P2 .074131 / E1 .065359；29：MrPro .137255 / P2 .221149 / E1 .183007）——标记为 hypothesis-generating agreement，不是独立验证。

---

## 3. 理论主张表

| # | 主张 | 证据等级 | 出处 | 后续是否被纠正/推翻 |
| --- | --- | --- | --- | --- |
| C1 | 同 multiset 频率置换会严重掉分 ⇒ 无序频谱不充分决定行为；槽位身份是真变量 | [已验证] | SYNTHESIS §2/§5（行 100-101） | 未被推翻；但其扩大（"每槽有唯一语义标签""频率必须单调"）被明确禁止 |
| C2 | 粗 ramp 与细 derived 表在本地 FineWeb 只差 ≈.000511/.000560 NLL；同端点 geometric 在 16K 比 derived 差 4.455995 ⇒ 粗分配结构有用 | [已验证，NLL 口径] | SYNTHESIS §4（行 77-79） | 保留；不推广为精细曲线或复杂校准必需 |
| C3 | 简单分配规则（MrPro）无需拟合 checkpoint 功能即可产生强结果 | [已验证，公开+本地] | SYNTHESIS §5 行 100；REVIEW-0907 作者简单性原则 | 保留为设计原则；"必须依靠复杂 checkpoint 机制才能解释成功"的方案应先受质疑 |
| C4 | 非可识别性（可证明版）：固定权重/输入/gain/评分下，Phi(T_A)=Phi(T_B) 而 Y(T_A)≠Y(T_B) ⇒ 不存在只读 Phi 精确解释这两结果的单值函数 | [已验证，直接反证] | SYNTHESIS §6（行 170-175） | 三个扩大版本已撤回（见 V-D1） |
| C5 | FullLagP2 在 Qwen2.5-1.5B 64K 三任务局部优于 MrPro（MK +25pp、配对 3胜4平1负；聚合对两参照 5胜3平0负） | [已验证，开发面板、每任务8条] | OVERNIGHT §2；LEDGER 各臂 | 边界固定：128K MK 两臂均 0；3B 有取舍；不得写成 SOTA 或跨长度普遍优势 |
| C6 | E1 s28_less 使 128K 78.125%→83.333%（两行改善、无回归） | [部分证据，开发面板] | AUDIT-0910 Verified 表 | 泛化未建立：新 seed 面板与 cross-model 冻结迁移须先完成 |
| C7 | s28 成功机制（当前最强解释）：prefix 形成与后续读取共同把一个竞争答案 margin 推过 greedy 边界，且贡献近似可加 | [已验证，单案例同前缀 margin 可检] | AUDIT-0910 §4 | 保留为"待解释的正例"，不得升为通用交互机制 |
| C8 | 精确保持位移 1 处全部 Q/K 点积 ⇒ ν=ω (mod 2π)，非平凡静态重分配不可能同时精确保持所有局部算子 | [已验证，数学恒等] | EXTRAP-0910 §6 | 保留；但不得反读成"冷换表不可能有益"（BM/P2/E1 条件收益为反权重） |
| C9 | 有限维有界相位码在无上界长度处无一致分离（鸽笼，差 ≤2π√K/M） | [已验证，数学] | EXTRAP-0910 §6 | 界在 K=64 极松，不得建立 128K/1M 实践天花板 |
| C10 | 干扰项 ×4 需 ≈log4 额外 logit 优势；有界 logit 读出的质量上限 `p*≤[1+(N−1)e^{−C}]^{−1}` | [已验证，条件恒等式] | EXTRAP-0910 §5 | 只是条件 softmax 极限，不证明整个模型或聚合任务必败 |
| C11 | checkpoint 已把功能分配进 rotary subspace（不同槽位有功能分工） | [假设，研究输入] | REVIEW-0907 行 271-273；SYNTHESIS §4（p-RoPE 来自从零训练不可外推） | 未验证；禁止由它直接推出最优分配 |
| C12 | DCA 证明单个绝对位置时钟不是唯一算子类（注意力与位置联合处理有成功先例） | [已验证，外部文献] | SYNTHESIS §4（行 86-88） | 重命名 DCA 不构成新方法 |
| C13 | 频率与槽位结合是真变量，但"增加旋转分量"需区分提高已有槽频/更多槽覆盖所需尺度/改非旋转内容通道 | [部分证据] | REVIEW-0907 行 101-110 | 保留为研究变量定义 |
| C14 | ALS 虚假收敛（previous=inf 时 inf<=inf）与 NumPy rotate_half 反号是两个真实 bug，均已修复且有独立参照回归 | [已验证] | SYNTHESIS §3；REVIEW-0907 行 210-217 | 修复不算方法收益 |
| C15 | 相干项可抵消也可放大：跨 head 共享时逐 head 能量和 5.09866 而共享响应能量精确为 0；真实缓存比值中段 .918–2.900、尾段 1.111–8.728 | [已验证，两篇暴露自然文本] | REVIEW-0907 行 182、228-231 | 不预设"修复相干性统一减少损伤" |
| C16 | E10（混合核）前 12 行任务分平 MrPro 而 NLL 改善；E9 相对时钟在边界连续（斜率折点≠不连续） | [部分证据] | EXTRAP-0910 §7.5；AUDIT-0910 §5 | 同钟扩窗 NLL 控制排队中，先不裁决 |

---

## 4. 失败机制清单 = 可机检否决清单（VETO LIST）

**用法**：统一理论的每一段推理都对照本表。检查规则：若理论文本中出现"被禁形式"一栏所描述的模式（或其改名/换参数/换统计量包装），即判违规，必须给出（a）证据等级标注、（b）出处、（c）被撤回版本的精确弱化形式，否则不得写入。证据等级词表：`[已验证]`＝有可检索原始数据/数组/恒等式；`[部分证据]`＝开发面板、小样本、单 seed 或口径受限；`[假设]`＝未被匹配证据唯一支持。

### 4.0 总根因（一句话）
SYNTHESIS §2（行 31-34）："循环的根因是推理链反复失效，且失效结论没有退出……已有证据反复否定的是从某个局部量到实际能力的跳接；后续却把该局部量改名、增加自由度，或把错误前提重新当起点。" 整改不是再猜频率函数。

### A 组：代理指标 ⇒ 能力结果的跳接（9 条）

| ID | 被禁形式（出现即违规） | 裁决与保留形式 | 出处 |
| --- | --- | --- | --- |
| V-A1 | 用低崩溃比/低 PPL 比值直接判"更稳定、PASSED"⇒ 能力收益 | 低崩溃比可以来自短端已经坏掉（sigmoid 短/长 PPL 37648/62474 旧例）。只能报数，不得判收益 | SYNTHESIS §2 行 38 |
| V-A2 | Gram、曲率、重构残差、phase risk、覆盖、平滑性变好 ⇒ 生成变好 | 已有直接反例；18样本64维行为梯度路线已失败，不得重新包装成"功能需求恢复" | SYNTHESIS §2 行 44；REVIEW-0907 行 178 |
| V-A3 | 背景响应（二次目标/能量）下降 ⇒ 能力提高 | 原 Carrier 背景目标改善与长 UUID/VT 全 0 同时发生。已列入"不保留为结论"清单 | OVERNIGHT §1/§5；SYNTHESIS §6 行 151 |
| V-A4 | NLL、源依赖、target block rank 提高 ⇒ 正确输出 | rank≈2000 仍全错；Far-pass 多种适配有完整生成地板；E/B、全词表竞争者、答案轨迹 margin、write decomposition 均已测过，不得再当作新代理复活 | SYNTHESIS §2 行 46 |
| V-A5 | PPL/条件困惑度改善当作指令遵循、目标绑定、自由生成成功的替代 | qa16k：EVQ-LoRA PPL 最好（32K 127.9 vs 991.5）但 F1 11.26%、exact 4/303 最差。PPL 度量的是它所测文本，不是能力 | COSH-REVIEW §4 |
| V-A6 | 冻结态选择器分数/局部导数外推 ⇒ 全模型任务改善 | E8 强冻结态选择分数对应 128K −13.889pp；Mr 处单点 Jacobian 预测旧 p2 有限变化相对误差 71.15–468.12。局部导数只能在明确 trust region 内用，不得接成能力优化器 | AUDIT-0910 Verified 表；REVIEW-0907 行 232-236 |
| V-A7 | attention top-1 / attention mass ≥0.5 ⇒ 生成正确；用它解释长度断崖 | top-1 只需目标 logit 最大（权重 .4/.3/.3 即反例）；attention top-1 ≠ 生成正确；未测得通用 S_dilution | SYNTHESIS §6 行 155 |
| V-A8 | LoRA 训练 loss / smoke 通过 ⇒ 生成改善或不遗忘 | 59/128 收尾中断、无最终 adapter、无训练后评测 ⇒ 既不能判有效也不能判无效（"未测≠否证"双约束） | OVERNIGHT §3；LEDGER 行 307-318 |
| V-A9 | 数值修复、数组合法性、hash 一致、guard 取消 ⇒ 效果提升可晋级 | 数值修复曾降低 128K 能力；FullLag 修复只证明采样混叠来源（99.36% 尖峰重现），未做同输入旧原表对照；"不宣称取消 guard 就是效果提升" | SYNTHESIS §2 行 43、§3 行 69；OVERNIGHT §2 |

### B 组：单因素归因跳接（6 条）

| ID | 被禁形式 | 裁决与保留形式 | 出处 |
| --- | --- | --- | --- |
| V-B1 | 一次候选失败 ⇒ 认定训练量/架构（或任何单一变量）为根因，再找新公式 | Phase18-23 旧例：长度、训练量、数据与架构同时变化不能单因素归因。tau=1 时最高频率仍约 .72967 rad/token，"高频几乎没有"与数组不符 | SYNTHESIS §2 行 40 |
| V-B2 | 修 bug 后低分 ⇒ "容量是唯一瓶颈" | QuALITY 旧例双向都不许：早期评分/长度错误不算方法负结果；修后低分也不识别容量 | SYNTHESIS §2 行 41 |
| V-B3 | 一个 routing/oracle 控制无效 ⇒ attention 不是瓶颈 / 关闭所有注意力处理 | 原代码仍保留其他块并用普通 QK softmax——该控制的无效性不外推 | SYNTHESIS §2 行 47 |
| V-B4 | 早压缩⇒128K 相位饱和；头/层数差⇒3B 错配；最少 orbit⇒频率缺口失败 | 数组差异、架构差异与失败各有记录，但因果解释均未隔离，全部标 hypothesis；不从模型名字、phase risk 或 D\* 追溯认定根因 | SYNTHESIS §6 行 147 |
| V-B5 | 把 BM(gain0.074) 的 32K +12.778pp 记作 gain 的独立因果效应 | 是表×gain 打包与长度取舍；须先补 MrPro×gain0.074 臂完成 2×2 factorial，在同表内评 gain | AUDIT-0910 Verified 表、§2 |
| V-B6 | 把 s29 的 32K +8.333pp 说成广泛短上下文恢复 | 增益全部来自一行 QA | AUDIT-0910 Verified 表 |

### C 组：身份 / 口径混淆（6 条）

| ID | 被禁形式 | 裁决与保留形式 | 出处 |
| --- | --- | --- | --- |
| V-C1 | 用变体（如 sqrt 增益 Y2）的零分证明"官方方法上限" | 官方 YaRN cos/sin 幅度 a=1+.1ln(s)、logits 乘 a²；Y2 是变体，两身份不得互换 | SYNTHESIS §2 行 42 |
| V-C2 | 把 arithmetic 旧表 / log-p2 / FullLag 当同一方法的成绩互相引用 | 不同身份；8 月已发现 stride16 高频混叠、log-p2 用 2048 整点格，数值修复历史不能互相覆盖 | SYNTHESIS §2 行 43 |
| V-C3 | 把 Carrier 的 YaRN 外底座与 Native-sector 的 MrPro 底座串成一个干预 | 两构造底座不同，不得拼接；负频率是合法旋转，不是 Carrier 失败的单因素根因 | SYNTHESIS §6 行 151 |
| V-C4 | 部分任务集均值（9 项 450 行 62.22）对比论文完整 13 项 macro（53.2）判胜 | 口径不匹配即无效；"部分 RULER 均值超过论文完整 macro"已列入不保留清单 | OVERNIGHT §1/§5 |
| V-C5 | 把 s2/s4/s8 的门槛与分数拼成"同一物理长度断崖" | 配置 scale、实际评测长度、gain 版本混用；干净反例只有原 Haar/MaxEnt 8K→16K 对照 | SYNTHESIS §6 行 156 |
| V-C6 | 把 EOS 口径统计（14/50 次结束）改写成"14/50 失败"或反之 | EOS 与主分分开报告 | SYNTHESIS §6 行 151；OVERNIGHT §2 |

### D 组：已永久撤回的强断言 / 定理扩大（16 条）

| ID | 被撤回的强断言 | 撤回依据 / 允许的最小保留 | 出处 |
| --- | --- | --- | --- |
| V-D1 | 非可识别性定理的三个扩大：(a) "任何 Phi"（含完整有序表本身）；(b) "Phi 不同、Y 相近"违反 Y=f(Phi)；(c) 任意 epsilon 趋零仍失败 | 函数可以多对一；`Phi(T)=T` 时相同 Phi 即同一表；一次非零 MAE 不证任意小扰动失败；greedy 需另论 margin/平局。只保留 4.0/C4 的直接反证版 | SYNTHESIS §6 行 177-188 |
| V-D2 | 由槽位范数不均推出任务对 attention 敏感、推 3-nat 损失下界 | attention logit 不是词表 logit，不能套 LM 交叉熵的 p−y；未证有限置换使线性项均值归零 | SYNTHESIS §6 行 181 |
| V-D3 | LoRA 容量错误界：rank-r 更新除以头数；best-found 残差当"不可修复下界"；各向同性平均当最坏情形界 | 每头切片可各有 rank-r；一般 LoRA 可读旧 Q/K 未保留的 hidden 方向；相应代码说明与 owner 已加更正 | SYNTHESIS §3 行 60-63 |
| V-D4 | "全部低维条件永久淘汰"的二元淘汰矩阵与三 fiber 定理 | 具体统计量的反例只否定其声明的充分性/不变性；"缺少预测、未满足前提、未测量均不能填 VETO"。改用 §4.7 的 C1-C8 最小裁决表 | SYNTHESIS §6 行 152 |
| V-D5 | 测试全绿 = 定理及否证矩阵被证明 | `test_challenger_remediation_verification.py` 把 VETO 字典与 fiber 名字写死再检查，属重述结论；自洽测试不能替代独立参照 | SYNTHESIS §6 行 159；REVIEW-0907 行 212-215 |
| V-D6 | "Qwen 实际损失变化 −1.81 nats、Taylor 预测 −252.04 nats" | 脚本硬设 m_ref=.25、delta_m=.044109，未加载实际 C2/reference 表；算的是单个 cos 函数不是 LM 损失，单位不是 nats。只可作构造例 | SYNTHESIS §6 行 149 |
| V-D7 | slot19 残差占 81.2% ⇒ 该槽功能敏感度最高（"slot19 最敏感"） | 81.2% 是曲线拟合平方残差占比，非 Jacobian/内容模长/损失敏感度；未测 peak modulus。"slot19 最敏感""harmonic mismatch""prefill 污染""softmax 稀释导致断崖"整体降为 hypothesis，带错误数学推理的强版本直接撤回 | SYNTHESIS §6 行 148、§5 行 106-108 |
| V-D8 | R(s)=R0+ln(s) 的普遍坐标律 ⇒ support 反转机制 | 该坐标定义域仅相应正频率、非退化 support 域；B→sB 增量是 (K−1)ln(s)/K 不是 +ln(s)；坐标变化不推出胜负方向 | SYNTHESIS §6 行 158 |
| V-D9 | "cos 收敛半径有限"；路径积分=免费预测器；旋转差算子界直接界定 LM 损失 | cos 是整函数，问题是低阶截断误差；路径积分是微积分基本定理，实际求值仍需路径上整网梯度；greedy 分数不适用光滑积分 | SYNTHESIS §6 行 157 |
| V-D10 | 独立相位能量 χ_i=\|\|∂y/∂φ_i\|\|²（或其×距离²）直接推逐频率压缩量 β | 丢失 i≠k 相干项（两 key 算例：能量同 (4,1)，导数 0 vs 4）；应改用有符号共享频率响应；"用非负响应直接推压缩方向"是三种被禁偷懒替代之一 | REVIEW-0907 行 129-146、164 |
| V-D11 | CoPE 光滑幅度窗 ⇒ 光滑修改频率一定改善注意力 | 频率映射改变谱原子位置≠乘窗（Δ=0,C=1 时频率置 0 得 1、幅度置 0 得 0 直接反例）；"用光滑外观代表更好的过渡"被禁 | REVIEW-0907 行 148-154、164 |
| V-D12 | Cosh 是"真实 RoPE 应当采用"的被发现最优族 | 一旦选定常系数局部平方项+min-kernel，任何 α、β 拟合都返回 Cosh 族——由目标形式选出，不由拟合结果发现；"唯一近似是 broadband projection"只指标量模型内部代数步骤 | COSH-REVIEW §1.2 |
| V-D13 | surrogate 最优定理解析确定部署 τ | √(β/α) 与 d/√L 比值 4.42/5.70 直接反证；早期"near-optimal law"排名（混 stage 评分/混 seed）不得继续采用 | COSH-REVIEW §2 |
| V-D14 | 高拟合 R² ⇒ 真实需求服从该先验/条件 | 先验曾被反向挑选提高拟合（24,000 配置找 886 个 R²>0.99）；GPT-2 attention 统计不是目标模型的任务敏感距离分布 | COSH-REVIEW §1.1 |
| V-D15 | 三频带（high/mid/low）是数学定理 / 1-32-cycle 阈值证明三个普遍功能模块 | 三带是连续谱上不同渐近机制的实际过渡；middle band 不是第三种数学上distinct的信息种类 | EXTRAP-0910 §2 |
| V-D16 | 低频覆盖一个圆 ⇒ 更高频率安全；未访问弧比例解释 E1 条件成功 | 联合学习对象（相位向量×内容系数×竞争 key）的边际覆盖不够；E1 槽 28/29 在参考窗内已转 12.37/9.97 圈，无法用单圆未访问解释 | EXTRAP-0910 §3 |
| V-D17 | 有限维相码鸽笼界 / 标量 proxy 首零 / 静态障碍定理 ⇒ 实践天花板（128K、1M）或"冷换表不可能有益" | 界在 K=64 极松；外部定理（Base-of-RoPE Thm1、RoPE-Distinguishes）绑定随机 Q/K 模型与正则幅度假设；manuscript 移植障碍只否定全算子精确等价，BM/P2/E1 条件收益必须作为反权重保留 | EXTRAP-0910 §6 |
| V-D18 | 提高频率 ⇒ 周期性位置分离普遍增大；gain ⇒ 恢复相位弧；E7 ⇒ 局部线性化失败 | 弦距离对压缩非单调（Δν=2π 分量为 0、减半反而最大；s28 成功改动在 3/5 距离降低分离度）；gain 只乘 logit 改温度；E7 的 161× 在 BF16 实现层且 NMSE 比是平方范数比非幅值比 | AUDIT-0910 §1-§3 |
| V-D19 | 二元任务分数的 crossover 顺序 ⇒ 强非线性隐藏态交互机制 | multikey margin (−2.125,−1.125,−1.000,+0.250) 近似可加，交互仅 +0.250（BF16 度量）；先按 margin-跨阈值机制解释，非交互不可证但不得预设 | AUDIT-0910 §4 |
| V-D20 | 家族级关闭："300–500 点搜索失败 ⇒ 静态表家族在 +5pp 封顶"；"若干形状失败 ⇒ support 约束表家族关闭"；"慢频置零崩溃 ⇒ 证明弧恢复机制/那些维是死重"；"NLL 可加+任务差 ⇒ 否证逐槽可加" | 搜索结论只描述被测试域/目标/算法/预算；置零保留内容通道，collapse 只证明依赖被移除旋转；可加 logit 经阈值化可产生不可加 accuracy（multikey 即例）；对已知 5 个结果拟合指标是 retrospective calibration 不是验证 | AUDIT-0910 §4-§6 |
| V-D21 | "training-free + static + nongeometric frequencies 无人占据"的新颖性声明 | MrRoPE-Pro 本身逐维 radix 进度、中段累计指数非线性、非单一全局几何级数；YaRN 也是分段；学一个非几何表或再叠 YaRN 不够（LeRoPE 已逐频学习+重训+组合）。新颖性必须落到新选择规则/内容依赖/允许的非单调性/预测理论 | AUDIT-0910 Prior-art；COSH-REVIEW §7 |
| V-D22 | 把旧 p2 新提案尾部标为"继承已验证的本方优势"；把 44.8%（1.44759 倍）上限当普遍约束 | 旧 p2 有效结果不证明本方低频加速有效；该上限仅是几何可行范围的条件性计算；"用排序约定代表模型不可变约束"被禁（降序是约定不是定理，旧 p2 槽 1/18 交叉仍有效） | REVIEW-0907 行 160、197、164；SYNTHESIS §3 行 66-69 |

### E 组：死区（明确禁止重启的路线）

| ID | 死区 | 状态 | 出处 |
| --- | --- | --- | --- |
| V-E1 | 自动恢复旧 Cosh 搜索、对手微调、seed42 权重恢复 | "禁止自动恢复"——除非作者显式重新授权 | REVIEW-0907 行 48 |
| V-E2 | 重启 18 样本/64 自由度的 margin-gradient 能力优化路线 | 已失败，"不能据此重新启动"；共享频率响应工具只作局部诊断 | REVIEW-0907 行 178 |
| V-E3 | 把正确局部 Jacobian/Fisher 接成"下一个能力优化器" | 局部 Taylor/Fisher 只在正则性与 trust region 内保留；不作全局退休也不作全局优化目标 | REVIEW-0907 行 236；SYNTHESIS §6 行 206-209 |
| V-E4 | 用频率置换重新发现同一个 multiset 限制 | 限制已确立；保留槽位身份，不重复置换 | SYNTHESIS §5 行 101 |
| V-E5 | 在没有可区分预测时继续科学 GPU 工作；把"更稳定/更平滑/局部改善"当完成目标 | 迭代只能产出三类：有依据的具体解 / 可区分剩余解释的决定性预测 / 非可识别性证明+明确缺失测量量 | REVIEW-0907 行 266-269 |
| V-E6 | 把选择器 normalization 修复后的 layer 27 排名当有效次选（修复后第二候选层已是 32） | layer 27 是修复前的测量候选，不得作为 runner-up 复活；该修复也未重跑独立复核 | AUDIT-0910 行 127 |
| V-E7 | 本夜旧队列：整夜收尾后所有旧"下一步启动"文字失效 | 需新授权/新材料审查后才可启动 | OVERNIGHT §7 |
| V-E8 | 静态碰撞指标当外推机制解释（旧 full-rope 审计遗产） | 静态 collision ≠ extrapolation 机制；cos-only kernel 只是半个故事 | 与 MEMORY.md full-rope-collision-audit 一致；COSH-REVIEW §1 层级分解 |

### F 组：扩展、比较与判据治理（8 条）

| ID | 被禁形式 | 保留规则 | 出处 |
| --- | --- | --- | --- |
| V-F1 | 用小开发收益（少数 VT 命中+相位合法）授权大扩展矩阵 | 9 项长矩阵耗 63.8% 作业时间未形成可比主结论；扩展前必须有"改动影响正确/干扰区分"的依据 | OVERNIGHT §4.3 |
| V-F2 | 事后放松判据并当作原判据通过 | 先写"三项收益保留才进入后续"，遇 128K 混合又用 64K 收益转入 LoRA——"不能把重写判据当作原判据通过"；事前固定主终点与各结果分支行动 | OVERNIGHT §4.5；SYNTHESIS §7 执行问题3 |
| V-F3 | 来源可追溯 ⇒ 效果可外推 | P2Middle 教训：模型、表组成、gain 全变的搬运须先有跨模型依据才能称"完成方法推导" | OVERNIGHT §4.2；REVIEW-0907 行 240-244 |
| V-F4 | 诊断工具正确 ⇒ 问题被解答 | trace 首样两臂在格式 token 即分歧；全层状态漂移与数值误差限制固定 Q/K 解释 | OVERNIGHT §4.4 |
| V-F5 | 准备、文档、代码完成、hash 核对、Git 提交 ⇒ 科研目标完成 | 无已证实方法突破/超 MrRoPE/可升格主贡献之前，一切"工程完成"不计为研究进展；hash 只在新增/修改/传输边界核对 | REVIEW-0907 行 11；SYNTHESIS §7 执行问题1；OVERNIGHT §6 |
| V-F6 | 把同一 prompt 的多答案/多 head 当独立样本增大功效 | 样本单位是 prompt；"high power"需要目标效应量与配对不一致率，两次开发胜利不是可靠效应量估计 | OVERNIGHT §2；AUDIT-0910 §6 |
| V-F7 | 要求方法先具备完整理论/复杂校准组件才许有效 | "不要求补全理论，不以复杂诊断作为方法的必需组件"；简单规则不需要完整最优性定理 | SYNTHESIS §5 行 100、§7 行 246-248 |
| V-F8 | 计划文档中的成功百分比/时长估计当校准概率或计时决定晋级 | 它们是 planning guesses；新耗时方案先按真实吞吐核算 | AUDIT-0910 §6；REVIEW-0907 行 55 |
| V-F9 | 新方案与适配不足的旧方案比较后把全部收益归给网格（范式不对齐） | 新方法必须与 Cosh 在相同训练/适配范式下比较；只赢 Geo 不足以证明升级；零训练/短 LoRA 不是自动继承的任务约束 | COSH-REVIEW 结论节、§7.1-7.2 |
| V-F10 | 把未测写成否证；把轻微门槛失败写成崩溃；用平均收益覆盖预设保持要求 | 三件套判据分开：科学预测/实用验收门槛/工程检查；同时报告效果量与逐行得失 | SYNTHESIS §7 执行问题3；OVERNIGHT 各表 |
| V-F11 | 用尾部-512 NLL 两文档 OOD 摘要冒充全文档困惑度或八文档面板 | live OOD 摘要每 source/length 只有 2 份文档（可用 8 份），测的是长前缀后 tail-512 NLL | AUDIT-0910 Verified 节 |
| V-F12 | 拿 stale 的 root `development_summary.json` 当权威证据 | 权威入口是每个 method 自己的 `results/<method>/summary.json` 与 `ruler.jsonl` | AUDIT-0910 Verified 节 |

### 4.7 C1–C8 历史条件的最小裁决表（SYNTHESIS §7 行 211-223）
统一理论若要引用这些条件，只能用下表的"最小裁决"措辞，禁止升级为"已证实"或"永久淘汰"：

| 条件 | 最小裁决 | 使用约束 |
| --- | --- | --- |
| C1 小 movement 距离保证保持 | C2 案例否定其自动通过；无普遍 epsilon 阈值 | 不以 MAE 放行；也不把轻微门槛失败写成崩溃 |
| C2 无序频谱充分 | 匹配置换直接否定 | 保留槽位身份；仍不能排序所有保槽位方法 |
| C3 orbit 计数/粗界决定收益 | Exact/ULP 与有效 p2 否定该选择规则 | 只作代数性质记录，不用于挑表 |
| C4 背景导数能量下降保证收益 | 目标下降与生成退化同时存在 | 退出能力目标；不永久禁止全部背景统计诊断 |
| C5 顺序/平滑保证稳定 | 同 support 有序几何表长端失败，旧有效 p2 有交叉 | 非充分也非必要 |
| C6 support/base 决定能力 | 训练干预与冻结 weights×表交叉否定简单归约 | 同时记录训练表、权重、部署 support |
| C7 32/1-turn 分区充分 | 未建立（原 C2 反例不满足其精确条件） | 不得填"已证实"或"永久淘汰" |
| C8 小有序算子差保证任务保持 | 无尺度/内容/裕量阈值不能认证 | 不以"差很小"放行；界变大也不反推必败 |

### 4.8 复发模式警告（复盘的复盘）
- 复发的固定形态：局部量被否定 → 改名/增加自由度/把否定前提重新当起点（SYNTHESIS §2 行 33-34）。机检法：统一理论中出现的每个统计量都须回答"它与谱系表中被否定的量是同一个量吗（允许代数恒等变形检验）？若是，其到能力的推理必须由本文档承认的证据支撑，而非由新名字支撑"。
- 反向也禁止：不能为了"统一失败理论"把正例、混合结果和后来纠正删掉；不同协议的正结果与有效负结果必须继续保留，不能一次修 bug 把真实负结果洗掉（SYNTHESIS §2 行 49-50、§7 行 224-226）。
- "没有让已有反例约束后续方法和扩展决定"是本夜自认的主要执行错误（OVERNIGHT §4 末段）。

---

## 5. 频率表 / 方法定义清单

（构造规则 + 关键数字；除注明外均为文档记载的既有定义，本轮未新建表。）

| 名称 | 构造规则 | 关键数字 / 得分 | 出处 |
| --- | --- | --- | --- |
| MrRoPE / MrPro | 相邻 radix 的对数增量设等差数列再归一化；高频不动、慢频在目标跨度内插值；官方 gain a=1+.1·ln(s)，logits 乘 a² | 公开强结果；13 项 RULER macro 53.2（论文口径，不可与 9 项部分集直接比）；等差性是设计假设非最优性必要条件 | SYNTHESIS §2/§4；REVIEW-0907 行 66；AUDIT-0910 Prior-art |
| YaRN | 官方线性 ramp vs repo smoothstep 是两种实现身份；主训练真实 64K、400步、global batch 64（名义 ~1.68B token），128K 再 200 步 | 变体（如 Y2 sqrt 增益）零分不证官方上限；与本项目 300 步 rank64 LoRA（batch2×accum4、max 8192）不可当相同难度 | SYNTHESIS §2 行 42；COSH-REVIEW §5 |
| CoPE | 对末 20 槽进一步减频、末槽变零（固定代码 commit f8957a1）；幅度高通 vs 频率搬移是两种对象 | 与 Mr 中段+本方尾频组合：128K 简单检索 8/8、UUID 0/8 | REVIEW-0907 行 112、26 |
| 旧 log-p2（Qwen2.5-1.5B） | Native 频率基 + 因果 lag 权重 + 条件残差 min-max + p=2 + rcond=1e-10 + 历史 gain；2048 个取整质心采样；槽 1/18 有频率交叉尖峰 | 64K/128K core4 = 0.7000/0.5875；部署 FP32 hash `ed8abbb27a...58ba86`（三哈希与旧 audit 一致） | REVIEW-0907 行 189-198；OVERNIGHT §2 |
| FullLagP2 | 同旧 p2 规则但把 2048 质心换成全部 32768 个 lag（数值修复）；本方 gain 1+.074·ln4；官方 Mr gain 1+.1·ln4；BF16/Flash SDPA/greedy/repetition penalty 1.1 | 数组 SHA `ecd0c280...d461096b`；与旧 p2 中段最大相对差 0.01243%；两槽独立计算重现尖峰 99.36%。得分：1.5B 64K MK 37.50/12.50/25.00、VT 87.50/82.50/77.50、FWE 70.83/45.83/45.83（P2/Mr/同gainMr）；128K MK 0/0、VT 85.00/72.50、FWE 50/50；3B 64K MK 75/50、VT 90/95、FWE 66.67/75 | OVERNIGHT §2；LEDGER |
| P2Middle | 旧 1.5B p2 中段搬到 Qwen3B，保留 Mr 高频、尾频、本方 gain | 长 UUID 0 vs Mr 12.5；VT 65 vs 62.5；短能力退化——混合，不判死搬运本身 | OVERNIGHT §1 |
| 原始 Carrier | 以 Native 背景二次目标（复响应相干和）为改造目标 | 20 条开发长 UUID/VT 均 0；数字检索 32/50、EOS 14/50（口径：结束次数） | OVERNIGHT §1；SYNTHESIS §6 行 151 |
| Native-sector Carrier | 相位约束变体（以 MrPro 为 Native-sector 底座） | 9 任务：100/30/6/60.8/7.2/70/96/98/92；均值 62.22（部分）| OVERNIGHT §1 |
| C2（coupling transfer） | 低维耦合律拟合；OLMo/Qwen 边界约 11.2865/7.7814 与 10.7764/7.4297 turns | 两模型 slot19 C2 movement=0；"轻微 Native 门槛失败、长端近似保持"混合裁决；小 MAE 不保证门槛 | SYNTHESIS §6 行 148-150 |
| scale-transport 提案（09-07，失败） | 响应统计+分位匹配投影 | 槽 24-39 慢于 MrPro（0.28205–0.85724×）；槽 40-63 约 1–1.71165×；8 对等频、56 个不同频率；槽 23→24 间距 0.22493→1.48037；j24 累计压缩指数 0.0065→0.912；短 QA F1 0.425；5 点 guard 仅 λ=0 通过 | REVIEW-0907 行 20-25、121-125 |
| E1 s28_less / s29_more | 单槽（Qwen 3B 表）减/增压缩 | m：槽28 MrPro .098039→P2 .074131→E1 .065359；槽29 .137255→.221149→.183007；native 槽 28/29 窗内转数 12.367/9.966；32K 相位跨度 67.83/51.77 rad；s28: 128K +5.208pp（36 行），s29: 32K +8.333pp（单行）| EXTRAP-0910 §3-4；AUDIT-0910 |
| E3 BM(gain0.074) / E7 / E8 | BM 表配 0.074 增益 / 局部投影选择 / 慢槽 51 置零 | BM: 32K +12.778 / 128K −8.125；E7: +2.778 / −9.514；E8: −13.889（前 12 行）| AUDIT-0910 Verified 表 |
| Z（native 替代臂） | 旧 guard 拒绝 Native 后改用的 Z 短训 | Z 短训 20/32 vs 完整输入 23/32——不拒绝≠等效；Native 该臂未跑 | SYNTHESIS §2 行 45 |
| Cosh | J[ρ]=α/2∫ρ²+β/2∫S_ρ²、单位质量 ⇒ ρ''=(β/α)ρ；部署规则 τ=d/√L | 存在/唯一/闭式为真数学贡献；但族由目标形式选出；Phase16 独立邻点只赢 3/9 | COSH-REVIEW §1.2、§2 |
| 共享频率响应（正确局部对象） | J_{q,j}=Σ_h W_{O,h}Σ_i Δ_qi ∂o/∂φ；G_jk=(1/Q)Σ_q J_qj^T J_qk——同 query 内跨 key/head 带符号求和，query 间取平均；参数化 ν=ν_ref+ω_native·δ | 代数验证：中心差分最大误差 8.44e−10；抵消算例（共享能量 0 vs 逐 head 和 5.09866）；只描述冻结 hidden 局部块响应 | SYNTHESIS §6 行 190-204；REVIEW-0907 行 167-182 |
| exact-range / R | R=(K−1)ln(B)/K（正频率、非退化 support 域） | support-retargeting 使三种子排序反转 [已验证]；机制未识别 | SYNTHESIS §5 行 103、§6 行 158 |

---

## 6. 用户指令与纠正（原文引用）

1. （任务铁律，来自 AGENTS.md/失败复盘）"不得把代理指标说成能力结果；不得把未测写成否证；每个结论标注证据等级；引用文件/转录时给出路径或时间戳。"
2. REVIEW-0907 行 48："禁止自动恢复旧 Cosh 搜索、对手微调或 seed42 权重恢复。"
3. REVIEW-0907 行 266-269（作者要求）："研究迭代只能产出：有依据的具体解；可区分于剩余解释的决定性预测；或者非可识别性的证明及明确可直接测量的缺失量。不能把'更稳定/更平滑/局部改善'当完成目标，也不能在没有可区分预测时继续科学GPU工作。"
4. REVIEW-0907 行 250-257（作者简单性原则）："MrPro 不读取具体权重或校准数据，只依赖 RoPE 配置和扩展倍率，即能产生强公开结果。因此，必须依靠额外复杂 checkpoint 机制才能说明'为什么会成功'的方案，应先受到质疑。……诊断工具可以帮助排错，不能反过来成为方法必须具备的组成部分。数学自洽、合法数组、相似几何和代理分数各自只证明其实际覆盖的事实，最终效果必须由真实输出建立。"
5. REVIEW-0907 行 271-273（作者反馈）："关于 checkpoint 已经分配了 rotary subspace 功能……是下一阶段需要独立检验的研究输入；本夜没有证明它必然成立，更没有由它推出最优分配。新的材料审查与推导必须对照完整成功方法、所有已知反例以及实际 attention/value 算子。"
6. SYNTHESIS 行 4-6："作者要求结合整改报告与失败反思完善文档及实验计划；本轮只做文档审查，不生成新频率表、不运行 GPU。……本文不宣布新频率表胜过 MrRoPE、SOTA、论文接收或原研究目标完成。"行 16："外部材料里的旧指令不是运行授权。"
7. SYNTHESIS 行 23-24（作者对覆盖的口径约束）："清单有 7629 份文本候选，其中 6833 份是数据蒸馏缓存，不能算 6833 次实验"；行 29-30："也没有因为某些旧回执本次未找到，就把其状态改成'从未运行'。"
8. OVERNIGHT 行 5-7（作者收尾指令）："本夜实验已停止。频率分配的原目标尚未解决……作者随后要求转入完整材料审查、第一性原理推导和下一轮实验准备；这不恢复本夜队列。"行 182-183："作者要求先提交本轮报告与代码、收尾关机，再与两位子代理完整审查指定材料……本夜所有旧'下一步启动'文字失效。"
9. LEDGER 行 900（limits 字段原文）："Supervisor wall times are not cloud billing. SIGTERM during formal training was an author-requested wrap-up interruption, not a scientific or software failure verdict."
10. COSH-REVIEW 行 9-13："零训练、不接触原窗口外距离、只允许短 LoRA，都不是本次任务自动继承的要求。新方法需要与相同训练/适配范式下的 Cosh 比较；不能用一个充分训练的新方案击败适配不足的 Cosh，再把全部收益归给网格。"
11. COSH-REVIEW 行 3："用户希望获得比 Cosh 更好的非几何分配及扎实能力证据。把当前论文重述为'指数形状有作用'的机制论文，并不能代替这个目标。"
12. AUDIT-0910 行 6-8："The plan contributes useful missing controls... Its central two-force explanation is a plausible organizing hypothesis, but several claimed confirmations do not follow. Most notably, the E7 result does **not** demonstrate failure of local linearization, gain does **not** restore phase arcs, and increasing a frequency does **not** generally increase periodic positional separation. Correct these before spending GPU time on the proposed separation optimizer."
13. AUDIT-0910 行 115："The plan's claim that 'training-free + static + nongeometric frequencies' is unoccupied is too broad."
14. REVIEW-0907 行 207（效率与授权）："作者最新明确三次是效率期待而非硬上限，授权本夜 GPU 研究；实时预算由 HANDOFF 管理，不把本节当无限计算授权。"

---

## 7. 未决问题

1. **中段分歧的机制**：为什么有效旧 p2 与 MrPro 在中段后半程（槽 30 起）采用不同压缩？"先解释……再用实际共享频率响应区分可保留的功能"是既定研究次序，尚未完成（REVIEW-0907 行 200）。
2. **support 反转机制**：support-retargeting 三子排序反转 [已验证] 的具体机制未识别；"尚无被 matched evidence 唯一支持的解释"，现有 L/P 不回答此题（SYNTHESIS §5 行 103）。
3. **L/P 可见性诊断是否执行**：同可见集合、两干预时点的 oracle 因果诊断已备好未启动，收紧为"必须能改变一个具体设计取舍"才执行（SYNTHESIS §5 行 115-130）。
4. **E1 冻结规则跨模型泛化**：新 seed 混合/多样面板与 `cross_model.py` 独立评测未跑完，之前不得宣称泛化；OLMo 迁移（4K −3.333/16K +3.819pp）仍显著低于其 BM 对照（AUDIT-0910、EXTRAP-0910 §7.1）。
5. **gain 的 2×2 factorial**：MrPro×gain0.074 臂未补，表×gain 的因果分解未完成（AUDIT-0910 §2）。
6. **s28/s29 方向性与可组合性**：等步长反转、邻槽控制、s28×s29 交互测试已固定未评；初始 12 行 ties 是天花板限制，需历史全 36 行（EXTRAP-0910 §7.2）。
7. **B5 重写**：需要绑定带符号内容竞争分数、在未见过数据上预测 margin 的选择器；现有几何指标只作诊断（AUDIT-0910 §3、minimal-continuation 5）。
8. **row-wise 压缩算子定义**：相位附着对象（token vs query 重标定）、cache 合同、W 边界回跳反例，定义前不得实现（AUDIT-0910 §5）。
9. **数值不变性诊断**：全局 position-ID +1 平移的 exact 对称性在 BF16 下的实现检验已排队（3 个正例 + MrPro 控制）；翻转将识别算术敏感而非任务失败（EXTRAP-0910 §7.4）。
10. **E10 收益归因**：频率混合 vs BF16 执行路径变化——同钟扩窗 NLL 控制未跑（EXTRAP-0910 §7.5）。
11. **FullLagP2 旧原表对照缺失**：本夜未做同输入旧完整 p2 vs 全 lag 修复对照，"修复是否改善能力"仍是未答问题（OVERNIGHT §2 行 43-44）。
12. **统一框架候选**："phase-dependent, content-conditioned score changes + altered prefix states + amplitude/numerical effects + a task decision boundary" 是当前有用框架；"simpler two-force rule may emerge from them, but has not yet been demonstrated"（AUDIT-0910 行 111）——统一理论必须从这四类可分离组件出发，或给出能合并它们且有决定性预测的规则。
13. **rotary subspace 功能分配假设**：作者研究输入，未验证；若统一理论以其为前提必须显式标 [假设] 并给检验路径（REVIEW-0907 行 271-273）。
14. **低频载波恒等式线索**：组内统一加 c 保持频率差、复 logit 乘 exp(icΔ)，但注意力取实部不保语义；"尚无选取 c 的充分依据，不应凭此启动实验"（REVIEW-0907 行 202）。
15. **G_W(δ) 联合窗相关量**：作为"值得测试"的几何量（依赖频率间距与距离测度），其与 learned 距离分布、内容坐标的关系未建立；FullLagP2-E1 两槽一致是 hypothesis-generating agreement（EXTRAP-0910 §4）。

---

## 附：统一理论自检脚本要点（机检接口）

把 §4 表转为规则引擎时，每条 VETO 至少三元组 `{id, pattern, permitted}`。建议的自动检查钩子：
1. 文本层：出现"⇒能力/⇒SOTA/⇒上限/证明……失败/淘汰全部/永久"与代理名词（PPL、NLL、margin、mass、Gram、曲率、残差、平滑、相位弧、覆盖、orbit、Jacobian、NMSE）共现时强制人工核对对应 VETO 行。
2. 数字层：任何被引用的分数须带（协议、任务集、样本单位、长度、gain、模型）六元组，否则触发 V-C4/V-B5/V-F6。
3. 谱系层：新统计量与 §5 表中已有量做代数恒等变形比对，命中即套用旧裁决（防"改名复活"）。
4. 双向层：既查"未测写成否证"（V-A8、V-D17、V-D20、V-F10），也查"代理写成分"（V-A 全组）。
