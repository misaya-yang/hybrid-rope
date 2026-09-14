# digest nongeo-code

- 范围：`experiments/nongeometric_screen/` 全部代码与回执 + 并行 20×10 计划及其审计 + `scripts/experiments/scale_transport/ruler_full_prepare.py`（含其"未提交"改动的落实）+ 承载 GPU 结果的 `docs/research/` 四份研究文档。
- 铁律执行口径：每条结论标注 [已验证]（有代码回执/文档数字且未被后续纠正）、[部分证据]（开发集/小面板/单例）、[假设]（机制解释或推断）。引用给出路径；远程回执给出目录。
- 核心问题（任务原问）：20×10 并行计划设计了哪些非几何机制候选、哪些已在 CPU 判负、哪些还等 GPU、与 bank/arc 二分法的关系（替代机制还是兼容成本项）——直接答案集中在 §5.6–§5.8。

## 1. 来源清单（文件路径/会话 id、大小、行数）

### 1.1 实验代码（experiments/nongeometric_screen/，行数=wc -l）

| 文件 | 行 | 作用 |
|---|---:|---|
| README.md | 82 | 模块地图；active run=`/root/autodl-tmp/nongeometric_screen_20260909`；最新用户协议原文（128K-first，见 §6） |
| worker.py | 232 | 常驻 Qwen2.5-3B（bf16, sdpa/flash-only），作业队列/SHA 漂移门/STOP 文件；`install_table`（64 有限非负频率、rope_type=default 拒绝叠加动态缩放）；`qualification()` 必须逐 token 复现归档 MrPro greedy；NLL=尾512、默认 4 文档 8K/32K；panel small=12 行(`_0`)、full=36 行；summary scope="Historical development inputs; not independent confirmation" |
| select.py | 190 | 固定态条件化选键 replay（提案生成 E1×32/E2×4/BM/E8×24；聚合 robust_gain=min(split)；E1 top2、E2 最大、E8 上限内最大、E4 对=两 split 正相关、E5=BM 层 top2、E6=组间差最大层；enq 010+）；scope="Whole-model tests are mandatory" |
| capture.py | 161 | 真实完整前缀无缓存捕获：top8+recent32+target+64 linspace 选键、全行 lse、baseline_lse/full_target_mass；"Explicit correct-answer trajectory on calibration only" |
| operators.py | 83 | layer/group 替换 + dual_frequency（E10）：16 槽 24–39 与 88–103 复制，Q 侧 ½ 权重，K160/V128，logits 合并后一次 softmax；SDPA 融合需临时 V 垫维、输出切回 head_dim |
| distance_operator.py | 98 | E9 距离算子：两个 FlexAttention BlockMask（local δ∈[0,w] 用原生时钟；far δ>w 用 MrPro 时钟 + Q 侧相位修正 w·(native−mr)），各自 lse 后 logaddexp 联合归一；batch-1 无 padding 约束 |
| checks.py | 65 | 稠密验证：conditional_dense_max_error=4.768e-07；dual_frequency_score_max_error=1.907e-06；layer/group 恒等=0.0；dual_frequency_bf16_rms=0.0578、relative_rms=0.0168(<0.03 门) |
| project.py | 95 | E7 构造：结合窗 w=校准 multikey 原子记录最长距离；J=带符号局部响应导数；H=J^TJ（W_O einsum）；预算=E1 候选局部 replay 步成本两侧均值之 min；λ 二分+径向帽 0.25 rad；enq 019_E7、040_E9_distance、041_E10_dual_frequency |
| project_local.py | 23 | E7 预算修正（同支撑预算）：修正后若表≡BM 则标 IDENTICAL_TO_HISTORICAL_BM、done/019 改 REUSED_IDENTICAL_BASELINE |
| local_check.py / local_precision.py / precision_check.py | 34/52/26 | E7 非线性复核 + 精度分解：理想相对 7.93640e-8 ≈ 线性 7.93668e-8；FP32 绝对 7.93255e-8；BF16 绝对 1.28074e-5（"160×"实为 BF16 实现） |
| repair_replay.py | 28 | 修复 2 条 oldp·exp(δ) 下溢记录（log-weight 归一）；重算 bm_layer_ranking：E5 第二名 27→32（27 的提名作废，已测输出保留） |
| smooth_budget.py | 64 | 最小粗糙度 KKT 解（qwen 界 23/40、预算 (n−1)/3；BM 恢复校验 6i(n+1−i)/(n(n+1)(n+2)) @ (n−1)/2）；enq 0440_Smooth_MrBudget、0441_MrUni（panel full、16 doc、8/16/32K） |
| long_bridge.py | 61 | 周期∈[32768,131072] 的槽（36–39）±1/131072 rad/token；enq 0442_Slower(−1)、0443_Faster(+1) |
| gap_budget_transfer.py | 86 | EVQ 启发的 log-gap 运输：donor=原生前缀 gaps 0–22、预算=均值 0.2158673516（高频 log 量程的 4.3478%）；recipient 按几何均值周期选带；enq 0445_HighGapToLong([32K,128K]→gaps36–39)、0446_HighGapToMid([2048,8192]→gaps26–31) |
| scale_taper.py | 55 | BM_ScaleTaper：w_j=clip(log(W/T_j)/log4,0,1)，ν=Mr·(BM/Mr)^w；槽 24–31=BM、32–35 权重 .8972/.6762/.4486/.2144、≥36=MrPro；enq 044d_（后随 hand-built 分支 deferred） |
| prepare_gap_probe.py | 47 | GPT-5.6 Pro 三臂（source: User-provided GPT-5.6 Pro analysis, 2026-09-10）：G1_gap_widen(24,48)/G2_gap_narrow(36,36)/G3_pair_shift_slow(36,48)，围绕 (m28,m29)=(30,42)/306，δ=6/306；enq 044a-c + 046_gap_probe_binding16 |
| prepare_diverse.py | 107 | 新 QA 队列：16 文章分层抽样（seed 20260910，排除开发文章），其余行逐字节复制 |
| prepare_long_sources.py / prepare_long_tokens.py / long_eval.py | 61/38/54 | PG19 test 书 + Proof-Pile arXiv test（官方哈希 b1bc923a…），每源前 8 篇 ≥131073 token；64K/128K 尾512 NLL；明示"not full-document PPL" |
| causal_cases.py | 76 | 前缀×读取交叉实验：cached-path qualification 先行；K 从原始 k_proj 重建（绝不反演取整缓存 K） |
| binding_swap.py | 51 | niah_multikey_2_131072_2 中互换 6683176↔9424151（等 token 长、多重集不变），报告首位数字 D_plus/D_minus |
| origin_shift.py | 34 | 匹配正例整体 +1 位置平移 |
| numerical_controls.py | 33 | E10 same-clock(Mr/Mr) 算术对照，对比 stock-Mr 与 E10 混合 |
| cross_model.py | 119 | E1 跨模型运输规则：过渡分数 5/17、取源槽 28→目标 m[j−1]（Qwen bounds[23,40]；OLMo bounds[14,32]→槽19）；对目标结果盲目 |
| holdout_eval.py / paired_summary.py | 54/72 | 冻结表新队列评测；任务×长度配对差 + 描述性 bootstrap（seed 20260910、10000 次、未校正多重比较） |
| pro_block_calibration.py | 115 | 用户提供的 Pro block-stretch 假说校准：MrPro 增益下原生频率、32768 全行捕获、runtime→exact replay KL |
| distance_checks.py | 59 | E9 算子稠密/核一致性检查 |
| native_reference.py / summarize.py / finish_screen.py / binding_window（结果） | 21/51/34 | Native S1/gain1 参照；开发表生成（retired_arms.json 退役臂保留已测行）；宽松 screen 规则（崩溃判据后自动扩全 36 行排名 060+；被 128K-first 协议取代） |

### 1.2 结果回执

- 本地镜像 `results/nongeometric_screen_20260909/`（自服务器同步的部分）：`selection/`（proposals.json、conditional_rows.jsonl、summary.json、replay_repair.json、selection.json）、`implementation_checks.json`、`runtime.json`（torch 2.8.0+cu128、transformers 5.15.1、RTX 4080 SUPER）、`qualification.json`、`done/`、`queue/`、`deferred_queue/`、`results/<job>/summary.json`、`long_nll/`、`causal_cases/`、`counterfactual_bindings/`、`origin_shift/`、`transfer/`、`holdout_results/`、`planned_controls/`（含 p2_gap_comparison.json、gap_budget_transfer.json、long_bridge.json、scale_taper.json、smooth_budget/）。
- selection.json（[已验证]本地回执）：e1=[E1_s29_more, E1_s28_less]，e2=E2_tail_more，e8=E8_zero51，e4_pair=(corr 0.1453, 槽25,29)，e5_layers=[21,27]（修复后第二名=32），e6_group=(0.3471, 层14, 组1)；gain=1.138629436111989。
- 远程权威目录：`/root/autodl-tmp/nongeometric_screen_20260909`；历史基线 `/root/autodl-tmp/bm_transfer_20260908`（run_qwen3_01/MrPro.jsonl 36 行、run_nll_01 48 行、prepared_qwen3_01、prepared_nll_01）。

### 1.3 研究文档（docs/research/）

| 文档 | 行 | 角色 |
|---|---:|---|
| NONGEOMETRIC_TEN_CANDIDATE_PLAN_20260909.md | 326 | codex 轨 E1–E10 全设计（每候选 6 条排除、阈值、宽松混合 RULER 裁决、"十项都失败后怎样得到有用结论"表）；用户两次纠正原文（§6） |
| NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md | 630 | 全部已执行 GPU 结果的权威数字来源 + 协议纠正（§2/§3/§5 大量数字取自此文） |
| PARALLEL_NONGEOMETRIC_20X10_PLAN_20260910.md | 227 | 并行轨 v2：20 候选（A1-A4/B5-B7/C8-C10/D11-D15/E16-E18/F19-F20）×10 实验（1′–10′）；状态"设计交付、未执行"；§1 codex 快照表、§2 八条抨击验证、§3 二力框架（arc restoration vs phase separation）、§8 锚点数字 |
| PARALLEL_NONGEOMETRIC_20X10_AUDIT_20260910.md | 127 | 对 v2 计划的审计：E7 160× 改判 BF16；弦距非单调反例表（s28_less 的 Δ² 变化 +0.1623/−0.3651/+0.5320/−1.5277/−0.4050 @5 个 Δ）→ 5′ 回测门 CPU 判负；二值分≠强交互；300–500 点失败不能封顶族；"别关族"；先验纠正（MrRoPE 本身即非几何 radix 日程） |
| UNIFIED_BUDGET_ALLOCATION_THEORY_20260910.md | 143 | bank/arc 二分法（§2）、不变量 I1/I2（§1）、27 面板预算落点读法、EVQ 调和（sink 对/source 反）、冻结队列 0446→0448→0449→0450→0451、"GPU 已关" |
| BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md | 63 | 预算分配模型（端点固定 + 守恒 ln S=1.386 + D_j + 洞 + 危险区）；全部已测统一读法表；两个独立长端机制（后端完成 LBS / 前端保真 s28_less）；三新候选参数表 + 预冻结判定规则；队列排布 |
| ROPE_BM_TRANSFER_20260908.md 及 0908 系列 | — | BM/GapCapped/P2 历史谱系与 OLMo 复核（P2 "1.5B/64K 旧协议"出处；OLMo S4 BM 独立复核过、S8 失败） |

### 1.4 scale_transport 关键入口与"未提交改动"

- `scripts/experiments/scale_transport/ruler_full_prepare.py`（106 行）：官方 RULER 单任务分片生成器——13 任务名、UPSTREAM_COMMIT c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a、`--count 50 --seed 137 --length 131072` 默认、validation 子集、chat 模板 + 原始答案预算校验（`0 < len(ids)+tokens_to_generate ≤ length`）、rows.jsonl/manifest.json 全 SHA 回执。这是并行计划实验 3′（MK2/MK3 50 行高功效面板）的基建。[已验证]
- 会话开始时 git status 显示 M；复查发现工作区在 09-10 07:49 被快照提交（a6e3aba），diff 已被提交吸收。`git diff 951f51e HEAD -- …` 显示改动恰为一行：`'NLTK_DATA': '/root/autodl-tmp/nltk_data'` → `'NLTK_DATA': os.environ.get('NLTK_DATA', '/root/autodl-tmp/nltk_data')`，目的=允许在非服务器环境（本地 CPU）跑官方生成器时覆盖 NLTK 数据路径。[已验证] 无其它未提交语义改动。
- 其余入口 docstring 略读：run.py（2 小时冻结 pilot）、ruler_run.py、ruler_prepare.py、evaluate_proposal.py、position_visibility.py、prepare.py——与本轮裁决无直接依赖。

### 1.5 会话/转录

- 本任务上游主转录：`analysis/unify_20260910/raw_thread-main.txt`、`raw_thread-0909-pm.txt`（含 candidate 命名的最早出处）。
- 本次 compact 前会话 jsonl：`/Users/[REDACTED_AUTHOR].yanghejazfs.com.au/.claude/projects/-Users-[REDACTED_AUTHOR]-yanghejazfs-com-au-paper-project-hybrid-rope/7cf3ec49-b384-45ca-99f1-edab805c575a.jsonl`。

## 2. 任务时间线（目标 → 方案 → 结果 + 关键数字）

约定：面板分数一律为 32K/128K 六任务宏平均（36 行开发面板，除非注明"12 行"=panel small `_0` 行）。所有结果为**开发集证据**，独立确认列于 T10。

**T0 历史复用基线（09-08，bm_transfer_20260908）**
MrPro 官方 gain：87.2222 / 78.1250。BM（MrProBM）：91.6667 / 70.8333。判别面锚点：MK2@128K MrPro 75 / BM 50 / GapCapped 50；MK3@128K 本地缺失（carrier floor 6%）；单 passkey 双方 100%（饱和，弃用）。NLL（16 篇 FineWeb-Edu 尾512, 8/16/32K）：Native 2.27607/2.14317/2.03850，MrPro 2.32333/2.18886/2.08899，BM≈MrPro±0.004。[已验证]（worker 启动即 SHA 校验 36+48 行基线完整性）

**T1 基建与资格门（09-09）**
目标：单常驻模型、可续跑、防漂移的多候选筛选。方案：worker.py 队列 + install_table 约束 + qualification 逐 token 复现归档 greedy。结果：qualification PASS；checks 全过（见 §1.1）。[已验证]

**T2 固定态条件化选键 replay 提名（CPU+捕获，09-09）**
目标：不花整网 GPU 就预排序 60+ 静态提案。方案：capture.py 真实前缀 top8+recent32+target+64 支撑，select.py 解析式重放（恒等门 nmse≤1e-8、|Δlogit|≤0.01），robust_gain=min(两 split 任务宏)。结果：提名 e1=[s29_more, s28_less]、e2=E2_tail_more、e8=E8_zero51、e4=(25,29)、e5=[21,27→32]、e6=(14,组1)。事后教训：E8 代理增益>E1 但实测 −13.9pp → 代理只能预排序。[已验证（作为"提名发生"）；作为预测器=已被否证，见 §4-F7]

**T3 E1 单槽手术（GPU 小面板→全 36 行）**
目标：测过渡区相邻步长交换。方案：slot∈[24,40) 取相邻槽 m。结果：
- `E1_s28_less`：32K 持平 87.2222，**128K 83.3333（+5.2083pp，2 胜 0 负**：multikey 行2、multiquery 行3）；16 文档 NLL(8/16/32K) = +0.0001544/−0.0006042/−0.0008822。[部分证据：开发集 36 行；确认在 T10/队列 0450]
- `E1_s29_more`：**32K 95.5556（+8.3333pp）**、128K 77.9167（−0.2083）。短端收益 = 一道 QA 答案；长端 VT 回收 1 变量、multiquery 丢 1 值。[部分证据：单行驱动，小样本噪声敏感]

**T4 端点违反臂（GPU 12 行首筛）**
- `E2_tail_more`（平台再压 ÷4.93）：128K 崩至 54.7；失败分型=误绑定（multiquery 同值填两键）。[部分证据(12 行)+机制分型]
- `E8_zero51`（槽 51 置零频）：12 行 128K −13.889pp（统一文档记 50.6）；分型=终止/格式型非检索型。[部分证据]
- 二者 + MrUni/HighGap 后被提炼为端点不变量 I1/I2 的反例（§3-C4）。

**T5 gain 臂（E3/A3 系）**
方案：表 × gain 系数析因 {MrPro,BM}×{.1,.074}、BM×1。结果：MrPro .1 87.2222/78.1250；BM .1 91.6667/70.8333；MrPro .074 98.3333/75.3472；BM .074 100/70；BM gain1 89.5833/58.8194（NLL 反而更好）。同 gain 下 BM 表效应 = +1.6667 短 / −5.3472 长。结论：gain 是强长度耦合的幅度变量，救不了 BM@128K；更宽 gain 网格未排队。[已验证（四格完整析因，开发面板）]

**T6 异质轴（E4/E5/E6）**
E4_pair25_29（槽 25/29 共享平均指数）：12 行持平。E5_layer21/32、E6_layer14_group0/1（单层 BM、单 KV 组替换）：12 行持平。[部分证据=空结果；下节 pair(28+29) 给出了同位置的超调反例]
后续（budget 文档）：pair(28+29) 组合 = 87.2/74.0——同位置双倍集中 → 2.9–4.2K 洞超调(1.46×) → "同位置叠加不稳健"。[部分证据]

**T7 E7 局部投影（构造→实测→两次修正）**
目标：沿 BM 方向做受局部输出预算约束的投影表。结果链：
1. 实测 019_E7：32K +2.778 / **128K −9.514pp**（68.61%）。负。[部分证据(开发面板)]
2. 计划 v2 曾引用"线性模型低估扰动 160×"为方法论判决；local_precision.py 精度分解证明 160× 属 **BF16 实现误差**（理想相对 7.93640e-8 ≈ 线性 7.93668e-8；FP32 7.93255e-8；BF16 绝对 1.28074e-5）。审计改判。[已验证（数值分解可复算）；"其余端到端退化是否舍入所致"=未证]
3. project_local.py 修正预算同支撑后，约束问题的解恰为 BM → 修正版 E7 从未需要 GPU（019 标 REUSED_IDENTICAL_BASELINE）。[已验证(构造)]

**T8 算子类（E9/E10，"替代机制"档）**
- `E9`（距离域双时钟，distance_operator.py + distance_checks.py）：算子建成、公式稠密验证过、040 号位排队（窗 w=binding_window 校准值）；**无任何模型分数见诸文档——等 GPU**。[假设（仅实现完成）]
- `E10`（双频核：中带 16 槽 Mr/BM 双频，½ 权重，K160）：分数误差 1.907e-6、bf16 relative_rms 0.0168；12 行 RULER **平 MrPro**；数值对照：same-clock 控制−stock = −0.000709/+0.000030 nats(8K/32K)，E10 混合−控制 = −0.005180/−0.012242 → NLL 微改善不能被算术对照解释；36 行全面板 pending。审计意见：BM 在 128K 是净负资产，作混合成分先验更差；"kink≠不连续"、需行定义。[部分证据(12 行平局+NLL 分解)]

**T9 因果干预与诊断（09-10，CPU/少量 GPU）**
- 前缀×读取交叉（causal_cases.py）：MK2 行2 四格 0/0/0/1；multiquery 行3 0.75/1/0.75/1；VT 行2 0.8/1/1/1；QA 行1 全 0（cached 路径失效，不能用于解释）。
- MK 首位数字 margin 分解：四列 −2.125/−1.125/−1.000/+0.250 → 读取 +1.000、前缀 +1.125、**交互残差仅 +0.250（BF16 分辨率级）**——"二值成功表≠强非线性机制"。[已验证（同一记录 token 级复算）]
- binding_swap：互换后两法都答出不在 prompt 中的 6624365/6624369 → 竞争干扰，不能作二元选择概率；D+/D− 分量：MrPro (−2.0, 0.0)、E1 (−0.6875, 0.9375)。origin_shift：+1 平移把 E1 的 MK 胜利翻成失败（但方向性 margin +2.25→+2.0 存活；multiquery/QA 收益保持）。[部分证据：三个回看个案]
- 校准距离记录：MK 行末查询到目标值首 token 88,725（及 117,487–117,493 / 95,676–95,682）；单键 51,403–51,409 / 29,024–29,030；multiquery 行0 混合 70–120,933 → "128K 分数≠128K 关系直测"。[已验证(字面几何)]
- 理论工件：z_j(ν) 与 ∂M 推导（mechanism §70）；s28_less 相位滑移 3.144 rad@32K、12.575@128K、8.512@88,725；旋转差公式 2|sin(d·Δν/2)|。[已验证(代数)]
- E10/Native 参照补：native S1/gain1 短参照 12 行 83.3333（vs MrPro 87.2222；QA 0 分拖低六任务均值）。[部分证据]

**T10 运输与新数据（09-10）**
- Qwen2.5-7B：原生网格与 MrPro 界相同 → 槽 28 表逐槽精确运输；18 行六任务面板 + 4 文档 NLL。已跑完、不再扩展（数字未录入本组文档）。[已验证(完成性)；结果数字不在所读文档 = 记录缺口]
- OLMo-2-0425-1B：固定规则=过渡分数 5/17→槽 19（界 14/32），目标结果盲目。已跑完、不再扩展。
- 新 3B 任务：四任务 ×16（32K/128K，seed 20260910，QA 16 文章分层）；新长文 PG19/Proof-Pile 尾512 NLL@64K/128K：slot28 差 −0.00155/−0.00121（Proof-Pile）、+0.00359/+0.00420（PG19），slot29 ≤+0.00304——"小样本尾 NLL，非全篇 PPL"。[部分证据]

**T11 并行 20×10 计划（v2）与其审计（09-10，全程未占 GPU）**
计划交付 20 候选 + 10 实验（§5.6 全列）；§0 三判断（headroom 真实、一阶代理定量否定、二力框架）。审计（同日，CPU）：
1. E7 160× 改判；2. 弦距 5 点回测表证 **全局"分离度"指标不可行**（s28_less 在 5 个 Δ 中有 3 个降低分离度：Δ² 变化 +0.1623/−0.3651/+0.5320/−1.5277/−0.4050）→ 实验 5′ 的 GPU 前置门**CPU 判负**，B5 须改为内容条件化带符号 margin；3. 先验纠正：MrRoPE 本身就是非几何 radix 日程，"零训练+静态+非几何"象限按机制细分；4. 方法论：300–500 点搜索失败不能给族封顶；"不要关族"；5. 需先给 row-wise/E9/E10 下"行定义"（缓存语义/核成本）。计划本体：设计交付、未执行。[已验证(审计计算)；计划状态=未执行]

**T12 整表规则族（09-10，codex 轨 GPU）**
- `Smooth_MrBudget`（0440，固定 B=16/3 最小粗糙度 KKT 表）：87.2222/**68.3333**（−9.7917@128K；multikey .75→.25、QA .5→.25，multiquery/VT 升），NLL +.000975/+.001925/+.003423 → **否定"固定预算下最小粗糙度即处方"**。CPU-KKT 证书（roughness .004886399，B=8 恢复 BM）只证明构造凸性。[已验证]
- `MrUni`（0441 原义，全表均匀斜坡）：32K 64.6 → I1 端点反例。[部分证据(文档记录，细表未展开)]
- `LongBridgeSlower`（0442）：80.5556/80.0694（**−6.6667/+1.9444**；VT .75→.95、FWE .75→.6667；3 升 3 降）；NLL +.001088/−.000231/−.000228。`Faster`（0443）：87.2222/73.9583（0/−4.1667，唯一变化 QA 行2 1→0；3 位等幅反向）→ 等幅反方向不复现长端信号。恒等式 A_δ(d)=cos(dδ)A(d)+sin(dδ)B(d)（CPU 复算 8.9e-16）说明该臂动的是**带相对内容的公共相位**，非带内间距。确认队列已排（VT16+MK16×2 长度=48 生成）。[部分证据：开发信号+短端代价公开]
- `BM_ScaleTaper`（044d）：构造+OLMo W=4096 几何构造检查通过（周期坐标可运输）；**无结果**——随 hand-built 分支 deferred。[未执行]
- G1/G2/G3（044a-c）：CPU 核算先拆台——G1/G2 使三增量为 (−δ,2δ,−δ)/反号、几何均值对频与总 B 不变，但 Dirichlet 粗糙度同增 20δ²、扰动范数相等且都连带动外隙；未测即撤（用户批评候选质量后整支 deferred）。[未执行]

**T13 EVQ 运输与 P2（09-10）**
- `HighGapToLong`（0445，EVQ 字面操作：高频抽 0.2158674 log 单位给 gaps36–39）：**32K 70.1389(−17.0833) / 128K 67.3611(−10.7639)，36 行 0 升 7 降**，NLL +.008165/+.017769/+.018295 → 付了短端成本、没买到长端收益。`HighGapToMid`（0446 原义）16/36 处停止、部分数据保留。EVQ 理论辨析：Cosh 导数 tanh(τ)/τ=.761594 高、sinh(τ)/τ=1.175201 慢 = **log-频率间距运输而非通道数运输**（密度在高发端实际更高）；训练态假说不自动搬到冻结模型。[已验证（该表判负）；不否证一切高频运输/不否证 EVQ 训练理论]
- `FullLagP2`×当前协议（36 行 + 48 NLL 全跑完）：72.9167/**81.6667**。vs 官方 MrPro +3.5417@128K（−14.3056@32K）；vs 同 gain .074 MrPro **+6.3194**@128K（−25.4167@32K；同 gain 参照 98.3333/75.3472）。128K 任务分解：P2 1/.5/1/.9/.75/.75 vs 同 gain 1/.75/.9375/.75/.8333/.25 → 长端优势由 QA+VT 驱动、multikey/FWE 付账；5 升 7 降(对同 gain)。NLL vs 官方 −.004285/+.000477/−.008876、vs 同 gain +.014000/+.017771/+.012728（48 输入哈希全对）。
- **P2≠高频运输证据（CPU 直查部署数组）**：P2 对高频 gaps0–22 总改动仅 +.0007754，比 HighGapToLong 的 .2158674 **小 280 倍**；其真正落点是 gap29 +.809123（gap28 +.149445、gap30 +.131703，gaps31–39 变窄；槽 29/30/31 周期 3,977/5,258/7,016→4,468/13,267/20,193，槽 40 保 141,332）。[已验证（算术）]
- 128K-first 协议下 P2 过 128K 初筛：尾512 NLL/PPL@128K：MrPro 官方 1.705390/5.503529；同 gain MrPro 1.680924/5.370516；P2 **1.678507/5.357549**；precheck 268.7s。已放 4 行新 128K QA pilot（0441 P2_QA128_pilot4，收尾中；12 generations 已释放）。[部分证据]

**T14 统一理论成文 + 新候选入队 + GPU 关机（09-10 深夜）**
UNIFIED + BUDGET 两文档把上述 27 个已测面板归入"预算落点"单一自由度模型（端点固定 ⇒ Σ17 过渡 log-gap=基础+ln S=1.386；唯一自由=额外 ln S 放哪）；确立 bank/arc 读法（§3-C6）；判词"**所有赢家都在把预算右移（桥变窄/危险区完成），所有输家都在左移或动端点**——单方向、8 个独立构造一致的梯度"。构造并冻结三张新候选 0446 StackFrontBack（MrPro⊕s28_less⊕LBS：m28=.065，m36–39=.625/.729/.847/.980→D=127.5K，max 洞 1.86×）、0448 MrProN16（径向族 m_q=q(q+1)/272，槽39完成，D=131K，洞1.76×）、0449 MrProN15（38 槽完成，洞1.80×），各 36 行+48 NLL（45–55 分/张）→ 0450 E1 新样本确认 → 0451 P2 长端确认，共约 6–7 小时。判定规则预冻结（§6）。**"GPU 已关。服务器队列已冻结（恢复后自动按序执行）"**（UNIFIED:143）。[已验证(文档+队列回执)；结果=未执行]

**簿记注**：queue 前缀号存在复用——`smooth_budget.py` 写 0440/0441，`gap_budget_transfer.py` 写 0445/0446，而末期文档中 0441=P2_QA128_pilot4、0446=StackFrontBack。合理解释：先前作业已 complete/退役后编号复用，HighGapToMid（0446 原义）与 hand-built 分支同撤入 deferred_queue。[假设：复用路径未留改名回执；核对以 done/ 与 contract.json 为准]

## 3. 理论主张表（主张 | 证据等级 | 出处 | 后续是否被纠正/推翻）

| # | 主张 | 等级 | 出处 | 后续状态 |
|---|---|---|---|---|
| C1 | gain=注意力 logit 幅度补偿（1+0.1·ln4=1.138629…，Q/K 双乘 ⇒ logits×gain²），改幅度不改相位弧；与表分配正交（四格析因） | [已验证] | mechanism §254–267；audit §"gain is amplitude" | 成立；细化为"gain 是强长度耦合变量"（32K +12.78 而 128K −8.13） |
| C2 | 单槽手术可在 128K 超 MrPro（s28_less +5.2083pp、2W0L） | [部分证据] | mechanism §52–57；selection.json | 未被推翻；确认在 0450 排队，未完成 |
| C3 | 任务收益在单槽粒度上 NLL 失明（两 E1 赢家 NLL≈0 而任务 +5.2/+8.3pp）⇒ NLL/固定态代理只能当门、不能当目标 | [已验证] | plan §5实验8′、§8 锚点；mechanism §56 | 成立；升级为搜索目标函数设计原则（实验 8′ v2 修正） |
| C4 | 端点不变量：I1 j≤23 恒等（bank 不可压）；I2 j≥40 精确 ÷S（平台是 p→p/S 的精确重参数化，改水平即破坏） | [已验证(反例集)] | UNIFIED §13–14；BUDGET §11–12 | 由 HighGapToLong/MrUni/E8（I1）与 E2_tail_more（I2，12 行）负结果归纳；仍是"归纳到当前数据"，非定理 |
| C5 | bank/arc 二分法：r_j=W/T_j≳8–10 的槽（≤~29）=整数对齐锐利局部滤波器组（零压缩容忍）；r_j≲8（~29–63）=弧钟（窗内容忍、越 D_j 恶性）；危险区=槽36–39（原生 r∈[1.15,2.2]，MrPro D=75/85/97/112K 全<128K） | [部分证据(机制解释)] | UNIFIED §26–41,§85 | 未被推翻；作为组织所有赢家/输家的解释框架；其预言（Core-C 距离-准确率曲线，MrPro 在 75–112K 带衰减）排队 0450/0451 待验 |
| C6 | 预算守恒水床恒等式：两端点固定 ⇒ 17 过渡 gap 之和固定（额外 ln S=1.3863 唯一自由度）；B=Σm_q=N−Σi·ε_i；D_j=W·S^{m_j} 弧安全距离；洞比 T_{g+1}/T_g>原生 1.241 | [已验证(代数)] | BUDGET §14；mechanism §209–221 | 恒等式本身精确；"预算守恒使 bank/桥/危险区三者不可兼得 ⇒ 零和、必须砍桥"是**对该参数化类的断言**，静态单表内成立，算子外（E9/E10/row-wise）不自动成立 [假设边界已被审计明确] |
| C7 | 两个独立长端机制：后端完成（LBS→VT@106K）与前端保真（s28_less→multikey@89K）；P2 是两者极端混合 | [部分证据] | BUDGET §35–39；UNIFIED §73 | 叠加可分性=0446 StackFrontBack 的直接检验（排队未跑） |
| C8 | "所有赢家右移预算、所有输家左移或动端点"——8 个独立构造同向梯度 | [部分证据(方向一致但均为开发集)] | BUDGET §58 | 待 0446–0451 升级为论文级证据 |
| C9 | EVQ/Cosh 有效成分=慢端间距的 sink；其 source（压高频密度）对冻结模型必须反转为"桥的渐进性" | [已验证(HighGapToLong 字面版判负) + 假设(反转表述)] | mechanism §496–505,§612–627；UNIFIED §91–93 | P2 的 gap-29 集中被读作"同一 sink 的另一实现"——[假设]，机制归因未分离 |
| C10 | 二力框架（arc restoration vs phase separation）可容纳四个数据点（BM、E2、s28、s29） | [假设→已降级] | plan §3 | 被审计+mechanism 修正：全局"分离度"弦距指标 CPU 判负（非单调+s28 反例）；框架保留形态=带符号内容条件化 margin + 前缀/读取双通路 + 数值实现项 + 决策边界位置 |
| C11 | E1 成功是交互主导的多通路现象（前缀形成 + 读取各 ~1 nat，交互残差 +0.25） | [已验证(三回看个案)] | mechanism §118–146 | 明确限定：三例、非普遍因果；QA cached 路径不保留现象（不可用作解释） |
| C12 | E7 的 160× 低估证否一阶工具 | **被推翻(归因)** | plan §0-2 → audit + local_precision.py | 理想相对 vs 线性同阶（7.93640e-8 vs 7.93668e-8）；160× 属 BF16 实现；端到端失败另有原因（后证修正约束≡BM） |
| C13 | 一阶/固定态代理系统性不可靠（E8 代理>E1 实测崩；Jacobian 外推史 71–468%） | [已验证(多例)] | mechanism §96–102；plan 实验4′ 排除清单 | 成立；定位降为假设生成器（UNIFIED §130） |
| C14 | "零训练+静态+非几何频率表"象限无人占据 | [部分证据(7 篇查新)] | plan §8 末 | 审计补正：MrRoPE 本身即非单调几何的 radix 日程——象限声明需按机制细分 |
| C15 | 50 行 MK 高功效面板是双方共同瓶颈；36 行面板既能假阳性(E3 首筛)也勉强检出真阳性(s28) | [已验证(方法事实)] | plan §2#2 | 基建已备（ruler_full_prepare.py + NLTK 本地化）；面板本身未跑 |
| C16 | 跨模型运输按周期/分数坐标表述（ScaleTaper OLMo 构造检查、E1 5/17→槽19 规则） | [部分证据(构造层) + 已验证(执行完成)] | scale_taper.py scope；mechanism §172–189；BUDGET §63 | 结果数字未归档于本轮文档 = 记录缺口（§7） |
| C17 | MrPro 是本族 N=17 均布成员；E1/LBS 是一阶扰动；N′∈{17,16,15} 是单参数收缩族（跨模型可直接表述 N→N−1） | [已验证(代数构造：MrPro 公式 vs 部署数组误差 ≤4.3e-8)] | BUDGET §41–49；UNIFIED §107 | 族单调性=0448/0449 检验（排队未跑） |
| C18 | 128K 任务分数≠128K 关系直测（校准距离谱 70–120,933 混杂；近目标也可因干扰而难） | [部分证据(字面几何)] | mechanism §440–451 | 成为 Core-C 距离分桶曲线的设计依据 |

## 4. 失败机制清单（试过什么、为什么、复发警告）

- **F1 高频抽预算喂长程（HighGapToLong）**：EVQ 字面移植，两端全输（−17.1/−10.8、0 升）。原因：bank 时钟在 W 内缠绕 ≥36 圈，是整数对齐滤波器；任何压缩直接付短端。警告：训练态密度论证不得平移到冻结模型；任何"发端"提案必须先过 I1 端点检查。
- **F2 平台水平再动（E2_tail_more ÷4.93、MrUni 全表 ÷4）**：分别崩 54.7@128K(12 行)、64.6@32K。原因：÷S 平台=p→p/S 精确重参数化；均匀 PI 不动跨度但破坏 I2 水平。警告：改端点=改流形同一性，与"分布自由度"无关的旋钮根本不存在。
- **F3 慢带置零（E8_zero51，单槽）**：代理友好、实测 −13.9pp、终止/格式型退化。原因：慢槽非死重（至少单槽承重）；代理与端到端脱钩。警告：置零类审计必须配失败分型（误绑定/漏答/终止三型）——沿用此分型法被统一文档评为好实践；整带版（实验 10′）仍未测。
- **F4 固定预算最小粗糙度（Smooth_MrBudget）**：数学上带 KKT 证书、唯一、可跨坐标复现——但 −9.7917@128K。原因：成本函数对方向失敏，把预算从中后段挪走=从危险区抽弧安全（反方向）。警告：凸证书只担保构造性质，不担保任务处方；"平滑"不是目标。
- **F5 局部投影 + 一阶响应（E7）**：128K −9.514pp。三层教训：(i) 线性预测的"160× 失败"实为 BF16 实现误差——先做精度分解再下方法论结论；(ii) 预算与响应矩阵支撑不一致（full-support 预算配局部支撑），修正后约束恰不激活、解≡BM——原实验部分测的是未定义对象；(iii) 局部输出 NMSE 受控≠端到端安全。警告：一阶工具保留"预排序"角色，其量级预测必须用冻结状态有限差分交叉验证（C9/C13）。
- **F6 全局几何代理（分离度/弦距）**：计划 v2 把它升为 #2 实验，审计用 5 点回测（免费 CPU）证死：4sin²(νΔ/2) 对 ν 非单调、随 Δ 变号；s28_less（实测赢家）在 5 距离中 3 个降低分离度。警告：任何 content-free 几何标量在注册 GPU 前先跑已判定点回测门；"回测门失败=交付否证，零 GPU"是本周期最便宜的一次止损。
- **F7 代理晋升过头（NLL/固定态当目标）**：E8 提名增益>E1；两 E1 赢家 NLL≈0。警告：NLL 只做门（Δ≤+0.03 不淘汰）不做目标；小面板首筛的"1胜0负"不算数（E3 gain1 首筛 1胜3负后全面板反转被流程自身抓住——正例示范）。
- **F8 实现层数值 bug 改变科学记录**：oldp·exp(δ) 全项下溢污染 2 条记录 → log-weight 归一修复 → E5 提名 27→32（27 作废，实测层输出保留）。repair_replay.py 明示"computational corrections, not scientific failures"。警告：代理链路的每个归一步骤都要有恒等门（identity nmse≤1e-8、|Δmax|≤0.01 已入 select.run）。
- **F9 手搭局部扰动当候选（G1/G2/G3、BM_ScaleTaper 初版）**：CPU 核算显示 G1/G2 粗糙度同增 20δ²、范数相等、不隔离中心 gap；用户批评"没有比既有有效分配更值得期待的理由"→ 整支撤出活动队列。警告：任意性构造须过"为什么可能赢"的先验陈述关；deferred 而非删除（数据可回收）。
- **F10 二值分数伪交互**：MK 交叉表 0/0/0/1 看似强非线性；token 级 margin 显示加性 + 0.25 残差。警告：交互结论需连续 margin 证据（同一记录即可复算，勿另跑 GPU）。
- **F11 cached 路径≠full-prefill 路径**：QA 行1 的 cached 交叉实验丢掉了 full-prefill 的 E1 成功。警告：机制干预必须逐案例先验证"该路径保留现象"。
- **F12 搜索封顶谬误（计划自带，审计重申）**：自由表搜索 300–500 点失败不能给静态族封顶（无单调性保证）。警告：负搜索结论只覆盖"该邻域该预算"。
- **F13 队列编号复用簿记风险**（0441/0446 双义，T14 注）。警告：改名或 done/ 回执绑定 contract SHA，避免跨文档引用串号。
- **F14 单行驱动的宏平均**（s29_more +8.33pp=一道 QA；native 参照被 2 道 QA 拉动）。警告：任务宏必须随附行级胜负与驱动行披露（worker summary 的 wins/losses 字段是正确实践）。

## 5. 频率表/方法定义清单（名称、构造规则、32K/128K 得分）

### 5.1 参照表

| 名称 | 构造 | 32K/128K |
|---|---|---|
| Native | ν_j=ω_j，m≡0（S=1） | 短 12 行参照 83.3333（NLL 2.27607/2.14317/2.03850） |
| **MrPro**（基准） | ν_j=ω_j·4^{−m_j}；界 23/40（0-based）；m_j=(t(t+1))/(n(n+1)) 二次过渡（t=j−23,n=17），j≥40 m=1；gain=1+0.1·ln4=1.138629436111989 | **87.2222 / 78.1250** |
| BM（MrProBM） | 同端点；过渡段 smoothstep 抛物形 | 91.6667 / 70.8333 |
| FullLagP2 | 历史 64 频率全表（原协议 1.5B/64K），gain .074 | 72.9167 / 81.6667 |

### 5.2 已执行静态候选（含裁决）

| 候选 | 构造规则 | 32K/128K | 裁决 |
|---|---|---|---|
| E1_s28_less | m28 取前邻 m（0.098→0.065；gap27→28 挪 0.045） | 87.2222 / **83.3333**（2W0L） | 开发正候选；确认=0450 |
| E1_s29_more | m29 取后邻 m | 95.5556 / 77.9167 | 短端正候选；单 QA 驱动 |
| E2_tail_more | 槽 40+ 频率 ×1e6^{−1/64}（平台→÷4.93） | 12 行崩 54.7@128K | 负（I2） |
| E2_boundary39/41、E2_tail_less | 界挪/反向 | 未录 | 未测/未记录 |
| E8_zero51 | ν_51=0 | 12 行 −13.889@128K | 负（I1 系、格式型） |
| E3 BM×gain074 / BM×gain1 / (MrPro×.074) | 表×增益析因 | 100/70；89.5833/58.8194；98.3333/75.3472 | 取舍/负/强耦合对照 |
| E4_pair25_29 | 槽 25/29 共享 (m25+m29)/2 | 12 行平 | 无信号 |
| pair(28+29)（组合） | 双槽同时减压 | 87.2 / 74.0 | 超调负（洞 1.46×） |
| E5_layer21(→32 提名) | 仅该层换 BM | 12 行平 | 无信号 |
| E6_layer14_group{0,1} | 仅该 KV 组换 E2 表（含反向控制） | 12 行平 | 无信号 |
| E7_local_projection | BM 方向局部响应约束投影 | +2.778 / −9.514pp | 负；修正版≡BM |
| Smooth_MrBudget | min-roughness KKT @B=16/3（增量的唯一非负最平滑分配） | 87.2222 / 68.3333 | 负 |
| MrUni | 过渡段均匀斜坡 t/n | 32K 64.6 | 负（端点） |
| LongBridgeSlower | 槽 36–39 ν ∓ 1/131072（公共相位 −1rad@128K） | 80.5556 / **80.0694** | 条件正信号（长+1.94/短−6.67）；确认队列 48 生成已排 |
| LongBridgeFaster | 同幅反号 | 87.2222 / 73.9583（0W1L） | 控制：反向不复现 |
| HighGapToLong | gaps0–22 各减 0.2158674/23，gaps36–39 各加 ¼ 预算 | 70.1389 / 67.3611（0W7L） | 负（字面 EVQ 运输） |
| HighGapToMid | 同预算给 gaps26–31 | 16/36 停 | 部分数据保留，无裁决 |
| P2 FullLag（本轮协议） | 历史整表 | 72.9167 / 81.6667；128K 尾512 NLL 1.678507/PPL 5.357549（同 gain 参照 1.680924/5.370516） | 过 128K 初筛；QA pilot 4 行释放中 |
| native 参照 | S=1/gain=1 | 短 12 行 83.3333 | 参照 |

### 5.3 算子/动态类

| 名称 | 构造 | 状态 |
|---|---|---|
| E9_distance | 本地 δ≤w 原生钟；远域 MrPro 钟 + Q 侧相位修正；双 BlockMask、联合 LSE；w=binding 校准窗 | 实现+数值验证完成；**无模型结果**（040 排队被 128K-first 搁置） |
| E10_dual_frequency | 槽 24–39∪88–103 双频 ½ 混合（Q 半权、K160/V128 垫维） | 12 行平 + NLL 分解（−0.005180/−0.012242 vs 算术控制）；36 行 pending |
| D11/G_t row-wise（实验 6′） | G_t=max(1,⌈(t+1)/W⌉) 只作用槽>23，位置映射实现，标准 SDPA | 仅设计；未执行 |
| row-wise NoPE 混合等 | 计划排除清单内 | 未执行 |

### 5.4 构造了但未测（deferred/queue）

BM_ScaleTaper（044d）；G1/G2/G3+046 binding16（044a-c）；A2 界审计 h=42；A4 S=2@64K；MrPro×c=0 缺失对照——全部属 hand-built/扩展队列，随协议转入 `deferred_queue/`。

### 5.5 统一理论新候选（已入队、GPU 关机后冻结）

0446 StackFrontBack、0448 MrProN16、0449 MrProN15（参数见 T14）；判定规则先行冻结（BUDGET §52–54）；随后 0450 E1 holdout16、0451 P2 长端确认。

### 5.6 20×10 并行计划候选/实验总表（v2 原文归类；除引用 codex 数据外均未执行）

- 组 A（失败混杂分解）：A1 形状×预算 2×2 析因；A2 边界坐标审计（h=42/38、l=20/26）；A3 gain 因果消融（MrPro×{0,.074,.1}×长度交互）；A4 尺度匹配 64K@S=2。
- 组 B（反误绑定机制设计）：B5 分离度优先表+5 点预注册回测门；B6 radix 完整性审计（并入 B5 前置）；B7 慢带整带置零审计。
- 组 C（测量驱动）：C8 双侧敏感度门控（压缩{溢出大∩代价小}、解压{代价大∩溢出小}，汇聚检验是否独立提名槽 28/29）；C9 G_attn 一阶交叉验证（预期系统性低估）；C10 门控跨模型（OLMo→BM 式/Qwen→MrPro 式）。
- 组 D（算子类）：D11 带限 row-wise 动态；D12 位置坡道（D11 对照）；D13/D14 GQA 双焦/层级梯度（等 codex E5/E6 全量）；D15 双频改良（第二频取 native——维持不进 10）。
- 组 E（谱系收尾）：E16 FullLagP2 gain 匹配 3B 裁决；E17 重算 p2（并入 E16）；E18 u-坐标运输。
- 组 F（评测科学）：F19 OOD-NLL 校准分析；F20 MK2/MK3 50 行高功效面板（`ruler_full_prepare.py` 即其基建）。
- 实验排序 1′–10′：3′ 面板(#1)→5′ 分离门(#2)→4′ 双侧门控(#3)→1′ 析因→2′ gain/边界→7′ P2 清算→8′ 搜索探针→6′ row-wise→10′ 整带置零→9′ 异质双臂。

### 5.7 状态分类（直接回答任务核心问）

**已在 CPU 判负（不烧 GPU 即出局）**
1. 5′ 全局分离度指标：弦距非单调 + s28_less 三距离反例回测 → 门失败（[已验证]审计计算；后续须以带符号内容条件化 margin 重述）。
2. E7"160× 低估"方法论判决：精度分解改判 BF16（[已验证]）——原主张作为论据作废；E7 本身仍是 GPU 实测负。
3. "P2 的长端胜利=高频运输兑现"推断：P2 只从高频取 .0007754 vs 需 .2158674（280× 差）（[已验证]算术）。
4. G1/G2/G3 gap 探针对称性：粗糙度同增 20δ²、不隔离中心 gap、等扰动范数 → 判别力不足 + 用户批评 → 撤（[已验证]CPU 核算；[假设]撤稿主因=用户候选质量批评）。
5. "固定预算最小粗糙度"作为处方：GPU 实测 −9.7917 补强；但其 KKT/唯一性构造本身 CPU 可验——失败属机制层，非构造层。
6. 搜索封顶谬误、二值分交互谬误、cached 路径解释力：CPU/回看层面即纠（F10/F11/F12）。

**GPU 已实测——负**：HighGapToLong、Smooth_MrBudget、MrUni、E2_tail_more(12 行)、E8_zero51(12 行)、E7、BM×gain1、LongBridgeFaster、pair(28+29)（超调）、BM@128K 本身（相对 MrPro 长端 −7.3）。
**GPU 已实测——正/取舍（均开发集）**：E1_s28_less（128K +5.21）、E1_s29_more（32K +8.33）、LongBridgeSlower（长 +1.94/短 −6.67）、P2 FullLag（128K +3.54/+6.32 vs 两参照）、E3 BM×gain074（32K 100/128K 70，纯取舍）。
**持平/无信号**：E4、E5、E6、E10（12 行）、native 参照短 12 行低于 MrPro。
**还等 GPU（全部 [未执行]）**
1. 冻结队列 0446→0448→0449→0450→0451（StackFrontBack 叠加性、N′ 族单调性、E1/P2 新样本确认——约 6–7 小时，恢复即跑）。
2. E9 距离双时钟（040 位排队；算子+稠密验证完备，仅差模型评测）。
3. E10 36 行全面板 + bf16 端到端确认（12 行平 + NLL 分解不构成裁决）。
4. HighGapToMid 补完（或正式作废）；64K 长度维（八条抨击 #3 唯一仍成立的缺口）。
5. 并行轨 10 实验中除引用 codex 数据者外全部未跑：3′ MK50 面板（基建就绪，NLTK 路径已本地化）、1′ 析因、2′ gain/边界、4′ 双侧门控、6′ row-wise、7′ P2 清算（其一半已由 P2 同 gain 参照实质完成）、8′ 搜索、9′ 异质、10′ 整带置零。
6. 7B/OLMo 运输的数字归档缺口（跑过未录入本轮文档）。

### 5.8 与 bank/arc 二分法的关系：替代机制还是兼容成本项？

判据：bank/arc 二分法（UNIFIED §2）是在**静态单表 ν_j=ω_j·4^{−m_j} 类内**刻画"额外 ln S 落点"的代价结构——bank 段（槽≤~29）落预算=无限成本（I1），arc 段/危险区完成到 m=1=低成本（÷S 精确重参数化），桥宽=唯一设计参数（洞预算定价）。据此分类：

1. **类内兼容成本项手术（不是替代机制）**：全部已赢/已输的静态表候选都是在给 bank/桥/危险区三段重新定价——s28_less=bank 边缘保真（前端）；LBS/Slower/N′ 族=危险区 arc 完成（后端）；P2=最窄桥+极早完成（双机制极端混合）；Smooth/HighGap/MrUni/E2/E8=违反 I1/I2 的反方向手术。赢家的共同性（BUDGET §58 单向梯度）恰恰**是**二分法的证据，而不是它的替代。
2. **正交补偿项**：gain/温度（E3、实验 2′、16′）只改 logit 幅度不改相位弧——析因证明其与表效应可交换；属"兼容成本"里的预算外幅度维度，长度耦合强但不能救表（−8.13@128K 连 gain 一起输）。
3. **真·替代机制（脱离单表类，二分法零和断言对其不自动适用）**：E9 距离域双时钟（同槽按 δ 选钟，试图同时买 bank 精度与 arc 安全而不付洞）；E10 双频核（每通道两频、logits 合并，绕开单选 ν）；D11 row-wise G_t（按上下文长度解耦，窗内恒等 Native 由分区不变性保证）。这三者是 20×10 计划"出路 2：按长度解耦"与 codex E9/E10 的合流——用户授权口径"赢 MrPro 即可重构全部理论"是其保留理由。现状：**全部未决**（E9 无分数、E10 平、D11 未跑），既未证替代有效、更未证无效——[假设]级，不得写成否证。审计附加先验：E9 kink 两侧压缩率差 4× 有 GapCapped 式跨 kink 误绑定风险；E10 混 BM 成分的先验因 BM@128K 净负而下调。
4. **异质轴（E4/E5/E6、D13/D14、10′）**：把 bank/arc 代价改在层/头/KV 组粒度落地——是二分法的**空间推广**而非替代；首轮持平（12 行），信息量上限被计划实验 10′ 明确标注。
5. **二力框架的历史位置**：它是 bank/arc 的前身表述——arc restoration 存活为 arc 钟窗内容忍性，phase separation 作为全局标量被 CPU 判负后，降级为"内容条件化带符号 margin + 前缀/读取双通路 + 数值实现项 + 决策边界"（AUDIT/§3-C10），二分法是其几何化、可预算化的稳定形态。

## 6. 用户指令与纠正（原文引用）

1. README.md（"Latest user-directed protocol (2026-09-10)"）："focus on 128K. Reuse valid passkey results and screen target-length PPL before releasing a small long downstream test. Do not automatically complete 32K panels or release the old large holdout jobs. The hand-built gap/pair/taper branch and the earlier expansion queue are retained under `deferred_queue/`, not active. P2 passes the initial 128K screen; only a four-row fresh 128K QA pilot has been released."
2. 任务口径（plan 头，"用户 USER #87 + #88 纠正"）："零训练，找最佳非几何分配方式；复用历史 baseline（归档 `results/bm_transfer_20260908/run_qwen3_01/MrPro.jsonl`）；NLL 和 passkey 轻微输不淘汰，继续混合 RULER；赢 MrPro 即可重构全部理论。"
3. TEN_CANDIDATE_PLAN："User subsequent correction is already incorporated: **NLL and passkey are initial screening and capability dimensions, and do not have the power to singly veto a candidate with slight degradation. Existing mixed RULER is the core evaluation.**"；"Second user correction: research should serve a paper that readers can understand and evidence can support, and does not demand beating all models, all tasks."；E9/E10 保留语："per user's explicit permission '赢 MrPro 即可重构全部理论'（as long as MrRoPE can be beaten, theory can be reconstructed）"。
4. mechanism §327–336："The user identified an important error in the recent research emphasis: fixed budget, smoothness, and neighboring-gap geometry were being analyzed without making protection and improvement of long-distance interactions the primary criterion."；"The user explicitly cautions against turning a local long-distance diagnostic into the entire research program."
5. mechanism §489–491："The user's latest correction is to investigate a resource exchange: tolerate some loss at short and intermediate scales if high-frequency capacity can be reassigned to produce a meaningful long-context improvement."
6. mechanism §22–25（协议纠正/撤回）："**Current priority correction (2026-09-10, after the user's candidate-quality critique):** the hand-built HighGap, G1/G2/G3 pair-gap, and BM_ScaleTaper branch has been withdrawn from the active queue … The mistake was proposing arbitrary local redistributions from a verbal analogy without a sufficient reason to expect improvement over existing effective allocations."
7. prepare_gap_probe.py 头注："source: User-provided GPT-5.6 Pro analysis, 2026-09-10"（G 三臂与 block-stretch 假说为用户转交的外部分析件，非自主生成）。
8. 工作纪律（本任务铁律，AGENTS.md/失败复盘）：不得把代理指标说成能力结果；不得把未测写成否证；每个结论标注证据等级；引用给出路径/时间戳。

## 7. 未决问题

1. **0446/0448/0449/0450/0451 冻结队列结果**：双机制可叠加性、N′ 族单调性、E1/P2 新样本外推——是把"方向一致梯度"(C8) 升为论文级证据的既定路径；全部悬置中，且判定规则已预冻结，不得事后挪门。
2. **E9 距离双时钟**：唯一带完整实现、零模型证据的替代机制。跑之前仍需审计要求：kink 处跨域记录对误绑定检验、w 的选取敏感性、batch-1/无 padding 约束下的吞吐成本、"行定义"。
3. **E10 双频核全面板**：12 行平 + bf16 relative_rms 1.68% 是否掩盖任务差异；K160 +12.5% KV 成本 vs 收益的口径要先定。
4. **64K 长度维**：抨击 #3 唯一仍成立的结构性缺口；64K@S=2 与 MK50 面板都未跑；"64K 谷"的尺度错配假说（实验 2′）无数据。
5. **s28_less/LBS/P2 的机制归因未分离**：前端保真 vs 后端完成的"两独立机制"读法是 [部分证据]；StackFrontBack 之外仍需按证据距离分桶的距离-准确率曲线（Core-C 预言：MrPro 在 75–112K 带衰减、completer 保持）。
6. **bank/arc 阈值的定量地位**：r≳8–10 vs ≲8 的分界、"洞预算决定桥宽"的定价曲线（原生 1.241 参照）都只是拟合当前 27 面板的形状，未做独立预测检验；危险区定义（r∈[1,2.2]）同样是本模型刻度。
7. **跨模型数字归档缺口**：7B/OLMo E1 运输"已完成不再扩展"，但其 32K/128K 数字不在本轮任何已读文档——写论文前必须从 transfer/ 回执补齐或明确弃用。
8. **HighGapToMid 处置**：16/36 停止的部分数据保留但既无裁决也未正式作废；0446 编号复用串号问题同此。
9. **"多少慢槽可牺牲"整带置零（10′）**：E8 单槽已证明慢槽承重，整带版是唯一有功效的下一步（15 min 成本），仍未跑。
10. **理论边界**：C6 零和断言只对静态单表（及因子化核）成立——若 E9/E10/row-wise 任一成功，"必须砍桥"要改写为"桥的代价函数在该算子类中被重新参数化"；目前把两者都留在 [假设]。
11. **失败分型自动化**：误绑定/漏答/终止三型目前是人工阅读生成文本；若要作为筛后规则，需要可复算的判据。
12. **P2 QA pilot（0441）结果**与 LongBridgeSlower 48 生成确认队列（VT16+MK16×2）同样悬置在关机点之后。
