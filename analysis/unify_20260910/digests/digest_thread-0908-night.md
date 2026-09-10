# digest thread-0908-night（09-08 深夜并行会话簇，22:49–23:59 启动的 8 个 codex 子代理会话）

> 生成时间：2026-09-10。提取方式：按统一脚本从 JSONL 提取 `response_item/message(user|assistant)`、`agent_message`、`task_started/task_complete`、`session_meta`，并额外提取全部 `send_message_to_thread` 工具调用入参（子代理完整清单只存在于这些调用里，纯消息提取会丢失）。时间戳为本地 America/New_York。
>
> **前提校正（重要）**：任务下发口径称本簇为"非几何候选并行探索"。核对 `session_meta.source.subagent.thread_spawn.agent_path` 后确认：**这 8 个会话是父线程 `01a0806f-3df5-74b1-bc56-bf00d89d238e`（paper-2027 稿件工作线程）派出的"稿件论证审查 + 实验资产盘点 + PDF 成稿审稿"并行簇**，全部为 depth-1 只读子代理。真正的"非几何候选"探索计划文档是 `docs/research/NONGEOMETRIC_TEN_CANDIDATE_PLAN_20260909.md`（09-09）与 `docs/research/PARALLEL_NONGEOMETRIC_20X10_PLAN_20260910.md`（09-10），不在本簇内。但本簇盘点了大量**候选机制的正/负结果与失败证据**（BM、normalized-index/K128/K32、PSR、dose-response、isotonic、C2、scale-cap），这些正是后续并行候选探索需要的"已试过什么、为什么失败"底账，因此本 digest 对该部分内容全量保留。[已验证：8 个文件的 agent_path 逐一读出]

## 1. 来源清单

| key | 启动时间(ET) | 结束(ET) | agent_path / 昵称 | 源文件大小/行数 | 提取文件 |
|---|---|---|---|---|---|
| night01 | 09-08 22:49:30 | 23:16（2 turns） | `/root/argument_reader_check` / Ampere | 3,672,835 B / 342 行 | `raw_night01.txt` |
| night02 | 09-08 23:14:12 | 23:21 | `/root/asset_foundations` / Descartes | 2,245,624 B / 240 行 | `raw_night02.txt` + `raw_night02_tools.txt` |
| night03 | 09-08 23:14:33 | 23:22 | `/root/asset_scale_architecture` / Jason | 3,083,315 B / 264 行 | `raw_night03.txt` |
| night04 | 09-08 23:14:51 | 23:22 | `/root/asset_llama_adaptation` / Kepler | 2,248,673 B / 186 行 | `raw_night04.txt` |
| night05 | 09-08 23:15:11 | 23:22（含 23:20 中途 MESSAGE） | `/root/asset_frozen_models` / Aristotle | 4,962,195 B / 301 行 | `raw_night05.txt` + `raw_night05_tools.txt` |
| night06 | 09-08 23:15:30 | 23:29 | `/root/asset_mechanisms` / Mencius | 4,555,915 B / 364 行 | `raw_night06.txt` + `raw_night06_tools.txt` |
| night07 | 09-08 23:46:23 | 09-09 00:04 | `/root/second_manuscript_review` / Hilbert | 3,754,256 B / 273 行 | `raw_night07.txt` |
| night08 | 09-08 23:59:44 | 09-09 00:09 | `/root/pdf_review_r01` / Poincare | 8,365,827 B / 183 行 | `raw_night08.txt` |

- 全部子代理的父线程：`01a0806f-3df5-74b1-bc56-bf00d89d238e`（本机 main 线程，cwd 均为项目根）。
- 会话任务指令（NEW_TASK/中途 MESSAGE）以 `agent_message` 传入，**payload 为 Fernet 加密**（`encrypted_content`，无法在本机解码）。可获得的指令信息只能从子代理对任务的复述与产出反推；本文所有"目标"字段均标注为转述推断。[已验证：加密字段直读；未解出任何原文]
- night08 审稿对象：`paper-2027/main.pdf`（编译稿；git 工作树当前显示 `paper-2027/main.pdf` 为 modified）。
- night07 交付物：`paper-2027/research/EXPONENT_SECOND_REVIEW_20260909.md`（35,131 B，写入于 09-09 00:03；本 digest 直接引用其第 17 行"3/6 Borderline Reject，接近 4/6"裁定）。

## 2. 任务时间线

### night01 / Ampere（argument_reader_check）— 成功（2 个 turn）
- 目标（转述自其开场白）：对 paper-2027 当前稿做只读"论证链 + 数学表达"审查——普通 ML 读者能否抓住主张 + PE/RoPE 数学是否自洽；不做 venue 决策；旧意见仅作待验证线索。
- Turn 1（22:49–22:58）结果：判定"第一页已能准确回答研究什么/为什么/怎么验证；数学主干自洽；没有把坐标重写冒充核心理论贡献，也没有把静态几何直接当成模型性能的硬跳跃；防御性措辞已收敛"。给出 4 个"确定需要处理"的措辞修正（见 §3 C1）+ 1 个 comparator 澄清（`improves Qwen's 64K RULER score` → `improves over matched YaRN at 64K`）。
- Turn 2（23:09–23:16，父线程追加 NEW_TASK）：还原旧 NeurIPS 3 分初审 + rebuttal + AC 门槛，按 NeurIPS 2026 六档量表复审。**判 5/6 — Accept，confidence 4/5；质量5、清晰度5、重要性4、原创性4**；逐条表格复核 8 项旧评审问题全部"已解决"（RDz6s.2 optimized Geo+YaRN 一项记"部分解决，但已不构成对现稿主张的反驳"）。另给 3 个最影响接收的剩余写作问题（intro 仍写 "variational problem" 应改 "convex surrogate variational problem"；第三项贡献应改写为 model-relative displacement profiles + BM 构造的正面对象；OLMo confirmation 与 Qwen development 同表需 caption 分层级）。
- 关键数字（其复核中引用）：454M `157.7→107.5`（fixed-index s=8 scaler 口径）；99-run receipt（a5:157 的 0.75/1/1.25× Cosh + matched exponential）。

### night02 / Descartes（asset_foundations）— 成功（F1–F9 受控训练/分配资产盘点）
- 目标（转述）：盘点"基础受控训练/指数分配"资产，判断哪些实验真正可区分、数据口径、稿件覆盖；只读，不改文件不跑模型。
- 结果：完整 F1–F9 清单两次发父线程（23:19、23:20 全文版）+ 23:21 摘要版 + 23:21:32 口径补充。**核心结论**：F1（151.9M exact-range 三 seed）是最强纯 allocation 因果证据；F4/F5 属同一 128-token campaign 不能重复计数；F6/F7 同 Phase11B family；**F9（旧 350M/454M 记录）存在模型身份与数值来源冲突，应暂不纳入证据榜单（audit lead only）**。
- 计数规则（可复用）：Top15 实验身份时，F2 全部 arms/exp-control/boundary 只算一个 family；F4+F5 算一个 campaign；F6+F7 算一个 Phase11B family；F1 的 target-matched 列不是第二实验。
- 详见 §3/§5 各条目数值。

### night03 / Jason（asset_scale_architecture）— 成功（S1–S8 规模/架构/跨模态资产盘点）
- 目标（转述）：按证据强度整理规模/架构/模态实验为"可写 claim"，区分已实施结果、代理指标与可写结论；"不把缺少 raw artifact 自动等同于不存在"。
- 结果：8 项可区分实验 S1–S8（数值见 §5），并给出正文/附录缺口清单：
  - **最大遗漏 = S1（454M 三 seed EVQ×固定索引 scaler 四臂）**：`table_evq_ramp.tex` 存在但 `main.tex` 未引用，caption 还引用不存在的 `sec:ramp-scaler`；EVQ+scaler 8K teacher-forced passkey `100±0%` vs Geo+scaler `61±3%`；PPL@8K/16K `70.9/107.5` vs `82.9/157.7`。
  - 孤儿资产：`table_evq_ramp.tex`、`a4_supporting_experiments.tex`、`fig4_phase17c_flagship.pdf`、`fig_range_composition_454m.pdf` 均未完整进入 `main.tex`。
  - 不要升级：125M GQA/MLA 单 seed、phase17c 单 seed、video 单 seed 只能 supporting；**454M QuALITY full-eval 准确率接近 25% 随机基线，只能作 negative/downstream diagnostic，不能拿旧 n=200 pilot 的夸大 accuracy 当 claim**。
  - 规模叙事边界：1.485B 是 same-initialisation/same-recipe persistence；750M 是 shared-Geo-checkpoint continuation；**两者不能合并成 multi-seed scaling law**。

### night04 / Kepler（asset_llama_adaptation）— 成功（A1–A8 成熟模型适配资产盘点）
- 目标（转述）：盘点 Llama-3-8B/OLMo 成熟模型 adaptation 证据；同一 run 的多 endpoint 合并为单项证据，区分 report/curated/raw。
- 结果：重大遗漏 4 项 + A1–A8 排序清单（数值见 §5）。要点：
  1. Llama-3-8B matched RULER（A1）最该升主文：Native-LoRA official 8K `94.44%`、16K `0.295%`；EVQ-LoRA 8K `77.60%`、16K `14.03%`（normalized exact EVQ 8K 21.54% > Native 17.69%）。
  2. **OLMo final full-answer+EOS lineage（A3）完全遗漏**：EVQ/Native 4K `100/95`、8K `98/18`、16K `60/0`；当前稿只用了较弱的 routing first-number `69/100`。
  3. OLMo matched RULER（A2）只有表没有正文 claim：Native 4K `82.16%` vs EVQ `37.51%`；8K EVQ `21.29%` vs Native `0.08%`；16K EVQ `6.13%` vs Native `0%`。
  4. **任务族监督 vs unseen-task 边界没有放在正结果旁**：A8 clean 4K Tulu/LongAlign→full RULER 负结果（0/39 cells beat control；4K `0.0974 vs 0.6535`）证明**低 natural-text NLL 不能推出 broad autoregressive capability**，应进附录 limitation。
- 从属去重声明：A5 temporal 曲线 + source intervention 是同一 seed-42 run 的 subordinate endpoints；A4 的 QA 与 RULER 是独立训练 adapters（不同 seed 20260728/20260729），不能当一个 run 两个 endpoint；A7 是 A3 链的 retention check。

### night05 / Aristotle（asset_frozen_models）— 成功（冻结模型资产 15 项盘点 + K128/K32 身份核验）
- 目标（转述）：盘点冻结模型 curated/raw JSON 与 attention-aware 证据，去重同一 run，输出 ≥5 个可区分资产及落点建议；不跑模型不下载。
- 结果（首轮 send_message 23:19 长清单 + 23:20 排序版 + 23:21–23:22 三轮身份/构造核验）：
  - 已入稿：OLMo unseen-nine 16K fixed-support（uniform `.56`、coarse ramp `61.04`、derived `60.47`、Native 0、YaRN .0794）；Qwen core4 64K（`57.75/64.00/66.50`、YaRN .6025）；Qwen0.5B full13 normalized-index 64K `51.4551 vs 45.3654`（+6.0897pp CI[2.7627,9.5835]），32K `-0.0256` 无差异；BM OLMo seed20260910（4K `81.81 vs 37.85`，16K `51.32 vs 2.78`）；OLMo natural BM 778/arm（+3.8194 CI[1.316,6.292]）。
  - **遗漏且最高价值**：K128 Gemma index-vs-physical 独立 confirmation（index `.790` vs physical `.728125`，+6.1875pp CI[2.8109,9.6250]，seed202609028，80/task×4=320/arm，16K）；K32 paired crossing（旧 physical 优势不复现：32K `-.036875` CI[-.0775,.0025]、64K `+.005` CI 跨零；index 过 Native retention `.9232` 而 physical `.8595` 不过）——**这条更正任何 "physical-x privilege" 说法**。
  - 身份核验结论：K128 两臂除 active table SHA 外全部 byte-frozen matched（gain 精确 `1.102585782722872`）；**K128 新 rows 上没有重跑 Native/YaRN，prior panel manifest 不同（`d5380f1b...` vs 新 `244a...`），不得写成同一新 rows 四臂比较**。K32/K128 的 Lref/s/g 是设计上不同（K32 b=1e6, Lref=32768, s=2, g=1.0512929；K128 b=10000, Lref=4096, s=4, g=1.1025858），只能说"冻结 law constants（xH/xL/c）与 within-panel 端点/增益 matched"。
  - 其余遗漏资产：reference-corrected K128 s2/s4 曲线、cross-model BM natural-NLL（OLMo BM−MrPro 16K `-.8259`；Qwen3B 16K `+.00425` CI 全正；Qwen7B 全 CI 跨零）、BM 128K cross-cache 机制诊断、isotonic endpoint tradeoff、allocation dose-response、low-dim C2、scale-independent g s2、BM scale cap。
  - 失败/未定项点名（低优先或非结果）：`DIRECT_Z` 在 downstream 前停止；`K32 packed-natural data NOT_RUN`；`PHASE_ISOTROPY/PHASE_ALLOCATION SCREEN_UNRESOLVED`；`HEAD_SELECTIVE exact candidate negative`；`FAR_PASS_CHORD internal negative`。
  - 建议优先级：P1 = K128 confirmation + K32 crossing/YaRN（一个 subsection，纠正 physical-privilege）；P2 = cross-model BM natural-NLL 表（model-dependent transfer 反重量）；P3 = dose-response/isotonic 进 appendix limitation；C2/reference-corrected K128 有空间才放。

### night06 / Mencius（asset_mechanisms）— 成功（M1–M11 机制/竞争资产盘点 + 一次自我纠错）
- 目标（转述）：机制与竞争性证据资产清单，保留单种子、冻结反事实、未打包 raw 等边界。
- 结果（23:21 与 23:23 两版全文 + 23:29 关键纠正）：
  - **重大遗漏 4 条**：(1) 当前稿没有"静态几何为何不能直接预测 LM、与 task-sensitive attention/gradient 的关系"的可见实证桥——应把已存在的 CPU-only 50M attention-Fisher/LM-gradient probe 与 per-pair NoPE/swap 因果谱作为机制边界/负结果短段或补图，**不能写成训练前 selector**；(2) base-only frozen control 完整数值未呈现；(3) 151.9M 权重×表 crossing 藏在附录，应进 compact mechanism panel；(4) **sparse/PSR 是独立路线的失败/未定结果，与 exponent-allocation 主线断开，不要塞入主线主图**。
  - Top 排序（机制/竞争子集）：1 M2 exact-range 151.9M；2 M1 full geometry+collapse；3 M3 50M crossing；4 M4 151.9M crossing replication；5 M9 mature same-support z；6 M8 factorial/Exp 竞争；7 M7 per-pair spectrum（负机制）；8 M5 base-only；9 M6 attention-Fisher/gradient；10 M10 dose-response。单一机制主图方案 = M3+M4 双 heatmap + M1 slow-collapse inset + M7 static-proxy-fails strip。
  - **自我纠错（23:29）**：初版把 151.9M crossing 的 raw owner 写成 `K32_PAIRED_CROSSING_CONFIRMATION_RECEIPT_20260901.json`，随后更正为 `SAME_SUPPORT_FROZEN_CHECKPOINT_RESULTS_20260823.json` 的 `small_model_crossing`（raw receipt SHA256 `9304752d885d…1996e22`）；20260901 K32 receipt 是另一项 Qwen crossing。并给出精确 NLL 单元、interaction 定义 `[L(WF,TC)-L(WF,TF)]-[L(WC,TC)-L(WC,TF)]` 与四个 checkpoint hash。→ 这是"资产 owner 归属易错"的一手案例。
  - M11 sparse/PSR 细节：`experiments/native_sparse_position/RESULT_20260908.md` 明确未建立 position-specific advantage 或 novelty；32-row：Dense 17/32 exact+EOS、RoPEMean 0/32、Quest 3/32、PostMetric4 10/32、PreMetric4 11/32、MatchedContiguous4 1/32；更强 32-token Quest 13/32 vs Post 3胜6负、Pre 3/5；记录为 "failed quality-cost improvement claim"，`CORE_DIAGNOSIS.md` 称路线仍 open 而非 rejected。

### night07 / Hilbert（second_manuscript_review）— 成功（二审报告落盘）
- 目标（转述）：对现稿做第二遍稿件审查，逐句核对主张、证据、历史审稿反馈，报告写入指定文件。
- 结果：`paper-2027/research/EXPONENT_SECOND_REVIEW_20260909.md`（35KB）。**判 3/6 Borderline Reject（接近 4/6）；需要实质结构重排，无需新增模型实验；报告外未修改任何文件。** 核心判断（其 23:58 独白）："现稿的数字和大多数逐句限定比旧稿严谨，主要风险来自**全篇证据层级和章节顺序仍把 Cosh 塑造成中心方法**"。报告含"对作者最新批评逐项裁定"表（第 24 行起）与"主图/主表重排"（第 232 行起）。
- 与 night01 对照：**同一晚、同一稿，Ampere 判 5/6 Accept，Hilbert 判 3/6 Borderline Reject**——评审结论显著分歧（见 §3 C2、§7 Q3）。

### night08 / Poincare（pdf_review_r01）— 成功（编译 PDF 成稿审稿，7 项分级问题）
- 目标（转述）：以 PDF 为唯一证据源完整读正文+附录，渲染关键页核对图表/公式/版式，先报影响主张成立性的硬问题。
- 结果：逐式检查 Proposition 1、Theorem 2/3/5 及附录证明，**未发现会推翻结论的公式硬错**；固定端点 151.9M 配对实验为全文最有说服力因果证据。独立录用倾向"边界，略偏可接收"，信心"中高"。7 项问题（3 主要/3 中等/1 次要）：
  1. [主要] 解析推导只定义带自由参数 τ 的分配族，**尚未定义统一免选择的部署方法**：Eq.(39) 的 τ=√(β/α) 无任务/模型确定 β/α；实测混用公式约定、pre-specified 经验值（MLA 1.414）、sweep 选优（Video DiT 文本式参考 2.83 却报最佳 1.5）。建议统一称 "analytic Cosh family" + 协议级 τ 表。
  2. [主要] **8B headline 选择性突出 32K 改善（PPL 991.48→127.91）却省略同协议 8K 退化（6.82→10.07）**；且 Native/midpoint-Cosh 有限网格端点不同，不能归因纯 interior allocation，应改为 "matched end-to-end table/adaptation effect"；RULER 侧同样有 8K 94.44→77.60 的代价。
  3. [主要] **50.9M factorial 文字结论强于统计证据**：reference Cosh vs Geo 仅 7/12，CI[-0.021,0.002]，sign-flip p=0.125；matched exponential 9/12 p=0.071；只有 1.25× Cosh p=0.027 且未校正多重比较。
  4. [中等] Cosh 变分目标是作者规定的设计偏好，未证明等价于最大化完整子空间有效秩，更未推出 LM 泛化；§A.6 只查 cosine slice 而 §A.3 自己证明 cosine-only ordering 可与 full-subspace ordering 相反。
  5. [中等] 统计单位混淆：多数 breadth 结果单训练对/单 seed；document/sample bootstrap 只量化条件于该训练对的评价行不确定性，不能代替训练随机性；主表应加 "training replications" 与 "inference scope" 列。
  6. [中等] 部分构造在 PDF 中不精确：§C.1 target-matched 无 `R(L_eval)` 公式；§C.2 factor-four crossing runtime table 离散构造不明确（同一张表是否同时用于 512/1024）；§A.3 数值反例未附频率表，PDF 自身无法检查存在性例子。
  7. [次要] 摘要需点名 21.62→25.44% 的 comparator 是 MrRoPE-Pro；Figure 1(b) 措辞（图内含恶化 1× 点）；§6.3 明写 95% paired bootstrap；首次展开 EVQ 缩写；Figures 5–6 字号过小。
- 新增实验必要性判据（仅当保留更强主张才需要）：Eq.(39) 跨架构 zero-search 规则→新模型预注册 c=1 配对训练种子；规模泛化→独立适配/训练种子；完整子空间几何为机制→预先几何排序预测 OOD NLL。

## 3. 理论主张表

| # | 主张 | 证据等级 | 出处（本簇内） | 后续是否被纠正/推翻 |
|---|---|---|---|---|
| C1 | 现稿"数学主干自洽，无静态几何→模型性能的硬跳跃"，防御措辞已收敛 | [部分证据]（只读审查判断） | night01 Ampere 22:58 | **部分推翻**：night08 Poincare 发现 8B headline 省略 8K 代价、50.9M 文字强于统计（其主要问题2/3）；night07 判证据层级仍把 Cosh 摆在中心 |
| C2 | 当前稿达到 NeurIPS 5/6 Accept（conf 4/5） | [假设]（单评审意见） | night01 Ampere 23:16 | **同夜冲突**：night07 Hilbert 3/6 Borderline Reject；night08 Poincare "边界偏可接收、conf 中高"。未在同夜仲裁，留待父线程（§7 Q3） |
| C3 | 固定端点/span 后只移动 30 个 interior exponents，训练后行为改变（allocation 是独立因果设计轴） | [已验证]（151.9M、3 seeds、499,974,144 tokens/arm、7,629 steps；256/512/1024/2048 tail-NLL +.026/−.281/−.176/−.146；OOD 3/3 seed 同向） | night02 F1/F3、night06 M2 | 未被推翻；night08 认可为"最有说服力因果证据"；target-matched 反转 3/3 被 night06 记为 support×allocation interaction 证据 |
| C4 | allocation 方向跨 base×L_train×d_head 存在但非 Cosh 独有：1.25× Cosh −0.0121（10/12, p=.027 未校正）；Cosh rule−matched exponential +0.00074（CI 跨零，p=.836） | [部分证据]（方向性、配置依赖） | night02 F2、night06 M8 | night08 主要问题3 明确警告：文字若写成"跨形状稳健成立"即过强；reference Cosh 自身 7/12、p=0.125 |
| C5 | 静态 r2/collision 不能预测 LM 性能；weights 与训练表 co-adapt | [已验证]（50M 冻结 2×2：GeoW+EVQT 时 static r2 由 4.57 升到 12.54 而 PPL 7.14→76.20；interaction −3.5367 CI[−5.165,−3.039]≈5.9× main effects；151.9M 2-seed 复核 interaction 3.400/3.251） | night06 M3/M4 | night06 自我纠正了 raw owner 归属（不影响数值结论）；50M probe 的 output JSON 在 /tmp 未入匿名包，只能作内部机制/附录证据 [部分证据→落点降级] |
| C6 | base-only 补偿：geometric base family 可追回 Geo→EVQ 冻结 loss gap 的 74.7%（LS-fit base8.06K：PPL 76.20→9.63），残余为 non-geometric shape+co-adaptation | [部分证据]（单 seed 冻结、非从头训练） | night06 M5 | 未被推翻；边界声明保留："不等于 from-scratch base-only equivalence" |
| C7 | mature checkpoint 内 normalized interior z 仍因果：OLMo 16K derived `.6047` vs Geo `.0056`（+0.5992 CI[+.5488,+.6480]）、vs 最近无标签 ramp `.6104`（−.0056 CI 跨零）；Qwen64K derived `.6650` vs Geo `.5775` | [已验证]（每模型单冻结 checkpoint；derived 与 ramp 不可区分→不能称 derived uniquely causal） | night05、night06 M9 | night05 明确 CI 只条件于 rows；Qwen 旧 128K `.6175` 因 aliased **作废**，corrected derived `.5400` vs Geo `.4550` |
| C8 | normalized-index 坐标优于 physical-x（Gemma K128,16K：.790 vs .728125，+6.19pp CI[+2.81,+9.63]） | [已验证（单 checkpoint 单长度）]，上限"一个 K128 ordering 被确认，拒绝 physical-x privilege；不确立 index 普适性" | night05 | K32 crossing 显示 **physical 的旧优势不复现**（64K 差 +.005 CI 跨零）；K128 新 rows 无 Native/YaRN 对照，禁止四臂混写（night05 身份核验） |
| C9 | BM（boundary-matched intermediate-band）收益 checkpoint/长度依赖：OLMo 4K/16K 大胜 MrPro（81.81/51.32 vs 37.85/2.78）；Qwen3B 128K 反转为负（70.83 vs 78.13）；Qwen7B 32K 80 vs 83.33、128K 71.11 vs 84.44 | [已验证]（跨 checkpoint 相反排序，即论文保留的反例） | night05 | 未被推翻；3B/7B 为 6-task screen 小 n（3B n=4/任务长样本，7B n=2/任务、chunked MLP）→"无 universal transfer"边界[部分证据强度限定] |
| C10 | per-pair 因果谱：Q/K norm/phase covariance 不能预测 held-out causal importance（12 cells median Spearman −.167/−.045）；保多重集 band swap 仍改 NLL（22/24 cells ≥.05）；Geo/EVQ 长程问题更像 OOD phase interference | [已验证（负结果，单 seed 冻结，tail-NLL oracle，无 downstream capability）] | night06 M7 | 未被推翻；明确判定 `DOES_NOT_SUPPORT_BAND_BRIDGE` / `HELDOUT_PRUNING_SIGNAL`；只可进 Appendix/Discussion，不能写成 EVQ 正机制 |
| C11 | 解析 Cosh 冻结 retrofit 有 full-vs-tail 剂量权衡：λ=.02 时 16K tail −.1153 但 4K full +.0169 CI[+.0153,+.0185] 破 +.01 guard；learned oracle 方向也非免费（λ=16 时 4K full +.5294） | [已验证（单 OLMo checkpoint 负结果）] | night05、night06 M10 | 支持"static r2 max ≠ behavioral optimum"；无 finished method |
| C12 | OLMo clean 4K Tulu/LongAlign→RULER 负 guardrail：无 binding/routing supervision 时 4K-only EVQ adaptation 不产生 broad RULER transfer（0/39 cells beat control） | [已验证（负结果，单 seed）] | night04 A8 | 是"低 NLL 不能推能力"的核心内部反例；night07 把它列为必进 limitation 的材料 |
| C13 | 自回归能力证据分层：8K full-answer+EOS EVQ 98/100 vs Native 18/100（16K 60/0 vs 0/0）强于 first-number routing 69/100，后者又强于 teacher-forced NLL | [已验证（同 task family、单 seed、显式长 phase 暴露——边界三条）] | night04 A3/A6/A5 | 不得把 first-number 冒充 full-answer；不得把 phase-exposure 写成"无长位置暴露训练"；K 端 teacher-forced/source-use 不得写成 benchmark accuracy |
| C14 | Video DiT 跨模态支持（128-frame far extrap MSE −35.42%） | [部分证据]（单 seed，τ=1.5 为 sweep 选优、非 Eq(39) 参考值 2.83） | night03 S5 | night08 主要问题1 点名 τ 口径混用；只能标探索性支持 |
| C15 | 454M 四臂 EVQ×scaler（EVQ+scaler 8K PK `100±0`、16K PPL `107.5`） | [已验证（3 seeds curated JSON；但 passkey 为 teacher-forced NLL-gap、非 AR exact；`\rs{}` 是 repo fixed-index smoothstep，不得称 official YaRN）] | night03 S1 | 未推翻；落点建议限定为 "matched-scale complementarity" |
| C16 | 旧 350M/454M Phase11 记录（raw 8K 268.4→167.8；YaRN 260.2→99.6） | [假设→冻结使用]：模型身份冲突（report 说 454M、curated run IDs `350m_*`、当前 A4 同数值标 125M），payload 明示 "not numerically identical" | night02 F9 | 裁定：audit lead only，不纳入证据榜单（后由 09-10 文档继续追） |
| C17 | PSR/sparse-position 未建立 position-specific advantage | [已验证（负/未定，独立路线）] | night06 M11 | 与主线断开；`CORE_DIAGNOSIS` 称路线 open 而非 rejected——不得写成已否证，也不得入主线 |

## 4. 失败机制清单

本簇未"执行"失败实验（全程只读），但盘点出项目既有失败/受损机制与复发警告：

1. **代理指标→能力跳接（最高频复发模式）**
   - 实例：teacher-forced passkey 100% ≠ AR exact（454M 48K overlay PF 100% 而 AR exact 95%，且是恢复训练单 seed）；routing first-number 69/100 ≠ full-answer+EOS；natural-text NLL 低 ≠ RULER transfer（A8 0/39）；collision/r2 静态几何与 LM loss 反向（50M crossing：r2 改善、PPL 7.14→76.20）。
   - 复发警告：night03/04/06/08 四人独立重申——claim 必须与 metric 类型同层；bootstrap CI 是"条件于该训练对/该 rows 的评估不确定性"，不是训练随机性；文档 bootstrap ≠ training-seed CI。
2. **归因混杂被当纯 allocation**：8B 与 OLMo LoRA 结果同时换 substrate+grid endpoints+adapter 路径，不是 pure interior-shape；learned inv-freq 有 32 参数+LR sweep；historical Geo/EVQ 同改 support/span/shape。夜审警告：这些只能写 "matched end-to-end table/adaptation effect"（night08）；纯形状归因专属 151.9M/M4 固定端点家族。
3. **重复计数同一训练活动**：F4+F5 同 campaign、F6+F7 同 Phase11B、F2 多 arms 同 family、A5 两 endpoint 同 run、A4 QA/RULER 却是独立 adapters（反向案例：不同 seed 不能合并）。计数规则由 night02/04 显式给出。
4. **资产 owner/身份错配**：night06 把 151.9M crossing raw owner 误写为 K32_20260901 receipt（23:29 自纠）；F9 454M/350M/125M 标签冲突；Qwen 128K 旧分 `.6175` aliased 作废；K128 prior panel manifest 与新 confirmation 不同不可混比。→ 任何 claim 前先核对 receipt/manifest/SHA。
5. **headline 选择性呈现**：摘要只报 32K 改善省略 8K 退化（night08）；454M QuALITY 旧 n=200 pilot 夸大 accuracy（实际≈25% 随机，night03）；"improves Qwen's 64K RULER score" 缺 comparator（night01）。
6. **τ/超参选择口径不一**：同一族里混用公式约定、pre-specified empirical、sweep 选优（Video DiT 2.83→1.5 报最佳）；τ=0.01 init 落 softplus dead zone 使 learnable-τ 叙事受限（F5）；"τ=1.5 最优"被 M4 exact-range 后续 supersede（F8）。
7. **命名失真**："endpoint grid" 实为 left-endpoint k/K（night01 第4点）；`\rs{}` 非 official YaRN（night03）；DAPE 旧身份 mislabel 已退役（night01 复核表）；"variational"应降为"convex surrogate"（night01/08）。
8. **候选机制失败底账（对非几何探索直接可用）**：PSR 未建优势（M11）；BM 128K 3B/7B 反转 + cross-cache 显示成败随 prefix 形成表而非 read table、VT 6-token 后 premature EOS；band-bridge 负（M7）；dose-response guard 破（M10）；isotonic 端点权衡（Iso−p2 4K/8K 改善、16K 跨零）；C2 部署法 gate fail（OLMo PPL retention .87097<.875）；scale cap（S8 32K 6.94%、CappedS4 与 Freq8Gain4 无长程恢复）；DIRECT_Z 停跑；PHASE_ISOTROPY/PHASE_ALLOCATION SCREEN_UNRESOLVED；HEAD_SELECTIVE/FAR_PASS_CHORD 内部负。

## 5. 频率表/方法定义清单（本簇出现者）

| 名称 | 构造规则 | 关键得分（本簇记录） |
|---|---|---|
| standard/endpoint grid | `u_k=k/K`（a6 误称 "endpoint grid"，应写 "standard k/K grid"）；K=64 标准 grid 23 pairs/46 dims/`r2=2.00013` | Geo K=16/32/64 collision .2250/.2383/.2432, r2 7.31/7.63/7.84 |
| midpoint EVQ | `(k+1/2)/K` | — |
| Cosh 解析族 | Eq.(39) τ=√(β/α)，strength multipliers 0.75/1/1.25/1.5 预注册 | Geo 6.211973 vs Cosh rule 6.202094、1.25× 6.199873 (PPL 492.69)、matched-exp 6.201354 |
| FMRoPE | 选择 base、normalized exponents 仍均匀（与本文对象不同） | — |
| anchored EVQ-Cosh（151.9M） | 固定 sampled extrema/span，动 30 interiors | tail-NLL Δ +.026/−.281/−.176/−.146 |
| τ 实测值清单 | MLA 1.414（pre-specified empirical）、750M 1.5、Video 1.5（sweep）、τ\*=64/√L_train（=4 @L256、=2 @L1024…） | S8: τ4 32× overlay PPL 99.6 vs Geo+scaler 260.2 |
| YaRN 类固定索引 scaler `\rs{s}` | repo fixed-index smoothstep（≠official YaRN） | S1 四臂：EVQ+s8 PPL@8K/16K 70.9/107.5；S6 48K overlay 2.63 vs Geo 14.22 |
| BM（boundary-matched intermediate band） | 模型相对 displacement profile 的中间带构造 | OLMo 4K/16K 81.81/51.32；3B 32K 91.67、128K 70.83(−7.3 vs MrPro)；7B 32K 80、128K 71.11(−13.3 vs MrPro 84.44) |
| physical-x 表 | `c_orth=(1−b^{−1/K})^{−1}`，`x_i=log(L_ref·ω_i/(2π·c_orth))`，`G(x)=clip((x_H−x)/(x_H−x_L),0,1)`，`ω'_i=ω_i·s^{−G(x_i)}`；xH=.7382780681078285, xL=.366403835112904, c=.074 | K128 16K .728125 |
| normalized-index 表 | 同一冻结 G 在 counterfactual K64 归一网格点采样，按 `i/(K−1)` 插值输送 | K128 16K .790（+6.19pp CI[+2.81,+9.63]）；K32 Qwen0.5B 64K 51.4551 vs YaRN 45.3654（+6.09 CI[2.76,9.58]），32K 打平；K32 crossing：index .9232 过 retention、physical .8595 不过 |
| K32/K128 面板参数 | K32: b=1e6, Lref=32768, target 65536, s=2, g=1.0512928913614359；K128 Gemma: b=1e4, Lref=4096, target 16384, s=4, g=1.102585782722872 | — |
| DAPE/Kerple | 独立 learned positional-operator family | F7：Geo+DAPE→EVQ4+DAPE 55.9→56.8，频率对比消失（negative control） |
| learned inverse-frequency | 32 learned params + positional LR 100×/10× | F4：PPL@8K 455.3/477.7 vs Geo 513.7 vs EVQ 333.7 |
| learnable τ | single τ via train loss | F5：收敛 1.1406±0.0034，PPL@8K 437.9±12.3（≠extrapolation 优选）；口径补充：表中 EVQ 333.7 是 seed42 fixed τ=5 arm（3-seed mean 335.7103±1.7449） |
| NoPE/band-swap（50M probe） | 每 pair 独立置 NoPE；等宽带交换保持 exact multiset | M7：swap 改 NLL≥.05 于 22/24 cells；selected pruning Geo −.2959/−.2308/−.2802 |
| base-only control | 仅改 scalar base，保留 geometric order | GeoW: 7.14→(span377.7K)7.16→(endpoint331.6K)7.30→(LS-fit 8.06K)45.63 vs hist EVQ 76.20；EVQW: 23.05→…→9.63 vs 7.16 |
| isotonic profile | Native 权重单调化 | Iso−p2：4K −.145、8K −.175、16K +.0075(CI 跨零)；PG-19 2×/4× NLL −.027/−.035 |
| 432M MLA 约定 | d_rope=32, K=16, τ=1.414 empirical | 16K PPL 138.8→95.6（−31.1%），8K 代价 ~1% |

32K/128K 专门得分：32K — Qwen0.5B index vs YaRN −0.0256（无差异）、BM 3B +4.44pp/7B −3.33pp、OLMo native .82 vs C2 .7125（C2 差）；128K — BM 3B 70.83 vs MrPro 78.13（负）、7B 71.11 vs 84.44（负）、Qwen corrected derived .5400 vs Geo .4550。

## 6. 用户指令与纠正（原文引用）

真实人类用户在 8 个子会话中只出现 environment_context 自动消息；任务指令 payload 加密不可读。以下为本簇可获得的、约束/指导本簇行为的**原文**（AGENTS.md 注入全文见每个 raw 文件；转述性指令标"转述"）：

- AGENTS.md（每会话注入）："**Distinguish hypotheses and proxy improvements from demonstrated outcomes**… Successful methods are evidence, not immutable constraints."；"Check relevant prior results, failures, and subsequent corrections before repeating a direction. Do not repackage failed assumptions or generalize a specific failure beyond its evidence."
- night01 Ampere 自我固定边界（22:50 原文）："评阅边界已固定为'论证链与数学表达'，不做完整 venue 决策，也不把缺少全套大模型实验本身列为问题。"
- night03 Jason（23:15 原文）："区分已实施结果、代理指标和可写结论，**不把缺少 raw artifact 自动等同于不存在**。"
- night04 Kepler（23:15 原文）："重点会把'同一 run 的多端点'合并为单项证据，并区分 report、curated、raw。"（转述自开场白）
- night03 S1 建议（原文）："正文 claim 只写 'matched-scale complementarity'，**不要写成优于所有 YaRN/scaler**。"
- night03 S6 建议（原文）："不要写 'validated 48K capability'，只写 'single-seed supporting PPL pattern'。"
- night02 计数指令（原文）："F4/F5 属于同一 128-token campaign，**不能重复计数**；F6/F7 …不能当作两个独立实验；F9 …**应暂不纳入证据榜单**。"
- night06 对主线保护（原文）："sparse/PSR …**不要塞入主线主图**；最多在独立研究记录中引用并明确 'future/independent direction'。"
- night08 修正指令（原文摘要）："§5.1 将结论改为 'matched end-to-end table/adaptation effect'，并明确指出纯 interior-shape 归因来自 §3.1 的固定端点实验。"
- night01→现稿的三处被采纳性纠正候选（Ampere 给父线程）：00_abstract 首句 range 表述、"effective positional dimension" 必须写 Rényi-2 effective rank（实际 span 仍 46 维）、"variational"→"convex surrogate"。
- 注：night07 中途（23:49:30）父线程发过一条加密 MESSAGE（1,868 字符）改变了其表述（其 23:58 独白从"逐项裁定"转向"结构重排"方向），内容不可恢复。

## 7. 未决问题

1. **5/6 vs 3/6 评审分歧未在簇内仲裁**：Ampere（night01, Accept 5/6）与 Hilbert（night07, Borderline Reject 3/6）、Poincare（night08, 边界偏可接收）同夜对同一稿给出不同档位；Hilbert 主张"结构重排（Cosh 去中心）"，Ampere 主张措辞级修补。父线程如何取舍、最终 main.pdf（当前工作树 modified）采纳哪些，需从 09-09 会话簇确认。
2. **F9（350M/454M/125M 标签）身份 reconciliation**：night02 裁定 audit-lead；09-10 是否解决未知（当前 docs/research 未见终局文档）。
3. **S1 四臂表 + A3 EOS lineage + A7/A8 是否升入正文**：night03/04 给了明确 Top 排序与落点建议，本簇内无执行证据。
4. **τ 协议级表格（Poincare 主要问题1）与 8B headline 8K 代价并列（问题2）是否已改**：night08 之后由父线程落地情况不在本簇。
5. **"physical-x privilege"叙事与 K32 crossing 的收口**：night05 已给出更正与措辞上限，主文是否仍残留旧叙事待查 09-09 稿。
6. **静态几何→行为的机制桥仍缺**：M7（band bridge 负）、M6（gradient 非训练前 selector）、M10（dose 权衡）共同说明任何"由几何预测 allocation 好坏"的统一理论候选都必须自带 failure-mode 条款；这直接约束 09-09/09-10 非几何候选的评估设计（E1 计划文档中的"代理指标不具单独否决权/混合 RULER 为核心"正是对应用户纠正）。
7. **K128/K32 之外是否还有 third checkpoint** 可把 index-vs-physical 结论从"单点确认"升格为坐标层面普适性——簇内无人执行，属后续实验问题。
8. **子任务指令原文（加密 NEW_TASK）**未归档明文：若统一理论需要"当时父线程如何下达任务"的证据，只能依赖父线程 `01a0806f` 会话转录本身。
