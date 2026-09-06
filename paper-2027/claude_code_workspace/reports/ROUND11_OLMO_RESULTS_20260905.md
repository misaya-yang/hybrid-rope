# Round 11 跨家族验证报告：OLMo-2-0425-1B-Instruct（2026-09-05）

> **状态（2026-09-06 更新）**：已被 Round 12 全矩阵取代（1B+7B × {N,Z,Y,M,Y2} +
> chat-template 审计 + E1）。09-06 审计证实 **OLMo 无 raw 模式崩溃**，故本轮
> 数字仍有效，但范围限于 1B、ZC/ZF/ON 三臂。本地回执 JSON 已于 09-06 按用户
> 指令清理；服务器原件保留于 `/root/autodl-tmp/claude_round11_olmo_20260905/`
> （Round 12 E1 注记引用的 task128 对角线 full_exact_eos 14/64 即出自该处）。

预注册：`round11_20260905_olmo/PREGLUCTION_OLMO_Z1.md`（任何 GPU 运行前冻结）。
引擎 = `code_release_008` 逐字不改；回执镜像 `round11_20260905_olmo/receipts/`。
严格成功 = 完整答案精确匹配 + EOS + 两个反事实世界都对（组级）。

## 1. 动机与差异声明

用户指令：同一配方换模型家族验证（"直接1.4那个"）。
**与 Qwen 轮（§10）的关键差异（全部预先声明）**：
- 布局相同（2048/8192/16384），但语义不同：Qwen native 32K → 全部 within-native；
  **OLMo native 4096 → near 8192 = 2×、far 16384 = 4× 外推**。本回合的 far 是真正的
  超窗口外推测试，比 Qwen 轮难得多，是配方跨家族的更强检验。
- 候选文件内嵌 tokenize 结果，必须用 OLMo tokenizer 重建（同一 2wiki zip，sha 95df2bf5…）。
- 基线自建（native0/task0），与候选同引擎产生。
- 比较是配方级的，不做跨模型条目配对。

## 2. 资产与引擎验证（全部通过，零代码修改）

- OLMo 是 P2/log_s4 witness 的**本家模型**：Z 表 float32_sha256 = 56ddfae2… = P2_SHA，
  冻结引擎守卫零修改通过（Qwen 当时需要 profile_source 绕行）。
- 资产：检查点 36d044c7…（1.48B，max_position_embeddings 4096）；合同 olmo1485_contract.json；
  固定控制表 fixed_controls（FIXED_NZGY_CONTROLS_FROZEN_V1，Z gain 1.102585782722872）；
  native_pool_v3（a060c665…）；teacher_cache_olmo（896 条，池哈希匹配）。
- 训练器/运行时/审查器无任何阻挡 OLMo 的模型特定硬编码（逐一核对过长度校验与合同检查）。

## 3. 数据准备（D0-D2，全部通过）

| 阶段 | 结果 |
|---|---|
| D0 候选重建（OLMo tokenizer，CPU） | 1536 候选，split/family 计数与 Qwen 池完全一致（512/384/384 + 32/16/16 + 64×3），零拒绝；pool sha cb1180f3… |
| D1 资格筛选（GPU 风险闸门） | **128/286 通过（45%）**，45.9 秒；已知"OLMo EOS 坍缩"只发生在长输出，紧凑 2K 作答健康 |
| D2 视图固化 | QUALIFIED_NATURAL_TRANSPORT_V1，views sha f174314d…，全部 cell 计数正确 |

## 4. 主案结果（OLMO-ZC：Z 表 + LoRA compact-only，128 步）

配方与 §10 E3a 同构：arm Z，seed42，all-linear r16/α16（12.06M 可训），128 步（32R+96T），
KL .02，`--compact-only`（训练全程输入 ≤2048，在 OLMo native 窗口内），212 秒完成。
训练回执：TRAINING_COMPLETE_NOT_FEASIBILITY_OR_CAPABILITY，step128，reload 精确。

### 4.1 三级对照（主家族 single_evidence，far 16K = 4× 外推）

| 阶段 | far 严格(组) | far 严格(行/64) | 16K EOS 率* | far 宽松含答案 |
|---|---|---|---|---|
| T0（native 表，无适配器） | 0/32 | 0/64 | 0.0 | 0/64 |
| Z0（Z 表，零训练） | 1/32 | 4/64 | 0.617 | 26/64 |
| ZC（Z 表 + LoRA compact-only） | 1/32 | 3/64 | 0.703 | 27/64 |
| （参考）Qwen E3a 同配方 | 20/32 | — | ≈1 | — |

\* 16K EOS 率按该长度全部家族行聚合（128 行：single 64 + binding 32 + double 32）。

其他家族（组级严格）：
- near 16K single：T0 0 → Z0 3 → ZC 4（训练 +1）；EOS 率 0.594→0.758（+0.164）。
- far 16K single：0 → 1 → 1；EOS 率 0.617→0.703（+0.086）。
  **训练对终止的改善随外推距离衰减：near +0.164 > far +0.086。**
- compact 2K single 27→25（**丢 2，零损失门未过**）、double 2→1；binding compact 8→8 持平。
- binding/double 在基线上就弱（binding 8/16、double 2/16）——OLMo 指令基线低。
- compact 的 EOS 率三案均为 1.0：坍缩只发生在长输入（≥8K），与已知"长输出 EOS 坍缩"一致。

### 4.2 Native 保留（全部通过，零损伤）

| 层 | step0 → step128 | 保留 |
|---|---|---|
| ppl（text NLL） | 2.5102 → 2.4990 | **1.005** |
| instruction | 0.312 → 0.328 | 1.05 |
| reasoning | 0.031 → 0.047 | 1.5 |
| position_format | 0.203 → 0.203 | 1.00 |

### 4.3 冻结审查判定

`single_evidence_v4`：**POINT_FEASIBLE_CONFIRMATION_PENDING**（Native 点值全过 0.88 门，
配对 CI：ppl [0.934, 0.975]、task [0.774, 1.166]，task 下界 <0.88 → 未确认级）。
按预注册分级：**far <6/32 → 配方未按原样迁移**（负/部分结果，如实登记）。

## 5. 失败分解：瓶颈不是内容，是接口兑现

far 16K 单证据 64 行五分类（宽松"含答案"= 答案串出现在输出任意位置，仅诊断用）：

| 类别 | z0 | task128 | 含义 |
|---|---|---|---|
| 含答案+EOS+精确（=严格成功） | 4 | 3 | 完全兑现 |
| 含答案+EOS+格式不合 | 10 | **18** | 答案在完整句子里（"…is directed by Charles Band."） |
| 含答案+无 EOS | 12 | 6 | 内容对、不终止 |
| 无答案+EOS | 20 | 20 | 干净终止、内容错 |
| 无答案+无 EOS | 18 | 17 | 完全失败 |

**训练把"含答案但不终止"（12→6）转化成了"含答案、终止、但句子格式"（10→18）；
严格兑现（4→3）原地踏步。** LoRA 学会了终止，没学会在 16K 上把答案裁成裸答案。

## 6. 原因分析："模型能力不够"的精确分解

**不是内容能力不够，是接口兑现能力不够，且被底模指令基线弱放大。**

1. **内容层：表解锁跨家族成立（机理 A 的物理层）**。零训练换表就把 4× 外推的
   EOS 率 0%→62%、输出含正确答案 0/64→26/64（41%）。注意量级：模型从未在 16K
   上训练过，只换了位置频率表，就有近一半样本的输出里出现正确答案串——
   潜在长程检索能力在 OLMo 里同样存在，表只是把位置可判别性还给它。
   若内容能力本身缺失，换表不可能凭空解锁。这部分与 Qwen 完全一致。
2. **接口层：LoRA 学到了一半**。训练前后"含答案"行数几乎不变（26→27），
   变化全在终止：contain+noeos 12→6，EOS 率 near +0.164 / far +0.086。
   终止是全局序列性质，易学、跨长度部分迁移；但终止改善随外推距离衰减。
3. **格式层：没学到，还退了一步**。contain+eos+句子包裹 10→18（变多），
   严格兑现 4→3（原地）。训练目标把模型往"含答案的完整句"推，而不是
   "裸答案"——compact-only 训练全程没见过长上下文展开，16K 激活态在训练
   分布之外，梯度从未在那个状态上施加。模型"找得到"，但在 4× 外推上说不干净。
4. **为什么 Qwen 行、OLMo 不行**：Qwen2.5-Instruct 指令服从基线强，16K 在其
   32K 窗内，解锁出来的潜在能力本来就自带"简洁作答"接口，LoRA 只做轻微对齐。
   OLMo-2-0425-1B-Instruct 指令基线弱（同一紧凑评估：double 2/16、binding 8/16、
   reasoning 层 native 仅 3%），默认完整句作答；16K 又是 4× 外推、完全出训练分布。
   128 步 compact-only 不足以从零搭建这个接口。
5. **与 E3a 的 position_format 现象统一**：Qwen E3a 唯一退化的层就是
   position_format（9→5，Z 案 9→9）。格式/终止接口正是对"长输入 exposure"
   敏感的组分；OLMo 回合把同一脆弱性放大成主失败模式（底模更弱 + 4× 外推）。
   两轮合起来给出机理边界：**内容解锁跨家族迁移；格式/终止接口的兑现依赖
   exposure × 底模指令基线。**
6. **对机理归因的净影响**：机理 A（表解锁）跨家族加强；"LoRA 只做接口修补"
   这半句有前提——底模要有足够的指令服从可被修补。E3a"长输入 exposure 不必要"
   只在 within-native 评估下验证过；4× 外推下由 ZF 案检验（§7）。
7. **对论文跨家族主张的净影响**：不能主张"配方整体跨家族成立"；可主张
   "谱表的位置可判别性修复跨家族成立（零训练可见），完整兑现依赖底模指令
   能力 × 训练暴露的交互"。这是更精确、更有信息量的结论。

## 7. OLMO-ZF（长输入暴露案，完成）

预注册纠偏路径：ZC 失败形态 = 格式兑现瓶颈 → 跑 Qwen Z 案的同构（全布局暴露，
训练含 8192/16384 输入 = 对 OLMo 是 2×/4× 外推训练）。区分两种解释：
- 若 ZF far 显著 >1/32：长输入 exposure 在超窗口区必要（E3a 结论受限于 within-native）；
- 若 ZF ≈ ZC：瓶颈在底模能力上限，暴露无用。

训练 128 步完成（回执：step128、reload 精确）；全布局训练梯度统计与 compact 案
相当（中位 250 vs 273，无失稳）。

### 7.1 结果（single_evidence 严格，组级；链：T0→Z0→ZC→ZF）

| 长度 | T0 | Z0 | ZC(compact) | ZF(全暴露) |
|---|---|---|---|---|
| near（2× 外推） | 0 | 3 | 4 | **16/32** |
| far（4× 外推） | 0 | 1 | 1 | **4/32** |
| compact 2K | 27 | 26 | 25 | 25（−2 保持） |

EOS 率：far 0.0→0.617→0.703→**0.914**；near →0.969。
其他家族：binding far 0→2/16、double near 0→2/16、double far 0。
审查：**POINT_FEASIBLE_CONFIRMATION_PENDING**；配对 CI ppl [0.937,0.977]、task [0.806,1.188]；
Native 保留点值全过（与 ZC 同套门值）。

### 7.2 far 64 行失败分解链（z0 → ZC → ZF）

| 类别 | z0 | ZC | ZF |
|---|---|---|---|
| 含答案+EOS+精确（=严格行） | 4 | 3 | **14** |
| 含答案+EOS+句子包裹 | 10 | 18 | 14 |
| 含答案+无 EOS | 12 | 6 | 1 |
| 无答案+有 EOS | 20 | 20 | **32** |
| 无答案+无 EOS | 18 | 17 | 3 |
| 宽松含答案合计 | 26 | 27 | 29 |

### 7.3 判读

1. **终止问题被暴露解决**：far EOS 0.914、near 0.969；contain+noeos 12→6→1、
   nocontain+noeos 18→17→3。全布局训练把终止行为基本修好，包括 4× 外推。
2. **兑现在 2× 大幅改善、4× 部分改善**：near 4→16（4 倍）；far 严格组 1→4、
   严格行 3→14（4 倍）。回答 §7 开头的预注册问题：**长输入 exposure 在超窗口区
   必要但不充分**——E3a"exposure 不必要"的结论限于 within-native 评估。
3. **内容上限由表设定，暴露不新增内容**：宽松含答案 26→27→29，几乎不动。
   零训练换表已达 26/64；exposure 转化的是表已解锁的内容，不在 4× 外推上
   解锁新内容。这把机理 A 与"训练学到内容"（机理 B）进一步分开。
4. **残余瓶颈转移为检索正确性 + 双世界一致性**：nocontain+eos 32/64——
   干净终止但内容错（"自信的错误"）成为主导失败形态；且严格行 14 只落成
   4 个组级成功（多数组只有一个世界过线）。128 步暴露训练没解决 4× 上的
   检索精度。
5. **紧凑零损失门仍未过**（27→25，与 ZC 同样丢 2 组）：跨两个训练案复现，
   是 OLMo 底模/配方的真实小损伤，不是单次噪声。
6. **与能力上限的区分**：ZF far（4/32）明显 > ZC（1/32）但 ≪ 饱和，
   "纯能力上限"解释被削弱、"暴露+更多步数/更大底模可继续抬升"留有余地。
   表是否必要由 ON 案（§7b）检验。

### 7b. OLMO_ON（arm N 全暴露对照，完成）

预注册（`PREGLUCTION_OLMO_ON_ADDENDUM.md`，依 PREGLUCTION §6）：arm N、native 表、
全布局训练、无换表——与 ZF 只差表。

结果（训练 128 步回执过；审查判定 **STOP_ZERO_PRIMARY_GENERATION**——主终点 far=0，
按预注册关闭该候选）：

| 量 | ON | （对照 ZF） |
|---|---|---|
| far 严格 | **0/32** | 4/32 |
| near 严格 | **0/32** | 16/32 |
| far 宽松含答案 | **0/64**（64 行全部"无答案+有 EOS"） | 29/64 |
| far / near / compact EOS | 1.0 / 1.0 / 1.0 | 0.91 / 0.97 / 1.0 |
| compact single | 27→27（零损失） | 27→25 |
| 配对 CI | ppl [1.003,1.013]、task [0.880,1.067] | — |

数据层面的分离（解释见 §10 后记，按用户指令降级为假设）：同样全布局训练下，
不换表 → 终止完美（EOS 1.0）但内容 0/64；换表 → 内容出现。终止可训、不依赖表；
4× 内容可达性依赖表。

## 8. 执行事故（如实记录，均已保存现场）

1. **native32/128 漏传 `--table/--gain`**：我的 round 脚本 COMMON 漏了这两个参数
   （E3a 脚本的 COMMON 全程带着），导致这两步用 Native 表评估，与 task128 的 Z 表部署
   不一致；审查器正确拒判（UNRESOLVED_RECEIPTS_OR_CONTROLS: "different deployment
   functions"）。现场改名保留为 `native{32,128}_stagingerror_nativetable/`，
   用正确部署（Z 表+gain+adapter）重跑，第二次审查通过。不影响训练与任务评估。
2. **审查器版本**：code_release_008 自带审查器无 v4 协议；v4 审查器是 round-10
   协议产物（code_round10，sha e9cd1933ab80…，与 Qwen 三案判定所用逐字节相同），
   已原样复制进 code_round11 并重跑。审查器本身跨模型无硬编码。
3. **review_firstattempt_unresolved.json** 保留在 out/ 与回执镜像中。

## 9. 轮后纠偏（用户裁定，2026-09-06）

用户对本轮及 §10 Qwen 轮的总体裁定：**研究目标发生漂移**。主线一直是两个硬问题：
1. 冻结 mature checkpoint + 一张静态表（无训练）→ 能否同时做到 long information transport + Native retention；
2. 静态 retrofit 能否产生**真实自然自回归长生成**——只认严格精确生成，不认
   teacher-forced likelihood、宽松含答案、EOS/格式等代理指标。

LoRA 适配本是次级补救分支，被抬成了主线，围绕它做了整套因果归因。据此：

- 本报告的机理叙述（"谱预算 binding""表解锁潜在能力""LoRA 只修接口"
  "exposure 分离"）全部**降级为未经识别的假设**，不得继续指挥 GPU。
  其中"LoRA 只修接口"与数据直接矛盾：ZF 的严格行 3→14 说明训练改变了
  内容利用/选择，不只是接口。
- 保留的数据观察只有三条：
  1. 同一静态表零训练在 OLMo 4× 外推上产生 far-content 可观测量变化 0→26/64；
  2. Qwen 窗内 16K 的改善不要求训练时见过 16K；
  3. OLMo 4× 的真实瓶颈表现为"正常终止但答案错"（ZF 32/64），
     生成失败不能再归因 EOS。
- 原三项待办（新确认池、351 盲标、§9.3 teacher-prefix 诊断）**暂停**——
  它们是在给已经偏掉的故事加统计置信度。
- 项目归零到三个问题：静态表方法到底是什么？为什么目前解决不了
  "Native 保留 + 4× 自然生成"的联合目标？远端 signal 已出现而正确生成未出现，
  缺的到底是什么？——由 Pro 模型/Codex 先钉死这三个问题，再决定下一块 GPU。
- GPU 实验停止；容器按用户指令关机。

## 10. 回执索引（服务器 `claude_round11_olmo_20260905/`）

- `olmo_candidates/`、`olmo_qualification/`（QUALIFIED_128_GROUPS）、`olmo_tasks/`（视图+清单）
- `out/`：task0、z0、train（step_128 adapter）、native0、native32/128（重跑版）、
  task128、review.json、review_firstattempt_unresolved.json、
  native{32,128}_stagingerror_nativetable/（事故现场）
- `out_zf/`：ZF 案（执行中）
- 本地镜像：`round11_20260905_olmo/receipts/`（含 ZC 全部 JSON 回执）
