# ICLR 2027 实验计划

创建:2026-08-06 · 状态:`PLANNING_ONLY`(每一项 GPU 实验都需用户逐项授权后才能启动)

格式遵循 AGENTS §1.5 triage:每项写明 ①对应审稿关切 ②已有证据 ③最小缺失证据 ④最小可执行方案 ⑤停止条件,并给出优先级分类(`required` / `optional` / `unnecessary`)。

审稿关切 ID 来自 `rebuttal/rebuttal_0723/00_REVIEWER_SCORES_AND_AC_METAREVIEW.md`;
机制事实来自 `research_notes/FABLE5_EVQ_MECHANISM_AUDIT.md`(下称"审计")。

---

## 0. 总原则

- **Venue 无关/双用途**:所有实验同时服务 NeurIPS camera-ready 与当前
  ICLR 2027 独立续作;09-24 决定日不改变实验优先级与协议。
- **时间盒**:W4 末(09/02)冻结主结果表 v1(纪律性冻结,camera-ready 与转投都需要);P2 未完项可在 9 月继续(已无会前硬截止)。写作不在本文件范围,启动条件见 02 §0。
- **优先顺序**:paper-blocking(P0-1、P0-2、P1-1)> 审稿人显式要价(P1-3、P2-2、P2-3)> 增强(其余)。
- **不做的事**:不重跑已完成的 matched arm(AGENTS §2.6);不为叙事对称加实验;不启动 32K/4× 新攻坚(已有多重负结果,论文以边界形式呈现);不做 EVQ vs LeRoPE 对照训练(无共同基准,时间不允许;见 `LEROPE_CONCURRENT_WORK_NOTE_20260728.md` §5)。

---

## P0 — 零成本 / 纯分析(W1,不需要 GPU,不需要授权)

### P0-1 Readout-trace 逐层因果分解 【required,paper-blocking】

- **关切**:机制链 link 4(跨层传播)是唯一"有数据、无分析"的环节(审计 §2 link4);同时支撑 AC.2"diagnostic-heavy → 机制证据要成体系"。
- **已有**:`results/readout_conversion_s42_20260715/raw/causal_{evq_cosh,native_geo}/records/*.pt`(5 对 16K 配对样本,dense vs gold_drop_all,bf16 `[3,32,128256]`);`results/lora_sparse_conversion_s42_20260714/phase0_{evq,geo}.json`(逐层逐头 QK 探针,8K/16K/32K)。
- **缺失**:逐层 δ̃_l 曲线、entry layer、retention ratio、final-layer 竞争分解、QK↔residual 对齐。
- **方案**:完全按审计 §4 执行(指标 M1–M5、判定规则 H-late-entry/H-decay/H-competition、Geo 阴性对照)。新脚本 `scripts/analysis/readout_trace_decomposition.py`,输出 `results/readout_decomposition_s42_<date>/` + `docs/exp/` 报告。CPU-only,需带 torch 的环境(系统 python3 无 torch,用 `requirements.txt` 环境)。
- **停止**:Geo 对照 δ̃≉0 → 停,先审 trace 实现;分析一次成型,不做事后指标搜索。
- **产出角色**:论文机制章的"链条最后一块"图;5 case/单 seed,标注 oracle-diagnostic 级。

### P0-2 λ* 对准后验检验(LeRoPE 实验 A) 【required,paper-blocking(决定 P1-2 形态)】

- **关切**:R27bE.1(推导给出标度形式给不出常数)、AC.3。
- **已有**:21 个配置的经验 τ 选择数据在库(`scripts/analysis/unification_plot*.py`、`verify_softmax_transport.py` 的输入);LeRoPE dominant band λ≈2.205·L_train(leave-one-out 0.762 nats,11×于次名)。
- **缺失**:H1 检验 —— 经验选中的 τ 倍数是否与「使某通道最接近 2.205·L_train」秩相关。
- **方案**:按 `LEROPE_CONCURRENT_WORK_NOTE_20260728.md` §7.2,纯 post-hoc:对每配置在 τ∈{0.75,1.0,1.25,1.5}× 算 `min_k |log(λ_k/λ*)|`,与实际选中倍数做 Spearman。半天工作量。
- **停止/分支**:成立 → P1-2 优先做 pinned-λ* 变体;不成立 → 干净排除该解释,P1-2 只做保护带 two-regime 变体,论文中如实报告阴性。

### P0-3 QK 探针 layer/head 结构重分析 + 32K 衰减图 【required】

- **关切**:AC.2、RDz6s.1(机制证据成体系);审计混淆 C8(不用全头平均)。
- **已有**:`phase0_{evq,geo}.json` 逐层逐头原始数据(3 长度 × 10 case × 2 arm),从未按层/头结构分析发表。
- **方案**:逐层 EVQ−Geo gold-block hit@16 / mass 优势热图;32 retrieval heads(层 11–31)与全体头对比;8K→16K→32K 衰减曲线(把 32K 失败画进主图,而非藏在文字里)。与 P0-1 同一脚本族。
- **停止**:纯描述性,一次成型。

### P0-4 证据整合表(论文素材) 【required】

- **关切**:AC.4(什么能改变决定)、send-gate。
- **方案**:把 submitted(Primary I–III、750M)与 post-sub(OLMo routing、8B causal、FMRoPE exact-range、负结果 G-*)全部 matched 结果按「设置/seed 数/metric 类型/tier/owner 路径」列成单表,供 02 号文档逐章引用。只整理,不升级任何 tier。

---

## P1 — 本地 RTX 5090 / 低成本(W2–W3,**每项需授权**)

### P1-1 P-EVQ:Native-importance-protected EVQ 微调 【required,paper-blocking;本计划核心】

- **关切**:用户核心问题(保 in-window + 增外推);AC.2/RzWsa.4(1B 成熟模型);03 号文档预测 P1/P2 的直接检验。
- **已有**:设计+离线实现已注册(`OLMO2_NATIVE_IMPORTANCE_PROTECTED_EVQ_HYPOTHESIS_20260728.md`;代码 `rebuttal/rebuttal_0723/experiments/olmo2_lora_maturity/native_protected_evq.py` 等 7 个文件 + 测试);背景事实:full-EVQ 换表 4K RULER 82.16→37.51、selective Q/K 已把 2Wiki 拉平(22.0%/21.5%)但 RULER 4K 仍 72.19% vs 42.44%(`OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md`)。
- **缺失**:Stage D 诊断(H-PROTECT 是否成立)→ E0 免训练筛 → 一次 144-step 修复训练 + 能力 gates;protected 通过后再跑 matched full-EVQ control。
- **方案**:严格按已注册文档 §4–§9 执行,不改任何阈值/超参。执行顺序:无 GPU preflight → 诊断(推理 only)→ E0 → (未过再)protected 训练 → gates → (过了才)control 臂。需要:OLMo-2-0425-1B-Instruct checkpoint + LongAlign 4K view(远端资产;5090 有该 shape 的吞吐 profile)。预算量级:诊断+E0 为小时级推理;训练 144 步 micro-batch 1×4,单次。
- **停止**:注册文档 §7 的五种结论各有出路,全部可写(诊断失败=放弃保护假设也是论文可用的负结果);任何 hash/gate 失败即停。
- **产出角色**:论文新方法章 + "in-window 保持 + 外推收益"主表。**这是把 03 号文档从假设变成证据的唯一路径。**

### P1-2 Scratch 无损分配变体(pinned-λ* / two-regime)3-seed 【required(形态依 P0-2)】

- **关切**:用户核心问题的 scratch 侧(+0.0381 内禀代价可否消除);R27bE.4(matched-τ 非 cosh 替代 schedule)。
- **已有**:`rebuttal/rebuttal_0723/experiments/evq_spectral_frame_50m/run_spectral_frame_50m.py`(50.9M from-scratch,CPU-ready,`CPU_READY_GPU_RUNTIME_UNVERIFIED_NOT_TRAINED`)支持替代频率构造;151.9M exact-range 基建(EXPERIMENT_REPORT §12)可复用。
- **缺失**:{geometric, EVQ-Cosh(τ*), pinned-λ*+cosh 尾, protected-band two-regime} 四臂 × 3 seeds 的 train-length NLL + 2×/4× 外推 NLL。
- **方案**:优先 50M(单 run 小时级,5090);构造函数新增文件(不改 `scripts/lib/rope/schedules.py`,新模块 + 独立 hash 验证,吸取 `I-HYBRID-ALIAS` 教训:所有频率张量独立 clone + 与期望 hash 断言)。判定:pinned/two-regime 是否消除 ≥ 2/3 的 in-window 差距且保住 ≥ 80% 外推收益(03 号文档 P3)。
- **停止**:任一臂 loss 非有限、realized hash 不符即停;50M 方向不稳定不升 151.9M。

### P1-3 τ basin 右边界补全 【required(审稿人显式要价)】

- **关切**:R27bE.4(独立调 τ)、G-TAU-FALLIBLE、weekend sweep 自记的 grid-edge 缺陷(`results/weekend_sweep/analysis/summary.md`:r≤1.7 单调未见谷底)。
- **方案**:即 summary.md 自己推荐的后续:L=1024、d_head=64,τ 扩到 r∈{2.0,2.4,3.0}(+已有 7 点),TinyStories + FineWeb-Edu 两数据集,3 seeds ≈ 18 新 run × ~50min ≈ **15h GPU**。产出:完整 basin 图(含谷底或明确的继续单调),τ 规则以"basin selector + 实测 basin"诚实呈现。
- **停止**:两数据集方向矛盾即如实双报,不挑数据集。

### P1-4 Aliasing vs OOD-phase 分离度量(LeRoPE 实验 C) 【required(下一版核心机制图)】

- **关切**:AC.3 / RDz6s.3(理论-实践链);03 号文档 T2 的经验支柱。
- **已有**:seed-42 的 151.9M Paper-Geo / EVQ / FMRoPE checkpoints + 逐对 NoPE 消融基建(EXPERIMENT_REPORT §11,checkpoint-only);频带删除脚本 `rebuttal/rebuttal_0723/experiments/frequency_band_usage_5090.py`。
- **缺失**:按 λ 相对评估距离 d 分组的两组消融曲线:组1(λ≪d,制造 aliasing)vs 组2(λ≳L_train,制造 OOD phase),各自对长程 NLL 的边际贡献随 d 的走向。
- **方案**:checkpoint-only 推理,无训练;沿用 §11 的 16/32 anchor 划分与 bootstrap 区间;3 checkpoints × 4 长度。小时级。
- **停止**:两曲线不分离 → 二分框架降级为叙述性动机,论文理论章相应收缩(这本身是干净结论)。

### P1-5 OLMo matched adapters 的 QK 探针 + Q/K norm/entropy 控制 【optional(强烈建议)】

- **关切**:审计混淆 C6(norm/temperature 未控);把 link-2 证据从"8B adapted"扩到"能力转换已被证明的 OLMo 设置"。
- **已有**:冻结 adapters(hash 在 `OLMO2_1B_4K_ONLY_ROUTING_CONVERSION_20260726.md` §Frozen evidence);phase0 探针代码可移植(`eval_sparse_conversion.py::_probe_one`)。
- **方案**:推理 only:Native/EVQ 两臂在 20 条 8K NIAH 行上的逐层逐头 gold-block 指标 + 每臂 Q/K norm 分布 + attention entropy(N_eff)。若 EVQ 的 QK 优势在 norm 匹配后仍在 → C6 关闭。
- **停止**:适配 OLMo 注意力模块的 hook 超过 1 天工作量即降级为 W4 备选。

---

## P2 — 视预算与进度(W3–W4,**每项需授权**)

### P2-1 8B causal suite 扩容(n≥100 cases,第 2 个 LoRA seed) 【optional】

- **关切**:审计 C8(10 cases/单 seed);AC.2。
- **方案**:复用 `eval_sparse_conversion.py` 全套(phase0 → gold-drop → oracle → rank),100 case、新 LoRA seed 一次训练(300 步,~40min 级)+ 推理。远端 GPU,预算中等。优先级低于 P1-1:机制章单 seed 标注可接受,能力章已有 OLMo 两 seed。
- **停止**:方向与 seed-42 矛盾 → 如实双报并调低机制章语气(这正是要买的信息)。

### P2-2 Held-out base/head_dim 小 factorial 【optional(27bE 显式要价)】

- **关切**:R27bE.2/R27bE.5(单一 base=500K、单一 lineage)。
- **方案**:151.9M × {b=10K, 1M} × {d_head=64,128} × {geo, EVQ(τ*)} × 3 seeds 的最小格(8 对照单元),train-length + 2× NLL。约 24–48h GPU。若预算紧,砍到 b=10K 一列(4 单元)。
- **停止**:任一单元 EVQ 反向 → 保留并写进 boundary(不是砍掉该单元)。

### P2-3 FMRoPE 对照打包升级 【required(写作侧),实验 optional】

- **关切**:AC.1 / RzWsa.1 / RzWsa.2(novelty 硬门槛)。
- **已有**:exact-range 3-seed(Cosh−FMRoPE −0.316/−0.195/−0.167 @512/1K/2K,`AUTHOR_CONFIRMED` 但 `local_per_seed_values_present: false`)+ G-FMR-DEPLOY 负向(target-aware range transport 更强)+ §11 interference 对比。
- **方案**:优先做 provenance 修复:找回/重出 per-seed raw 使其升 tier(远端 hash 或一次 3-seed 复算);写作上以"allocation shape(我们)与 range control(FMRoPE)是两个正交旋钮,各有胜负"呈现双向结果。**不需要新的科学实验,需要的是 owner 升级。**
- **停止**:per-seed raw 无法恢复且复算超预算 → 论文按 AUTHOR_CONFIRMED 语气降级引用。

### P2-4 EVQ 闭式表 vs Fixed-LeRoPE 学出表 【unnecessary(本轮不做)】

- LeRoPE note §6.4 认可其价值,但无共同基准、7 周内建立公平协议风险高;写入 future work。

---

## 5. 排期与预算汇总

| 项 | 分类 | 周 | GPU | 预算量级 | Paper-blocking |
| --- | --- | --- | --- | --- | --- |
| P0-1 readout 分解 | required | W1 | 无 | 0 | ✅ |
| P0-2 λ* 对准 | required | W1 | 无 | 0 | ✅(决定 P1-2) |
| P0-3 QK 结构图 | required | W1 | 无 | 0 | — |
| P0-4 证据整合表 | required | W1 | 无 | 0 | — |
| P1-1 P-EVQ | required | W2 | 5090/远端 | 诊断小时级+一次 144 步 | ✅ |
| P1-2 无损变体 3-seed | required | W3 | 5090 | ~12 run × 小时级 | — |
| P1-3 basin 补全 | required | W2–W3 | 5090 | ~15h | — |
| P1-4 aliasing/OOD | required | W3 | 5090 | 小时级(推理) | — |
| P1-5 OLMo QK 探针 | optional | W3 | 远端 | 小时级(推理) | — |
| P2-1 8B 扩容 | optional | W3–W4 | 远端 | 中 | — |
| P2-2 base/head_dim | optional | W4 | 5090/远端 | 24–48h | — |
| P2-3 FMRoPE owner 升级 | required(写作) | W2–W4 | 视情况 | 小 | — |

表中 Paper-blocking 指"对下一版叙事关键(不论 venue)";原 ICLR 周期压力已解除(00 §1),P2 项在 W4 后仍可继续。

授权流程提醒:每项启动前按 AGENTS §2.2 出 READY/receipt(5090 有限豁免仅适用于资产已备齐的 bounded diagnostic);launch 后按 §2.5 记录 receipt;所有新频率张量独立 clone + hash 断言(`I-HYBRID-ALIAS` 教训)。
