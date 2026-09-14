# Stable Accept计划审查与冻结执行版（2026-09-14）

## 结论

原提议的科学优先级成立：冻结TailSpline，先做同剂量C归因，再补真实任务、
强静态基线和Native保持；不把第三checkpoint当本轮门槛，不从roughness直接
推出任务最优。

执行上不原样照抄。当前冻结队列采用以下修正：

1. RULER正式主表使用13任务各200条纯`source-order`输入，不复用旧
   depth-balanced 10条，不做multi-evidence profile选择，不插入内容padding。
2. 推理batch=2允许attention-mask屏蔽的左侧pad token；pad不属于prompt，
   非pad position仍从0开始。这是批处理实现，不是数据干预。
3. Natural-QA以`source_context_sha256`作为文档簇；bootstrap抽文档簇后按
   抽中问题数加权，使区间与问题等权点估计一致；另外报告真实Llama
   `input_tokens>8192`的315条子层效果。
4. HELMET-RAG保留为独立应用确认，但不让尚未取得的数据资产阻塞当前GPU。
   官方仓库说明完整数据包约34GB；当前数据盘只余约22GB。必须先做选择性
   NQ/Hotpot资产取得、ID排除和manifest冻结，不能直接下载全包或用自造RAG
   冒充HELMET。

## 当前冻结队列

服务器：`ssh -p 37849 root@connect.westc.seetacloud.com`

主链：

`/root/autodl-tmp/today_rope_plan_20260914/tailspline_llama_final_benchmark_chain.pid`

执行顺序：

1. 完成已开始的padding单任务T/P诊断；该结果不进入正式表；
2. E1：Llama classic 390 generation + 138 LM中的单臂同剂量C；T raw复用；
3. 32K clean RULER-13×200：TailSpline 2600，再MrPro 2600；
4. Natural-QA631 V2：TailSpline 631，再MrPro 631；
5. classic 390上只补YaRN与BM各390 generation + 138 LM；T/P raw复用；
6. 写四份report的SHA和`stable_accept_ready_queue_complete.txt`。

当前队列不包含新曲线、训练、Qwen/OLMo重复Full-13、HELMET未冻结数据或
短能力面板。服务器无Codex定时任务，也不自动关机。

## 已冻结资产

### E1同剂量C

- 表：`tailspline_llama_s4_matched_dose_c/tables/llama_s4_tailspline_dose_control.json`
- formula：`C=(1-w)U+w Front`，`Front=2U-MrPro`，
  `w=3n/[2(2n+1)]`；
- band `[18,35]`，S4，gain `1.138629436111989`；
- C table SHA256 float32：
  `cc2d04faba2eb92a50221579e8d5fd1bb1e7cde501ae8a475fb0fd7df6590d5a`；
- TailSpline/C部署FP32分析坐标的`sum_m`分别为
  `38.942857036846945/38.94285704449242`；解析式精确同剂量，差异来自FP32
  部署与反算，不为bitwise追平而修改表。

E1只检验shape-beyond-total-log-displacement；它不能独立证明tail jump是
唯一中介。主终点为同一classic面板上的Full-13 log-length AUC T-C；PPL为
次终点，NIAH为嵌套分解。

### Clean RULER-200

- manifest：`tailspline_llama_s4_32k_ruler200_clean/assets/manifest.json`；
- rows：2600，13任务各200；
- 输入SHA：
  `60fb1fbce690dffc367d487f531b0effc4e9816035ea0f8888d5d3a0420cdda4`；
- 实际输入长度28270--32606；
- `depth_balancing=false`、`multi_evidence_profile_selection=false`、
  `content_padding=false`。

### Natural-QA631 V2

- manifest：`tailspline_llama_s4_naturalqa631/assets/manifest.json`；
- 输入SHA：
  `7c4524fc89b6910fde8239ebbc1cf440fff61ed4523569052a7e67f933011767`；
- 631行、524个source-context文档簇；
- Llama原生8K内316行、超过8K 315行；超过8K的五任务计数为
  Hotpot142、2Wiki52、Qasper11、Narrative53、MultiField57；
- 这是历史冻结源池的Llama重分词，不是独立新问题，也不是完整LongBench。

### classic强基线

- YaRN table SHA256 float32：
  `ad4c0e740a42022551b834478a1208c81f8e5e41d11facc64c5692bf41eb908b`；
- BM table SHA256 float32：
  `c62e6514f39c2a6de9f6a8901cb1a773c96c5764cf3899481334e3dde6f2faa0`；
- 两者只在已有390 prompt/138 LM格上补跑；T/P不重跑。

## HELMET-RAG处置

官方HELMET代码已只读克隆到服务器：

`/root/autodl-tmp/benchmarks/HELMET`

commit：`af609c4d51b97fc35012099380aa889da961c42d`。

官方`configs/rag.yaml`默认含NQ/TriviaQA/HotpotQA/PopQA，数据下载脚本只提供
完整`data.tar.gz`。原提议的NQ128+Hotpot128、6/3排列、8/16/32K四扩展臂加
Native8K需要独立的选择性数据资产与排除清单；当前不能从已克隆代码推断这些
资产已经存在。故本轮先完成E1、clean RULER、Natural-QA与classic强基线；
HELMET只有在磁盘与选择性数据取得问题解决、且分数揭盲前manifest冻结后才能
进入GPU主链。

## 理论力度

采用原计划的三个边界：

- 超额roughness可控制相对TailSpline的位置表示偏离，不能证明更接近Native；
- 同剂量T/C消除的是统一槽响应，保留坐标化的有符号响应差异；
- attention有限变化公式解释证据与干扰的相对logit变化，不直接推出任务收益。

因此论文可写构造保证、受控量分解和条件任务响应；在没有实际中介结果时，
不写“定理保证TailSpline性能最优”。

## 完成判据

当前ready队列完成要求四份配对报告及raw均完整：

- `tailspline_llama_s4_matched_dose_c/reports/tailspline_vs_dose_control_c_classic.json`；
- `tailspline_llama_s4_32k_ruler200_clean/reports/tailspline_vs_mrpro_full13_32k_200_per_task_clean.json`；
- `tailspline_llama_s4_naturalqa631/reports/tailspline_vs_mrpro_naturalqa631.json`；
- `tailspline_llama_s4_classic_strong_baselines/reports/tailspline_vs_mrpro_yarn_bm_classic.json`。

不能用进程退出、单臂partial或结果文件名代替状态、行数、prompt集合、table
receipt和raw hash验证。HELMET、额外32文档PPL与短能力面板仍是后续确认工作，
不冒充当前ready队列已覆盖。
