# OLMo BM：从已有RULER正结果迁移到自然QA

状态：COMPLETE。两批合计567条/臂全部完成，固定BM未调参。

依据：BM在OLMo静态S4的独立六任务16K面板有已核实收益。按作者“胜出后换模型/
任务”的规则，保留同一模型、BM数组、gain与S4，检验自然任务范围。Qwen3B
失败不抹除OLMo局部收益，也不使BM成为已解决的通用方法。

## 输入与比较

- OLMo-2-0425-1B-Instruct，实际1,484,916,736参数，原模型revision及完整元数据
  复用`prepared_bm_02`，不下载或更改权重。
- HotpotQA、2WikiMQA、Qasper；从既有来源SHA排序的未截断LongBench token池，
  各取前60条。原池分别171/199/197条，原官方源各200条。
- 已核对原始zip中的context与参考答案、现tokenizer解码与旧完整chat prompt SHA，
  原文和问题均完整保留。没有补答案、拼接背景、padding或截断原文。
- 输出预算分别32/32/128，输入+预算必须<=16384；输入真实长度单独报告。
  原token池按旧64-token reserve已筛过长例，不声称这是原官方200条的全量。
- 这三个自然QA面板未找到同表/输入的MrPro基线；现有E1_MrPro_long是构造
  double_evidence任务，不能冒充匹配自然基线。因此只补跑一次MrPro并归档，再
  跑固定BM；后续相同输入复用此次Mr结果。

主读数：输入>4096的自然QA token-F1，按任务等权；<=4096另报。保留完整生成
和EOS，不把F1当完整字符串正确率，不声称证据位于固定16K距离。本池曾用于
旧项目工作，不叫全新盲确认；BM未按此池答案选择或修改。

## 执行

远端`/root/autodl-tmp/olmo_fast_screen_20260908/`：
`code_natural_01`、`prepared_natural_01`、`run_natural_01`；worker PID4395。
共180条/臂，使用已验证stock HF generate、BF16、Flash SDPA；频率/增益固定。
运行器`layer_screen.py`走单表分支，无层hook、无CausalGain、无缓存干预。
实际预算和吞吐随回执记录。

在GPU运行时并行推进3B频率分配机制分析。若BM自然任务有益，继续其有价值的
范围检验；若无益，保留清晰任务边界，不通过修改BM或挑样本将它救成正例。

## 完整池结果

两批分别180和387条/臂，覆盖既有可用池的全部567条；并非官方各200条全部输入。
输入超过4096的458条，三个任务等权F1为BM26.01%、MrPro21.37%，+4.64pp。
Hotpot166条33.54%对28.63%；2Wiki173条25.43%对21.55%；Qasper119条19.06%对13.94%。
短输入109条另报：2Wiki与Qasper均值提高，Hotpot仅5条出现1负、-6.67pp。
这两组都比较静态S4部署，不能冒充相对原Native表的短端保留。
完整输出、token与EOS、逐行评分及回执通过复算，见
[完整结果JSON](ROPE_OLMO_BM_NATURAL_RESULT_20260908.json)。
