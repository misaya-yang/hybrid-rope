# Native增强：实验与结果入口

## 2026-09-17正式结果

冻结`OLMo-2-0425-1B-Instruct`后，公开参数构造的NCP相对geometric Native得到两个正式正结果：

- 同目标Native-4K语言建模：103个源文档、128个窗口上，full-context NLL从`2.915269`
  降至`2.902483`，差`−0.012786` nat/token，约等于PPL降低`1.27%`；NCP从额外早期上下文
  获得的NLL收益比Native多`0.007941`。
- 新Full-13×10：130条/臂，NCP比Native高`+3.2564pp`。正式点分按固定13任务等权；
  配对bootstrap区间`[+0.8077,+5.9231]pp`作为稳定性分析。

同一轮Native-window Natural-QA有99条/臂。当前论文主指标为任务内问题等权、再对三任务等权：Native/NCP为`39.4116/38.4554%`，差`−0.9562pp`，区间`[−4.4410,+1.5666]pp`。原报告的source-equal敏感性为`−0.8549pp`，区间`[−4.4495,+1.9234]pp`；两种聚合不能互换。

[Full-13与原QA报告](reports/server_20260917/native_quick_gate_x10_and_qa.json)和[同目标NLL](reports/server_20260917/native_lm128_parallel.json)保留原身份；问题等权复算使用[便携配对分数](../../paper-2027/figs/revision_evidence_inputs.json)，口径见[论文附录E](../../paper-2027/appendix/compact_e_native.tex)。这些是不同终点，分别解释。

机制拆分给出更具体的边界，而不反向抹掉正式性能结果：

- 既有780行开发面板重聚合：H/NCP/V1相对Native分别`+0.1346/+1.4081/+3.0107pp`，
  reverse为`−1.4765pp`；见[五臂重分析](reports/server_20260917/existing_five_arm_reanalysis.json)。
- 新288行反事实面板中，NCP在1K为`+2.083pp`、2K持平、4K为`−11.458pp`；
  这说明该合成binding/chain面板不是NCP正式收益的中介解释；见
  [NCP机制报告](reports/server_20260917/mechanism_ncp_vs_native.json)。
- 固定末四层相位块干预使Native从`9.375%`降到`6.25%`，反向干预净效应为0；
  因而这一个固定层块不解释NCP收益；见
  [末四层干预](reports/server_20260917/final_quarter_intervention.json)。

结论按结果优先表述：**NCP在冻结OLMo的Native-4K NLL/PPL和新Full-13面板上增强原生能力；
Natural-QA与两组机制面板完整保留其各自分数，用于任务分解和机制排除。**

## 2026-09-16准备记录

既有NCP在OLMo原生Full-13上的总体收益已完成并入稿；本目录随后新增的288题、独立确认与
同目标NLL已于上节登记。执行优先级由
[统一准备清单](../../docs/research/next_stage_20260912/PAPER_NEXT_REVISION_PREPARATION_20260916.md)安排；
参考风险符号不自动升级为模型准确率预测。

## 2026-09-16 Native相位研究代码

新方案不再继续搜索Z5，而是分开检验行为、局部相位响应和独立确认。以下记录当时的执行合同；
对应GPU结果现已在本页首节完成登记：

- `prepare_server_cpu.sh`准备既有五臂输出重分析、更新后的288题四臂面板、96题Q/K/V捕获清单、
  Full-13×10新世界快速门、Natural-QA三任务完整4K普查（每任务最多80）及128个同目标NLL窗口。
- `run_server_gpu.sh`默认只打印计划；显式`--execute`才运行Native/H/NCP/V1反事实、固定四层Q/K/V、
  末四层双向相位干预及独立Full-13/Natural-QA/NLL确认。
- Q/K/V只用于解释冻结表，不用于拟合新表。固定干预块为OLMo零起始层12–15；64题在模型输出前冻结。

当时的无卡准备、`cpu_ready.json`与显式`--execute`边界均已按合同使用；该次结果回传记录服务器已关机，
本地只保留紧凑正式报告，完整raw仍在服务器数据盘。


本包只新增CPU准备与分析路径，复用现有NCP构造和冻结模型评估器。

最新优先级见[native定义、旧方法谱系与LeRoPE审查](../../docs/research/reviews/NATIVE_BENEFIT_AND_METHOD_LINEAGE_REVIEW_20260915.md)。
NCP和reference_predictions保留为候选/参考模型资产，不再默认决定下一项方法或accuracy预测。
当前研究规格见[准备计划](../../docs/research/next_stage_20260912/NATIVE_ORAL_PREPARATION_PLAN_20260915.md)。

## 已完成的CPU回执

- [服务器验证与执行边界](reports/cpu_validation.json)、[288条资产清单](reports/mechanism_manifest.json)。
- [三模型runtime公式构表](reports/runtime_matrix.json)、[独立积分oracle](reports/ncp_cpu_oracle.json)。
- [完整相位理论复算](reports/theory_audit.json)、[输出前距离预测](reports/reference_predictions.json)。
- [同目标NLL索引canary](reports/lm_alignment_canary.json)、[现有2600×2生成吞吐剖析](reports/throughput_cpu.json)。
- [未来生成计划](reports/generation_plan.json)、[等log剂量表](reports/ncp_dose_control.json)、[相位反射表](reports/ncp_phase_reflection.json)。

上述JSON属于CPU准备/既有输出分析；已完成的GPU结果由页首正式报告单独维护。完整prompt资产保留在服务器；
可用prepare.py和原有tokenizer按冻结seed重建。

## 代码

- [既有报告与27个方法的raw复算](evidence_review.py)：配对真实prompt身份，保留不完整开发面板。
- [旧原生正结果的缺失gain对照准备](prepare_native_factorial.py)：Qwen1.5B两新臂，各108条；不构造新频率曲线。
- [两格短追加与四格配对报告](run_native_factorial.py)：默认只输出计划；显式`--execute`才执行216次新生成，复用旧两格并共同bootstrap，分别报告36/72条历史块。它不接入或改变当前GPU队列；历史未记录的runtime版本在报告中保留为缺失，不能据此宣称已完成严格同环境资格。
- [本轮实际证据回执](reports/existing_evidence_review.json)。

| 需要做的事 | 入口 |
|---|---|
| 检查NCP积分目标、构造等log剂量控制 | [ncp.py](ncp.py) |
| 从公开Native表适配不同模型几何 | [multimodel.py](multimodel.py) |
| 逐槽等相位位移的反向控制 | [phase_control.py](phase_control.py) |
| 准备288条查询/重连反事实 | [prepare.py](prepare.py) |
| 冻结96条Q/K/V与64条层块干预输入 | [prepare_capture.py](prepare_capture.py) |
| 在任何输出前冻结真实距离的参考风险预测 | [predictions.py](predictions.py) |
| 查看未来生成命令（默认不加载模型） | [run.py](run.py) |
| 已完成生成的完整输出配对分析 | [report.py](report.py) |
| 既有Native/H/B/V1/NCP五臂CPU重分析 | [reanalyze_existing.py](reanalyze_existing.py) |
| 固定层块相位覆盖与报告 | [layer_phase_override.py](layer_phase_override.py) · [report_intervention.py](report_intervention.py) |
| 新Natural-QA与同目标NLL准备/执行 | [prepare_native_naturalqa.py](prepare_native_naturalqa.py) · [prepare_native_lm.py](prepare_native_lm.py) · [run_native_lm.py](run_native_lm.py) |
| 新Full-13/Natural-QA确认报告 | [report_confirmation.py](report_confirmation.py) |
| 相位与内容margin界CPU复算 | [theory.py](theory.py) |
| 同目标full/recent NLL合同 | [lm_context.py](lm_context.py) |
| 真实评测成本剖析、CPU batch计划 | [throughput.py](throughput.py) |

## 服务器命令

所有命令从服务器repo `/root/autodl-tmp/hybrid-rope` 执行。Python为`/root/miniconda3/bin/python`。
准备过程采用`CUDA_VISIBLE_DEVICES=''`、`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、
`MKL_NUM_THREADS=1`、`TOKENIZERS_PARALLELISM=false`，并使用`nice -n 19 taskset -c 127`。
CPU编号是本次服务器实际可用编号；换机先查询可用affinity，不照抄。

```bash
python -m experiments.native_enhancement_oral_20260915.run
python -m experiments.native_enhancement_oral_20260915.report --help
python -m experiments.native_enhancement_oral_20260915.multimodel --help
python -m experiments.native_enhancement_oral_20260915.throughput --help
```

`run`默认输出PLAN_ONLY。`--execute`是未来明确授权的GPU入口，持有共享GPU锁，发现已有GPU进程就拒绝，
不支持共享运行、不停止其他任务。本轮没有调用它。

输入资产位于`/root/autodl-tmp/today_rope_plan_20260914/native_enhancement_oral_cpu/assets/mechanism`，
表在同根`tables/`，CPU回执在`reports/`。已有780条NCP开发任务仍由
`experiments/native_contrastive_proximal_20260915/`负责；这两个实验不能互相冒用baseline。

配对报告示例（第二臂减第一臂）：

```bash
python -m experiments.native_enhancement_oral_20260915.report \
  --panel /root/autodl-tmp/today_rope_plan_20260914/native_enhancement_oral_cpu/assets/mechanism/inputs.jsonl \
  --run native=/root/autodl-tmp/today_rope_plan_20260914/native_enhancement_oral_cpu/runs/mechanism/native \
  --run ncp=/root/autodl-tmp/today_rope_plan_20260914/native_enhancement_oral_cpu/runs/mechanism/ncp \
  --out /root/autodl-tmp/today_rope_plan_20260914/native_enhancement_oral_cpu/reports/ncp_vs_native_mechanism.json
```

## 验证

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_native_enhancement_*.py
```

测试包括独立积分oracle、相位余项界、反事实答案、评分失败情形、同目标索引和默认不执行GPU。
随机张量/合成分数只存在于明确标记的CPU测试；不产生模型准确率。
