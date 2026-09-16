# Native增强：实验准备入口

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

上述JSON都是CPU准备/既有输出分析，没有本包新方法的GPU准确率。完整prompt资产保留在服务器；
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
| 在任何输出前冻结真实距离的参考风险预测 | [predictions.py](predictions.py) |
| 查看未来生成命令（默认不加载模型） | [run.py](run.py) |
| 已完成生成的完整输出配对分析 | [report.py](report.py) |
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
