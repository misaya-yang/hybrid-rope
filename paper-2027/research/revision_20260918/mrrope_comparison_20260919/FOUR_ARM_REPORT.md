# Llama‑3 / Llama‑3.1 × MrRoPE‑Pro / TailSpline：128K 四臂实验报告

日期：2026-09-19。状态：四臂模型执行完成；520条原始输出已同步本地并通过离线逐行复核。
本报告记录已完成实验，不启动补跑，不修改论文。15分钟监控已暂停。

## 主要结果

原生8K的Llama‑3在本次128K测试上，TailSpline为 **18.59%**，MrRoPE‑Pro为 **0.51%**，
TailSpline领先 **18.08个百分点**。更换为经过长上下文训练的Llama‑3.1后，
MrPro为 **54.15%**，TailSpline为 **49.54%**，排序反转，MrPro领先 **4.62个百分点**。

| 检查点 | 检查点原生窗口 | MrPro | TailSpline | TailSpline − MrPro |
|---|---:|---:|---:|---:|
| Meta-Llama-3-8B-Instruct | 8,192 | 0.51% | 18.59% | +18.08 pp |
| Llama-3.1-8B-Instruct | 131,072 | 54.15% | 49.54% | −4.62 pp |

每臂均为全部13类RULER任务、每类10条，共130条。分数为任务等权平均，
不是“答对条数/130”的二元准确率；多答案任务允许部分命中。
本次Llama‑3上的方法差距很大，但18.59%的绝对分数也表明该极高倍率下仍有严重能力损失。
不能把这一端点的相对优势写成128K全面可用，或所有检查点统一获胜。

同一方法跨检查点：MrPro从0.51%升至54.15%（+53.64 pp），TailSpline从18.59%升至49.54%
（+30.95 pp）。这直接显示检查点身份对本协议结果的重要性；两个检查点训练经历等同时不同，
不能将增量单独归因于某一训练因素。Llama‑3.1这里也替换了RoPE表，并非原生未修改模型成绩。

## 实验合同与实现

- 顺序：Llama‑3.1/MrPro → Llama‑3.1/TailSpline → Llama‑3/MrPro → Llama‑3/TailSpline。
- 两种方法都以L=8192、S=16构表；目标窗口131072，base=500000，64个频率对，频段[18,35]。
  Llama‑3.1的8192是构表参考，不能改写其真实131072窗口或长上下文训练经历。
- MrPro使用会议补充材料的原始构表与rotary forward；TailSpline安装旧gate已冻结的频率表，
  复用同一个forward。所有臂gain=1.2772588722239782，无权重更新、量化或结果驱动调参。
- 相同方法在两个检查点上的64个实际频率值逐项一致。不同方法之间频率表不同，gain相同。
  此前[CPU审计](CPU_TABLE_AND_CHECKPOINT_PLAN.md)记录了我方旧MrPro构造与官方构造的6个1-ULP差异；
  当前MrPro直接使用官方构造，未将“近似相同”说成旧实现全表bitwise相同。
- 输入使用既有gate的完整prompt IDs，按预存源顺序取每任务前10条，选择不依赖模型输出。
  目标为128K档，实际输入118900–130905 token，并非每条都恰好131072 token。
- 四臂统一FlashAttention2 2.8.3.post1、BF16、batch=1，单张RTX PRO 6000 Blackwell Server Edition。
  torch=2.8.0+cu128，Transformers=5.15.1，accelerate=1.14.0。
- 四臂共享Llama‑3.1 generation_config：do_sample=true、top_p=0.9；官方evaluate_one_task调用覆盖
  temperature=0.7、max_new_tokens=30、num_beams=1、eos_token_id=[128009,198]（EOS或换行）。
  配置快照里的temperature=0.6被逐行generate_kwargs覆盖，最终实际调用为0.7。
- 每行seed=20260919 + task_index×10 + source_index；四臂配对一致。
- 使用官方补充材料中的评分函数语义：prediction/reference转小写后做子串匹配；
  QA为任意reference命中，其余任务为reference命中比例；每任务均值×100先保留2位小数，再取13项均值。

实现入口与兼容性细节见[执行记录](FOUR_ARM_RUN.md)、[执行器](official_single_arm.py)
及[队列](four_arm_queue.py)。正式生成期间执行器保持一致，本地文件SHA与四臂protocol一致。
补充材料、作者示例与本次协议之间的差异见[协议审计](RULER_PROTOCOL_AUDIT.md)。
本次不是原作者HF预生成数据、历史运行环境及Table 2完整评测的逐项复现。

## 完整逐任务结果

单位：百分数；差值为百分点。每个单元格均含10条。

| 任务 | L3 MrPro | L3 TailSpline | L3 Δ(T−P) | L3.1 MrPro | L3.1 TailSpline | L3.1 Δ(T−P) |
|---|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 0.00 | 60.00 | 60.00 | 80.00 | 80.00 | 0.00 |
| niah_single_2 | 0.00 | 70.00 | 70.00 | 90.00 | 80.00 | -10.00 |
| niah_single_3 | 0.00 | 10.00 | 10.00 | 100.00 | 100.00 | 0.00 |
| niah_multikey_1 | 0.00 | 20.00 | 20.00 | 90.00 | 80.00 | -10.00 |
| niah_multikey_2 | 0.00 | 0.00 | 0.00 | 40.00 | 50.00 | 10.00 |
| niah_multikey_3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| niah_multivalue | 0.00 | 25.00 | 25.00 | 32.50 | 20.00 | -12.50 |
| niah_multiquery | 0.00 | 10.00 | 10.00 | 37.50 | 30.00 | -7.50 |
| vt | 0.00 | 0.00 | 0.00 | 74.00 | 74.00 | 0.00 |
| cwe | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| fwe | 6.67 | 36.67 | 30.00 | 60.00 | 60.00 | 0.00 |
| qa_1 | 0.00 | 0.00 | 0.00 | 70.00 | 60.00 | -10.00 |
| qa_2 | 0.00 | 10.00 | 10.00 | 30.00 | 10.00 | -20.00 |
| **13项等权平均** | **0.51** | **18.59** | **+18.08** | **54.15** | **49.54** | **−4.62** |

Llama‑3上TailSpline逐任务8胜5平0负；MrPro仅fwe非零，TailSpline有8项非零，
主要分数来自single_1/single_2（60/70）、fwe（36.67）、multivalue（25）。
Llama‑3.1上TailSpline相对MrPro为1胜6平6负；QA2差20 pp，single_2与multikey_1各差10 pp，
multivalue差12.5 pp，multiquery差7.5 pp，qa_1差10 pp；multikey_2赢10 pp。

## 与旧128K gate的关系

旧结果来源：[原始报告](../../../../experiments/iclr2027_strong_evidence_20260915/reports/pro6000_llama_s16_128k_gate.json)。
该旧报告为report-backed结果；本轮独立raw-row复核针对当前四臂，未重新复核旧raw文件。

| 原生8K Llama‑3，128K端点 | MrPro | TailSpline | TailSpline − MrPro |
|---|---:|---:|---:|
| 旧gate，原decoder / SDPA Flash | 1.282051% | 22.051282% | +20.769231 pp |
| 本次，官方评测函数解码 / FlashAttention2 | 0.513077% | 18.590000% | +18.076923 pp |

因此，“旧MrPro约3分、本次约0.3分”不是上述两份Full13报告的准确数字。
旧结果是1.28 vs 22.05，本次是0.51 vs 18.59。两次都显示此端点TailSpline优于MrPro。
本次输入沿用旧gate，但同时改变了解码设置、注意力后端、batch（旧2→当前1）及MrPro构造路径，
不能把分数变化全部归因于30-token上限、换行停止、采样或FA2中的某一项。

旧→新TailSpline任务变化：single_1为70→60，multikey_1为30→20，multivalue为37.5→25，
multiquery为22.5→10，qa_2为20→10，fwe为26.67→36.67；其他任务不变。
MrPro唯一非零任务fwe由16.67→6.67。两次方法排序与主要失效任务结构一致。

## 复核、耗时与证据保存

[离线审计脚本](audit_four_arm_results.py)对本地520条prediction重新评分并与每臂report、队列comparison逐项核对：

- 每臂130个唯一row_id、13项各10条，无缺项、重复或跳臂。
- 四臂逐行task、source_index/source_order_index、row_id、prompt与input-ID摘要、input_tokens、references、seed及generate_kwargs一致。
- 两个检查点保持不同的真实配置；软件版本、tokenizer停止ID、共享generation_config、gain与执行器身份一致。
- 同方法跨检查点的实际64频率表完全相同；设备回执全为cuda:0、attention全为flash_attention_2。
- 独立重算结果全部匹配保存报告。审计基于保存的prediction及输入身份摘要；没有再次运行模型或重新解码全部输出token。

审计为 **PASS**：[审计JSON](four_arm_results/audit.json)、[逐任务CSV](four_arm_results/per_task.csv)、
[队列完整汇总](four_arm_results/comparison.json)、[最终状态](four_arm_results/queue_status.json)。
Llama‑3两个臂的未舍入macro为0.512820513%与18.589743590%；报告采用任务先舍入的官方口径，
分别为0.513076923%与18.590000000%。展示为0.51/18.59不受影响。

| 臂 | 样本生成计时合计（秒） | 输出token合计 | 达到30-token上限的行数 |
|---|---:|---:|---:|
| llama31_mrpro | 2587.42 | 2018 | 30 |
| llama31_tailspline | 2582.75 | 1966 | 29 |
| llama3_mrpro | 2590.14 | 2071 | 44 |
| llama3_tailspline | 2571.38 | 1638 | 21 |

四臂样本生成计时合计约2.87小时，不含模型加载、安装和排队时间，不能当作完整账单耗时。
达到预算上限不必然等于错误截断；这些计数仅描述运行行为。

原始产物已同步至本目录four_arm_results，各臂保留protocol.json、runtime.json、generations.jsonl、report.json。
为维持来源身份，下载的原始文件未改写。protocol/runtime是生成前快照，其中model_execution=false
未由执行器回填；完成状态由520条实际生成行、report中的model_execution=true、COMPLETE队列和本轮审计共同确定。

| 臂 | 原始输出 | 完整报告 | 配置 | 运行时频率表 |
|---|---|---|---|---|
| L3.1 MrPro | [jsonl](four_arm_results/llama31_mrpro/generations.jsonl) | [report](four_arm_results/llama31_mrpro/report.json) | [protocol](four_arm_results/llama31_mrpro/protocol.json) | [runtime](four_arm_results/llama31_mrpro/runtime.json) |
| L3.1 TailSpline | [jsonl](four_arm_results/llama31_tailspline/generations.jsonl) | [report](four_arm_results/llama31_tailspline/report.json) | [protocol](four_arm_results/llama31_tailspline/protocol.json) | [runtime](four_arm_results/llama31_tailspline/runtime.json) |
| L3 MrPro | [jsonl](four_arm_results/llama3_mrpro/generations.jsonl) | [report](four_arm_results/llama3_mrpro/report.json) | [protocol](four_arm_results/llama3_mrpro/protocol.json) | [runtime](four_arm_results/llama3_mrpro/runtime.json) |
| L3 TailSpline | [jsonl](four_arm_results/llama3_tailspline/generations.jsonl) | [report](four_arm_results/llama3_tailspline/report.json) | [protocol](four_arm_results/llama3_tailspline/protocol.json) | [runtime](four_arm_results/llama3_tailspline/runtime.json) |

远端原始owner：`/root/autodl-tmp/mrrope_official_20260919/results/four_arm_fa2/`，
服务器`connect.westd.seetacloud.com:42853`。完整输入token IDs仍在同实验根下
`preserved_ruler/today_rope_plan_20260914/tailspline_llama_s16_128k_gate/assets/full13/inputs.jsonl`；
本地输出带身份摘要与references，可离线重评分，但本次没有再次复制约93MB输入文件或模型权重。

## 对论文86.6分争议的支持范围

[归档MrRoPE论文](paper_A.pdf)Table 2将86.6写在LLaMA3-8B-Instruct、8K→128K、Full13 RULER下；
图3却出现Llama3.1-8B (8K)，补充材料示例也指向Llama3.1，详见[来源审计](RULER_PROTOCOL_AUDIT.md)。
这些身份表述冲突与本次结果差距，都值得要求具体解释。

在本次固定输入和解码下，官方MrPro在Llama‑3、Llama‑3.1分别为0.51与54.15，
均未复现86.6；更换为Llama‑3.1显著抬高分数，但仍不能解释全部差距。
本实验提供方法/检查点的配对证据，并非作者历史Table 2运行记录，
也不是500×13数据集的总体成绩或估计区间。每任务10条足以记录本次具体差异、提出协议质疑，
不需要先补跑6500条才可报告发现；它也不把未测数据集的精确成绩、历史检查点身份或故意造假直接变成已核实事实。

本报告仅保存和分析已有实验。未增加任务、改参数追逐分数、重新启动监控、关闭实例或发送外部投诉。
