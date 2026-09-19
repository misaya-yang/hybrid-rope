# RULER 86.6% 与 1.28%：评分和协议核对

2026-09-19 UTC。此记录补充之前仅凭PDF的比较，不修改独立审阅原文或论文。
作者要求核对判定标准；只做公开代码、现有评分路径与报告算术的静态检查，没有运行模型。

## 结论

**目前没有依据把86.6%与1.28%的差异判为我方基线实现错误，或作为现稿的拒稿理由。**
新获得的原文补充材料显示，评分函数一致，但示例模型路径、默认任务集和生成协议与
我方完整13任务实验未对齐。原论文表格与所发示例代码之间也存在需要澄清的身份差异。

这不是证明86.6%不可能、虚假，或已确定其来源；脚本默认值不是原始运行记录。
在缺少实际检查点配置、运行命令和逐任务原始结果的情况下，不应把原因归给任何一方。

## 检查到的直接事实

1. 从[原论文OpenReview补充材料](https://openreview.net/attachment?id=1J63FJYJKg&name=supplementary_material)
   下载的ZIP保存在本机 `ruler_protocol_sources/mrrope_supplement.zip`（未入库，
   按`identity.json`记录的sha256核对）。这是与当前同名GitHub开发仓库不同的代码包；
   结论以会议补充材料为准。
2. 同一目录下本机保存的`evalr.sh`第1—5行示例使用
   `--original-max-position-embeddings 8192 --yarn 16 --model models/llama3.1-8b-ins`。
   该脚本没有单列Llama MrPro命令。原文表2标记Llama3-8B-Instruct，原文图3又标记Llama3.1-8B。
   这些标注不能确定论文实际加载的权重身份。
3. 本机保存的`ruler.py`第125行默认只有7个任务：
   `qa_1, vt, fwe, niah_single_1, niah_multikey_1, niah_multiquery, cwe`。
   发布的evalr.sh没有传`--tasks`覆盖项；原文表2却写全部13任务。
   默认样本数是100/task。不能因此断言原论文最终仅跑7任务，但发布命令不足以复现其表格身份。
4. 同文件第99行读取预生成的`SaylorTwift/RULER-{length}-llama-3.1-tokenizer-chat-template`。
   生成统一使用`max_new_tokens=30`，且换行token也作为EOS（第56、60行）。
   我方13任务则使用自己的固定源顺序生成面板、逐任务输出预算，并保留所有空输出和截断输出。
5. 两个补充材料评分函数与[NVIDIA官方实现](https://github.com/NVIDIA/RULER/blob/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/scripts/eval/synthetic/constants.py)
   的AST完全一致，均为字符串包含匹配：QA取任一参考答案命中；其他任务按参考答案命中比例给部分分。
   官方版本正是我方固定的`c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`。
   核对身份与官方函数快照见同目录`identity.json`与`nvidia_constants.py`（均在本机保存，未入库）。
6. 我方评分路径为`recovery_v2_eval.py`调用
   [ruler_bench.py](../../../../scripts/experiments/olmo_fast_screen/ruler_bench.py)的`score`。
   它采用上述包含匹配与官方控制字符预处理，不要求完整字符串exact match或末尾EOS才给RULER分。
   [128K报告](../../../../experiments/iclr2027_strong_evidence_20260915/reports/pro6000_llama_s16_128k_gate.json)
   保留每任务10条、全部13任务；重算逐任务均值后MrPro为1.282051%，TailSpline为22.051282%。
   此次是报告算术和代码定义验证，没有重新回放原始生成行。

## 模型身份的重要性

Meta的[Llama-3.1-8B-Instruct模型卡](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
给出128K上下文。**若示例路径确实加载该检查点，就不能把它当作仅在8K训练的Llama-3-8B。**
设置构造参数`original_max_position_embeddings=8192`不会抹去权重已有的长上下文训练经历。
我方报告明确指定`Meta-Llama-3-8B-Instruct`、原生8192。
但本轮没有补充材料中本地模型目录的权重／配置文件，因此只确认身份存在歧义，不判定最终运行身份。

## 对此前讨论的纠正

之前将这个分数差异放在“可能拒稿”的例子里，未先查会议补充代码，证据不足。
现在已有直接材料说明可比性尚未建立；应撤回它作为我方已知缺陷的暗示。
相同判定函数不等于相同检查点、任务集、数据和解码协议，也不能把NIAH的ROUGE-1召回
与完整RULER13任务均分混为一谈。本文既有方法结论继续以内部配对比较为依据。
