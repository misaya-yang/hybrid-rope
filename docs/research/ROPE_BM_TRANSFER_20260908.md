# BM 跨模型复测：MrRoPE 论文的 Qwen2.5-3B-Instruct

## Material Passport

日期2026-09-08，用户明确要求在其他模型复测。当前：作者追加要求直接使用MrRoPE官方3B。1.5B已按要求中止，不作方法判定；3B已完成；结果见[完整报告](ROPE_BM_TRANSFER_RESULT_20260908.md)。模型为Qwen/Qwen2.5-1.5B-Instruct与3B-Instruct，分别revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`、`aa8e72537993ba99e69dfaafa59ed015b17504d1`。权重及metadata依据见[资产定义](ROPE_BM_TRANSFER_MODELS_20260908.json)，本轮边界重新核对缓存哈希，之后仅复用未变stat；不下载替换checkpoint。

## 固定设计

原生长度32768，base1000000，head_dim128，静态S4，两模型按自身配置生成MrPro/BM。相同几何下边界均l23/h40/N17；gain=1+.1ln4。BM闭式与OLMo完全相同，不根据Qwen输出选参数。

六个官方RULER任务沿用上轮：niah_single_2、niah_multikey_2、niah_multiquery、VT、FWE、SQuAD QA。seed20260913，QA pre_samples256；每任务32K两条、128K四条，共36条/模型。两臂各完整生成，默认greedy、repetition_penalty1，模型默认EOS、原官方每任务输出预算；输入不裁切，原始文本/token IDs和实际长度留存。32K与128K为生成上限，不声称每条精确填满。

主终点128K六任务等权平均；32K同口径另报。长端增益且短端不下降支持该小面板的迁移；长增短降为取舍；无长增益不支持此次迁移。地板/天花板和逐题得失保留，不因不利结果删任务。两种规模同属Qwen2.5家族，不把两次checkpoint比较当成两种全新架构或完整RULER。

先跑1.5B，再3B。若tokenizer.json、聊天模板、特殊token与解码合同相同，3B复用完整输入；否则重新生成，不能悄悄改变模板。不复用历史使用repetition_penalty1.1或不同任务面板的聚合分数。

## 事前依据与执行审核

OLMo S4的BM独立复核16K显著高于MrPro、MrUni及官方YaRN，但S8及两项恢复方案失败，说明跨尺度不是已建立规律。历史FullLagP2在Qwen1.5B/3B有checkpoint相关取舍，且改变的不只是本轮BM中段；不能据此预判BM有效。

运行器新增32K短端与128K长端的配对汇总、分片权重stat核对，并检查GPU实际加载参数量与权重头一致；CPU相关9项测试通过。标准FP32相位、完整注意力、gain及原有decoder/scorer路径沿用已验证实现。GPU运行期间不改其源码，阶段结果与故障单独记录。没有自行添加墙钟截止时间。

## 作者追加要求：直接使用论文3B

已核对[MrRoPE原文Table2(b)与Appendix B](https://arxiv.org/html/2601.22181v1)：模型明确为Qwen2.5-3B-Instruct，32K→128K，S4；正文给出中段边界23/40，与当前模型表一致。论文的完整RULER成绩覆盖13项，本轮仍先完成已有六任务混合面板的同输入MrPro/BM比较，不把本轮均分与论文13项53.2直接判胜负。

1.5B在MrPro臂完成19/36条后由STOP中止，监督器STOPPED/exit−15；没有完整baseline或BM结果，不能据此判定容量或方法效果。原始部分输出保留，不复跑该模型。3B缓存与模型身份已经完成核对，直接启动，无下载替换。

## 冒烟范围确认

作者进一步明确：先做3B的类似冒烟；若BM仍有此前胜率，再考虑全量。当前任务只完成已冻结的36条/臂，不自动扩展到完整13任务或大样本。决策同时依据128K均分差、逐题胜负及32K损失，不能仅用某一任务或总胜场建议全量。

## 最终状态

3B两臂均完成，32K为BM91.67%/MrPro87.22%，128K为70.83%/78.13%；NO_LONG_GAIN。未启动全量。逐题重算及语义复核见完整报告，1.5B保持用户中止状态。
