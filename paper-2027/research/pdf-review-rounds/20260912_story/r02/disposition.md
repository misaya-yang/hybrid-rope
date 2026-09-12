# 第二轮处理记录

1. 采纳并按代码补写：MLA使用共享5M-token缓存、GPT2、RandomState9999、每长度8窗口、全序列L-1目标、先每seed取exp(meanNLL)再算三seed均值/SD；恢复代码与provenance登记SHA一致。明确历史准备代码用FineWeb-Edu sample-10BT训练split的shuffle99999/buffer10000，不能将其声明为已认证文档独立holdout；旧cache上游revision与token-array hash不在portable记录中。M4明确WikiText2 train/validation分开构造、repeat/trim、NeoX tokenizer、1,048,576验证tokens、4 offsets、完整目标与配置统计单位。没有编造旧数据版本。151.9M绝对四格继续记为未追回，保留权威配对结果。
2. 在式(7)首次出现位置说明geometry确定对象、目标为显式design prior；不把它作为已证明的下游预测器。保留构造、理论及全部正结果。
3. 主图与绘图脚本统一whole-response token F1，caption明确包含触顶响应并指向EOS表。原有分项与EOS计数保留。
4. 主文增加factorial实际128步/重复语料/约0.01NLL说明；六任务直接标48 long prompts，不重跑或加benchmark。
5. 首次RULER使用补hsieh2024ruler；同时发现并消除refs.bib中RULER和Resonance的重复键，保留完整正式条目。

最终修改后格式编译通过：正文9页，声明另计；0未定义引用、0pt overfull；正式模板字节已核对。新增文字与图注的最终视觉/源码包复验记录于STORY_RESTRUCTURE_20260912.md。
