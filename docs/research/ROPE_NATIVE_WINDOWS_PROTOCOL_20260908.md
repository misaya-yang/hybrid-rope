# 独立Native窗口形成与全局读取

状态更正（2026-09-15服务器回查）：两模型`native_windows_full_01`均已完成。
属于分块形成/缓存坐标交接，不宣称文献新颖性。以下保留事前协议，不再当作未跑新方案。

实际NativeWindowMrPro在Qwen3B32K/128K为62.22/43.75%，相对同输入MrPro低25.00/34.375pp；
OLMo4K/16K为51.53/16.49%，相对MrPro高13.68/13.72pp，但不是相对原始Native的收益。
Qwen36条、OLMo72条均完整，本轮重新聚合已有raw分数。详见
[当前审查与回执](reviews/NATIVE_BENEFIT_AND_METHOD_LINEAGE_REVIEW_20260915.md#42-原生窗口独立形成已经测过)。

前两项否证约束了设计：Mr完整长前缀交给BM读取未保住OLMo正例；只恢复纯bias位置项
未修复Qwen负例且破坏正例。这里保留的是每块完整Native计算，而非再拆一个学习组件。

## 唯一形成规则

取模型原生配置L0=max_position_embeddings；OLMo4096、Qwen3B32768。
将前n−1个原prompt token按源顺序分成长度最多L0的连续块。每块以空cache、局部位置
0..块长−1、Native频率和gain1独立完整prefill。无复制header、重套chat模板、padding或
按答案选块。每个历史token恰好处理一次，捕获真正pre-RoPE K及V。

对冻结读取表T∈{MrPro,MrProBM}，把每层原始K重新旋转成a R_T(全局位置)K_raw，V保留；
所有块合并为一个完整历史cache。最后prompt token及全部生成使用T，所有历史token可见。
Mr/BM两个读取分别报告，不逐题择优或混成一个方法分数。源窗口形成在两读臂间复用。

同一块内共同位置平移在RoPE中抵消，因此保留了独立原token片段的Native函数，包括
bias/content项之间的关系；不保证原完整prompt状态保持。跨块历史依赖被明确删除：
各块历史K/V不依赖别块tokens，全局重旋不能恢复这些依赖。指代、赋值链、跨文档推理
及chat角色解释可能受损，必须由最后query和decode承担。

## 判别样本与规则

OLMo沿用single_1_16384_1、qa_2_16384_13两条0→1正例；Qwen3B沿用MK2_131072_1、
VT_131072_1两条BM负例。Qwen多键正确记录位于第二个32K块内部，检验完整局部绑定
能否恢复；VT/QA暴露跨块推理代价。两读取都无法改善简单检索例则停止当前分块规则；
若检索恢复而VT/QA下降，按整体配对面板评估实际收益与代价，不要求每子项都赢。

每块核对源Native旋转K与真实cache逐位一致；全局重旋读取原始K，不逆转已舍入cache。
CPU对Qwen/OLMo微型真实架构检查了：改变前块不会影响后块K/V、末尾非整块、独立复数
全局旋转对照、全量cache长度及继续解码。尚不是GPU能力或性能结论。

预填充attention算量从约L²降到LL0；投影/MLP和完整KV存储仍在，decode仍读全历史。
实际耗时记录，不能只靠这个估算宣称加速。

## 历史区别

撤回的w512/P2是保留全attention的按距离换相位；9月3日Native前缀交接仅首窗口
原生形成，后续恢复跨块传播，且只有CPU验证。本方案每块独立Native形成，未找到
同构的已完成本地负结果。与PCW等分块先例的关系需继续核对，不以此占新颖性。
