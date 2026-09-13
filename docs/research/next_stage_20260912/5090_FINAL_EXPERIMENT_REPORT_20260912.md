# 5090 收尾实验报告：OLMo BM 适配与 Llama 8B 倍率交叉

更新：2026-09-12（服务器运行跨入 2026-09-13）。本报告是本轮 5090
执行的结果 owner；服务器原始输出保留在 ignored 本地盘，不属于 Git 交付。
代码入口见[OLMo recovery实验目录](../../../experiments/olmo_recovery_20260912/index.md)。

## 结论与停止决定

1. **BM 是有效但非统一的解。** OLMo-2-1B 的 BM 适配把简单检索能力稳定推到
   16K；Llama-3-8B 的 BM 在 32K 检索和 16K–64K 自然生成均远好于 Native。
   但同一 Llama 上，g8 表在 4×/8×合成检索转为 MrRoPE-Pro 小幅领先，排序
   随模型、目标倍率和任务改变，不能把一张静态 BM 表当成最终长窗方案。
2. **本轮最佳 Llama 训练端点是 BM_g4 8K-step100。** 它在 4–32K held-out
   LM 上一致降低 NLL，13个自然任务×长度格子的宏均值由 33.23% 提高到35.01%，
   且 8/16/32K 小型检索保持 100/100/97.92%。从该点继续10步16K暴露只进一步
   小降 NLL，自然任务总体回落到34.58%；该分支停止且checkpoint已删除。
3. **64K能力需要正确倍率，而非把g4硬外推。** BM_g4在64K三项检索均为0；
   换成目标8K→64K的g8后，BM/MrPro分别达到72.92/77.08%。这不是评分口径
   差异：主分数逐条调用NVIDIA RULER官方contains式`string_match_all`，另记
   exact、EOS和hit-cap仅作诊断。
4. **按作者决定停止继续堆实验。** 当前证据要求新的、能预测模型/倍率/任务
   交叉的理论，而不是继续给BM或LoRA做无边界调参。本轮没有运行完整RULER，
   下列小面板不能冒充论文公开总分。

## 共同口径

- 模型：OLMo-2-0425-1B-Instruct与Meta-Llama-3-8B-Instruct；均使用可信克隆，
  不做重复SHA扫描。
- RULER主分数：参考答案经官方大小写不敏感contains匹配出现在完整输出中；
  多答案任务按命中比例计分。规范化完整答案+EOS、空答和生成上限命中率单列，
  不替代official task score。
- 自然任务：held-out LongAlign、LongAlpaca、LongCite及短UltraChat，报告完整
  响应token F1；生成预算等于金答案token数并包含终止EOS。它比contains严格，
  但不是事实正确率或人工质量评价。
- 64K推理使用8K分块prefill并累计完整KV cache，不截断上下文。该路径先在8K的
  single/multikey/multivalue各一条上与一次性prefill做逐token比对，三条完全相同。

## OLMo-2-1B：BM训练有效到16K，不能据PPL外推检索

BM_g4采用r32/alpha32、全层QKVO+FFN LoRA、CPT+长SFT+短replay+Native KL。
8K200步耗时260.24秒、主输入3,349,872 tokens、峰值CUDA 16,415,366,144
bytes；继承adapter和优化器的16K200步耗时515.74秒、主输入6,395,688 tokens、
峰值29,504,773,120 bytes。所有loss和梯度有限。

| 端点 | 4K NIAH | 8K NIAH | 16K NIAH | NLL 4K / 8K / 16K / 32K |
|---|---:|---:|---:|---|
| frozen BM | 100% | 100% | 68.75% | 3.1595 / 3.1586 / 3.1777 / 4.9214 |
| BM 8K200 | 100% | 100% | 81.25% | 2.8809 / 2.8688 / 2.8625 / 4.5403 |
| BM 8K200→16K200 | 100% | 100% | 93.75% | 2.8831 / 2.8734 / 2.8747 / 4.4707 |

8K200在980行E3上的4K/16K task-equal official分数为73.80/45.09%；frozen BM
为66.70/35.71%，即训练后增加7.11/9.39pp。16K仍低于C42V24的50.08%。
631条长输入自然QA的task-equal F1为25.10%，与frozen BM 25.33%和C42V24
25.80%接近；任务间有重新分配，不能用总均值声称全面提升。

BM 8K200和16K200在32K/64K简单检索均为0/16，且全部打满生成上限。
32K NLL下降而检索仍为零，直接否定“更低PPL即可证明使用远距证据”。

### 零训练区间曲线

512/1K仅single，2K–16K为single/multikey/multivalue各16行。2K–16K三任务
均值/log-length AUC/最弱点如下：BM为87.24/91.14/70.31%，对称Beta
gamma=1.5为86.20/91.14/66.15%，gamma=3为90.36/92.57/79.17%，Native均值
24.54%。gamma=3是有用的OLMo局部方向，但本轮未在Llama或自然任务上确认，
不升级为通用新方法。

## Llama-3-8B：BM_g4 LoRA与32K内健康度

Llama使用与历史frozen BM逐FP32值相同的BM_g4表，七类线性层r32/alpha32，
BF16、activation checkpointing、eager Blackwell flash路径。8K-step100峰值
CUDA 22,399,232,512 bytes；16K单步试探峰值27,060,842,496 bytes、GPU利用率
100%，证明32GiB 5090可以稳定进行同语义16K训练。

| 端点 | NLL 4K | 8K | 16K | 32K |
|---|---:|---:|---:|---:|
| frozen BM_g4 | 2.5882 | 2.5950 | 2.5319 | 2.5044 |
| 8K-step100 | 2.4733 | 2.4857 | 2.4221 | 2.3960 |
| 再加16K-step10 | 2.4720 | 2.4843 | 2.4206 | 2.3943 |

8K-step100的三项小型检索在8/16/32K为100/100/97.92%，没有退化。自然生成
13格宏均值/EOS率由frozen BM的33.23/32.69%升至35.01/51.92%；分长度看短程、
8K和32K提高，16K由29.83%降至27.94%。16K-step10总体为34.58%，低于
8K-step100；“NLL继续下降但任务不升”再次出现。因此没有继续到50/100/200步。

## Llama零训练：倍率交叉是核心新事实

下表固定Llama checkpoint、任务、样本、gain和目标倍率；每格只有三任务×4行，
使用official contains分数。g8表示按8K原生窗目标64K重新构造，而不是把g4
表直接跑到64K。

| g8表 | 8K（1×） | 16K（2×） | 32K（4×） | 64K（8×） |
|---|---:|---:|---:|---:|
| BM_g8 | **97.92%** | **100.00%** | 89.58% | 72.92% |
| MrRoPE-Pro_g8 | 95.83% | 85.42% | **93.75%** | **77.08%** |
| BM−MrPro | +2.08pp | +14.58pp | −4.17pp | −4.17pp |

同尺度g4@32K则是BM 97.92%、MrPro 72.92%，BM明显占优。由此不能写成
“MrPro总是更远”或“BM总是更强”：BM在4×目标和g8的前半段更好；g8到
4×/8×时MrPro的渐进分配更稳。这个交叉应成为后续理论必须预测的现象。

### 自然任务不复现同一排序

512–32K的13格自然F1宏均值为：Native 22.54%、BM_g4 33.23%、MrPro_g4
32.37%、BM_g8 33.23%、MrPro_g8 33.32%。Native在16/32K仅2.25/2.09%；
四张扩展表在这些长度约28%–30%。g8的32K自然F1为BM 29.68%、MrPro
28.17%，与32K合成检索排序相反。

64K自然小面板只含LongAlign和LongCite各4条，task-equal F1为Native 2.11%、
BM_g8 21.70%、MrPro_g8 18.70%。其中LongAlign为2.74/29.11/23.12%，LongCite
为1.47/14.29/14.29%（顺序均为Native/BM/MrPro）。这支持真实长输入能力，
但样本极小且LongAlign/LongCite标签各有既有限制，只能作最后诊断。

## 执行修正与未计入科学结果的失败

- 首次64K frozen运行不必要地挂载零初始化LoRA外壳，在`lora_B`申请显存时OOM；
  frozen推理移除该恒等外壳后确认仍有一次性65K prefill的MLP峰值问题。
- 分块prefill第一版使用默认SDPA，对累计KV的非方形因果形状无Blackwell kernel；
  改为PyTorch lower-right causal bias后通过短样本逐token一致性验证，再运行64K。
- BM_g4 64K的0%同时由项目scorer、上游NVIDIA `string_match_all`和存档分数
  三路重算一致；原始输出表现为空答或无关重复并命中cap，不是exact/contains混淆。
- Llama compile smoke实际完成但早期轮询竞态误判退出，随后其checkpoint被删除；
  正式step1–100全部来自同一eager lineage，没有混用两条优化器轨迹。

## 理论交接

下一步不应再问“BM还是MrPro谁更强”，而应解释并预测：为什么同一g8表在
1×/2×偏好BM、4×/8×偏好MrPro，同时自然任务又不跟随检索排序。最低要求是：

1. 将目标倍率、模型已学频段使用和任务所需局部/远距分辨率放入同一可检验模型；
2. 由机制预测排序交叉位置，而不是事后拟合一条新曲线；
3. 允许层或距离承担不同分配，但必须给出静态表失败的具体反例与可证伪预测；
4. 保留OLMo上BM/C42V24成功和Llama/Qwen上MrPro长端成功，不删反例换统一叙事。

一个直接但尚未验证的方向，是把BM与MrPro看作同支持、同gain下不同“压缩预算
到达时序”，推导随相对距离变化的最优分配或层间职责；本轮数据只支持提出问题，
不支持宣布长度自适应blend或层混合已经有效。

## 远端结果位置与保留状态

- OLMo根：`olmo_recovery_20260912/`；主要owner为
  `bm_g4_{8k,16k}200_*_20260913/`与`zerotrain_interval_20260913/`。
- Llama根：`llama8b_recovery_20260912/`；主要owner为
  `bm_g4_step100_*`、`{native,bm_g8,mrpro_g8}_32k64k_small_20260913/`、
  `*_transfer_512_32k_small_20260913/`及`*_transfer_64k_*_small_20260913/`。
- 保留模型权重、OLMo最终实验checkpoint及Llama BM_g4 step100 checkpoint。
  Llama 16K-step10过渡checkpoint（约961MiB）、未完成64K LongCite分词目录
  （约2.5GiB）、失败运行目录及680MiB编译缓存已经清理；已固定的8行64K
  自然面板保留。清理后数据盘剩余约23GiB。
- 最终远端审计确认16份关键summary均为`COMPLETE`、无GPU任务或实验进程；
  `shutdown -h now`正常返回，随后SSH banner连接超时，服务器已离线。本报告
  不声称远端原始输出已进入Git，也未提交或推送仓库改动。
