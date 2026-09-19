# MrPro CPU构表核对与检查点诊断

## 已完成：CPU构表

[可重跑脚本](cpu_table_audit.py)，仓库根运行：

```sh
python3 paper-2027/research/revision_20260918/mrrope_comparison_20260919/cpu_table_audit.py
```

[完整报告](cpu_table_audit/report.json)同时比较作者GitHub
`LlamaYaRNRadix.yarn_radix3()`、会议补充材料`LlamaMrRoPE.pro()`，
以及当前项目`build_analytic(method='mrpro', low=None, high=None)`的实际路径。
后者调用`cross_audit.tables.transform`，而非仅比较分析用的`analytic_exponents`文本。

共同输入：rotary dim=128、64对、base=500000、原生参考8192、scale=16。

| 检查 | 结果 |
|---|---|
| 频段 | 三者均为[18,35]，n=17 |
| 累计配置 | 同一有限网格二次累计公式q(q+1)/(n(n+1)) |
| gain | 精确相等，1.2772588722239782 |
| 两份作者代码的频率表 | 64/64逐位相同 |
| 作者表与当前项目表 | 58/64逐位相同；另外6项各差1 ULP |
| 最大绝对频率差 | 9.313225746154785e-10 |
| 最大相对差 | 1.1226319342308327e-7 |
| 距离131071处最大未折叠相位差 | 0.00012206938117742538 rad |
| 用相同公式重放作者FP32除法路径 | 与两份作者表均64/64逐位相同 |

差异来自计算路径：作者以FP32 tensor除以转成FP32的标量；项目用FP64计算缩放／除法后
转成FP32。重放这一步已经解释全部不同字节。没有发现频段、公式或gain错误；
不能将结果写成原项目表与作者表逐位全等，也不能据此认证GPU安装、生成或历史权重身份。

三张精确静态表供后续诊断复用：
[项目表](cpu_table_audit/project_table.json)、[GitHub作者表](cpu_table_audit/github_table.json)、
[会议补充表](cpu_table_audit/conference_supplement_table.json)。
历史gate使用的已存表仍须从其owner读取并逐项比较，当前CPU重建不是历史run原始回执。

## 待运行：最小检查点诊断

后续2026-09-19作者改为先准备[官方配方单臂](OFFICIAL_SINGLE_ARM.md)，当前端点已提供。
下述六条旧gate配对方案保留为历史方案，尚未执行；不与官方配方单臂混为同一实验。

状态：**等待用户提供当前GPU端点、现有Llama3.1检查点和gate资产路径；未运行模型。**
不下载权重，不使用历史SSH地址，不启动既有队列。

### 问题与配对合同

问题：保留原Llama3参考下的同一张MrPro S16静态表，换为Llama3.1权重后，
现有128K检索prompt是否恢复？这是检查点干预诊断，不是复现论文86.6%。

首轮固定6条：从原gate的`niah_single_1/2/3`各取源顺序前2条。
选择仅依赖任务及原始source顺序，不按任何臂的输出选择；记录其row_id、prompt IDs、
references、预算和旧MrPro输出。原gate这三个任务每个均有10条，不新增生成题。
保留同一prompt token序列、131072预算、greedy解码和原任务计分。
此小面板用于检测大幅恢复，不报告为完整RULER或显著泛化结果。

- 优先复用原`Meta-Llama-3-8B-Instruct`同6条已完成结果；核对已存静态表与新臂逐项一致。
- 新臂加载现有`Llama-3.1-8B-Instruct`权重，安装上述同一静态表和gain。
- 不按Llama3.1的128K窗口重新计算MrPro表；模型真实训练身份与表的8192参考分别记录。
- Llama3.1有自带的`llama3` RoPE scaling。现有`install_static`拒绝scaled checkpoint，
  `recovery_v2_runtime.table_for_config`也不支持直接把该模型当原始Llama3处理。
  不绕过其检查后叠加表；为这个明确诊断直接替换完整rotary模块，保留权重原样。
  使用会议补充材料式的FP32 position_ids乘法和同一静态表，并检查forward前后表/gain未变。
- 记录并对齐tokenizer、特殊token身份、EOS/stop集合、输出预算、精度和generation参数。
  不把模型各自的generation_config静默变化当作权重效果。
- 根据已授权主机的现有receipt选择稳定prefill路径；必要的原Llama3同输入runtime核验
  仅用于定位新实现引入的差异，不重跑完整130条。

### 结果解释

- 若Llama3.1在同表、同prompt下大幅恢复：支持checkpoint是本地差异的重要原因；
  不能证明作者86.6%历史上就是用Llama3.1跑的。
- 若没有恢复：不自动排除作者checkpoint身份问题。原生Llama3.1表与被替换表不同；
  可先检查原生Llama3.1在同少量输入的表现，再决定是否需要下一项诊断。
- 需要解释论文86.6%的来源，最终仍要其真实运行配置／权重身份和逐任务结果。
  成功的小诊断不升级成论文全部13任务分数的复现。

## 自然QA旁证的口径纠正

现有41条LongBook QA报告中的20.17与13.21是**平均token F1百分数**，并非正确题比例。
按报告已存逐样本分数，MrPro/TailSpline分别24/18条F1>0，两者F1=1均为0条。
因此“MrPro答对16/41、比TailSpline多4题”不由该报告支持。
这些结果表明仍有部分答案重合，但不能用来证明实现一定正确。
未重新评分原始生成文本，未改写已有结果。
