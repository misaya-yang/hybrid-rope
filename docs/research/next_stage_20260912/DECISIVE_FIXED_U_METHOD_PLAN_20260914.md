# 固定-u方法的决定性实验闭环

更新：2026-09-14。状态：**解析构表与本地统计代码已准备；没有启动GPU、没有新增模型结果。**
本文替代 `ROPE_THEORY_AND_TODAY_PLAN_20260914.md` 的执行优先级，不覆盖其中已经成立的
数学恒等式或历史结果。

## 1. 先回答是否闭环

原E1（table×gain）、E2（单节点phase/donor patch）、E3（仅OLMo fixed-u）即使全部成功，
也**不能**闭合“零训练静态表跨模型稳定超过MrRoPE/YaRN”这一核心问题：E1只能做归因，
E2只能证明一个局部计算中介，E3只验证一个checkpoint和一条倍率迁移。它们不会自动产生
跨模型方法。

当前可以形成的最小方法候选是 **scale-covariant fixed-u allocation**。它尝试闭合一个比
“普适最优表”更窄、但可执行和可证伪的问题：给定一条全局冻结的S4参考allocation，目标
倍率变化时不再机械冻结指数 `m`，而保持band内归一化log-frequency坐标 `u`。参考曲线的
`0.75`及32圈/1圈band仍是冻结经验常数；理论唯一导出的是倍率迁移，不应谎称从第一原理
推出0.75。

因此，它不是靠CPU推导已经闭环；闭环与否由下面的跨模型结果决定：

- 若下面三个判决全部通过，论文可以形成一个完整零训练静态方法及其跨模型证据；
- 若任一核心判决失败或未决，当前方法没有解决核心问题，停止而不是继续发明proxy；
- 即使全部通过，也只支持所测checkpoint、倍率、任务和冻结推理YaRN对照，不支持普适最优，
  也不支持胜过配套继续训练后的完整YaRN。

## 2. 一句话科学问题与中心主张

**科学问题：** 已在S4冻结的有效内部指数分配，怎样随目标倍率迁移，才能避免MrRoPE/YaRN
式固定transition在更大倍率下继续放大中频资源偏移？

**待判中心主张：** 在已登记且事前冻结的band、固定参考曲线和固定gain规则下，保持band内归一化
log-frequency坐标的静态表，比固定指数迁移更稳定，并在冻结Llama、OLMo和未参与设计的
Qwen2.5-3B上获得高于MrRoPE与static YaRN的区间质量。

这比现有“z matters”前进一步：它提出一个确定的倍率迁移规则。当前它是待实验判决的主张，
不是已有结论。

## 3. 理论链与最终方法

固定Native band端点 `omega_l,omega_h`，令

\[
A=\log(\omega_l/\omega_h),\qquad
t_k=\frac{\log(\omega_l/\omega_k)}{A}.
\]

部署表写成 `nu_k=omega_k S^{-m_k}`。槽位在扩展后整个band中的归一化log坐标为

\[
u_S(k)=\frac{A t_k+(\log S)m_k}{A+\log S}.
\]

MrRoPE、BM以及此前的倍率迁移都固定 `m_k`。但

\[
u_S-t=\frac{\log S}{A+\log S}(m-t),
\]

所以倍率越大，任何 `m!=t` 的内部资源偏移都会被系统放大。现有mix075满足内部槽
`m>t`；固定m会在S8比S4进一步强化中频减速。这与Llama S8的FWE/Native损伤和任务交换
相容，但不是由这些结果反推公式。

固定参考倍率 `S_0=4` 和参考曲线

\[
m_0=\tfrac14 m_{BM}+\tfrac34 m_{front},
\]

要求 `u_S=u_{S_0}`，唯一得到

\[
\boxed{
m_S(k)=\alpha m_0(k)+(1-\alpha)t_k,
\quad
\alpha=\frac{\log S_0(A+\log S)}{\log S(A+\log S_0)},
\quad
\nu'_k=\omega_k S^{-m_S(k)}.
}
\]

对 `S>S_0`，`0<alpha<1`，所以单调性、Native高频前缀和完整`/S`低频后缀都保留；
没有alpha搜索。实际构造必须使用保存的Native FP32数组计算 `A,t`。主方法使用事前固定
gain `g(S)=sqrt(1+0.1 log S)`；纯频率归因同时使用同gain的fixed-m对照。若后续事实证明
该gain规则不稳定，结果应写成频率方法在共同gain下的结论，而不是按模型重选gain。

当前Llama `[16,34]`、OLMo `[14,31]`、Qwen `[22,39]` 是已有owner冻结的实验band，并不
精确等于同一个32圈/1圈解析边界；fixed-u理论也没有导出这些边界。Qwen2.5-3B沿用相同
RoPE几何下已冻结的`[22,39]`，不能看分数再改。若论文要求一个对任意新几何都无需既有
band的完整规则，当前方案仍未闭合该更强目标。

MrRoPE少考虑的具体一步是**倍率间的资源坐标协变性**：它给出一条累计radix/exponent
曲线，却不约束同一内部槽在总扩展log-band中的相对位置随S保持什么。fixed-u补上这一点。
这能解释为什么它相对fixed-m更少过度减速中频；它不构成任意未知Q/K上准确率必升的定理。

## 4. 只保留三个实验判决

### A. OLMo便宜首判：方法是否比自己的fixed-m反事实更好

- checkpoint：OLMo-2-0425-1B-Instruct，S4→S8，band `[14,31]`；
- 两臂：fixed-u 与 fixed-m，只改迁移规则，表、band、端点、gain、prompt、decoder全匹配；
- 数据：只从已有完整测试池冻结，使用4K/16K/32K，按16/24/8行每任务分层；不生成72条
  新小集，不读取中途分数；
- 主终点：配对task-equal log-AUC `fixed-u - fixed-m`；Native、32K、worst、FWE/VT/MK2、
  EOS/cap同时报告；
- 强基线BM/MrPro/static-YaRN仅复用完全同prompt输出，否则补同块，不用历史总体均值。

**推进核心问题的方式：** 这是scale-covariance原则的第一个真实任务反事实，不是再次证明
mix075有用。若它不能优于fixed-m，方法的新增理论步骤没有任务价值，后续立即停止。

### B. Llama跨模型复现：是否修复已知倍率迁移损伤

仅A通过后运行。使用Llama S4→S8 `[16,34]`、已有完整测试集及已有输出；补齐fixed-u、
同gain fixed-m和缺失的同prompt强基线。主终点仍为区间AUC差；Native@8K与FWE为预注册
guardrail，64K不得因短端改善而隐藏。

**推进核心问题的方式：** OLMo与Llama都正，才说明新增规则不是单checkpoint修补；同时检验
“减少S8额外中频减速”是否缓解Llama已知Native/FWE代价。

### C. 未参与设计的Qwen2.5-3B最终方法确认

仅A、B都通过后运行。checkpoint身份明确为Qwen2.5-3B-Instruct（Native 32K，不冒称官方
YaRN训练checkpoint）；S4目标区间32/64/128K。参考S4曲线在这里直接安装，因此本实验判决
完整预注册方法对canonical MrPro、BM、static YaRN，而不是再次估计alpha。使用现成完整测试
池、同prompt、完整生成和official scorer；不以72条小样本作主结论。

**推进核心问题的方式：** 这是未参与曲线选择的第三checkpoint；只有它对MrPro与static YaRN
的AUC均为正，才能把方法写成跨模型稳定改进。

### 条件消融：原E1

只有A、B、C支持最终方法后，才补Llama S4的2×2 table×gain缺口，回答收益是否主要由gain
造成。它用于论文归因，不是方法选型，不允许加新gain点。原E2单节点干预从主队列删除；即使
成功也不改变候选排序，仅在正文机制证据确有缺口且主方法已成立时作为附录诊断。

## 5. 明确停止条件

1. 表hash、Native数组、band端点、gain或prompt身份不匹配：工程停止，修复身份后原样恢复，
   不把它记为理论失败。
2. OLMo完整预注册块的fixed-u相对fixed-m AUC不为正，或区间跨零且Native/worst没有预注册
   的明确改善：方法证据不足，停止Llama/Qwen，不扫alpha、band、tail或gain。
3. OLMo通过而Llama不通过，或Llama Native/FWE仍保持原有大幅损伤且没有AUC/worst补偿：
   拒绝“稳定scale-covariant改进”，停止Qwen扩展。
4. Qwen2.5-3B未同时超过MrPro与static YaRN的配对AUC，或改善只来自单任务/单端点：不写
   跨模型方法claim，不追加第四模型救平均。
5. 三项全部通过后停止搜索，登记raw、合同、表和报告，进入论文改写；不再用E2、margin、KL、
   NLL、Q/K均值或新proxy追求更漂亮的解释。

“区间跨零”可以在统计学上称未决，但在本轮论文决策中等同于**不能支持该方法并停止扩展**。

## 6. 成功后的论文结构

- **claim：** 固定支持下z是独立设计自由度；对有效参考allocation，倍率迁移还必须保持资源坐标
  协变，fixed-u在所测冻结模型上比MrRoPE/static YaRN更稳定。
- **method：** 事前冻结的已登记band + 冻结mix075参考曲线 + 上述唯一fixed-u公式 + 事前gain规则；全层、
  全长度一张静态表，零权重更新。
- **theory：** 从RoPE相位band、固定m的倍率偏移放大，到唯一保持u的构造及可行性证明。
- **experiment：** A/B为fixed-u对fixed-m的跨模型纯迁移识别，C为未参与设计checkpoint上的
  强基线确认，E1只做必要gain消融。
- **narrative：** YaRN/MrRoPE说明不同频段不能统一缩放；本文先识别内部allocation，再指出
  现有方法缺少跨倍率资源坐标合同，给出唯一协变迁移并在真实任务上判决。

若A/B/C不能全部成立，论文保留现有z发现与条件构造，不把fixed-u提升为最终方法；这时当前
资料仍缺的是一个**能事前选择参考shape且跨模型保持任务方向的功能原则**。未知的signed
`A_k,B_k`意味着仅由RoPE几何不能证明该原则；继续改名发明几何proxy不会填补它。

## 7. 代码与证据边界

- `experiments/fixed_rope_three_interfaces_20260913/tables.py`：新增fixed-m/fixed-u解析迁移并
  直接产生现有resident runner可接受的冻结表receipt。
- `math_and_transport.py`：31项CPU恒等式和实际数组capsule；不是模型证据。
- `factorial_report.py`：E1四格同prompt分解；不称Core-6新AUC。
- `causal_intervention.py`：E2实现保留但不进入主队列；没有作者再次指定时不运行。
- 运行继续复用现有`pipeline.py`、`resident_eval.py`和`matched_generation_report.py`；不得
  重放旧`night_run`。

构造两条唯一主对照（OLMo示例；实际路径以新服务器clone后的owner为准）：

```bash
python -m experiments.fixed_rope_three_interfaces_20260913.tables scale-transport \
  --config /path/to/OLMo/config.json --parent /path/to/s4_parent_receipt.json \
  --scale-from 4 --scale-to 8 --low 14 --high 31 --mode fixed_m \
  --gain 1.0990651274 --candidate-id olmo_s8_fixed_m_common_gain \
  --model-id olmo2_1b --role control --changed-variable scale_transport \
  --out /path/to/tables/olmo_s8_fixed_m_common_gain.json

python -m experiments.fixed_rope_three_interfaces_20260913.tables scale-transport \
  --config /path/to/OLMo/config.json --parent /path/to/s4_parent_receipt.json \
  --scale-from 4 --scale-to 8 --low 14 --high 31 --mode fixed_u \
  --gain 1.0990651274 --candidate-id olmo_s8_fixed_u_common_gain \
  --model-id olmo2_1b --role candidate --changed-variable scale_transport \
  --out /path/to/tables/olmo_s8_fixed_u_common_gain.json
```

Qwen2.5-3B的S4前瞻表不是从测试分数拟合，直接用同一冻结参考式：

```bash
python -m experiments.fixed_rope_three_interfaces_20260913.tables analytic \
  --config /path/to/Qwen2.5-3B/config.json --method mix075 --scale 4 \
  --low 22 --high 39 --gain 1.0670658068 \
  --candidate-id qwen25_3b_s4_fixed_reference --model-id qwen25_3b \
  --role candidate --changed-variable exponent_allocation \
  --out /path/to/tables/qwen25_3b_s4_fixed_reference.json
```

两个receipt进入现有`pipeline.py`合同后，由`plan_queue`按prompt hash只补缺失行；不得绕过
coverage直接全量重跑。主报告继续使用`matched_generation_report.py`。只有三项主判决通过后，
才运行`factorial_report.py`；`causal_intervention.py`没有主队列命令。

已有A31–A35仍按[证据索引](../../../paper-2027/research/evidence/index.md)的身份使用。新代码通过
或新manifest生成不新增paper evidence；只有GPU完整结果产生后，才更新对应owner、最近索引和
asset registry。
