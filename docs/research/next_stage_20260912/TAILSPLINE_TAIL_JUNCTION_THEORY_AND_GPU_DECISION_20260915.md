# TailSpline尾部接入理论与GPU判决

更新：2026-09-15。状态：**理论增量经独立CPU复算；不改变当前GPU主队列；clean T--C仅为条件性追加，不自动启动。**

## 1. 外部材料与结论

本轮审查作者提供的`RoPE_Theory_Rebuilt_From_YaRN_MrRoPE_20260915.md`，
文件SHA256为
`5cc80dd90477343d7b82c56859ad885227e6b491643c9094b6a08da726d620c2`。
材料中提到的`YaRN_MrRoPE_TailSpline_Theory_Audit_20260915.json`未在同一
Downloads目录找到，因此本结论不引用缺失的机器可读摘要；核心闭式与数值由本轮独立
CPU计算复核。

这份材料不是YaRN、MrRoPE和TailSpline的统一任务最优理论。它的有效新增是：将
TailSpline相对MrPro的`tail landing`从视觉平滑改写成尾部邻域相对共同`/s`参考的
有限旋转响应性质。其余`m`/`epsilon`坐标、YaRN index-linear身份、TailSpline的
one-sided唯一解、T--C同位移分解和gain边界，已经分别由A37、A38与A42覆盖。

## 2. 可进入论文的精确性质

令中频宽度为`n`，在低频边界前第`r`个槽取`q=n-r`，并定义残余缩放指数
`delta_r=1-m_{n-r}`。MrPro与TailSpline分别满足

\[
\delta_r^P=\frac{r(2n+1-r)}{n(n+1)},\qquad
\delta_r^T=\frac{r(r+1)(3n+1-r)}{n(n+1)(2n+1)}.
\]

固定`r`并令`n`增大时，MrPro为`O(n^{-1})`，TailSpline为`O(n^{-2})`。因此MrPro
以一阶残差接近fully-scaled tail，TailSpline以二阶残差接近。对`0<=d<=sL`，

\[
\sup_d\left\|R(d\nu)-R(d\omega/s)\right\|_2
=2\sin\!\left(\frac{\min\{L\omega(s^\delta-1),\pi\}}2\right).
\]

在Llama-3-8B的`K=64,b=500000,L=8192,[18,35],s=4`上，最后一个中频槽的
最大相位残差为：MrPro `1.280272`、同位移控制C `0.224123`、TailSpline
`0.102176` radians。独立复算在`n=2,...,128`上验证了闭式恒等式，数值与材料一致。

该性质只描述低频tail附近的局部接入。Llama上最坏旋转界在约第五个tail邻近槽已经
饱和，不能据此排序整个transition，也不能推出任务分数。论文安全表述是：

> MrPro approaches the fully scaled tail with first-order residual, whereas
> TailSpline approaches it with second-order residual and therefore provides a
> tighter finite-window rotation bound in the tail neighborhood.

不得将`tighter bound`改写为任务收益的已证中介。

## 3. 与现有任务证据的关系

- A39/A40检验TailSpline完整表相对MrPro完整表的任务价值；它们同时包含总位移与
  高阶shape差异。
- E1检验同总位移T--C。当前Full-13点估计为`-0.41pp`，95%区间
  `[-2.63,+1.82]pp`，且TailSpline为batch 1、C为batch 2。这个结果不支持
  one-sided shape已经产生总体任务收益；E0用于判断批处理身份是否足以影响该结论。
- A42已经说明相位变化可通过key竞争和value聚合进入模型计算，但没有证明低频tail邻域
  是A39/A40收益的经验中介。

因此这项新理论强化“TailSpline与MrPro结构上为何不同”，不升级“为何TailSpline在
任务上获胜”的因果主张。

## 4. GPU执行决策

当前顺序保持不变：

1. 完成clean 32K TailSpline--MrPro Full-13配对；
2. 完成Natural-QA631 TailSpline--MrPro真实任务配对；
3. clone GPU先完成E0的39行batch敏感性；
4. Native参照与Native-Z5沿既有队列执行；
5. YaRN后置，BM不进入当前最终队列。

不因本理论材料启动新倍率、新band、新gain、新曲线或等剂量YaRN--MrPro。

clean T--C只有在以下任一条件成立时才作为单一追加臂考虑：

- E0显示batch 1/2足以使当前E1不可解释；或
- 作者决定把tail-junction行为机制升级为论文核心claim，并接受一整个C臂的成本。

若追加，冻结当前clean 32K的2,600个prompt、batch 1、checkpoint、band、gain、
decoder与scorer，只运行C并复用TailSpline raw。禁止从输出选择子集或修改C。判决为：

| 结果 | 允许结论 | 后续动作 |
|---|---|---|
| T--C区间下界大于0 | 等总位移后的高阶shape具有任务价值；仍不称tail landing为唯一中介 | 可增加一个小型路径诊断，但不再搜表 |
| 区间跨0 | 二阶接入只保留为结构保证 | 停止机制追加 |
| T显著低于C | one-sided任务偏好被否证；T--P完整方法结果仍按原范围有效 | 停止该理论分支，不救曲线 |

## 5. 论文安放

主文最多加入一个命题、一组Llama有限网格数字和一句证据边界。完整离散推导、多频组合
界、高频入口代价以及operator-bound饱和范围进入TailSpline附录。不要把外部材料整体
并入正文，也不要使用“从YaRN/MrRoPE重建统一理论”的标题；推荐内部名称为
`Boundary regularity of scaled-tail transitions`。

