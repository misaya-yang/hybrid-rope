# 固定 RoPE 表的两边界理论草案：慢端覆盖、快端失真与 checkpoint 修正

更新：2026-09-13。状态：**可证伪理论草案，不是已证明方法或最终 band**。
本文承接[Band位置最小筛选](BAND_POSITION_MINIMAL_SCREEN_20260913.md)和
[跨模型与跨倍率早期结果](BAND_NEXT_STAGE_RESULT_20260913.md)，尝试把 transition
band 的两个边界分开解释。当前三个模型上的开发最优点只用于检验假说，不作为边界定义。

## 1. 对象与三类陈述

对模型 $M$，令 Native 长度为 $L_M$，RoPE base 为 $b_M$，每个 attention
head 有 $K_M$ 个二维 rotary pair：

\[
\omega^N_{M,k}=b_M^{-k/K_M},\qquad
\nu_{M,k}=\omega^N_{M,k}S^{-m_k},\qquad H=SL_M.
\]

一张固定表在全部层和全部运行长度使用同一个 $m$。本文先讨论单调、三段可读的
profile：$m=0$ 的快频平台、从 $l$ 到 $h$ 的 transition，以及 $m=1$ 的
慢频平台。最终优化不要求预先存在三段；这里的 (l,h) 是解释和低成本初始化参数。

下文严格区分：

1. **恒等式**：由 RoPE 和 $H=SL$ 直接得到，不依赖模型效果；
2. **经验假设**：关于模型保留和长距检索，需要实验支持；
3. **checkpoint 修正**：由冻结 Q/K 权重与激活估计，不允许查看目标 band 分数后再拟合。

## 2. 慢端：约一圈是覆盖边界，不是能力定理

定义 Native 窗口内的转圈数：

\[
t^N_{M,k}=\frac{L_M\omega^N_{M,k}}{2\pi}.
\]

在慢平台 $m_k=1$ 上，目标长度 $H=SL_M$ 的转圈数满足恒等式：

\[
t^H_{M,k}
=\frac{H\nu_{M,k}}{2\pi}
=\frac{SL_M\omega^N_{M,k}S^{-1}}{2\pi}
=t^N_{M,k}.
\]

因此，如果慢端的设计意图是把“Native窗内不足一圈”的槽全部放入完整压缩平台，
使这些槽在目标窗的相位暴露不超过Native，那么transition结束点的几何锚点与
$S$ 无关：

\[
h_{1\rm turn}(M)
=\frac{K_M}{\log b_M}\log\frac{L_M}{2\pi}.
\]

整数表取最近合法槽或包围它的两个槽。三个配置的连续锚点为：

| 模型 | $L$ | $b$ | $h_{1\rm turn}$ | 对应整数邻域 |
|---|---:|---:|---:|---:|
| OLMo-2-1B | 4096 | 500000 | 31.604 | 31--32 |
| Llama-3-8B | 8192 | 500000 | 34.984 | 34--35 |
| Qwen2.5-1.5B | 32768 | 1000000 | 39.651 | 39--40 |

这与当前独立 high 搜索的现象相容：OLMo 的 high=31/32未分离，Qwen S2 的最佳
开发点 high=39，Llama S4 的 high=34也在一圈邻域。但它不是定理：

- 单个正弦槽少于一圈仍可提供局部位置信息；多槽联合码也不由一个槽决定；
- “一圈”不保证正确绑定、无 collision 或完整生成；
- Llama S2 的当前点 high=32对应1.844圈，说明固定宽度搜索、shape和剂量可能把
  high 从几何锚点拉开；Llama S8的匹配C42比较也尚未独立扫描high。

所以第一条假设是一个**覆盖包络**：

> **H-slow**：在其它配置固定时，可部署的 high 应落在
> $t^N_h=O(1)$ 的邻域。若high向更慢频率越过到 $t^N_h\ll1$，部分Native内
> 不足一圈的槽仍只得到部分压缩，目标相位暴露可能超出其训练范围；若high过早，
> 又会把更多已完成多圈的槽完整压缩并增加Native代价。本假设不保证
> $t^N_h=1$ 取得最高分。

## 3. 快端：用 Native-window phase-distortion 预算决定从哪里开始动

band 的 low 槽本身满足 $m_l=0$，实际相位扰动为零。为了给 low 一个不依赖
transition 首个小增量的坐标，定义“若该槽被完整压缩时”的反事实 Native 相位预算：

\[
\bar D_N(k;S)
=L_M\omega^N_{M,k}\left(1-\frac1S\right)
=2\pi t^N_{M,k}\left(1-\frac1S\right).
\]

它不是 low 槽的实际扰动，而是一个保守判据：高频槽若连完整压缩的相位差都很大，
就先留在 $m=0$ 平台；当 $\bar D_N$ 下降到 checkpoint 可承受预算 $B_{f,M}$ 后，
才开始 transition。

若 $B_{f,M}$ 在同一 checkpoint、同一任务族上近似固定，则：

\[
l_M(S)
=\frac{K_M}{\log b_M}
 \left[\log L_M+\log\left(1-\frac1S\right)-\log B_{f,M}\right],
\]

从 $S_1$ 到 $S_2$ 的预测位移为：

\[
\Delta l
=\frac{K_M}{\log b_M}
 \log\frac{1-1/S_2}{1-1/S_1}.
\]

Llama 的 $K=64,b=500000$ 给出：

| 倍率变化 | 固定槽位 / Native winding | 固定 $\bar D_N$ | 固定目标端相位差 $D_H=S\bar D_N$ |
|---|---:|---:|---:|
| S2→S4 | 0 | **+1.98槽** | +5.36槽 |
| S2→S8 | 0 | **+2.73槽** | +9.49槽 |

当前 Llama C42V24 小屏由 S2 `[14,32]` 移到 S4 `[16,34]`，low/high 都平移2槽。
两个 fast 坐标的 $\bar D_N(l)$ 分别为232.12/231.05 rad；slow坐标也为
5.79/5.77 rad。这一数值吻合使 $\bar D_N$ 成为当前最值得证伪的跨S假说，
但它只有一个已观察跃迁。

Llama S8已有匹配C42的 `[14,32]` 与 `[16,34]`：两者64K均81.25%，前者因32K
更高而AUC领先。固定 $\bar D_N$ 真正预测的是 `[17,35]`，尚未测试；因此S8
现有结果既没有证实，也不允许用 `[14,32]` 的开发AUC直接否定该预测。

第二条假设是：

> **H-fast**：固定 shape 和 gain 时，同一 checkpoint 随 S 改变的 low，主要沿
> $\bar D_N(l;S)=B_{f,M}$ 迁移；固定槽位和目标端相位差是预先列出的反例规则。

## 4. checkpoint 修正：几何给锚点，Q/K谱决定预算

同一层、头和相对距离 $d$ 的 rotary attention 项可写为：

\[
a(d)=\sum_k \operatorname{Re}\left(C_ke^{id\nu_k}\right).
\]

其中 $C_k$ 由 pre-RoPE Q/K 内容系数决定。局部导数为：

\[
\frac{\partial a}{\partial m_k}
=\log S\,d\nu_k\operatorname{Im}\left(C_ke^{id\nu_k}\right).
\]

所以相同几何相位改动在不同 checkpoint、层、任务和距离上可以有不同符号。只看

\(|C_k|\) 或投影权重范数能估计“承载量”，不能给出任务方向。

定义一个在目标 band 分数开封前计算的任务条件化谱敏感度：

\[
F_{M,k}
=\mathbb E_{\ell,h,q,x,d}
 \left[\left(\frac{\partial\mathcal L}{\partial\phi_{\ell,h,k}(d)}\right)^2\right].
\]

在小相位近似下，Native代价可写成：

\[
\mathcal C_N(m)
\approx\sum_k F_{M,k}\,\mathbb E_{d\le L}
 \left[d^2(\omega^N_{M,k}-\nu_{M,k})^2\right].
\]

若边界附近的 $F_{M,k}$ 变化较慢，单槽预算

\(\sqrt{F_{M,l}}\bar D_N(l;S)\le\epsilon_N\) 给出一阶 checkpoint 修正：

\[
\delta l_M
\approx\frac{K_M}{2\log b_M}
 \log\frac{F_{M,l}}{F_{\rm ref,l}}.
\]

高敏感槽会把 low 推向更慢频率；低敏感槽允许更早开始压缩。实际预测应使用完整
chordal代价而不是无限小近似：

\[
\mathcal C_N^{\rm chord}(m)
=\sum_kF_{M,k}\,\mathbb E_{d\le L}
 \left[4\sin^2\frac{d(\omega^N_{M,k}-\nu_{M,k})}{2}\right].
\]

这里的 $F$ 必须来自冻结 checkpoint 的 Q/K 权重加激活，或在固定校准集上的
phase-gradient；不能用目标band的最终分数回归得到。权重范数积只能作为廉价null：
现有CPU测量中OLMo/Llama的per-slot engagement相关高达0.970，但激活相位核对比度
相关约−0.025。这正说明“有多少承载”跨模型相似，而“哪些槽携带位置差异”高度
checkpoint相关。

第三条假设是：

> **H-QK**：几何锚点的跨模型残差可由开封前的Q/K phase-sensitivity谱方向性预测；
> 若只能在看过任务分数后拟合每模型offset，它就不是统一理论，只是经验调参。

## 5. 当前候选的坐标数值

下表的 $\bar D_N$ 都是假设“该边界槽完整压缩”得到的坐标；实际low处
$m_l=0$，不要把表中数值误写成真实low相位改动。$D_H=S\bar D_N$。

| 模型 / 配置 | band | $t_l^N$ | $t_h^N$ | $\lambda_l/L$ | $\lambda_h/L$ | $\bar D_N(l)$ | $\bar D_N(h)$ | $D_H(l)/D_H(h)$ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OLMo S4 当前点 | `[14,31]` | 36.943 | 1.132 | .0271 | .8836 | 174.09 | 5.33 | 696.36 / 21.33 |
| OLMo S4 当前点 | `[14,32]` | 36.943 | .922 | .0271 | 1.0847 | 174.09 | 4.34 | 696.36 / 17.38 |
| OLMo S4 Llama直迁 | `[16,34]` | 24.515 | .612 | .0408 | 1.6345 | 115.53 | 2.88 | 462.10 / 11.53 |
| Llama S2 开发点 | `[14,32]` | 73.886 | 1.844 | .0135 | .5423 | 232.12 | 5.79 | 464.24 / 11.59 |
| Llama S4 开发点 | `[16,34]` | 49.031 | 1.224 | .0204 | .8173 | 231.05 | 5.77 | 924.21 / 23.06 |
| Llama S8 $\bar D_N$预测 | `[17,35]` | 39.941 | .997 | .0250 | 1.0033 | 219.59 | 5.48 | 1756.71 / 43.84 |
| Llama S8 已测C42 | `[14,32]` | 73.886 | 1.844 | .0135 | .5423 | 406.21 | 10.14 | 3249.66 / 81.10 |
| Llama S8 已测C42 | `[16,34]` | 49.031 | 1.224 | .0204 | .8173 | 269.56 | 6.73 | 2156.48 / 53.83 |
| Qwen S2 literal | `[16,34]` | 164.919 | 3.387 | .0061 | .2953 | 518.11 | 10.64 | 1036.22 / 21.28 |
| Qwen S2 $\bar D_N$候选 | `[20,38]` | 69.546 | 1.428 | .0144 | .7002 | 218.48 | 4.49 | 436.97 / 8.97 |
| Qwen S2 当前最佳点 | `[22,39]` | 45.162 | 1.151 | .0221 | .8689 | 141.88 | 3.62 | 283.76 / 7.23 |
| Qwen S2 canonical | `[23,40]` | 36.393 | .927 | .0275 | 1.0783 | 114.33 | 2.91 | 228.67 / 5.83 |

当前数据呈现两件事：

1. OLMo、Llama S4和Qwen当前点的high都接近一圈邻域，支持继续检验H-slow；
2. fast预算明显是checkpoint相关的：OLMo约174、Llama S2/S4约231、Qwen当前
   点约142 rad。一个跨模型常数不能解释全部，需要H-QK或承认模型条件化。

由两边界独立而非固定宽度出发，第一批最小新候选应是：

- Llama S2 `[14,35]`、S4 `[16,35]`、S8 `[17,35]`；
- Qwen S2 low取20--22的预注册预测，high独立固定39--40；
- OLMo S4继续把14固定，只区分31/32，不再把low/high绑成同宽平移。

这些是理论探针，不是替换当前实测winner的命名。

## 6. 已有反例限制了什么

### 6.1 单一慢端或端点相位不充分

[Winding-Matched作者方案](WINDING_MATCHED_ROPE_AUTHOR_PROPOSAL_20260913.md)在指定
单lag上实现端点同余，但Llama 8K为89.58%、16--64K均为0。它直接反驳“一个端点
相位等式足以保证真实长任务”。H-slow只能作为必要性启发，不能单独排序表。

### 6.2 几何相同不等于checkpoint相同

OLMo与Llama同为 $b=500000,K=64$，Llama S4的 `[16,34]` 原样迁到OLMo后，
16K NIAH相对OLMo `[14,32]`下降25pp，tail NLL增加0.4119 nat。绝对槽位和纯
winding都不充分，支持加入H-QK。

### 6.3 band响应非单调

Llama S2从 `[14,32]` 移到 `[16,34]` 时16K NIAH由100%降到87.5%，再移到
`[18,36]`又回到97.92%；PPL几乎不变。S4的 `[18,36]` 则在32K降到18.75%，
tail128 NLL显著恶化。任何只依赖单槽标量并假定性能随band平滑的推导都已被削弱。

### 6.4 shape与band强交互

Llama S8的Solver shape中 `[14,32]` 比 `[16,34]` 的64K高25pp；匹配C42 shape
后两者64K同为81.25%，差异主要转到32K。此前25pp不能归因于band主效应。

### 6.5 权重幅值不是任务方向

CPU engagement跨OLMo/Llama相关0.970，但激活相位核对比度相关约0；旧Fisher和
多个逐槽几何标量也不能排序任务分数。H-QK必须以留一预测接受检验，不能凭一张
漂亮的谱图宣布机制成立。

## 7. 每一环怎样被证实或否定

### E-slow：独立 high 扫描

固定模型、S、low、shape、gain和输入，只测

\(h_{1\rm turn}-1,h_{1\rm turn},h_{1\rm turn}+1\)。已有点直接复用。

- 支持：三个模型的endpoint/AUC非支配basin都落在一圈邻域，越过到明显
  $t_h\ll1$ 后稳定恶化；
- 否定：至少两个模型在匹配控制下稳定偏好远离一圈的high，且不是Native代价、
  shape或gain解释；
- 允许结论：只写“one-turn neighborhood is a useful slow-boundary prior”，不写必要充分条件。

### E-fast-S：同一模型 S2/S4/S8

先用Llama，因为S2/S4已有同shape结果。新增S8 `[17,35]`，并补独立high后的
`[14,35]`、`[16,35]`可复用为对照；先8K/64K low层，再只给非支配两臂补全长度。

- 支持：同一checkpoint各S的最佳low使 $\bar D_N(l;S)$ 近似稳定，且明显优于
  固定槽位和固定 $D_H$ 预测；
- 否定：预测low在endpoint与AUC上被固定槽位明确支配，或最佳 $\bar D_N$ 随S系统漂移；
- 允许结论：成功也只能写checkpoint内的scale law。

### E-QK：跨模型留一预测

在不读取留出模型band分数的前提下，用固定的Native文本与NIAH校准行提取：

1. 权重范数 engagement null；
2. 激活 $|C_k|^2$；
3. phase-gradient $F_{M,k}$。

先在两个模型上冻结“几何锚点+谱修正”的low预测，再直接给第三模型一张表；目标
模型邻点只在预测解封后用于计算prediction regret，不回写公式。轮换三次。

- 支持：phase-gradient修正在三折中均把预测送入非支配basin，并系统优于无修正
  geometry及weight-norm null；
- 否定：修正方向在留出模型上反复相反，或只在看过目标结果后才成立；
- 允许结论：成功可称“checkpoint-conditioned predictive initialization”，不能称闭式最优解。

### E-factorial：shape、band、gain与剂量

在一个有清晰band差异的S8点做最小控制：

- C42V24/MrPro shape × 两个band，同一标准gain；
- winner表用标准gain与区间均值gain；
- winner相对anchor移动时，加一张保持共同 $\sum m$ 的最小tilt companion。

如果band排序在shape或gain下反转，则两边界理论只能作为完整profile的条件化先验；
如果matched-dose后收益消失，原观察主要来自压缩剂量，不能称placement机制。

## 8. 论文可写范围

若H-slow、H-fast、H-QK及factorial均通过，可写：

> A fixed-table transition admits two empirically distinct design pressures: a
> slow-boundary coverage prior near one native-window turn, and a fast-boundary
> Native-phase-distortion budget corrected by checkpoint Q/K spectral sensitivity.

仍不能写唯一band或全模型保证。

若只有H-slow通过，可写慢端存在跨模型几何prior，快端必须checkpoint条件化。
若H-fast只在Llama跨S通过，可写单checkpoint尺度律。若H-QK失败，保留失败本身：
在本项目测试的slot、winding、phase-distortion与谱修正中，没有一个几何—权重标量
稳定预测全部模型×倍率；最终方法应直接优化全64槽任务区间目标，而非继续发明
固定边界公式。不能由有限反例升级为“不存在任何通用理论”。

## 9. 证据和实现入口

- 当前band结果：[BAND_POSITION_MINIMAL_SCREEN_20260913.md](BAND_POSITION_MINIMAL_SCREEN_20260913.md)
- 跨模型/跨S开放格：[BAND_NEXT_STAGE_RESULT_20260913.md](BAND_NEXT_STAGE_RESULT_20260913.md)
- 固定表区间结果：[4080_FIXED_TABLE_RANGE_RESULT_20260913.md](4080_FIXED_TABLE_RANGE_RESULT_20260913.md)
- Winding-Matched反例：[WINDING_MATCHED_ROPE_AUTHOR_PROPOSAL_20260913.md](WINDING_MATCHED_ROPE_AUTHOR_PROPOSAL_20260913.md)
- C42V24构造：`experiments/rope_fast_5090_20260912/e3_tables.py`
- band remap：`experiments/olmo_recovery_20260912/transfer_range_profile.py`

本页没有启动GPU、修改主稿或把计划实验登记为已完成结果。
