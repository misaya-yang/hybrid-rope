# Winding-Matched RoPE：作者提出的无人工频带新思路

提出日期：2026-09-13。**本方案由项目作者（本会话用户）提出。** Codex负责理论
审计、实现、实验和报告，不将其改写为代理自行提出的方法，也不在GPU结果之前把
理论性质写成任务有效性结论。

## 核心构造

对Native频率\(\omega_i\)、Native长度\(L\)和目标倍率\(S\)，令

\[
r_i=\frac{L\omega_i}{2\pi},\qquad
n_i^\star=\left\lfloor(S-1)r_i\right\rfloor,
\]

并构造

\[
\widetilde\omega_i=
\frac{\omega_i}{S}+\frac{2\pi n_i^\star}{SL}.
\]

它不输入YaRN的`l/h`或连续profile；每个槽在“端点旋转与Native端点同余、频率不
快于Native且不慢于full interpolation”的集合中，选择最大的合法频率。第一轮
gain固定为同倍率BM/MrPro的标准gain，使唯一新变量是频率表。

## 精确算术下的待审定性质

- \(SL\widetilde\omega_i=L\omega_i+2\pi n_i^\star\)，所以目标最大相对距离的
  每个2D旋转块、以及拼接后的block-diagonal RoPE算子与Native最大相对距离同余。
- 在\(\omega_i/S\le\nu\le\omega_i\)且满足该端点同余的逐槽候选中，它是唯一
  最大频率，因而最小化未取模的减速量。
- \(0\le\omega_i-\widetilde\omega_i<2\pi/(SL)\)；对本项目\(S\ge2\)和
  \(0\le\Delta\le L\)，单块旋转差可由\(2\sin(\pi/S)\)界定。
- 频率严格顺序需要用正确的缩放坐标证明并在实际Native FP32表上复验；频率有序
  不自动意味着exponent序列\(m_i=-\log_S(\widetilde\omega_i/\omega_i)\)单调。

这些是构造性质，不证明模型会在所有中间长度使用该端点同余。整数floor可能产生
逐槽锯齿；实际runtime还会把表和phase降到FP32，使实数域的精确同余变成有限精度
近似。两者都必须先测，不能被闭式公式掩盖。

## 判决实验

| 阶段 | 目的 | 成功有什么用 | 失败有什么用 | 距最终解法 |
|---|---|---|---|---|
| WM0 CPU审计 | 在OLMo L=4096/S=4与Llama L=8192/S=8的真实Native表上检查频率上下界、严格顺序、整数绕圈、FP64/模拟FP32端点残差、m的单调违例和最小间距 | 确认闭式表可按现有runtime部署，并准确限定定理 | 数值或顺序失败直接修公式/实现；m非单调只记录机制，不自动否决 | 构造正确性，约20% |
| WM1 Llama五长度 | 复用当前8/16/32/48/64K三任务面板与BM/MrPro/full-profile，只新增WM一臂 | 若Native、内部点和64K共同强，得到第一个无人工band候选 | 端点强短窗弱说明端点同余不足；端点也弱则否定本闭式判据的任务充分性 | 开发候选，约40% |
| WM2 64K八任务 | 只有WM1有独特收益时，复用runner-matched BM/MrPro 32行，只跑WM | 检查周期构造是否跨VT/FWE/QA/多证据 | 负结果限定为任务专用或机制反例，不反推所有无band方法失败 | 任务广度，约55% |
| WM3 全区间确认 | 与最多两个非支配候选进入八任务×全长度新seed | 支持声明范围内的可部署方法 | 反转则保留端点/任务条件，转回full-spectrum robust solver | 最终证据的一部分 |

WM只新增一个闭式表，不在失败后追加平滑项、floor温度或参数扫描。失败只否定当前
“最大合法绕圈”构造的任务充分性，不把无固定`l/h`的全谱优化空间一并删除。

## WM0 CPU回执

真实Native FP32频率表上的构造已经完成：

| 模型 / 配置 | FP64端点圆周残差 | 模拟runtime FP32残差 | m非单调下降数 | sum(m) |
|---|---:|---:|---:|---:|
| OLMo L=4096, S=4 | 7.21e-13 | 3.22e-4 | 15 | 28.1719 |
| Llama L=8192, S=8 | 3.30e-12 | 1.95e-3 | 20 | 19.9181 |

两张频率表都在`[omega/S, omega]`内且严格递减，逐槽整数绕圈为最大合法值。
因此实数域端点同余成立；附件建议的实际FP32 `<1e-6`门槛不成立，改为如实报告
有限精度残差。频率有序但m明显非单调，确认WM属于整数winding阶梯，而不是
YaRN/MrPro式单调exponent allocation。Llama WM的总位移远小于BM约37和当前
full-profile约41.95；这既可能改善Native保留，也可能导致中间长度或任务能力不足，
必须由WM1真实生成裁决。
