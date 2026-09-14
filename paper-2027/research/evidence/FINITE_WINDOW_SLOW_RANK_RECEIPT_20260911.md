# 有限窗慢频塌缩 Receipt（b=256 核心实验：闭合 min ωW=1.1892 缺口）

日期：2026-09-11。状态：**verified**（双实现一致 + 论文锚点复现）。
性质：纯几何精确计算（无 GPU、无模型），定义完全采用论文 `03_theory.tex` §Full-pair geometry
的块白化 Gram 口径（eq:budget-identity 精确恒等式 r₂(Γ)=(trΓ)²/trΓ²）。

## 解决的问题

交接档 §3.2 警告（原登记：`external-reviews/pro-materials-20260908/RoPE_ICLR2027_Major_Revision_20260906.md` §5.2）：
核心实验 min ωW = 256·256^(−31/32) = **1.1892 > 1**，无任何通道满足 ωW≤1，
故 Proposition (shared slow-frequency subspace, ωL→0) 的渐近**不覆盖**核心实验，
机制叙事此前缺少有限窗版本。

本 receipt 给出有限窗精确结果：**塌缩不依赖 ωL→0 渐近**。在 b=256, K=32, L=256
（paired-training 网格）下，最慢 8 对（ωW∈[1.19,4]）的块白化 Rényi-2 有效秩仅 **2.1147**
（渐近情形 2.00013）；塌缩随纳入更快频率**渐进**退化，不是突然失效。

## 定义与口径

- x_ω(Δ)=[cos(ωΔ), sin(ωΔ)]，Δ ~ Uniform[0,L]（declared separation measure，与论文
  b=500K 口径一致）；归一化后即 Δ~Unif[0,1]、频率以 ωW 计。
- 块白化：Q_ων = S_ω^(−1/2) H_ων S_ν^(−1/2)，对角块置 I₂，堆叠成 Γ。
- r₂(Γ) = (trΓ)²/trΓ²（Rényi-2 / participation ratio）。
- 解析式：E[cos kΔ]=sin(k)/k，E[sin kΔ]=(1−cos k)/k，全部矩阵元素闭式计算；
  另以 2×10⁵ 点数值积分独立复核（n=8 两路线一致到 5 位小数）。

## 数字

**锚点自检**（b=5×10⁵, K=64, L=4096，ωL≤1 的 23 对）：
r₂ = **2.00013**，与论文正文数值逐位一致（实现正确性检验）。

**核心网格**（b=256, K=32, L=256，最慢 n 对，ωW 升序
1.189, 1.414, 1.682, 2.000, 2.378, 2.828, 3.364, 4.000, 4.757, 5.657, 6.727, 8.000, …）：

| 纳入最慢 n 对 | ωW 范围 | r₂(Γ) |
|---:|---|---:|
| 2 | ≤1.41 | 2.00030 |
| 4 | ≤2.00 | 2.00361 |
| 6 | ≤2.83 | 2.02263 |
| 7 | ≤3.36 | 2.05253 |
| **8** | **≤4.00** | **2.11474** |
| 10 | ≤5.66 | 2.38075 |
| 12 | ≤8.00 | 2.89256 |
| 16 | ≤16.00 | 4.43339 |

**谱证据**（n=8）：前两大特征值 7.9681 + 7.5752 = 15.54 / 16.0（97.1% 质量），
第三方向仅 0.4246——即 16 个名义维度的位置方向几乎全部压在 2 维上，
弱正则方向随 ωW 增大缓慢增重（这就是 r₂ 退化的机制）。

## Provenance

- 首次计算：GPT 第一轮推导（五开放问题 Q5，2026-09-11，sandbox 脚本）。
- 独立复核：本仓库同日用上表解析实现复算，n=4/6/7/8 四个值与 GPT 逐位一致
  （2.00361/2.02263/2.05253/2.11474），且 500K 锚点复现论文值 2.00013。
- 早期手算质疑（"ωW=4 的 sin 分量不应塌缩"）已被谱证据否定：
  极端频率对的第二 canonical 方向虽弱（~0.40），但对秩质量的贡献可忽略。

## 论文与交接档联动

- `paper-2027/sections/03_theory.tex`（shared slow-frequency limit 段，标准网格句之后）
  已插入有限窗延续句：核心网格 8 对 r₂=2.11（4 对 2.004；≤ωL=8 时 2.89），
  定性为 finite-window subspace overlap，ωL→0 极限为其极端情形。
- 交接档 `PAPER_REVISION_HANDOFF_20260911.md` §3.2 的 ⚠ 警告已改为 ✅ 闭合并指向本文件。
- 改稿措辞纪律：引用本结论一律用 "finite-window subspace overlap"，
  不得退回 "塌缩需要 ωL≪1"，也不得把 ωL→0 渐近当作核心实验的解释。

## 边界（本 receipt 不声称什么）

1. 只陈述**最慢频率对之间**的相互冗余结构；30 个内部指数的整体几何（含中高频段）
   不在此定理范围内。
2. 静态秩不保证下游外推（论文原句保留）；本结果的作用是使机制解释覆盖核心实验的
   参数区间，不是新的因果主张。
3. measure 固定为 Uniform[0,L]；换 measure 需重算（脚本直接支持）。

## 复算脚本（自包含；运行即自检锚点）

```python
import numpy as np
def Ecos(k): return 1.0 if abs(k)<1e-12 else np.sin(k)/k
def Esin(k): return (1-np.cos(k))/k if abs(k)>1e-12 else 0.0
def S_block(w):
    return np.array([[0.5+0.5*Ecos(2*w), 0.5*Esin(2*w)],
                     [0.5*Esin(2*w),     0.5-0.5*Ecos(2*w)]])
def H_block(w,v):
    return np.array([[0.5*(Ecos(w-v)+Ecos(w+v)), 0.5*(Esin(w+v)-Esin(w-v))],
                     [0.5*(Esin(w-v)+Esin(w+v)), 0.5*(Ecos(w-v)-Ecos(w+v))]])
def inv_sqrt(A):
    ev,V=np.linalg.eigh(A); return [REDACTED_EMAIL](1/np.sqrt(ev))@V.T
def r2_of(ws):
    n=len(ws); G=np.zeros((2*n,2*n))
    for i,w in enumerate(ws):
        G[2*i:2*i+2,2*i:2*i+2]=np.eye(2)
        for j,v in enumerate(ws):
            if j!=i:
                G[2*i:2*i+2,2*j:2*j+2]=inv_sqrt(S_block(w))@H_block(w,v)@inv_sqrt(S_block(v))
    ev=np.linalg.eigvalsh(G)
    return (ev.sum()**2)/((ev**2).sum())

wa=[4096*np.exp(-np.log(5e5)*j/64) for j in range(41,64)]      # anchor, 23 pairs
assert abs(r2_of(wa)-2.00013)<1e-4
ws=[256*np.exp(-np.log(256)*j/32) for j in range(31,15,-1)]    # core grid, slowest 16
for n in [4,6,7,8,12,16]:
    print(n, '%.5f'%r2_of(ws[:n]))
# 期望输出: 4 2.00361 / 6 2.02263 / 7 2.05253 / 8 2.11474 / 12 2.89256 / 16 4.43339
```
