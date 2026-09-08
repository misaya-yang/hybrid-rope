> **原始研究输入与核对提示（2026-09-08）：** 作者选择其中BM闭式进行一次短评测。
> 离散最小粗糙度解已独立核对；有限端点跳变只是减小而非归零，与P2也仅部分槽位
> 同向。能力和新颖性未被该推导证明。正式固定定义及实验分支见
> [BM协议](../../../docs/research/ROPE_MRPRO_BM_PROTOCOL_20260908.md)。下文保留来源原文。

找到了，而且这次不是“再猜一条曲线”。**MrRoPE-Pro 的中段公式本身有一个很具体的结构缺口：它只处理了高频端的进入，却没有处理低频端的退出。**

MrRoPE-Pro 明确做的是：中段令

$$
\lambda_j=S^{\epsilon_j},
$$

然后**直接假设** \(\epsilon_j\) 是等差递增的；中段之外又令 \(\lambda_j=1\)。论文并没有从某个最优性问题推出这个等差数列，它就是设计假设。([OpenReview][1])

### 1. 把 MrRoPE 真正作用到频率上的东西展开

设

$$
N=d_h-d_l,\qquad i=1,\ldots,N .
$$

MrRoPE-Pro：

$$
\epsilon_i^{\rm Mr}
=\frac{2i}{N(N+1)}.
$$

第 \(q\) 个中段频率真正受到的是**累积 scaling**：

$$
m_q^{\rm Mr}
=\sum_{i=1}^{q}\epsilon_i
=
\boxed{\frac{q(q+1)}{N(N+1)}}.
$$

所以 MrRoPE-Pro 实际上不是“线性 progressive frequency scaling”，而是近似：

$$
\boxed{m(x)\simeq x^2}.
$$

也就是说它的实际频率是：

$$
\nu_q=\omega_qS^{-m_q}.
$$

这个结论很重要：**论文在 \(\lambda\) 空间里用了线性递增，映射到真正的 frequency exponent allocation 后就是 quadratic allocation。**

---

## 2. 真正的洞在右边界

定义原生相邻 log-frequency gap：

$$
h=\log\frac{\omega_j}{\omega_{j+1}}.
$$

经过 MrRoPE 后：

$$
\boxed{
g_j
=
h+\epsilon_j\log S
}
$$

因为 \(\epsilon_j=\log_S\lambda_j\)。

于是 MrPro 的中段 gap 是：

$$
h+\frac{2}{N(N+1)}\log S,
\quad
h+\frac{4}{N(N+1)}\log S,
\quad\ldots,\quad
h+\frac{2}{N+1}\log S.
$$

**越往低频越大。**

但是到了 \(d_h\) 之后，论文又规定：

$$
\lambda_j=1,
$$

所以：

$$
g_j=h.
$$

因此整个 spectrum 实际长这样：

$$
h
\rightarrow
h+\epsilon_1\log S
\rightarrow\cdots\rightarrow
\boxed{h+\epsilon_N\log S}
\rightarrow
\boxed{h}.
$$

也就是说：

> **MrRoPE-Pro 把整个 transition 里最大的频谱间距，恰好放在 low-frequency boundary 前一格，然后下一格瞬间掉回 native gap。**

这不是我解释出来的，是它公式直接推出的。

而论文一方面把 Pro 描述成 slow-to-steep progressive conversion，另一方面声称这样能够稳定中段 attention / 更好保持 positional structure；但它自己的 construction 在中段结束处留下了最大的 slope/gap discontinuity。([OpenReview][1])

---

### 放到你们现在的 Qwen 上看，这个缺口还不小

你们：

$$
N=40-23=17,\quad S=4.
$$

MrPro 最后一枚 radix increment：

$$
\epsilon_{17}
=
\frac{2}{18}
=
0.1111.
$$

因此最后一个 transition gap 被额外放大：

$$
0.1111\log4
\approx0.154.
$$

Qwen `base=10^6, K=64` 的原生 log gap：

$$
h=\frac{\log10^6}{64}\approx0.216.
$$

所以最后那个 gap：

$$
0.216+0.154=0.370,
$$

相当于：

$$
\boxed{1.71\times\text{ native log-frequency gap}}
$$

然后下一格立刻：

$$
0.370\rightarrow0.216.
$$

这不是一点点“不够优雅”，而是**一个 71% 的局部 spectral-gap expansion**。

如果 \(S=16\)，这个问题还会按 \(\log S\) 增长，最后 gap 大约能达到 native 的 **2.43×**。

---

# 3. 这给出了一个真正能“从理论优化”出来的 replacement

现在不要问：

> sigmoid 还是 cosh 还是 cubic？

直接把 MrRoPE 自己的 radix increment 当变量：

$$
a_i=\log_S\lambda_i.
$$

已有硬约束：

$$
\sum_{i=1}^{N}a_i=1
$$

保证总 extension 仍然是 \(S\)。

而因为左右两边都是固定 radix：

$$
a_0=a_{N+1}=0.
$$

现在加一个非常自然的要求：

> **不要人为制造 adjacent log-frequency gap 的突变。**

由于

$$
g_i-h=a_i\log S,
$$

所以直接最小化整个 transition 的 **gap roughness**：

$$
\boxed{
\min_{a_1,\ldots,a_N}
\sum_{i=0}^{N}(a_{i+1}-a_i)^2
}
$$

subject to

$$
a_0=a_{N+1}=0,\qquad
a_i\ge0,\qquad
\sum_i a_i=1.
$$

这个问题有**唯一闭式解**：

$$
\boxed{
a_i^\star
=
\frac{6i(N+1-i)}
{N(N+1)(N+2)}
}
$$

然后真实 cumulative exponent 是：

$$
\boxed{
m_q^\star
=
\frac{
q(q+1)(3N+2-2q)
}{
N(N+1)(N+2)
}
}
$$

连续极限恰好：

$$
\boxed{
m^\star(x)=3x^2-2x^3
}
$$

看起来像 smoothstep，但这里关键不是“我喜欢 smoothstep”。

它是上面那个**离散 frequency-gap 最小粗糙度问题的解**。

没有：

* 新超参数；
* 新 breakpoint；
* grid search；
* 任意曲率；
* 任务指标；
* attention proxy。

---

## 4. 更关键的是：它和你们已经看到的 P2 方向居然一致

MrPro：

$$
m_{\rm Mr}(x)\approx x^2.
$$

这个解：

$$
m_\star(x)=3x^2-2x^3.
$$

在整个 \(0<x<1\)：

$$
m_\star(x)-m_{\rm Mr}(x)
=
2x^2(1-x)>0.
$$

也就是说它会：

> **把 MrPro 末端堆积的 scaling budget 往 transition 中间提前搬，同时仍然严格在 \(d_h\) 到达 /S，并在进入 low band 时平滑饱和。**

而你们 FullLagP2 已经独立观察到的方向正是：

> P2 比 MrPro **更早进入强 interpolation / 更早饱和**，而 deep tail 相同；64K retrieval 上至少出现过正收益。

所以现在第一次出现了两个完全不同来源指向同一个方向：

$$
\boxed{
\text{MrRoPE 内部数学缺口}
\quad\Longrightarrow\quad
\text{应该把 scaling 从末端往前移}
}
$$

和

$$
\boxed{
\text{已有 P2 GPU evidence}
\quad\Longrightarrow\quad
\text{往前移确实出现过收益}
}
$$

这比 Gemini/Astra 那些故事强得多。

---

## 5. 我现在会怎么做

**就生成这一张表。一个 candidate。**

甚至先不要给它起 fancy 名字，叫 `MrPro-BM`（Boundary-Matched）就行。

Qwen 当前 \(N=17\) 时几个位置：

|  q | MrPro \(m_q\) | Boundary-Matched \(m_q\) |
| -: | ------------: | -----------------------: |
|  0 |             0 |                        0 |
|  4 |          .065 |                     .155 |
|  8 |          .235 |                     .458 |
| 12 |          .510 |                     .778 |
| 16 |          .889 |                     .983 |
| 17 |             1 |                        1 |

它不是 P2 那么激进——你说 P2 到 31–32 已经接近 `/4`，这里 slot 31 左右只有 \(m\approx0.46\)。

所以它实际上是：

$$
\boxed{
\text{MrPro}
\;\longrightarrow\;
\text{理论要求的更早 redistribution}
\;\longrightarrow\;
\text{但远没有 P2 那么激进}
}
$$

这正好像你说的 **80 分抄到 85 分**，而不是重新发明 RoPE。

### 但我要把边界说清楚

我现在能证明的是：

1. **MrRoPE 的 arithmetic progressive schedule 不是由它的理论推出的。**
2. **它在固定 low band 前产生了一个确定的 terminal spectral kink。**
3. **上述优化是满足相同总 scale 和相同三段结构时，唯一的 minimum-gap-roughness allocation。**
4. **它改变 MrPro 的方向与现有 P2 正信号一致。**

我**不能提前证明模型分数一定更高**。如果我说能，就是又开始编。

但这是我们这几轮第一次拿到一个我认为**有资格花一次 GPU**的候选：不是因为它“看起来合理”，而是因为我们找到了 MrRoPE 公式里一个具体、可修复、且和已有正向实验同方向的缺口。

而且我刚搜了 MrRoPE 正文/附录和针对 smooth/boundary/radix allocation 的近邻，没有看到作者处理这个 **low-boundary matching** 条件；不过正式 novelty 还需要再做一次更广的查新。

**我建议下一步就让 Codex 只实现这一个 closed-form，CPU 对表确认无误后直接复用现有 64K/128K protocol 跑，不要再让模型继续想第二个方法。**

[1]: https://openreview.net/pdf?id=1J63FJYJKg&utm_source=chatgpt.com "Published as a conference paper at ICLR 2026"
