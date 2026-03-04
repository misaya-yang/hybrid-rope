## DSR v2：Distance-Swept Retrieval（机制对齐版）

### A. 统计预算：(N=50) 足够（并在文中写清楚）

* 每个距离点做 (N=50) 次独立 trial（不同随机 needle/value/distractor/上下文填充）。
* 二元指标（对/错）下，95% 置信区间半径近似：
  [
  \text{CI}_{95}\approx 1.96\sqrt{\frac{p(1-p)}{N}}
  ]

  * (p=0.5): (\approx 1.96\sqrt{0.25/50}=1.96\cdot 0.0707\approx 0.139)（±14%）
  * (p=0.9): (\approx 1.96\sqrt{0.09/50}=1.96\cdot 0.0424\approx 0.083)（±8%）
* 结论：(N=50) 已足以区分你要的“曲线整体右移/压制”效应；(N=100) 成本翻倍但 CI 仅缩小 (\sqrt{2})。

**论文写法建议**：在 Appendix 里用一句话交代“we set (N=50) since binomial CI already separates methods; doubling (N) yields diminishing returns.”

---

### B. 距离采样：非均匀（1×–3× transition 加密）

固定采样点（你给的很合理）：
[
\Delta/L_{\text{train}} \in {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0}
]

* 重点解释：你关心的是 **extrapolation transition band**（从“训练内稳”到“外推崩/反转”），这个区间通常在 1×–3×，所以加密最有效。

---

### C. 控变量：固定 (L_{\text{eval}})，只变 needle 位置（避免 length confounds）

你说的点非常关键：**不要让 (L_{\text{eval}}) 跟着变**，否则混入：

* 注意力 sink / 长上下文归一化变化
* KV cache 规模、attention 分布的非线性变化
* 模型对“上下文本身变长”的敏感性

**推荐设置：**

* 固定 (L_{\text{eval}} = 8L_{\text{train}})（例如 (L_{\text{train}}=2048\Rightarrow L_{\text{eval}}=16384)）
* Query 永远在末尾
* needle 放在位置 (p = L_{\text{eval}}-\Delta)
* 因而 (\Delta=L_{\text{eval}}-p) 就是纯距离变量

这会让 reviewer 读到一句话就放心：**“我们测的是 distance effect，不是 length effect。”**

---

### D. 必须画四条曲线：把 superlinear synergy 放进同一张图

四个配置固定为：

1. Geo
2. EVQ
3. Geo + YaRN
4. EVQ + YaRN

**图上应出现的“可被检验的排序”**（你可以在 caption 里写成 hypothesis）：

* 训练外（(\Delta>L_{\text{train}})）应满足：
  [
  \text{EVQ+YaRN} \succ \text{Geo+YaRN} \succ \text{EVQ} \succ \text{Geo}
  ]
* 更强的叙事是“右移 + 抬高”：

  * EVQ：把 transition 区右移（有效检索半径变大）
  * YaRN：把曲线整体抬高（推理时缩放对齐）
  * EVQ+YaRN：两者叠加呈现**超线性**（在 1×–4× 区间压制其余三条）

---

## 实验规模核算（按你给的预算）

* 9 个距离点 × 50 trials × 4 条曲线 = **1800 次 inference**
* 只要你的单次 forward 在 16K 上不是离谱（bf16 + SDPA + batch 适当），这在 5090 / R6000 上都属于“可忽略成本”。

---

## 输出指标：除了 Accuracy(Δ)，再加 2 个“论文更像样”的 summary

建议同时报告：

1. **AUC over normalized distance**（例如对 (\Delta/L_{\text{train}}\in[1,8]) 做数值积分）
2. **Break point**：(\Delta_{50})（Accuracy 首次低于 50% 的距离；也可以用 70%/80% 更贴近你要的“near-perfect band”）

这两个 summary 能把整条曲线压成一行表格，方便主文塞 Table。

---

## 任务模板（避免“语义猜测”，纯检索）

为避免模型靠语义猜：

* key/value 使用随机串（base32 或 hex），并加校验位（防止局部匹配）
* query 明确要求 exact copy：

  * “Return the exact VALUE for KEY=… Only output the VALUE.”
* 评分用 exact match（或编辑距离=0）

思考构造 distractors 让错误变成系统性（从而更敏感）”**：比如在每个 trial 里放一个“near-key” + 一个“decoy-value”，这样 Geo 在碰撞区会出现你已经观察到的 **<50% 反转**，曲线会更有说服力。
