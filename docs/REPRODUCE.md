# 复现路径指南 (Reproduction Guide)

> 最后更新：2026-02-22
> 依照开放科学准则，本文档提供三种最具代表性的最短验证路径。

## 环境前置依赖
硬件：单卡 A100 80GB (推荐) 取保序列内存支持。
`pip install torch transformers datasets numpy`

---

## 路径一：微型尺度强健性证据 (50M 级别多随机种子)
本步骤将复现我们在 50M 从零训练过程中的稳固统计结果，证明 Hybrid 结构的稳定 PPL 收缩增益。

**执行命令:**
```bash
python a100/unified_search_3cfg_3seed.py
```
**原理与预期:**
该脚本内嵌了 42, 123, 7 三个随机种子集合，利用串流形式截断 TinyStories 做前向训练与滑动 PPL 评估，产出完整的均值和方差统计。

---

## 路径二：核心图谱重现 (机理崩溃验证与注意力分布)
本步骤针对理论分析，可视化由于 Theta 暴增导致的距离 $D(\Delta)$ 失效，以及对应的注意力崩塌。

**执行命令 (绘制崩溃相位的机理论证图):**
```bash
python scripts/plot_yarn_compare.py
```
*(注：仓库内有多个相关的绘图脚本配合，您也可直接观察 `knowledge_base/06_phase_collision_D_analysis.md` 引用的数据探针，运行探针命令如下：`python scripts/debug_sigmoid_rope.py`)*

---

## 路径三：大尺度极严公平协议验证 (8B LoRA)
为了保证和 Baseline (如 PI, YaRN) 能够平权测试，我们移除了框架内置的 API 偏差，采用最严密的底层显存替换执行比较：

**执行单测的快捷路径 (验证逻辑链合法性):**
由于完整的 8B 继续预训练时间开销巨大，我们提供专门探针检测修改的 RoPE `inv_freq` 确实贯穿前向过程发挥功效：
```bash
python 2026-02-22/scripts/_test_inject.py
```

**执行完整 4 方法全景评测管线 (包含 PPL 与 NIAH):**
```bash
python 2026-02-22/scripts/run_overnight_8h.py
```
*(由于资源要求巨大，可能耗时8小时+，该脚本中包含严格的变量检查锁定)*
