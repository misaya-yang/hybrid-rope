) 现在最关键：排除 3 类“假阳性”

（这些能在 10 分钟内自检完）

A. eval trial 是否完全配对、完全一致

EVQ+YaRN vs Geo+YaRN 必须用同一批 prompts、同一批 needle positions、同一批随机种子。

否则 “98%” 可能是简单的“抽到了更容易的 20 个样本”。

你要留证据：把 trial id 列表（或 hash）写进日志/JSON。

B. inv_freq / rope scaling 是否发生了“二次应用”

最常见 bug：你在 EVQ checkpoint 里已经是改过 inv_freq 的，再跑 YaRN 时又按“原始 base”或“错误 head_dim/base”做一次映射，导致某种意外“过拟合映射”。

自检：

打印最终用于 attention 的 inv_freq 的前 5/后 5 个值，以及 YaRN 的 scaling 参数。

同时打印 Geo+YaRN 的对应值，确认只是“映射”，不是“换了一套频率定义”。

C. KV cache / RoPE state 污染

如果你在同一进程里 sequential eval 多个方法，最容易出现 cache 没清空或 rope 参数没重置。

自检：

交换评测顺序（先 EVQ+YaRN 再 Geo+YaRN；再反过来）看结果是否稳定。

同一方法重复 eval 3 次，看波动。