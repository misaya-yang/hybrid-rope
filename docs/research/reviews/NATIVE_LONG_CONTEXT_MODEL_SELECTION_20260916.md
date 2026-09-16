# 非Llama、原生长窗口、小RoPE base模型选择

## 后续状态更新

GLM下载回执已确认完成，相关partial RoPE支持正在当前实验分支接入，GLM S4队列已启动。
见[带时间戳执行记录](../../../experiments/iclr2027_three_track_sprint_20260915/SERVER_TASK_LAYERS.md)。
下文“未下载/需接入”描述选型时状态；几何与架构差异仍有效，不将队列启动称为结果完成。


## 结论

**首选 `zai-org/GLM-4-9B-0414`，使用普通Chat版本。** 它是独立GLM家族、约9.4B参数，
官方明确原生训练32K；原始发布配置为RoPE base10000、无预置rope_scaling。
这是当前找到最贴合“约7B、非Llama、原生32K/64K、原始小base”的候选。
9B是对“7B左右”的小幅放宽，原生长度与base条件保持。

本次只研究公开资料、读取小型配置并做CPU几何/显存计算。未下载模型权重、未修改GPU任务，
也未宣称该模型上的TailSpline结果已经产生。

## 1. 官方证据

- [GLM-4官方仓库](https://github.com/zai-org/GLM-4#model-list)：GLM-4-9B-0414列为Chat，
  32K→128K；表下注明确models are natively trained with a32K context，超32K建议YaRN。
- [官方发布配置](https://huggingface.co/zai-org/GLM-4-9B-0414/blob/main/config.json)：
  max_position_embeddings32768，rope_theta10000，partial_rotary_factor0.5，head_dim128，
  40层、32个Q heads、2个KV heads，model_type glm4。
- [官方模型卡](https://huggingface.co/zai-org/GLM-4-9B-0414)：普通Chat模型，区别于GLM-Z1推理版。
- [Transformers GLM4实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm4/modeling_glm4.py)：
  只旋转指定子空间，剩余Q/K坐标原样保留；相邻偶/奇维度组成旋转对。

本次固定读取revision `645b8482494e31b6b752272bf7f7f273ef0f3caf`。
[配置快照](model_selection_20260916/glm4_9b_0414_config.json)和
[CPU几何/显存估计](model_selection_20260916/glm4_geometry_memory.json)可供后续接入复用。

公开证据支持“原生32K及发布base10000”；不把发布配置当作未公开的逐阶段训练日志。

## 2. 候选比较

| 候选 | 原生/发布窗口 | RoPE base | 本轮判断 |
|---|---:|---:|---|
| **GLM-4-9B-0414** | **原生32K** | **10000** | 首选；独立家族，partial RoPE需正确接入 |
| GLM-Z1-9B-0414 | 原生32K | 10000 | 同几何但为推理版本；本轮选普通Chat以匹配既有短答案评测 |
| Mistral-7B-Instruct-v0.3 | 32K | 1000000 | 可补家族，但不满足本轮小base选择目的 |
| Falcon3-7B-Instruct | 32K | 1000042 | TII独立家族，但同样是大base；head_dim256 |
| InternLM2.5-7B-Chat | config32K | 1000000 | 发布配置带dynamic scaling factor2；不作为本轮原始静态表首选 |
| Llama-2-7B-Chat | 4K | 10000 | 作者已排除：同Llama家族且原生窗口过短 |
| DeepSeek-LLM-7B-Chat | 4K | 10000 | 独立家族但原生窗口不满足要求 |

候选配置来源：
[Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/main/config.json)、
[Falcon3](https://huggingface.co/tiiuae/Falcon3-7B-Instruct/blob/main/config.json)、
[InternLM2.5](https://huggingface.co/internlm/internlm2_5-7b-chat/blob/main/config.json)。

`model_type=llama`只是软件实现路径，不能独自判断训练家族。例如Falcon3使用此实现名，
仍是TII模型；本轮排除它的实际理由是base，不是这个字段。

## 3. 与128K/256K评测的对应

- 32K Native →128K：S4，与现有Qwen128K的倍率一致。
- 32K Native →256K：S8，可以利用现有大显存开展更长端点评价。
- 每个倍率分别生成自己的T/P静态表。若研究同一表的窗口内曲线，则另明确该表的目标倍率；
  不混淆“S4表在64K评价”与“S2表在64K评价”。

BF16单序列KV cache按 `2 × layers × KV_heads × head_dim × tokens × 2 bytes` 计算：

| 上下文 | KV cache |
|---|---:|
| 32K | 1.25 GiB |
| 64K | 2.5 GiB |
| 128K | 5 GiB |
| 256K | 10 GiB |

官方safetensors参数量9,400,279,040，纯BF16权重约17.51GiB；256K时权重加KV约27.51GiB。
这为96GB机器留下较大容量空间。运行峰值还包括prefill激活、workspace和分块lm_head；
上述是容量估计，不是已完成的256K稳定性实测。可沿用现有chunked-prefill思路，按实际GPU
测试选择direct或chunk，不能仅以权重+KV相加推断总峰值。

## 4. 接入时必须处理的具体差异

GLM4 head_dim128，但partial_rotary_factor0.5，因此rotary_dim64，**只有32个旋转对**。
公开几何下32/1-turn band为 `[17,30]`，过渡宽度13；T/P公式本身支持该有限网格。

当前仓库有三处全RoPE/64对假设，接入前需要相应修改，而不是直接套Llama64对表：

1. `experiments/fixed_rope_three_interfaces_20260913/tables.py::model_geometry`：
   pairs目前直接由head_dim/2计算，需按实际rotary_dim确定。
2. `scripts/experiments/cross_audit/tables.py::install_static`：
   表长度校验应匹配模型的rotary inv_freq长度；保留GLM自己的相邻维旋转与非旋转部分。
3. `recovery_v2_eval.py`与`recovery_v2_runtime.py`：存在shape==(64,)断言及native建表路径，
   需要按模型配置/实际原生频率确定。不能只放宽一个assert，留下错误native初始表。

最小验证应检查Native表安装前后输出一致、T/P表真的安装到32对旋转子空间、
pass-through维度未受影响，然后再运行固定合同。单独的架构兼容性修正不改变方法系数。

## 5. 选择的研究价值

它提供与Qwen相同32K原生长度、但发布base10000的独立家族，也引入32对partial RoPE场景。
因此可增强“配置原则跨模型/旋转预算起作用”的证据。若观察到更大收益，仍需将“跨模型
验证”与“base的单因素因果证明”区分，因为模型权重、架构和训练数据也不同。

当前推荐已明确；后续下载、接入和GPU执行由作者的下一步指令或实验owner已有授权决定。
