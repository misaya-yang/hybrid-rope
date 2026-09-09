# 三项后续方法：固定执行协议

用户明确要求全部试完三项。复用首次运行冻结的 Qwen2.5-1.5B、32 篇 PG19 校准/8 篇留出、8189-token 检索题、192 content + 64 rotary、BF16 推理。每项各层 500 步、seed 42、最大位置尺度 8，学习率/参数集/value 权重不变。现有 output KD（NLL 3.9427165）是复用的比较起点。三项分开，不把后面的改动叠在前一项结果上。

1. **BKV 初始化 + output KD**：按剩余 K / 全部 V 的平均逐通道 L2 范数得到 ratio，在 PCA 前把 K 除以 ratio，在 K 解码映射中乘回。对应 BKV 的平衡原理，适配本布局的完整 V；不声称逐行复刻上游 V slicing。保存 `work/bkv_initialization`，独立测 NLL，再从它执行原 output+value 500 步，保存 `work/bkv_kd`。唯一干预是初始化，包括其导致的参数坐标尺度改变。
2. **attention KL + output + value**：从原 `work/initialization` 开始，三个权重均为 1、raw score 权重 0；保存 `work/attention_kd`。KL 方向为 teacher || student，因果 softmax，直接比较输出 KD 是否因 attention 分布约束而改善。没有按测试结果调权重。
3. **学生前缀上的逐层 output KD**：从原初始化开始，每层 target 仍为冻结原模型 Q/K/V 所定义的 output/value；student Q/K/V 来自已经压缩前序层的实际 hidden states。拟合后用该层完整 residual/MLP/layernorm 前向，把状态传给下一层。仅训练同一组 factor 参数；不改 LM 权重。保存 `work/progressive_kd` 及每层 student_inputs。第一次层输入严格等于原始捕获；每步 teacher 文档/距离轨迹与旧 output KD 对齐。

每项读取同一已知答案条件 NLL、原始生成和 8 篇真实 8K 留出 NLL；留出书籍/检索题不进入拟合。记录初始张量哈希、实际训练轨迹和耗时。全模型 NLL、实际答案与局部误差分别解释，单道题不估计广泛任务准确率。三项全部完成后才汇总结论，不根据一个中间读数提前停止整个请求。

命令从 `/root/autodl-tmp/operator_family_prepare_20260909` 执行，Python 为 `/root/miniconda3/bin/python`。

```bash
python -m experiments.rope_operator_family.run fit --capture work/capture --out work/bkv_initialization --content-rank 192 --rotary-dim 64 --initialize-only --balance-kv --device cuda
python -m experiments.rope_operator_family.run fit --capture work/capture --init-from work/bkv_initialization --out work/bkv_kd --content-rank 192 --rotary-dim 64 --steps 500 --max-position-scale 8 --score-weight 0 --output-weight 1 --value-weight 1 --device cuda
python -m experiments.rope_operator_family.run fit --capture work/capture --init-from work/initialization --out work/attention_kd --content-rank 192 --rotary-dim 64 --steps 500 --max-position-scale 8 --score-weight 0 --output-weight 1 --value-weight 1 --attention-weight 1 --device cuda
python -m experiments.rope_operator_family.progressive --model /root/autodl-tmp/qwen25_1p5b_32k --data work/data --capture work/capture --init-from work/initialization --out work/progressive_kd --steps 500 --device cuda
```

各份 checkpoint 的 evaluate/generate/diagnose/profile 复用现有 CLI，输出单独命名；旧结果不覆盖。18 个代码检查（原 15 + 新 3）验证数值/接口，不作为方法效果证据。
