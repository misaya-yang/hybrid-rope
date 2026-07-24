# FMRoPE vs EVQ：125M / L=256 定向实验

状态：spec/code present；仓库内无训练结果。远端数据、preflight 与 GPU 状态
必须现场核验，不能由本文件推断。

## 唯一问题

在完全相同的短上下文语言模型训练中，FMRoPE 的
`theta_train=L_train`、`theta_infer=L_target` 是否足以解释或超过
EVQ-Cosh 的外推收益？

这个实验不尝试证明 EVQ 是长上下文 SOTA，也不扩张到 RULER、LoRA 或
200K。它只回答 reviewer 提出的近邻方法问题。

## 为什么不用当前 8B LoRA

8B checkpoint 已在超大规模预训练中形成稳定的位置使用方式，短 LoRA 既可能
没有足够梯度去重写 Q/K 对频带的依赖，也把“频率分配是否有效”和“预训练模型
是否愿意迁移”混在一起。本实验从随机初始化开始全参数训练，三种固定频率从
step 1 起直接参与 attention，因此更适合回答方法本身的 matched comparison。
CPU 单测会验证三臂都产生 finite、非零且 schedule-dependent 的 QKV gradient；
这只证明梯度路径存在，不预先保证最终外推收益。

## 方法身份

FMRoPE 按 ICLR 2026 论文 Section 6.1 实现：

```text
theta_train = L_train
theta_infer = L_target
omega_i(theta) = theta^(-2i/d)
```

截至 2026-07-23 未找到作者公开实现。因此结果必须称为
**paper-faithful local FMRoPE implementation**，不能称为 official-code
reproduction。论文来源：
`https://openreview.net/forum?id=PR1PPxvG9Q`。

## 冻结协议

| 字段 | 值 |
| --- | --- |
| 模型 | 仓库历史 “125M” 配置；精确参数量 151,898,880 |
| 架构 | 12 layers, hidden 768, 12 heads, head_dim 64, SwiGLU 3072 |
| 数据 | FineWeb-Edu；三臂共享同一 token 顺序 |
| 训练上下文 | 256-token window（255 个 next-token predictions） |
| 请求预算 | 100M source tokens |
| 实际预算 | 99,942,400 source tokens；1,525 optimizer steps |
| 优化器 | AdamW, lr 6e-4, betas (0.9, 0.95), wd 0.01 |
| LR schedule | 152-step linear warmup，随后 cosine decay 到 6e-5 |
| batch | global 256 = micro 64 × accumulation 4 |
| 精度 | FP32 master weights + BF16 autocast/activations；三臂完全一致 |
| seed | 42 pilot |
| 评测长度 | L-token window：256 sanity；512、1024、2048 为目标外推组 |
| 评测统计 | 32 个固定 validation anchors；相同末端 128-token target |

三次训练仅在不可学习的 frequency tensor 上不同。令
\(K=d_{\mathrm{head}}/2\)：

1. `paper_geo_base500k`（Paper-Geo）：
   \(u_k=(k+\tfrac12)/K,\ \omega_k=500000^{-u_k}\)；
2. `fmrope_base256`：Std-Geo endpoint quantizer
   \(u_k=k/K,\ \omega_k=256^{-u_k}\)；
3. `evq_cosh_tau4_paper_grid_base500k`：与 Paper-Geo 相同的 midpoint
   quantizer，\(\phi_k=1-\operatorname{asinh}((1-u_k)\sinh\tau)/\tau\)，
   \(\omega_k=500000^{-\phi_k}\)，
   tau=4，base=500K。这里 `4 = d_head / sqrt(L_train) = 64 / 16`，
   是论文 operating default，不是看到本实验结果后调出的值。

论文方法的主比较仍是 `Paper-Geo` 对 `EVQ-Cosh`；FMRoPE 是 reviewer
近邻方法对照，不会替代或重定义该主比较。

Paper-Geo 与 EVQ-Cosh 因而共享量化约定；EVQ 的 `tau→0` 极限精确返回
Paper-Geo，而不是 Std-Geo。Std-Geo 的小规模消融放在
`../reviewer27be_shape_base/shape_l128`，不为此把 100M-token FMRoPE 实验
扩成第四次训练。三个 canonical schedule 的 float32 通道值与 SHA-256 固定在
`../../theory_results/FREQUENCY_DEFINITION_MANIFEST.json`。

## 推理条件

| checkpoint | condition | 作用 |
| --- | --- | --- |
| Paper-Geo-500K | raw | 投稿主链路实际使用的几何对照 |
| Paper-Geo-500K | YaRN-derived virtual-coordinate ramp, inference-only | 在 checkpoint 真实 midpoint 频率上检查 FMRoPE/YaRN 相似性；不是 official native-grid YaRN，也不是完整 YaRN 训练协议 |
| FMRoPE-256 | fixed base=256 | 分离训练时 base 与推理 retarget 的作用 |
| FMRoPE-256 | target base=L_eval | FMRoPE 主结果 |
| EVQ-tau4-500K | raw | 提交方法的直接结果 |

不向 EVQ 额外叠加 YaRN，避免把“FMRoPE vs EVQ”的问题扩成组合方法搜索。

## 为什么使用 paired tail-NLL

旧 Phase11B 每个长度抽取 8 个不同随机 chunk，长度间和方法间方差较大，而且
旧 validation 后来被发现与训练前缀重合。本实验使用独立 held-out shard，并为
所有长度、方法复用相同 32 个序列终点；任意相邻终点至少相隔 2048 tokens，
避免把重叠窗口误作独立样本。每个窗口只把最后 128 个预测位置作为 primary
metric，因此不同上下文长度看到的 target token 完全相同，唯一变化是左侧
可见上下文和 RoPE 方案。

这里的 `L` 与历史 Phase11B evaluator 保持一致：从 validation 取一个
L-token window，以前 L-1 个 token 为 model input，预测后 L-1 个 token。
FMRoPE 的推理 base 仍按论文公式设为该目标 window 长度 `L`，而不是人为改成
`L-1`。

同时保留 full-window NLL 作为 diagnostic，但不让它覆盖 paired tail 结论。

## 结果解释

- 若 FMRoPE target-matched 明显优于 EVQ：承认 performance novelty 不成立；
  rebuttal 只能区分 closed-form allocation shape 与 target-length base retargeting。
- 若两者接近：不得说 reviewer “错了”；应承认实用效果重叠，并把 novelty
  收窄到参数化与训练时频率预算。
- 若 EVQ 在 512/1024/2048 的 paired tail-NLL 稳定更低：这是直接的
  matched evidence，但 single-seed pilot 仍不能升级成普遍优越性。
- 若排序随长度反转：如实报告 trade-off；不要挑选单点。
- checkpoint 必须持久化 `inv_freq`；加载后 checkpoint、NPY sidecar 与
  metadata 的 float32 hash 必须完全一致，禁止构造函数静默重建。
- 任一臂 non-finite、checkpoint/hash 不一致或 held-out gate 失败：结果无效。

只有当 seed-42 差异足够影响回复且 reviewer 确实要求统计稳定性时，才考虑
seeds 137/256；本轮不预先扩张。

## 运行

默认数据准备不再依赖旧服务器或旧实验 manifest。它从固定 revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` 重建：

- 训练：FineWeb-Edu `sample/10BT/000_00000.parquet` 的确定性 token prefix；
- 验证：不同的 `004_00000.parquet` 的 5M-token prefix；
- tokenizer：固定 revision
  `EleutherAI/gpt-neox-20b@c292233c833e336628618a88a648727eb3dff0a7`。

两个 parquet（约 4.31 GB）、tokenizer 和最终 NPY 均做 SHA-256 校验。
训练/验证 NPY 约再占 0.84 GB；准备前至少保留 7 GB 可用空间。若旧的
native-150M manifest 仍存在，也可通过 `FMR_SOURCE_MANIFEST` 快速复用，
但它不再是运行前提。

无卡机器：

```bash
export FMR_WORK_DIR=/path/to/fmrope_l256_work
bash rebuttal/rebuttal_0723/experiments/fmrope_125m_l256/run_5090.sh prepare
bash rebuttal/rebuttal_0723/experiments/fmrope_125m_l256/run_5090.sh preflight
```

`prepare` 必须在无卡阶段执行，可中断续传原始 parquet。若默认镜像不可用，
可设置 `FMR_HF_ENDPOINT=https://huggingface.co`，或预先放入其他镜像的
同名文件；后者只有通过固定的原始 HF SHA-256 后才会被接受。成功后应备份整个
`$FMR_WORK_DIR/data/`（约 0.84 GB）和 `preflight.json`；原始 parquet
可随时按固定 hash 重下。

以上步骤完成并生成 `$FMR_WORK_DIR/preflight.json` 后才开 5090：

```bash
bash rebuttal/rebuttal_0723/experiments/fmrope_125m_l256/run_5090.sh run
```

最终原始结果和人类可读汇总分别位于：

```text
$FMR_WORK_DIR/evaluation/results.json
$FMR_WORK_DIR/evaluation/summary.md
```

如中途某一 arm 失败，先关 GPU 并诊断；不得覆盖已有目录。确认修复后可用
`FMR_ARMS="remaining_arm"` 只启动尚未完成的 arm。
