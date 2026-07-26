# RTX 5090 Blackwell training profile

最后验证：2026-07-25，PyTorch `2.8.0+cu128`，CUDA capability `sm_120`。

这是本仓库后续 RTX 5090 训练的默认性能基线。第一性原则是用最快的稳定路径
最小化回答研究问题所需的总 GPU 时间与成本。它不固定科学 batch size；若已有
相同或足够相近 shape 的完成 receipt，直接复用。只有没有可复用证据时，才做
能够确认显存、吞吐、finite loss 和算子资格的最短丢弃式 probe。

## 已实测配置

```python
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)

compiled_loss = torch.compile(
    loss_module,
    mode="default",
    dynamic=False,
    fullgraph=False,
)
optimizer = torch.optim.AdamW(..., fused=True)
```

Shell runtime:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="$WORK_DIR/torchinductor_cache"
```

Training and evaluation use BF16 autocast. Before the first real step, require:

1. `torch.cuda.get_arch_list()` contains the active architecture;
2. BF16 is supported;
3. Flash SDPA is available and enabled;
4. math, memory-efficient and cuDNN SDPA fallbacks are disabled;
5. the shortest compile/steady-state probe needed for this shape passes;
6. finite loss, expected parameter count and adequate disk space.

If Flash SDPA is ineligible for a new attention shape or mask, the probe must
fail before paid training. Do not silently enable quadratic math attention.

## 长上下文 LoRA：不要把显存占满当成优化目标

2026-07-25 的 OLMo-2 1.5B、16K、QKVO LoRA（rank 64，global batch
4，micro-batch 1，gradient accumulation 4）运行给出了一个需要长期保留的
反例：启用整层 activation checkpointing 后，进程只驻留约 12,152 MiB，
但连续 12 秒采样中 SM 利用率为 99--100%，板卡功耗为 573--575 W，
驱动报告 `SW Power Cap: Active`，同时没有 thermal slowdown。训练步骤
110--190 的稳态吞吐为 12,454 tokens/s。

因此：

- GPU 时间与成本是执行层的首要优化目标。只要模型/方法、数据与 split、
  有效训练预算与目标、优化器/LR 语义以及评估样本/指标保持广义一致，就应
  使用最快的稳定运行方式。
- micro-batch、accumulation、checkpoint、compile、kernel、allocator、
  dataloader 和评估 batching 属于执行细节；它们可以跨 arm 不同，不因形式
  不对称而重跑已完成实验。
- 低显存占用不能推出 GPU 没有吃满；显存容量、memory-controller
  utilization、SM utilization 和功耗墙是不同约束。
- 该工作负载是计算/功耗受限，不是 dataloader 受限。仅增大 batch
  以“填满显存”没有已知收益，还可能降低核心频率或破坏科学协议。
- 整层 activation checkpointing 用空闲显存换取了 backward 重算。这里
  真正值得测的是用额外显存减少重算，而不是追求显存占用百分比。
- 不得为了显存数字好看而中断健康的 registered run。Geo/EVQ 配对只要求
  上述广义科学契约一致；所有执行细节按最快稳定方案选择并写入 receipt，
  不因执行配置不同而自动重跑已经完成的 arm。

对新的长上下文 LoRA shape，按以下顺序选择运行配置：

1. 若没有可复用 receipt，用足以判断的最少稳态样本记录 SM、
   memory-controller、功耗、时钟、限频原因、tokens/s、finite loss 和峰值
   显存；一旦候选明显足够好就停止 probe。
2. 先复用相近 shape 中总 GPU 成本最低的完成配置。只有存在明确提速机会时，
   才短测另一个 micro/accum/checkpoint/compile 组合，不穷举配置。
3. 以预计到完成的总 GPU 时间/成本选优；显存只需稳定可行，不设固定占用率
   或固定空余比例目标。
4. 若当前配置仍不够快，只测试最有可能降低总成本的下一项，例如关闭
   checkpoint、启用 compile，或在前两者不可行时采用 selective activation
   checkpointing；记录编译开销和持久 cache。
5. 快速 probe 只负责排除 OOM、非法算子和非有限 loss，并选择足够快的候选；
   不要求为了形式完整再跑固定网格或长 probe。若一条 matched arm 已完成，
   只要广义科学契约不变，可将更快执行配置用于未完成 arm；不要仅为执行层
   对称性重跑已完成 arm。

### 16K QKVO LoRA 的实测运行配置

同一 Geo checkpoint、数据哈希、global batch 4、BF16、Flash-only SDPA 和
fused AdamW 下，2026-07-25 的丢弃式 probes 为：

| checkpoint | compile | micro / accum | mean tokens/s | peak allocated / reserved | 相对基线 |
| --- | --- | ---: | ---: | ---: | ---: |
| on | none | 1 / 4 | 12,506 | 6.309 / 6.670 GiB | baseline |
| on | none | 2 / 2 | 12,518 | 9.568 / 10.227 GiB | +0.10% |
| on | none | 4 / 1 | 12,612 | 16.023 / 17.346 GiB | +0.85% |
| off | none | 1 / 4 | 17,360 | 29.034 / 29.230 GiB | +38.81% |
| off | none | 2 / 2 | OOM | requested 512 MiB with 193.69 MiB free | infeasible |
| off | max-autotune-no-cudagraphs | 1 / 4 | 24,020 | 21.739 / 22.236 GiB | +92.08% |
| off | max-autotune-no-cudagraphs, warm cache | 1 / 4 | 23,988 | 20.835 / 21.025 GiB | +91.82% |

`max-autotune-no-cudagraphs` 的首次冷编译步为 111.7 秒；复用约 301 MiB
持久 cache 后，同配置首步为 9.7 秒，10-step 稳态吞吐只变化 0.13%。
进一步的 warm-cache 50-step receipt 为 mean 23,827、median 23,818
tokens/s，即相对同脚本 checkpoint 基线仍为 +90.5%。该 50-step 运行的
12 秒硬件采样为 97--100% SM、573--576 W、约 22,170 MiB 进程显存。

因此这个 shape 的下一条 matched 训练候选是：

```text
micro_batch=1
gradient_accumulation=4
gradient_checkpointing=off
torch.compile=max-autotune-no-cudagraphs
TORCHINDUCTOR_CACHE_DIR=<persistent data-disk path>
```

这不是所有模型的全局默认。完整 300-step arm 的 sustained throughput
仍须替换 50-step 估计；compile 与 eager 存在轻微浮点路径差异，因此必须
记录实现哈希、compile 模式、首步 loss 和完整训练曲线，但这类执行差异本身
不要求重跑已完成的 matched arm。只有当验证发现明显数值漂移、非有限 loss
或评估定义变化时，才升级为配对重跑问题。单纯增大 checkpoint-on
micro-batch 已被实测排除为有效提速手段。

### 训练饱和不代表评估也饱和

同一运行进入 `eval_batch_size=1` 的 4K--32K teacher-forced 评估后，另一组
连续 12 秒样本只有平均 69.9% SM（范围 11--100%）、平均 429 W
（范围 407--455 W）和约 3,944 MiB 进程显存，呈明显的锯齿空洞。这才是
真实的未充分利用，不能与上面的 power-capped 训练混为一谈。

长上下文评估应单独 probe，并优先：

- 按每次约 32K 总输入 token 设分长度 batch，例如 4K/8K/16K/32K 使用
  8/4/2/1；实际值仍以 OOM-free receipt 为准。
- 不要在每个小 batch 后无条件 `gc.collect()` 和
  `torch.cuda.empty_cache()`；只在长度切换、已验证的内存压力点或 OOM
  恢复路径清理 allocator。
- 对 source/deleted/swapped 等同 shape 反事实，可在显存允许时合并 forward
  或至少批处理，减少 Python 和 kernel-launch 空洞。
- 改变评估 batch 或合并顺序后，先在固定样本上验证 metric parity。若指标
  在约定容差内逐样本一致，可直接沿用已有结果并记录执行变化；若不一致，
  才需要统一 evaluator 或重算受影响的 matched arms。

## Verified receipt

For the 50.1M MLA, sequence length 4096, micro/global batch 32 workload:

| item | K=8 | K=32 |
| --- | ---: | ---: |
| steady throughput | 427,546 tokens/s | 428,000 tokens/s |
| peak allocated CUDA memory | 32,171,419,136 bytes | 32,171,419,136 bytes |
| first compile step | 32.09 s | 10.82 s with shared cache |
| estimated 299.9M-token train | 701.4 s | 700.7 s |

GPU utilization during training was 99–100%, approximately 31.4 GiB reported
process memory and 575–576 W. The shared cache materially reduced later compile
startup.

Across the six complete 299,892,736-token runs, mean sustained throughput was
406,311 tokens/s (range 405,765–406,593), and total training-loop time was
73.81 minutes. Each run peaked at 32,171,420,160 allocated CUDA bytes. The
five-step probes therefore overestimated sustained end-to-end training
throughput by 5.3%; use complete-run throughput for cost estimates after the
first arm finishes.

These numbers validate this workload only. For another model, preserve the
scientific global batch unless the protocol explicitly changes it, then tune
micro-batch/accumulation with discarded probes rather than guessing from free
memory.

For the registered MLA operator-parity follow-up, evaluation uses 8/4/2/1
windows at 4K/8K/16K/32K. This keeps every inference forward at no more than
32K total input tokens, preserves per-window NLL, and reduces launch overhead.
It is an evaluation-only execution optimization, not a training variable.
