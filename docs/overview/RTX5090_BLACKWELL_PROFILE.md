# RTX 5090 Blackwell training profile

最后验证：2026-07-24，PyTorch `2.8.0+cu128`，CUDA capability `sm_120`。

这是本仓库后续 RTX 5090 训练的默认性能基线。它不固定科学 batch size；每个
新模型仍须先执行丢弃式 probe，确认显存、吞吐、loss 和算子后才能训练。

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
5. one compile step and at least five steady-state discarded steps pass;
6. finite loss, expected parameter count and adequate disk space.

If Flash SDPA is ineligible for a new attention shape or mask, the probe must
fail before paid training. Do not silently enable quadratic math attention.

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
