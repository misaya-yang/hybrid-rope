# EVQ-Cosh AI Agent Handoff

> **阅读顺序**: Part 0 (GPU准则) → Part 1 (架构) → Part 2 (EVQ公式) → Part 3 (禁令)
> **其他参考**: `docs/overview/PAPER_CLAIMS_MAP.md` | `paper/main.tex` | `results/video_dit/REPORT_FINAL.md`

---

## Part 0: GPU 实验核心准则

**硬件**: RTX 5090 (32GB) + RTX 6000 Pro (96GB)，均为 Blackwell sm_120
**环境**: PyTorch ≥ 2.7.0, CUDA 12.8, bfloat16

### 10 条铁律

**1. 必须 torch.compile** — 不用 = 浪费 30-50% 算力

```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
model = torch.compile(model, mode="default")  # 首次用 default，稳定后可 max-autotune
```

实测: compile 给 +40~46% 吞吐，VRAM 反而下降。

**2. 每步调用 cudagraph_mark_step_begin** — 缺少则 compile 退化到 eager

```python
for step in range(total_steps):
    torch.compiler.cudagraph_mark_step_begin()
    ...
```

**3. batch size 查表一次到位，不要反复试错**

| 模型 | GPU | seq_len | 推荐 batch | VRAM |
|------|-----|---------|-----------|------|
| 129.6M DiT | 5090 | 32f | 24-28 | ~25GB |
| 454M Text | 5090 | 512 | 测试中 | ~17.6GB |
| 454M Text | 5090 | 2048 | micro=2, accum=5 | ~25GB |
| 750M Text | 6000 Pro | 2048 | 8-12 | ~60-80GB |

OOM → batch 减半；VRAM < 80% → batch 加 50%。不要从小 batch 逐步试上去。

**4. gradient accumulation 解决显存不够**

```python
for micro_step in range(grad_accum):
    loss = model(batch) / grad_accum
    loss.backward()
optimizer.step(); optimizer.zero_grad()
```

**5. eval 不占训练 VRAM** — eval 前 `torch.cuda.empty_cache()`，用小 eval_batch

**6. 不要反复调试** — 本地写代码 → 远程跑 5 步验证 → 直接启动完整训练

**7. GPU 任务分配**: ≤350M / DiT 用 5090；750M+ / 长序列 eval 用 6000 Pro

**8. dtype 必须 bfloat16** — Blackwell 原生支持，不要用 float16

**9. 结果保存到 JSON** — 不要只 print 到 stdout，中间 checkpoint 必须有

**10. 启动前 30 秒 checklist**

- [ ] torch.compile + cudagraph_mark_step_begin + expandable_segments
- [ ] batch 查表，不是瞎猜
- [ ] bfloat16
- [ ] 先 5 步验证再全量跑
- [ ] τ = d_head/√L_train（不是 0.707！）
- [ ] YaRN = Progressive per-channel（不是 NTK-aware！）
- [ ] YaRN scale = eval_len / train_len
- [ ] inv_freq 用 float64 计算
- [ ] 架构与 Part 1 表格一致
- [ ] 结果自动存 JSON + checkpoint

### 性能参考

| 配置 | eager | compile(default) | 提升 |
|------|-------|-----------------|------|
| 454M L=512 5090 | 231ms 44K tok/s 25.1GB | 165ms 62K tok/s 17.6GB | +40% |
| 129.6M DiT bs=16 5090 | ~73 samp/s | bs=24: ~108 samp/s | +46% |

---

## Part 1: 架构规范

**任何偏离 = 结果不可比 = 浪费 GPU。**

### 架构表

| Tier | Params | Layers | Heads | head_dim | Hidden | FFN | Vocab |
|------|--------|--------|-------|----------|--------|-----|-------|
| 50M  | ~50M   | 6      | 8     | 64       | 512    | 2048| 50304 |
| 125M | ~125M  | 12     | 12    | 64       | 768    | 3072| 50304 |
| 350M | ~350M  | 24     | 16    | 64       | 1024   | 4096| 50304 |
| 454M | ~454M  | 24     | 16    | 64       | 1024   | 4096| 50304 |
| 750M | ~750M  | **18** | **24**| 64       | 1536   | 6144| 50304 |

⚠️ 750M 是 18层/24头，不是 24层/16头！454M 与 350M 共享架构，参数差来自训练设置。

### 训练超参

| Tier | LR | Batch | Seq Len | Tokens |
|------|----|-------|---------|--------|
| 50M  | 6e-4 | 32 | 2048 | 50M |
| 125M | 3e-4 | 16 | 2048 | 100M |
| 350M | 2e-4 | 2  | 2048 | 100M |
| 500M | 1.5e-4 | 4 | 2048 | 500M |

共用: AdamW (β1=0.9, β2=0.95, wd=0.1), cosine decay → 1e-5, tokenizer=`EleutherAI/gpt-neox-20b` (50304), dataset=FineWeb-Edu, RoPE base=500000

Continued pretrain: LR=1e-5, warmup=500, micro_batch=2, grad_accum=5

---

## Part 2: EVQ-Cosh 公式

```
φ_k(τ) = 1 - (1/τ) × arcsinh((1 - u_k) × sinh(τ))
u_k = (k + 0.5) / K,  K = head_dim/2 = 32
inv_freq_k = base^(-φ_k(τ)),  base = 500,000

τ* = d_head / √L_train
  L=2048 → τ*≈1.414 | L=256 → τ*=4.0 | L=512 → τ*≈2.828
```

规范实现: `run_evq_sweep.py:141-157`（midpoint u_k）

```python
def evq_cosh_inv_freq(head_dim, tau, base=500000.0):
    K = head_dim // 2
    idx = torch.arange(K, dtype=torch.float64)   # 必须 float64
    u = (idx + 0.5) / float(K)
    if abs(tau) < 1e-8:
        phi = u
    else:
        phi = 1.0 - (1.0/tau) * torch.arcsinh((1.0-u) * math.sinh(tau))
    return torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi).float()
```

⚠️ `schedules.py` 用 u_k=k/n（无 midpoint），与论文不一致，以 `run_evq_sweep.py` 为准。

---

## Part 3: 禁令清单

| # | ❌ 禁止 | ✅ 正确做法 |
|---|---------|-----------|
| 1 | NTK-aware YaRN（破坏 EVQ 频率结构）| Progressive YaRN per-channel ramp。参考 `phase14c_multiscale_evq_yarn.py` |
| 2 | Geo checkpoint 直接换 EVQ inv_freq | 从头训练或 continued-pretrain（phase11e） |
| 3 | τ=0.707（≈Geo，无差异）| τ*=d_head/√L_train |
| 4 | max-autotune 做首次运行 | 先 mode="default" 确认逻辑，再切换 |
| 5 | eval 改 seq_len 忘改 YaRN scale | scale = eval_len / train_len |
| 6 | 混用 tokenizer | 统一 gpt-neox-20b (50304) |
| 7 | 纯文本训练后期望 passkey 100% | 需混入 5-10% passkey 数据 |
| 8 | 454M QuALITY 用 accuracy 做指标 | 用 Gold NLL（-30.1% @8K） |
| 9 | DiT 跨 run 比较（CUDA 非确定性造成 70%+ 虚假差异）| **必须 head-to-head** 同 run 对比 |
| 10 | DiT 用 τ*_AR=K/√T（中频抽空 → 位置指纹崩坏）| τ*_DiT ≈ 0.53×K_t/√T_train，K_t=16,T=32 → τ≈1.5 |
| 11 | Power-Shift 族 φ_k(α)=1-(1-u_k)^(1+α) | 已证伪：α=0.25 差 22x。DiT 仍用 Cosh |
| 12 | 忽略 base=10000 死通道（θ_k×Δ≈0 → 相变）| 注意 base_t 选择，小 base 可消除 |

---

## 关键文件索引

| 用途 | 文件 |
|------|------|
| 论文 LaTeX | `paper/main.tex` |
| 论文↔实验映射 | `docs/overview/PAPER_CLAIMS_MAP.md` |
| τ-sweep 主脚本 | `scripts/core_text_phases/run_evq_sweep.py` |
| RoPE schedule 库 | `scripts/lib/rope/schedules.py` |
| Video DiT 报告 | `results/video_dit/REPORT_FINAL.md` |
| DiT 理论分析 | `DiT_frequency_allocation_analysis.md` |
| DiT 训练脚本 | `scripts/video_temporal/run_dit_temporal.py` |
