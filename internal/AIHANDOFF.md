# EVQ-Cosh AI Agent Handoff

阅读顺序: Part 0 (GPU准则) → Part 1 (架构) → Part 2 (EVQ公式) → Part 3 (禁令)
其他参考: `docs/overview/PAPER_CLAIMS_MAP.md` | `paper/main.tex` | `results/video_dit/REPORT_FINAL.md`

## Part 0: GPU 实验铁律

硬件: RTX 5090 (32GB) + RTX 6000 Pro (96GB), Blackwell sm_120, PyTorch ≥2.7, CUDA 12.8, bfloat16

1. 必须`torch.compile(model, mode="default")`，不用=浪费30-50%算力。实测454M +40%吞吐且VRAM下降，DiT +46%。首次用default，稳定后可切max-autotune。
2. 训练循环每步开头调用`torch.compiler.cudagraph_mark_step_begin()`，缺少则compile退化到eager。
3. 脚本头部设`os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"`。
4. batch size目标是吃满显存(VRAM使用率>90%)。查表一次到位: 129.6M DiT/5090/32f→bs28(~25GB), 454M/5090/L2048→micro=2 accum=5(~25GB), 750M/6000Pro/L2048→bs8-12(~60-80GB)。启动后立刻用nvidia-smi检查，VRAM<80%必须加大batch，不要浪费显存跑完全程。
5. dtype必须bfloat16，Blackwell原生支持，不要用float16。
6. 多run流水线(如h2h多个τ值)切换时必须彻底释放显存，否则前一个run的残留直接导致下一个OOM或显存浪费。标准流程: `model.cpu(); del model; del optimizer; del scheduler; del kv_cache; del all_tensors_on_gpu; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()`。每个run开始前打印`torch.cuda.memory_allocated()`确认归零。eval同理，eval完必须释放再恢复训练。
7. 不要反复调试: 本地写代码→远程跑5步验证loss/VRAM/无报错→直接启动完整训练。禁止上传→OOM→改batch→再OOM→改compile的循环。
8. GPU分配: ≤350M和DiT用5090，750M+和长序列eval(16K-48K)用6000Pro。
9. 结果必须保存到JSON/log，不要只print到stdout。中间checkpoint必须有，防中断丢失。
10. gradient accumulation解决显存不够: `loss = model(batch) / grad_accum; loss.backward()`循环后再`optimizer.step()`。
11. 每次训练启动前必须验证数据量是否合适: 打印total_steps、tokens_per_step(=batch×seq_len)、total_tokens，确认总训练量与目标一致。训练不足导致欠拟合，训练过多浪费GPU时间。参考: 50M→50M tok, 125M→100M tok, 350M→100M tok, DiT 129.6M→15000 steps×bs×frames。
12. 5步验证时必须同时打印: step时间(ms)、VRAM峰值(GB)、loss值、学习率、total_steps，确认全部正常再启动全量训练。VRAM峰值<总显存80%说明batch太小，必须加大。
13. 引用第三方方法(RIFLEx/NTK/FIRE等)时，必须先阅读官方repo代码确认关键参数和实现细节，不要凭论文描述自己重写。已踩坑: RIFLEx实现漏了0.9安全系数且用L_train代替L_test选intrinsic frequency，导致结果完全错误浪费一轮eval。正确做法: clone官方repo→找到核心函数→确认所有参数含义→在我们代码中精确复现，不要"大概理解了就自己写"。
14. eval pipeline必须有sanity check: 对比方法的绝对值必须合理(如inference-time优化方法应该比raw好，否则实现有bug)，新加的eval指标先在已知结果上验证正确性再跑新实验。如果结果违反常识(如优化方法反而让所有模型变差)，第一反应是检查实现而不是解读为"方法在此setting不work"。

性能参考: 454M L=512 5090: eager 231ms/44Ktok/25.1GB → compile 165ms/62Ktok/17.6GB (+40%)。129.6M DiT 5090: eager bs=16 ~73samp/s → compile bs=28 ~108samp/s (+46%)。

启动前checklist: torch.compile+cudagraph+expandable_segments, batch吃满显存(>90%), bf16, 5步验证(打印ms/VRAM/loss/lr/total_steps), 验证数据量, τ=d_head/√L_train, YaRN=Progressive not NTK-aware, scale=eval_len/train_len, inv_freq float64, 架构与Part1一致, 结果存JSON+checkpoint, 多run间显存彻底释放, 第三方方法对照官方repo实现不要自己重写, eval结果做sanity check(优化方法应比raw好否则有bug)。

## Part 1: 架构规范

任何偏离=结果不可比=浪费GPU。

| Tier | Params | Layers | Heads | head_dim | Hidden | FFN | Vocab |
|------|--------|--------|-------|----------|--------|-----|-------|
| 50M | ~50M | 6 | 8 | 64 | 512 | 2048 | 50304 |
| 125M | ~125M | 12 | 12 | 64 | 768 | 3072 | 50304 |
| 350M | ~350M | 24 | 16 | 64 | 1024 | 4096 | 50304 |
| 454M | ~454M | 24 | 16 | 64 | 1024 | 4096 | 50304 |
| 750M | ~750M | 18 | 24 | 64 | 1536 | 6144 | 50304 |

750M是18层/24头不是24层/16头。454M与350M共享架构。

训练超参: 50M(lr=6e-4,bs=32,50M tok), 125M(3e-4,16,100M), 350M(2e-4,2,100M), 500M(1.5e-4,4,500M)。共用AdamW(β1=0.9,β2=0.95,wd=0.1), cosine→1e-5, tokenizer=gpt-neox-20b(50304), FineWeb-Edu, base=500000。Continued pretrain: LR=1e-5, warmup=500, micro=2, accum=5。

## Part 2: EVQ-Cosh 公式

```
φ_k(τ) = 1 - (1/τ) × arcsinh((1 - u_k) × sinh(τ))
u_k = (k + 0.5) / K,  K = head_dim/2 = 32
inv_freq_k = base^(-φ_k(τ)),  base = 500,000
τ* = d_head / √L_train → L=2048:τ≈1.414, L=256:τ=4.0, L=512:τ≈2.828
```

规范实现 `run_evq_sweep.py:141-157`(midpoint u_k)。`schedules.py`用u_k=k/n无midpoint，与论文不一致，以run_evq_sweep.py为准。

```python
def evq_cosh_inv_freq(head_dim, tau, base=500000.0):
    K = head_dim // 2
    idx = torch.arange(K, dtype=torch.float64)  # 必须float64
    u = (idx + 0.5) / float(K)
    if abs(tau) < 1e-8:
        phi = u
    else:
        phi = 1.0 - (1.0/tau) * torch.arcsinh((1.0-u) * math.sinh(tau))
    return torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi).float()
```

## Part 3: 禁令

1. 不要用NTK-aware YaRN，破坏EVQ频率结构。用Progressive per-channel ramp，参考phase14c_multiscale_evq_yarn.py。
2. 不要在Geo checkpoint上直接换EVQ inv_freq，必须从头训练或continued-pretrain(phase11e)。
3. 不要用τ=0.707，≈Geo无差异。用τ*=d_head/√L_train。
4. 不要用max-autotune做首次运行，先default确认逻辑。
5. eval改seq_len必须同步YaRN scale=eval_len/train_len。
6. 不要混用tokenizer，统一gpt-neox-20b(50304)。
7. 纯文本训练不会passkey 100%，需混入5-10% passkey数据。
8. 454M QuALITY不要用accuracy做指标(容量地板~25%≈随机)，用Gold NLL(-30.1%@8K)。
9. DiT禁止跨run比较，CUDA非确定性造成70%+虚假差异，必须head-to-head同run对比。
10. DiT不要用τ*_AR=K/√T，中频抽空导致位置指纹崩坏。τ*_DiT≈0.53×K_t/√T_train，K_t=16,T=32→τ≈1.5。
11. Power-Shift族已证伪(α=0.25差22x)，DiT仍用Cosh。
12. base=10000+K_t=16+T=32产生死通道(θ_k×Δ≈0→相变)，注意base_t选择。
13. 实验work_dir必须放数据盘(`/root/autodl-tmp/`)，不要放系统盘(`/root/`)。系统盘30G容易满，每个DiT实验~2GB(checkpoints+npy)。命令示例: `--work_dir /root/autodl-tmp/hybrid-rope/results/video_dit/xxx`。
14. 不要凭论文描述自己重写第三方方法，必须对照官方代码逐行确认。已踩坑: RIFLEx自己实现漏了0.9系数+用错L_train/L_test，导致结果完全错误。RIFLEx官方实现: `freqs[k-1] = 0.9 * 2π / L_test`（注意是L_test不是L_train，k由官方intrinsic frequency detection选出，不要自己猜）。
15. eval结果必须做sanity check: 如果一个公认有效的inference-time方法(RIFLEx/YaRN)让所有模型都变差，第一反应是检查实现bug，不是解读为"方法不适用"。

## Part 4: Wan2.1 视频微调避坑 (Phase 17)

### 模型选型坑

| 模型 | RoPE | Flow-matching | 参数量 | 结论 |
|------|:---:|:---:|--------|------|
| CogVideoX-2B | ❌ sinusoidal | ❌ DDPM | 2B | **废弃**: `use_rotary_positional_embeddings=false`，monkey-patch无效 |
| CogVideoX-5B | ✅ | ❌ DDPM | 5B | 可用但非FM，推理慢(50-100步) |
| HunyuanVideo | ✅ | ✅ | 13B | 太大，训练慢 |
| **Wan2.1-1.3B** | ✅ | ✅ rectified flow | 1.3B | **选定**: FM+RoPE+小模型，推理快(20-30步) |

选型前必须验证: (1) 模型config中RoPE是否启用 (2) 调度器类型(FM vs DDPM) (3) 参数量能否fit显存。

### Wan2.1 原始模型 vs diffusers

diffusers `WanPipeline` 需要 diffusers 格式模型(含 `model_index.json`)。服务器上只有原始格式，不要浪费时间下载/转换。直接用原始 `WanModel`:

```python
sys.path.insert(0, "/root/autodl-tmp/wan21/Wan2.1")  # wan module
from wan.modules.model import WanModel
model = WanModel.from_pretrained("/root/autodl-tmp/wan21")  # 传目录不是文件
```

关键接口差异:
- **forward**: `model(x=List[Tensor], t=Tensor, context=List[Tensor], seq_len=int)` — x 和 context 是**列表**不是 batch tensor
- **RoPE**: `model.freqs` 是 complex tensor `[1024, 64]`，前22列=temporal，后42=spatial(21h+21w)
- **LoRA targets**: `"q", "k", "v", "o"` 不是 `"to_q", "to_k", "to_v", "to_out.0"`
- **VAE encode**: `vae.encode([tensor])` 输入是列表，每个 `[C,T,H,W]`

### 分辨率与序列长度

VAE stride=(4,8,8), patch_size=(1,2,2)。分辨率必须被 8 整除且 latent 空间被 2 整除:

| 分辨率 | latent H×W | patch后 | seq_len (T=21) | step time (bs=1) | VRAM |
|--------|-----------|---------|----------------|-------------------|------|
| 480×832 | 60×104 | 30×52 | 32,760 | ~45s | 87GB (bs=8+ckpt) |
| **288×288** | 36×36 | 18×18 | **6,804** | **2.8s** | 79.8GB (bs=1) |
| 280×280 | 35×35 | ❌不整除 | — | — | — |

**280 不行**: 280/8=35(奇数), patch_size=2 无法整除。用 288。

### 显存坑

- bs=4 @ 288×288 无 gradient checkpointing: **OOM** (>96GB)
- bs=2 @ 288×288 无 gradient checkpointing: **OOM** (>95GB)
- bs=1 @ 288×288 无 gradient checkpointing: **79.8GB** ✓
- 之前 480×832 bs=8 + gradient checkpointing: 87.7GB ✓ 但 45s/step

要 bs>1 必须开 gradient checkpointing，但会拖慢每步速度。

### 编码坑

1. **T5 必须加载到 GPU**: 96GB 显存放 T5(11GB)+VAE(0.5GB) 绰绰有余，不要照抄旧脚本的 CPU 加载
2. **文本嵌入长度不一致**: 不同 caption 产生不同长度的 T5 embedding，必须 pad 到 max_length 再 stack
3. **PYTHONPATH**: `wan` 模块在 `/root/autodl-tmp/wan21/Wan2.1/`，必须加到 PYTHONPATH

### temporal RoPE 修改

```python
# model.freqs: complex [1024, 64], 前22列是temporal
K_t = 22  # t_dim=44, 22 pairs
positions = torch.arange(1024, dtype=torch.float64)
raw = torch.outer(positions, inv_freq.double())
new_freqs = torch.polar(torch.ones_like(raw), raw)
model.freqs[:, :K_t] = new_freqs
model.freqs = model.freqs.to(device)
```

注意: Wan2.1 base=10000 + K_t=22 → 10/22 通道死/近死(45.5%)。τ*_DiT = 0.53 × 22/√13 ≈ 3.2。

### 训练配置 (已验证)

```
Model: Wan2.1-T2V-1.3B, 1419M params
Resolution: 288×288, 81 frames
Latent: [16, 21, 36, 36], seq_len=6804
LoRA rank=16, targets=["q","k","v","o"], 11.8M trainable
bs=1, lr=1e-4, AdamW(0.9,0.95), cosine schedule
VRAM: 79.8GB, step time: 2.8s/step
500 steps GEO + 500 steps EVQ ≈ 47 min
```

### 服务器操作铁律

16. 永远不要在付费GPU服务器上调试。所有代码在本地写好、检查好，一次传上去跑。
17. 每次上传脚本前: (a) 确认所有 import 路径正确 (b) 确认 device=cuda (c) 确认 tensor shape 匹配 (d) 确认 batch_size 不会 OOM
18. 用 `nohup python -u script.py > /tmp/log 2>&1 &` 后台跑，`python -u` 保证 unbuffered 输出
19. 不要用 sleep+poll 检查进度，直接 `tail /tmp/log`

## 关键文件

论文: `paper/main.tex` | 映射: `docs/overview/PAPER_CLAIMS_MAP.md` | sweep脚本: `scripts/core_text_phases/run_evq_sweep.py` | RoPE库: `scripts/lib/rope/schedules.py` | DiT报告: `results/video_dit/REPORT_FINAL.md` | DiT理论: `DiT_frequency_allocation_analysis.md` | DiT脚本: `scripts/video_temporal/run_dit_temporal.py` | Wan2.1训练: `scripts/video_temporal/wan21_raw_train.py`
