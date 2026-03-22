# EVQ-Cosh AI Agent Handoff

参考: `docs/overview/PAPER_CLAIMS_MAP.md` | `paper/main.tex` | `results/video_dit/REPORT_FINAL.md`

---

## Part 0: GPU实验铁律

硬件: RTX 6000 Pro Blackwell 96GB (`ssh -p 23173 root@connect.bjb1.seetacloud.com`), PyTorch≥2.7, CUDA 12.8, bfloat16

### ⚠️ VRAM计算（历次OOM根源，必须掌握）

```
VRAM总量 = 静态 + 激活 + LOGITS峰值

静态  = params × 12B  (bf16模型2B + bf16梯度2B + fp32优化器8B)
          125M→1.5GB,  350M→4.2GB

LOGITS = B × seq_len × vocab × 2B   ← 不随模型大小变化！只随batch增大
          vocab=50257(gpt2-neox)
          B=20,L=4096 → 20×4096×50257×2 = 8.2GB
          B=32,L=4096 → 13.2GB  ← 这是bs=32 OOM的真正原因

激活   = 从同seq_len实测数据线性外推（不同seq_len不可互推！）
```

**MLA架构必须对齐业界标准（DeepSeek-V2/V3, Kimi K2）：**

| 参数 | 业界标准 | 旧配置(已废弃) |
|------|---------|--------------|
| d_rope | **64** | 32 |
| d_nope | **128** | 32 |
| v_head_dim | **128** | 64 |
| kv_lora_rank | **512** | 128~256 |
| K (RoPE频率数) | **32** | 16 |
| base | **10000** | 500000 |
| RoPE占attention | **33.3%** | 50% |

**实测安全配置（直接用，不要重新估算）：**

| 模型 | GPU | seq_len | bs | VRAM | ETA/run |
|------|-----|---------|----|----|---------|
| 50M MLA(v2) | 5090/32GB | 4096 | **2** | ~16GB | ~? |
| 125M MLA | 96GB | 4096 | **20** | 91GB ✅ | ~100min |
| 350M MLA | 96GB | 4096 | **12** | 95GB ✅ | ~3.5h |
| 350M MLA | 96GB | 8192 | **5**  | 80GB ✅ | ~2.3h |

**OOM时反推最大bs（不要猜）：**
```
max_bs = (GPU_total - M_static) / (M_used + M_req - M_static) × bs_oom × 0.9
例: M_used=78.03, M_req=18.42, bs=24, GPU=94.97, static=1.5
  → (94.97-1.5)/(96.45-1.5)×24×0.9 = 21.2 → 取bs=20
```

### 启动前checklist（逐项确认，缺一不可）

1. `torch.compile(model, mode="reduce-overhead")` — **不是default**。**但多tau sweep必须每tau单独Python进程**（CUDA graphs跨run碎片化可达46GB，即使del model+empty_cache也无法清除）
2. 训练loop每步第一行: `torch.compiler.cudagraph_mark_step_begin()`
3. 脚本头: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
4. `--passkey_mix_ratio 0.0` — passkey混入导致动态shape破坏CUDA graphs；passkey可后续100步微调补
5. batch_size按上表，不要随意调整
6. 开GPU后**直接运行训练**，不要跑ls/nvidia-smi等诊断命令（浪费付费GPU时间）
7. 用`nohup ... > log 2>&1 &`后台跑，`tail -f log`查看进度

### 其他规则

- 多run间彻底释放显存: `del model,optimizer; gc.collect(); torch.cuda.empty_cache()`
- 结果存JSON+checkpoint，中间50%/75%必须保存，防中断丢失
- 先seed=42单run验证方向，确认后再多seed
- 第三方方法必须对照官方repo逐行确认（RIFLEx踩坑：漏0.9系数+L_train/L_test用错）
- eval结果违反常识（优化方法反而更差）→第一反应是检查bug，不是解读
- work_dir必须在数据盘`/root/autodl-tmp/`，不要放系统盘（30GB容易满）

---

## Part 1: 架构规范（任何偏离=结果不可比）

| Tier | Params | Layers | Heads | head_dim | Hidden | FFN |
|------|--------|--------|-------|----------|--------|-----|
| 125M | ~125M | 12 | 12 | 64 | 768 | 3072 |
| 350M | ~350M | 24 | 16 | 64 | 1024 | 4096 |
| 750M | ~750M | **18** | 24 | 64 | 1536 | 6144 |

Vocab=50304(所有tier)。750M是18层/24头，不是24层/16头。

训练超参: AdamW(β1=0.9,β2=0.95,wd=0.1), cosine→1e-5, bf16, base=500000
- 125M: lr=3e-4, bs=20(L=4096)
- 350M: lr=2e-4, bs=12(L=4096)
- 续训(continued pretrain): lr=1e-5, warmup=500

---

## Part 2: EVQ-Cosh 公式

```python
def evq_cosh_inv_freq(head_dim, tau, base=500000.0):
    K = head_dim // 2
    idx = torch.arange(K, dtype=torch.float64)  # 必须float64
    u = (idx + 0.5) / float(K)                  # midpoint，不是k/n
    if abs(tau) < 1e-8:
        phi = u
    else:
        phi = 1.0 - (1.0/tau) * torch.arcsinh((1.0-u) * math.sinh(tau))
    return torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi).float()
```

**τ* 经验值 (d_rope=32, base=500000, MLA):**
- L=4096 → τ*=**2.5** (MLA v2, K=32, base=10000, 50M sweep验证: 4K -3.7%, 8K -6.7%, 16K -14.5%。τ=1.414也赢但幅度小。EVQ正确pattern完全恢复)
- L=8192 → τ*=**1.414** （3-seed验证，PPL@16K -31.1%）
- L=2048 → τ*≈2.0 （待验证）

通用公式: `τ*(L,b,K) = max(1.20 + 0.45×x, 4/√K)`，`x = 1 - ln(L/2π)/ln(b)`
DiT修正: `τ*_DiT ≈ 0.53 × τ*_AR`

**4K模型 YaRN FT结果 (scale=2, seed=42):**

| Method | PPL@8K | PPL@16K |
|--------|-------:|--------:|
| GEO+YaRN+FT | 31.8 | 97.0 |
| **EVQ+YaRN+FT** | **31.0** | 115.3 |

EVQ+YaRN+FT@8K比GEO好**-2.5%**（超线性融合）。GEO在16K以上更优。

**YaRN FT配置（已验证）:**
- lr=2e-6（不是5e-5！5e-5灾难性遗忘）, warmup=50, 500步全参数
- 混合数据: 50%原始长度 + 50%目标长度（防遗忘，关键）
- batch=4(8K), batch=2(16K)；FT数据用rechunked训练数据，不用test_*.pt（太少）

**数据 (`/root/autodl-tmp/data/`):**

| 数据集 | seq_len | tokens | 备注 |
|--------|---------|--------|------|
| 1b_diverse_4k | 4096 | 1B | seed=42, buf=10k |
| 1b_diverse_4k_v2 | 4096 | 1B | seed=2024, buf=100k，非重叠 |
| 1b_diverse_4k_v3 | 4096 | 1B | seed=3000, buf=200k，准备中 |
| 8k_mixed/b/c | 8192 | 500M×3 | FineWeb/SlimPajama/OWT |
| train_750m_clean | 2048/4096 | 1.47B | FineWeb-Edu |

测试集: `data/train_750m_clean/test_{4096,8192,16384,32768,49152}.pt`

---

## Part 3: 禁令

1. YaRN必须用Progressive per-channel ramp，不用NTK-aware（破坏EVQ频率结构）
2. 不要在GEO checkpoint上直接换EVQ inv_freq，必须从头训练
3. τ=0.707用于我们自己454M模型(L=2048)是错的(≈GEO无差异)；LLaMA-3.2-1B(L=8192)的τ*=64/√8192=0.707才是对的
4. eval改seq_len必须同步YaRN scale=eval_len/train_len
5. DiT禁止跨run比较（CUDA非确定性造成70%+虚假差异），必须head-to-head同run
6. DiT用τ*_DiT≈0.53×K_t/√T_train，不要用AR公式（中频抽空导致崩坏）
7. passkey训练必须`--passkey_mix_ratio 0.0`+事后微调，不要混入主训练
8. 第三方方法必须对照官方repo（不要自己重写）

---

## Part 4: Wan2.1 (Phase 17，已降优先级)

详见 `scripts/video_temporal/wan21_raw_train.py`。关键坑：
- 用原始`WanModel`不用diffusers（服务器无diffusers格式）
- forward输入是List[Tensor]不是batch tensor
- 分辨率必须288×288（280不行，35×35 patch无法整除）
- LoRA targets: `"q","k","v","o"` 不是 `"to_q"...`
- bs=1@288×288=79.8GB，bs>1需gradient checkpointing

---

## 关键文件

`paper/main.tex` | `scripts/core_text_phases/run_gqa_evq_experiment.py` | `scripts/core_text_phases/run_evq_sweep.py` | `results/video_dit/REPORT_FINAL.md`
