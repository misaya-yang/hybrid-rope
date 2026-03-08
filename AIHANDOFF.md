# Project Handoff

## Repository Scope

This repository is now a submission-oriented EVQ-Cosh workspace. The goal is not to preserve every historical branch of exploration, but to keep a clean, defensible path from theory to experiments to anonymous paper draft.

## Current Topology

- `scripts/`: only paper-core experiment, evaluation, and figure code
- `docs/`: curated experiment records and theory derivation material
- `paper_draft/`: narrative, theory source of truth, matrix docs, and submission source
- `team/`: advisor and collaborator coordination material
- `results/`: classified outputs

## Recommended Reading Order

1. `README.md`
2. `paper_draft/mainstory.md`
3. `paper_draft/CORE_THEORY.md`
4. `paper_draft/figs/README.md`
5. `docs/exp/2026-03-04_phase11_L256_results.md`
6. `docs/exp/2026-03-03_passkey_mix_results.md`
7. `docs/exp/2026-03-06_phase15_750m_2k_to_4k_continue_results.md`

## Primary Evidence Hierarchy

### P0: main anchors
- `paper_draft/figs/fig2_evq_yarn_synergy.pdf`
- `paper_draft/figs/fig3_pe_dominant_scaling.pdf`
- `docs/exp/2026-03-03_passkey_mix_results.md`
- `docs/exp/2026-03-04_phase11_L256_results.md`
- `docs/exp/2026-03-05_phase11b_125m_results.md`

### P1: strong supporting evidence
- `results/core_text/phase15/`
- `results/core_text/phase9f_750m_2k_1b/`
- `results/core_text/phase14_yarn_passkey/`

### P2: secondary / scope-expanding evidence
- `results/supporting_video/video_temporal/`
- `results/supporting_cross_model/`
- `results/theory/`

## Submission Source

Anonymous draft source is here:

- `paper_draft/submission/main.tex`

The current draft should obey these structural rules:

- page 10 starts with `References`
- body keeps Figure 2 and Figure 3 as the two main figures
- Figure 1 stays supporting, not primary

## Current Narrative Lock

Do not drift away from these three claims unless new evidence justifies it:

1. Closed-form theory: RoPE frequency allocation is a variational inverse problem.
2. Extreme extrapolation: EVQ beats learnable PE in DAPE-style regimes.
3. Systems result: `EVQ + YaRN >> Geo + YaRN`.

## Paper Improvement Priorities

Optimize the paper under three principles:

1. **Protect the evidence hierarchy**
   - Keep `P0` as the real paper core: `EVQ+YaRN` and PE-dominant / DAPE-style extrapolation.
   - Keep `P1` as robustness support.
   - Keep `P2` as scope-expanding evidence only.
   - Do not promote single-seed or cross-modal results into body-level anchors without new multi-seed support.

2. **Protect the theorem / conjecture boundary**
   - Closed-form ODE solution and geometric limit are paper-grade theory.
   - `tau*` remains an empirical law / conjecture unless the derivation is tightened.
   - New mechanism ideas, including the capacity-compensation hypothesis, stay in `team/plans/` until they are experimentally closed.

3. **Optimize around the current submission skeleton, not around loose notes**
   - Submission source of truth: `paper_draft/submission/main.tex`
   - Submission structure and redlines: `paper_draft/NEURIPS_SUBMISSION_PLAN.md`
   - Core paper narrative: `paper_draft/mainstory.md`
   - Theory source of truth: `paper_draft/CORE_THEORY.md`
   - Figure / table matrix: `paper_draft/figs/README.md`
   - Known paper-side corrections and claim hygiene: `paper_draft/PAPER_ERROR_CORRECTIONS.md`
   - Missing evidence and collaborator-facing next steps: `team/open_gaps.md`, `team/plans/`

Practical rule: improve the paper by strengthening the current three-claim package, not by adding parallel storylines.

## What Has Been Deliberately Demoted

These are not headline claims in the current submission package:

- single-seed `+40pp` passkey outliers
- `Hybrid strict superiority`
- `video confirms tau*=2.0`
- `750M phase9f` as a primary result
- `750M continue` as a primary result

## Where To Put New Work

- New experiment report: `docs/exp/YYYY-MM-DD_<topic>.md`
- New theory derivation note: `docs/theory/`
- New submission-facing figure: `paper_draft/figs/`
- New advisor/collaboration note: `team/`
- New structured output bundle: `results/<bucket>/`

## Blackwell GPU (RTX 5090 / RTX 6000 Pro) Training Setup

RTX 5090 和 RTX 6000 Pro 都是 Blackwell 架构 (sm_120, compute capability 12.0)，训练配置完全相同。

### 环境要求

- **PyTorch >= 2.7.0**，推荐 2.8+
- **CUDA 12.8**（sm_120 必须 CUDA 12.8+）
- 验证：`python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"`
- 预期输出：`2.8.x+cu128 12.8 (12, 0)`

### torch.compile 配置（关键！）

Blackwell 上 **必须** 使用 `torch.compile` 才能获得正常 GPU 利用率。不用 compile，GPU 利用率只有 ~30%（kernel launch 开销），训练时间会膨胀 3-4 倍。

```python
model = GPT(cfg, inv_freq).to("cuda")
model = torch.compile(model, mode="max-autotune")  # 关键！
```

训练循环每步前加：
```python
torch.compiler.cudagraph_mark_step_begin()  # 防止 CUDA Graph 覆写错误
```

#### 常见坑

| 问题 | 原因 | 解决 |
|---|---|---|
| `mode="reduce-overhead"` crash | CUDA Graph 张量覆写冲突 | 改用 `mode="max-autotune"` 或 `mode="default"` |
| `mode="max-autotune"` backward crash | 缺少 `cudagraph_mark_step_begin()` | 每步 forward 前调用 `torch.compiler.cudagraph_mark_step_begin()` |
| Triton "OutOfMemoryError: out of resource" 警告 | 部分 triton kernel 配置超出 sm_120 寄存器限制 | 无害警告，autotune 会自动跳过这些配置，选择可用的 |
| compile warmup 慢（首次 2-3 分钟） | Triton autotune 为每个 kernel 搜索最优配置 | 正常现象，只在首步发生 |

### 性能参考（454M 模型, L=512, RTX 5090 32GB）

| 配置 | ms/step | tok/s | VRAM | ETA (2B tokens) |
|---|---|---|---|---|
| eager (无 compile) | 231ms | 44K | 25.1GB | 12.6h |
| `compile(mode="default")` | 165ms | 62K | 17.6GB | 8.9h |
| `compile(mode="max-autotune")` | 183ms | 56K | 20.9GB | **9.9h** |

注意：`max-autotune` 在 454M 模型上比 `default` 略慢，因为 autotune 的 triton kernel 在此模型尺寸上不一定优于 cuBLAS。两者都远优于 eager。实际选择可根据模型尺寸 benchmark 决定。

### 其他训练优化

- **bf16 不需要 GradScaler**：bf16 与 fp32 共享指数范围，GradScaler 只对 fp16 有意义，且会引入 CPU-GPU 同步开销
- **loss.item() 避免每步调用**：每 200 步调用一次，否则每步都触发 CPU-GPU 同步
- **clip_grad_norm_ 只在 warmup 期间使用**：之后跳过可省 ~55ms/step
- **数据预存为 int64**：避免每步 int32→int64 转换开销
- **passkey 混合数据缓存到磁盘**：`torch.save(mixed_data, "mixed_data_seed{seed}.pt")`，避免每次重启重新生成

### 经验教训

> **不要因为 torch.compile crash 就认定"硬件不支持"。** Blackwell (sm_120) 在 PyTorch 2.7+ / CUDA 12.8 下完全支持 compile。crash 通常是 mode 选择或 CUDA Graph 配置问题，换个 mode 或加一行 `cudagraph_mark_step_begin()` 就能解决。先排查再下结论。

## Practical Warning

This repo has already been intentionally pruned. Do not recreate old root-level clutter. If a new artifact does not clearly belong to one of the five visible top-level directories, the default answer is that it does not belong in this repository.
