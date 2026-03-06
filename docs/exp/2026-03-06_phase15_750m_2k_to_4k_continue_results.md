# Phase15: 750M 2K->4K Continue Results (Geo vs Full EVQ r=0)

> 日期: 2026-03-06
> 状态: **COMPLETE**
> 服务器: R6000
> 实验: `phase15_750m_2k_to_4k_continue_ckpt_eval`
> 本地镜像: `results/core_text/phase15/*.json`

---

## 0. 实验设置

| 参数 | 值 |
|------|-----|
| 模型 | 750M (H=1536, L=18, heads=24, d_head=64, FFN=6144) |
| 初始化 ckpt | `phase9 geo_750m_2k_1bdata_ckpt step_15258.pt` |
| continue 长度 | `2048 -> 4096` |
| continue 数据量 | `500M tokens` |
| Base | `500,000` |
| Seed | `42` |
| 对比 | `Geo` vs `EVQ tau=1.5, r=0` |
| mix 配置 | `passkey_mix=3%`, `downstream_mix=10%` |
| batch | effective `14` (`micro=7`, `grad_accum=2`) |
| lr | `2e-4` |
| EVQ 训练时长 | `15322.4s = 255.4 min` |

**导回本地的关键 JSON**

- `results/core_text/phase15/phase15_evq_r0_seed42_result.json`
- `results/core_text/phase15/phase15_evq_r0_seed42_checkpoint_eval_progress.json`
- `results/core_text/phase15/phase15_geo_seed42_result.json`
- `results/core_text/phase15/phase15_continue_summary.json`

---

## 1. Final Head-to-Head

### 1.1 PPL

| 长度 | Geo | EVQ r=0 | Delta |
|------|-----|---------|-------|
| 2K | 25.922 | 26.160 | `+0.9%` |
| 4K | 21.955 | 22.282 | `+1.5%` |
| 8K | 23.386 | 19.607 | `-16.2%` |
| 16K | 45.136 | 24.407 | `-45.9%` |

**结论**：
- continue 到 4K 后，EVQ 在 `2K/4K` 仍有轻微代价
- 但在真正长程段，优势非常大，尤其 `16K -45.9%`

### 1.2 Passkey (40 trials final eval)

| 长度 | Geo ret / AR | EVQ ret / AR | Delta |
|------|--------------|--------------|-------|
| 2K | `100% / 100%` | `100% / 100%` | — |
| 4K | `100% / 100%` | `100% / 100%` | — |
| 8K | `100% / 0%` | `100% / 77.5%` | `AR +77.5pp` |
| Global | `100% / 66.67%` | `100% / 92.5%` | `AR +25.8pp` |

**结论**：
- retrieval 在这条设置里已经全部饱和到 `100%`
- 真正拉开的是 **AR exact match**
- EVQ 在 `8K` 上把 `AR exact` 从 `0%` 拉到 `77.5%`

### 1.3 Multi-needle (12 trials final eval)

| 长度 | Geo per/all | EVQ per/all | Delta |
|------|-------------|-------------|-------|
| 4K | `100.0% / 100.0%` | `96.7% / 83.3%` | Geo 略优 |
| 8K | `70.0% / 16.7%` | `78.3% / 25.0%` | EVQ `+8.3pp / +8.3pp` |
| Global | `85.0% / 58.3%` | `87.5% / 54.2%` | mixed |

**结论**：
- 4K 上 Geo 略优
- 8K 上 EVQ 更好
- 当前 multi-needle 不是这轮最强 headline，仍以 supporting evidence 为宜

### 1.4 Downstream NLL

- **Geo**: LongBench NLL 正常跑出，aggregate `mean_nll=3.3983`, `ppl_from_nll=29.91`
- **EVQ**: `qasper / hotpotqa / 2wikimqa / narrativeqa / multifieldqa_en / musique` 全部因为远端数据下载失败而报错

**关键 caveat**：
- 这不是 EVQ 模型崩了
- 是远端 `LongBench` 数据拉取失败，导致 **没有形成公平 downstream 对比**
- 下一轮前必须先把这批数据预置到服务器

---

## 2. EVQ 训练轨迹 (50% / 75% / 100%)

### 2.1 PPL 轨迹

| Checkpoint | 2K | 4K | 8K | 16K |
|------------|----|----|----|-----|
| 50% | 28.847 | 24.919 | 21.686 | 25.577 |
| 75% | 26.895 | 23.129 | 20.269 | 24.586 |
| 100% | 26.160 | 22.282 | 19.607 | 24.407 |

**结论**：
- EVQ 在 `50% -> 75% -> 100%` 全程单调改善
- 其中 `16K` 从 `25.577 -> 24.407`，稳定继续下降

### 2.2 Passkey / RULER 轨迹

| Checkpoint | Global AR | 8K AR | 4K multi-needle all | 8K multi-needle all |
|------------|-----------|-------|---------------------|---------------------|
| 50% | 98.33% | 95% | — | — |
| 75% | 98.33% | 95% | 75.0% | 37.5% |
| 100% ckpt eval | 91.67% | 75% | 100.0% | 25.0% |
| 100% final eval | 92.5% | 77.5% | 83.3% | 25.0% |

**解读**：
- PPL 继续改善到 100%
- passkey AR 在更严格的 40-trial final eval 下仍然很强
- RULER 的 8K all-needle 依然偏难，不适合在这轮做主 claim

---

## 3. 这轮实验最重要的结论

### 3.1 continue 到 4K 后，EVQ 的长程优势没有消失

最关键的对比不是 `2K/4K`，而是：
- `8K: 23.386 -> 19.607` (`-16.2%`)
- `16K: 45.136 -> 24.407` (`-45.9%`)

这说明 continue 到更长训练长度后，EVQ 的收益仍然集中在真正长程段，而不是被 4K 训练“洗掉”。

### 3.2 retrieval 已饱和，AR exact 变成更有区分度的指标

这轮 `Geo` 和 `EVQ` 的 retrieval 都是 `100%`，因此最有信息量的是：
- `8K AR exact: 0% vs 77.5%`
- `Global AR exact: 66.67% vs 92.5%`

也就是说，EVQ 不只是“找到 passkey”，而是显著提高了精确复现能力。

### 3.3 这轮可以支撑的主叙事

- `750M` 最大规模 continue setting 下，EVQ 仍然表现出 **短程微亏、长程大赢**
- 在 passkey retrieval 饱和后，EVQ 的优势主要体现在 **AR exact / long-range precision**
- 这轮和 earlier `750M phase9f` 一起，强化了 “模型变强后，tradeoff 更集中于真正长程段” 的故事

---

## 4. 下一轮加训准备

### 4.1 应保留的 checkpoint

**EVQ 下一轮 resume 基座**
- 远端路径：
  - `/root/autodl-tmp/evq_phase15_750m_2k_to_4k/seed42/evq1.5_r0_750m_2k_to_4k_continue/checkpoints/step_08719.pt`

**Geo 对照保留**
- 远端路径：
  - `/root/autodl-tmp/evq_phase15_750m_2k_to_4k/seed42/geo_750m_2k_to_4k_continue/checkpoints/step_07629.pt`

### 4.2 可删除的中间 checkpoint

**EVQ**
- `step_04359.pt`
- `step_06539.pt`

**Geo**
- `step_03814.pt`
- `step_05721.pt`

### 4.3 下一轮前的必要修复

1. 预置 `LongBench` 数据，避免 EVQ downstream eval 再次因远端下载失败而空跑
2. 明确下一轮是：
   - 继续拉长训练长度
   - 还是保持 4K 训练长度、只增加 token
3. 保留 `result.json`、`checkpoint_eval_progress.json`、summary json，不删

---

## 5. 建议如何进入主文叙事

这轮结果最适合的表述是：

- **750M continued-pretraining @4K confirms the same qualitative pattern**:
  EVQ pays a very small cost at `2K/4K`, but substantially improves true long-range extrapolation, with `PPL@16K -45.9%` and `8K AR exact 0% -> 77.5%`.

不建议把这轮写成：
- “EVQ retrieval 更强”  
因为 retrieval 在这轮已经 ceiling 了，真正区分度来自 `AR exact` 和 `16K PPL`。
