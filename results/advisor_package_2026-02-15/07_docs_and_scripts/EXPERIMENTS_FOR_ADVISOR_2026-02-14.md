# Hybrid-RoPE 实验全量整理（给导师）

- 报告日期：2026-02-14
- 仓库：`e:/rope/hybrid-rope`
- 整理口径：基于当前仓库内脚本 + 已同步结果文件；未落地到本地的实验单独标注为“待同步/进行中”

## 1. 实验总览（按主线分组）

| 主线 | 状态 | 代表结论 | 证据路径 |
|---|---|---|---|
| A100 from-scratch（50M/100M/350M） | 已完成 | `hybrid_a0.2_t100k` 在 16K 外推稳定优于 `geo_500k` | `results/unified_search_3cfg_3seed/results.json`, `artifacts/a100_2026-02-13/data/100m_scaling/results.json`, `results/350m_final/results.json` |
| 50M 频谱搜索（A100+A800） | 已完成 | 单次搜索最佳从 `geo_500k` 转向 `hybrid/anchpoly` | `results/unified_search/results_A.json`, `artifacts/a800_2026-02-13/results/unified_search/results_B.json` |
| LLaMA 长上下文推理侧实验（8B/13B） | 已完成 | 几何 RoPE 在长上下文存在崩溃边界；sigmoid/anchored 可显著延后或缓解 | `results/llama_shape_theta_min/results.json`, `results/llama13b_triangle_boundary/results.json`, `results/anchored_sigmoid_v3_followup/ANCHORED_SIGMOID_V3_SUMMARY.md` |
| A800 Llama-3-8B Hybrid-LoRA | 已完成 | 16K 上 `hybrid_lora` 相对未微调 base 显著改进 | `artifacts/a800_2026-02-13/A800_LLAMA3_HYBRID_LORA_EVAL_SUMMARY_2026-02-13.md` |
| A800 机制验证（P0/P1/P2 + 3h follow-up） | 已完成 | 频谱与 LoRA 耦合是关键，单独改其中一项不足以稳定 16K | `artifacts/a800_2026-02-13/results/mechanism_p1/summary.md`, `artifacts/a800_2026-02-13/results/mechanism_p2_framework/summary.md` |
| R6000 Qwen Hybrid-LoRA + eval suite | 已完成（结果已落地） | LoRA 后 PPL 上升，但 Passkey/KV 检索保持高准确率 | `results/qwen_hybrid_lora/summary.json`, `results/qwen_hybrid_lora/eval_suite.json` |
| Qwen 即插即用频谱对比（wikitext） | 已完成 | Qwen 原生配置最稳；人为 hybrid 在该协议下退化明显 | `results/qwen_plugandplay_wikitext_v1/results.json` |
| 导师 follow-up 扩展夜跑（9h extended） | 本地未同步 | 脚本已就绪，结果目录空 | `scripts/run_night_run_9h_extended.py`, `results/night_run_9h_extended/` |
| 700M 训练期频率对比 | 脚本已完成，数据已同步（data-only） | 已有训练/评估脚本与远程启动器 | `scripts/run_train_freq_comparison.py`, `scripts/train_700m_wikitext.py`, `scripts/train_700m_local.py`, `scripts/eval_trained_700m.py`, `tools/remote_legacy/run_train_freq_comparison.py` |

## 2. 关键已完成实验与结果

### 2.1 A100 主线：从 50M 到 350M

#### A. 50M 单次 unified_search（跨机 split）

- A100 split A 最优：`anchpoly_p3.9_omf0.3_t500k`，`PPL@16384=16.459`
- A800 split B 最优：`hybrid_a0.2_t100k`，`PPL@16384=16.316`
- 对比基线 `geo_500k_ALIGN`：
  - A100：16.991
  - A800：17.947

证据：
- `results/unified_search/results_A.json`
- `artifacts/a800_2026-02-13/results/unified_search/results_B.json`

#### B. 50M 3配置×3seed 稳健性（论文主证据）

| Config | PPL@2048 (mean±std) | PPL@16384 (mean±std) |
|---|---:|---:|
| geo_500k | 6.826 ± 0.048 | 18.207 ± 0.768 |
| hybrid_a0.2_t100k | 6.699 ± 0.168 | **17.324 ± 0.360** |
| anchpoly_p3.9_omf0.3_t500k | 6.634 ± 0.192 | 19.133 ± 1.135 |

证据：`results/unified_search_3cfg_3seed/results.json`

#### C. 50M 2配置×10seed

- `geo_500k`：`PPL@16384 = 18.798 ± 1.643`
- `hybrid_a0.2_t100k`：`PPL@16384 = 17.777 ± 2.824`
- 结论：均值仍优于 geo，但 hybrid 方差更大。

证据：`artifacts/a100_2026-02-13/data/unified_search_2cfg_10seed/results.json`

#### D. 50M theta/freq 公平因子实验（6配置×3seed，16K）

| Config | PPL@16384 (mean±std) |
|---|---:|
| geo_100k | 11.904 ± 0.473 |
| geo_200k | 18.524 ± 3.298 |
| geo_300k | 13.797 ± 1.127 |
| geo_500k | 13.757 ± 0.348 |
| hybrid_a0.2_t100k | 13.508 ± 1.100 |
| hybrid_a0.2_t500k | **13.495 ± 1.005** |

证据：`artifacts/a100_2026-02-13/data/50m_theta_factorial/results.json`

#### E. 50M YaRN 对照（修正版）

| Length | Hybrid native | Geo native | Geo + YaRN progressive |
|---|---:|---:|---:|
| 2048 | 6.672 | 6.839 | 6.839 |
| 4096 | 6.748 | 7.045 | 8.640 |
| 8192 | 8.688 | 8.833 | 16.899 |
| 12288 | 13.333 | 13.588 | 29.352 |
| 16384 | **16.861** | 17.966 | 39.479 |

证据：`results/50m_yarn_compare_v2/results.json`

#### F. 100M 扩展

- `geo_500k`：`PPL@16384 = 10.888`
- `hybrid_a0.2_t100k`：`PPL@16384 = 9.417`
- 相对下降：约 13.5%

证据：`artifacts/a100_2026-02-13/data/100m_scaling/results.json`

#### G. 350M 最终验证（500M tokens 训练）

- `geo_500k`：`PPL@16384 = 14.653 (std 3.851, n=10)`
- `hybrid_a0.2_t100k`：`PPL@16384 = 12.646 (std 3.093, n=10)`
- 相对下降：约 13.7%

证据：`results/350m_final/results.json`

### 2.2 LLaMA 推理侧（eval-only）实验

#### A. 我们的方法（位置压缩）vs baseline（长长度）

在 `results/our_method_comparison/results.json` 中：
- 40K：baseline 37.928；`cf2.0` 为 24.714
- 49K：baseline 69.794；`cf2.0` 为 41.860
- `cf2.5/cf3.0` 在更长长度出现 OOM

证据：`results/our_method_comparison/results.json`

#### B. LLaMA shape vs theta（最小对照）

| Config | PPL@2K | PPL@16K | Collapse ratio |
|---|---:|---:|---:|
| geo_10k | 518.013 | 11214.916 | 22.026x |
| sigmoid_t100k | 10.372 | 12.510 | **1.077x** |

证据：`results/llama_shape_theta_min/results.json`

#### C. Cross-model WikiText（LLaMA + Qwen）

| Model | PPL@2K | PPL@16K | 16K/2K |
|---|---:|---:|---:|
| llama_geo_10k | 549.85 | 12111.06 | 22.03x |
| llama_sigmoid_best_t100k | 11.67 | 12.57 | 1.08x |
| qwen_orig_theta | 8.46 | 6.98 | 0.82x |
| qwen_geo_100k | 8.58 | 7.16 | 0.83x |

证据：`results/cross_model_wikitext_v1/results.json`

#### D. LLaMA triangle boundary（geo_500k / geo_2M / anchored_x20）

- `geo_500k`：边界约在 16K（`PPL@16K=262.278`）
- `geo_2M`：16K 最优（`PPL@16K=9.264`），但更长长度仍上升
- `anchored_x20`：16K 明显优于 geo_500k（19.041 vs 262.278），边界后移到约 24K

证据：`results/llama13b_triangle_boundary/results.json`

#### E. Anchored Sigmoid v3 follow-up

- `geo_500k`：`PPL@2k=10.05`, `PPL@16k=194.96`, collapse `19.40x`
- `anchored_x10`：`PPL@2k=10.01`, `PPL@16k=19.65`, collapse `1.96x`
- anchor_factor 扫描：`x5=246.26`, `x10=25.12`, `x20=9.28`（16K）

证据：
- `results/anchored_sigmoid_v3_followup/ANCHORED_SIGMOID_V3_SUMMARY.md`
- `results/anchored_sigmoid_v3_followup/exp1_results.json`
- `results/anchored_sigmoid_v3_followup/exp3_results.json`

#### F. Night run（anchored_x20, 9h包）

本地 `results/night_run_anchored_x20_9h/results.json` 为空，但 `summary.md` 已给出核心结论：
- 16K：`geo_500k=262.278`, `geo_1M=25.475`, `geo_2M=9.264`, `anchored_x20=19.041`
- anchored_x20 相比 geo_500k 显著延后崩溃。

证据：`results/night_run_anchored_x20_9h/summary.md`

### 2.3 A800：Llama-3-8B Hybrid-LoRA 与机制线

#### A. Hybrid-LoRA 主实验

- 训练：`seq=8192`, `max_steps=600`, `C4-en`, 用时 3.883h
- 评测（base_unfinetuned vs hybrid_lora）：
  - 2K：15.889 vs 15.352
  - 8K：13.633 vs 13.506
  - 16K：190.566 vs **15.400**（相对下降 91.92%）

证据：`artifacts/a800_2026-02-13/A800_LLAMA3_HYBRID_LORA_EVAL_SUMMARY_2026-02-13.md`

#### B. 机制 P1（2x2 因子）

16K（sequential）关键值：
- M00(base_orig) = 190.566
- M10(base_hybridfreq) = 7612.143
- M01(lora_origfreq) = 231.308
- M11(lora_hybridfreq) = **15.400**

结论：频谱与 LoRA 需要耦合，单独改其中之一不足以稳定长上下文。

证据：`artifacts/a800_2026-02-13/results/mechanism_p1/summary.md`

#### C. 3h follow-up（重训+复评）

16K 排名（random_start）：
1. `sigmoid_th100k_steep8_mid0.5_omf0.3` = 25.847
2. `sigmoid_th500k_steep8_mid0.5_omf0.3` = 26.116
3. `geo_500k` = 27.217
6. `geo_10k_baseline` = 76.989

证据：`artifacts/a800_2026-02-13/results/h800_3h_followup/summary.md`

#### D. Poly follow-up

- `poly_th500k_p3.9_omf0.3`：`PPL@16k=31.231`
- `poly_th100k_p3.9_omf0.3`：`PPL@16k=35.146`

证据：`artifacts/a800_2026-02-13/results/h800_3h_poly_followup/summary.md`

### 2.4 R6000：Qwen Hybrid-LoRA 与 Qwen 插件式对照

#### A. Qwen Hybrid-LoRA 训练与评测

训练摘要：
- 模型：Qwen2.5-7B-Instruct
- 数据：TinyStories，目标 40M tokens
- `max_steps=500`, `seq=8192`, 训练时长约 2.101h

Eval suite：
- Base PPL：8K 3.1369；16K 3.0642；24K 3.0404；32K 3.2449
- Hybrid-LoRA PPL：8K 8.8666；16K 9.6371；24K 11.0923；32K 10.4202
- Passkey MC：base 与 hybrid_lora 在 8K/16K/24K/32K 全部 1.0
- KV MC：两者总体高，个别长度为 0.8333

结论：LoRA 后语言建模 PPL 变差，但检索任务保持高准确率。

证据：
- `results/qwen_hybrid_lora/summary.json`
- `results/qwen_hybrid_lora/eval_suite.json`

#### B. Qwen plug-and-play（wikitext）

- `qwen_orig`：32K/2K 比值 0.777（稳定）
- `qwen_geo_100k`：32K/2K 比值 1.166（高长度退化）
- `qwen_sigmoid_best_t100k`：整体 PPL 高于 orig
- `qwen_hybrid_a0.2_t100k`：明显崩溃（32K/2K 比值约 17.59）
- `qwen_yarn8`：该次运行加载失败（错误见结果文件）

证据：`results/qwen_plugandplay_wikitext_v1/results.json`

## 3. 目前最稳结论（可直接汇报）

1. 在 from-scratch 路线（50M→100M→350M）下，`hybrid_a0.2_t100k` 对 `geo_500k` 的 16K 优势是可复现的。
2. 在 LLaMA 推理侧，纯 geometric 在长长度上出现明显崩溃边界；sigmoid/anchored 形状可显著降低 collapse ratio。
3. 在 A800 LoRA 路线，16K 上 `hybrid_lora` 相比未微调 base 有数量级提升（190.566 → 15.400）。
4. 在 Qwen 路线，hybrid-LoRA 目前呈现“检索保持、PPL上升”的 trade-off，尚不足以作为“全面优于 base”的结论。

## 4. 当前缺口与风险（导师可能追问）

1. 700M 训练期频率实验目前只有脚本，仓库内无对应 `results/train_freq_comparison` 与 `results/train_700m_*` 落地结果。
2. `results/night_run_9h_extended/` 与 `results/advisor_followup_2026-02-14/` 为空，本地尚未同步扩展夜跑结果。
3. 个别实验目录中存在空结果文件（如 `results/night_run_anchored_x20_9h/results.json`），当前以 `summary.md` 为准。
4. Qwen 路线在“PPL不升反降且检索不掉”的目标上仍未闭环，需要进一步协议统一和参数回扫。

## 5. 代码进展（你当前正在推进）

以下脚本已具备直接开跑能力，覆盖“训练期频率对比 + 700M训练 + 评估”：

- 训练期频率总脚本：`scripts/run_train_freq_comparison.py`
- 远程启动器（历史归档）：`tools/remote_legacy/run_train_freq_comparison.py`
- 700M 真实语料训练：`scripts/train_700m_wikitext.py`
- 700M 离线本地训练：`scripts/train_700m_local.py`
- 训练后评估：`scripts/eval_trained_700m.py`

建议下一次同步优先拉回：
- `results/train_freq_comparison/*/results.json`
- `results/train_700m_wikitext/*/results.json`
- `results/train_700m_local/*/results.json`
- `results/eval_700m/*/results.json`

---

如果导师希望“只看最强证据链”，优先阅读这 4 个文件：

1. `results/unified_search_3cfg_3seed/results.json`
2. `results/350m_final/results.json`
3. `artifacts/a800_2026-02-13/A800_LLAMA3_HYBRID_LORA_EVAL_SUMMARY_2026-02-13.md`
4. `results/llama_shape_theta_min/results.json`
