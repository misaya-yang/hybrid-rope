# Minimal Rebuttal Experiment Runbook

日期：2026-06-10

用途：把 rebuttal 阶段“还能补什么”压缩成一个最小、可执行、可停止的实验队列。核心原则是：

- rebuttal 不是二次提交论文；
- 只做能直接回答 reviewer 问题的补证据；
- 没有 exact numbers、seed scope、日志/结果路径的结果不得写成结论；
- supporting/exploratory 证据不得升级成 primary claim；
- 如果一个实验会迫使我们重写主叙事，它优先级自动下降。

当前默认状态：

- Path A：若拿到 exact Geo+LoRA numbers，可把 LoRA 作为强控制证据写入 rebuttal。
- Path B：若拿不到 exact Geo+LoRA numbers，LoRA 只能保留为 post-hoc/supporting observation；最终 response 使用 `AUTHOR_RESPONSE_PATH_B_COMPACT.md` 或 `AUTHOR_RESPONSE_PATH_B_READY_DRAFT.md`。

## 0. Global Stop Rules

### 0.1 任何实验进入 rebuttal 前必须有

| 字段 | 必填原因 |
| --- | --- |
| command | 让 reviewer/AC 相信不是口头补丁 |
| git commit / working tree note | 防止脚本版本漂移 |
| checkpoint identity | 避免把不同 stage/seed 混在一起 |
| data identity | 尤其 LoRA LongAlign、FineWeb-Edu、passkey mix |
| seed list | 单 seed 必须明确标 single-seed |
| metric definition | PK 必须区分 TF NLL-gap 与 AR exact |
| raw result path | 方便最终表格追溯 |
| response sentence | 先写“这条结果允许我们说什么”，防止过度解释 |

### 0.2 立即停止的情况

| 情况 | 停止动作 |
| --- | --- |
| 结果需要解释成一个新故事才能有用 | 不进 rebuttal，只进 future work/limitation |
| 新结果反转主 claim | 不硬拗，收缩 claim |
| 只能报趋势、没有 exact number | 不写正式 response |
| 只能靠 broad benchmark 加分 | 停止，rebuttal 不做榜单扩张 |
| 需要从 scratch 跑多天才能完成 | 除非已经在跑，否则不作为 rebuttal 必需项 |

### 0.3 结果文件位置建议

为了不污染 reviewer supplement 和当前 repo，不建议把临时实验输出写进 repo 根目录或现有 `results/`。如果必须跑：

```bash
export EVQ_REBUTTAL_RUN_DIR=/tmp/evq-cosh-rebuttal-runs
mkdir -p "$EVQ_REBUTTAL_RUN_DIR"
```

LoRA 脚本可用环境变量改输出：

```bash
export EVQ_LORA_RESULT_DIR="$EVQ_REBUTTAL_RUN_DIR/lora_results"
export EVQ_LORA_CKPT="$EVQ_REBUTTAL_RUN_DIR/checkpoints/evq_r64_tau1414"
```

注意：`scripts/2026-04/*.sh` 多数写死远端训练路径，不能在本机直接当作可复现实验入口，只能当作 protocol reference。

## 1. P0: No-GPU / Result-Assembly Work

这些是 rebuttal 必做项，因为它们不依赖新训练，主要解决 trust、scope、provenance。

### P0.1 Geo+LoRA Exact Number Intake

**问题对应**：R2 的 LoRA confound、undertraining / insufficient-training 质疑。

**当前状态**：当前 allowed-scope 搜索只找到脚本和计划，没有找到可直接写入论文/response 的 Geo+LoRA exact result table。

**使用文档**：

- `rebuttal/TABLE23_LORA_WORKSHEET.md`
- `rebuttal/AUTHOR_RESPONSE_PACKET.md`
- `rebuttal/AUTHOR_RESPONSE_PATH_B_COMPACT.md`

**如果用户提供结果，必须填入**：

| Row | 8K | 16K | 32K | seed | checkpoint | note |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Base | 待填 | 待填 | 待填 | 待填 | 待填 | no adapter |
| Geo+LoRA | 待填 | 待填 | 待填 | 待填 | 待填 | same data/rank/steps as EVQ |
| EVQ-LoRA | 待填 | 待填 | 待填 | 待填 | 待填 | same data/rank/steps as Geo |

**允许写的 response**：

- 若 Geo+LoRA 无外推增益而 EVQ-LoRA 有：可以说 matched LoRA/LongAlign control attributes the extrapolation gain to frequency injection rather than LoRA adaptation alone。
- 若 Geo+LoRA 同样改善：LoRA 不能作为 EVQ-specific attribution，只能说 adaptation itself contributes substantially。
- 若 8K in-range cost 主要来自 LoRA：可以分解 adaptation cost 与 EVQ incremental cost，但必须给 exact numbers。

**禁止写**：

- “LoRA proves industrial-scale validation.”
- “EVQ-LoRA closes undertraining concern.”
- “+30% cost is EVQ-specific.”

除非 Base / Geo+LoRA / EVQ-LoRA 三行齐全，否则这些说法会反噬。

### P0.2 Figure 8 / Table 21 Consistency

**问题对应**：trust / paper hygiene。

**当前状态**：已修复。旧图是 accuracy bars，但 caption/table 是 Gold-answer NLL；现在 figure 已 regenerated 为 NLL plot。

**相关文件**：

- `rebuttal/FIGURE_TABLE_AUDIT.md`
- `scripts/figures/fig5_downstream_qa_nll.tex`
- `scripts/figures/build_fig5_downstream_qa.sh`
- `paper/figs/fig5_downstream_qa.pdf`
- `paper/figs/fig5_downstream_qa.png`
- `paper/main.pdf`

**允许写的 response**：

- “We thank the reviewer for catching this stale/mislabeled figure. We have replaced the panel with the Gold-answer NLL plot matching Table 21.”

**禁止写**：

- “The reviewer misread Figure 8.”

这是我们自己的图表一致性问题，应该主动承认。

### P0.3 Primary Provenance / Token Budget

**问题对应**：R2 对训练强度、seed、protocol 混淆的攻击点。

**当前状态**：已整理。

**相关文件**：

- `rebuttal/PRIMARY_PROVENANCE_NOTE.md`
- `paper/appendix/a2_experiment_details.tex`

**当前可写事实**：

| Evidence | Protocol |
| --- | --- |
| Primary I EVQ x YaRN | 454M, `L_train=2048`, 100M tokens, seeds 42/123/7, FineWeb-Edu + 10% passkey mix, fixed YaRN scale 8 |
| Primary II DAPE-style | 125M, `L_train=128`, 15M tokens, FineWeb-Edu; Geo/DAPE/EVQ seed 42; learnable tau seeds 42/137/256 |
| Phase 11B support | separate 125M, `L_train=256`, 100M protocol; do not mix with Primary II |
| Primary III MLA | 432M MLA, `L_train=8192`, 500M tokens, seeds 42/43/88, d_rope 32/base 500K |

**允许写的 response**：

- token/seed/protocol were underreported; we will add a compact reproducibility table.
- Primary II is seed-scoped diagnostic evidence, not broad dominance over all learned PE methods.

**禁止写**：

- “Primary II is 3-seed for every method.”
- “Phase 11B validates Table 4 directly.”

### P0.4 1B MLA Label Fix

**问题对应**：R2 会抓 “robustness to training saturation” 与 1B reversal 的矛盾。

**当前状态**：已把相关 wording 改为 schedule-sensitivity check / limitation。

**相关文件**：

- `paper/tables/table_evidence_tier.tex`
- `paper/sections/05_experiments.tex`
- `paper/appendix/a4_supporting_experiments.tex`
- `rebuttal/PAPER_ISSUE_AUDIT.md`

**允许写的 response**：

- the 1B MLA continuation is evidence of schedule sensitivity under scarce rotary channels, not robustness to saturation.
- we will relabel this row and discuss it as a limitation.

**禁止写**：

- “The 1B run proves training saturation robustness.”
- “The reversal is just noise.”

## 2. P1: Eval-Only / Low-Risk Experiments

P1 的共同特点：不训练新模型，只评估已有 checkpoint 或已有 adapter。它们是 rebuttal 最合适的补实验类型。

### P1.1 LoRA Training-Free Scaler Reference

**问题对应**：R2 可能说 “raw extrapolation baseline is too weak; LLaMA context extension defaults to training-free scaling such as Dynamic NTK/YaRN.”

**目标**：在 LoRA 板块给出 Base / Geo+LoRA / Geo+YaRN or Geo+Dynamic-NTK / EVQ-LoRA 的 eval-only reference。

**可用入口**：

- `experiments/lora_evq_v2/eval_evq_lora.py`
- `experiments/lora_evq_v2/eval_positional_ppl.py`
- `experiments/lora_evq_v2/run_pe_comparison.sh`

**脚本能力核对**：

| Script | 能做 | 不能默认声称 |
| --- | --- | --- |
| `eval_evq_lora.py` | PPL at `8192,16384,32768`; passkey generation; LongBench subset | 不自动构成 matched Geo+LoRA unless adapter_dir/method 对齐 |
| `eval_positional_ppl.py` | 16K WikiText positional windows for base/geo/evq/yarn | 默认只分到 16K；不能当 32K evidence |
| `run_pe_comparison.sh` | 3 methods x 3 seeds LoRA train/eval protocol | 写死远端训练路径；训练 9 runs，非最小 rebuttal |

**最小命令模板**：

```bash
cd experiments/lora_evq_v2

python eval_evq_lora.py \
  --model_name "$EVQ_LORA_MODEL" \
  --adapter_dir "$GEO_LORA_CKPT" \
  --output_dir "$EVQ_REBUTTAL_RUN_DIR/lora_geo_eval" \
  --ppl_lengths "8192,16384,32768" \
  --passkey_lengths "8192,16384,32768" \
  --ppl_data_path "$EVQ_LORA_WIKITEXT"
```

如果要评估 YaRN，需要确认 load-time `rope_scaling` 与 adapter 方法没有冲突。`eval_positional_ppl.py --method yarn --yarn_factor 2.0` 是 16K 分窗入口；32K 结论优先使用 `eval_evq_lora.py` 或专门 patch 后的 PPL 脚本。

**决策规则**：

| 结果 | Response 更新 |
| --- | --- |
| Geo+LoRA 无外推增益，Geo+YaRN/NTK 也弱，EVQ-LoRA 强 | Path A 强化；LoRA 板块基本闭环 |
| Geo+YaRN/NTK 接近 EVQ-LoRA | 不说 EVQ dominates default scaler；写 EVQ is competitive/supporting |
| Geo+LoRA 本身强 | LoRA 不能作为 EVQ-specific attribution |
| EVQ-LoRA 只赢 raw base | 不够；不要把 LoRA 写成最硬证据 |

### P1.2 Primary I Geo+YaRN Scale Sweep

**问题对应**：R2 可能攻击 fixed YaRN scale `s=8` 不是 tuned baseline。

**目标**：确认 Geo+YaRN 在 scale sweep 后是否仍不能解释 EVQ+YaRN 的收益。

**可用入口**：

- `scripts/core_text_phases/phase11_yarn_eval.py`
- `scripts/core_text_phases/eval_phase17h_yarn_strict.py`

**脚本能力核对**：

| Script | Protocol | 风险 |
| --- | --- | --- |
| `phase11_yarn_eval.py` | Phase 11 `L_train=256`, methods `geo,evq2.0,evq4.0`, scales `[1,2,4,8,16,32]` | 不是 Primary I 454M `L_train=2048`；不能直接替代 Table 2 |
| `eval_phase17h_yarn_strict.py` | strict FineWeb-Edu raw + YaRN auto eval for Phase17H checkpoints | hardcoded remote paths; one specific seed/checkpoint |

**最小可写结果**：

- 如果 scale sweep 只在 supporting protocol 上完成，只能写 “in a supporting scale-sweep, the advantage is not explained by a single mistuned fixed scale”，不能写成 Primary I 完全 tuned baseline。
- 如果能在 Primary I checkpoints 上复用同一 eval logic，才可以写 “we added a matched scale-sweep for Primary I.”

**决策规则**：

| 结果 | Response 更新 |
| --- | --- |
| best Geo+YaRN 仍弱于 EVQ+YaRN | 回应 “fixed-scale not obviously mistuned” |
| best Geo+YaRN 接近或超过 EVQ+YaRN | 收缩为 “EVQ and YaRN are complementary under matched scale, but tuned Geo+YaRN can close this specific gap” |
| 只跑了 non-primary protocol | 标为 supporting sanity check |

**禁止写**：

- “EVQ beats tuned YaRN” unless tuned Primary I result exists.

### P1.3 Passkey AR Exact Check

**问题对应**：R2 可能把 teacher-forced NLL-gap PK 当成 retrieval benchmark，或要求 autoregressive exact match。

**当前 metric 口径**：

- Primary PK = teacher-forced NLL-gap retrieval diagnostic。
- AR exact 只能单独报告，不得混入 Primary PK。

**可用入口**：

- `scripts/core_text_phases/eval_passkey.py`
- `scripts/supporting_eval/eval_passkey_scratch.py`

**脚本能力核对**：

| Script | 能做 | 不能说 |
| --- | --- | --- |
| `eval_passkey.py` | sweep checkpoints and write `passkey_results.json`; current main path is TF-style passkey eval | 不能直接声称 AR exact |
| `eval_passkey_scratch.py` | explicitly implements TF NLL-gap and autoregressive exact-match; writes `passkey_nll_<run_id>.json` | 需要 checkpoint compatibility and exact work_dir |

**命令模板**：

```bash
python scripts/supporting_eval/eval_passkey_scratch.py \
  --work_dir "$PRIMARY_I_OR_SUPPORTING_WORK_DIR" \
  --tier 50m \
  --tau 1.5 \
  --seed 42 \
  --lengths "2048,4096,8192" \
  --depths "0.1,0.5,0.9" \
  --num_trials 10
```

上面只是模板；实际 `tier/tau/seed/work_dir` 必须与 checkpoint 命名匹配。

**决策规则**：

| 结果 | Response 更新 |
| --- | --- |
| AR exact 和 TF NLL-gap 同方向 | 可以补一句 “AR exact follows the same trend in a small check” |
| AR exact 弱但 TF NLL-gap 强 | 仍可保留 TF diagnostic，但必须承认 generation exact-match is harder / not the main metric |
| AR exact 不稳定 | 不写进 rebuttal；只澄清 metric definition |

**禁止写**：

- “PK = exact retrieval.”
- “NLL-gap retrieval implies generation exact match.”

### P1.4 Learned Tau Trajectory

**问题对应**：R1 对 shape/scale 和 learnable tau negative result 的疑问。

**目标**：如果已有 logs 记录 learned tau 轨迹，可以把 Table 4 的 negative result 从 “odd failure” 转成 “training loss is myopic for extrapolation allocation.”

**可用证据条件**：

- logs/checkpoint metadata 已经存在；
- tau trajectory 可追溯到 exact run；
- 不需要重新训练。

**允许写的 response**：

- “The learned parameter tends to drift toward the in-range optimum / remains weakly identified, consistent with the fact that training loss observes only `L_train` while extrapolation benefit is out-of-range.”

**如果没有日志**：

- 不写这条；只用理论解释和 Table 4 negative result。

**禁止写**：

- “learnable tau proves EVQ closed form is optimal.”

## 3. P2: Training-Class Experiments

P2 只有在已有 checkpoint/compute/脚本非常近时才做。它们不应成为 rebuttal 成败的默认条件。

### P2.0 LLaMA-3-8B Clean Positional Distillation Pilot

**问题对应**：原 Base vs EVQ-LoRA 两行同时改变 RoPE、LoRA target 和
LongAlign 监督目标，无法判断 8K `+30%` 来自频率切换还是窄域微调遗忘。

**当前状态**：seed-42 脚本、冻结数据协议、四组评测和自动 gate 已准备，
尚未启动 GPU 实验。

**四组诊断拆分**（不是无 LoRA 实验）：

| Arm | Schedule | Training | 解释 |
| --- | --- | --- | --- |
| Base-Geo | native Geo | none | 原模型 |
| Base-EVQ | EVQ tau 1.414 | none | 直接换频率冲击 |
| Geo-Null（文件名仍为 Geo-Distill） | native Geo | 1-step q/k hidden null | pipeline/null sentinel |
| EVQ-Distill | EVQ tau 1.414 | q/k-only hidden distillation | 位置重校准 treatment |

训练只使用固定 revision、document-disjoint 的 FineWeb-Edu plain-text token
序列和 teacher hidden states；没有 LongAlign、token-label CE、任务答案或检索
监督。EVQ 为 seed 42、300 steps；Geo teacher/student 初始相同，因此只保留
1-step null sentinel。两者都是 8K、effective batch 8、`r=64`、`alpha=128`、
`lr=2e-5`。

这个矩阵去掉了 LongAlign/task supervision 并收窄了 adapter 作用面，但仍然
使用 q/k LoRA，同时引入 Geo-teacher hidden distillation。它不能单独证明旧表
的 `+30%` 来自 LoRA、LongAlign、v/o 或学习率，也不能称为 full fine-tuning。

**入口**：

```bash
export CUDA_VISIBLE_DEVICES=0
bash scripts/2026-07/01_lora_positional_distill_seed42.sh prepare
bash scripts/2026-07/01_lora_positional_distill_seed42.sh benchmark
bash scripts/2026-07/01_lora_positional_distill_seed42.sh train
bash scripts/2026-07/01_lora_positional_distill_seed42.sh eval
```

RTX PRO 6000 正式跑前，用相同 frozen microbatches 比较 compile on/off；若显存
有余量，再比较 B2/GA4 与 B4/GA2，始终保持 effective batch 8。正式入口默认
只 compile student backbone，teacher eager，并记录 CUDA/arch、tokens/s、峰值
显存与 5 秒间隔 GPU utilization/power。FP8、QLoRA、DDP/FSDP 不进入首轮。
`benchmark` 使用独立的 12-step non-claim 目录，不得与正式 checkpoint 混用；
正式 checkpoint 只有在 immutable protocol 和覆盖全部 fresh/resume 段的
invocation ledger 均通过后才会生成 `claim_ready.json`。

**固定停止条件**：

- Geo-Null 在 8K/16K/32K 任一长度相对 Base 的绝对漂移超过 1%：先查 pipeline；
- EVQ-Distill PPL@8K 高于 Base 超过 10%：不补 seed，先诊断 rank/target/LR；
- 正式通过要求 EVQ 8K cost `<=5%`、相对匹配 Geo-Null 的 16K improvement
  `>=2x`、32K improvement `>=4x`、表示误差恢复 `>=90%`；
- quick RULER 不是 A-stage gate。没有 AR capability 提升时，只能说 clean
  positional recovery，不能说 industrial context capability。

必须报告四个 contrast：Base-EVQ − Base-Geo、Geo-Null − Base-Geo、
EVQ-Distill − Geo-Null、EVQ-Distill − Base-EVQ。只有 exact JSON、WikiText/hash、
canonical frequency hash、adapter hash、model/tokenizer fingerprint 与 seed scope
全部通过 fail-closed summary 后，结果才可进入 rebuttal intake。

**与论文旧 LoRA 行的桥接优先级**：如果 rebuttal 要声称旧混杂被关闭，仍需
原协议的 Base-Geo / Base-EVQ / Geo+LongAlign-LoRA /
EVQ+LongAlign-LoRA matched block；clean pilot 不能替代它。clean seed-42 成功后
再决定是否补 43/44，失败或 8K cost `>10%` 时不补 seed。

**相关文件**：

- `experiments/lora_evq_v2/prepare_positional_distill_data.py`
- `experiments/lora_evq_v2/train_positional_distill.py`
- `experiments/lora_evq_v2/eval_positional_distill.py`
- `experiments/lora_evq_v2/summarize_positional_distill.py`
- `docs/superpowers/specs/2026-07-10-llama8b-positional-distillation-design.md`

结果未完成前，LoRA 仍走 Path B，不得把“脚本已准备”写成新证据。

### P2.1 MLA Tau Sanity

**问题对应**：R1/R3 可能质疑 MLA 中 `d_eff=d_head` convention，而非 theorem。

**目标**：在 scarce-channel MLA 下比较 `tau=1.414` 与更接近 `d_rope/sqrt(L)` 的 tau。

**可用入口**：

- `scripts/core_text_phases/run_gqa_evq_experiment.py`
- `scripts/core_text_phases/run_350m_mla32_500m.sh`
- `scripts/core_text_phases/run_125m_mla_v2_500m.sh`
- `scripts/core_text_phases/run_50m_mla_v2_tau_sweep.sh`

**脚本能力核对**：

| Script | 能做 | 风险 |
| --- | --- | --- |
| `run_gqa_evq_experiment.py` | general MHA/GQA/MLA training with `--taus`, `--d_rope`, `--seq_len`, `--train_tokens`, `--eval_16k` | training cost; results write to work_dir |
| `run_350m_mla32_500m.sh` | 350M MLA, `d_rope=32`, `seq_len=8192`, 500M tokens, tau `0.0,1.414` | hardcoded remote path; expensive |
| `run_50m_mla_v2_tau_sweep.sh` | 50M MLA-v2 tau sweep at 4K/200M | different architecture/protocol; supporting only |

**dry-run sanity command**：

```bash
python scripts/core_text_phases/run_gqa_evq_experiment.py \
  --tier 350m \
  --taus 0.0,0.354,1.414 \
  --seeds 42 \
  --attn_type mla \
  --d_rope 32 \
  --seq_len 8192 \
  --batch_size 6 \
  --train_tokens 5000000 \
  --work_dir "$EVQ_REBUTTAL_RUN_DIR/mla_tau_sanity_dryrun" \
  --dry_run
```

真正训练前必须重新确认：

- `tau=0.354` 是否确实对应 intended convention；
- 是否只跑 50M/125M quick sanity，而不是 350M/500M full rerun；
- 是否能在 rebuttal deadline 前得到 stable result。

**决策规则**：

| 结果 | Response 更新 |
| --- | --- |
| `tau=1.414` 仍优 | MLA convention 得到支持，但仍写成 operating rule |
| `tau=d_rope/sqrt(L)` 更优 | 主动收缩 MLA story：current default is not universally optimal under scarce channels |
| 结果 mixed | 不写强结论，只写 limitation |

**禁止写**：

- “`d_eff=d_head` is a theorem.”
- “MLA result proves production DeepSeek behavior.”

### P2.2 Fixed-L Continuation For Schedule-Sensitivity

**问题对应**：1B reversal / tau-L mismatch。

**目标**：用固定 `L_train` continuation 证明 1B reversal 不是 “training saturation kills EVQ”，而是 schedule mismatch/scarce-channel sensitivity。

**现实判断**：

- 这是概念上最干净的实验；
- 但训练成本高，不应作为 rebuttal 必做项；
- 如果没有现成 checkpoint/compute，不要临时启动大实验。

**最小可写替代**：

- 把 1B row relabel 成 schedule-sensitivity limitation；
- 解释 MLA scarce-channel `K` 小，allocation mismatch 更难被冗余吸收；
- 不宣称该解释已经被 fixed-L rerun 证明。

**禁止写**：

- “We resolved the 1B reversal” unless fixed-L evidence exists.

### P2.3 Extra Seeds For Primary II

**问题对应**：Primary II single-seed diagnostic。

**目标**：如果已有 125M `L_train=128`, 15M protocol 能快速复跑，补 Geo/DAPE/EVQ seed。

**注意**：

- 不能用 Phase 11B `L_train=256`, 100M 结果替代 Primary II Table 4。
- 新 seed 只应该回答 variance；不应该把 Table 4 升级成 broad PE dominance。

**决策规则**：

| 结果 | Response 更新 |
| --- | --- |
| 多 seed 同向 | “we added seeds; trend is stable in this diagnostic setting” |
| 多 seed variance 大 | “we scope this as a diagnostic; not a broad benchmark claim” |
| only one extra seed | 明确标 `n=2` or single additional seed |

## 4. Do-Not-Run List

这些方向内部讨论可以保留，但不建议在 rebuttal 窗口执行。

| Direction | 原因 |
| --- | --- |
| broad LongBench/RULER leaderboard | 454M capacity ceiling 容易让数字贴地板；不能直接回答 mechanism concern |
| VideoRoPE fairness comparison | 像抱怨评审文化；不回答 reviewer 具体问题 |
| 1B multi-seed from scratch | 成本高、风险高、deadline 不现实 |
| new theorem / new derivation package | 可能打开新攻击面，且 rebuttal 不能变第二版论文 |
| broad baseline sweep over many PE methods | 会稀释主线；只做 reviewer 指名且低成本的 baseline |
| massive LoRA 9-run retrain | 除非 checkpoints 已经存在；否则不是最小 rebuttal |

## 5. Response Upgrade Matrix

这张表决定实验回来后改哪份 response 文档。

| Evidence arrives | Update file | Upgrade allowed |
| --- | --- | --- |
| Exact Base/Geo+LoRA/EVQ-LoRA table | `TABLE23_LORA_WORKSHEET.md`, `AUTHOR_RESPONSE_PACKET.md` | Path B -> Path A |
| LoRA Geo+YaRN/NTK eval-only | `AUTHOR_RESPONSE_PACKET.md`, `REBUTTAL_CLAIM_LEDGER.md` | Add training-free scaler reference if exact |
| Primary I Geo+YaRN scale sweep | `REBUTTAL_DRAFT_EVIDENCE_SCOPED.md`, `AUTHOR_RESPONSE_PACKET.md` | Address tuned baseline concern |
| AR exact passkey | `REVIEWER_RESPONSE_SKELETON.md`, `AUTHOR_RESPONSE_PACKET.md` | Add separate AR exact sentence |
| Learned tau trajectory | `REVIEWER_RESPONSE_SKELETON.md` | Strengthen R1 response |
| MLA tau sanity | `PAPER_ISSUE_AUDIT.md`, `AUTHOR_RESPONSE_PACKET.md` | Adjust MLA convention/limitation |
| Clean positional-distillation JSONs | `TABLE23_LORA_WORKSHEET.md`, `AUTHOR_RESPONSE_PACKET.md` | Separate injection shock and adaptation forgetting; supporting-only until multi-seed |

如果 evidence 没有回来：

- 使用 `AUTHOR_RESPONSE_PATH_B_COMPACT.md`；
- 明确 LoRA confound has been scoped down；
- 不让缺失的 Geo+LoRA 数字拖住 Figure fix、provenance、1B relabel 这些已经完成的 trust 修复。

## 6. Minimal Final Rebuttal Shape

无论 Path A 还是 Path B，最终 response 应该按这个顺序，不要按“我们做了很多工作”的顺序。

1. Thank reviewers and state scope: EVQ-Cosh studies training-time RoPE frequency allocation, not universal long-context SOTA.
2. Correct trust issues first: Figure/Table mismatch fixed; metric definitions clarified.
3. Answer empirical confounds with exact data only:
   - Path A: include Geo+LoRA table.
   - Path B: concede LoRA row was confounded and keep it supporting.
4. Address training strength without “overtraining”:
   - progression evidence;
   - 750M continuation;
   - token/protocol table;
   - LoRA only if exact control exists.
5. Address 1B reversal as limitation:
   - schedule-sensitivity / scarce-channel allocation mismatch;
   - no saturation robustness claim.
6. Address theory:
   - shape derived under surrogate;
   - scale is operating default / basin selector;
   - learnable tau negative result is expected if supported by logs, otherwise framed cautiously.
7. Close with concrete paper edits, not new grand claims.

## 7. Current Go / No-Go Checklist

| Item | Status | Rebuttal impact |
| --- | --- | --- |
| Raw source verbatim archive | Done | reviewer-material traceability |
| Comprehensive prep doc | Done | internal planning |
| Figure/Table NLL fix | Done | trust repair |
| Primary token/protocol note | Done | training-strength response |
| 1B relabel | Done | avoids contradiction |
| LoRA wording scoped down | Done | avoids false attribution |
| Path B compact response | Done | safe default |
| Exact Geo+LoRA numbers | Missing | only blocker for Path A |
| LoRA training-free scaler reference | Missing | optional but useful if fast |
| Primary I tuned Geo+YaRN | Missing | optional if reviewer specifically attacks scale |
| AR exact PK | Missing | optional if metric attack is severe |
| Learned tau trajectory | Missing | optional R1 support |
| LLaMA-8B positional-distillation pilot | Prepared, not run | cleaner diagnosis before any multi-seed LoRA claim |

Final decision rule:

- If exact Geo+LoRA numbers arrive before response drafting: use Path A.
- If not: use Path B and do not apologize for missing broad new experiments; emphasize scoped claims, fixed trust issues, and primary evidence already in the paper.
