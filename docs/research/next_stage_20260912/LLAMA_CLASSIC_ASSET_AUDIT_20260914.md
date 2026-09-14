# Llama TailSpline经典评测资产审计与可续跑合同

更新：2026-09-14。状态：**服务器与本地历史资产已只读审计；正确Core-6补齐与
Full-13/PPL50流水线代码就绪，尚未由本文启动GPU。**

## 1. 直接结论

现有`llama_low108`不是8/16/32K面板，而是Core-6 × 8/32/64K × 6行/格：

- 路径：`/root/autodl-tmp/fixed_rope_three_interfaces_20260913/panels/llama_low108/screen.jsonl`；
- 108行，SHA256 `d031dc797ba3c9452ab4f6ee038de006cf24a1fdb193d261fd46187f62f65edf`；
- 以`--length-cap 8192 --length-cap 16384 --length-cap 32768`过滤只会保留8K和32K，
  因而实际是72行，不能称三长度实验。

现有Llama资产没有一套可直接复用为TailSpline最终经典评测的完整逐行基线。能够复用的
是冻结checkpoint/tokenizer、pinned RULER生成器及其原始数据、50份长文源和部分历史
prompt；不能把旧LoRA聚合、旧表结果或缺raw的摘要拼进当前零训练比较。

## 2. 已核对资产

### Checkpoint与上游生成器

- Llama checkpoint：`/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct`；Native 8192、
  32层、32 Q heads、8 KV heads、head dim 128、`rope_theta=500000`。
- `config.json` SHA256：
  `61f3de03a16ca8046b05dc777bce72717022bc8522152eee61a69072272ef54b`。
- `tokenizer.json` SHA256：
  `e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7`。
- RULER upstream：`c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`；13任务所需
  `niah.py`、VT、CWE、FWE、QA生成器和SQuAD/HotpotQA/Paul-Graham原始输入均存在。

### 旧Llama RULER-derived面板

- Plan-B P：96行，8任务 × 8/16/32K × 4行/格；rows SHA256
  `2786b205a3c8fc2b7d89b93191123501e91336350788c3dfcc6e72a4dc83e6ec`。
- Plan-B S：160行，8任务；8K每格4行、16/32K每格8行；rows SHA256
  `90dfb3e9f17ba246e44884c3acba71170a32439adcf510d54b7535afce600219`。
- 它们只有`niah_single_2`、`niah_multikey_2`、`niah_multivalue`、
  `niah_multiquery`、VT、FWE、QA1、QA2；缺Full-13中的single1/single3、
  multikey1/multikey3与CWE。
- 旧深度是10/35/65/90及多证据cluster，不具备注册的10/30/50/70/90五点合同。
- 旧64K compact也只有8任务和旧深度，不补上述缺口。

本地历史Full-13记录
`rebuttal/rebuttal_0723/theory_results/llama8b_matched_ruler_mix_20260726.json`
是516步LoRA的聚合证据，部分臂不完整且当前没有全部逐行raw；不得作为冻结TailSpline基线。

### PPL长文

服务器已有50份带来源回执的长文：36份ProofPile test与14份PG19 test，manifest SHA256
`1d60909bddc18a1626aaf8addc03cf0d6bd877b7e1b9615a40693405f89d73d0`。
每份文本足够重新用当前Llama tokenizer冻结32K+1前缀。旧
`llama_minimal_band_s4_20260913/lm.npy`只有2份PG19文档，虽确为Llama PPL资产，仍不足
以支持最终PPL结论；其他本地旧PPL多为LoRA、小样本或缺逐行身份。

### Passkey/NIAH

本地旧`niah_results.json`和LoRA `niah_recall_results.json`覆盖10/30/50/70/90，
但每格仅1行，且不保存完整冻结prompt token/hash；只能说明已有生成器经验，不能复用为
当前配对raw。最终流水线将RULER `niah_single_1`作为Passkey读出并明确它与Full-13共享
同一raw，不伪装成独立数据；其余七种NIAH共同构成NIAH family。

## 3. 无缝修正块：只补遗漏的16K

`run_tailspline_llama_s4_complete108.sh`先核对当前四臂的72行确为
Core-6 × {8K,32K} × 6，再从现有冻结面板
`llama_s4_core6_fresh6_16k32k_seed20260917_v2`只取16K的36行/臂。该面板SHA256为
`99fc70a17376d4792ae8d8220196c05a48db3c01ea2b39194bfa533a235c65b5`，与现有72行
prompt交集为0。

完成后报告严格合并同一表/gain下的72+36，得到真实Core-6 × 8/16/32K × 6 =
108行/臂。它是正确三长度诊断，不冒充Full-13/PPL闭环，也不重复已经生成的8K/32K。

## 4. 主经典评测合同

`prepare_tailspline_llama_classic_assets.sh`在CPU上并行准备：

1. 首次Full-13 RULER：8/16/32K均为10行/任务，共390行；六个单答案NIAH
   任务在每个长度对10/30/50/70/90各2行，选择不读取模型输出；
2. PPL50：36 ProofPile + 14 PG19，每份冻结32769个Llama tokens，运行时评估
   8/16/32K前缀；按语料分别报告，并保留组合与source-equal诊断；
3. 当前只生成TailSpline/MrPro两表，统一S4、canonical `[18,35]`、gain
   `1.138629436111989`，无lambda、band、gain、depth或混合搜索。

`run_tailspline_llama_s4_classic.sh`只接受已完成资产，两臂顺序运行；每臂同一模型驻留
完成390次生成与150个LM文档-长度格。`generations.jsonl`和`lm_rows.jsonl`按冻结顺序
append，已有文件必须是合同前缀才可续跑。

`tailspline_llama_classic_report.py`最终核对表/gain/band、390行Full-13 cell、
150行PPL、prompt集合与五深度覆盖，输出：

- token-weighted PPL及log-length PPL-AUC（越低越好），ProofPile/PG19分列；
- Passkey（复用`niah_single_1`，不称独立数据）、八项NIAH task-equal AUC及五深度；
- Full-13 task-equal曲线/AUC、逐任务、worst、EOS/cap与paired bootstrap。

## 5. 样本量与耗时依据

- 修正块只新增36行/臂，共144次生成。历史同runner的36个16K Llama rows从contract到
  status约144秒/臂，因此四臂约10分钟，允许加载与任务输出造成的波动。
- 正确Core-6 108行/臂的现场组成耗时约：8K+32K 72行403秒，16K 36行144秒，
  合计约9.1分钟/臂。
- 首次Full-13每臂prompt输入约7.41M tokens，是上述正确Core-6约2.05M的3.61倍；按
  当前实测线性外推，生成约0.55小时/臂。50文档PPL增加150次forward；由旧2文档
  PPL+36生成的219秒记录估计再加约0.3–0.7小时/臂。
- 因此当前TailSpline/MrPro两臂保守预计约1.75–2.5 GPU小时。它是当前机器/runner的规划估计，首个完整
  TailSpline arm结束后必须以实际wall time更新，不能当成实测承诺。

首次主评测规模来自已经冻结的Full-13合同，不因首臂分数缩减任务或新增中间候选。
只有PPL、Passkey/NIAH、Full-13三个family endpoint中至少两个相对MrPro呈正向，
才允许将确认块扩大到50行/格；否则不以追加样本或新候选挽救结论。
