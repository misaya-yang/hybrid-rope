# EVQ-Cosh experiment and rebuttal rules

最后更新：2026-07-24

在任何 rebuttal、LoRA、长上下文或付费 GPU 工作前先读本文件。

## 1. 当前真实状态

- 当前 rebuttal 入口是
  `rebuttal/rebuttal_0723/README.md`。
- 仓库内目前唯一可逐字核验的正式 review 是
  `rebuttal/rebuttal_0723/00_REVIEWER_27BE_OFFICIAL_REVIEW.md`：
  Reviewer `27bE`，评分 3，置信度 4，稳定 concern IDs 为
  `R27bE.1`–`R27bE.5`。
- 0723 宽实验计划对 `zWsa`、`Dz6s` 的描述不是仓库内原始 review。补齐原文
  和来源哈希前，不得称为 reviewer 原话或据此宣称“必须做”某个实验。
- 尚未形成可发送的 author response。
- 当前数值结论入口是
  `rebuttal/rebuttal_0723/EXPERIMENT_REPORT_20260724.md`；其中
  shape/base/FMRoPE 结果已完成，但必须保留报告中的负面边界。
- `rebuttal/rebuttal_0723/mla_scarcity_5090/` 只有冻结代码与本地测试，
  尚无训练结果。
- 外部服务器上可能存在的数据、preflight 或进程必须现场核验；本地代码存在、
  过去的口头状态或远端路径均不等于已完成证据。
- 所有 7 月 23 日前材料已移入 `rebuttal/pre_rebuttal/`。它们是事实底稿和
  历史档案，不再是当前行动入口。

## 2. 当前 rebuttal 重点

| Concern | 回答重点 | 当前边界 |
| --- | --- | --- |
| `R27bE.1` | 分开 exact surrogate、pure-tether choice、conditional small-\(\tau\) proxy 与 empirical deployment rule | 不再声称端到端理论闭环 |
| `R27bE.2` | 如实列出模型规模、base 和架构覆盖；held-out base/`d_head` 已有定向补充 | 当前没有新的大模型结果 |
| `R27bE.3` | 纠正 submitted “DAPE” 的实际身份，并用 fixed-schedule controls 回答 shape 问题 | 旧行实际只是 shared learnable `inv_freq`，不能继续当 DAPE head-to-head |
| `R27bE.4` | predicted/tuned \(\tau\) 与 matched non-cosh schedule | 三 seed 结果支持 shape 轴，但不支持 Cosh 唯一最优 |
| `R27bE.5` | held-out base=1M、`d_head=128` 已回答；大模型部分明确保持 open | 前半问已完成，生产规模仍缺 |

回答只围绕 reviewer 实际问题展开。不要因为仓库里有旧材料，就把 YaRN、
RULER、LoRA、所有理论错误和所有 supporting 结果塞进每条回复。

## 3. 必须遵守的科学事实

- Submitted Primary I 的所谓 “YaRN” 是 repository-defined fixed-index
  smooth-ramp scaler，不是 official YaRN。
- Submitted Primary II 的所谓 “DAPE” 是 layer-shared learnable inverse
  frequencies，不是 DAPE。`R27bE.3` 已直接涉及这条比较，回答时必须纠正。
- 当前 rebuttal 定义固定为：`Std-Geo` 使用 \(u_k=k/K\)；`Paper-Geo`
  使用 \(u_k=(k+\tfrac12)/K\)；`EVQ-Cosh` 使用与 Paper-Geo 相同的
  midpoint quantizer。Primary I/Primary II 主链路支持 Paper-Geo 身份，但
  Phase11/11B L=256 supporting Geo 实际是 Std-Geo，不能把所有旧 `Geo`
  名称视为同一 lineage。精确 hash 见
  `rebuttal/rebuttal_0723/FREQUENCY_DEFINITION_MANIFEST.json`。
- Cosh 是 stated convex surrogate 的唯一 minimizer；不是 exact oscillatory
  collision kernel、attention objective 或 LM/task objective 的全局最优解。
- Ordinary baseline-to-perturbed KL 的一阶变分为零，从 \(O(\tau^4)\) 起。
  `tau=d_eff/sqrt(L)` 只能称 conditional proxy 提供结构动机后的 empirical
  operating default / basin selector。
- `c_coll=1.171` 的旧 calibration 没有完成所声称的优化；旧 Phase16
  “27 configurations / all <1%” 叙述也不成立。
- PK 默认是 teacher-forced NLL-gap retrieval，除非 artifact 明确标为
  autoregressive exact match。
- 8B LoRA 的长位置 PPL/NLL 改善没有转化成 registered downstream QA 提升，
  且主要能力缺口出现在 \(\le8\)K。不得把稳定 PPL、routing probe 或
  source-removal effect 写成 downstream capability。
- “8B 短 LoRA 梯度不足以重写频带依赖”目前是机制假设，不是已证明原因。

若回复不使用 fixed-ramp、`c_coll` 或 Phase16，它们不需要在每条 reviewer
回复中发散；但一旦引用相关证据，就必须使用上述真实身份。是否发送一条合并的
material-integrity disclosure 由作者决定。

## 4. 当前实验登记与授权边界

### FMRoPE 125M/L=256

- 实际模型为 151,898,880 参数，随机初始化、全参数训练、`L_train=256`。
- 主配对是 Paper-Geo base=500K 与同 quantizer 的 EVQ-Cosh tau=4；
  paper-faithful local FMRoPE
  (`theta_train=256`, `theta_infer=L_eval`) 是第三个近邻方法对照。
- 只回答 base retarget 与 frequency-allocation 的方法级差异，不回答 RULER、
  8B 迁移或 production-scale 泛化。
- FMRoPE 作者公开实现尚未找到，只能称 paper-faithful local implementation。
- 在对应正式 reviewer 原文补齐前，它是作者要求的 novelty probe，不得伪装成
  已核验 review-triggered experiment。

### Reviewer 27bE shape/base

- `shape_l128`：主比较 Paper-Geo vs EVQ-Cosh；seed-42 预注册 tau scan
  和 Std-Geo 小消融；三 seed 的 Paper-Geo、span-matched uniform、EVQ、
  power 和 exponential shape 对照。
- `heldout_b1m_d128`：base=1M、`d_head=128`、Paper-Geo vs EVQ-Cosh
  三 seed；不重复扩张 Std-Geo。
- 两者都是 fresh matched ablation，不能与旧 Primary-II 数字直接合并，也不能
  支持 universal optimum 或大模型 SOTA。
- 两包的 schema-v2 manifest 都必须包含 frequency contract；checkpoint 必须
  持久化 `inv_freq` 并通过 checkpoint/NPY/metadata 三方 hash 与 strict-load
  round-trip。旧 schema-v1 manifest/preflight 必须重生，但 token cache 不必重下。

### 不自动执行

`rebuttal/rebuttal_0723/EVQ_Cosh_NeurIPS2026_Rebuttal_Experiment_Design.md`
保留为宽方案库。其中的 1.5B、每臂 4B tokens、RULER 和 96GB RTX Pro 6000
假设不是当前运行授权，也不匹配当前 5090 资源。只有补齐正式 reviewer source、
证明 score-changing 价值并由作者明确批准后，才重新立项。

历史 8B repair、continuous-frequency adaptation、real-DAPE compare 和
simulated-review 路线均位于 `rebuttal/pre_rebuttal/`，默认不启动。

### Rebuttal short-context SFT data

作者已明确授权
`experiments/rebuttal_2026/sft_distillation/` 的数据生成准备。该授权只覆盖
一份 Paper-Geo/EVQ 共用的短上下文能力 SFT 数据，不自动授权 1.5B SFT
训练或 RULER campaign。

- 程序必须先生成 structured facts、唯一 oracle 和 evidence；DeepSeek 只能
  生成表面 naturalization、等价问题与不相关干扰段落。
- API key 只从 `DEEPSEEK_API_KEY` 读取，模型只从 `DEEPSEEK_MODEL` 读取；
  不得将凭据放入命令、日志、缓存、manifest 或代码。
- 先完成 100 条 audit；逐条人工 annotation 和程序门禁生成
  `audit_gate.json` 前，pilot 命令必须 fail closed。
- Pilot 固定为 train/validation/test = 3000/400/400，按 world/template
  family 隔离；不能用随机样本拆分。
- 最终 `dataset_manifest.json` 必须证明 Paper-Geo/EVQ 使用相同 messages
  文件 SHA-256、样本数和顺序。

## 5. Response evidence rules

- 保留 reviewer 原文和稳定 ID；模拟 reviews 不是证据。
- 分开 reviewer request、repository fact、proposed action、completed evidence。
- 每个 response claim 必须指向 manuscript location 或 traced artifact；否则标
  `AUTHOR_INPUT_NEEDED` 或明确为 open limitation。
- 方法身份或数学错误不能靠新实验修复。
- 代码、preflight、GPU 进程和 checkpoint 都不是结果；只有通过冻结协议、
  evaluation、raw artifact 和 provenance gate 的产物才可候选进入 response。
- Null/negative 结果仍然是有效结果，不得只报告有利长度、seed 或 metric。

## 6. Never repeat

- 不用 LongAlign 替代 paper-lineage LongAlpaca，也不从文件名猜数据身份。
- 不用 paid GPU 做下载、tokenization、tensor preparation、协议设计或普通调试。
- 未核验 command、PID、log 和 artifacts 时，不报告 ready/running/ETA/completed。
- 不把 code SHA、GPU、runtime、compile、checkpointing、cache 或 telemetry 当作
  scientific variable。
- 不用 WikiText-only gain 支持广义泛化，不用短 LoRA run 支持能力提升。
- 不从 class、注释、citation 或旧报告继承 official method identity。
- 每个 named baseline 都必须从 paper row 追到 artifact、runner、forward path
  和固定的官方定义。
- 未通过 representative-output parity 时，使用描述性 local label。
- 明确 native endpoint 与 midpoint frequency grid；两者都是 geometric，
  但不是同一个 control。
- 在把 proxy 称为 theorem 前，独立重推 leading order、assumptions 和
  optimization target。

## 7. Historical 8B clean-pair rules

这些规则只在作者明确重启 8B 路线时适用，不是当前默认实验：

- Match exact model/tokenizer, LongAlpaca tokens/order, objective/labels, LoRA,
  BF16, optimizer, steps, B2/GA4 and external evaluator.
- Within a seed, only the frequency schedule may differ; pure-shape claims also
  require the same endpoint/midpoint quantizer.
- Retain the seed-42 baseline/EVQ artifacts, but label the baseline
  `Paper-Geo` or `Std-Geo` only after its saved `inv_freq` receipt establishes
  the identity. EVQ seeds 43/44 against the seed-42 baseline can only be
  labeled as fixed-reference comparisons, not paired multi-seed controls.
- LongAlign remains a separate experiment and cannot fill a LongAlpaca arm.
- Registered QA negative and short-context harm must be reported with any
  long-position NLL result.

## 8. Execution and GPU

- **GPU-on is experiment-only.** Before paid GPU startup, every code file,
  model, dataset/tensor, manifest/hash, output path, environment and exact launch
  command must be present and validated in no-GPU mode.
- Run SHA-256 checks, downloads, copying, compilation, tests, dry-runs, token
  counting, case generation and transfer preparation off GPU.
- A GPU session may only consume a completed READY receipt and immediately run
  its frozen command. If a prerequisite is missing, shut it down and finish
  preparation off GPU.
- After launch, verify PID, first optimizer step, finite loss, speed,
  memory/utilization and ETA. Never launch another arm without permission.
- Before launch record `df`/largest artifacts and enforce the experiment's
  free-space floor. Delete full checkpoints only after matching evaluation
  JSONs and hashes pass; retain raw metrics and cleanup receipts. Keep a shared
  compile cache only until the registered suite is terminal.
- Record execution metadata, but do not require irrelevant global equality.
  FP8/FP4, quantization, packing, sample order, labels or scientific batch
  changes create a new protocol.
- Shut down paid GPU before offline analysis/packaging when no authorized GPU
  work remains, unless the user explicitly asks to keep it running.

## 9. Large dataset downloads and mirrors

- Canonical upstream repository, immutable revision, file path, byte size and
  SHA-256 define data identity. A mirror is transport only.
- If `hf-mirror.com` redirects a large object to slow overseas Xet storage,
  check whether ModelScope exposes the exact object. Keep the Hugging Face
  revision and hash as acceptance criteria.
- Prefer a resumable multi-connection transfer from ModelScope's current CDN
  redirect rather than hard-coding a temporary signed URL:

```bash
api="https://modelscope.cn/api/v1/datasets/${MS_REPO}/repo?Revision=master&FilePath=${URL_ENCODED_PATH}"
url="$(curl -fsS -D - -o /dev/null "$api" |
  awk 'BEGIN{IGNORECASE=1} /^Location:/{sub(/^[^:]+:[[:space:]]*/,""); sub(/\r$/,""); print; exit}')"
test -n "$url"
aria2c --continue=true \
  --max-connection-per-server=16 --split=16 --min-split-size=4M \
  --file-allocation=none --auto-file-renaming=false --allow-overwrite=true \
  --dir="$DOWNLOAD_DIR" --out="$FILE.incomplete" "$url"
```

- Fetch redirects immediately because signed URLs expire; never commit or
  publish them.
- Preserve `.incomplete` and aria2 control files for resume. Apparent file size
  is not completion: require aria2 `OK`, exact bytes and full SHA-256 before an
  atomic promote to the canonical path.
- Remove slow partials or mirror caches only after verified canonical bytes
  exist. Reject any hash mismatch before tokenization.

## 10. Valid result

A result counts only when training/artifacts are valid, every claimed pair obeys
the frozen contract, external evaluation is complete, and raw per-seed NLL/PPL
is retained. A negative result still counts. A missing arm cannot be replaced by
another dataset, model, seed or protocol.
