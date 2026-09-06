# Fixed-table and FFN experiment execution

- **Status/date:** BOUNDED EXECUTION COMPLETE, 2026-09-04; final Native confirmation recorded on this date. Frozen candidates fail; N128 passes the declared aggregate Native confirmation while format/indexing regresses. Beyond-Native and FFN-specific claims remain unresolved.
- **Questions:** Does the frozen Z witness meet separate Native/long endpoints?
  Can Native-constrained all-linear adaptation transport existing compact skills?
- **Protocol:** [reviewed stages](../preflights/CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md).
  Original checkpoint and fixed N/Z/G/Y identities; full generated output/EOS,
  physical contexts, independent retention, conditional stopping.
- **Authorization:** author authorized execution on the supplied server, runtime
  repairs, outcome-dependent prioritization and high-value follow-ups. Author
  additionally authorized justified cleanup or moving data to the system disk.
- **Artifacts:** private run bundle `ffn_review_execution_20260904`; per-stage
  commands, logs, JSONL outputs and content hashes retained there. Public hashes
  and results will be added after each completed stage.
- **Supported:** actual GPU is RTX 4080 SUPER, 32760 MiB; existing Conda runtime
  is PyTorch2.8.0+cu128, Transformers5.15.1, PEFT0.20.0. The frozen Z table and
  amplitude exported successfully, and actual cached/full-prefix greedy smoke
  passed. These facts establish only that tested runtime boundary.
- **Unsupported:** confirmed global Native feasibility, blind natural transfer, FFN necessity,
  farther-length capability, final-test generalization and paper promotion.
- **Corrections:** prior local-only readiness is superseded for the completed
  runtime smoke only. The non-interactive SSH PATH lacked `python`; Conda was
  present and is used explicitly. No environment reinstall was needed.

## 初步结论与建议 Pro 优先解决的问题

1. **零训练尚未过关：**当前 OLMo 固定Z和静态YaRN都未同时通过Native指标；单位幅度拆分也未通过，两张表都不是去掉幅度就能修好。不能据此关闭方法类别。
2. **LoRA确实改善完整生成：**Qwen单证据16K远端从3/32到24/32；同宽源证据控制后仍为24/32，但增益混合格式与内容，不能全部说成新学会检索。
3. **总体保留已确认、分项遗忘仍存在：**独立确认集宏观保留98.06%，95%区间94.35%–101.65%，通过88%门槛；但格式/位置从61/500降到47/500，保留仅77.05%，不能说所有能力均保住。
4. **主要设计缺口：**自然双证据compact只有4/16可解；NIAH严格0→100%主要是输出格式。不要用这些分数给位置机制或FFN必要性下结论。
5. **最高ROI待检验：**YaRN启发的分布式prefix LM监督；已有固定128位置/0.1权重实现和通过的GPU smoke，未做正式训练。请结合3,642答案标签、192条T阶段Native replay和表/幅度拆分，提出一个最小、可证伪的新协议。
6. **论文范围已收紧：**NIAH＋一项自然QA＋一项标准短能力保留，优先两种checkpoint；不追求穷举所有benchmark或立即宣称通用无损。

下面按协议、结果、失败/无效记录和收据展开。两项幅度对照现已完成；后续新增工作另立明确协议与状态。

## Runtime repair

Transformers5.15.1 moved default RoPE initialization from the common function
registry to `Olmo2RotaryEmbedding.compute_default_rope_parameters`, and uses
`rope_parameters` for the YaRN configuration. The exporter now supports this
official API while preserving the old API branch. No table search or amplitude
change was introduced. Native float32 tensor SHA-256:
`dde15c31724177356ae954d6e11fb337e6fccef56e4520a905cac3f0d9885b34`.
Z tensor SHA-256 remains
`56ddfae2800d4bbf9e6bd2d20bae751edc9865dcbaf641c7c8c4f1d7f1c15e5b`;
amplitude1.102585782722872, logit multiplier1.2156954082626084.

## Running and pending work

The independent24-group Native source diagnostic completed and did not resolve.
Qwen natural training qualification and source-separated Native assets are ready.
Official 2Wiki source recovery, qualification and physical-view construction
completed. Constructed counterfactuals are not untouched benchmark examples.
Corrected N training and paired C/N/F validation completed. Z was stopped at
restoration step26 because original-Native double-evidence controls do not
resolve. Native-window and simple NIAH diagnostics are the current bounded queue.

The data disk initially had about13GB free and the system disk about21GB.
New scratch/cache work uses the system disk where appropriate; no historical
raw evidence was deleted. GPU stages are bounded (runtime smoke600s, first
diagnostic1800s, later stages prospectively bounded). No other training job was
running at connection time. No automatic host shutdown has been performed.

This report is updated from receipts, not launch messages. A failed instrument
will be distinguished from a valid negative candidate result. Adaptation will
not be launched on fabricated qualification metadata to bypass missing assets.

## Completed observations and decisions

### Frozen OLMo Z: Native feasibility fails

The independent reader used115 paired, historically exposed Native validation
rows: ten text anchors and105 generated-task rows. This is a fresh execution,
not newly blind data. The source tokenizer's original seven-file tree was
reconstructed byte-exactly; its composite hash matches the original manifest.
No old outputs were substituted for the new model outputs.

| Endpoint | Native | Fixed Z | Retention |
| --- | ---: | ---: | ---: |
| Text NLL | 3.197478 | 3.319650 | PPL88.50% |
| Five-task macro | .350751 | .283596 | 80.85% |
| EOS-completed task macro | .343606 | .241562 | 70.30% |

Paired source-group95% intervals: PPL[86.00%,90.99%], task[61.40%,101.58%],
EOS-completed task[50.49%,90.48%]. The point task-retention gates already fail;
there is no qualified joint Native/long working point here. Stop this fixed
zero-training candidate's farther matrix. This does not close the table family
or the separately adapted method. Different rows/decoder endpoints prevent
splicing this result with the historical five-task aggregate.

Within the33 summary cases, Native ended within512 tokens on29, Z on3.
Ordinary summary ROUGE changed only .20426->.19842, while EOS-completed ROUGE
changed .18142->.01685. Qasper also lost ordinary answer quality (.39826->.21227),
so termination is not the only measured Native problem.

Native receipt SHA-256:
`0dba8492ad73e3ae1b9d3a79506261ae28d4284a9d629fb438d2cc2755c673fe`.
Z receipt SHA-256:
`79e4a38883151a4241a03ca8c5287aa309f7cd9f79edff701980f2559824a85e`.

### Synthetic instrument: format and background controls do not resolve

Original compact exact+EOS was37.5% for lookup and8.3% for chain. Inspection
showed many correct numeric values accompanied by unwanted labels. A fixed
syntax example, with unchanged full-output scoring, raised compact scores to
75.0% and83.3%. Full near-oracle scores remained0% and50%, respectively.
Both instruments remain **Unresolved** for isolating remote transport. Their
remote zeros cannot close a RoPE or FFN method class. No training was used to
teach this synthetic format, and no additional toy sweep is scheduled.

### Small-model screen: Qwen1.5B gets generation priority

Same48 source-article QA cases and16 worlds per synthetic family, unmodified
models, full strings and native termination configuration:

| Checkpoint label | Compact natural QA | Lookup format | Chain format | Configured Native window |
| --- | ---: | ---: | ---: | ---: |
| OLMo2 1B | 33.33% | 87.50% | 75.00% | 4K |
| Qwen2.5 0.5B | 16.67% | 68.75% | 43.75% | 32K |
| Qwen2.5 1.5B | 54.17% | 100% | 100% | 32K |
| Gemma1.1 2B | 25.00% | 100% | 100% | 8K |

These are development screens, not model-wide rankings. Qwen1.5B was chosen
before seeing its long-task outcomes. Its actual cached/full-prefix parity
also passed. Its64K endpoint is2x configured Native and4x maximum adaptation
length;32K is inside Native. Never import OLMo's4K denominator into its claims.
The existing K64 movement is transported to Qwen's Native basis, then frozen;
this is a new model-specific candidate, not the old OLMo tensor or a new sweep.

Pilot receipt SHA-256:
`0b99f2a19da48a339f277248a1c72e4e1686f4988c3ad073f747dbc8bda17a7d`.

### EOS diagnosis: direct terminal-decision shift plus delayed stopping

Four failure-conditioned summaries had room for a1024-token diagnostic without
leaving the Native4K physical cap. At exactly the original Native terminal
prefixes, Native EOS margins were[3.125,.5625,1.0625,1.9375]; Z margins became
[-1.5,-2.6875,-2.5,-1.9375]. All four eventually ended at635–727 tokens under the
longer diagnostic budget. This supports delayed termination in these cases;
it does not rescue the registered512-token result or establish a population rate.
It also does not uniquely identify FFN as the cause.

The full-vocabulary terminal logits were preserved. A NumPy boolean prevented
one report serialization; recovery reused the exact saved logits and reran only
the missing extended-decode receipts. Recovered logits SHA-256:
`ceafc2ab82a8ba4295e8bccc6f68b82bc03df6facc6ef3283b1c0ce48ea12ccf`.
Completed diagnosis receipt SHA-256:
`ff01cec729e62f45676646ff265e8025d7d09d4769131df9238de505fd703f69`.

## Adaptation: frozen data and corrected numerical contract

Official [2Wiki annotations](https://github.com/Alab-NII/2wikimultihop) supply
source paragraphs and relation triples. Controlled relabeling/binding worlds
carry graph proofs and source identities. Whole source articles and annotated
relation chains are separated; the unused task-calibration source bucket is
reserved for test. This repairs data availability before any long-test outputs,
without repeated instances or fake qualification flags.

Original Qwen Native passed both complete compact worlds for128 training groups
(64 single,32 double,32 binding) after229 candidates. Raw qualification receipt:
`e5eb1f1605dd8863e4d84842350ee131d4f569268b4f8caf5a9d10c9e3ff11ff`.
Training has768 views; validation64 groups; test256 groups. The extra64 SQuAD
cases establish a held-out source-dataset probe, not an unseen-reasoning theorem.
Natural-source counterfactuals are not untouched official benchmark scores.
Near/far within each world uses exact equal-size block exchange. World source
text lengths may differ; no claim of identical evidence-token counts is made.

Native data has512 train,128 calibration,256 validation and1756 test rows across
text, instructions, reasoning and declared synthetic format regression. Text
uses Wikipedia and FineWeb; reasoning uses official GSM8K numeric references.
Model-pilot articles and task source articles are excluded from the current
Native pool. Replay KL is measured on fixed source-truth prefixes, not claimed
to be the teacher's actual greedy trajectory. Short Native checks cover<=4K;
a separate real-report guard covers the chosen model's wider Native window.

The main all-linear candidate has18,464,768 trainable Qwen parameters. Replay is
stratified shuffled sampling without replacement:448 different rows across the
full recipe,112 per stratum. Of these,256 rows are used during restoration and
192 during task learning (48 per stratum). N restoration has zero updates, so
only those192 rows constrain the changed model during transfer. The first actual smoke had nonzero attention/FFN
updates. Its one-step Native/task gradient ratios were about16–17 with positive
cosines; this alone does not prove conflict or blocked learning.

### Numerical invalid run: zero KL must not move original Native

The first continuous N run is **Invalid as the registered identity-restoration
control**, and was stopped with all partial receipts retained. Initial
calibration KL was exactly zero, but the first restoration step had gradient
norm1.248488e-6. AdamW amplified normalization-roundoff gradients; later Native
restoration KL drifted away from zero. Do not use this trajectory as method
failure or a valid main result.

The fix retains the KL forward value and uses its analytic first-order logit
gradient `softmax(student)-softmax(teacher)`, exactly zero at equal logits.
Work-machine tests passed wide-vocabulary zero-gradient identity and float64
finite-difference gradient checks. This is first-order training code, not a
Hessian/Fisher implementation. N restoration now has an explicit zero-gradient
invariant. The corrected trajectory starts fresh.

A separate serialized-resume smoke did not meet its original parameter tolerance
(maximum difference about2e-4 despite identical task loss/Native KL). Its cause
is not yet fully identified. Formal runs therefore use the fixed uninterrupted
32R+96T schedule and preserve all registered saves. No bitwise-resume claim is
made, and no formal result depends on that unresolved path.

## Corrected N seed42: completed training, retention interval unresolved

**Observation / 2026-09-04:** immutable engine release008 completed all128 steps
(32 restoration +96 transfer) in1,721.87 seconds. It consumed6,329,249 task tokens,
768 task forwards and448 distinct Native replay rows. All32 identity-restoration
steps had exactly zero gradients; saved/merged greedy reload passed. Training
JSONL SHA-256: `682da90ffe3e10256e1cce40741f0f5e061640cdd5b0981c198c40e19d9f1992`.

Independent256-row Native validation uses the clean current pool. NLL changed
2.26473431→2.26764774: PPL retention99.7091%. Three-stratum complete-answer/EOS
macro changed25.5208%→23.9583%: retention93.8776%. Both point gates pass88%.
Paired source-group bootstrap95% task/EOS interval is[84.8797%,102.4317%]; its
lower bound fails88%, so the retention conclusion is **Unresolved**, not proven
non-inferiority. Absolute baseline task accuracy is modest; neither this aggregate
nor replay KL establishes general capability preservation. Wider Native-window
and independent confirmation endpoints remain required before that claim.

Receipts: `qwen_N_s42_fixed/complete.json`,
`qwen_N_s42_fixed_native128/native_evaluation.json`, raw `examples.jsonl`, and
`retention_screen.json`. Baseline engine007 and candidate008 differ in training
code; a CPU AST audit confirmed identical evaluation/asset-validation functions
and strata. This provisional retention screen does not replace the full reviewer
checks of shared runtime, token identities and complete generated controls.
The completed generated-validation comparison below establishes a scoped
training-associated improvement; it does not establish FFN necessity or blind
extrapolation. Subsequent baseline controls caused the Z queue to stop.

## Still required before a positive claim

Complete corrected Z/Y runs, evaluate independent short and full-window Native
retention, then full C/N/F validation at the registered saved points. Freeze the
selection before opening farther task data. Scope negative candidates, keep
unresolved controls separate, and add seeds or longer lengths only when the
preceding evidence warrants the cost. Completed adaptation now has a validation-task observation below; blind
extrapolation, FFN-specific attribution and manuscript promotion remain absent.

## Paired natural validation: learned generation, unresolved full protocol

**Observation, 2026-09-04:** same immutable384 validation views and greedy full-
answer/EOS scorer, original Qwen Native versus all-linear N128 (seed42):

| Family | Groups | Native compact | N128 compact | Native near16K | N128 near16K | Native far16K | N128 far16K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single evidence | 32 | 26/32 | 26/32 | 20/32 | 26/32 | 3/32 | 24/32 |
| Binding | 16 | 10/16 | 9/16 | 4/16 | 9/16 | 0/16 | 6/16 |
| Double evidence | 16 | 4/16 | 4/16 | 1/16 | 3/16 | 0/16 | 3/16 |

A success requires both counterfactual worlds to produce the complete lawful
answer and EOS; these are semantic-group counts, not independent token trials.
Unfiltered far macro increased3.125%→43.75%. On Native-compact-correct groups,
N128 compact/near/far is single100%/100%/88.46%(26 groups),
binding90%/90%/60%(10), double100%/75%/50%(4).

The receipt reviewer returns **UNRESOLVED_CONTROLS**: double has only4 qualified
compact groups, below registered8; its near75% is also below80%. Do not hide this
family or lower thresholds after seeing outcomes. Z was stopped at26 completed
restoration steps with partial receipts preserved; Y, replication, FFN placement
ablation and farther tests have not started. This is not a Z negative result.

**Supported:** the complete fixed all-linear recipe learns these controlled
natural-source tasks and improves16K far generation. **Unsupported:** FFN necessity,
Z superiority, fully preserved Native capability, official2Wiki performance,
blind32/64K transfer, or16K as Native extrapolation. Qwen's actual Native window
is32768;64K would be2xNative and4x the maximum training cap.

Native baseline/candidate receipts SHA-256:
`0098e89a7ec3b6156bd0fc6693a6170b22c428e2692608e8b5b17c88972bb7a8`,
`05037dd2ea54ed855d2a73eb4da8e5d8e8fad5952b7ab329122678e44228aca1`.
Task baseline/candidate receipts:
`d216dd81645e5b22c698abff5d50def376fde5270016262d26f0f84307fa693c`,
`74b4446413e28f6a5409a17c62cee7dbffe26f31e2fd25851735eb375c356072`.
Selected diagnostic adapter checkpoint receipt:
`a31cfd710ebf9759fe819140fea32beb8db9d66467dcc4d4ecf2efdcbbc56a8f`.
The full reviewer independently rescored raw token outputs, checked common
runtime/tokenizer/data/deployment hashes, and verified equivalent engine007/008
evaluation ASTs. Its retention screen is now backed by these completed checks.

## What the low Native score means

| Stratum | Native correct+EOS | N128 correct+EOS | Lost | Gained | Source groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| Instructions | 35/64 | 36/64 | 0 | 1 | 11 |
| Numeric reasoning | 5/64 | 3/64 | 2 | 0 | 64 |
| Format/indexing | 9/64 | 7/64 | 2 | 0 | 64 |

All192 generated Native cases ended with EOS in both arms. Reasoning prompts
require direct numeric answers without a reasoning trace; these scores are not
standard few-shot/CoT GSM8K. Format errors mix substantive mistakes and unwanted
Markdown/explanations. Do not normalize or strip these away to pass the registered
full-string endpoint. Losses on4 formerly correct cases are observed, but low
baseline counts and source clustering prevent a broad catastrophic-forgetting
or full-retention verdict from this small validation pool.

The simple Qwen raw-row join has0 lost strict-correct worlds in all four
families and1 gained world in two-hop tracing. OLMo NIAH has all correct gold
surfaces present at4K, despite30/48 strict successes; its lower paired NIAH
score is also principally an output-format effect. These diagnoses do not
replace the full-output gate.

A read-only official Hugging Face metadata check matched the deployed weight
SHA exactly to [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, excluding accidental use of the
base model. This identity check does not certify the experimental decoder.

## Independent subagent review and resulting changes

One author-authorized subagent reviewed code without mutations or GPU work.
Its opinions are analysis; only separately verified code facts enter this report.

- Verified: LoRA covers Q/K/V/O and FFN gate/up/down; source-truth answer and
  EOS are supervised; Native replay uses the deployed table; stable KL gradient
  is `q-p`. No new disconnected-FFN fault was found.
- Limitation: source-truth replay prefixes are not the original teacher's actual
  generated trajectories. Small sampled KL gives no theorem of retained generation.
- Verified control gap: original natural views have C/N/F only. The independent
  source-only companion fixes source widths and confines cross-world differences
  to the source block, then constructs near/far by exact block exchange.
  Identical deleted prompts have structurally zero paired success; report their
  single-world predictions and source-present differences, not a fake independent
  negative-control pass. Companion data is prepared, GPU evaluation pending.
- Verified confirmation gap: the old reviewer suggests confirmation but accepts
  only validation folds. A future confirmation report must freeze the candidate
  before opening the predeclared pool; do not repeatedly select checkpoints from
  confirmation outcomes. No confirmation claim is made today.
- Verified parameter accounting: Qwen all-linear r16 has18,464,768 parameters;
  attention-only r68 has18,522,112(+0.31%). That future comparison identifies
  allocation of a fixed parameter budget, not FFN necessity. OLMo's r46 is separate.

## New bounded diagnostics and model-size decision

The [NIAH canary](../../../../scripts/experiments/niah_retention_canary.py) freezes
8 random seven-digit passkeys in2 worlds, compact plus physical4K/16K/32K at
10/50/90% depths:160 generations per model. Natural FineWeb text supplies
haystacks; Native replay sources are excluded. Data-manifest SHA:
`0eae92cde7fa33a28891b162431a08d94f2d4077700354237775b36a35340877`.
The comparison is original Native versus existing N128; no retraining or test
selection. HF generation parity uses a fresh explicit greedy configuration,
matching single EOS and repetition penalty1; four compact and two4K/32K
8-token boundaries are checked. Native generation defaults are retained in a
separate receipt. A parity failure invalidates canary interpretation first.

This is a declared small NIAH variant, not an exact RULER reproduction. Repeated
length/depth views leave8 independent semantic groups. NIAH probes retrieval,
not all knowledge, reasoning or instruction retention; [RULER](https://arxiv.org/abs/2404.06654)
explicitly motivates tasks beyond simple retrieval. Keep full output+EOS scores
and per-cell lost/gained cases; do not turn this canary into a high-confidence
88% non-inferiority claim.

Decision: if original Native fails simple compact NIAH, inspect loading/prompt
and parity before any model-size conclusion. If original passes and N128 loses,
inspect first divergence and merged versus unmerged adapter before retraining.
If both retrieve well but natural double stays weak, a larger **Native-only**
pilot is worthwhile: use the same semantic validation groups and require at
least8/16 compact-correct double groups and near>=80% on that cohort before
paying for8B training. If it still fails, repair the assay rather than increase
training budget. Current loaders require a single weight file and model-specific
RoPE initialization; a sharded Llama8B needs a new verified weight/operator/
Native-length contract. No8B training or download has been started.

## Further paired checks and actual OLMo size

CPU joins of the original and N128 natural raw outputs find **20/32 single,
6/16 binding and2/16 double** groups satisfying all four conditions: original
compact correct, N128 compact correct, original far wrong, N128 far correct.
These are observed rescue counts for already-solvable compact skills on this
fixed validation set. They strengthen the task-learning interpretation while
leaving table-specific, FFN-specific and beyond-Native claims open.

The24-report Native-window guard completed for both models: original NLL
1.79178420 versus N1281.79507383, PPL retention99.6716%; ordinary summary task
retention99.5216%, EOS-weighted task retention96.5762%. EOS completion was19/24
versus18/24. These point values pass;24 reports are a development check, not a
high-confidence global retention certificate. Candidate guard receipt SHA:
`d7ff13683ee02fffe2c3b6123eb281b50dc8bdcd106cf3d5bf77d5dcee375376`.

The author requested additional simple tasks and OLMo about1.485B. Reading the
weight headers confirms existing OLMo-2-0425-1B-Instruct has**1,484,916,736**
parameters (Native4096); Qwen has1,543,714,304(Native32768). OLMo is the already
inventoried checkpoint, now receiving new diagnostics rather than being treated
as a new model-size tier. Its checkpoint contract SHA is
`116bd264824a382c2a5add069fdad253ce21ff35909660cef7f2a571664bddbe`.

A [simple capability canary](../../../../scripts/experiments/simple_capability_canary.py)
freezes128 compact generations,16 paired groups each of key-value binding,
two-hop variable tracing, three-word reversal and two-digit addition. The exact
same semantic prompts run on Qwen Native, Qwen N128 and OLMo Native; tokenizer
and input lengths remain model-specific. These are synthetic skill diagnostics,
not a replacement natural benchmark. OLMo also gets the paired NIAH compact/4K
subset, within its actual Native window. Completed outcomes follow below.

## Completed simple tasks and NIAH: content versus output contract

**Observation:** fixed128 compact generations per deployment,16 two-world groups
per family. Qwen results before/after N128 and OLMo Native:

| Family | Qwen Native | Qwen N128 | OLMo1.485B Native |
| --- | ---: | ---: | ---: |
| Key-value binding | 16/16 | 16/16 | 5/16 |
| Two-hop variable tracing | 13/16 | 14/16 | 5/16 |
| Reverse three words | 13/16 | 13/16 | 0/16 |
| Two-digit addition | 16/16 | 16/16 | 15/16 |

OLMo failure examples include wrong numeric bindings and wrong order, not only
extra explanatory text. All128 OLMo outputs ended with EOS. These exact-prompt
results make Qwen the more resolving current compact instrument; they do not
establish a universal ranking of the models or isolate model size causally.
Qwen aggregate task counts do not decline here, but individual lost/gained rows
and16-group uncertainty must remain visible before any broad retention claim.
Receipts SHA-256, Qwen Native/N128/OLMo Native respectively:
`eba1ccf710c8e8df90f2dedf7f5684a8999d11dab1df655297f362cde2c34941`,
`5ac2df9665c597ae6cbb0d46600b69d2ed6cdaeddfa8f8ad2609f46a8d1a57be`,
`42a4655d26a5b078c2179cd7e600b1e3f8fff354084e7b6aa20c3a3832b07c02`.

NIAH full paired output+EOS scores:

| Cap / depth | Qwen Native | Qwen N128 | OLMo Native |
| --- | ---: | ---: | ---: |
| Compact | 8/8 | 8/8 | 8/8 |
| 4K /10% | 8/8 | 8/8 | 6/8 |
| 4K /50% | 8/8 | 8/8 | 1/8 |
| 4K /90% | 7/8 | 7/8 | 3/8 |
| 16K /each10,50,90% | 0/8 | 8/8 | Not run |
| 32K /each10,50,90% | 0/8 | 8/8 | Not run |

**Critical interpretation correction:** every original-Qwen16K/32K output
already contains the correct passkey, but as a sentence, e.g. a declarative
sentence naming the record and number. N128 emits only the requested number.
All these rows terminate with EOS. Thus the0→100% strict gain here diagnoses
output-format compliance, not retrieval recovery. Surface-presence counts are
an error diagnostic only, never substituted for the registered complete-output
metric. At4K one originally format-only failure becomes an actual wrong-number
failure after adaptation despite unchanged aggregate strict score. Preserve
that case; aggregate equality is not per-instance preservation.

Six same-setting HF/custom greedy checks passed for each Qwen deployment,
including4K/32K eight-token boundaries. The OLMo run checks its compact/4K
boundaries. These scoped checks do not prove every decoding configuration equal.
Qwen Native/N128 and OLMo NIAH receipt hashes:
`3e058bb2a5791fd4c0ddf7c122da5c9bb5f10dcf76bf7959f2dbfef3b1281c55`,
`a5017985bc6472889027043d7f628930f9c293a8b66228fafd36133e112eb6b2`,
`36aaee2519aabfa879f7e95761a573dcbe6af67a491e2a1555ed891ef6b51687`.

Natural QA also mixes content and format gains. Among newly strict-correct far
**worlds**, prior outputs contain the gold surface in29/39 single,12/16 binding,
and6/10 double cases. In the remaining10/4/4, inspected examples include refusing
to name supplied evidence or naming the wrong person. These world counts are not
independent semantic-group uncertainty. Gold-surface presence is not a semantic
correctness scorer: it may occur in a negation or alongside alternatives. The
valid headline remains complete-task improvement; it must not be rewritten as
all gains representing newly recovered retrieval or FFN reasoning.

## YaRN comparison and frozen Native control

The author narrowed the next paper increment to2–3 complementary benchmark
families. Primary-source verification and the concrete follow-up design are in
[theory §10](../theory/CONSTRAINED_GENERATION_FIRST_PRINCIPLES_20260904.md#10-yarn-correspondence-supervision-density-and-a-smaller-next-claim).
The main difference from YaRN is not merely FFN inclusion: today's768 task views
supply3,642 supervised answer/EOS positions over6,329,249 input/target tokens,
plus separate Native KL. YaRN's published main recipe is dense full-parameter
language-model training. A next experiment should isolate supervision density
on fixed inputs before simultaneously increasing model size and training budget.

**Observation:** the original frozen official-HF YaRN-s4 control was evaluated
on the exact same115 OLMo Native rows using the identical archived evaluator
SHA `198d5155bb731e55b141e8a7a090e43efb69918910d1e914af092d97672c0762`.
No parameter training or dynamic length routing was used. Fixed Y tensor SHA
`cc9da456982ffce5ca0558e9ea661abc4a880ec002179ce6b9149d45aa4a016c`,
amplitude1.138629436111989, logit multiplier1.2964769927807063.

| Static control on OLMo | PPL retention | Ordinary task retention | EOS-weighted task retention |
| --- | ---: | ---: | ---: |
| Z, fixed p2 s4 | 88.50% | 80.85% | 70.30% |
| Y, fixed HF YaRN s4 | 66.32% | 87.28% | 84.06% |

Both candidates fail current separate88% Native gates. Y improves the task side
relative to Z but worsens NLL; neither dominates the other across endpoints.
Do not extend either zero-training candidate's matrix under this protocol. This
is no refutation of YaRN's trained Llama results or of every possible static table.
The restored/generated adaptation question remains separate.

A CPU tensor check finds15 exactly unchanged frequencies inY and4 inZ;32 versus
39 exactly equal the Native/4 value (float32 comparisons). This describes the
realized band profile; it does not establish which frequency or amplitude change
caused the observed endpoint differences. Y versus Z also changes gain.

## Execution review: two prospective bounded additions

**Registered before their outputs:** a fixed amplitude-component diagnostic
uses the exact existing Z and Y frequency files, with rotary amplitude set to
Native1. It changes no frequency bytes and has no grid or fitted coefficient.
Compare each to its original-amplitude receipt on the same Native-selection
rows, using the same archived evaluator. Two runs are capped at1800s each.
This distinguishes frequency-associated damage from the marginal effect of
amplitude at that table. It neither proves a global optimum nor overrides the
separate Native/full-generation gates. Any longer matrix remains conditional.

The YaRN-inspired training companion now has runnable code: `--prefix-lm`
adds weight0.1 CE on at most128 uniformly sampled prompt next-token positions
per view, alongside unchanged full-answer/EOS and Native KL objectives. Sampling
is fixed by a separate keyed seed and stored position-set hashes; source inputs,
LoRA rank and primary data do not change. The same causal model forward projects
both answer and sampled-prefix positions. No extra position-scale or LR sweep.

**Derived scope:** for fixed parameters and uniform subset sampling, the expected
per-view CE and first-order gradient equal that view's full-prefix mean objective.
The realized subset/trajectory is not the same as dense training, and view means
are not a token-weighted corpus mean. Two CPU tests check index boundaries and
the exact averaged subset gradient on a small enumerated case; work-machine
stable-KL plus prefix tests passed4/4. New task-gradient diagnostics mean joint
answer+prefix gradients. Only a two-step N GPU smoke is queued; formal training
has not been launched and still needs resolving natural controls.

A review also corrected an old accounting error: N128's old receipt calls
6,329,249 `actual_cumulative_task_tokens`, but its student forward omits final
EOS, so actual input count is **6,328,481**. The difference is768, one per task
view. Old receipts remain unchanged;6,329,249 is now explicitly prompt+target
sequence budget. Target supervision remains3,642 answer/EOS positions. New
receipts separate these three counts and sampled-prefix target positions.
This accounting correction changes no trained weights or evaluation scores.

## Latest completed source-only guard and prefix smoke

The full384-row source-only companion completed for original Qwen and N128.
Frequency/operator/data identities are paired; cross-world differences are
confined to the common-width source region and near/far uses an exact block swap.

| Family | Native near/far groups | N128 near/far groups | Total groups |
| --- | --- | --- | ---: |
| Single evidence | 16 /3 | 26 /24 | 32 |
| Binding | 4 /0 | 9 /6 | 16 |
| Double evidence | 1 /0 | 4 /3 | 16 |

On the original compact-correct cohort, N128 near/far is26/26 and23/26(single),
9/10 and6/10(binding),3/4 and2/4(double). The double-control precision remains
insufficient. Padding did not erase the far-generation improvement, but content
versus formatting remains the separately documented interpretation boundary.
Deleted prompts yield an answer matching either world's truth in0/128 Native
and7/128 N128 cases. Deleted two-world success is structurally zero; this is not
an independent reasoning certificate. Raw single-world predictions are retained.
Native/candidate source-guard receipt hashes:
`a68cfcfbde88ffde942f12848fcb7ea905cb4135a1c4e20b64c87493dc2aa22a`,
`7be0c9801bb374edf306de3974f448b350af06f20c50bd2e293510f4638e5883`.

The sampled-prefix N smoke completed1 restoration and1 task optimizer step,
8 task views including physical16K, in30.52s. It used58,014 actual input tokens,
39 answer/EOS labels and1,024 sampled-prefix labels; peak allocated GPU memory
was7,737,387,520 bytes. Native identity restoration remained zero, joint loss
was finite, attention/FFN updates were nonzero, and saved/merged greedy reload
passed. Receipt SHA:
`411168b423f2ea75b37eafb738dabe8b342a45b5a130673be7891490279479d4`.
**Runtime readiness only:** no completed full-prefix-companion training, retention
or generated-task improvement is claimed. Targeted local checks passed83/83;
work-machine numerical tests passed4/4. The initial local test-import collision
with the unrelated scripts/train.py was repaired without installing PEFT locally.

The two unit-amplitude Native evaluations subsequently completed; the verified
results below supersede their earlier running/queued status.
The initial report is ready for external-model analysis; raw source receipts and
immutable code releases remain available on the work machine.

## Completed unit-amplitude decomposition

**Observation:** both additional115-row Native runs completed. Paired frequency
bytes, checkpoint/config/tokenizer, evaluator, data and order were checked; all
525 generated rows across N/Z/Y/Z-unit/Y-unit were independently rescored from
raw tokens. Stored NLL scalars were reaggregated, not rerun by GPU.

| OLMo static deployment | PPL retention | Ordinary task retention | EOS-weighted task retention |
| --- | ---: | ---: | ---: |
| Z, original amplitude | 88.50% | 80.85% | 70.30% |
| Z, amplitude1 | 71.06% | 90.76% | 81.59% |
| Y, original amplitude | 66.32% | 87.28% | 84.06% |
| Y, amplitude1 | 41.08% | 75.96% | 72.37% |

Removing amplitude does **not** repair either fixed-table candidate. At fixedZ,
it improves task scores but worsens NLL; at fixedY all three endpoints worsen.
Thus the original amplitude partly compensates frequency changes for language
modeling; there is no single scalar improvement established across endpoints.
This closes these two Native-null amplitude candidates under the current budget,
not every allocation or every calibration algorithm. No grid, extra scale or
farther matrix follows. Current zero-training work needs a new identified
intervention, not more runs of these failing settings.

Z-unit/Y-unit receipt SHA-256:
`1790824039d71a3f449561f76aec04c3517d2fd8e7d78fd3e7f3dd3414c00852`,
`62867a0411ed39c81abd86ace2d68b57b339a12f7421a19cbef7b9ad70f165bf`.
This is a fixed-table within-arm amplitude effect; it does not identify the
optimal amplitude or the causal contribution of each frequency band.

## Repository organization

The author requested a full `paper-2027/` organization pass by a Sol subagent.
Thirty historical reports and32 preflights were grouped into five parallel
topic folders. Theory/analysis gained grouped navigation. Today's report,
current preflight, HANDOFF and manuscript entrypoints keep their public paths.
Two script-addressed phase reports retain their old paths intentionally.
A parent check matched all62 moved documents to unique originals and found
no changes after normalizing path/link targets. The full tree has533 checked
Markdown links and5 pre-existing missing private-artifact links, with no new
missing target. The85 targeted checks pass. This is an organization change,
not new scientific evidence. TeX/PDF and immutable `paper/` remain unchanged.

## Fixed N128 Native confirmation — registered before opening the pool

To resolve retention uncertainty without checkpoint reselection, the completed
end-of-budget N128 and original Native are frozen by table/gain, adapter/config,
checkpoint/tokenizer and clean Native-pool hashes. Each receives exactly the
predeclared held-out test pool:256 text rows and500 rows per generation stratum,
1756 per arm, capped at1800s. No task test is opened and no training follows
automatically. The [new confirmation reviewer](../../../../scripts/analysis/review_native_confirmation.py) accepts confirmation folds only;
the earlier validation reviewer remains a distinct tool.

A point failure closes N128 under this retention protocol. A confidence-interval
failure remains unresolved; it does not authorize trying earlier checkpoints on
this same pool. Passing can support only the declared task/domain distribution
and fixed checkpoint, alongside the separately completed Native-window guard.
This follow-up is a confirmation of one fixed model, not test-driven model selection.

## Final fixed Native confirmation

**Observation / declared-scope confirmation:** both predeclared1756-row runs
completed. The dedicated reviewer verified frozen model/data/adapter locks,
matching evaluation code, source-group pairing and raw full-output/EOS scores.
It returns `NATIVE_CONFIRMATION_PASS_AT_DECLARED_SCOPE`.

| Endpoint | Original Native | Fixed N128 | Retention |
| --- | ---: | ---: | ---: |
| Text NLL | 2.25430848 | 2.25457272 | PPL99.9736% |
| Generation macro, including EOS | 24.0000% | 23.5333% | 98.0556% |
| Instruction answers | 240/500 | 250/500 | 104.17% |
| Numeric reasoning | 59/500 | 56/500 | 94.92% |
| Format/indexing | 61/500 | 47/500 | 77.05% |

Paired source-group bootstrap95% retention intervals are PPL[99.8121%,100.1260%]
and task/EOS[94.3452%,101.6479%]. All aggregate lower bounds exceed88%.
There are149 text source groups,27 instruction source groups and500 source
groups in each remaining stratum. These are not training-seed intervals.

**Material limitation:** aggregate confirmation does not imply every skill was
preserved. Format/indexing loses15 formerly correct cases and gains1; reasoning
loses9/gains6; instructions lose5/gain15. This is an observed subtask regression,
not negated by the macro pass. The originally declared gate was aggregate; we
retain that outcome while rejecting any stronger per-task no-forgetting claim.
The older validation interval remains its own valid, less precise observation,
not a score to overwrite. This fixed confirmation must not select another saved
checkpoint or tune the next variant; a later adaptive method needs fresh validation
and confirmation cases.

Baseline/candidate receipt SHA-256:
`8266f78a15ee0837f7d54d546455e03439795bae7c59a526c9425dccf9a86067`,
`e338d594714dcf824c30e624268087cc52145a5f634c09935b256527d913e73d`.
Lock hashes:
`c543d96b8b9fa6c8e82d36d8a4bdf80b49d27f5da0073f8195230e0e630fb9b3`,
`8419cd106e01eb72e5d00006533368e7c97db44f0aee1febb0d310407da38886`.
Reviewer code SHA:
`73b9df07efb5df10f0989bef9b468cb2ab9952b5fdc7330eebad9cca633a9ede`.
The new reviewer has2 passing boundary tests locally and on the work machine;
all85 targeted local checks pass.

### Next experiment: what this changes

**2026-09-05 priority amendment:** The [Pro audit reconciliation](../analysis/PRO_REPORT_AUDIT_RECONCILIATION_20260905.md) supersedes the next-training order below: first matched N_compact, then old-recipe Qwen Z/Y. Double underqualification is family-local; the old joint verdict remains. Teacher-trajectory replay and prefix-LM remain conditional later hypotheses. No new run or result is implied.

Do not expand the failed zero-training settings or restart the unresolved double
matrix. For adaptation, distinguish two remaining questions before more training:
(1) can supervised content use improve after removing the measured formatting
confound; (2) can the format/indexing regression be prevented on fresh cases?

The sampled-prefix arm is runtime-ready but unproven. Its extra language-model
supervision addresses distributed learning signal, not a guarantee of retention.
A complementary working hypothesis is inadequate Native decision coverage:
truth-prefix replay may miss the teacher's actual generated paths, and only192
rows constrain N during task learning. The smallest retention-oriented comparison
should draw actual teacher trajectories from training/calibration sources only,
keep the answer/position/optimizer budget fixed, and use new held-out cases.
Do not feed today's confirmation failures into training and then call this pool
unseen. More model parameters or GPU memory do not by themselves solve that issue.
