# Native-Z5原生窗口探索结果（2026-09-15）

## 结论

在冻结`OLMo-2-0425-1B-Instruct`全部权重后，V1通过五自由度的RoPE内部
`z`校准，在独立46文档面板的1K、2K和4K NLL上都显著优于geometric Native。
这构成一个单checkpoint的**存在性结果**：发布时的geometric RoPE不是该冻结模型
在这组原生窗口自然文本上的精确事后最优点。

证据不能升级为“强Native增强”或普适构表规则：V1的RULER和Natural-QA点估计
虽为正，95%区间都跨0；split-consensus方向没有复现显著4K NLL改善，all-50 refit
也未通过预先设置的开发门。因此本探索不支持“改z可稳定增强原生任务能力”，也不支持
将checkpoint校准得到的方向迁移到其他模型或数据。

## 固定合同

- 模型：`allenai/OLMo-2-0425-1B-Instruct`，Native长度4096，模型权重更新数为0。
- 干预：七个固定pair-index knots、两端固定、五个有效内部`z`自由度；正gap保证
  严格有序；最快/最慢实际频率、全support、base、`gain=1`保持不变。
- 优化：PG19 validation的50个冻结4097-token窗口；design/selection/internal-confirm
  为16/16/18本；Native初始化，AdamW，lr `0.003`，40步，seed `20260914`；
  由selection选择step 35，不扫超参数。
- 独立NLL：ProofPile test 32文档与PG19 test 14文档，共46文档；长度1K/2K/4K；
  主要终点为paired-document 4K NLL。
- 任务：Native-4K RULER Full-13×10（130条）及未截断Natural-QA99
  （2WikiMQA 24、HotpotQA 5、Qasper 70）；任务分数不参与选表。
- 强判据：4K NLL差值95%区间上界小于0，且RULER或Natural-QA至少一个
  task-equal差值95%区间下界大于0。V1只满足前者。

冻结资产SHA256：PG19-validation 50×4097为
`26226aac62a30e3b29d521dfa177ee6c22c8bc6affa086b735eea628d864d12b`；
PPL46 array为`0864553fbed3d4967dcb517c867284d0c0856d6fd714d0ecc9888bfce7724f1a`；
RULER130为`cdf69c94ecc8fd3d95fac7547137ad5f5cdcc2c2ae139ae63f49275e8c06b5b2`；
Natural-QA99为`a94428d3513f7578c9d013aecd015f427c02566ee20bb72caee95e5ab087f877`。
预注册合同来自Git提交`dd100f2`中的
`docs/research/next_stage_20260912/NATIVE_Z5_ENHANCEMENT_PREREG_20260914.md`。

## V1结果

候选表SHA256为
`12cbf88c3e7f0d013376869d45814087094effe70cfb108c8c7c523b69f2579b`。
以下NLL差值均为`candidate − Native`，负值更好：

| 长度 | Native NLL | V1 NLL | 差值 | paired 95% CI |
|---:|---:|---:|---:|---:|
| 1K | 2.6081037 | 2.6014541 | -0.0066496 | [-0.0087493, -0.0045881] |
| 2K | 2.4419243 | 2.4391494 | -0.0027749 | [-0.0047224, -0.0008381] |
| 4K | 2.2978120 | 2.2957033 | -0.0021088 | [-0.0042071, -0.0001483] |

优化集自身的dense NLL差值为design `-0.007976`、selection `-0.008425`、
internal-confirm `-0.008954`。独立PPL46仍保持改善，因此V1不是只在selection上成立。

任务证据没有达到强判据：

- RULER Full-13×10：Native/V1 macro为`69.12%/70.79%`，差`+1.68pp`，
  95% CI `[-3.15,+6.44]pp`。
- Natural-QA99：Native/V1 task-equal macro为`39.41%/41.39%`，差`+1.98pp`，
  95% CI `[-2.75,+6.73]pp`。该面板不平衡，尤其HotpotQA仅5条。

所以V1支持“原生自然文本NLL存在可校准改善”，不支持“原生任务能力稳定增强”。

## 后续验证为何没有升级结论

### Split-consensus方向

三个design block存在共同一阶下降方向，局部margin为`0.01716`，selection选择
`alpha=0.25`；正方向也显著优于反方向。这只验证了局部符号控制，不等于一条
可迁移构表规则。

在PPL46上，consensus-plus相对Native的4K NLL差为`-0.000631`，95% CI
`[-0.003530,+0.002141]`，跨0；而它相对V1反而差`+0.001478`，95% CI
`[+0.000295,+0.002630]`。RULER相对Native为`-1.28pp`
（95% CI `[-5.63,+2.87]pp`），Natural-QA为`+2.73pp`
（95% CI `[-2.71,+8.19]pp`）。因此`strong_consensus_native_enhancement=false`：
consensus验证没有把V1的偶然/路径依赖解释排除掉。

### All-50 refit

all-50使用同一五自由度参数化、V1固定的lr与35步，不再划分selection。校准50本
上的NLL差为`-0.007168`，但复用PPL46的4K开发门只有`-0.001543`，95% CI
`[-0.003855,+0.000623]`；相对V1为`+0.000566`，95% CI
`[-0.000676,+0.001698]`。两个推进条件均失败：
`refit_4k_nll_ci_below_native=false`且`refit_4k_point_below_v1=false`。
因此流程按合同停在fresh-task confirmation之前，没有重复使用已看过的
RULER130/Natural-QA99宣称确认。

## 可用与不可用的论文表述

可用：

> 对一个冻结的成熟OLMo checkpoint，五自由度的post-hoc z校准在独立46文档
> 面板上显著降低了原生4K NLL，说明geometric Native并非该checkpoint在该
> 语言建模面板上的精确事后最优点。

不可用：

- “Native-Z5稳定提高了原生任务能力”——两个V1任务区间均跨0。
- “得到了一条普适的Native z规则”——候选依赖checkpoint、梯度和数据选择，
  consensus与all-50复验也未给出稳定升级。
- “任何模型都能通过改z变强”——只验证了一个checkpoint和一个自然文本面板。
- “零搜索解析方法”——模型权重虽冻结，但表由模型NLL梯度和selection校准得到。

## 远端证据源

服务器根：
`/root/autodl-tmp/today_rope_plan_20260914/olmo_native_z5_enhancement`

- V1汇总：`reports/native_vs_z5.json`
- V1优化与表：`optimization/optimization_result.json`、`optimization/table.json`
- V1逐文档NLL：`optimization/heldout_nll_rows.jsonl`
- V1任务原始输出：`runs/native/generations.jsonl`、
  `runs/native_z5/generations.jsonl`
- consensus汇总：`reports/native_z5_consensus.json`
- consensus逐文档NLL与表：`consensus/heldout_nll_rows.jsonl`、
  `consensus/table_plus.json`、`consensus/table_minus.json`
- consensus任务原始输出：`runs/consensus_plus/generations.jsonl`、
  `runs/consensus_minus/generations.jsonl`
- all-50结论：`all50_refit/refit_result.json`、`all50_refit/status.json`、
  `all50_refit/complete.txt`
- 冻结资产与合同：`assets/manifest.json`

这些远端JSON/JSONL是结果源；本文只负责压缩证据边界，不替代逐行复核。
