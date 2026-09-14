# Sol 夜间交接：PC2 + PMKeep

2026-09-09。今晚围绕这两个核心持续改进，目标是拿到真实任务收益。**不设自动时间截止，不自动关机；用户早上自行停止。** 不启动第三条研究主线。已有基线落盘后直接复用，方法失败时定位原因、改方法并继续，不重新跑一轮相同基线。

用户最新指令覆盖旧附件中的8小时上限、仅运行PM及禁止备用分支等限制。PM原始方案是[Pro决策文件](/Users/yang/Downloads/ICLR_Position_Research_Decision_Codex_20260909.md)。PC2与PM的结果分别归因。

## 最新执行要求：两个核心，十二个具体实验

用户明确要求立刻落实12项实验；两个核心不限制实验数量。以下E01–E12为实际执行安排，覆盖并扩展下文旧备用菜单。不能把12种方案思考、12个样本或12个对照臂称为12项实验。各项均包含生成质量比较。当前队列对应E01与E07；读取已完成结果，不为新编号重跑。接管任务负责顺序调度和方法改进，无自动截止或关机。

共同条件：PC2使用NOSA、16K公共四任务、原生reader和DMA；PM使用Qwen3B、8K自然文档、问题到达前压缩。DEV第一轮每任务4条，需区分效应时扩大DEV；最终确认使用尚未用于方法选择的TEST。记录逐样本真实输出、原定任务分数、EOS合同适用项、秒数、显存和方法状态字节。保持输入与预算相同的对照只计算一次并缓存。预算变化只补该预算缺失对照。成本优化不计入12项科研实验。

| ID | 实际干预及对照 | 要区分的问题和预期 | 失败后具体动作 |
|---|---|---|---|
| E01 PC2原始构造 | topk64，pc2 vs native/cobs_rank2/split2；沿用正在运行的首轮 | pair covariance是否带来真实检索收益，能否面对强对照 | 按共同样本失误进入E02/E03/E04；不将格式地板当作机制失败 |
| E02 PC2测度修正 | 仅换pc2_unweighted，reader/DMA不动；同E01对照 | CIS倾斜是否令selector偏爱干扰块；若是，取消倾斜应恢复漏选任务 | 无恢复则不继续扫CIS系数，检查E03与E04中的结构贡献 |
| E03 PC2跨频率修正 | pc2_rank1：主方向加pair残差；同E01对照 | 被丢弃的跨频率相关是否解释cobs优势；预计改善相关漏选样本 | 若仍不如cobs，定位高阶/多峰块，不重复增加同类低秩项 |
| E04 PC2机制消融 | weighted_mean vs pc2，同topk64；pc2已有生成直接复用 | 去除二阶项后收益是否消失；验证covariance的边际价值 | 若均值相同或更好，停止声称二阶贡献，依据实际错误修改协方差构造 |
| E05 PC2预算压力 | 固定DEV选定PC2版本，topk64降48；在48补native/cobs_rank2/split2 | 更紧读取预算下优势是否保留；质量和读取成本共同报告 | 若优势消失，限定64预算结论，按丢失来源改排序；不拿64对照比较48候选 |
| E06 PC2独立确认 | DEV锁定版本，topk64，在完整未用TEST与同预算强对照比较 | 开发收益是否迁移到未用于选择的输入 | 未复现就报告泛化失败；继续开发须用DEV，后续确认换新的未见输入，不反复调这份TEST |
| E07 PM原始构造 | uniform-prefix/H512/保留25%；F/E/E_author_policy/K/P/C/U；沿用当前队列 | 相位范围平均能否改善保留策略，超过合法强对照 | 由E08/E09分开识别query错配与相位平均过宽 |
| E08 PM近期query | 仅换recent-prefix，H512/25%；P/C/U同步换query；复用F/E/K | 近期query是否更接近未来检索，改善内容保留 | 若无恢复，不继续盲扫近期窗口；检查E09的相位问题 |
| E09 PM短未来范围 | uniform-prefix/H128/25%，其余不变 | H512是否抹平有效相位；较短范围应恢复特定任务 | 若不恢复且C/U更好，当前相位目标缺少任务价值，修改目标而非继续扫H |
| E10 PM交互检验 | recent-prefix/H128/25%，与E07/E08/E09形成2×2比较；复用三项结果 | 两个修正能否组合，有无互相抵消；不能把组合收益全归给一项 | 组合无益则保留单项有效构造；两项均无益也用这一项排查交互救援，随后改机制 |
| E11 PM预算压力 | 固定DEV选定policy/H，保留率25%降12.5%；同12.5%补E/E_author_policy/K，F可复用 | 紧预算下是否仍有质量—容量价值 | 若崩溃则定位被丢弃内容、保留25%结论，不用扩充缓存掩盖同预算失败 |
| E12 PM独立确认 | DEV锁定policy/H，25%，完整未用TEST对F及强对照 | 开发改进是否迁移，报告任务得失和实际成本 | 未复现则收窄结论，回DEV修方法；测试见过后不可再称独立确认 |

E05/E06与E11/E12必须明确填入DEV选定版本；选择依据为公共任务质量和成本，不按TEST挑版本。若DEV没有胜者，选DEV最好的合法候选，E06/E12仍可测“开发失败能否迁移/当前构造是否稳定无收益”，如实标为负结果确认，不能包装为成功确认。任务或实现失效时先修该具体缺口；若某项经已知证据已无区分力，接管者必须在同一核心给出替代干预、预测与生成比较后执行，不能静默删掉实验或重复失败凑数。

执行顺序：完成现有E01/E07；依据错误优先交错E02/E08、E03/E09、E04/E10；随后做E05/E11和E06/E12。四个DEV不够判决时扩大相应DEV，不先无差别扩大所有对照。12项完成后继续依据证据改进两核心，直到用户停止；不是机械跑完就结束。

### 已实现的逐项执行入口

`experiments/position_overnight/twelve.py` 默认只打印命令，加`--execute`才运行；与现有run_all共用`queue.lock`，不会启动第二个争抢GPU的队列。每项结果用独立目录保存，代码改动后换`--tag`；同一版本中断可用相同tag恢复。E01/E07已有结果直接认领，不用此入口再执行它们。

```bash
cd /root/autodl-tmp/position_overnight_20260909/code
TASKPY=/root/miniconda3/bin/python
# 例如E02；E03/E04/E08/E09/E10同样只换ID。
$TASKPY -m experiments.position_overnight.twelve --id 2 --tag v1 --execute
# 下列pc2与uniform-prefix/H512是显式示例值，接管者替换为DEV已选版本。
$TASKPY -m experiments.position_overnight.twelve --id 5 --pc2-method pc2 --tag selected_v1 --execute
$TASKPY -m experiments.position_overnight.twelve --id 6 --pc2-method pc2 --tag selected_v1 --execute
$TASKPY -m experiments.position_overnight.twelve --id 11 --pm-policy uniform_prefix --pm-horizon 512 --tag selected_v1 --execute
$TASKPY -m experiments.position_overnight.twelve --id 12 --pm-policy uniform_prefix --pm-horizon 512 --tag selected_v1 --execute
```

PC2支持`--chunk-size`及`--query-chunk`，采用接管者正在验证的高吞吐等价配置；不要将默认128/16视为已优化。`--per-task 0`扩大完整DEV；E06/E12始终使用完整TEST。运行前检查该项是否已存在等价结果，已有方法生成也可直接用于配对分析；runner内置缓存只承诺复用固定对照，并不自动跨目录复用候选。

## 机制研究与可执行补充（以真实实验推进，不设额外前置关卡）

用户最新因额度异常要求关闭核心任务的半小时唤醒：`pc2-pm`已设为PAUSED，不再后台唤醒本任务；接管任务`pc2-pm-keep`保持12分钟心跳并统一GPU调度，实验继续。核心任务完成现有准备后集中交接，最多两个子代理收口，不扩展后台工作，不使用重置卡。用户强调有效吞吐和实验价值：每次开新试验先明确事实、干预、比较及成功/失败如何改变下一步，不为占显存并发重复工作，也不因理论未完备拖延有效比较。

**立即顺序：**完成E01/E07；PC2优先E04，只补weighted_mean并复用PC2，再决定E02/E03。PM先得到正式初始生成，下面工具只用于具体DEV分歧，不挡主实验。

**共同失败优先于继续变体。** 用户指出：每核心六项共享基础假设，不能把六项都失败当六个无关失败。E05/E06/E11/E12主要检验预算与泛化，并不提供独立救援机制。发生连续同类失败时，先从已有原始输出找共同原因，再选一个能区分它的最小干预；不是必须跑完六项才允许改变构造，也不另建大审查流程。

| 共同原因 | 最小区分证据 | 对下一项的实际影响 |
|---|---|---|
| PC2评分改了，但关键支持集合几乎没变：mandatory/CIS配额掩盖QK排序作用 | 在同一已测DEV query上检查最终集合差异、哪些新增块真正进入reader；不能只看logmass差 | 不继续堆摘要阶数，先判断干预是否到达实际瓶颈；要改配额必须明确是一项新干预并配同预算控制 |
| PC2与rank/均值变体都在优化错误代理，或近似误差不在当前修正覆盖部分 | 用实际Q/K/CIS分开exact-vs-full二阶、full二阶-vs-PC2误差，再与具体生成得失联系；现有单点repair仅回答局部，不回答全历史 | cross误差主导才修cross；高阶误差主导才换表示；准确mass也不改善则检查value/历史形成/读取目标，不继续称更丰富摘要必有效 |
| PM所有分支都在用不代表后续问题的前缀Q | 观察P/C/U集合变化与任务收益；必要时用Full轨迹真实future-Q做明确标为oracle的诊断，不能回流部署或TEST选参 | 实际Q有恢复而前缀Q没有，说明要改可部署需求估计或保留策略，不再只调H/M；oracle无恢复还需查目标和干预范围，不能直接判主题死亡 |
| 两核心都只改善mass而没保护答案所需的读取信息，或共享任务/接口存在能力地板 | 结合完整模型/有效强对照与保留集合实际生成；value匹配对照或受控集合替换区分目标遗漏与无效接口 | 方法缺陷就改方法，共同提示/接口缺陷才共同修复；不因候选输就随意换数据，不把局部负结果包装成全面否定 |

若六项重复出现同一种错误却没有改变下一次干预，就是研究设计与判断没有发挥作用，不能归咎于运气或“研究本来会失败”。上述检查服务下一项选择；继续以真实比较推进，不等完整因果链证明才运行。

**当前判断需要修正。** `pc2_rank1`没有单调改善保证，不能默认排第一。令Σ=I+0.1vvᵀ、v跨两RoPE pair，CPU两块真实logmeanmass=[0.5061,0.65]，PC2=[0.525,0.65]排序正确，而rank1+pair残差=[0.8,0.65]错排：残差负cross相关被投影丢掉，破坏抵消。真实质量与成本应决定是否采用。现共同少量DEV有PC2独有收益和COBS独有收益，不能只挑赢家解释。

**PC2的可检验链条：**准确目标是`logZcis+a·mu+log E_w exp(a·(k-mu))`，其中`w=softmax(CIS), a=q/sqrt(d)`；PC2误差分为`0.5 aᵀ(Σ-DΣ)a + R3`，即cross-pair缺失与高阶余项。先由E04判断二阶有无任务边际价值；若要修rank，观察应支持cross项主导。若full二阶本身已错，可考虑两簇混合pair统计，但必须面对同簇mean-only和原强对照；该mixture尚未实现，不当作已准备作业。原生pair的特殊贡献还需同状态量随机pair控制，旋转等变性并非PC2独有。[COBS](https://arxiv.org/html/2607.09052v1)已有二阶、低秩和query子空间；本地COBS受控变体不等于完整论文复现。

E02不是修复double-count实现bug：selector近似`logsumexp(qk+CIS)`、reader保留实际CIS均有正确计算含义。去倾斜是在检验不同路由测度，不能赢后改写成“修好了重复加bias”。

**PM的可检验缺口：**真实query的分母含后来新增keys。删除旧集合D、剩余质量m时，精确有`F_S-F=sum_D a_j(F-v_j)/m`；仅保留mass不控制value方向与抵消。前缀Q也未必代表未来问题，改H不能生成缺失内容方向。官方[EA](https://arxiv.org/html/2510.00636v1)已有value norm，因此匹配value对照是检查目标遗漏，不是新颖性。value分散程度也有[近期工作](https://arxiv.org/html/2608.21541v1)，不能当首次发现。

**新增可运行资产（独立目录，不覆盖主队列）：**服务器`/root/autodl-tmp/position_overnight_20260909/prepared/mechanism_20260909`。

- `experiments.pm_keep.fast_scores`：FP32 batched GQA/global-prefix softmax，保持同一评分数学目标，减少逐head/小tile Python循环。CPU8项及小型算子检查通过；真实GPU速度和完整模型行等价尚待实测。独立目录下运行`flock -n /root/autodl-tmp/position_overnight_20260909/queue.lock /root/miniconda3/bin/python -m experiments.pm_keep.fast_scores --device cuda --shape qwen3b_11589 --query-batch-size 256 --warmup 1 --repeats 3 --output /root/autodl-tmp/position_overnight_20260909/runs/pm_fast_scores_gpu_v1.json`。这只测算子，不算方法质量；确认节省时间后在固定真实DEV输入核对集合和生成，才接入后续run，不据随机张量直接替换所有实验。无须重算已完成科学对照。
- `experiments.pm_keep.run_followup --value-objective raw_norm`：P/C/U同时乘原cached V的L2 norm，Q/H/预算/reader不变；EA/F/K保持原样复用。`value_only`只按V norm选，正式只运行P这一行（manifest另有目标标记），避免P/C/U重复。若三项同幅改善，只支持value贡献；若V-only同样好，就缺少昂贵位置评分的价值。`centered_norm`保留作平移不变性控制，不默认铺开网格。
- `experiments.pm_keep.causal_probe`：一个共同prefill与Q、其余层C，仅指定一层替换为P或U，并做同移出条目/同插入数量的随机sham，完整自由生成。P与sham的编辑数匹配；U是完整U集合，与P同KV预算但编辑数不一定相同。固定seed，不用未来问题/答案选集合。默认dry-run；`--execute`才占GPU，遵守queue.lock。CPU真实tiny Qwen接口与分支缓存隔离已验证；尚未有该probe的真实checkpoint质量结论。
- `experiments.nosa_position.causal_probe`已完成，CPU10项通过：共同prompt[:-1]历史，只在最后prompt token指定层做control/exact-mass/sham等量block交换，其余计算继续原candidate。精确mass读取所有keys，只是诊断访问；单点无恢复不否定整个prefill机制。

成功层级：开发信号值得继续；锁定版本在新输入上面对同预算强对照且收益/成本成立，才叫方法成功；匹配干预与消融进一步支持所声称机制。弱基线获胜、只有proxy改善、仅DEV调参收益、以显著额外成本换不实用微增益，均不能升级为完整成功。小样本未决和局部负结果也不意味着方向判死，下一步必须指向具体缺口。

```bash
TASKROOT=/root/autodl-tmp/position_overnight_20260909
TASKPY=/root/miniconda3/bin/python
cd "$TASKROOT/prepared/mechanism_20260909"
# 主队列释放lock且出现目标失配证据时，接管者运行这个固定小比较。
flock -n "$TASKROOT/queue.lock" "$TASKPY" -m experiments.pm_keep.run_followup --value-objective raw_norm --model "$TASKROOT/runs/pm_gpu_ready_20260909_v3/model_view" --data "$TASKROOT/data/pm_keep/rows.jsonl" --output "$TASKROOT/runs/pm_value_dev_v1" --baseline-cache "$TASKROOT/baselines/pm_keep" --split dev --per-task 4 --arms F E E_author_policy K P C U
flock -n "$TASKROOT/queue.lock" "$TASKPY" -m experiments.pm_keep.run_followup --value-objective value_only --model "$TASKROOT/runs/pm_gpu_ready_20260909_v3/model_view" --data "$TASKROOT/data/pm_keep/rows.jsonl" --output "$TASKROOT/runs/pm_value_only_dev_v1" --baseline-cache "$TASKROOT/baselines/pm_keep" --split dev --per-task 4 --arms P
```

## 入口与现有证据

服务器：`ssh -p 24941 [REDACTED_EMAIL]`。根目录`/root/autodl-tmp/position_overnight_20260909`；代码在`code`，数据在`data/pc2/rows.jsonl`与`data/pm_keep/rows.jsonl`，结果写`runs`，基线库写`baselines/pc2`与`baselines/pm_keep`。

模型：NOSA `/root/autodl-tmp/NOSA-1B`；PMKeep Qwen3B 使用已验证的加载视图 `/root/autodl-tmp/position_overnight_20260909/runs/pm_gpu_ready_20260909_v3/model_view`。原始3B目录缺分片索引，该视图通过真实shard header补齐索引，原权重未改。Python `/root/miniconda3/bin/python`；作者KVpress实现位于根目录`vendor/kvpress`。

NOSA首批36条DEV已报告：2K四项RULER recall均为1；16K single=1、multikey=1、multiquery=.25、vt=.2。24条CF完整exact全部0，多数生成换行。因此**当前有效主面板是公共检索**，尤其multiquery/VT；CF可做一次共同prompt repair，未修好便暂不拿其地板分裁决PC2。上述是首批DEV观察，不能冒充完整benchmark或独立确认结果。

启动后先看已有`status.json`、`load_report.json`、`generations.jsonl`，续做未完成工作；只补与新修改有关的最小检查，随后立即进入有效实验。付费GPU运行时CPU可并行分析已有输出、准备下一项修改。

```bash
cd /root/autodl-tmp/position_overnight_20260909/code
export PM_KEEP_KVPRESS_ROOT=/root/autodl-tmp/position_overnight_20260909/vendor/kvpress
TASKPY=/root/miniconda3/bin/python
TASKROOT=/root/autodl-tmp/position_overnight_20260909
NOSADATA=$TASKROOT/data/pc2/rows.jsonl
PMDATA=$TASKROOT/data/pm_keep/rows.jsonl
NOSAMODEL=/root/autodl-tmp/NOSA-1B
PMMODEL=$TASKROOT/runs/pm_gpu_ready_20260909_v3/model_view
```

统一入口：`$TASKPY -m experiments.position_overnight.run_all --tag initial`。默认先各任务4个DEV，顺序执行两核，不并发争GPU。扩展DEV用`--per-task 0 --tag dev_full`；只改PC2用`--core pc2 --pc2-method pc2_rank1 --tag pc2_rank1_v1`；只改PM用`--core pm --pm-query-policy recent_prefix --tag pm_recent_v1`。相同对照从缓存读取。`--dry-run`列出准确命令。入口第一批结束不代表研究目标完成，Sol继续根据结果改进。

## 基线只做一次，后续比较只追加方法

PC2主对照是native、cobs_rank2、split2；weighted_mean、quest已实现，可在相应问题出现时补一次。使用`--baseline-cache "$TASKROOT/baselines/pc2"`，每个模型/输入/预算/解码合同只算一次；候选命令中继续列对照名字，已完成项直接命中缓存，不重新生成，结果表仍能直接配对。

PMKeep使用`--baseline-cache "$TASKROOT/baselines/pm_keep"`复用固定F/E/K：F为完整cache参考，E/K采用当前代码所记录的作者实现适配；不能把适配称完整论文复现。P为PMKeep，C/U是对应操作消融。改变query-policy/horizon时P/C/U的操作变了，应产生新结果；不让无关参数变化强迫重算相同F/E/K。

复用依据是科学计算与输入一致；只改报告/无关代码不重跑baseline。若确实修改了baseline的计算、模型、prompt、decode或读取预算，则保留旧结果、只补受影响比较。每个方法修订用新output目录留原始输出和源码快照，不覆盖失败证据。

## 两个核心先各得到第一轮有效结果

PC2保持reader、原生RoPE和DMA不动，只改selector。先在四类公共DEV上跑当前PC2并与缓存基线比较；不用CF格式地板否定它。PC2已有均值加pair covariance结构，不能把二阶展开或协方差统计称为首次提出。

```bash
$TASKPY -m experiments.nosa_position.run --model "$NOSAMODEL" --data "$NOSADATA" --output "$TASKROOT/runs/pc2/core_dev16k_b64" --selectors native cobs_rank2 split2 pc2 --split dev --lengths 16384 --tasks niah_single_1 niah_multikey_1 niah_multiquery vt --per-cell 8 --topk 64 --baseline-cache "$TASKROOT/baselines/pc2"
```

PMKeep使用prefix-only query样本和预定未来范围估计保留价值，问题到达前完成保留；问题和答案不得用于打分。第一轮一起跑F/E/P/C/U/K，已有F/E/K从缓存命中；重点看P能否改善真实生成并超过相关强对照。

```bash
$TASKPY -m experiments.pm_keep.run --model "$PMMODEL" --data "$PMDATA" --output "$TASKROOT/runs/pm_keep/core_dev_uniform_h512" --arms F E P C U K --query-policy uniform_prefix --horizon 512 --split dev --baseline-cache "$TASKROOT/baselines/pm_keep"
```

两核顺序按GPU实际就绪情况安排：一条遇到明确工程阻塞时先运行另一条已准备好的比较，CPU修阻塞；不是启动无关实验填满GPU。已有健康作业不中断。

## 四条预写备用分支：有触发就继续改，不必再询问

**1. PC2u：检查CIS重复偏好或测度错配。** 若weighted_mean与PC2都偏向同一批高CIS干扰块，或PC2比native下降而pair结构本身没有证据被否定，运行不带CIS倾斜的`pc2_unweighted`。主reader的原生DMA保留；这检测selector测度，不是关掉整网DMA。

```bash
$TASKPY -m experiments.nosa_position.run --model "$NOSAMODEL" --data "$NOSADATA" --output "$TASKROOT/runs/pc2/u_dev16k_b64" --selectors native cobs_rank2 split2 pc2_unweighted --split dev --lengths 16384 --tasks niah_single_1 niah_multikey_1 niah_multiquery vt --per-cell 8 --topk 64 --baseline-cache "$TASKROOT/baselines/pc2"
```

若u恢复收益，沿正确的selector测度继续，报告收益来自哪里；若没改善而强对照也无收益，改查可恢复source与reader，不继续盲调CIS系数。

**2. PC2+rank1：补被丢弃的跨频率相关性。** 若cobs_rank2比PC2好，或真实激活显示明显跨pair相关且影响漏选，运行`pc2_rank1`：`Σ̂ = Σ_rank1 + pair(Σ−Σ_rank1)`，只对残差做pair投影，避免双计第一主方向。

```bash
$TASKPY -m experiments.nosa_position.run --model "$NOSAMODEL" --data "$NOSADATA" --output "$TASKROOT/runs/pc2/rank1_dev16k_b64" --selectors native cobs_rank2 split2 pc2_rank1 --split dev --lengths 16384 --tasks niah_single_1 niah_multikey_1 niah_multiquery vt --per-cell 8 --topk 64 --baseline-cache "$TASKROOT/baselines/pc2"
```

若该分支胜，检查额外状态/构建代价并面对COBS强对照；若仅统计误差改善而生成不涨，保留诊断，不宣布任务收益。对稀有尖峰导致二阶过估，可据观察另做有理由的稳健化，仍属于PC2核心，不必重启方案审批。

**3. PM-recent：修query分布错配。** 若uniform-prefix样本偏向早期局部语境，P保留的内容与后续问题需要明显不符，用recent-prefix样本；统一改变P/C/U的query样本，F/E/K从缓存复用。

```bash
$TASKPY -m experiments.pm_keep.run --model "$PMMODEL" --data "$PMDATA" --output "$TASKROOT/runs/pm_keep/recent_dev_h512" --arms F E K P C U --query-policy recent_prefix --horizon 512 --split dev --baseline-cache "$TASKROOT/baselines/pm_keep"
```

若recent改善，继续检验近端query是否可代表未来检索；不把此结果泛化成任意Agent未来查询都可预测。若仍失败，区分query方向不足和相位范围抹平。

**4. PM-H128：减轻未来范围过宽的相位平均。** 若P相对C/U在H512下出现明显低模长、过度平滑，且任务实际问答更短，先仅把horizon改为128；保持uniform-prefix以隔离该改变。

```bash
$TASKPY -m experiments.pm_keep.run --model "$PMMODEL" --data "$PMDATA" --output "$TASKROOT/runs/pm_keep/uniform_dev_h128" --arms F E K P C U --query-policy uniform_prefix --horizon 128 --split dev --baseline-cache "$TASKROOT/baselines/pm_keep"
```

只有两项单独证据支持时再组合recent+H128。上述分支不是封闭菜单：Sol可根据真实错误自行修改这两个核心；每次写一句改动机制与预期可区分现象，然后直接做最小有效比较。

## 拿到收益后怎么确认，没赢时怎么继续

方法开发优先公共任务的真实自由生成；proxy、attention mass和几何反例用于定位错误。public按官方recall报告，CF按完整字符串+EOS另报，PM自然任务采用数据声明的任务指标；保留全部raw outputs。观察过的数据叫开发数据，不叫holdout；不要用固定测试集调方法后声称未见泛化，选定版本再用尚未用于决策的数据确认。

PC2若某版本胜出，用同一版扩大独立public样本，再测topk48确认预算变化；已有相同合同baseline直接读取，48缺失的baseline只补一次。PMKeep选定query-policy/horizon后，在未用于方法选择的test split运行P与强对照缓存比较。对照全胜不是强制条件，但要清楚主要改善与代价；部分任务涨、部分降就如实解释，不靠平均掩盖。

失败继续指向方法：PC2只胜native却不胜weighted_mean，优先改出协方差的边际价值；PM只胜弱消融不胜E/K，继续处理query估计/保留目标与强对照的差距。实现错误修代码，格式错误修共同提示；基本能力不足不误判位置机制。只有明确训练能修的失配且具备独立训练数据时才匹配校准；不拿test调参，也不因没赢就从零训新模型。

当前没有候选能承诺accept。目标是研究到出现可信收益，并把有效负结果用于下一次具体改进；不是跑完预定表格就停止，也不是维护已被结果否定的解释。用户停机前持续处理这两个核心的有价值工作；不添加倒计时、自动kill、自动关机或到点终止任务。

## 真实状态由接手者追加

- NOSA真实311 tensors/28层/1,622,179,896参数加载与36条DEV生成完成：`/root/autodl-tmp/nosa_position_20260909/runs/native_dev_pilot_01/`。PC2原生参考后端标签必须保留；不是官方CUDA kernel速度复现。
- PMKeep真实GPU：`runs/pm_gpu_ready_20260909_v3/`，同执行方式keep-all logits/cache差0、greedy一致。完整8K单例F/K成功，E/P/C/U失败；单例不裁决方法。正式模型加载与六臂生成可运行。
- 已部署并启动首批双核队列：主PID **4164**，当前PC2 child **4165**，`queue_initial.json`状态RUNNING；PC2已从缓存命中首个native结果，随后运行cobs_rank2/split2/pc2，完成后自动进入PM。36条既有native生成已导入`baselines/pc2`。
- 用户指定由Astra low任务 `01a0870a-cc6e-7262-ae4c-300394ae57b3` 接管持续研究；保持该任务的模型设置，不重启重复队列。
- 最新最佳方法、领先的任务、尚未解决的错误：每轮只更新此处一小段，不另开重复文档。

实时汇总：`$TASKPY -m experiments.position_overnight.report --runs "$TASKROOT/runs/pc2_dev_initial" "$TASKROOT/runs/pm_dev_initial" --output "$TASKROOT/reports/initial"`，生成CSV、配对差和DECISION.md。只有已完成共同样本进入配对统计，缺失不记0。`queue_initial.json`记录队列PID/当前child；各run有自己的status与raw JSONL。不要另开重复队列抢同一GPU。

### 夜间接管 2026-09-09T16:43:40.903235+00:00

本任务 `01a0870a-cc6e-7262-ae4c-300394ae57b3` 已按用户指令接管后续实验。心跳 `pc2-pm-keep` 已启用，每12分钟接续本任务（用户因token额度于接管后调整）；没有自动截止或自动关机。已实查 RTX 4080 SUPER 与现有队列 PID 4164 / PC2 child 4165，PC2 初始64个 row-arm 比较从1推进到2，native命中已有baseline缓存，cobs_rank2完成16251-token真实输入生成；PM会由同一队列随后启动。当前样本不足以选择方法分支，保持健康队列不打断、不重复启动。下一次读取新增raw输出与queue状态，完成共同样本后决定PC2备用分支；PM按既定初始比较执行。服务器没有rg，远程检索用grep或Python。

### Execution handover 2026-09-09T17:01:33.486339+00:00

Same 16251-token input, 128/16 -> 128/64: native prefill 20.856 -> 13.729 s (779 -> 1184 tokens/s); PC2 23.190 -> 18.747 s (701 -> 867 tokens/s). Both final prefill logits and full generated tokens are exactly equal. 512/64 is faster but changes logits (max 0.3125 native, 0.390625 PC2); not adopted. Receipt: runs/chunk_probe_v1/results.json. Adopted 128/64; run_all and twelve use baseline query chunk 16 to reuse verified-equivalent cached controls. This is one-input execution evidence, not a task-quality gain.

Original PC2 stopped at a full-row boundary with 32/64 results; PARTIAL_BUDGET is its generic SIGTERM label, not a deadline. Those raw rows were copied unchanged into runs/pc2_dev_initial_q64/generations.jsonl; original directory preserved. New child PID4999, supervisor4998; queue_q64_resume.json owns this continuation. Parent4164 is intentionally paused until supervisor finally resumes it, then the original PM command runs automatically. Check both queue files; do not start a competing queue. New run already advanced to34/64. Analyze q64 directory as E01; copied rows are not independent replications.

Latest 12-experiment plan received: E01/E07 reuse existing runs, then E02/E08, E03/E09, E04/E10, DEV-selected budget pressure and TEST confirmation. Expand PM to64 DEV before choosing and128 TEST after locking. Continue within two cores after the12 experiments as authorized; no automatic shutdown. Heartbeat remains12 minutes.

### Joint execution review 2026-09-09T17:16:47.611210+00:00

User explicitly requested both tasks to reason together. Agreed priority: finish E01/E07, then E04 (weighted_mean only is new work; import PC2), before treating E03 rank1 as promising. PC2 rank1 can break cross-pair cancellation; E02 tests a different selector measure/routing policy, not a duplicated-reader-bias bug. Do not wait for new causal tools before real generation.

E01 q64 completed; parent4164 resumed and PM child5671 now runs. E04 supervisor5828 is queued under the SAME queue.lock, status queue_e04_mean_check_v1.json (WAITING until PM ends). It will execute E04 with --reuse-from runs/pc2_dev_initial_q64; do not enqueue E04 again. Explicit candidate reuse is now available in both runners and twelve via --reuse-from RUN_DIR... . Only matching candidate computation/config/model/input/scoring rows import; fixed controls still use existing baseline cache. Imports preserve raw token outputs and timings and carry reused_candidate/reused_from_run metadata. Three CPU tests passed locally and remotely. No metric/reader/scoring formulas changed.

For PM full-DEV expansion, use a new tag, --per-task 0, and --reuse-from runs/pm_dev_initial to avoid regenerating initial P/C/U; different policy/H/value objective cannot reuse candidate rows. Actual PM natural DEV lengths: Hotpot 5816-11589 and2Wiki5116-8411 tokens, not uniformly8K. Preserve full documents and compare same-input same-fraction controls.

## Pro 方案底层分析与故障预案（2026-09-09 再次深入）

用户本轮要求深入分析原始 Pro 决策文件并做好预案。本节完整核对其第4–10节、当前PM源码、固定版本EA源码和已经完成的首轮生成。保留上文两个核心的执行授权与统一GPU调度；本任务的半小时心跳仍暂停。本次新增的是推导、CPU有限算例和结果分析，没有启动新GPU实验，也没有修改正在执行的runner。

### 1. 真正要保护的对象是“不同需求下的读取选择”

记前缀post-RoPE keys为K，内容query为u，未来位置为t。当前P与C的差异准确写成：

\[
\pi_j=\mathbb E_{u\sim\widehat P_U,t\sim\nu}
 \operatorname{softmax}_j(KR(t)u/\sqrt d),\qquad
\pi^C_j=\mathbb E_u
 \operatorname{softmax}_j(K\bar Ru/\sqrt d),\quad\bar R=\mathbb E_t R(t).
\]

先选集合、再看未知问题，是一个信息受限的决策问题。固定保护区和每头B个槽时，按精确\(\pi\)排序，确实最大化**这个代理分布下**的期望保留mass。P用抽样近似它；C先删去了位置引起的query变化。两种位置各自强烈读取不同key时，先平均query可能把一个“始终中等”的干扰key排到最前；保留不同读取情景可以避免这种特定损失。这是方案最有力的构造理由。

原始K/V、模型权重和实际读取位置均不变，贡献发生在未来读取需求的估计。实现中的“合法位置”仅保证同一个query各频率使用同一个t；它不证明重新定位的u就是模型在未来历史上会产生的query。pre-RoPE向量仍含历史和位置相关信息。

三个操作需分开：位置平均、内容分布近似、归一化顺序。EA还使用全前缀内容统计与Gaussian MGF、期望分子的归一化、V norm；固定版本源码在评分分母中排除了四个sink，再强制保留它们。P/C/U分母保留全部prefix。因此P对E是实际方法比较，**P对C才是当前匹配的位置处理比较**，不能把P对E的差异全部归因于平均RoPE。[EA原文§2.2](https://arxiv.org/html/2510.00636v1)；本地作者源码固定于`71640b4f9061054a7630c5049bb9ee659a01523c`。

[GVote原文§3.2](https://arxiv.org/html/2509.03136v1)已经逐样本选key、用未来平均cos/sin并做union；其预算和聚合目标不同。未来采样本身不能算本方法的新意。上述数学工具与以下推导也不单独构成新颖性证明。

### 2. 比“协方差丢失”更接近实际选择的两个结论

**应看logit差的变化，而非单条key的方差。** 原文件的
\(\Delta C=\mathbb E[(R-\bar R)\mathbb E(uu^T)(R-\bar R)^T]\succeq0\)
在product proxy下成立。但是softmax对所有logits共同加一个数完全不敏感。令
\(K_c=(I-\mathbf1\mathbf1^T/T)K\)，应进一步观察

\[
\Gamma=K_c\Delta C K_c^T/d.
\]

它是位置坍缩丢失的**logit差空间**协方差。若\(\Gamma=0\)，则在该proxy下，\(K(R(t)-\bar R)u\)几乎处处只有共同偏移，P/C的精确attention相同。证明由\(\operatorname{tr}\Gamma=\mathbb E\|K_c(R-\bar R)u\|^2/d=0\)直接得到。若\(\Gamma\ne0\)，仍不保证平均概率或top-k变化；对称抵消和cutoff间隔都可能消除影响。\(\Gamma\)也不是完整softmax误差公式。

因此：不按\(\operatorname{tr}\Delta C\)大就选择层，不按单key的\(k_j^T\Delta Ck_j\)大就预测收益；优先看真实cutoff附近**交换候选之间**的logit差与最终进入reader的集合。实现上不需要构造T×T矩阵，少量key对的\((k_i-k_j)^T\Delta C(k_i-k_j)/d\)就能检查此问题。只有出现待解释的DEV分歧才计算。

CPU新增有限例：三条key丢失的logit方差均为3.13355，但全部来自共同偏移；\(\Gamma=0\)，P/C概率差精确为0。这说明很大的方差图可以完全没有选择价值。

**U恢复单位模，不一定恢复一个合法的共同位置。** 对均匀H个位置，\(\bar R_k=\lambda_kR_k(t_c)\)，其中\(\lambda_k\)可为负。U将其改为\(\operatorname{sign}(\lambda_k)R_k(t_c)\)，不同频率可能各自额外转\(\pi\)，通常找不到共同t。

严格反例取H=8、频率1和3/10、起点0。第一pair的\(\lambda<0\)，第二pair的\(\lambda>0\)。若U等于某个\(R(t)\)，必须同时有
\(t=3.5+(2n+1)\pi\)与\(0.3t=1.05+2m\pi\)，因而\(3+6n=20m\)，左奇右偶，矛盾。这里连任意实数位置都不成立。CPU只检查了有限整数区间，确认U仍正交、却不等于任何区间内位置；无限范围结论来自上面的代数证明。

实验含义：P优于U，并不足以单独证明“需要完整位置分布”，也可能是合法共同相位比这个逐pair控制好。如果P在真实任务上出现值得归因的收益，再加**预先固定的单一合法位置**\(t=T+\lfloor(H-1)/2\rfloor\)，保持同内容样本M和同预算。它用于区分多位置情景与单点合法旋转；不现在另排一组无收益条件下的大消融，不扫描位置挑赢家。若U已同样好，则原构造的幅度恢复已足够，仍不能坚持多位置分布必要。

### 3. 实验能赢的条件可以写成一个明确的差值

先固定一条共同参考轨迹的query和所有可见K/V，包括后来的问题与生成keys。记\(b_j(q)\)为prefix内归一化attention，\(\alpha(q)\)为prefix占全attention分母的份额，\(a=\mathbb E\alpha\)。当a>0时定义

\[
\rho_j=\frac{\mathbb E_{\rm real}[\alpha(q)b_j(q)]}{a}.
\]

\(\rho\)是按“实际有多大程度读取旧记忆”加权的真实需求，而非每个prefix查询等权。令\(d_j=1_{j\in S_P}-1_{j\in S_C}\)，两个集合同预算、同保护区，则**固定参考轨迹上的真实保留mass差**准确等于

\[
\Delta_{P,C}=a\,\rho^Td
=a\underbrace{\pi^Td}_{\text{代理位置决策收益}}
+a\underbrace{(\rho-\pi)^Td}_{\text{真实需求错配}}.
\]

精确P集合使第一项非负；有限M实际集合没有这个确定保证。第二项可以反向且更大。CPU构造的第一项未乘a前为+0.30，错配项为−0.95，a=0.6，实际差为−0.39。修复平均RoPE可以完全正确，同时优化了不代表未来任务的需求。

这给预案一个可操作顺序：**相位处理是否产生有用的proxy集合收益 → 该收益是否转移到实际future-Q → 保存的信息是否改变最终答案。** 不把任何一环的成立当作后一环的证明。全模型自由生成中，不同集合还会改变后续Q/K、轨迹和EOS；上述共同轨迹公式是诊断合同，不是两条自由生成轨迹差异的精确分解。

令r为保护区外名额，\(\hat\pi\)为实际估计，真实固定轨迹最优集合为\(S^*\)。有确定性界

\[
a[\rho(S^*)-\rho(\hat S)]
\le 2ar\|\rho-\hat\pi\|_\infty
\le 2ar\big(\|\rho-\pi\|_\infty+\|\pi-\hat\pi\|_\infty\big).
\]

证明：插入\(\hat\pi\)，利用\(\hat S\)对它的最优性，剩下两组至多r项误差。若a=0，旧cache的该项贡献为0。这个界区分**proxy偏差**与**采样/数值误差**，通常很松，不作为M=256的排序认证。更大的M只能减小后一项；改变H仅调整位置分布，也不能补出未被内容proxy覆盖的检索方向。用P自己的采样给P评分有选择乐观偏差；需要解释proxy改进时，用独立固定小批proxy或真实future-Q诊断，不能用训练式自评分冒充转移证据。

prefix内容Q以正文续写需求为主，未知问题可能需要跨段关系或精确实体查找；recent-prefix仍可能只是最近一段正文。因此E08不是必然修复，E09也没有“越短越好”保证。最有希望的适用区间是：内容需求可以由前缀合理覆盖，位置变化会改变关键key之间的竞争，预算能容纳这些不同需求，而且被保住的读取信息会影响任务答案。

### 4. mass到答案仍有独立缺口

同一query下，删除集合D、剩余全attention质量m时，精确有
\(F_S-F=\sum_{j\in D}a_j(F-v_j)/m\)。因此mass只能给带\(V_{\max}\)的上界，不能识别value方向、抵消、\(W_O\)与后续网络对该方向的敏感性。均匀聚合GQA关联heads也只是一个声明的目标，不代表每个head对任务同等重要。

value norm是已有启发式：所有V同时加同一个向量时，原输出与压缩输出都平移相同向量，二者之差不变，但raw norm排序可以改变。因此已准备的`raw_norm`、`centered_norm`不是最优性保证；先用raw_norm对P/C/U同步干预及V-only廉价对照区分“没有衡量输出贡献”与“位置项仍有边际价值”。只有证据指向平移敏感性时才用centered控制，不能直接展开三目标网格。

同样，期望mass不等于每个未知问题都能保住全部证据。多跳任务常需要若干低频证据同时存在，独立重要性排序没有集合互补性的保证。这是解释多需求失误的候选原因；必须看到具体丢失证据或匹配恢复才成立，不能仅因题目是多跳就改成新coverage算法。

### 5. 已完成首轮：16个输入，不是112个独立样本

本轮远程读取`runs/pm_dev_initial/status.json`确认COMPLETE、112/112，并拉取完整原始JSONL。四任务各4个输入、7个arm；固定25%prefix槽位。自然任务为context-first未知问题协议，分数是F1；检索为完整字符串加terminal EOS。以下0–100分属于DEV小样本，不能给正/负泛化结论。

| 任务（每项n=4） | Full | EA统一保护 | EA作者保护 | P | C | U | KeyDiff适配 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2WikiMQA | 100.00 | 50.00 | 50.00 | 44.23 | 44.23 | 44.23 | 0.00 |
| HotpotQA | 69.40 | 39.17 | 37.50 | 37.50 | 41.07 | 41.20 | 15.28 |
| Multi-KV order | 25.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 25.00 |
| Single-KV | 100.00 | 50.00 | 50.00 | 50.00 | 50.00 | 25.00 | 100.00 |

P/C逐输入分数15项相同，1项P较低；没有P对C的任务得分胜例。P/U互有一个得失。16个输入的P/C/U完整集合hash均不同，说明集合非完全相同，**hash不能告诉我们重叠率或是否只换了少量无关条目**，不能据此声称大幅改变选择，也不能立即判`STOP_NO_SELECTION_EFFECT`。

已有短接口回执：相同native增量路径与adapter的logit差为0、相同集合路径差为0、greedy一致；单次整段prefill对增量native有独立的BF16分区数值差异，不能把它隐藏，也不能误归为PM独有错误。这只核对了短接口，未把整个科学实验“认证无bug”。

当前最有区分力的已有案例是：`single_kv_dev_000/001` Full与KeyDiff能答对而P/C/U及EA均失败，适合检查共享query/mass目标；`2wikimqa_dev_001/003` Full能答对但各压缩臂失败，可检验压缩信息损失；`hotpotqa_dev_003` P对C退化，但各臂答案表述与F1都较复杂，不宜独自承担干净的机制结论。`single_kv_dev_003` P/C答完整字符串、U少最后字符，不能由此一例断言U的位置几何全面失效。包含得失的共同样本用于DEV诊断，后续成功确认仍用未见输入。

### 6. 触发后做什么：每次修复针对一条被证据支持的原因

| 触发信号 | 下一项最小有效动作 | 会改变的决定 |
|---|---|---|
| 同保留集合、相同增量路径仍输出不同，或问题泄漏/位置重编号 | 修已定位接口；只使受影响结果失效并重跑 | 工程责任明确，不把接口故障算方法阴性；已有效基线保留 |
| P/C仅少量边界交换、采样种子导致主要排序反转 | 先复用固定DEV张量看cutoff及两额外预定种子；不按答案挑seed | 采样噪声主导才处理估计精度，不先扩大H或更换内容目标 |
| P/C/U共享错误，而Full/合法强对照能做 | 按现有E08/E09/E10的2×2设计区分内容proxy和位置范围；扩大必要DEV，复用基线 | 有内容修复只归因内容，有H修复再查位置作用；连续无恢复不重复扫同类参数 |
| 实际future-Q排序能在匹配干预下恢复，prefix-Q不能 | 以已准备的共同cache/局部干预定位；明确oracle访问和轨迹范围 | 改可部署需求估计；不将oracle当成绩，不认为多采位置可解决内容缺失 |
| mass改善但输出/答案不改善，或value差异解释现有失败 | 运行已准备的P/C/U相同raw-V权重与V-only控制，固定其余配置 | value-only同样好就没有昂贵位置评分的边际证据；组合有益须分开归因 |
| P真实优于C/U、且优于EA有可重复迹象 | 再补固定合法单位置；需要时做少量paired集合替换/sham | 确定是多位置情景、合法相位或简单幅度修复，随后冻结做TEST |
| 局部repair无恢复 | 先确认该层/时点实际改变了所需读取 | 局部阴性不能否定全历史机制；不为解释局部阴性自动铺全层搜索 |
| 质量可重复改善但评分昂贵 | 对已准备`fast_scores`做真实GPU与固定输入等价检查，复用所有科学对照 | 数学不变的加速只修系统开销；相同budget若PM和EA decode近似一致，PM额外评分不能靠decode凭空摊平 |
| 开发有益，新输入未复现 | 报告实际效应与配对区间，继续开发用DEV；再次确认须另留新输入 | 修泛化缺口，不反复查看同一TEST改参数后仍称独立确认 |

成本另有简单判据：相对Full，如每个后续token节省\(\Delta\tau>0\)、额外压缩成本为c，则仅在\(N>c/\Delta\tau\)时可能摊平这部分时间；两种同预算方法若每token成本相同，则不存在此类时间回收。共享前缀多问题可以摊销一次评分，必须实测；一次full-prefill再压缩的峰值、配对实验保留的完整缓存、可部署单分支常驻bytes分别报告。时间较慢仍可能有质量/容量价值，准确给出取舍，不自动否定全部用途。

**本次完成的预案资产：**本节推导；完整112行首轮结果复核；`results/two_core_analysis_20260909/pm_core_audit.py`和对应JSON的4组CPU有限检查（contrast null、U共同位置反例、真实mass差式、选择regret界）；上文现成value/causal/fast-score入口。CPU算例不新增任何语言模型正结果。原文件中的“可能有效”应由接下来有区分力的生成比较兑现；实现和实验判断由执行者负责，不能把保证研究命题必胜当作工程验收条件。

### Next interventions 2026-09-09T17:42:11.637987+00:00

E04 completed32/32, including16 reused PC2 observations: PC2 vs weighted_mean recall multikey .50/.00; multiquery .3125/.1875; single1/1; VT .40/.55 (4 rows per task). Covariance has task-dependent marginal effect, not a universal win. PM initial112/112 has no P>C score wins (15 ties,1 loss). Do not call seven arms112 independent samples.

Launched E02 unweighted_v1 and queued E08 recent_v1 under common queue.lock; inspect queue_e02_unweighted_v1.json and queue_e08_recent_v1.json before further scheduling. E02 tests selector measure/quota preference while reader/DMA are unchanged. E08 tests prefix-query proxy shift, sameH512 and25percent retention. Fixed controls reused; only candidate configurations changed. After these finish choose E09/E10 or value followup according to real errors, not a blind parameter sweep. E03 still part of plan but rank1 is not presumed monotonic. Keep common failure mechanisms in view; budget/TEST runs are not separate rescue hypotheses.

### Heartbeat decision 2026-09-09T17:55:40.277742+00:00

E02 and E08 COMPLETE. E02 unweighted equals PC2 on multikey(.5), multiquery(.3125), single(1), but VT drops.4 to.2; no evidence for removing CIS tilt. E08 recent-prefix P/C:2Wiki .5 vs original.4423, Hotpot .125 vs originalP.375/C.4107, single .25 vs original.5, multi-order0. Shared proxy errors persist; recent is not a general repair. Only4 DEV rows/task, no independent confirmation.

Started E09 h128_v1 and queued E03 cross_rank1_v1 with the shared GPU lock; inspect queue_e09_h128_v1.json and queue_e03_cross_rank1_v1.json. E09 isolates horizon from content proxy. E03 is justified by COBS multiquery.5 vs PC2.3125 together with E04 showing some covariance value, but rank1 is not monotonic and may hurt other tasks. Preserve original controls. Next review: completeE10 interaction if needed; if PM content/H still jointly fail, change objective/query estimation using existing value tools rather than more H/M settings. Local raw mirrors are results/position_overnight_20260909/; no model downloads.

### Heartbeat decision 2026-09-09T18:09:42.001716+00:00

E09 H128 COMPLETE112/112. Uniform H128 P/C/U all recover multi-order1/4 vs H5120/4; single stays P/C2/4, U improves1/4 to2/4. 2Wiki all.4423 unchanged; HotpotP.3967/C.3950/U.4135. This supports a horizon effect on some DEV cases, not a PM-specific improvement. No broad H sweep.

E03 cross_rank1_v1 is healthy44/64 (child7642): partial3-row multikey rank1.6667 vs native/cobs.3333, multiquery rank1.25 vs cobs.3333. Do not select from incomplete task means. E10 recent_h128_v1 queued behind E03 under same lock to finish the planned2x2 content-policy/horizon comparison. After both finish, decide DEV expansion and value-objective correction. Preserve all task-specific tradeoffs, especially P-C shared improvements and weak Full multi-order basic capability.

### Heartbeat decision 2026-09-09T18:24:51.190034+00:00

E03 COMPLETE64/64:rank1 multikey.75 (native/cobs.5), multiquery.4375 (native.25,cobs.5), single1, VT.2 (native/PC2.4). Promising retrieval gain plus VT harm, only4 DEV rows/task; no test selection yet. E10 recent/H128 COMPLETE:2WikiP/C/U.5;HotpotP.1435/C.1417/U.125;multi-order0;singleall.25. Combined recent/H did not rescue initial proxy errors. No more H/M variants now.

Started queue_value_pc2_devfull.json using root/value_then_expand.py under the common lock: (1)pm_value_h128_v1 from prepared/mechanism_20260909, uniform/H128, P/C/U all multiplied byraw V norm, fixedF/E/E_author_policy/K reused; (2)pm_value_only_v1, onlyP label used for pureVnorm control (not PM), same16 DEV; (3)pc2_rank1_dev_full_v1, all32 public16K DEV rows, importing16 prior rank1 rows via--reuse-from e03_cross_rank1_v1 and reusing existing fixed controls. This tests whether mass objective ignores useful output magnitude and expands the PC2 improvement/tradeoff beyond4 samples/task. Value-only equal quality would undercut phase attribution. Keep scientific quality separate from previously prepared fast-score execution asset.

### Heartbeat decision 2026-09-09T18:39:48.572714+00:00

PM value16 DEV COMPLETE. raw_norm P/C/U:2Wiki .6923/.5909/.5909 (originalH128 .4423 each;EA.5), Hotpot .4044 each (EA.3917), multi-order .25 each, single .5 each. PureVnorm:2Wiki .6923,Hotpot .25,order0,single0. Value conditioning improves some outputs; 2Wiki gain is also achieved by V-only, so phase-exclusive claim is not established. Combined scores outperform V-only on other tasks.

PC2 full DEV is healthy80/128 with five paired rows/task, continue untouched. Queued queue_pm_value_devfull.json (root/pm_value_devfull.py) behind it:64 DEV raw_norm uniform/H128 all7 arms, then64 DEV V-only(P label). Each imports prior matching16 candidate rows and baseline caches; no repeated initial generations. New isolated prepared/pm_value_devfull_20260909 copies the delivered method unchanged and only adds current runner/reuse helper; original prepared mechanism snapshot untouched. Candidate source/config fingerprints remain matched. Wait for these DEV results before lock/TEST; do not enqueue duplicate expansions.

### Heartbeat decision 2026-09-09T19:11:08.516386+00:00

PC2 full32 DEV COMPLETE. rank1/native/COBS: multikey.75/.375/.75; multiquery.59375/.5/.625; single1/1/1;VT.3/.4/.3. No quality advantage over COBS at this scope. q64 timing medians rank1 prefill37.27,total39.08s;COBS36.04,38.83s (rank1n32,COBSq64n24; descriptive, not matched speed claim). rank1 descriptor25.65MB vsCOBS21.99MB. Do not declare quality-cost win or keep adding rank. PM64 value DEV healthy201/448 when checked, continues untouched.

Queued queue_pc2_exact_mass_bridge.json after current PM queue: only3 new full-model exact_mass generations on existing multikey002 and multiquery000/002; native/COBS cache reused, PC2 and rank1 rows imported. New exact_probe.py vectorizes the existing explicit exact_mass reference, keeping same causal visibility, GQA normalization, NOSA quota and reader throughout prefill/decode. CPU comparison against explicit per-block reference plus future-key perturbation passed. This asks whether removing logmass approximation errors changes task outcomes; it is not a general answer upper bound or a deployment-cost claim. Samples deliberately cover previous gains/losses and are diagnostic DEV, not new confirmation. Outcome will decide whether more moment approximations are justified, before automatic budget/TEST expansion of an uncompetitive construction. Original12-work-item objective remains; avoid treating repeated shared failures as independent rescue ideas.

### Heartbeat decision 2026-09-09T19:30:24.837250+00:00

PM64 value DEV COMPLETE. P/C/EA task scores:2Wiki.3533/.3123/.3721;Hotpot.2699/.3074/.3577;order.0625/.0625/0;single.3125/.3125/.4375 (KeyDiff1). Value-only2Wiki.4189,Hotpot.2801,order/single0. Expanded value candidate does not beat strong controls; key shared problem remains content-demand proxy/retention objective, not just horizon or missing V magnitude.

Exact block-mass3-row reference COMPLETE:multiquery000/002 mean.875 vs COBS.5,rank1.375,native.25;multikey002 exact/rank1/COBS1 vsPC2/native0. This selected diagnostic supports testing removal of approximation error, not a universal upper bound. Started queue_pc2_exact_full_dev.json:32 public16K DEV rows, native/COBS/rank1 reused and3 existing exact rows imported;29 new exact generations. Runtime of diagnostic was56.4seconds total for3 new generations; full scorer remains dense-QK work, not asymptotically sparse or a new approximation claim.

Queued queue_pm_type_balanced.json behind exact32:16 DEV, uniform-position H128/rawV objective held fixed, change only prefix Q sampling to inverse within-prefix token frequency. Repeated background can dominate uniform/recent samples; each observed token type now has equal total probability, a deployable prefix-only proxy. P/C/U share samples and original future-position draws. Two CPU tests passed for repetition invariance, reproducibility, and prefix-only index bounds. File balanced_queries.py lives locally and in prepared/pm_value_devfull_20260909, original adapter/ops unchanged. If it restores synthetic retrieval but harms QA, report the distribution tradeoff and do not call it a general proxy repair. No TEST used; budget/confirmation items deferred pending a useful construction or explicit negative confirmation choice.

### Heartbeat decision 2026-09-09T19:49:18.046155+00:00

Exact mass32 DEV COMPLETE: exact/COBS/native multikey.875/.75/.375;multiquery.90625/.625/.5;single1/1/1;VT.4/.3/.4. Strong point-estimate recovery over approximate routing at matched selected-block budget; still DEV and dense-QK scoring, not PC2 approximation success. Median exact total17.76s vsCOBS39.21s includes reused controls across execution timings, not an isolated official-kernel benchmark. Need full-attention comparison before practical quality-cost claims.

Type-balanced16 DEV COMPLETE: P/C/U single1/1/1 (old rawnorm.5),order.25 each,Hotpot.375 each,2Wiki.6923/.5625/.3125. Sampling correction restores single retrieval and is shared by all phase arms; no position-exclusive claim. Started queue_balanced_full_dense.json:64 DEV type-balanced rawnorm H128, imports48 prior P/C/U rows and reuses all fixed controls, then NOSA dense full causal KV reference on32 DEV. Dense metadata now topk=null,reader_support=all_causal_KV; it is NOT equal-reader-budget evidence. Queue supervisor11592, current PMchild11593.

Read Full multi-order raw outputs: all16 terminate withEOS and use requested comma format; only3 exact. Failures mostly concern returned value/order, so do not call this a simple formatting failure or attribute all multi-order failure to compression. Keep the strict existing metric.

### Candidate lock and confirmation 2026-09-09T20:06:03.256588+00:00

Type-balanced64 DEV COMPLETE. P/C/U:single1/1/1,order.1875/.1875/.1875;2Wiki.3383/.3065/.2440;Hotpot.2625 each. Matches Full/KeyDiff on synthetic exact, exceeds EA there, but QA remains below EA(.3721/.3577). Sampling correction is established within DEV; PM phase-specific/practical superiority remains limited. Dense NOSA32 DEV still running (11/32 when checked).

Locked PM candidate BEFORE reading TEST outcomes: inverse-token-frequency prefix Q sampling, rawVnorm objective,uniform legal future positions H128,M256,seed20260909,25percent total retention,original sink4/recent256. No model/metric changes. Queued queue_pm_pressure_test.json after current dense run: E11 e11_balanced_k125_v1 (16 DEV,12.5percent plus same-budget controls; Full reusable), then E12 e12_balanced_test_locked_v1 (all128 previously unused TEST,25percent,7 arms). This confirms the selected tradeoff; it is not a rescue or configuration search. Do not alter candidate based on TEST or call subsequent reuse of this TEST unseen confirmation. Preserve raw output and code snapshots.

Dense reference finished32/32:multikey.75,multiquery.78125,single1,VT.275;median total4.23s. Exact routing has better DEV quality on three tasks but is slower than full dense reading (17.76s median), while both keep full raw KV resident. Do not present exact as asymptotically sparse, a PC2 approximation win, or reduced resident KV memory.

Queued queue_pc2_pressure_test.json after PM confirmation: E05 e05_pc2_exact_k48_v1 (16 DEV,topk48,native/COBS/rank1/exact at same selected budget), then E06 e06_pc2_exact_test_locked_v1 (full unused public16K TEST,topk64,native/COBS/rank1/exact plus full-KV dense reference). PC2 candidate is frozen rank1 construction; exact and dense are separately labelled references, not renamed PC2. Rank1 tied or slightly trailed COBS on32 DEV; TEST is honest confirmation of that uncertain tradeoff, not a rescue. No candidate or score policy changes after TEST begins. This schedules the remaining original budget/confirmation work, with the exact diagnostic carried along to test the observed approximation gap.

### Pressure result 2026-09-09T20:24:21.530701+00:00

E11 COMPLETE16 DEV at12.5percent. P/C/U single1 each (KeyDiff.75,EA.5),order.25 each (Full/KeyDiff.25),Hotpot.25 each(EA.3977),2WikiP.0556/C.2308/U.0625(EA.4423). Tight-budget retrieval retains gains but QA degrades sharply; do not generalize synthetic retention to strong natural-QA compression or describe P as dominating collapse. E12 locked25percent TEST healthy206/896 (child12530); no selection from partial TEST, no config changes. PC2 E05/E06 queue remains waiting under shared lock.

### 用户所问两条进展的原因：数据与计算核对

本次只读分析运行中的结果和DEV前缀分布，未改变锁定TEST配置、未启动模型实验。问题是解释“类型均衡25/26”与“精确块评分90.6%/87.5%”，不是据测试继续挑版本。

**PM现在改善的是采样测度，不只是抽样方差。** `balanced_queries.type_weights`按词表token ID统计前缀非sink出现次数n_v，给每个位置i概率`1/(U*n_token_i)`，U为观察到的不同token ID数。每种token的总概率都是1/U；位置分布、H128、raw V norm、原始reader和25%预算保持不变。源码不做importance correction，所以即使M趋于无穷，它也不会回到原来的均匀位置目标：这是有意改变未来需求proxy。“类型”指token ID，不是实体类型、句子语义或问答类别。

实际合成数据重复使用同一句背景。对16条Single-KV DEV，prefix为8187–8190 tokens，每条有339句相同背景，唯一Record行15–18 tokens，非sink只有57–61种token ID。现场用冻结tokenizer重编码、核对所有prompt IDs后计算Record跨度的概率；记录跨度仅用于离线解释，没有进入部署采样函数。

以`single_kv_dev_000`为例：prefix8188，排除sink后N=8184，Record行16 tokens，U=58。均匀位置采样的Record概率为16/8184=0.1955%，每Q-head的M=256中期望0.5005个来自Record行；一个head完全没抽到Record的概率为60.59%。均衡后概率24.143%，期望61.806个，约提高123倍。所有16条DEV相应期望范围为均匀0.469–0.563、均衡58.40–71.36。

这些是**每个Q-head的抽样概率与期望**，不是整个模型看不到Record的概率，也不是实际保留KV覆盖率。所有关联GQA heads及所有层共同作用；Record之后的Q也可能携带记录信息。它们强烈支持“频繁重复背景把评分需求挤占”的解释，但尚未完成逐层Q→选择集合→答案的因果定位。原均匀估计即使采样充分，仍只给Record约0.2%的频率权重，单纯增加M不会改掉这个目标偏置。

完整64 DEV有可比的条件干预：`pm_value_h128_devfull_v1`与`pm_type_balanced_h128_devfull_v1`使用同模型、输入、backend，adapter/ops/run_followup/run/baselines五项源码hash相同；配置差异只有采样override及来源记录。P/C/U的Single-KV均由5/16到16/16，逐输入11改善、0退化。H与value norm没有同时改变，因此这里支持采样修改的边际效果。TEST中展示的是锁定组合方法相对EA等对手的表现，没有旧uniform采样TEST对照，不能把其全部对手差值单独归因于采样。

现场复核用户引用的前26条共同Single-KV TEST：P/C/U/F/KeyDiff均25/26，且失败的都是`single_kv_test_001`；EA11/26。该例期望`tbebwpqhbh`，上述五者都输出`tebebwpqhbh`并正常EOS，是共同的字符复制错误，不是候选独有的压缩失败，也不是缺EOS。检查时已增加到28条共同输入，前五者均27/28、EA12/28，仍是同一失败输入；这只是当时的中间快照。

**为什么P/C/U一起改善、自然QA却没有同步改善？** 当前Single-KV面板的唯一记录具有很强的稀有token特征，25%预算仍约有2048个槽，远多于该行的十几个tokens。一旦不同相位臂都能保住足够记录信息，位置处理就可能没有可观察的答案边际；具体哪些层保住哪些条目仍需集合证据。当前结果是输出层面的等效，不能泛化为RoPE完全无关。

自然文档不满足“出现少=重要、重复=背景”。低频token可能属于无关人物/日期/文献，重复的实体和关系反而可能是必要线索；token ID均衡也不能保证语义需求均衡。这是QA收益有限的机制解释，与完整DEV的实测方向一致：P的Hotpot由26.99到26.25、2Wiki由35.33到33.83；这两项并没有因均衡采样提高。均衡P仍高于这里的KeyDiff适配，但低于EA，是不同保留偏好的取舍。KeyDiff本地作者score按key与平均方向的相似度排除重复项，本身就有保留独特key的倾向；因此在低多样性背景里与均衡法同时成功有合理机制解释，尚未直接验证其保留集合。

未见随机key/value与插入位置上的复现是真实进展，但仍沿用同一重复背景生成机制。不能把“未见样本”扩写为已经跨文本分布、跨自然任务泛化。多顺序任务Full本身仅3/16，已检查主要是内容/顺序错误而非缺EOS或分隔符；均衡法到3/16说明这组输入上没有新增净分数损失，不能当作解决了普遍顺序推理。

**PC2方面，精确打分去掉的是整个块mass近似误差。** 保持原NOSA选块配额、mandatory blocks、CIS和reader时，每块准确目标是

\[
L_b(q)=\log\sum_{j\in b}\exp(q^Tk_j/\sqrt d+c_j)
=\log Z_{c,b}+a^T\mu_b+\log\mathbb E_w e^{a^T(k-\mu_b)},\quad a=q/\sqrt d.
\]

PC2用二阶项近似最后的log-MGF，并仅保留RoPE pair内协方差；rank1补一个跨pair方向。本地COBS对照保留两个低秩方向。精确参考同时消除了跨方向误差和高阶/尾部误差，所以生成改善支持“当前近似损失会影响任务”，尚不能指认哪一项为主因，更不能单独支持RoPE-pair特殊性。

一个64-token块的合法标量例说明为什么“继续补协方差”不必解决问题：A块scaled logits为[10,0,...,0]，B块全为1，CIS恒定。准确logmass为A=10.002856、B=5.158883；完整均值+方差二阶式却为A=5.084176、B=5.158883，排名反转。A中一个稀有强匹配在指数读取中占优，却被低阶统计淹没。该例只有一个变化方向，完整协方差已知，增加跨pair rank无济于事。CPU数值保存在`results/two_core_analysis_20260909/pc2_tail_counterexample.json`；它是机制反例，未声称真实NOSA失败块恰有这些logits。

实际32 DEV是四任务各8条。Multiquery有每题4个答案项，exact29/32项=90.625%，COBS20/32项=62.5%；逐输入5条改善、0退化、3条相同。Multikey是exact7/8、COBS6/8，仅1条改善、0退化。前者不是32个独立输入，后者也不是大样本的稳定12.5pp。原始输出中COBS有把同一个数字填给多个询问对象的情况，exact恢复更多不同正确数字；这与选块失误解释一致，但集合/块内误差分解仍未测定。

这里“精确”指评分算子。90.6%是原定官方答案项substring recall，不是完整回答+EOS正确率；部分输出有重复、额外文本或不同顺序，不能与PM的strict exact数字混为一种成功率。

**为什么精确参考慢、甚至能比dense得分高？** `ExactBlockSelector`首先对所有raw keys做FP32 QK与block logsumexp，然后仍要top-k、gather并执行selected SDPA。Dense直接做一次全可见SDPA；精确路由没有省掉全量打分，且增加了中间张量和第二次读取，当前慢是可以解释的。两者仍保留全部raw KV，不能把选25%块解释为常驻KV减少75%。本地COBS SVD/参考实现慢也不能变成精确法具有部署加速的证据。

NOSA本来在稀疏读取结构中训练，改为全可见KV会改变其attention分母与计算分布；Full不保证每个任务准确率最高。[NOSA原文](https://arxiv.org/html/2510.13602v1)。精确选择可能滤掉干扰，故其DEV得分高于dense并不矛盾；当前尚未将训练适配、干扰抑制与数值/轨迹差异分开，不能宣称已经证明稀疏优于稠密。

**当前研究含义：**PM的强信号是重复背景下更合适的需求权重，PC2的强信号是准确块mass确实有任务价值。下一项应利用既有DEV查可部署采样是否适用于非重复内容，以及真实cutoff误差究竟来自cross-covariance还是高阶尾部；这两项都不需要从运行中的TEST反向调参。精确打分已展示一种昂贵修复，但低成本、同状态量的修复是否存在仍是待解决的研究问题。

### 用户要求扩大测试：已交付统一跨分布面板

已完成`experiments/broad_position_eval/{prepare,scoring,run,report}.py`及13项CPU检查；详细清单、命令和证据边界见[扩展面板说明](../../experiments/broad_position_eval/README.md)。服务器独立目录为`prepared/broad_panel_20260909`，`code/`内是当前reader及新入口快照，`data/`内是冻结数据，`CPU_VERIFY.json`保存实际服务器校验回执。没有覆盖主队列代码、旧数据或已完成结果。

Qwen228行：76 DEV（48个源文档/对话/材料单位）+152 TEST；NOSA181行：61 DEV（41单位）+120 TEST。新增Qasper、MultiFieldQA、NarrativeQA，以及未用Hotpot/2Wiki文档；同一记录材料比较重复背景、自然prose背景和256条竞争记录；12个未使用MRCR源对话提供两种序数问题及保留全部竞争回答的compact控制。完整自然文档不截断，超过NOSA现有16K评估设置的长对话仅在Qwen上运行。MRCR来源是公开合成对话，派生面板不冒称原版全量benchmark。

PM数据SHA256=`7891a0e5f8ef85fdc832d0a917b497edc6248445f1a24b11789064e1631c0283`；PC2=`a29938aabebdbc924966d2cca7601f0cf2581f027cff66bc060fe213680e00b9`。本地13项检查通过；远程实际tokenizer完整核验228/181行、旧新文档重叠0，并通过实际tiny-Qwen CPU生成/分支缓存隔离。这些不是新面板的GPU方法结果。

后续优先跨分布DEV：PM均衡候选全部7臂 → 同输入uniform/rawV/H128的P/C/U采样对照（固定基线复用）→ NOSA native/COBS/rank1/exact/dense。按实际结果冻结方法后，才打开该新面板TEST；旧TEST已完成结果保留。不要只继续增加同一背景模板的样本。已运行健康作业和锁定TEST不改配置，具体队列顺序由接管任务统一安排。

**锁语义：**新`run`入口默认dry run，`--execute --wait-for-lock`自行取得既有`queue.lock`；调用者不要再外包同一把flock，避免嵌套等待。具体模型路径、预算、结果目录及报告命令均已写在README，接管者无需另写一套runner。

### PM confirmation and broader-panel handover 2026-09-09T21:20:24.421537+00:00

E12 COMPLETE128 independent inputs,896 row-arm results. P/C/U single31/32 each,Full/KeyDiff31/32,EA13/32;orderP7/32,C/U8/32,Full9/32,KeyDiff6/32,EA0. Natural QA:P/C/U/EA2Wiki.4283/.4055/.4361/.4505;Hotpot.4915/.4915/.4851/.4841. Clear repeated-background retrieval retention versus EA, no stable PM-specific advantage over C/U, natural QA has tradeoffs. Record the exact-task and QA metric families separately.

Source core reports new USER authorization to broaden evaluation: newQA sources, repetition/natural-prose/competitive-record contrasts and newMRCR long/compact pairs. Core is freezing broad_position_eval and will hand over independent staging. Prioritize that unified panel. Locally authored prepare_natural_background.py overlaps; it is only prepared/synced, NOT run and NOT a result. Do not launch duplicate natural-background experiments. Current PC2 E05/E06 locks/configuration stay unchanged; coordinate broad-panel execution after receiving actual asset manifest.

### Broad DEV execution accepted 2026-09-09T21:34:58.882686+00:00

Received broad_panel_20260909 delivery and reviewed README/run plus server CPU_VERIFY. Actual panel:PM76 DEV/48 source units +152 TEST,PC2 61 DEV/41 units +120 TEST; full documents,10 PM task cells and model-specific NOSA coverage. CPU receipts are deployment/boundary evidence only. Current E06 already running with640 planned row-arm pairs; preserve it and its locked candidate.

Started queue_broad_dev.json via root/broad_dev_queue.py: broad_pm_balanced_dev_v1 (76 DEV,all7 arms), broad_pm_uniform_dev_v1 (same DEV,P/C/U only), broad_pc2_dev_v1 (61 DEV,native/COBS/rank1/exact/dense), in that order. Each broad entrypoint takes the existing lock; outer supervisor deliberately does not flock. First child waits behind healthy E06. Outer status RUNNING_OR_WAITING requires checking current run/status.json to distinguish waiting from model execution. NewTEST not scheduled. New-input controls run once in versioned broad cache. Metrics and unit-clustered analyses remain separate; short NOSA prefill controls must not dilute actual sparse comparisons. Do not run my overlapping natural-background prototype.

### E05 interpretation and pending selector idea 2026-09-09T21:55:10.252887+00:00

E05 COMPLETE16 DEV/topk48. rank1/COBS/native/exact:multikey.75/.5/.5/.75;multiquery.4375/.4375/.3125/.875;single1 all;VT.4/.2/.4/.4. Rank1 VT improves from.2 at topk64 to.4 at48 on the same4 DEV inputs. Support selection/fill policy is another plausible factor; no TEST-driven configuration changes. E06 healthy46/640; broad DEV entrypoint waiting normally.

Prospective structural refinement, NOT implemented or GPU-tested: for a fixed context, let A be mandatory anchors, M the native Q-branch quota, q=M-|A| extra query slots, and K the total budget. Original Q-first/CIS-fill support necessarily contains the top(K-q) CIS blocks (anchors forced). Keeping this guaranteed floor and selecting the q highest query-score blocks OUTSIDE it maximizes the same additive query score under that floor, avoiding query slots overlapping guaranteed CIS coverage. A4000-case seeded CPU toy check (K48/64,continuous/tied scores,stable ordering) preserved floor/anchors/budget and never reduced surrogate score. This says nothing automatic about learned reader values, full trajectories or generation. Could test as a selector-policy change on DEV after broad-panel evidence; do not let this idea preempt the authorized cross-distribution panel or change locked TEST.

### USER PRIORITY OVERRIDE 2026-09-09T23:11:33.527074+00:00

User explicitly rejected oversized640-arm confirmation before finding a competitive method and GPU underutilization. Source core is preparing a bounded discriminating broad comparison and checking execution cost. Stop automatic old queues: broad supervisor14951 and waiting child14952 terminated (no broad GPU run started); E06child14745 received SIGTERM to stop at its complete-row boundary. Preserve all outputs/cache/config; do not auto-resume E06 or the original large broad queue, even if old queue wrappers say COMPLETE. Remote RESEARCH_PRIORITY_PAUSE.json records override. Do not launch replacement GPU jobs until the core delivers the narrowed configuration; execution owner remains this task. This is a scheduling correction, not a server shutdown or cancellation of research.

### LATEST USER OVERRIDE: mechanism review 2026-09-09T23:15:57.821844+00:00

Latest core-relayed user correction supersedes both automatic completion and the briefly proposed acceleration/resume plan. E06 has exited at250/640 results (50 complete inputs, all5 arms); last row multiquery TEST012. Child14745,parent12446,broad supervisor14951/waiter14952 all exited; shared lock verified free. PARTIAL_BUDGET is the runner SIGTERM label, not a time budget or scientific failure. Preserve raw evidence/config/cache; no automatic restart of E06 or large broad matrices. SVD benchmark was NOT started and has NO receipt. PM fast path also not run.

Priority is now full review of the already observed mechanisms and failures, then a concrete discriminating experiment from the core task; no new GPU work merely because the card is idle, no auto-shutdown. Acceleration alone does not justify resuming640 items. Heartbeat prompt updated to enforce this override. This is a research-design/scheduling pause, not cancellation of the research objective.

### Third-party source-retention evidence — 2026-09-09 23:45 UTC

Source task: `01a0887e-696c-78c3-ba30-75ef76e09845` (user-authorized third party for causes and solutions). This is a completed offline analysis, no additional GPU run. Direct task-message delivery to the core currently reports no active turn id; monitoring task has received these results. GPU scheduling remains with the core for its current one-hour work.

`experiments/pm_keep/retention_evidence.py` audits the six retrieval traces already produced by `hour_pm_future_queries_v1`. All frozen prefix token IDs were retokenized and matched. All 30 trace keep hashes (six rows × P/C/U/K/F) match the previously completed `hour_pm_structure_balanced_v1` generations. Thus these are the actual sets used by those generations, not a resampled proxy. Three focused CPU tests passed. Full per-layer/head/record evidence and summary: `results/position_overnight_20260909/three_party_retention_evidence_v1/` (same remote run name).

| Traced input | PM target-value source-token retention, mean over 72 layer/head units | KeyDiff retention | PM / KeyDiff full-answer+EOS | All record source tokens / slots per head |
| --- | ---: | ---: | --- | ---: |
| 000 prose16 | 33.10% (4/72 units retain all value tokens) | 82.41% (48/72 complete) | wrong / correct | 284 / 1911 |
| 001 prose16 | 20.83% (1/72 complete) | 70.24% (36/72 complete) | wrong / correct | 286 / 1871 |
| 000 prose256 | 14.12% | 9.49% | wrong / wrong | 4565 / 1846 |
| 001 prose256 | 3.97% | 2.98% | wrong / wrong | 4556 / 1836 |
| 000 repeat16 | 94.68% | 89.35% | wrong / wrong | 284 / 1964 |
| 001 repeat16 | 90.08% | 85.91% | wrong / wrong | 286 / 1963 |

For prose16, the current rule discards much of the answer-bearing source although the literal records fit comfortably in the slot budget, and the legal KeyDiff control recovers both answers. Together with the actual-Q probe's objective-order reversal, this directs repair toward query-demand/selection coverage, not another H or token-frequency sweep. All eight prose16 generation families give KeyDiff 8/8 and P/C/U 1/8. Uniform-P gives 0/8, so reverting the sampling measure is not a repair.

The 256-record condition is different: retaining every original record token cannot fit in the same per-head slots. This is not an impossibility theorem for compressed representations or cross-layer storage. It does mean that a blanket promise to preserve all original spans at 25% budget is false for these inputs. Span absence alone also does not prove that other hidden states contain no answer information. The six traced Full answers are wrong, so similarity to their local attention outputs is not a task-quality upper bound.

The main paper claim remains distinct from either engineering fix: improved sparse positional computation must demonstrate a positional contribution under matched content retention/read budget. PM P/C/U results currently do not do so. A full-covariance PC2 repair may be a useful task result; its overhead and difference from COBS must be measured before treating it as a new efficient positional method.

### Covariance-directed exact-tail candidate — 2026-09-10 00:08 UTC

Third-party source file: `experiments/nosa_position/covariance_tail.py`; separate from the core's running `tail_pair.py`. The full-covariance DEV intervention has now improved five of 32 inputs over each of COBS/rank1 with no observed regressions. That is the empirical reason to target omitted covariance; it does not make a surrogate minimizer a task winner.

The new candidate stores exactly the same tensors as max-radius tail1. For each block it chooses the original key whose exact extraction most reduces squared Frobenius error of the remaining off-RoPE-pair covariance, compared with plain PC2. Let E=Off(Sigma), r_i=k_i−mu, a_i=w_i/(1−w_i). Extracting i leaves E_i=E−a_i Off(r_i r_i^T). The computable gain is 2a_i r_i^T E r_i−a_i²||Off(r_i r_i^T)||F². A B×B centered-key Gram computes it without SVD; blocks with no resolved positive gain retain plain PC2. Near-singleton exclusions are skipped for numerical stability. This targets second-order covariance only, not arbitrary query distributions, higher cumulants or guaranteed answers. The monitor independently checked the weighted decomposition in FP64.

Six focused local CPU tests passed, including brute-force weighted exclusions, the actual logmass Hessian at zero, no-extraction identity, common native-pair rotation, no future-key leakage and complete tiny-NOSA chunk equivalence. The source and dependencies are staged at `prepared/covariance_tail_20260910/code`; `CPU_VERIFY.json` records the server checks. No GPU job has been started by the third party and no pretrained-model benefit or speedup is claimed yet. Existing max-radius tail/PM jobs retain their owner and queue.

The concrete first comparison is the already frozen first four DEV rows in each of the four 16K tasks, with the same topk64/select_blocks16/chunk128/querychunk64. Reuse the existing native, COBS, rank1, exact and max-radius-tail outcomes; only this new arm needs generation. The choice of the first four is input-order based, not selection of the five improved examples. With the active scheduler's lock and from the staged code directory, its command is:

```bash
flock -n /root/autodl-tmp/position_overnight_20260909/queue.lock /root/miniconda3/bin/python -m experiments.nosa_position.covariance_tail --model /root/autodl-tmp/NOSA-1B --data /root/autodl-tmp/position_overnight_20260909/data/pc2/rows.jsonl --output /root/autodl-tmp/position_overnight_20260909/runs/pc2_covariance_tail_dev_v1 --selectors pc2_covariance_tail1 --split dev --lengths 16384 --tasks niah_single_1 niah_multikey_1 niah_multiquery vt --per-cell 4 --topk 64 --select-blocks 16 --chunk-size 128 --attention-query-chunk-size 64
```

Only actual same-input generation quality and total cost can decide whether this criterion is useful. The claim is not that a better Frobenius score guarantees a better answer. The candidate has not been inserted into any automatic queue.

The separate completed order-error audit also matters for the original positional claim: on the already observed E12 Full-KV order TEST, 30/32 first fields correctly retrieve latest A, only 9/32 retrieve first B, and 18/32 instead retrieve latest B. Sixteen complete outputs are exactly latestA,latestB; all 32 have EOS and two comma-separated fields. This is a native occurrence-reading error beyond token eviction. The classification leaves the original scores unchanged and is not an experiment on a new method. Data: `results/position_overnight_20260909/three_party_retention_evidence_v1/order_error_categories.json`.

### Existing finish-queue parameter repair — 2026-09-10 00:11 UTC

The cached covariance shadow stage had exited before loading the model: `--row-ids` received two comma-separated IDs, but `run.select_rows` requires the path to a JSON list. The third party checked the queue status was NEEDS_REVIEW and both recorded parent/child PIDs were gone, then backed up `hour_finish_queue.py` and its status to `hour_finish_queue.before_row_ids_fix_20260910.{py,json}`. Only the `rows=` assignment was corrected, pointing to `prepared/three_party_readonly_20260909/cached_shadow_row_ids.json`; it contains the same two original IDs. No model, input selection, stage order, deadline or scoring changed. The original failed log remains. The launcher was not started by the third party; the monitor received the completed repair and request to resume only the core's already scheduled stages after checking no concurrent restart.

Covariance-tail update before any real-model run: the implementation preserves an exact zero-cross-covariance tie. When a residual lies in only one rotary pair, extracting that key leaves the entire off-pair covariance error unchanged; among such safe zero-cost keys, the largest radius is retained. This keeps the already known one-pair rare-key repair without sacrificing the covariance criterion. The corresponding explicit rare-key test is the seventh CPU check. Server CPU_VERIFY was refreshed to bind the final source hash. No real-model claim is added.

### Third-party bounded continuation ownership — 2026-09-10 00:17 UTC

The current third-party user goal explicitly asks to continue through actual solution validation and a paper. After the core's existing one-hour finish queue has ended, the third party will own exactly the already specified 16-input covariance-tail DEV comparison, using its separate staged snapshot and run directory. This is not a restart or change of the core's deadline queue. Before starting, it will verify STOP, parent/child liveness, GPU jobs and the shared lock; no existing run will be interrupted or displaced. All existing controls are reused, no test split is opened, and no automatic further matrix is added. The monitoring task has been informed. Final staged source SHA is `9c5a74846e25f538b5c73be587e0db18461d69522bc718461b6259b198bb1663` with seven server CPU checks. The full experiment outcome remains unknown.


### Finish-queue recovery and completed weighted replay — 2026-09-10 00:18 UTC

The monitor resumed the already authorized finish queue after the third party corrected `--row-ids` to the unchanged two-ID JSON file. Cached shadow and timing both finished. Shadow outputs match the original full-covariance generated token IDs on both inputs, but 56 of 1,786,624 compared KV-head/query rows changed selected support (813 slot differences); this is not bitwise support parity. Cached metadata was 458–477 MB. Standalone cached totals were 23.60/24.61 s on those two inputs; this does not establish a runtime improvement.

A second launch error was a missing required `--baseline-cache` for the original key-novel PM stage. The monitor added the existing broad cache path and preserved remote `before_baseline_fix_20260910` script/state backups. Successful cached stages were reused. By then the original 330-second stage estimate plus 90-second margin no longer fit before 00:22:09 UTC, so key-novel generation was skipped. The scheduler was minimally corrected to continue later independent, already scheduled stages that still fit; stage order, scientific arguments, deadline and 90-second margin stayed fixed. This change has `before_deadline_skip_fix_20260910` backups. Final status is explicitly `PARTIAL_DEADLINE`, not full completion.

The original weighted replay completed all nine saved inputs in 12.85 s (`runs/hour_pm_weighted_replay_v1.json`). It runs one native prefix prefill per input and reuses saved actual Q, fixed keep sets and per-head prefix mass; no new answer generation or selection occurs. Reconstructed unweighted mass differs from the original trace by at most 7.16e-7. At the last-question Q, P-minus-C in real mass times raw V norm is negative on seven of nine inputs, despite a positive original P proxy advantage on all nine. The conditional-prefix version retains the same seven negative signs, so new-token denominator weighting alone does not explain the reversals. The six retrieval rows represent only two shared material families, not six independent units. For prose16 inputs 000/001, real weighted mass P/K is 1.97070/2.64832 and 1.94900/2.51941. This aligns the metric with the original objective and strengthens the proxy-versus-actual-demand mismatch evidence, without proving a generation repair.

Supervisor 22095 and child 22096 exited; GPU was empty and `queue.lock` free when checked after replay. The third-party task separately accepted ownership of its newly authorized, bounded covariance-tail 16-DEV generation; the monitor did not add that candidate to the original hour queue or launch it.


### Covariance-directed tail DEV result — 2026-09-10 00:26 UTC

The independently owned `pc2_covariance_tail_dev_v1` completed all 16 predetermined DEV inputs (four per task), with raw generations, contract and code snapshot synchronized locally under `results/position_overnight_20260909/pc2_covariance_tail_dev_v1/`. The actual process exited. This candidate did not recover the full-covariance reference's quality.

On exactly these inputs, official answer-item recall for covariance-tail / max-radius-tail / rank1 / COBS / full covariance / exact is: multikey 0.50/0.50/0.75/0.50/0.75/0.75; multiquery 0.4375/0.50/0.4375/0.50/0.9375/0.9375; single-key all 1.0; variable tracking 0.20/0.20/0.20/0.20/0.40/0.40. Covariance-tail has 1 win and 1 loss against radius-tail, 1 win and 2 losses against rank1, 2 wins and 2 losses against COBS, and 0 wins with 5 losses against each full-covariance/exact reference. This is DEV evidence under the pre-existing recall contract, not full-string/EOS or unseen-data success.

Covariance-tail median total/prefill times are 22.647/20.419 seconds; matched radius-tail 21.213/18.649 and exact 18.009/16.566. The full-covariance run contains observer instrumentation on the first two inputs per task, so its mixed timing aggregate must not be used as a clean speed comparison. Reducing the chosen second-order Frobenius surrogate by one-key extraction has not yielded a task improvement here. It neither invalidates the full-covariance recovery nor identifies query weighting as the sole remaining explanation; a single extracted key and the remaining approximation retain substantial limitations.

The core task has renewed the continuous research objective and is preparing query-weighted covariance review plus a minimal prefix-only reconstruction-demand PM interface. No old 640-item or large broad queue was resumed. Subsequent execution should use one complete, prepared configuration with a single active GPU owner.

Third-party final cross-check: `pc2_covariance_tail_dev_v1/paired_analysis.json` now contains the full paired row differences and source-contract checks. Candidate is not promoted or expanded. Some old imported native/COBS rows omit row-level execution chunk fields; their recorded durations and the fullcov observer timings do not support an optimized deployment-speed comparison. All ten focused tests for the third party's two new components pass; the real-model outcome remains the negative DEV comparison above.

The core resumed the continuous goal and the monitor has accepted the next single-GPU ownership for the already prepared key-novel PM comparison. The third party released the GPU after its 16 completed rows and schedules no further job. Primary-source recheck of COBS §5.4–5.6 confirms existing query-second-moment subspace projection, low-rank factors, FP4 and Gram construction. Any next query-weighted method needs an explicit distinction from those mechanisms and a matched strong comparison; this observation does not rule out further improvements. Source: https://arxiv.org/html/2607.09052v1 .


### Prepared key-novel continuation and nearest-method check — 2026-09-10T00:33:00.551389+00:00

Under the renewed continuous objective, the monitor started the previously specified but deadline-deferred key-novel P/C comparison: all 24 DEV inputs from the three matched background cells, unchanged H128/M256/raw-V/25% budget and reader. This is a new standalone run `pm_key_novel_continuation_dev_v1`, not a restart of the old hour queue. Wrapper PID22979 and actual GPU PID22980 were verified live; `queue_key_novel_continuation.json` records the exact command and single owner. Existing same-input controls are reusable. No new TEST or automatic follow-on matrix was scheduled. The question is whether key-geometry-guided sampling repairs the lexical-frequency proxy failure.

Primary-source check: [COBS §5.4–5.6](https://arxiv.org/html/2607.09052v1) already projects covariance into the leading eigenspace of the query second moment, then uses low-rank factors and FP4; it also describes the Gram trick. Query-subspace compression is applied at inference to a corresponding trained full-space checkpoint. Our full-space rank2 NOSA adaptation lacks this complete method. Query weighting alone therefore does not establish novelty or a state-of-the-art comparison; a new design needs an explicit distinction and appropriately matched controls.


### Key-novel and KVzip reconstruction outcomes — 2026-09-10T00:44:42.006823+00:00

The key-novel continuation completed all 48 row/arm pairs in 263.14 seconds and its GPU process exited. Exact-string-plus-EOS P/C scores were repeat16 4/8 and 3/8, prose16 0/8 and 0/8, prose256 0/8 and 0/8. Versus type-balanced sampling over all 24 inputs, P had 1 improvement/3 regressions and C had 1/4. Inputs, score contracts, H/M, seed, value objective and retention budgets were checked equal; only the prepared sampling construction changed. Raw outputs are local under `results/position_overnight_20260909/pm_key_novel_continuation_dev_v1/`.

The core subsequently delivered `prepared/kvzip_reconstruction_20260910/READY_COMMAND.json` and server CPU/tiny validation; the monitor executed that exact command with its single lock, source SHA02e17d9e1799de749156db2feae1f59963acd96ab7ca380d99b5878b0d4284d7. The fixed-budget KVzip score comparison completed all 24 DEV inputs in 79.78 seconds, preserving raw free-generation tokens and keep traces. Scores: repeat16 5/8, prose16 1/8, prose256 0/8. Against balanced P and C it had 1 improvement/1 regression each; against KeyDiff 2/7, against Full 0/11. Every keep hash differs from each of those controls. Median end-to-end row cost, including reconstruction, was 3.209 seconds. This is the pinned author reconstruction scorer with our fixed per-head selection/reader policy, not full original KVzip reproduction or a new positional method. It changes aggregation and scoring normalization as well as the query source; do not treat it as a pure query-source intervention.

The complete KVzip directory is synchronized locally under `results/position_overnight_20260909/pm_kvzip_reconstruction_dev_v1/`. Family/cluster labels are available in broad scoring output; other omitted row metadata should be joined from the frozen data using the contract identity. The third task is performing a bounded, read-only check of R target-key/value retention for prose16 inputs 000/001 against the actual generation keep hashes. No extra GPU was delegated for that audit.

### Third-party KVzip retention check — 2026-09-10 00:47 UTC

Completed without GPU on prose16 inputs 000/001. Model identity, tokenizer, frozen prefix IDs, and both saved R keep hashes match the actual generated runs. Full layer/head/source-token output is in `results/position_overnight_20260909/three_party_retention_evidence_v1/kvzip_summary.json` and the two `kvzip_broad_retrieval_dev_*_prose_16.json` files.

R target key/value source-token retention is 31.71%/30.56% for 000 and 28.24%/33.73% for 001; complete target value is retained in only 2/72 and 3/72 layer/KV-head units. KeyDiff values are 82.41%/70.24%, with 48/72 and36/72 complete units. Across all16 records, R retains 37.45%/36.34% of source record tokens, versus KeyDiff71.75%/70.61%; all-record value tokens are38.82%/36.81% versus75.46%/73.68%. Thus this reconstruction adaptation also leaves low source coverage. It is not a demonstrated case of retaining the complete needed source and still failing, nor does source-token absence prove absence of information in all other hidden states.

R generates `bdpcffuxnw` instead of `bdztcffuxnw` and `bddcrvbq` instead of `bddgcrvbmqg`; approximate lexical similarity is not strict success. The R intervention changes query source, max aggregation, scoring normalizers and other author semantics. Its fixed-per-head allocation is also distinct from the author's original cross-head/layer allocation, so this result is not a rejection of the full KVzip method or identification of one unique failure cause. The monitor received the complete two-input audit; no new queue or scorer change was made.


### Target-record causal intervention — 2026-09-10T00:56:05.960972+00:00

Executed the core READY_COMMAND unchanged for the two predetermined Full-correct prose256 DEV inputs 002/003. Original P was reused only after its contract/input/new keep hash matched. Four new generations (target-record oracle and same-removal/equal-insertion sham) completed in 16.046 seconds. Whole target Record spans were restored at the original per-head budget, with original sink/recent positions protected. This uses future-query target lookup and is privileged headroom-selected diagnosis, not a query-blind method or an unbiased population effect. All original outputs and intervention/keep receipts are synchronized locally under `results/position_overnight_20260909/pm_target_record_oracle_dev_v1/`.

| Input | Expected value | Original P / sham | Target-record oracle | Mean swapped slots per layer/KV-head | Budget |
|---|---|---|---|---:|---:|
| 002 | bdqynkghuvn | bdbvavgebrv / bdbvavgebrv | bdqynkghuvn (correct) | 18.0278 | 1854 |
| 003 | bdxudqajtgx | bdxwmetc / bdxwmetc | bdxxwmetc (wrong) | 14.2361 | 1851 |

All six outputs terminate with EOS; only oracle002 becomes full-string correct. Both target spans contain 19 original prefix tokens, fully restored in all 72 layer/KV-head units. Swap counts range 11–19 for002 and 0–19 for003, with identical removals/counts in each corresponding sham. In003 the oracle output exactly copies the queried key `bdxxwmetc` rather than its value; this is a verified error category, not proof of a specific internal binding mechanism. Target-record loss causally matters for002, while source-span restoration alone is insufficient for003. Other lost contextual states, distractor effects and question-state changes remain competing explanations. Do not extrapolate the observed 1/2 repair rate beyond these selected DEV cases.


### Independent follow-up: canonical KeyDiff completed — 2026-09-10T01:07:21.405765+00:00

Following the user instruction to independently track existing experiments and stop messaging other chats, no further cross-chat messages were sent. The previously delivered canonical-KeyDiff configuration completed all 24 DEV inputs. All raw outputs and keep sets have finished synchronizing locally under `results/position_overnight_20260909/pm_canonical_keydiff_dev_v1/`; `paired_analysis.json` records exact paired wins/losses.

Full / native post-RoPE KeyDiff / pre-RoPE-scored KeyDiff exact+EOS: repeat16 5/8, 3/8, 5/8; prose16 6/8, 8/8, 7/8; prose256 6/8, 0/8, 0/8. This is a condition-dependent tradeoff, not a universal advantage of removing RoPE from the score. The final reader still uses original post-RoPE KV and absolute positions. No unseen split was tested.

The latest server check found no active GPU compute process and no newer prepared READY_COMMAND than this completed run. Existing old queues were not restarted. Further execution remains limited to a concrete, prepared experiment within the user's current tracking scope.
