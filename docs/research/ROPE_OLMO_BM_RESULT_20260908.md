# OLMo 零训练 BM 实验结果（2026-09-08）

BM在OLMo静态S4设置下取得已复核的六任务局部收益，且超过MrUni和官方YaRN同条件对照。静态S8仍接近长端地板；降低部署scale或gain也未恢复32K能力。进一步提出的选择性幅度分配没有超过BM，已记录为负结果。

## Material Passport

类型：代码实验与原始输出重算。模型：allenai/OLMo-2-0425-1B-Instruct，revision `48d788eca847d4d7548f375ad03d3c9312f6139e`，1,484,916,736参数；BF16、Flash SDPA、greedy、repetition_penalty=1。没有训练。公式、真实频率、幅度、token IDs、官方任务来源及代码文件身份均在准备manifest。

任务：niah_single_2、niah_multikey_2、niah_multiquery、VT、FWE、SQuAD QA。各长度内六任务等权，保持官方逐题回答预算及match-all/match-any评分；输入不截断。长度标签是生成上限，逐题实际长度可短于该值；复核集4K实际2262–4065、16K实际14839–16354。

## 主结果

| 面板与方法 | 4K均分 | 长端均分 | 长端上限 |
| --- | ---: | ---: | ---: |
| 开发 MrPro | 37.22% | 14.93% | 16K |
| 开发 MrProBM | 79.44% | 49.03% | 16K |
| 独立seed复核 MrPro | 37.85% | 2.78% | 16K |
| 独立seed复核 MrProBM | 81.81% | 51.32% | 16K |
| 同复核输入对照 MrUni | 76.88% | 32.12% | 16K |
| 同复核输入对照 OfficialYaRN | 54.38% | 6.94% | 16K |
| S8迁移 MrPro | 4.17% | 0.69% | 32K |
| S8迁移 MrProBM | 23.89% | 6.94% | 32K |

开发seed20260909：每任务4K两条、16K四条，共36条。复核seed20260910：4K四条、16K八条，共72条，QA使用pre_samples=64；两组prompt SHA无重叠。复核在BM公式冻结且开发比较后进行，未调公式。QA跨长度仍共享题号；不能将答案数量、频率槽或同题两个长度当成更多独立样本。

开发BM相对MrPro为24胜1负11平；复核44胜0负28平。复核相对MrUni为22胜6负44平，相对官方YaRN为40胜2负30平。分项和逐条负例见[重算JSON](ROPE_OLMO_BM_RESULT_20260908.json)。这些计数是prompt级得失，不是显著性检验或全任务无损声明。

MrUni和BM的中段累计exponent总和均约8.5（MrPro约5.6667），幅度相同；BM高于MrUni说明该面板上的差异不能仅归约成指数总和。它没有隔离所有频率差分，更不证明最小粗糙度是模型损失的因果最优目标。

## 后续方法与边界

- 选择性gain：4K80.69%、16K44.79%；统一匹配gain：79.44%/44.24%；均低于原BM开发16K49.03%，不晋级、不再调系数。详见[定义和负结果](ROPE_BM_SELECTIVE_GAIN_20260908.md)。
- Native在相同复核24条4K上75.63%，BM81.81%；BM的QA从100%降到75%、多查询从93.75%降到87.5%，VT从10%升到70%。均分提高不等于原生逐任务保持。
- 静态S8在4K/32K同时明显退化，32K检索及VT双方地板。即便机械判据标出正差，也不解释为解决了32K能力；[尺度上限实验](ROPE_BM_SCALE_CAP_20260908.md)继续比较同输入的频率/gain取舍。

## 实现与可复现性

[可复用BM函数](../../scripts/lib/rope/boundary_matched.py)接收native FP32频率、base、reference_length和scale，返回FP32频率表及cos/sin幅度；不会训练、改位置、处理KV缓存或自行选scale。其输出与本轮S4实际数组逐位一致；S8及Qwen几何也与独立离散约束求解对表，但后者CPU对表不是Qwen能力验证。

扩展幅度运行器后，重复BM开发36条的生成token与原运行器完全一致。累计针对性测试当前29项通过（原20项、长度主终点1项、幅度3项、BM部署5项），未重复验证无关代码。

本地完整输入、原始生成和执行源码快照：`results/olmo_fast_screen_20260908/`；远端：`/root/autodl-tmp/olmo_fast_screen_20260908/`。不同执行阶段保留独立源码快照，不能用最新源码直接声称旧manifest无漂移。汇总脚本[summary](../../scripts/analysis/summarize_olmo_fast_screen.py)核对完成回执、原始文件哈希、行顺序，并按官方评分逐行重算。大型原始输入未纳入论文文件。

重算命令：

```bash
python -m scripts.analysis.summarize_olmo_fast_screen --root results/olmo_fast_screen_20260908 --out docs/research/ROPE_OLMO_BM_RESULT_20260908.json
```

## 本轮完成状态

7个GPU监督阶段全部正常完成，共780次完整生成（包含对照与重放，不是780条独立样本），监督阶段合计1475.99秒，含模型加载；输入准备和本地分析时间另计。

32K后续两臂：BM原S4频率/gain在同32K输入上为0%，仅将S8的gain降到S4为5.56%，均低于完整S8的6.94%。两项不进入进一步32K确认。当前已验证适用范围是此OLMo、静态S4、最高16K的六任务子集；本轮没有跨模型能力验证。

使用已验证表的示例（模型已经转为BF16后安装）：

```python
import torch
from scripts.lib.rope.boundary_matched import boundary_matched_inv_freq

native = 1 / (500000 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
inv_freq, gain, meta = boundary_matched_inv_freq(
    native, base=500000, reference_length=4096, scale=4)
rotary = model.model.rotary_emb
rotary.inv_freq = inv_freq.to(next(model.parameters()).device)
rotary.original_inv_freq = rotary.inv_freq.clone()
rotary.attention_scaling = gain
```

此示例用于本轮默认静态OLMo RoPE结构；不得叠到已经scaled/dynamic的checkpoint或中途更换已有KV cache的频率。
