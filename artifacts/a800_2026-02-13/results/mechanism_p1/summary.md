Mechanism Analysis P1 Summary

## 4.1 2x2 因子实验结论
- M00(base_orig) @16K seq PPL: 190.566
- M10(base_hybridfreq) @16K seq PPL: 7612.143
- M01(lora_origfreq) @16K seq PPL: 231.308
- M11(lora_hybridfreq) @16K seq PPL: 15.400
- 结论：单独改频谱或单独上 LoRA 均未改善长上下文，二者耦合（M11）显著最优。

## 4.2 Token-wise loss 曲线分析
- M00 head/tail NLL: 2.648 / 9.508, tail/head=3.591
- M11 head/tail NLL: 2.629 / 2.721, tail/head=1.035
- 结论：M00 在后段 token 出现显著抬升，M11 基本保持平稳。

## 4.3 Attention 行为差异 & Collapse 指标
- M00 avg entropy/sink/long/midDist: 0.3550 / 0.4881 / 0.5909 / 6306.6
- M11 avg entropy/sink/long/midDist: 0.6440 / 0.1714 / 0.3695 / 3846.4
- 结论：M11 的注意力熵更高且 sink 质量更低，显示较少塌缩与更均衡的全局分配。

## 4.4 Phase Collision 指标趋势 vs 长度
- L=2048: base_orig=0.282738, hybrid=0.217758
- L=4096: base_orig=0.236111, hybrid=0.196925
- L=8192: base_orig=0.235119, hybrid=0.206349
- L=12288: base_orig=0.239087, hybrid=0.203373
- L=14336: base_orig=0.208333, hybrid=0.208829
- L=16384: base_orig=0.223710, hybrid=0.195933
- 结论：碰撞指标随长度变化可直接量化频谱近重合趋势，可作为外推风险判据候选。

## 4.5 LoRA 权重频段重分布行为
- Q_proj avg low/mid/high: 2.9865 / 4.0625 / 4.5749
- K_proj avg low/mid/high: 2.2641 / 3.3639 / 5.6829
- 结论：LoRA 更新在频段上呈非均匀分布，可与碰撞/注意力指标联合解释稳定化来源。

## 4.6 关键机制判断结论
- 综合 2x2、token-wise NLL、attention 统计与 phase collision 指标：当前证据支持“外推失稳是结构性问题”，且稳定化来自 RoPE 频谱与参数适配（LoRA）的耦合，而非单一因素。