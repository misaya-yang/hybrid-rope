# OLMo S8 fixed-u倍率迁移结果

更新：2026-09-14。状态：**冻结推理完成；fixed-u在当前机制判别块上未通过。**

## 判决问题

给定同一条OLMo S4参考allocation、同一band、端点、gain、prompt和decoder，把目标倍率
从S4迁到S8时，保持band内归一化log-frequency坐标的fixed-u规则是否优于直接保持指数
的fixed-m反事实？本实验只检验倍率迁移规则，不重新选择band、0.75参考曲线或gain。

## 冻结身份

- checkpoint：`OLMo-2-0425-1B-Instruct`，冻结权重，零模型更新；
- S4父表：`olmo_transition_bm_frontmix_w0p75_s4`，band `[14,31]`；
- S8共同gain：`1.0990651274`；fixed-m与fixed-u只改变倍率迁移规则；
- fixed-m表SHA256（float32数组）：
  `8656450637c6723a8610a7dbe9ea92171825e956b02fa45991e5ce463f07dd38`；
- fixed-u表SHA256（float32数组）：
  `019f6e6f58ac5c2818801dacfbed1a5374f65039d3c5f57c12a8a60d600cc4b6`；
- 数据：Core-6，4K/16K/32K，每任务每长度6行，共108个完全配对prompt/arm；
- 推理：greedy，batch 1，prefill chunk 8192；指标为RULER official contains，先做
  task-equal长度宏平均，再做log-length梯形AUC；
- 两臂均为`COMPLETE`，各108行；prompt SHA集合、generated ids、official score、合同内
  table/gain与receipt已经逐项核验。

## 结果

| 长度 | fixed-m | fixed-u | fixed-u − fixed-m | fixed-m cap/EOS | fixed-u cap/EOS |
|---:|---:|---:|---:|---:|---:|
| 4K | 0.54676 | 0.47315 | -0.07361 | 0.5278 / 0.4722 | 0.5278 / 0.4722 |
| 16K | 0.29769 | 0.24120 | -0.05648 | 0.5000 / 0.5000 | 0.4722 / 0.5278 |
| 32K | 0.11852 | 0.12593 | +0.00741 | 0.5000 / 0.5000 | 0.4167 / 0.5833 |

- log-AUC：fixed-m `0.35085`，fixed-u `0.29931`，差`-0.05154`；配对95%区间
  `[-0.09730,-0.00934]`，重采样中差值大于零的频率为`0.0075`；
- worst-length：fixed-m `0.11852`，fixed-u `0.12593`，差`+0.00741`；95%区间
  `[-0.05741,+0.07778]`，不能确认端点改善；
- 逐任务log-AUC差：FWE `-0.06481`、MK2 `-0.16667`、Multiquery `-0.02778`、
  Single `0`、QA `0`、VT `-0.05000`；
- 逐长度差值95%区间：4K `[-0.14167,-0.01806]`、16K
  `[-0.13380,+0.01389]`、32K `[-0.05741,+0.07963]`。

## 结论边界

fixed-u在当前OLMo机制判别块上降低了全区间AUC，理论新增的scale-covariance步骤没有
转化为任务优势；因此不扩大该规则到Llama/Qwen，也不扫alpha、band、tail或gain。
32K点估计和cap率略改善，但区间跨零，不能抵消4K/16K损失。

这是每格6行的已有测试池机制块，不冒充预案中的16/24/8最终样本分配。它足以否决
当前fixed-u分支，但不否决S4直接安装的mix075、`z`设计自由度或其他已有正结果。作者随后
要求整体研究继续，Qwen2.5-3B的S2完整强基线比较作为独立问题继续运行。

## 原始证据

服务器：`ssh -p 37849 [REDACTED_EMAIL]`。

- 根目录：`/root/autodl-tmp/today_rope_plan_20260914/olmo_fixed_u_decisive/`；
- 配对报告：`reports/fixed_u_vs_fixed_m_existing6.json`，文件SHA256
  `ecfbb151761e2863180840215d4f7abe3420c95d59b005d59194720e819670ae`；
- fixed-m raw：`runs/fixed_m_existing6/generations.jsonl`，SHA256
  `618b2fbf6840173735827561132016ad746d297a5e7fc51cb756da157f1dfc58`；
- fixed-u raw：`runs/fixed_u_existing6/generations.jsonl`，SHA256
  `000ec3d05d42db57836d41c9cf533aeaa8684092c3ed69619c81bc21880e10d7`。

原始生成仍保留在服务器；本owner是report-backed登记，不声称仓库内保存了raw副本。
