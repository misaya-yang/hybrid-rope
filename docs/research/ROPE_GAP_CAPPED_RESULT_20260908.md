# Gap-capped / 收窄MrUni：完整开发结果为负

状态COMPLETE / NO_LONG_GAIN。Qwen2.5-3B-Instruct、静态S4、同gain、原36条
输入与解码。只运行本方，MrPro基线直接复用；没有基线重跑。

| 长度 | MrPro | Gap-capped | 差值 |
| --- | ---: | ---: | ---: |
| 32K | 87.22% | 84.44% | -2.78pp |
| 128K | 78.13% | 62.15% | -15.97pp |

| 128K任务 | MrPro | Gap-capped |
| --- | ---: | ---: |
| single_2 | 100% | 100% |
| multikey_2 | 75% | 50% |
| multiquery | 93.75% | 81.25% |
| vt | 75% | 75% |
| fwe | 75% | 66.67% |
| qa_1 | 50% | 0% |

36条为0胜、7负、29平。恢复此前BM的两条负例，只是回到MrPro已有的输出/召回，
没有新赢过MrPro的行。多查询两个负例还生成了把magic number解释成心理状态
阈值的长说明；QA一例偏离动物的题意，一例用P/co-NP未知严格包含关系作答。
这里按原官方评分判定，完整字符串、token IDs和EOS另存，未更换判分标准。

## 该负结果改变什么

闭式和独立LP验证正确，候选确实满足更小原生频率位移及原最大gap限制，但
两项都不足以选出更好的部署。它还压缩了过渡区、增加了大gap的数量及粗糙度，
改变了长距联合相位关系；不能将这些几何事实任一项单独指定为失败的唯一原因。

这个结果不支持继续扫描cap、挪动边界或把表和BM插值以修当前分数。也不能据此
声称所有原生保护方法无效。用户进一步明确不要进入“改善一个代理、破坏另一项、
十分钟换一个猜想”的循环；下一项CausalGain仅有CPU代码，暂停未运行。
先建立对原生关系与长距区分具有联合判别力的分析，不能把又一个proxy最优解
自动送到GPU。

## 回执

36次生成全部完成，899.87秒，峰值分配21.89GB。逐行重算分数/EOS、prompt身份
及完成回执SHA通过；本地解包的16个实际执行源码hash与runtime一致。
原始目录`results/bm_transfer_20260908/gap_capped_run_01/`，代码快照
`results/bm_transfer_20260908/code_gap_capped_01/`；远端同名目录位于
`/root/autodl-tmp/bm_transfer_20260908/`。GPU进程已结束。

[完整比较JSON](ROPE_GAP_CAPPED_RESULT_20260908.json)、[事前协议](ROPE_GAP_CAPPED_PROTOCOL_20260908.md)。
复算命令：

```bash
python3 -m scripts.analysis.summarize_candidate_screen \
  --run results/bm_transfer_20260908/gap_capped_run_01 \
  --method GapCapped \
  --baseline results/bm_transfer_20260908/run_qwen3_01/MrPro.jsonl \
  --out docs/research/ROPE_GAP_CAPPED_RESULT_20260908.json
```
