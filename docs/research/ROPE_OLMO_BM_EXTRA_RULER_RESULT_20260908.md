# OLMo固定BM：新增七项RULER结果

COMPLETE，静态S4，16K，每任务50条，共350条/臂。此前缺少这些匹配MrPro输入的基线，补一次归档。未调BM。

| 任务 | MrPro | BM |
| --- | ---: | ---: |
| niah_single_1 | 20.00% | 90.00% |
| niah_single_3 | 0.00% | 50.00% |
| niah_multikey_1 | 12.00% | 60.00% |
| niah_multikey_3 | 0.00% | 0.00% |
| niah_multivalue | 5.00% | 60.50% |
| cwe | 0.60% | 5.20% |
| qa_2 | 12.00% | 26.00% |

七项等权41.67%对7.09%，+34.59pp。multikey_3仍双双0%，CWE仍低；不以其余收益掩盖失败。
结合此前六项结果，已有13个任务类型的覆盖，但采样数/seed不同，尚不是统一全量RULER。
700次完整生成用1384.98秒，峰值分配5.72GiB。已核对候选SHA、完整行身份、逐行评分、EOS及预算。
[逐行输出与token结果](ROPE_OLMO_BM_EXTRA_RULER_RESULT_20260908.json)。
