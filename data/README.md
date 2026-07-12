# Data Layout

`data/` 同时包含 tracked reviewer assets、历史小型结果和本机大型缓存。它们不能按同一证据等级解释。

| 路径 | 用途 | 提交策略 |
| --- | --- | --- |
| `curated/` | 匿名化、小型、可验证的 reviewer-facing artifact | tracked；用途和 SHA256 见 provenance manifest |
| `results_5090b/` | Primary I 的历史小型结果来源 | tracked provenance source |
| `evq_128tok_results/` | Primary II fixed-EVQ 恢复来源 | tracked historical source |
| `evq_phase9_L2048_50M_tau0_tau1.5/` | 小型 supporting sweep | tracked supporting source |
| `fineweb_val_cache/` | 本机数据缓存 | ignored；不属于 release |
| `video_temporal/` | 视频 supporting 数据与缓存 | 大型/生成内容 ignored；不是 primary evidence |

当前 reviewer-facing 数字只能沿：

`data/curated/*` → `docs/overview/RESULT_PROVENANCE_MANIFEST.md`

不得因为文件位于 `data/` 就默认它是 raw-backed、matched 或可用于 rebuttal。
