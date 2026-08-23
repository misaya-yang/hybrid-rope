# REAL_CONTEXT_TARGET_FREE_PREFLIGHT_20260822

## 状态

`DOWNLOAD_COMPLETE_TOKEN_MANIFEST_PENDING`

这是准备状态，不是实验结果。当前没有加载模型权重、没有执行全量
tokenization、没有启动评测、没有使用 GPU，也没有提交或推送。

本次完整本地 staging root 为 `/tmp/hybrid-rope-target-free-real-data-20260822`
（不在仓库内）。无卡主机的持久 root 以 `$TARGET_FREE_DATA_ROOT` 表示；大
文件传输因 SSH 数据通道截断而尚未被接受为远端完成状态。

## 已完成的代码准备

- `scripts/lib/rope/target_free.py` 新增通用 `ModelRoPEProfile` 和无状态
  `TargetFreeRoPE`。
  - profile 记录 `native_context_length` 及证据来源、`head_dim`、
    `rotary_dim`、pair 数量、Native phase/inv_freq SHA-256、原生 RoPE 与
    scaling 配置、movement/边界斜率哈希、gain coefficient 及来源。
  - phase 在 `p <= L_native` 使用 Native phase；超出边界使用
    `native_theta(L_native) + (1-m_k)*native_boundary_slope_k*(p-L_native)`。
  - query-only gain 为
    `[1+c*log(max(1,(p+1)/L_native))]^2`；key 不缩放，算子没有未来长度
    状态，因此已写入 KV 的历史 key 不会随未来请求改变。
  - 接口只接收 profile 和实际 `position_ids`，不接收 `L_target` 或
    request budget；模块无 `nn.Parameter`。
  - 通用实现不硬编码 4096/8192/16384、64 pairs 或某个 RoPE base。`c=0.1`
    只出现在 current OLMo test profile 的来源标记中，不是通用默认结论。
- `tests/test_target_free_rope.py` 覆盖 4K/32K Native fixture、不同 theta/
  rotary_dim/pair 数量与 pair layout、窗口内严格 phase、边界连续性、
  query-only gain、KV phase 稳定、无 target/budget 接口、零参数和 current
  OLMo movement/table 哈希复现。
- `scripts/data_prep/download_target_free_real_data.py` 下载并冻结真实
  数据、revision、许可证、文件大小和 SHA-256；只下载 OLMo tokenizer/config，
  明确拒绝 `model.safetensors`。
- `scripts/data_prep/target_free_context_builder.py` 实现 chat-template
  prompt、relative-Native 分桶、row/prompt/source/tokenizer hash、证据距离
  记录和 PG-19 nested anchor 构建；默认不 padding、不拼接无关文本、不造
  synthetic needle、不截断主结果。
- `tests/test_target_free_context_builder.py` 和
  `tests/fixtures/target_free_contexts/longbench_fixture.jsonl` 提供 CPU-only
  微型 fixture。
- 后续命令草案见
  `scripts/data_prep/TARGET_FREE_EVAL_COMMANDS_20260822.md`；命令仅生成，
  尚未执行。

## CPU 验证 receipt

```text
python3 -m pytest tests/test_target_free_rope.py \
  tests/test_target_free_context_builder.py -q
13 passed
```

本机没有 `conda` 命令，不能提供 `conda run -n aidemo` receipt；系统 CPU
Python 为 `torch 2.8.0`, `transformers 4.57.6`, `pytest 8.4.2`。远端只有
`base` Conda 环境，没有 `aidemo`。上述测试没有 CUDA 路径和模型导入。

## 真实数据冻结 receipt

逻辑原始数据根目录为 `$TARGET_FREE_DATA_ROOT`，目录结构如下；原始数据不
进入仓库：

```text
$TARGET_FREE_DATA_ROOT/
  longbench/data.zip
  longbench/official_config/{LICENSE,README.md,dataset2maxlen.json,dataset2prompt.json}
  longbench/extracted/{longbench,longbench_e}/*.jsonl
  pg19/metadata/{metadata.csv,validation_files.txt,test_files.txt}
  pg19/books/{validation,test}/*.txt
  olmo2_tokenizer_config/{config.json,generation_config.json,merges.txt,
                          special_tokens_map.json,tokenizer.json,
                          tokenizer_config.json,vocab.json}
  download_manifest.json
```

Pinned sources and counts:

| Source | revision | split/files | license | bytes / SHA-256 / rows |
| --- | --- | --- | --- | --- |
| `THUDM/LongBench` archive | `5e628be450b7e67fb7ae6e201bd6d8f7056f7672` | selected v1 + v1-E | LongBench repository MIT; constituent sources acknowledged upstream | `113932529` / `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64` |
| `deepmind/pg19` | `4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b` | validation 50, test 100 | Apache-2.0 | book payload `59022103` bytes; per-book SHA/size in `download_manifest.json` |
| `allenai/OLMo-2-0425-1B-Instruct` | `48d788eca847d4d7548f375ad03d3c9312f6139e` | tokenizer/config only | Apache-2.0 | 7 files; composite SHA `9892a04df88b699433335d42b4cb6875361181b0d6beb0c4fea241f329fa10e8` |

Selected LongBench extracted files (all `test` rows):

| family/task | bytes | SHA-256 | rows |
| --- | ---: | --- | ---: |
| LongBench/qasper | 4,829,368 | `29aa07d2a63f36f4fb8e8cd200a3428ee3126d750bde6af1c8f9bc41c2366854` | 200 |
| LongBench/narrativeqa | 22,715,627 | `0fb8d08ba5cdad4b74244224b0dc2e8b41ee6b850d954a13eb2d282621ce2f71` | 200 |
| LongBench/multifieldqa_en | 4,483,926 | `0aac182fd317dcf6d74f8e1e0f3e61029407435346c2e0b3ff9fb45ae49c5c3f` | 150 |
| LongBench/hotpotqa | 11,483,614 | `a0005ab2a1bc2ac3a70352dccbf96cccc4e0aac6bb677f6a55180fa51b92ef6f` | 200 |
| LongBench/2wikimqa | 6,052,108 | `dda279cf93a99e1e5bfa3291fb199fd55978d10a1feb31822953cf77a1742e37` | 200 |
| LongBench/gov_report | 11,620,138 | `d28112beb3a9b41d80aa390837fa1a31c9e3da84a5262009c97585cc49f597c4` | 200 |
| LongBench-E/qasper | 6,899,114 | `97d95c01221a17a2ce51f9180d65a671bf2998504a2ff0cccffe97e9b08444d8` | 224 |
| LongBench-E/multifieldqa_en | 4,484,226 | `678a51335e3c90e0dd43bf1131045e4f2859cc693e7eded89cc6a1ee8d18faff` | 150 |
| LongBench-E/hotpotqa | 12,452,228 | `26a90a291cca5b2515bf466c6c3d1f57d8a4e67b0cf5aa39e1834913a15e6309` | 300 |
| LongBench-E/2wikimqa | 11,450,883 | `525b5b182089a4012cc7429c33f4208358778615173c4a09349429fc80c89641` | 300 |
| LongBench-E/gov_report | 14,298,882 | `0a3902fcf3d49f228549f02a2ef1ae84dbe8578b8be3d4611d54459487bdef84` | 300 |
| LongBench-E/narrativeqa | — | — | pinned upstream does not publish `narrativeqa_e` |

LongBench official config receipts: `dataset2maxlen.json` is 459 bytes,
SHA-256 `72966b3c0933e214591637fb085798c5e687ebff4ddaab5d99bbc31120532022`;
`dataset2prompt.json` is 5,437 bytes, SHA-256
`56d22ad4f382169c2b8a11ff4c982a4a1bea096c8152b0f0b85b64686b157c30`.
GovReport's official reserve is 512; QA reserve is fixed at 64.

PG-19 split-list receipts: validation list 1,036 bytes,
`e66d9f76a39a7a73a62270e98395c9532d306ecd2b7a10ae4098dcdd133d179b`;
test list 1,475 bytes,
`c84c08139695f3312df83239a1a41e7b9cde1baf7c08bfcf230ae09eaaf18d8c`;
metadata.csv is 2,737,447 bytes,
`fbb2fdb48522927b2e16aa52950f2afeb83c6fa8fed45f0c3dd834e9bc9b43b9`.
The 150 individual book hashes and sizes remain in the raw
`download_manifest.json`.

OLMo metadata files:

| file | bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 625 | `0d15ebb6cb8d998513b46ef337214176a6fd59fe5f16b30387c70d5f87795a9c` |
| `generation_config.json` | 121 | `437b97826bb4430c205083a0986fbe31f4959cffbe445e47fb7c5192fdcb8d58` |
| `merges.txt` | 916,646 | `b6fe424e334903f7fb84d3a106d9730455f4744b9fe3c21ee136d97a00e72502` |
| `special_tokens_map.json` | 581 | `78afb564e81264029b25f9caf24bda2521d5bdaeff5cd3fdbc01d3da2e8ce2f2` |
| `tokenizer.json` | 7,137,177 | `73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca` |
| `tokenizer_config.json` | 4,884 | `50c412c57d832057a3d5db42064c741f751e570f7c8788f037bfb0d2dd6e5f49` |
| `vocab.json` | 1,611,056 | `9e14712c91b37c7aab74b1306baa46ac342d620637a4b44523cdc3aec7d24195` |

`model.safetensors` was not downloaded.

## Data construction contract

- Use the target model tokenizer and its chat template. Record row, rendered
  prompt, source context, and tokenizer hashes.
- Total budget is `prompt_tokens + generation_reserve`. Buckets are retention
  `<= 1x`, near `(1x,2x]`, and far `(2x,4x]` relative to the explicit
  `L_native`. Rows over `4x` are rejected and counted, never truncated.
- QA reserve is 64. GovReport uses official LongBench budget 512.
- No padding, synthetic needle, unrelated concatenation, or main-result
  truncation. Selection is deterministic by row hash.
- PG-19 uses at most one test anchor per book. An eligible book must have at
  least `4 * L_native` history; its `1x/2x/4x` views are nested right-aligned
  suffixes and score only the shared ending 512 tokens.
- Evidence distance is populated only when the released row has a token-level
  offset. Supporting-fact annotations without token offsets are recorded as
  present-but-distance-unavailable.
- The 64K/128K natural-sample sufficiency check is pending until the target
  tokenizer manifest exists. No complete 34GB HELMET package was downloaded;
  only if the tokenized LongQA counts are insufficient should a separately
  pinned minimum HELMET subset be prepared.

## 未完成项 / blockers

1. The local staging root is complete and has the receipt above. Copying the
   raw tree to the no-card persistent root was attempted but the SSH data
   channel stalled after two small files; the large archive was not accepted on
   the server as complete. Retry the same `rsync --partial`/download script
   after the host data channel recovers, then compare every manifest SHA.
2. Token manifest and relative-Native bucket counts remain pending by design;
   no full tokenization was run on CPU.
3. PG-19 nested tokenized anchors remain pending for the same reason.
4. GPU evaluation and all four arms remain unexecuted. RULER remains smoke only.

## Git / scope receipt

- 当前分支 `main_0726` 与 `origin/main_0726` 同步，HEAD 为
  `25102673077529b6b6de94b2eca04f9fcd125b1b`。
- 本轮没有 commit、push、reset、switch 或清理动作；本轮新增文件仍是
  工作树未提交改动。
- 工作期间观察到并行外部进程把先前用户改动提交到了上述 HEAD；本轮没有
  回滚或改写这些提交。
- `paper/` 在当前工作树和 `03c4909..HEAD` 范围内均无变化。
