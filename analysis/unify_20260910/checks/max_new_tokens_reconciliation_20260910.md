# 核查记录：自然文本结果的 max_new_tokens 口径（astra10 线索结案）

日期：2026-09-10。CPU 只读核查，按 astra10 回传线索（"部分既有自然文本结果的顶层 max_new_tokens 与个别行长度可能不一致，需按实际运行来源核定，不能只读一个顶层字段"）逐文件核对。**结论：线索属实且已定界；登记为口径事实，不构成任何结果作废。**

## 1. `docs/research/ROPE_OLMO_BM_FIVE_QA_RESULT_20260908.json`（LongBench 自然文本 F1，不一致实例）

- 顶层 `contract.generation_config.max_new_tokens = 16`，且带 `_from_model_config: True`——该字段是 **HF model config 回显，从未被实际生成路径采用**。
- 实际逐行长度（778 行 × 双臂）：**min 2 / max 64**；两臂各有 **45/778 行恰停 64**（=真实生效上限，5.8% 截断面），直方图无 >64 行。
- EOS：baseline 725/778、candidate 728/778（行级 `*_eos` 真值计数；与 OLMo RULER 文件里的 BM119/MrPro197 是**不同实验、不同评分口径**，不得混引——sol20 已警示）。
- 判读：该文件所有自然 F1 分数按**有效 cap=64** 理解；引用方写 contract 参数时必须写"顶层 16 为回显、实际 64"。分数本身不需要重算（截断行两臂对称，各 45）。

## 2. `results/nongeometric_screen_20260909/qualification.json`

- 行内自带 `max_new_tokens=128`、`generated_ids` 长 11、`ended_eos` 字段在行级——**行级字段自洽**，无问题。此文件证明正确姿势：读行级字段，不读顶层。

## 3. `docs/research/ROPE_LOCAL_FAILURE_EVIDENCE_20260908.json`

- `requested_max_new_tokens = 0`（local_frequency_review 审计 profile 四处）——0 是该审计的"未指定"占位，非生成参数，非不一致。留此免误报。

## 4. 规则（并入 INTEGRATION §8-4，已结案）

任何自然文本/生成分数引用前：以**行级长度分布**核定有效 cap（找恰停于某值的行数），顶层 generation_config 只在与行级一致时才引用。本次核查后五-QA 文件可安全用于 BM/MrPro 自然 F1 对比（cap=64 对称截断）。
