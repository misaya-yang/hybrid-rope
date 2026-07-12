# Raw Source Index

最后更新：2026-07-12

状态：`tracked control index / payloads local-only`

本目录只为作者内部核对原文与输入来源。它不是真实 NeurIPS review 记录，不进入 paper、author response、supplement 或 reviewer archive。策略与科学判断分别以 `../REBUTTAL_MASTER_QUESTION_LEDGER_20260711.md` 和 `../THEORY_REBUTTAL_MATHEMATICAL_AUDIT_20260711.md` 为准。

## 保留的安全 local-only 输入

| 文件 | 允许用途 | 安全边界 |
| --- | --- | --- |
| `01_review_panel_a_verbatim.md` | 核对内部模拟 panel A 的原话与问题覆盖 | 不是实际 reviewer 原话；不得公开引用或据此声称 reviews 已收到 |
| `02_review_panel_b_verbatim.md` | 核对内部模拟 panel B 的原话与问题覆盖 | 不是实际 reviewer 原话；不得公开引用或据此声称 reviews 已收到 |
| `04_user_five_point_note_verbatim.md` | 核对作者提供的五点准备要求 | 只作内部需求来源；对外表述必须重新匿名化并由证据支持 |
| `05_probability_calibration_verbatim.md` | 核对作者提供的概率/决策校准材料 | 只作内部 planning 输入；不是 acceptance 概率事实或 reviewer evidence |

以上文件即使内容本身不含明显凭据，也一律按 local-only 处理：不提交、不打包、不复制到公开文档。

## 已退役的 03 reasoning attachment

`03_reasoning_attachment_verbatim.md` 已从受控 source set 移除。不得从历史目录、聊天附件、备份或其他副本恢复，也不得在索引、rebuttal、paper 或 supplement 中转述其中的身份信息或推理过程。

## 模拟审稿原文的新位置

两份 2026-07-10 模拟审稿已字节不变迁移到：

- `../simulated_reviews/2026-07-10_fable5_committee_output.md` — SHA256 `520ff82bb04c4d552f1d36a5573ef613fc17d9bb15838b780c8f16de1d751864`
- `../simulated_reviews/2026-07-10_gpt_pro_committee_full_v2.md` — SHA256 `07fac44c2080bf0a3440cf092596c0656ee1c2246246e486e5fb3e7ed4b9d942`

这些文件是内部模拟 / pressure test，不是真实 NeurIPS reviews。其原文只用于覆盖潜在攻击面；其中的科学判断已被 2026-07-11 theory、LoRA 与 master-ledger audits supersede。不得把模拟 reviewer 身份、评分、措辞或概率写成真实 review 事实。

## 使用规则

- 总入口：`../README.md`。
- 真实 review 到来后的分流：`../REVIEWER_TRIAGE_PLAYBOOK.md`；真实原话必须另行 local-only 保存并分配稳定 ID。
- 查完整 decision / evidence 状态：`../REBUTTAL_MASTER_QUESTION_LEDGER_20260711.md`。
- 本索引 `00_INDEX.md` 保持 tracked；其列出的 `*_verbatim.md` payload 不得 commit、package、export 或进入 supplement。
- 只有匿名化、raw-backed、被真实 review 触发的 reviewer-grade 事实，才可按 README 的分轨权威规则进入最终 response。
