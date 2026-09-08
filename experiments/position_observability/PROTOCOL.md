# Core-result search: current bounded experiment

User principle: do not lead with six arms; demonstrate a consequential core effect
before broad comparisons. No acceptance probability is asserted.

First runnable check is native Qwen3.5-0.8B, pinned official revision
2fc06364715b967f1860aea9cf38778875588b17. It is an inexpensive modern recurrent/global
hybrid for task qualification, not a replacement for Qwen3.8/Kimi K3 experiments.
No pretrained frequency/attention path is changed.

Four frozen event-log families each have two permutations of the same records.
Marker-based queries check access to both competing records; first/current/history
queries check chronology. Every output must terminate. 40 short prompts total,
48-token generation cap, nonthinking greedy decoding. Score complete decoded answer
and EOS separately and jointly; canonical whole-answer match is an additional
format-tolerant endpoint, not a substring or first-token success rule.

This determines whether the model understands the task before paying for long
inputs or interventions. A failure here indicates a task/model qualification
problem, not positional-encoding failure. A success is not a paper result either;
it enables a meaningful longer/native-context counterfactual test.

Longer natural-background inputs and actual information-channel interventions
remain conditional on this qualification and the independent research decision.
No six-arm training or competing model download is automatically queued.

Task qualification and the one-time Pro follow-up are now complete. The source
reply is saved in `../native_sparse_position/DESIGN_SOURCE.md`; no further polling.
Current causal experiments and evidence are in `../native_sparse_position/CORE_DIAGNOSIS.md`.
