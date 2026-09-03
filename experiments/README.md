# `experiments/` — standalone supporting packages

These packages sit outside the February–March phase chain because each has its
own model, protocol, or rebuttal-specific lifecycle. Code or a prepared runner
is not evidence that a run occurred.

| Package | Role | Evidence boundary |
| --- | --- | --- |
| [`lora_evq_v2/`](lora_evq_v2/) | LLaMA-3-8B LoRA, temporal evaluation, and provenance utilities | supporting adaptation; use the July canonical owner, not the package README, for results |
| [`mla_yarn_short_s42/`](mla_yarn_short_s42/) | short-context MLA/YaRN operator package | protocol-specific supporting code; no generic YaRN identity |
| [`native_rope_evq_150m/`](native_rope_evq_150m/) | 151.9M Native/EVQ control implementation | current exact-range evidence is owned under `paper-2027/research/evidence/` |
| [`rebuttal_2026/`](rebuttal_2026/) | rebuttal-triggered SFT/distillation assets | prepared or supporting protocol unless a named result owner proves execution |

The historical main phase chain remains in
[`../scripts/core_text_phases/`](../scripts/core_text_phases/); reusable RoPE
implementation remains in [`../scripts/lib/rope/`](../scripts/lib/rope/).
Current evidence and claim scope route through [`../INDEX.md`](../INDEX.md),
and live compute authorization exists only in
[`../paper-2027/HANDOFF.md`](../paper-2027/HANDOFF.md).

Do not start a model run from this directory without an explicit current
protocol and user authorization. Missing raw data, runtime identity, or matched
controls cannot be repaired by proximity to a script.
