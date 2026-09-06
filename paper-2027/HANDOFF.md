# ICLR 2027 active handoff

- **Updated:** 2026-09-05; offline preparation complete; no GPU job launched this turn.
- **Role:** live Git, manuscript, machine, authorization and next-action state; no scientific verdict.
- **Active manuscript:** `paper-2027/`; **immutable baseline:** `paper/`.

Read [`../AGENTS.md`](../AGENTS.md), [`../README.md`](../README.md), then search
[`../INDEX.md`](../INDEX.md). Do not inherit old queues from historical memos.

## Latest changes

This file owns no scientific verdict. The next session runs the prepared bounded comparison.

The user will reopen the machine after their usage refresh (about three hours
from the preparation request). This turn is preparation only. Do not wake,
launch training, consume a usage reset, or start a new automation from this note.

Read [protocol §10](research/attention-aware-retrofit/preflights/CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md#10-开机后的固定比较轮次--2026-09-05-准备版).
It contains exact configuration fields, commands, caps, stops and result-to-action mapping.
The [Pro audit reconciliation](research/attention-aware-retrofit/analysis/PRO_REPORT_AUDIT_RECONCILIATION_20260905.md)
explains why compact-only precedes prefix changes and why double limits its own claim.

- Copy the prepared launcher/reviewer/annotation code into a **new** work-machine code directory.
- Fill private paths; run `matched_transfer_round.py prepare` with work-machine Conda Python.
- After the user's machine/experiment restart, run the frozen N_compact, Z, Y cases.
- Reuse exact N128 release008: seed42, all-linear r16, uninterrupted32R+96T, final128 only.
- No Z26 resume, prefix change, new rank/gain/table, further seed, blind test or model download.
- Annotation can run on CPU alongside training; strict success is not semantic mechanism proof.
- Preserve raw logs/checkpoints; read each `review.json` and `execution.json`; completed processes exit without powering off the host.

## Completed evidence to preserve

Read the [execution owner](research/attention-aware-retrofit/results/SINGLE_TABLE_FFN_SERVER_EXECUTION_20260904.md).
OLMo Z/Y and both unit-amplitude variants fail joint Native gates; those candidates stop.
Qwen N128 far16K: single3→24/32, binding0→6/16, double0→3/16.
Double compact4<8 keeps the old joint protocol unresolved; v4 restricts that family.
N128 independent1756-row Native confirmation: macro retention98.06%, CI94.35–101.65%;
format/indexing77.05% regresses. The exposed confirmation cannot tune new variants.
Qwen configured Native32K means16K is within Native; original-Native8x/16x/32x remains untested.
Prefix-LM only has a two-step smoke. Actual teacher trajectories remain a later retention hypothesis.

## Machine, verification and Git

- Last verified machine: RTX4080 SUPER32760MiB; Conda Torch2.8/cu128, HF5.15.1, PEFT0.20.
  Author also reports509032GB; recheck live device and [Blackwell profile](../docs/overview/RTX5090_BLACKWELL_PROFILE.md).
- Server refresh on2026-09-05 closed before reporting state. User now plans a later boot;
  current assets/processes are unverified. Previous bounded queue completed; heartbeat was paused.
- Personal PC: documentation/planning host and light CPU checks only. Do not
  install or recreate the work-machine environment here.
- Branch `main_0726`, HEAD `7895c0f`; tracking divergence0/0 without fetching.
  Earlier work was committed outside this turn; current edits are unstaged. No Git mutation performed here.
- Active `paper-2027/main.pdf` SHA: `37aa6402a65d68b21909b0b3479c4e8edd811079e3922c2c1be915ddeab167e4`.
  Immutable `paper/main.pdf` SHA: `fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772`.
  TeX/PDF and `paper/` unchanged; no compile or new visual QA.
- Current round's checks are recorded in protocol §10; CPU tests do not establish server/GPU readiness.
