# ICLR 2027 active handoff

- **Updated:** 2026-09-04
- **Role:** live Git, manuscript, machine, authorization, and next-action state
- **Active manuscript:** `paper-2027/`
- **Immutable baseline:** `paper/`

Read [`../README.md`](../README.md), [`../AGENTS.md`](../AGENTS.md), and search
[`../INDEX.md`](../INDEX.md) only for the exact claim. This file owns no scientific verdict.

## Latest changes

- Current research has two routes only: a Native-constrained zero-training
  static-table 4x-to-8x frontier, and few-step physical-2x/4x LoRA with blind
  8x/16x/32x evaluation. Exact Native equality is not required.
- Submission dates do not ban training or GPU work. Experiments and manuscript
  evidence remain separately authorized and validated.
- Operator-bound/scale-orbit quantities failed as LM selectors. Same-table Q/K
  LoRA improved PG-19 NLL but not generated capability. These are scoped
  negatives, not a static-table or adaptation impossibility result.
- Current protocol:
  [`CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904`](research/attention-aware-retrofit/preflights/CONSTRAINED_FRONTIER_AND_2X4X_LORA_PREFLIGHT_20260904.md).
  Evidence/problem handoff:
  [`SINGLE_TABLE_ROPE_OPEN_PROBLEMS_HANDOFF_20260904`](research/attention-aware-retrofit/analysis/SINGLE_TABLE_ROPE_OPEN_PROBLEMS_HANDOFF_20260904.md).

## Git and manuscript

- Branch/upstream: `main_0726` / `origin/main_0726`; base SHA
  `fb1e0206aa32831f37ca6b2b77401f957cf82cde` was synchronized `0/0` before this change.
- Publication of the current documentation/research-code change is requested;
  delivery must verify local/tracking/remote equality.
- Active PDF SHA-256: `37aa6402a65d68b21909b0b3479c4e8edd811079e3922c2c1be915ddeab167e4`;
  prior validated body/total is 9/31 pages. It was not rebuilt.
- Immutable `paper/main.pdf` SHA-256:
  `fa41499486e53c982bd2afae26fe4f532e02fe61c1b9b92e64299dff37d94772`.
  `paper/` has no diff and neither manuscript was compiled.

## Machine and authorization

- No GPU run is active; the author shut the work machine down after the prior session.
- The work machine owns canonical PyTorch, Flash-attention, packaging, and GPU validation.
  Private rows, predictions, adapters, logs, manifests, and sidecars stay there.
- The low-configuration personal PC is a documentation/planning host. Do not
  install or recreate the work-machine environment here.
- A future run needs its exact machine/budget authorization and stop/shutdown
  condition. Stopping the prior session creates no ban on a new authorized run.

## Validation

- Current local navigation: `26/26`; scoped CPU suite: `69/69`; Python/Bash syntax,
  changed-document links, diff checks, and secret scan pass.
- Local Torch-dependent RULER tests and new work-machine asset/Flash smoke are skipped.
- Prior work-machine evidence: scale-orbit preflights, unit-gain and `.074` Q/K
  runs, 240/240 generated rows per arm, finite-scale `24/24`, and a resolving
  positive control. The corresponding result owners retain exact receipts/hashes.
- The previous session did not run the new factor frontier, route-explicit LoRA,
  rank/gain sweeps, or full-13 expansion.

## What to do now

1. On an authorized work machine, run only no-GPU asset/table/data preflights first.
2. Run the descending s8-to-s4 factor/gain retention gate; open long evaluation
   only for the largest arm passing both separate Native limits.
3. Complete route-explicit scoring and QK-first/parameter-matched-QKVO controls;
   train on physical 2x/4x only, then gate 1x before blind 8x/16x/32x.
4. Stop a validated candidate on zero primary long generation or Native damage
   above `0.12`; preserve receipts and execute the declared shutdown plan.
