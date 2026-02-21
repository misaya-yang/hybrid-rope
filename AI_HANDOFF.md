# AI Handoff (Single-File Context)

Last update: `2026-02-21`

## 1. Project Intent

- This is a paper-first experiment repository for Hybrid-RoPE style long-context research.
- The target workflow is:
  1. design method
  2. run controlled experiments
  3. preserve evidence (json/log/figures)
  4. convert to paper claims

## 2. Current Ground Truth

- `results/advisor_package_2026-02-15/` is the curated baseline evidence pack.
- `server_artifacts_2026-02-21/` is the latest full server data-only snapshot pulled to local.
- `sigmoid_rope_experiments/` contains phase-based mechanistic and training-time scripts.
- `sigmoid_rope_experiments/run_phase4.py` is actively used for training-time validation.

## 3. Hard Constraints

- Do not commit weights/checkpoints.
- Keep logs, csv/json, and plots for reproducibility.
- Prefer local/offline datasets and models in server paths.
- Never assume HuggingFace external internet access.

## 4. Where To Look For Evidence

- From-scratch and 8B evidence:
  - `results/advisor_package_2026-02-15/`
- Raw server mirror (latest):
  - `server_artifacts_2026-02-21/results/`
  - `server_artifacts_2026-02-21/logs/`
- Sigmoid phase outputs:
  - `sigmoid_rope_experiments/data/`
  - `sigmoid_rope_experiments/results/`

## 5. Storyline For Paper Drafting

1. Baseline scaling and long-context collapse diagnosis.
2. Frequency-shape controls and hybrid/sigmoid mitigation.
3. Fair 8B LoRA baselines (YaRN/PI/Hybrid/PI-soft) with downstream checks.
4. Sigmoid phase:
   - formula validation
   - model selection/refit
   - robustness/sensitivity
   - training-time validation (phase4)

Reference: `docs/RESEARCH_STORYLINE_2026-02-21.md`.

## 6. Operational Commands

Server evidence sync (data-only):

```powershell
powershell -ExecutionPolicy Bypass -File tools/sync_server_evidence_data_only.ps1 `
  -RemoteHost connect.bjb1.seetacloud.com `
  -RemotePort 42581 `
  -RemoteUser root `
  -RemotePassword "<PASSWORD>" `
  -RemoteRepoRoot "/root/autodl-tmp/dfrope/hybrid-rope" `
  -LocalRepoRoot "." `
  -LocalTargetRel "server_artifacts_2026-02-21"
```

## 7. If You Continue From Here

1. Check active training status first (`run_phase4.log` and process).
2. Sync new evidence snapshot before changing conclusions.
3. Update storyline docs and manifest after any major result change.
