# Server Artifacts Snapshot (Data-Only)

This directory is a **data-only mirror** from the remote training server as of **2026-02-21**.
It is intended as the canonical offline evidence bundle for paper writing and AI handoff.

## Scope

Included:
- `results/` experiment outputs (JSON/CSV/logs/figures)
- `logs/` runtime logs
- `sigmoid_rope_experiments/data` and `sigmoid_rope_experiments/results`
- `sigmoid_rope_experiments/run_*.log`

Excluded by sync policy:
- model/checkpoint weights (`*.safetensors`, `*.pt`, `*.bin`, `*.pth`)
- optimizer states and checkpoint directories
- oversized tokenizer payload duplicates

## Source

- Remote host path: `/root/autodl-tmp/dfrope/hybrid-rope`
- Sync script: `tools/sync_server_evidence_data_only.ps1`

## Usage

Treat this folder as **read-only evidence**. Use top-level docs (`README.md`, `AI_HANDOFF.md`, `docs/`) for narrative and roadmap.
