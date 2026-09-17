# CA-NCP Native experiment

Status: implementation and CPU validation package. No CA-NCP checkpoint statistic
or task-quality result is claimed until real Native Q/K capture and five-arm
generation complete.

## Frozen question

On `OLMo-2-0425-1B-Instruct` at Native 4K, keep all model parameters frozen,
snap NCP slot 35 to the fixed `2.205L` carrier, and use one unlabeled Native Q/K
statistic per layer/KV group to install the minimal rank-2 unitary plane that maps
the selected direction to that slot.

The first task panel is exactly Full-13×10. The arms are:

| Arm | Frequency table | Coordinate map |
|---|---|---|
| N0 | Native | identity |
| C0 | frozen NCP | identity |
| P0 | carrier-NCP | identity |
| N1 | Native | frozen shared alignment |
| P1 | carrier-NCP | frozen shared alignment |

N0/P0/N1/P1 form the frequency×alignment factorial. C0 retains the existing NCP
baseline. Nothing in this package expands the first panel to ×100 or tunes the
carrier, active boundary, rank, band, gain, or direction from task outputs.

The exact N0/C0 outputs already completed on this panel are hash- and
runtime-validated, then referenced by symlink. They are not copied or rerun. The
only new formal arms are P0/N1/P1 (390 generations total).

## Entry points

- `core.py`: split-half complex math, carrier construction, signed moment,
  deterministic eigenspace rule, minimal unitary plane, grouped Torch runtime map.
- `prepare_statistics.py`: freezes 16 PG19-train + 16 ProofPile-train fit documents
  and 4+4 source-disjoint report documents, with 512 deterministic causal pairs each.
- `capture_statistics.py`: explicit-GPU Native forward capture; default `PLAN_ONLY`.
- `build_alignment.py`: public table construction or CPU solve from captured moments.
- `runtime.py`: post-Q/K-norm, pre-RoPE frozen plane buffers and hooks.
- `prepare_pilot.py`: hash-validates and references the existing source-order
  Native-4K Full-13×10; no prompt IDs are copied or retokenized.
- `reuse_baselines.py`: verifies panel, table, dtype, decoder and runtime identity,
  then references the completed N0/C0 runs without copying outputs.
- `parity.py`: Native and carrier-table identity-path token parity; default `PLAN_ONLY`.
- `run.py`: five arms, global GPU lock, resumable rows, default `PLAN_ONLY`.
- `report.py`: exact fixed-panel scores, factorial contrasts, all tasks/output health,
  and common paired bootstrap stability.
- `benchmark_runtime.py`: opt-in one-row plane/prefill/decode/cache cost receipt;
  default `PLAN_ONLY` and never contributes an accuracy score.

`recovery_v2_eval.py` now accepts the explicit opt-in pair
`--ca-ncp-alignment-npz/--ca-ncp-alignment-label`. Runs without those arguments
retain the old path.

## Corpus manifest

The unlabeled input is a JSON list, JSONL file, or `{ "documents": [...] }`.
Each row needs:

```json
{
  "source": "pg19 or proofpile",
  "source_split": "train",
  "doc_id": "stable source identifier",
  "text_path": "one path relative to the manifest"
}
```

`token_file` (a one-dimensional integer `.npy`) or inline `text` may replace
`text_path`, but exactly one content field is allowed. Selection is SHA256(doc_id)
order within source; short documents are skipped; the 4K window and 512 pairs are
then fixed by content-independent hashes. Test/validation documents are rejected
for the formal statistic.

## No-card preparation

```bash
bash experiments/ca_ncp_native_20260917/prepare_server_cpu.sh

bash experiments/ca_ncp_native_20260917/prepare_server_cpu.sh --execute \
  --unlabeled-manifest /path/to/formal_pg19_proofpile_train.jsonl
```

Without the formal train manifest, the CPU script still materializes the exact
Native/NCP/carrier tables and registers the existing Full-13×10 panel by reference, then writes
`CPU_READINESS.json` with the missing asset. It never substitutes the available
ProofPile-test or PG19-validation files.

## Future attached-GPU sequence

```bash
ROOT=/root/autodl-tmp/today_rope_plan_20260914/ca_ncp_native_20260917
MODEL=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct

python -m experiments.ca_ncp_native_20260917.capture_statistics \
  --model "$MODEL" --assets "$ROOT/assets/statistics" \
  --construction "$ROOT/construction" --out "$ROOT/statistics"

# The command above is PLAN_ONLY. Explicit future execution:
python -m experiments.ca_ncp_native_20260917.capture_statistics \
  --model "$MODEL" --assets "$ROOT/assets/statistics" \
  --construction "$ROOT/construction" --out "$ROOT/statistics" --execute

python -m experiments.ca_ncp_native_20260917.build_alignment \
  --statistics "$ROOT/statistics" --construction "$ROOT/construction" \
  --out "$ROOT/alignment"

python -m experiments.ca_ncp_native_20260917.run \
  --model "$MODEL" --root "$ROOT"

# After reviewing the printed commands, a future explicitly authorized run:
python -m experiments.ca_ncp_native_20260917.run \
  --model "$MODEL" --root "$ROOT" --execute
```

`run --execute` reuses N0/C0, runs three eight-row parity paths concurrently
(Native+identity, carrier plain, carrier+identity), then launches P0/N1/P1 with
three workers after parity passes. All new generation runs hold `/tmp/hybrid-rope-gpu0.lock`; the queue finally writes
`reports/paired_report.json`. It does not stop other processes or queue behind an
occupied GPU.

## Deliverables and evidence identity

- `construction/METHOD_RECEIPT.json`
- `statistics/STATISTICS_RECEIPT.json`
- `alignment/alignment.npz` and `ALIGNMENT_RECEIPT.json`
- `parity/PARITY_REPORT.json` (the eight canaries are selected from the reused
  panel at runtime; no second token asset is created)
- `runs/{N0,C0,P0,N1,P1}/{contract,status,generations}.json*`
- `reports/paired_report.json`

The formal fixed-panel score is the observed result. Paired bootstrap is reported
after it as stability analysis. CPU identities and synthetic tests do not become
checkpoint-quality evidence.
