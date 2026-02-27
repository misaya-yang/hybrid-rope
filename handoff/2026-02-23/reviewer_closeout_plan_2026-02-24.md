# Reviewer-Closeout Plan (2026-02-24)

## Goal
Use limited GPU-hours on one RTX PRO 6000 (96GB) to maximize NeurIPS reviewer confidence by addressing the highest-risk objections:

1. fairness/protocol confounds,
2. statistical weakness,
3. long-context robustness beyond one benchmark,
4. reproducibility.

---

## Current blocking facts (as of 2026-02-24)

1. Ongoing campaign is memory-saturated (multiple LongBench workers overlapping), causing OOM/invalid runs.
2. `needle` stage failed for some runs due `peft/accelerate` compatibility (`TypeError: unhashable type: 'set'`).
3. Historical LongBench JSONs often keep only preview examples; this is insufficient for valid paired significance.

---

## Implemented fixes (already landed)

1. `scripts/run_campaign_parallel.sh`
   - added single-instance lock (`artifacts/logs/.campaign_lock_*`)
   - added `MAX_LONGBENCH_PARALLEL` guard (default `2`) to limit heavy concurrent jobs
2. `scripts/eval_niah_recall.py`
   - normalized `_no_split_modules` before LoRA load to avoid accelerate hash/type crash
3. `scripts/eval_longbench.py`
   - now writes `per_sample_scores` for each task, enabling real paired bootstrap/permutation tests

---

## Reviewer-first experiment ladder (high ROI / low cost)

### P0. Protocol-locked main table refresh (must-have)
- Keep methods fixed: `baseline_native`, `PI`, `YaRN`, `hybrid`.
- Keep manifest fixed across methods (`--manifest_json`).
- Re-run E1 at `MAIN_CTX=32768` first, then optional 64K only for top-2 methods.
- Required outputs:
  - `table1_main.csv`
  - `registry_flat.csv`
  - run-level `summary.json` + `rope_params.json`.

Success criterion:
- no INVALID rows in the final table.

### P1. Statistical closure with paired tests (must-have)
- Use full `per_sample_scores` from new LongBench outputs.
- Run paired bootstrap (and optionally permutation) for:
  - `hybrid vs YaRN`, `hybrid vs PI`, `hybrid vs baseline`.
- Report:
  - mean delta, 95% CI, p-value, effect size.

Success criterion:
- at least one primary comparison has CI excluding 0, or all are explicitly reported as non-significant with calibrated claims.

### P2. Robustness beyond “easy retrieval” (high reviewer value)
- Keep existing Needle/Passkey, but add one stronger stress axis:
  - distractor density increase (NoLiMa-style hard negative pressure),
  - or length-depth stress slices (RULER-style regime stress).
- Run only top-2 methods (`hybrid`, strongest baseline) to save compute.

Success criterion:
- show either performance margin under hard distractors or clear failure boundary with honest claim scope.

### P3. Cost-performance Pareto (cheap, persuasive)
- For top-2 methods, report score vs cost:
  - x-axis: GPU-hours / tokens/sec proxy,
  - y-axis: LongBench avg, Needle metric.
- This directly addresses practical utility objection.

Success criterion:
- at least one Pareto-positive region for proposed method.

---

## Suggested execution order (for current server)

1. Stop duplicated old campaign processes.
2. Relaunch with guarded parallelism:

```bash
export MAIN_CTX=32768
export MAX_PARALLEL=2
export MAX_LONGBENCH_PARALLEL=2
export LAUNCH_GAP_SEC=240
bash scripts/run_campaign_parallel.sh
```

3. After E1/E2 finish, run summarization:

```bash
python scripts/summarize.py --registry artifacts/registry.jsonl --out artifacts/tables
```

4. Run paired significance on new full-score artifacts:

```bash
python scripts/import_2024/significance_test.py \
  --data_dir <new_run_result_root> \
  --n_bootstrap 10000
```

---

## Compute budget estimate (incremental)

- E1+E2 rerun with guardrails: ~8-14 GPU hours (depending on queueing and retries)
- P2 robustness add-on (top-2 only): ~2-4 GPU hours
- P3 cost-performance aggregation: <1 GPU hour (mostly post-processing)

Total incremental target: **10-18 GPU hours**

---

## Claim discipline for paper writing

1. If significance remains weak: use “numerically higher under fixed protocol; not significant at α=0.05”.
2. If hard-stress win appears: prioritize that as main practical claim.
3. Keep theory claim and empirical claim separated:
   - theory = structural/variational result,
   - experiments = consistency + robustness evidence under locked protocol.

