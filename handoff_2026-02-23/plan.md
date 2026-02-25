# NeurIPS 2026 Hybrid-RoPE — **Lean** Experiment Plan (v2, Mentor-Adjusted)
**Target:** single GPU **RTX PRO 6000 (96GB)**, 8B model  
**Paper ratio:** ~70% theory / 30% experiments → experiments must be **high ROI**, **low risk**, **protocol-clean**.  
**Primary objective:** strengthen acceptance odds by addressing the **most fatal reviewer attack**: protocol confounds, especially **base vs shape**.

---

## 0.4) Operator quick card (2026-02-25, read first)

- **Tuned schedule to use for next controlled reruns**
  - `anchor_factor=4`
  - `slope_raw=20`
  - `center_ratio=0.70`
- **Current server cross-model run status**
  - `llama baseline seed=1337`: completed (`checkpoint-600`)
  - `llama anchored_sigmoid seed=1337`: running
- **Important mismatch**
  - current server `cross_model_finetune.sh` / `train_cross_model_lora.py` path does not pass tuned params.
  - default code path still uses anchored-sigmoid legacy defaults (`center_ratio=0.47`, `slope=16.05/head_dim`, auto anchor for 16K ~= 5).
- **Action rule**
  - before launching Mistral/Qwen continuation, verify schedule source and record `inv_freq_sha256` in run summary.
  - future speed-first relaunch entrypoint: `scripts/cross_model_finetune_fast_tuned.sh`

Reference docs:
- `AI_HANDOFF.md`
- `handoff_2026-02-23/local_tuning_proof_2026-02-24.md`
- `handoff_2026-02-23/tomorrow_tuned_param_runbook_2026-02-25.md`
- `handoff_2026-02-23/fast_tuned_training_runbook_2026-02-25.md`

---

## 0.5) Implementation status (2026-02-24)

The following plan scripts are now implemented in-repo and ready for execution once model/data mounts finish:

- `scripts/run_probe.py`  
  E0 probe entrypoint. Writes:
  - `artifacts/results/vram_probe.json`
  - `artifacts/results/main_ctx.txt`
  - probe run folder under `artifacts/results/probe_<timestamp>/`
- `scripts/run_eval.py`  
  Unified E1/E2 runner. Creates:
  - `runs/<run_id>/config.yaml`
  - `runs/<run_id>/rope_params.json`
  - `runs/<run_id>/metrics.jsonl`
  - `runs/<run_id>/summary.json`
  - `runs/<run_id>/stdout.log`
  - `runs/<run_id>/git_state.txt`
  - appends one row into `artifacts/registry.jsonl`
- `scripts/run_attn_hist.py`  
  E3-lite runner (online distance histogram + power-law fit). Writes:
  - `artifacts/figures/Dhat_loglog.png`
  - `artifacts/results/prior_fit.json`
- `scripts/summarize.py`  
  Registry/table aggregator. Writes:
  - `artifacts/tables/registry_flat.csv`
  - `artifacts/tables/table1_main.csv`
  - `artifacts/tables/table2_e2.csv`
  - `artifacts/tables/overview_counts.csv`
- `scripts/run_campaign_parallel.sh`
  Staggered parallel orchestrator for E2/E1/E3 on single-GPU nodes:
  - default `MAX_PARALLEL=2`
  - `LAUNCH_GAP_SEC` prevents simultaneous peak-memory PPL stages
  - auto-cleans stale E1/E2/TEST registry rows before relaunch

Supporting modules added:
- `rope/schedules.py`
- `rope/inject.py`
- `rope/attn_hist.py`
- `eval/ppl/eval_ppl.py`

`scripts/eval_longbench.py` is upgraded with paired-manifest support:
- `--manifest_json <path>`
- fixed per-task indices are reused across methods (or generated once then saved), satisfying paired-eval protocol lock.

Practical method mapping used by `run_eval.py`:
- `baseline_native` -> `results/.../baseline/final_lora`
- `PI`/`pi` -> `results/.../pi/final_lora`
- `YaRN`/`yarn` -> `results/.../yarn/final_lora`
- `hybrid` -> `results/.../anchored_sigmoid/final_lora`
- can override with `--adapter_override` and `--custom_inv_freq_path`.

Preflight note on the new server (`connect.bjb1.seetacloud.com:52592`):
- During code sync, base model mount was still incomplete (`Meta-Llama-3-8B-Instruct` lacked `config/tokenizer` files).
- `sentencepiece` and `tiktoken` are installed in `base` env to avoid tokenizer bootstrap failures once model files finish syncing.
- MIG capability is present on this GPU, but enabling MIG mode is blocked by host permissions (`Insufficient Permissions` from `nvidia-smi -mig 1` inside the current container).
- Current execution strategy therefore uses process-level parallelism with staggered starts (`run_campaign_parallel.sh`) to avoid multi-job PPL OOM spikes while still improving utilization.

## 0.6) Tomorrow tuned-parameter execution focus (2026-02-25)

This is the priority block for the next run day: replace default anchored-sigmoid settings with theory-calibrated parameters, then rerun the locked protocol.

- tuned region (local `aidemo` proof, 16K/32K/64K joint objective):
  - `center_ratio ~= 0.70`
  - `slope_raw ~= 20`
  - `anchor_factor ~= 3..5`
- recommended run setting:
  - `anchor_factor=4`, `slope_raw=20`, `center_ratio=0.70`
- evidence file:
  - `handoff_2026-02-23/local_tuning_proof_2026-02-24.md`

Execution policy for 2026-02-25:
- keep all fairness controls unchanged (same adapters/checkpoints, tokenizer, manifest, decode settings).
- change only the custom `inv_freq` schedule for anchored-sigmoid branch.
- run `E2` first (shape contribution check), then `E1` refresh table, then paired significance.
- if significance still weak: report as directional + mechanism consistency; do not overclaim.

## 0.7) Seven-day sprint code status (2026-02-24, implemented)

The following code-level upgrades from the "v6 seven-day整改计划" are implemented and ready:

- `scripts/eval_longbench.py`
  - added `--score_scale {raw,pct}` (default `raw`)
  - each task now outputs:
    - `score_raw`, `score_pct`, `score`
    - `metric_unit`, `score_scale`
    - `per_sample_scores_raw`, `per_sample_scores_pct`, `per_sample_scores`
  - explicit task->metric mapping is enforced (no implicit fallback)

- `scripts/run_eval.py`
  - added `--longbench_score_scale {raw,pct}`
  - longbench stage summary now records:
    - `longbench_avg_raw`, `longbench_avg_pct`
    - `longbench_score_unit`
  - keeps protocol traceability via `inv_freq_sha256`

- `scripts/import_2024/significance_test.py` (rewritten)
  - added `--task_list`, `--seed_grouped`, `--hierarchical_bootstrap`
  - outputs three evidence levels:
    - `per_task`
    - `per_sample`
    - `cross_seed`
  - writes `significance_seeded.json/csv` by default

- `scripts/import_2024/diag_residual_grid.py` (new)
  - computes diagonal residual grid over `(b, L, prior_family)`
  - outputs:
    - `diag_residual_grid.csv/json`
    - `diag_residual_grid.png/pdf`
    - `recommended_domain.md`

- `scripts/run_attn_hist.py`
  - added `--save_hist` to persist reusable prior histograms into `prior_fit.json`
    (`overall_hist`, `overall_hist_rebinned`, `by_layer_hist_rebinned`)

- `scripts/import_2024/attention_prior_bridge.py` (new)
  - compares baseline vs anchored prior-fit artifacts
  - estimates theory-predicted density from fitted `alpha`
  - aligns prediction with observed `inv_freq` density (if provided)
  - outputs:
    - `prior_fit_comparison.json`
    - `prior_fit_comparison.png/pdf`

- `scripts/import_2024/longbench_scale_audit.py` (new)
  - validates `score_pct == 100 * score_raw`
  - checks ranking invariance under scale conversion
  - outputs `longbench_scale_audit.md`

### Quick execution snippets (copy-run)

```bash
# P0: LongBench scale audit
python scripts/import_2024/longbench_scale_audit.py \
  --metrics_csv archives/batch_report_2026-02-23_downstream_eval/report/method_metrics_best_available.csv \
  --out_md artifacts/results/theory_validation/longbench_scale_audit.md

# P3: diagonal residual applicability grid
python scripts/import_2024/diag_residual_grid.py \
  --b_grid 1e3,1e4,1e5,5e5,1e6 \
  --L_grid 4096,8192,16384,32768,65536 \
  --prior_family uniform,powerlaw,bimodal \
  --out_dir artifacts/reviewer_2026-02-24/diag_residual_grid

# P4-1: attention prior fit (baseline / anchored), keep hist for bridge
python scripts/run_attn_hist.py --model <BASE_MODEL> --variant base --save_hist \
  --out_json artifacts/reviewer_2026-02-24/prior_bridge/baseline_prior_fit.json \
  --out_fig artifacts/reviewer_2026-02-24/prior_bridge/baseline_Dhat.png

python scripts/run_attn_hist.py --model <BASE_MODEL> --variant custom \
  --adapter_path <ANCHORED_ADAPTER> --custom_inv_freq_path <CUSTOM_INV_FREQ_PT> --save_hist \
  --out_json artifacts/reviewer_2026-02-24/prior_bridge/anchored_prior_fit.json \
  --out_fig artifacts/reviewer_2026-02-24/prior_bridge/anchored_Dhat.png

# P4-2: empirical prior bridge
python scripts/import_2024/attention_prior_bridge.py \
  --baseline_prior_json artifacts/reviewer_2026-02-24/prior_bridge/baseline_prior_fit.json \
  --anchored_prior_json artifacts/reviewer_2026-02-24/prior_bridge/anchored_prior_fit.json \
  --anchored_inv_freq <CUSTOM_INV_FREQ_PT> \
  --out_dir artifacts/reviewer_2026-02-24/prior_bridge
```

## 0.8) H1/H4 pipeline hardening update (2026-02-25)

New assets added for the NeurIPS sprint remediation:

- LongBench parity hardening:
  - `scripts/longbench_official_config/dataset2prompt.json`
  - `scripts/longbench_official_config/dataset2maxlen.json`
  - `scripts/import_2024/longbench_pipeline_audit.py`
  - `scripts/eval_longbench.py` now supports:
    - `--task_set {lb6,lb21}`
    - `--prompt_source {official,legacy}`
    - `--chat_template {auto,on,off}`
    - `--truncate_mode {tail,middle}`
    - `--max_new_tokens_policy {official,manual}`
    - `--strict_parity_check`

- Protocol lock propagation:
  - `scripts/run_eval.py` now records parity knobs and writes `baseline_protocol_lock.json` in each run folder.

- Statistical rigor:
  - `scripts/import_2024/significance_test.py` now supports FDR output:
    - `p_raw`, `p_fdr_bh`, `p_fdr_by`, `claim_grade`
    - `--fdr_method {bh,by,both}`
    - auto `claim_policy_report.md`

- Theory strengthening:
  - `scripts/import_2024/functional_residual_real_prior.py`
  - `scripts/import_2024/theorem3_adversarial_bimodal.py`

---

## 0) Non‑negotiables (protocol lock)

### 0.1 Equality contract (comparisons must be valid)
All methods compared in the same experiment must share:
- Same **base checkpoint** (weights)
- Same **tokenizer**
- Same **evaluation manifest** (exact same examples/IDs)
- Same **decode settings**
- Same **RoPE injection path** and `inv_freq.copy()`-style replacement
- Same seeds for deterministic eval

If any deviation occurs → mark run **INVALID** and exclude from paper.

### 0.2 Single‑GPU compute policy
- Prioritize **evaluation-only** evidence over training.
- Do **not** implement a “new baseline” unless you already have a clean, verified implementation.
- If **64K** is unstable or memory-tight, **use 32K** as the paper’s main context length (acceptable to NeurIPS).

---

## 1) One decision upfront: does Hybrid‑RoPE change **base**?

You must explicitly define your method as:
- `base` (e.g., RoPE θ)
- `shape` (the warp/density schedule)

**Case A (common):** Hybrid only changes **shape**, base stays `θ = 500k` (same as Llama)  
→ **E2 simplifies** to a 2-condition “same base, different shape” proof (great outcome).

**Case B:** Hybrid changes **both base and shape**  
→ E2 uses the 4-condition matrix (A/B/C/D) to isolate shape contribution.

> Codex must extract base+shape from your implementation and write it to `rope_params.json` per run.

---

## 2) Priority ladder (mentor-approved)

| Priority | Experiment | Time | Why it matters |
|---|---|---:|---|
| P0 | **E0** Sanity + **64K VRAM probe** | 1–2h | decides 32K vs 64K for everything |
| P1 | **E2** Base vs Shape | 2–4h | **highest ROI**; defuses critical reviewer attack |
| P2 | **E1** Main comparison at chosen context (32K or 64K) + **full LongBench** | 6–12h | replaces weak benchmark evidence; strengthens “equal conditions win” |
| P3 | **E3-lite** Attention distance prior **histogram + α fit** (no schedule derivation) | 3–6h | figure-level bonus, lower risk than full pipeline |
| P4 | **E4 (optional)** Minimal ablations | 1–3h | only if time remains |

**Hard cut:** drop E5 (longer LoRA / rank sweep). Low ROI for this paper.

---

## 3) Required repo structure (minimal but strict)

```
project/
  rope/
    inject.py
    schedules.py
    attn_hist.py
  eval/
    ppl/
    longbench/
    needle/
  scripts/
    run_eval.py
    run_probe.py
    run_attn_hist.py
    summarize.py
  artifacts/
    registry.jsonl
    results/
    figures/
    tables/
  runs/<run_id>/
    config.yaml
    rope_params.json
    metrics.jsonl
    summary.json
    stdout.log
    git_state.txt
```

### 3.1 Run ID scheme
`{date}_{exp}_{model}_{ctx}_{method}_{seed}`  
Example: `2026-02-24_E2_llama8b_32k_hybrid_1337`

### 3.2 Registry line (paper-critical)
Append one JSON per run to `artifacts/registry.jsonl`:
```json
{
  "run_id": "...",
  "exp": "E2",
  "model": "<CKPT_8B>",
  "ctx": 32768,
  "method": "hybrid",
  "seed": 1337,
  "status": "valid",
  "rope_base": 500000.0,
  "rope_shape": "anchored_sigmoid",
  "inv_freq_sha256": "...",
  "notes": ""
}
```

---

## 4) E0 — Sanity + **64K VRAM feasibility probe** (P0)

### Goal
- Verify injection correctness
- Decide whether **64K** is feasible (bs=1 eval) on this exact stack
- If not feasible → set **main_ctx = 32K**

### Steps
1) **Injection unit tests**
   - baseline method: same input → identical output across two runs
   - method swap modifies only RoPE tensors (and logged `inv_freq_sha256`)
2) PPL sanity at 4K and 16K
3) **VRAM probe at 64K**
   - run a *single forward* on a fixed synthetic batch (bs=1) and log peak VRAM

### Output
- `artifacts/results/vram_probe.json` containing:
  - peak_vram_gb, tokens/sec, success/fail reason
- Decision file: `artifacts/results/main_ctx.txt` with either `32768` or `65536`

### Acceptance
- If 64K peak VRAM leaves <~10–15GB headroom or OOMs → choose **32K**.

---

## 5) E2 — Base vs Shape disentanglement (P1, must do)

### Goal
Demonstrate performance gains are not a base-parameter artifact.

### Choose baseline B (must be clean)
Pick **one** strong baseline you already trust, e.g. `YaRN` *or* `PI`.  
(Only add `NTK-aware` if implementation is already verified.)

### Case A: Hybrid changes **shape only** (base same as checkpoint)
Run **2 conditions** at `main_ctx`:
- A: `B(base=b0, shape=sB)`
- B: `H(base=b0, shape=sH)`  ← your method

Interpretation:
- If B > A → **shape-only contribution proven** (best possible outcome).

### Case B: Hybrid changes **base + shape**
Run **4 conditions** at `main_ctx`:
| Cond | base | shape | meaning |
|---|---:|---|---|
| A | bB (e.g., 500k) | sB (geometric / baseline) | baseline |
| B | bB | sH | **shape-only change** |
| C | bH | sB | **base-only change** |
| D | bH | sH | full method |

Key inference:
- If **B beats A** and **D beats C** → shape has **independent** contribution ✓

### Metrics (minimal)
- PPL @ main_ctx
- LongBench composite @ main_ctx (or a stable subset if full is too slow, but prefer full)

### Acceptance criteria (paper-ready)
- Clear ordering that supports:
  - shape-only improvement (B>A), and
  - full method improvement (D>C), if Case B

---

## 6) E1 — Main comparison (P2) at chosen context + **full LongBench**

### Goal
Produce the **clean centerpiece table**: under equal conditions, Hybrid‑RoPE outperforms strong baselines at long context.

### Context
Use `main_ctx` from E0:
- Prefer 64K if feasible; otherwise 32K.

### Methods set (do not bloat)
- `baseline_native`
- `PI`
- `YaRN`
- `Hybrid-RoPE` (yours)
- Optional: `Sigmoid` / `Anchored-Sigmoid` if they are core to your story and already clean
- Optional: `NTK-aware` only if already verified

> Avoid adding any baseline you cannot guarantee is bug-free.

### Evaluation suite (must be stronger than current paper)
1) **Full LongBench** (standard tasks + standard sample counts)  
2) PPL at main_ctx  
3) Needle/Passkey retrieval at main_ctx (fast sanity)

### Paired design
- One `eval_manifest.jsonl` with example IDs used **across all methods**
- Paired bootstrap CI on per-example differences

### Acceptance (paper headline)
- Hybrid is best or tied-best on LongBench aggregate and has better PPL / retrieval than the strongest baseline at main_ctx.
- Provide effect sizes + CI (even if p-values are borderline).

---

## 7) E3-lite — Attention distance prior **histogram + α fit** (P3, simplified)

### Goal
Get a **Figure-level** empirical support for the paper’s distance prior assumption, without risking a negative “derived schedule performs poorly” result.

### Key simplification
- **No storing attention matrices**.
- Perform **online accumulation** of distance histograms.

### Design (safe)
- Samples: **N=32** sequences (length 4K–8K)
- Layers: 3 layers (early/mid/late), e.g. `{2, 16, 30}`
- Heads: 8 heads per layer (fixed indices)
- dtype: fp16/bf16 ok; histogram accumulation in fp32

### Online histogram algorithm
For each selected attention matrix `A` (shape `[q, k]`):
- For each query position `i`, add attention mass into bin `Δ=|i-j|`:
  - `hist[Δ] += Σ_{j} A[i, j]` grouped by distance

Implementation requirement:
- Use vectorized ops to avoid O(L^2) Python loops.
- Never materialize or store all `A` for all samples.

### Fit α (power-law)
- Fit `D(Δ) ≈ c * Δ^{-α}` on log-log scale over a stable range (exclude tiny Δ where locality dominates).
- Report α with CI via bootstrap over samples.

### Outputs
- `figures/Dhat_loglog.png`
- `results/prior_fit.json` with α estimates per layer/head and averaged

### Acceptance
- α is stable (not wildly varying) and roughly near the regime your theory assumes (e.g., around 1).  
Even if not exactly 1, the *shape* and *broadband behavior* can still be discussed.

---

## 8) E4 — Optional minimal ablations (P4)

Only if you have time and E1/E2 are completed.

Recommended minimal set:
- anchoring on/off (if anchoring is a core mechanism claim)
- 2-point λ sweep (not 3-point): λ ∈ {0.3, 0.7}

Metrics:
- PPL @ main_ctx
- Needle @ main_ctx

---

## 9) Exact run list (copy/paste)

> Replace `<CKPT_8B>` and set `MAIN_CTX` from E0.

### E0
```bash
python scripts/run_probe.py --model <CKPT_8B> --ctx 65536 --bs 1 --dtype bf16
python scripts/run_eval.py  --exp E0 --model <CKPT_8B> --method baseline_native --ctx 4096  --seed 1337 --suite ppl
python scripts/run_eval.py  --exp E0 --model <CKPT_8B> --method baseline_native --ctx 16384 --seed 1337 --suite ppl
```

### E2 (Case A: shape-only)
```bash
export MAIN_CTX=$(cat artifacts/results/main_ctx.txt)
python scripts/run_eval.py --exp E2 --model <CKPT_8B> --method YaRN  --ctx $MAIN_CTX --seed 1337 --suite ppl,longbench_full
python scripts/run_eval.py --exp E2 --model <CKPT_8B> --method hybrid --ctx $MAIN_CTX --seed 1337 --suite ppl,longbench_full
```

### E2 (Case B: 4-condition)
```bash
export MAIN_CTX=$(cat artifacts/results/main_ctx.txt)
python scripts/run_eval.py --exp E2 --model <CKPT_8B> --method A_baseline        --ctx $MAIN_CTX --seed 1337 --suite ppl,longbench_full
python scripts/run_eval.py --exp E2 --model <CKPT_8B> --method B_shape_only      --ctx $MAIN_CTX --seed 1337 --suite ppl,longbench_full
python scripts/run_eval.py --exp E2 --model <CKPT_8B> --method C_base_only       --ctx $MAIN_CTX --seed 1337 --suite ppl,longbench_full
python scripts/run_eval.py --exp E2 --model <CKPT_8B> --method D_full_hybrid     --ctx $MAIN_CTX --seed 1337 --suite ppl,longbench_full
```

### E1 (main comparison)
```bash
export MAIN_CTX=$(cat artifacts/results/main_ctx.txt)
for m in baseline_native PI YaRN hybrid; do
  python scripts/run_eval.py --exp E1 --model <CKPT_8B> --method $m --ctx $MAIN_CTX --seed 1337 --suite ppl,longbench_full,needle
done
```

### E3-lite (prior histogram)
```bash
python scripts/run_attn_hist.py --exp E3 --model <CKPT_8B> --ctx 8192 --seed 1337 \
  --N 32 --layers "2,16,30" --heads "0,1,2,3,4,5,6,7" --bins 256
```

### Summarize
```bash
python scripts/summarize.py --registry artifacts/registry.jsonl --out artifacts/tables
```

---

## 10) Paper deliverables (what must exist)

### Table 1 (main)
Methods × {LongBench(full)@MAIN_CTX, PPL@MAIN_CTX, Needle@MAIN_CTX} + deltas vs best baseline.

### Table 2 (E2 disentanglement)
- Case A: {baseline vs hybrid} at MAIN_CTX
- Case B: {A,B,C,D} at MAIN_CTX

### Figure 1 (E3-lite)
- `Dhat(Δ)` log-log curve + fitted power-law slope α (with CI).

### Appendix (reproducibility)
- One-page protocol lock, run registry excerpt, inv_freq checksums, and exact manifests.

---

## 11) Time budget target (mentor constraint)
Total GPU time: **15–20 hours** typical.
- E0: 1–2h
- E2: 2–4h
- E1: 6–12h (dominated by LongBench(full))
- E3-lite: 3–6h
- E4 optional: 1–3h

---

## 12) “INVALID run” conditions (auto reject)
- Different `eval_manifest` across methods inside the same table
- RoPE injection changes anything besides intended tensors
- Missing `inv_freq_sha256` log
- Different tokenizer or different decoding parameters
- Unpinned benchmark version

---

## Appendix A — Required schedule metadata format

Every run must save `runs/<run_id>/rope_params.json` with:
```json
{
  "method": "hybrid",
  "base": 500000.0,
  "shape": "anchored_sigmoid",
  "shape_params": {"lambda": 0.5, "anchor": {"enabled": true, "k0": 32}},
  "inv_freq_sha256": "...",
  "notes": ""
}
```

---

## Appendix B — Minimal config schema (YAML)
```yaml
model:
  checkpoint: "<CKPT_8B>"
  dtype: "bf16"
  attn_impl: "flash"   # or "sdpa"
rope:
  method: "hybrid"
  base: 500000.0
  hybrid:
    lambda: 0.5
    anchor:
      enabled: true
      k0: 32
eval:
  ctx: 32768
  batch_size: 1
  seed: 1337
  suite: ["ppl", "longbench_full", "needle"]
logging:
  out_dir: "runs/${run_id}"
  registry: "artifacts/registry.jsonl"
```
