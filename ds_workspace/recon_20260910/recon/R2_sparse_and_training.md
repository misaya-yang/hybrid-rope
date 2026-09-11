# R2 — Sparse Attention & Training Harness Reconnaissance

Branch `09_09`, repo `/Users/yang/projects/hybrid-rope`. Read-only survey; nothing outside this file was written.
Every number/path below was either read directly or is marked `not found` / `not recorded`.
Arithmetic derived from recorded receipts is labelled `[derived]` with the inputs named.

---

## (a) Executive summary

1. The project's entire from-scratch training history is **dense attention**. Every production harness ends in `F.scaled_dot_product_attention(..., is_causal=True)`; no trainer in the repo can produce a windowed, compressed, or selected attention layer.
2. The 151.9M model exists as **architecture only**: `experiments/native_rope_evq_150m/model.py` (6.9 KB) has no optimizer, no loss, no train loop — only `nn.Module` definitions.
3. The **only** trainer that can train that 151.9M model in-repo is `experiments/rotary_budget/train_budget.py`, and it **cannot import**: line 46 pulls `rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m.run_experiment`, a package that does not exist in this checkout (verified by `ls`; `rebuttal/rebuttal_0723/experiments/` holds four empty dirs). **However — this is a recovery, not a rewrite.** That package survives intact on the archived branch `main_0726` (9 files), and its full transitive closure is 10 files: it plus `geo_rope_contract.py`. Everything else it needs — `experiments/native_rope_evq_150m/model.py` (its `GPT` import, line 25) and `scripts/lib/rope/{official_yarn,schedules}.py` — is **already present** in the current tree with the required symbols (`official_yarn_on_inv_freq`, `evq_cosh_phi`). See §1.
4. Sparse attention **does** exist in-repo, but as **inference-time references on pretrained checkpoints**, not as trainable layers: `experiments/nosa_position/runtime.py` (NSA-style compress+select+local, loads pretrained weights via `from_pretrained`), `experiments/native_sparse_position/*` (hooks Qwen2/MiniCPM via `ALL_ATTENTION_FUNCTIONS`). `RESULT_20260908.md:15` states plainly: "No training."
5. The only *trainable* sparse attention is `experiments/deepseek_mini_position/runtime.py`, which monkeypatches `DeepseekV4Attention.forward` on an externally downloaded model — a workbench, not a from-scratch harness.
6. The only from-scratch *sparse* trainer is a synthetic toy: `scripts/experiments/sparse_memory/` — vocab 64, width 192, 3 layers, synthetic event-stream `npz`. It is a mechanism assay, not an LM pretraining harness.
7. **No hybrid per-layer local/global schedule exists** anywhere in runnable code (`layer_types` / per-layer attention-type dispatch: not found). The closest artefact is a per-layer *rotary table* policy, which "neither changes the attention mask nor reinterprets existing cache contents".
8. **No 150M-class checkpoints exist on disk.** All `.pt` hits under `results/`/`experiments/` are data caches, frequency vectors (`inv_freq.npy`), or eval traces. Training weights live only on servers.
9. Measured cost of a 4-arm 151.9M comparison at 500M tokens/arm and L=256 is **3.04 GPU-hours on an RTX 5090** `[derived, §5]` — cheap. But **no 150M throughput at seq_len ≥ 4096 is recorded anywhere**, and sparse/hybrid attention is only interesting at long length, so the real cost of the intended experiment is unmeasured.
10. **VERDICT: NO — a sparse-attention 150M four-arm comparison cannot be run with what exists in-repo *today*, but the gap is smaller than it first appears.** The blocker is no longer the trainer: a runnable 151.9M trainer is 10 files away on `main_0726` (a restore, ~1-2 days including re-qualification). The real blocker is (ii): the 150M model is dense-only (`model.py:101`, `is_causal=True`) and **no sparse or hybrid layer has ever been trained from scratch at LM scale anywhere in this repo**. What exists is a set of high-quality *inference-time reference implementations* written against frozen pretrained checkpoints — they must be re-derived for training (differentiable selection, no `from_pretrained`, per-layer type dispatch that does not exist yet). That is a build task of weeks, not a configuration task.

---

## (b) Findings by question

### 1. Training harnesses (from scratch)

**Canonical production harnesses — dense, no 150M tier:**

- `scripts/core_text_phases/run_evq_sweep.py` — plain dense GPT. `TIER_CONFIGS` at line 80 defines **only** `50m` (line 81, `hidden_size: 512`), `125m` (line 96, `hidden_size: 768`), `350m` (line 111, `hidden_size: 1024`), and a 4th tier at line 128 (`hidden_size: 1024`). **There is no 150m tier** (not found). Attention uses `F.scaled_dot_product_attention(q, k, v, is_causal=True)`. RoPE is the repo's own `RotaryEmbedding` (line 301) fed by `from scripts.lib.rope.schedules import evq_cosh_inv_freq` (line 57) — not HF. Tokenizer `AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")` (line 1150), matching `vocab_size: 50304`. Checkpoints: `torch.save(model.state_dict(), run_dir/"model.pt")` + `inv_freq.npy` under `results/core_text/{tier}_sweep/<run_id>/`.
  Recorded entry point, `docs/overview/REPRODUCE.md:36-41`:
  ```
  python scripts/core_text_phases/run_evq_sweep.py \
      --tier 50m \
      --taus 0.0,0.5,1.0,1.5,2.0 \
      --seeds 42 \
      --passkey_mix_ratio 0
  ```
- `scripts/core_text_phases/run_gqa_evq_experiment.py` — MHA/GQA/MLA dispatch (`attn_type`), reuses `run_evq_sweep` internals (import at line 64). Same dense SDPA. Checkpoints `.runs/{tier}_{attn_type}_b{base}/`. ~12 launcher `.sh` files sit beside it (e.g. `run_350m_mla32_500m.sh`, `run_50m_mla_v2_tau_sweep.sh`) with verbatim `/root/autodl-tmp/...` server commands.
- ~14 phase trainers in `scripts/core_text_phases/` (`phase11_L256_extrap.py`, `phase17e/17f/17g/17h`, `phase18_*`, …) — all dense, all reuse the same `GPT`/`Attention`/`RotaryEmbedding`.

**The 150M family — architecture without a trainer:**

- `experiments/native_rope_evq_150m/` contains exactly one file: `model.py` (6870 bytes). `grep -n "AdamW|optimizer|backward|step()"` returns **no hits**. Its attention is dense: `experiments/native_rope_evq_150m/model.py:101-102`
  ```python
  attended = F.scaled_dot_product_attention(
      query, key, value, is_causal=True
  ```
  Config lives outside it, in `experiments/rotary_budget/protocol.yaml`: 12 layers, `hidden_size: 768`, 12 heads, `head_dim: 64`, `vocab_size: 50304`, `expected_parameters: 151898880`, `sequence_length: 2048`, 3 seeds, `planned_input_tokens_per_arm: 499974144`.
- `experiments/rotary_budget/train_budget.py` — the live trainer for that model. Line 46 is the blocker:
  ```python
  from rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m import run_experiment as old
  ```
  verified absent (`ls: rebuttal/rebuttal_0723/experiments/fmrope_125m_l256_500m/: No such file or directory`). Even the model object comes from the missing package: `model=old.GPT(CONFIG, ...)` (line 60); only `CONFIG` and the frequency table are local. Its own docstring: *"Budget adapter around the recovered exact-range model, loss, sampler and runtime."*
- `experiments/rotary_budget/reference/run_experiment.py` — same missing import at lines 25-33 (`prepare`, `protocol`). `REPORT.md:63-65` is explicit: *"Recovered original GPT architecture from `/root/autodl-tmp/hybrid-rope/experiments/native_rope_evq_150m/model.py`. Original trainer/protocol snapshots are in `reference/` for comparison, **not new trainers**."*
- `experiments/rotary_budget/launch_probe.py:32-33` records the intended invocation, which shows the external dependency directly:
  ```
  /root/miniconda3/bin/python train_budget.py --original-root /root/autodl-tmp/hybrid-rope \
    --data <root>/data/manifest.json --output <root>/probe_mb8_01 --arm G32 --probe --micro-batch 8
  ```

**The missing dependency is recoverable.** `git ls-tree -r --name-only main_0726 | grep fmrope_125m_l256_500m` lists the package intact on the archived pre-slimming branch: `__init__.py`, `prepare.py`, `protocol.py`, `run_experiment.py`, `SPEC.md`, `run_5090.sh`, `run_full_queue_5090.sh`. Its transitive closure is just one further file, `rebuttal/rebuttal_0723/experiments/geo_rope_contract.py`, likewise on `main_0726`. The recovered code's own imports resolve against the **current** tree — verified: `run_experiment.py:25` → `from experiments.native_rope_evq_150m.model import GPT` (present); `protocol.py:23-24` → `official_yarn_on_inv_freq` (`scripts/lib/rope/official_yarn.py:167`) and `evq_cosh_phi` (`scripts/lib/rope/schedules.py:162`), both present; `geo_rope_contract.py:17` → `scripts/lib/rope/schedules.py` (present). Signature/numerical compatibility across the `main_0726` → `09_09` slimming is **not** verified and is the thing to test first.

**Priority dirs that are NOT from-scratch trainers (explicit):**
- `experiments/nosa_position/` — inference/analysis only. Sole optimizer hit is `learned_cutoff.py`, which trains a small residual cutoff map with the LM frozen.
- `experiments/native_sparse_position/` — eval/analysis of pretrained Qwen2.5-3B / MiniCPM4.1. `RESULT_20260908.md:15`: *"No training."*
- `experiments/deepseek_mini_position/` — `runtime.py` is a monkeypatch workbench (`install(model, ...)` over `DeepseekV4Attention` modules); `test_runtime.py` is tests. No from-scratch loop.
- `scripts/train.py` says so itself, lines 5-7: the paper's from-scratch primary runs use `scripts/core_text_phases/run_evq_sweep.py`, **"not this LoRA pipeline."** `scripts/train/`, `scripts/text_eval/llama3_continued_pretrain.py`, `experiments/evq_recovery/train.py` are all LoRA/continued-pretraining.

### 2. Sparse attention

| Artefact | Path | Verdict |
|---|---|---|
| NOSA reference (compress + top-k select + local window) | `experiments/nosa_position/runtime.py` | **Runnable, inference-only.** Loads a checkpoint: `NosaReferenceForCausalLM.from_pretrained(local_dir, ...)` (line 398). |
| DeepSeek-V4 Mini workbench (HCA/CSA/NSA) | `experiments/deepseek_mini_position/runtime.py` | **Runnable, trainable** (has `index_kl` aux loss, line 64/126) but monkeypatches an external model. |
| Native-sparse oracle adapter (block-topk over cached prefix) | `experiments/native_sparse_position/{mean,oracle,envelope,mixture}_native.py` | **Runnable instrumentation**, batch-1 decode; hooks Qwen2/Qwen3.5 via `ALL_ATTENTION_FUNCTIONS`. Header of `oracle_native.py`: *"explicitly NOT an efficient method"*. |
| Vendored MiniCPM 4.1 compressed attention | `experiments/native_sparse_position/assets/minicpm41_source/modeling_minicpm.py` (`compressed_attention()` L70) | **Third-party reference asset**, not repo-authored; CUDA kernels optional under `try/except`. |
| Synthetic banded-mask model | `scripts/experiments/sparse_memory/model.py:90-98` | **Runnable, toy scale** (vocab 64, width 192, 3 layers). |
| Sliding window in a flash kernel | `scripts/experiments/olmo_fast_screen/causal_flash.py:60-64` | **Explicitly rejected**: `sliding_window` raises `ValueError`. |

The banded mask in `scripts/experiments/sparse_memory/model.py:90-98` is the only real window mask in the repo:
```python
local = (k <= q) if cfg.dense_control else (k <= q) & (k > q-cfg.window)
ends = (torch.arange(length//cfg.ratio, device=device)+1)*cfg.ratio-1
compressed = ends.view(1, -1) <= q
```
RoPE placement differs by implementation and matters for the new direction: in NOSA, RoPE is applied **inside** the attention forward *before* selection (`runtime.py:342` `q, k = apply_rope(...)`); in the native-sparse adapter, RoPE is applied by the **host model before** the interface (the adapter verifies raw-vs-RoPE capture parity).

**Not found (searched, zero hits):** per-layer hybrid local/global schedule (`layer_types`, per-layer attention-type dispatch) in runnable code; triton block-sparse kernels; `torch.tril` diagonal-offset band masks; any sparse layer inside `scripts/lib/`, `scripts/train/`, or `scripts/mac_train/`; any sparse path in the 150M model.

**Environment.** `.venv` (Python 3.9.6) with torch 2.8.0, transformers 4.57.6, numpy 2.0.2. `flash-attn`, `xformers`, `triton`, `einops` are **not installed** and appear in neither `requirements.txt` nor `requirements-lock.txt`. None of the in-repo sparse implementations need them — they are pure PyTorch (SDPA/einsum/gather) and CPU-testable. `torch.cuda.is_available()` is `False` in this local venv.

### 3. What has actually been RUN

**FINISHED (with receipts):**

| Run | Budget | GPU | Result | Receipt |
|---|---|---|---|---|
| Exact-range 151.9M, 3 seeds, FMRoPE vs anchored EVQ | 499,974,144 tok/arm, L=256 | RTX 5090 | OOD NLL Δ `-0.281/-0.176/-0.146` at 2x/4x/8x; in-domain cost `+0.026` | `paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md` (`COMPLETE_RAW_HASH_RECEIPTED`) |
| 350M MLA-32, 3 seeds | 500M tok, L=8192 | RTX PRO 6000 Blackwell 96GB | GEO @16K 138.8±5.5 → EVQ 95.6±4.1 (−31.1%) | `results/350m_mla32_evq_report.md`, `results/350m_mla32_results_final.json` |
| Phase 11 (454M, L=256) | 100M tok, 3 seeds | R6000 Blackwell 96GB | VALID | `docs/exp/2026-03/2026-03-04_phase11_L256_results.md` |
| Phase 11B (125M, L=256) | 100M tok, 3 seeds | RTX 5090 32GB | VALID | `docs/exp/2026-03/2026-03-05_phase11b_125m_results.md` |
| Phase 22 (50M MLA) / Phase 23 (60M MLA-v2) | 200M / 50M tok, L=4096 | Blackwell 96GB / RTX 5090 | τ=2.2 (−24.1% @16K) / τ=2.5 (−14.5% @16K) | `results/PHASE22_23_MLA_TAU_SWEEP_REPORT.md` |
| Six 50.1M MLA runs | 299,892,736 tok each, L=4096 | RTX 5090 | throughput receipt only; **no PPL found** | `docs/overview/RTX5090_BLACKWELL_PROFILE.md:168-170` |
| Primary I retrain (454M) | 100M tok, L=2048 | "AutoDL server" (GPU name not recorded) | `train_time_sec: 2408.7` (geo) | `results/rebuttal_primary1_20260713/raw/primary1_seed42/summary.json` |

**ABORTED / NOT LAUNCHED:** the `rotary_budget` 18-run (6 arms × 3 seeds) 151.9M plan. `REPORT.md:9-14`: *"Real-data probe `probe_mb8_02` completed ... Steady throughput ~74,650 tokens/s; ten discarded updates, total loop 38.0 seconds ... The old 18-run plan would require about 33.49 GPU-hours of pure training at this short-probe speed ... It is **not launched**."* Its probe evidence does exist: `experiments/rotary_budget/evidence/probe_mb8_02.json` (`"parameter_count": 151898880`, `"updates": 7629`, `"input_tokens": 499974144`). Note `REPORT.md:76-78`: *"A reduced budget below roughly 268M tokens/arm must be called a pilot, not a stable frontier. No result-dependent cancellation of a seed42 arm, tau change, or baseline removal is allowed."* This is a **standing pre-registration constraint** on any 150M run.

**Checkpoints on disk: none.** Every `*.pt`/`*.safetensors`/`*.ckpt` hit is a data cache, `inv_freq.npy` frequency vector, or eval trace. Weights live on servers (`/root/autodl-tmp/...`); deliberately excluded from the local snapshot.

### 4. Evaluation of trained checkpoints

- **PPL / loss, beyond-window:** `scripts/text_eval/eval_454m_multilength.py` — `PPL_LENGTHS = [2048, 4096, 8192, 16384, 32768]` (line 100), `PK_LENGTHS = [4096, 8192, 16384, 32768, 65536]` (line 103) against `train_len=2048`, i.e. **16x PPL and 32x passkey**. Lengths are CLI args `--ppl_lengths` / `--pk_lengths`. Usage (docstring lines 13-29):
  ```
  python scripts/text_eval/eval_454m_multilength.py \
      --model_dir /root/autodl-tmp/evq_phase17c_2048_continue \
      --geo_ckpt .../geo_454m_1024_to_2048_continue/model.pt \
      --evq_ckpt .../evq1.41421_454m_1024_to_2048_continue/model.pt \
      --output_dir results/454m_multilength
  ```
  Output `eval_results.json` + console table.
- **Arbitrary lengths:** `scripts/core_text_phases/eval_super_extrap.py` — `--eval_lengths 512,2048,4096,8192,16384,32768 --passkey_lengths ...` (docstring lines 10-27), per-length YaRN, up to 64x.
- **Retrieval:** `scripts/supporting_eval/eval_passkey_scratch.py` (`--work_dir --tier --tau --seed --base --num_trials`), `scripts/core_text_phases/eval_multi_needle.py`, `scripts/supporting_eval/eval_niah_recall.py` (depth-sweep) and `eval_niah_heatmap.py`.
- **Per-distance:** `scripts/core_text_phases/eval_dsr.py` — *"Measures retrieval accuracy as a function of query-needle **distance** (not sequence length)"*, `L_eval = 8 x L_train`, `Delta/L_train in {0.5,...,8.0}`. Backed by `scripts/lib/rope/attn_hist.py` (`accumulate_distance_histogram`, *"Accumulate attention mass by distance"*) and visualised by `visualize_attention_distance.py`.
- **Interface:** from-scratch evals take `.pt` checkpoints via `--model_dir`/`--work_dir`/`--geo_ckpt`/`--evq_ckpt`; output JSON under `results/`. Beyond-window support is **yes, throughout** — that part is solved.

### 5. Compute reality check

**GPU types actually used** (all recorded in-repo): RTX 5090 32GB; RTX PRO 6000 Blackwell 96GB ("R6000 Blackwell 96GB", AutoDL); RTX 4080 SUPER 32GB (`experiments/rotary_budget/REPORT.md:61` — *"Live server: RTX 4080 SUPER, 32 GiB"*); Apple M4 Max 36GB MPS; A100 (Feb 2026, inferred from `artifacts/a100_2026-02-13/` path only — the directory **does not exist** locally). AutoDL hosts are rented and the endpoint is unstable (`experiments/curvature_20260910/RUNBOOK.md:8-16`).

**Recorded throughputs:**
- `docs/overview/RTX5090_BLACKWELL_PROFILE.md:159` — 50.1M MLA @ seq 4096: `427,546 tokens/s` (K=8) / `428,000 tokens/s` (K=32) on a 5-step probe; line 168-170: *"Across the six complete 299,892,736-token runs, mean sustained throughput was 406,311 tokens/s (range 405,765–406,593), and total training-loop time was 73.81 minutes."*
- `paper-2027/research/evidence/EXACT_RANGE_151M_3SEED_RESULT_20260820.md:166` — *"The four new arms consumed `10,941.55` training seconds (`3.04` GPU-hours of summed arm time) on an RTX 5090."*
- `experiments/rotary_budget/REPORT.md:11` — 151.9M @ L=256 on RTX 4080 SUPER: *"Steady throughput ~74,650 tokens/s"* (10-update probe).
- `results/m4_max_36gb/exp4_progressive_350m/REPORT.md:21` — 350M on M4 Max MPS: `Stage1: 1.5-1.8K tok/s`.
- **Not recorded:** any 150M-class throughput at seq_len ≥ 4096. Phase 18/19/22/23 and the Primary I retrain logs print step/ETA only; the retrain log even prints a placeholder `~0.0k tok/s`.

**Checkpoint/disk conventions:** `<work_dir>/<run_id>/model.pt` (plain `state_dict`, or `{"model": ..., "metadata": ...}`), plus `inv_freq.npy`, `results.json`, aggregate `results_checkpoint.json`. Work dirs: `results/core_text/*`, `.runs/*`, or server-side `/root/autodl-tmp/*`. Checkpoints are gitignored (`**/*.pt`, `**/*.safetensors`, `**/*.bin`) and rsynced, never committed.

**Cost of a 4-arm 150M matched-compute comparison** — `[derived]`:

Inputs: 4 arms × 499,974,144 tok = 1,999,896,576 tokens (from `protocol.yaml` / `probe_mb8_02.json`); 10,941.55 s per 4 arms on RTX 5090 (exact-range receipt, L=256); 74,650 tok/s on RTX 4080 SUPER (`REPORT.md:11`).

- Per-arm, RTX 5090 @ L=256: `10,941.55 / 4` = **2,735.4 s = 0.760 h**
- Implied throughput: `499,974,144 / 2,735.4` = **182,780 tok/s**
- **4-arm total, RTX 5090 @ L=256: 1,999,896,576 / 182,780 = 10,941 s = 3.04 GPU-hours**
- 4-arm total, RTX 4080 SUPER @ L=256: `1,999,896,576 / 74,650` = **26,790 s = 7.44 GPU-hours**
- *Method cross-check:* the same arithmetic applied to the 18-run plan gives `18 × 499,974,144 / 74,650` = 120,557 s = **33.49 h**, which reproduces `REPORT.md:12`'s recorded *"about 33.49 GPU-hours"* exactly. The method is sound.

**The honest caveat:** all of the above is **L=256**. Sparse/hybrid attention is only interesting at long sequence length, and **no 150M-class throughput at 4K/8K is recorded**. The nearest long-length datum is a *50.1M* model at 4K (406K tok/s sustained). Extrapolating that to 151.9M is an unmeasured estimate, not a receipt, and is **not** offered here as a number. A ~3 GPU-hour figure is therefore the cost of a *short-context* 150M four-arm run only; the cost of the experiment this direction actually wants is unknown until a 150M @ 4K throughput probe is run. `RTX5090_BLACKWELL_PROFILE.md:175-178` states the governing rule: *"These numbers validate this workload only. For another model, preserve the scientific global batch unless the protocol explicitly changes it, then tune micro-batch/accumulation with discarded probes rather than guessing from free memory."*

### 6. Standing constraints governing such a run

**GPU authorization is not standing; it is re-granted per instance.**
- `docs/research/USER_PROMPT_TRANSCRIPT_20260909.md:5`: *"这些是历史证据，不是重新授权执行的指令。… 过去的开机、训练、代理、提交及推送授权不因归档自动恢复。"*
- `docs/research/USER_INTENT_GUIDE_20260909.md:51`: *"本次不是启动GPU、继续旧方案或修改活动论文的授权"*; `docs/research/ACTIVE_RESEARCH_GOAL.md:54`: *"The user has closed the GPU..."*
- `scripts/README.md` (Safety and maintenance): *"A phase can authorize necessary preparation, runs and ordinary fixes; a script or old plan alone never authorizes a launch."* and *"Before paid GPU work, read `../docs/overview/RTX5090_BLACKWELL_PROFILE.md` and freeze code/config/data/checkpoint/table/output identities plus stop and shutdown plans."*
- Single scheduler: `docs/research/THREE_PARTY_RESEARCH_HANDOFF_20260910.md:8` names the 接续执行与决策 task the *"唯一 GPU 调度负责人"*.

**Utilization rules — note the precise form, they are not occupancy targets.**
- `analysis/kkt_20260910/mine2/ALL_USER_OPS.txt:48646` (verbatim user): *"GPU也绝对不能空转超过2min"* — a 2-minute idle ban.
- `:48656`: *"所用实验配置，要吃满GPU计算，最大化显存，最少超过28gb"* — ≥28 GB memory.
- Counter-rule, `docs/overview/RTX5090_BLACKWELL_PROFILE.md:85-86`: *"以预计到完成的总 GPU 时间/成本选优；显存只需稳定可行，不设固定占用率或固定空余比例目标。"* and `:68`: *"低显存占用不能推出 GPU 没有吃满"*. Also `:74-76`: *"不得为了显存数字好看而中断健康的 registered run."*
- One process per card: `docs/research/ROPE_FREQUENCY_LUNA_ROI_20260907.md:104`: *"不能只为填显存而让多个模型进程争用一张卡"*. MPS: `scripts/m4_max_36gb/README.md:248`: *"同一时间只跑一个实验"*.

**Experiment count — the "3" is NOT a hard cap.**
- `docs/research/USER_PROMPT_TRANSCRIPT_20260909.md:532`: *"后续最多 3 个实验，不能隐藏多候选扫描或重置原有预算。"*
- But `:647` walks it back: *"3个实验不是硬性上限…"*, codified in `USER_INTENT_GUIDE_20260909.md:23-29`: *"不能永久保存成'超过三次必须停机'"*. Later superseded by `TWO_CORE_SOL_HANDOFF_20260909.md:9`: *"用户明确要求立刻落实12项实验；两个核心不限制实验数量。"*

**Pre-registration of success/failure criteria is a standing user requirement.**
- `analysis/kkt_20260910/mine2/ALL_USER_OPS.txt:48666` (verbatim user): *"每次做实验，自己定义好，什么算成功，什么算失败，失败如何改进和吸收经验，成功了如何优化和扩大战果"*
- Practised form: `experiments/curvature_20260910/RUNBOOK.md:77-90`: *"it is pre-registered and must not be chosen after seeing the curve: **`EPS`**… Do not change `EPS` itself to make a branch appear."*; `:158-161`: *"`rho_trust` must land in [0.5, 1.5] and the native-KL exponent in [1.7, 2.3]. On failure the driver prints the three-branch diagnosis and stops. **Do not proceed past a failed `s4`. That stopping rule is pre-registered.**"* And the diagnosis of why this matters, `docs/research/PROJECT_DIRECTION_SYNTHESIS_20260909.md:115`: *"判决标准从不预先写死 → 任何结果都杀不死任何假说。"*
- **Directly binding on a 150M run:** `experiments/rotary_budget/REPORT.md:76-78` — *"A reduced budget below roughly 268M tokens/arm must be called a pilot, not a stable frontier. No result-dependent cancellation of a seed42 arm, tau change, or baseline removal is allowed."*

**Shutdown is part of the run, not after it.**
- `experiments/curvature_20260910/RUNBOOK.md:237-242` records it as *"a **standing end-of-run duty**, quoted from the user's own standing instruction"*: *"按照项目规范写好实验报告和相应文档，提交并推送代码，**同时检查 autoDL 是否成功关机**，先服务器命令关机，不行的话，用 chrome control"*.
- Four procedural rules (RUNBOOK.md:250-279), load-bearing two: *"**Never power off a machine that is not ours**"* (check `nvidia-smi`/worker log; *"if there is work here that this package did not start, stop and ask"*) and *"A shutdown command returning success is not evidence the instance stopped billing. Verify the instance actually reaches a stopped state in the AutoDL console"*. `docs/research/PC2_FAILURE_AND_CLAIM_AUDIT_20260910.md:9`: *"尚未成功发出关机命令，不能以断连推断已关机。"*
- Data retention: `docs/overview/SERVER_STORAGE_CLEANUP_20260907.md:4-5`: *"retain one or two small base models and reconstruction-critical assets."*

**Ban on re-running failed routes.**
- `docs/research/TWO_CORE_SOL_HANDOFF_20260909.md:3`: *"方法失败时定位原因、改方法并继续，不重新跑一轮相同基线。"*
- `experiments/curvature_20260910/RUNBOOK.md:216-222`: *"Do not re-run the chain hoping for a different number: change the thing the diagnosis names, and record which branch it was."*
- `docs/overview/RTX5090_BLACKWELL_PROFILE.md:93`: *"不要仅为执行层对称性重跑已完成 arm."*
- Repo `AGENTS.md:17-20` (item 4): *"Check previous results, failures, and corrections before repeating a direction. Do not repeat failed assumptions or generalize failures beyond their evidence."*

**Not found:** any rule requiring a *named person* to approve GPU runs beyond the user; any minimum-*utilization-percentage* rule (explicitly forbidden, see above); any hard maximum on experiments per round (retracted); the anonymity/credentials quote attributed to "AGENTS.md" at `RUNBOOK.md:354-356` is **not** in the repo's `AGENTS.md` (which has only the four principles, lines 3-20) — it comes from a system-level file outside this repo.

---

## (c) GAPS / WOULD HAVE TO BE BUILT

Ordered by cost, cheapest first.

**1. Restore a runnable 150M trainer — CHEAPER THAN EXPECTED (small: ~10 files + re-qualification).**
`experiments/rotary_budget/train_budget.py:46` and `reference/run_experiment.py:25-33` import `rebuttal.rebuttal_0723.experiments.fmrope_125m_l256_500m`, absent from this checkout but **intact on `main_0726`**:
```
git ls-tree -r --name-only main_0726 | grep fmrope_125m_l256_500m
  .../fmrope_125m_l256_500m/{__init__,prepare,protocol,run_experiment}.py
  .../fmrope_125m_l256_500m/{SPEC.md,run_5090.sh,run_full_queue_5090.sh,...}
```
The closure is exactly two packages — that one plus `rebuttal/rebuttal_0723/experiments/geo_rope_contract.py` (also on `main_0726`). Verified: the recovered `run_experiment.py:25` imports `from experiments.native_rope_evq_150m.model import GPT`, which **already exists** in the current tree; `protocol.py:23-24` imports `official_yarn_on_inv_freq` and `evq_cosh_phi`, both **present** at `scripts/lib/rope/official_yarn.py:167` and `scripts/lib/rope/schedules.py:162`; `geo_rope_contract.py:17` needs only `scripts/lib/rope/schedules.py`. So a `git checkout main_0726 -- <2 paths>` plausibly restores the trainer outright.
**Residual risk to check before trusting it:** `scripts/lib/rope/` may have diverged between `main_0726` and `09_09`. Presence of the symbols is confirmed; *signature and numerical* compatibility is **not** — that must be verified by re-running the probe and matching the frozen SHAs in `experiments/rotary_budget/evidence/probe_mb8_02.json` (`initial_trainable_sha256: fb176482...`, `row_order_sha256: 19c69384...`).

**2. Add sparse/hybrid attention to the 150M model (medium — the core build task).**
`experiments/native_rope_evq_150m/model.py:101` is dense SDPA only. A sliding-window / block-compressed / top-k layer must be written into it, with a per-layer type dispatch (which does not exist anywhere yet). The good news: working reference implementations of all three primitives exist and are pure PyTorch — `experiments/nosa_position/runtime.py` (compress+select+local, `AttentionSettings` at line 40 with `dense: bool` already a clean per-model toggle) and `scripts/experiments/sparse_memory/model.py:90-98` (banded mask). They are written for inference on frozen checkpoints, so they must be re-derived for training (differentiable selection, no `from_pretrained`).

**3. Choose which harness hosts the sparse layer (design decision, not a build task).**
Two viable hosts, and the choice should be made deliberately rather than by default. (a) **The recovered 150M path** — gap 1 plus gap 2; keeps the paper's 151.9M lineage and its exact receipts, but the trainer is currently unqualified on `09_09`. (b) **`scripts/core_text_phases/run_evq_sweep.py`** — already trains 50M/125M/350M from scratch with a working loader, tokenizer, checkpointing, resume, and multi-length eval, so swapping `Attention.forward` for a windowed/compressed variant is a fork of ~1 class; but it has **no 150M tier** (`TIER_CONFIGS`, line 80) and would move the comparison off the paper's 151.9M lineage. Given gap 1 turned out to be a restore, (a) is now genuinely competitive and preserves comparability with the existing 151.9M receipts.

**4. Decide and justify the arm set against pre-registration (small, but must precede any launch).**
`experiments/rotary_budget/REPORT.md:76-78` forbids result-dependent cancellation and requires ≥268M tokens/arm to be called anything but a pilot. `ALL_USER_OPS.txt:48666` requires success/failure criteria written **before** the run. A 4-arm sparse comparison must state its contrast, its kill criteria, and its compute budget up front.

**5. Measure 150M throughput at the target sequence length (small — hours of GPU).**
No 150M-class throughput at seq_len ≥ 4096 is recorded. `RTX5090_BLACKWELL_PROFILE.md:175-178` requires discarded probes rather than guesses for any new workload. Without this the four-arm cost is unknown, and given the 2-minute idle ban and ≥28 GB memory rule, a probe is not optional.

**6. Long-context eval for sparse models (medium-large).**
Existing evals (`eval_454m_multilength.py`, `eval_super_extrap.py`, `eval_dsr.py`, passkey/NIAH) all assume a dense causal mask and a single global RoPE table. A hybrid model's eval must respect per-layer windows and the local/global split; `experiments/nosa_position/run.py` is the closest working example of a sparse-aware multi-length harness, and it is written against a frozen pretrained checkpoint.

**7. Data and tokenizer for the sparse run (small).**
The 150M lineage used `499,974,144` tokens at L=256 from a manifest at `/root/autodl-tmp/iclr_exact_range_multiseed/data/data_manifest.json` (`REPORT.md:66-70`), tokenizer GPT-NeoX revision pinned in `protocol.yaml`. `REPORT.md:69-70` warns: *"Verify actual existence/hashes before use. Held-out long-document evaluation must be rebuilt from shard004; the old packed validation anchors are unsuitable."* The 4K/8K data for a long-context sparse run does not exist locally (no 150M-class checkpoints or token arrays on disk).

**8. Environment for sparse kernels (small, only if efficiency claims are wanted).**
`flash-attn`, `xformers`, `triton`, `einops` are all absent from `.venv` and from both requirements files. Pure-PyTorch sparse attention will train, but any *efficiency* claim about sparse vs dense would be unsupported without them.
