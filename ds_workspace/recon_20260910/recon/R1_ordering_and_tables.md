# R1 — Recon: MrRoPE table construction, increment ORDERING, and the frozen-Qwen measurement stack

Survey date 2026-09-10. Branch `09_09`. Read-only survey; no experiment run.

## (a) Executive summary

1. **MrRoPE's table is reproduced exactly** at `analysis/unify_20260910/tables/rebuild_ground_truth_tables.py:165-172` (`radial_family_m`) and `experiments/curvature_20260910/tables.py:71-83` (`m_mrpro`). Both are the canonical `m_j = q(q+1)/(N(N+1))`, `q = clip(j-23, 0, 17)`, `N=17`, `m=1` for `j>=40`. Bit-exact vs the deployed fp32 tensor (0 error).
2. **Slot convention confirmed**: `23` and `40` are **0-based slot indices** on K=64, not gap indices and not pair counts. `low=23, high=40` ⇒ **17 increments**, 24 native slots (0–23), **16 strictly-interior slots (24–39)**, 24 fully-compressed slots (40–63). `Σm = 29.3333`.
3. **The ordering question has NOT been measured.** No experiment varies the order of `eps_q` with the increment multiset, endpoints, and span held fixed. The "regressive" reverse sequence `eps_q = 2(N+1-q)/(N(N+1))` does not exist anywhere in the corpus.
4. **Ordering is not untouched conceptually, but the receipts are for a *different object***: three measured whole-interior random permutations (of final *frequencies* / dilation factors, not of `eps` confined to 23–40), all catastrophic. Their own receipt flags the confound: "Dilation permutation also changes final spectrum; no isolated slot cause."
5. **The closest-designed-but-unrun experiment is F9/K6, the four-cell counterfactual** (`ν^fast = max(ν^Y, ν^M)`, `ν^slow = min(…)`) — declared "highest priority GPU verdict", never executed. It is a *two-sided splice*, not an ordering permutation.
6. **MrRoPE vs YaRN on frozen Qwen has NO local receipt.** YaRN_linear is table-verified bit-exact but its panel score columns read `—`. This is a real hole in the existing comparison set.
7. **The `87.22 / 78.13` numbers are MrPro's 6-task RULER dev-subset macro accuracy in PERCENT out of 100, at 32K / 128K.** Verified against the raw JSON: `0.8722222 / 0.78125`. The `/128` reading is a numeric coincidence.
8. **No per-distance-bin or per-position logit-margin measurement exists as a reusable tool.** The single genuine correct-vs-competitor margin measurement (`−2.125 / −1.125 / −1.000 / +0.250` nats) is a hardcoded single case, and its own doc asks for exactly what the new experiment wants: "A transferable method should improve this signed margin on new inputs."
9. **V-E2 exists and is load-bearing.** Exact origin: `docs/research/ROPE_RESEARCH_FAILURE_REVIEW_20260907.md:178`. The label V-E2 is a later addition in a digest. A near-neighbour veto, **V-E4 (do not re-derive the same multiset limit by permuting frequencies)**, is directly on the path of any ordering experiment and must be answered explicitly.
10. **`FrozenRoPE` is new and untracked.** It lives only in `experiments/curvature_20260910/model.py:63`, a directory `git ls-files` reports as empty. There is no older, established frozen-Qwen eval class to inherit.

---

## Q1. Where is MrRoPE's table constructed? What is the 23/40 convention?

### 1.1 Canonical reimplementations

**`analysis/unify_20260910/tables/rebuild_ground_truth_tables.py:165-172`** — `radial_family_m(N_, dl=23)`:

```python
def radial_family_m(N_, dl=23):
    """MrRoPE Eq.14 径向族：m_j = q(q+1)/(N(N+1)), q=clip(j-dl,0,N)；
    N=17 即 MrPro（dl=23, dh=40），N'=16/15 为收缩族（完成槽=23+N'）。"""
    m = np.zeros(DR)
    for j in range(DR):
        q = min(max(j - dl, 0), N_)
        m[j] = q * (q + 1) / (N_ * (N_ + 1))
    return m
```

Built at `:175` as `mr_m_formula = radial_family_m(17)`, converted at `:176` via `NATIVE * np.power(4.0, -mr_m_formula)` — i.e. **S=4 is baked into the table identity**, not passed in.

**`experiments/curvature_20260910/tables.py:71-83`** — `m_mrpro(n=17, low=23, k=K)` (the newest, M-coordinate-native version, with an explicit `m[...] = 1.0` completion clause at `:82`). Its docstring at `:73-78` states the design logic verbatim:

> "Increments arithmetic in q and the endpoint m(n) = 1 together force the quadratic -- there is no free parameter left once n and the support are fixed. n = 17 is the member whose transition width equals YaRN's dim\*ln(beta/alpha)/(2 ln theta); n = 16 and n = 15 are the two neighbouring members of the same family, constructed but never evaluated."

**Verbatim spec string** at `rebuild_ground_truth_tables.py:352`:

> `'MrRoPE-Pro：nu_j=omega_j*4^(-m_j)，m_j=q(q+1)/(N(N+1))，q=clip(j-23,0,17)，N=17（dl=23, dh=40，YaRN 式 alpha=32/beta=1 界；Eq.14 径向 λ 族 N=17 成员）；j>=40 m=1；gain=1+0.1*ln4=1.138629436111989'`

**YES — the `m_q = q(q+1)/(N(N+1))` with `q = clip(j-23, 0, 17)` implementation exists, in two places.**

**False positive to avoid**: `experiments/rope_operator_family/progressive.py` is *progressive output distillation* (layer-at-a-time student-prefix fitting). Nothing to do with progressive increments.

### 1.2 The 23/40 convention — exact meaning for K=64

Confirmed independently in three places:

- `docs/research/rope_allocation_20260910/agents/sol16.md:33` — "The current MrPro transition is exactly zero-based `low=23`, `high=40`, so `N=17` transition gaps."
- `analysis/kkt_20260910/mine/R2_ground_truth.md:78` — "`m_q = q(q+1)/(N(N+1))`, `q = clip(j−23, 0, N)`, `N=17`, 界 `dl=23, dh=40`；`j≥40` 时 `q=17 ⇒ m=1`."
- `analysis/unify_20260910/tables/GROUND_README.md:49`

**Convention (K=64, zero-based pair index `j = 0..63`):**

| j range | count | q | m_j | band |
|---|---|---|---|---|
| 0 … 23 | 24 | 0 | 0 | high-frequency, native, untouched |
| 24 … 39 | **16** | 1 … 16 | `q(q+1)/306`, strictly in (0,1) | **strictly interior transition** |
| 40 … 63 | 24 | 17 | 1 | low-frequency, fully compressed by /4 |

`23` and `40` are **slot indices**, not gap indices. The **17 increments** are the gaps *between* slots 23→24 … 39→40; `eps_q = m_q − m_{q−1} = 2q/306`. Slot 40 is the *completion* slot (m reaches 1), not an interior member.

Sanity check that reproduces the published `Σm`: `Σ_{q=0}^{16} q(q+1)/306 = 1632/306 = 5.3333`; plus 24 slots at m=1 → **29.3333**. Matches `GROUND_README.md:49` (`Σm = 29.333`). ✓

**The boundaries are derived, not tuned.** `analysis/kkt_20260910/mine/R4_table_constructions.md:138`:

> "`low=23, high=40` 是**几何推导出来的、不是超参**：`low = floor(d·ln(W/(β_fast·2π))/(2 ln b))`、`high = ceil(d·ln(W/(β_slow·2π))/(2 ln b))`，`β_fast=32, β_slow=1` … 对 Qwen3B 精确算出 `23.5959 / 39.6509` → `[23,40]`；对 OLMo1B → `[14,32]`。**这与 MrPro 的索引集完全相同**——两族共用同一分带边界。"

Reproduced in code at `experiments/curvature_20260910/tables.py:86-99` (`yarn_bands`), which also flags the discretisation: ramping over the raw float edges instead of floor/ceil "moves slot 39 by 0.034 in m, which is a third of a native log-gap."

**Notation trap**: `N` is overloaded. In `m_q = q(q+1)/(N(N+1))`, `N=17` counts **increments**. Elsewhere the same doc set writes `N=17` for the *radial family member index* and `K=64` for pairs. `sol16.md:44` states the effective freedom correctly: "The 17 positive radix increments are `ε_i = 2i/[17·18]`, `i=1..17`, and sum to one. Hence the constrained transition has **16 effective allocation degrees of freedom**."

---

## Q2. Has anyone varied the ORDER of the eps increments (same multiset / endpoints / span)?

### Verdict: NO — this exact experiment does not exist.

**Classification: (c) NOT FOUND.** Three independent search passes (MrRoPE/progressive/ordering keywords; eps/increment/permutation keywords; whole-repo ripgrep of `permut|shuffle|reverse|reorder|sorted|order`) found no arm, result file, contract, or plan-with-receipt that reorders the 17 middle-band increments while holding the multiset fixed. The reversed sequence `eps_q = 2(N+1−q)/(N(N+1))` does not appear anywhere.

### What DOES exist — measured, but a different object

**(a) MEASURED — three whole-interior random permutations.** All receipts in one file: `paper-2027/research/attention-aware-retrofit/results/coupling-transfer/SCALE_CONSISTENT_LOG_PROFILE_RESULT_20260831.md`.

- `:271-279` (OLMo, pre-registered, final-frequency multiset): candidate 1× PG-19 NLL **`6.864926`** vs reference **`3.104234`**, `ΔNLL = +3.760692`, crossing the pre-registered H1 threshold `+0.10`. SHA-256 `ee03b6dae…`.
- `:261-268` (OLMo, random interior dilation permutation): NLL **`4.068625`**, core-4 **`0/0`** at 8K/16K. The receipt's own caveat: *"The permutation also changed the final frequency multiset and introduced 14 order crossings."*
- `:186-192` (Qwen 64K, interior slot assignment only): *"All four 20-row task cells scored zero, versus macro `0.7000` for the ordered reference."* SHA-256 `056938bce…`.

**Why these do not answer the ordering question**: they permute **final frequencies or per-gap dilation factors across the whole interior**, not `eps_q` confined to slots 23–40. Permuting increments changes `m_j` at every downstream slot, so they are not a fixed-`m`-multiset test either. And the permutation is random, not the specific progressive→regressive reversal.

**The corpus already records this limitation.** `docs/research/ROPE_LOCAL_FAILURE_EVIDENCE_20260908.json:3703` (case F10):

> `"valid_evidence": "random1xNLL4.068625/RULER0/0"`, `"cannot_conclude": "Dilation permutation also changes final spectrum; no isolated slot cause"`

**(a) MEASURED — one single-slot sign mirror (not an increment permutation).** `E1_s28_reverse_matched`, deployed `m28 = 0.132270`, changed slots `[28]`. 36-row panel **100.0 / 64.4444**, flat vs the 12-row baseline. Recorded at `analysis/kkt_20260910/tables/ground_truth_tables.json:4051` and `analysis/kkt_20260910/mine/R1_panel_data.md:93-97`. Flagged `部分证据-单源`: the mirror formula was not locally recoverable (candidate `2m28 − m27 = 0.130719` does **not** match the deployed `0.132270`; `GROUND_README.md:67`).

### (b) CLAIMED only — reverse-control proposals with no receipt

`docs/research/rope_allocation_20260910/agents/astra04.md:67`:

> "use `v = x_fit − x_Mr`; scale both `+v` and `−v` by the same largest factor ≤1 preserving order if a reverse control is needed. Both directions keep endpoints and cumulative compression. Evaluate positive, reverse, and MrPro with identical gain on newly generated independent full-model long tasks."

Echoed future-tense at `analysis/unify_20260910/digests_codex/digest_calibration.md:36` and `digest_constructive.md:74`. **No `x⁺`/`x⁻` result JSON or `contract.json` exists** — and note these reverse a *fitted direction vector*, not the `eps_q` sequence.

### Shape comparisons across DIFFERENT multisets (measured, but not an ordering test)

- **MrUni** = transition linear ramp `m_j = (j−23)/17` ⇒ **constant** increments. Measured **64.58 / 73.33**. `GROUND_README.md:55` explicitly corrects a widespread misreading: "(**非全表÷4**)" — it is not the whole table ÷4.
- **MrPro** = arithmetic increments `2q/306`. Measured **87.22 / 78.13**.
- So Pro-vs-Uni *is* a same-endpoints, same-span, different-increment-**multiset** comparison (progressive vs constant). It is **not** a permutation test — it changes the multiset by construction, and it also changes `Σm` (29.333 vs 32.000).
- The corpus claim `analysis/unify_20260910/digests/digest_mrrope-evq.md:73` — "Pro（progressive）> Uni（uniform）> YaRN（regressive）在中间带" — is sourced to the **external MrRoPE paper**, marked `[已验证-论文]` with the in-repo panel only `[部分证据]` reproducing "YaRN<Pro；Uni 未测真身". **No in-repo receipt ranks all three.**

### The nearest designed-but-unrun experiment: F9 / K6 four-cell

`analysis/unify_20260910/STARTING_POINT_YARN_VS_MRPRO.md:89-99`, §7:

> 定义 `ν^fast_j = max(ν_j^Y, ν_j^M)`、`ν^slow_j = min(ν_j^Y, ν_j^M)`（=M_A+Y_B / Y_A+M_B）。零新增曲线参数、同端点、同频率数、同 gain、**保排序**。**必须从完整 prefill 执行**（不得用固定状态重放替代）。

Designated at `analysis/unify_20260910/INTEGRATION_20260910.md:111`: "**K6 四格实验 = 最高优先 GPU 判决**". **Never run — no receipt.** It is a two-sided *splice* (per-slot min/max of two families), which reorders *which family supplies which slot*, not the increment multiset.

### Two claims I checked and am correcting

1. **A derived report claimed `sol16`'s softmax-Helmert family "cannot represent permuted/reversed orders." That is incorrect.** `docs/research/rope_allocation_20260910/agents/sol16.md:46-53` defines `eps(η) = softmax(log eps^Mr + Bη)` with a `17×16` Helmert basis `B` for the zero-sum subspace. The image is the **entire open simplex** `{eps > 0, Σeps = 1}` — which includes every permutation and the full reversal. What the family guarantees is that `Σeps = 1` and `m` is strictly increasing (which holds for *any* positive `eps`, so it is not a constraint on ordering). So the family **can** express the ordering question; it simply was never used that way. Its CPU reference (`sol16.md:55`, `.agents/rope_unification_20260910/code/sol16_frequency_calibration_reference.py`) is also **not in this repo** — `.agents/` was archived and deleted (see memory / commit `78421f1`).
2. **No doc argues ordering is irrelevant.** The nearest thing is a *pre-registered contingency interpretation* in an unrun package: `experiments/curvature_20260910/README.md:218` — "**this is a positive finding.** Budget alone determines the outcome, which means every geometric family is a budget choice wearing a shape, and the band structure is a consequence of the endpoints, not of the interior profile." That is a planned reading of a not-yet-obtained result, not a measurement.

### The veto that sits directly on this path

**V-E4** — `analysis/unify_20260910/digests/digest_failure-records.md:203`:

> `| V-E4 | 用频率置换重新发现同一个 multiset 限制 | 限制已确立；保留槽位身份，不重复置换 | SYNTHESIS §5 行 101 |`

Any same-multiset reordering experiment must be **explicitly distinguished from V-E4** (offer: V-E4 forbids re-deriving the *multiset* limit; an ordering test holds the multiset fixed and varies only the arrangement — a different question) or it will read as re-running a dead route.

---

## Q3. Pre-existing MrRoPE / YaRN / EVQ comparisons on the frozen Qwen models

### 3.1 Disambiguation of the headline numbers — RESOLVED

| number pair | method | metric | status |
|---|---|---|---|
| **87.22 / 78.13** | **MrPro** | 6-task RULER dev-subset macro accuracy, **percent out of 100**, columns = **32K / 128K** | **MEASURED** |
| 91.67 / 70.83 | MrProBM ("BM") | same metric/denominator/columns | MEASURED |
| 87.2 / 68.3 | **Smooth_MrBudget** (a *different* method) | same metric | digest-curated; mirror absent locally |

The metric is the mean of 6 per-task accuracies over the dev subset `{niah_single_2, niah_multikey_2, niah_multiquery, vt, fwe, qa_1}`, 36 rows total (12 @32768, 24 @131072). **A `/128` reading of 78.13 is a numeric coincidence** (`0.78125 = 100/128`). There is a *separate real* trap: the 12-row subset's MrPro baseline is `100.000 / 64.4444` and must not be compared against the 36-row `87.22 / 78.13`.

### 3.2 MEASURED receipts

**MrPro vs MrProBM, 36-row Qwen2.5-3B-Instruct panel, frozen, static S=4.**
File: `docs/research/ROPE_BM_TRANSFER_RESULT_20260908.json`, key path `models.qwen3.arms.MrPro.summary.by_length` → `macro_accuracy` = `0.8722222222222222` @32768 and `0.78125` @131072. Raw jsonl: `results/bm_transfer_20260908/run_qwen3_01/MrPro.jsonl`. Verdict `models.qwen3.comparison.status = "NO_LONG_GAIN"`, `macro_delta_by_length = {32768: +0.04444, 131072: −0.072917}`.
MrPro per-task @32K: single 1.0, mk2 1.0, mq 1.0, vt 0.9, fwe 0.8333, qa_1 0.5.
Config: model revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, seed 20260913, greedy, BF16, Flash SDPA, FP32 freqs, gain `1.138629` (=1+0.1·ln4), 29,740–131,039 tokens/input, 3.5M input tokens/arm.

**NLL (PPL proxy), 16 docs × 8K/16K/32K, tail-512.**
File: `docs/research/ROPE_QWEN3_BM_NLL_RESULT_20260908.json`. Native `2.27607 / 2.14317 / 2.03850`; MrPro `2.32333 / 2.18886 / 2.08899`; BM `2.32581 / 2.19311 / 2.08789`. 16K bootstrap CI `[0.00086, 0.00747]` excludes 0.

**GapCapped** 36-row receipt: `results/bm_transfer_20260908/gap_capped_run_01/GapCapped.jsonl`; scores `84.4444 / 62.1528` (summary null locally; computed in `digest_panel-results.md:70`).

**Qwen2.5-1.5B P2 transfer**: `docs/research/ROPE_QWEN15_FULL_LAG_P2_RESULT_20260907.json` (keys `summary64`, `summary128`, `checkpoint_transfer_3b64`). @64K mk2: P2 37.5 / MrPro 12.5 / matched-gain 25.0; vt 87.5/82.5/77.5; fwe 70.83/45.83/45.83.

**Qwen2.5-7B BM transfer**: `docs/research/ROPE_QWEN7_BM_RESULT_20260908.json`, raw `results/bm_transfer_qwen7b_20260908/run_screen_02/`. MrPro 83.33 / 84.44; BM 80.00 / 71.11 (0W/3L/15T).

**OLMo-2-0425-1B (contrast case)**: `docs/research/ROPE_OLMO_BM_RESULT_20260908.json`. Dev 36 rows: MrPro 37.22/14.93, BM 79.44/49.03. Confirm 72 rows: MrPro 37.85/2.78, BM 81.81/51.32 (44W/0L/28T). Controls: MrUni 76.88/32.12, **OfficialYaRN 54.38/6.94**. Native (24-row 4K) 75.625 vs BM 81.81.

### 3.3 (b) CLAIMED only — 26-arm nongeometric screen

`results/nongeometric_screen_20260909/` **does not exist locally** (verified: `ls` → No such file or directory). The underlying `summary.json`/`contract.json` live only on the remote server. Numbers are recorded in curated digests `analysis/unify_20260910/tables/GROUND_README.md` §3, `ground_truth_tables.json`, `digest_panel-results.md` §5.1. Headlines (32K/128K): `E1_s28_less` 87.22/**83.33**; `E1_s29_more` 95.56/77.92; `pair28_29` 87.22/73.96; `Smooth` 87.22/68.33; `LBS` 80.56/80.07; `LBF` 87.22/73.96; **`HighGapToLong` 70.14/67.36 (0W/7L — the literal EVQ operation, falsified)**; `MrUni` 64.58/73.33; `FullLagP2_Transfer3B` 72.92/81.67.

### 3.4 THE HOLE: YaRN on Qwen is NOT measured

`GROUND_README.md:74` lists **YaRN_linear（官方）** as `bit=True` (bit-exact vs the deployed carrier tensor) but its score columns read **`—` / `—`**. **There is no local receipt measuring YaRN on Qwen2.5-3B or 7B against MrPro or native.** The only measured YaRN receipts are on other models: OLMo OfficialYaRN 54.38/6.94 (above), and Gemma in `paper-2027/research/attention-aware-retrofit/evidence/REFERENCE_CORRECTED_K128_S4_RULER_RECEIPT_20260901.json`.

The external MrRoPE paper's numbers (Qwen2.5-3B full 13-task RULER, Pro 53.2 @128K vs YaRN 50.1) are **CLAIMED**, and `digest_mrrope-evq.md:238` explicitly states they are **not comparable** to 78.13 (different task subset, gain, scoring).

**Implication for the new experiment**: a "beat the model's own un-tuned YaRN at 2×/4×" success criterion cannot currently be evaluated on frozen Qwen — YaRN must be run, or the criterion restated. (On OLMo, where YaRN *was* run, Y2 scored 0/64 and MrPro 37.85/2.78 — see the project memory note.)

### 3.5 Passkey and KL

- **No dedicated Qwen3B passkey benchmark.** The analog is the `niah_single_2` row: MrPro and BM both 1.0 at 32K and 128K (4/4 rows at 128K). `results/legacy/passkey_long`, `baseline_passkey`, `phase14_yarn_passkey` are old small-model/training-era runs.
- **No native-output KL head-to-head for these methods.** `output_kl` exists (`experiments/curvature_20260910/model.py:188`) but is unscored. Only fixed-state proxy KL artifacts exist, and the referenced `bias_prior_local_KL_all_heads.json` is remote-only (NOT FOUND locally).

---

## Q4. The measurement stack for a FROZEN Qwen2.5-3B checkpoint

### 4.1 `FrozenRoPE`

**`experiments/curvature_20260910/model.py:63`** — **the only implementation in the repo**, and **untracked** (`git ls-files experiments/curvature_20260910/` returns nothing; the directory appears as `??` in git status).

```python
class FrozenRoPE:
    """A frozen causal LM whose 64 RoPE frequencies are a settable knob."""
    def __init__(self, path, dtype="bf16", device=None, attn=None, log_softmax_dtype=torch.float32):
```

Public surface:

| member | line | role |
|---|---|---|
| `K = 64` | `:28` | pair count guard |
| `load_frozen(path, dtype, device, attn=None)` | `:35` | loader helper |
| `pick_device()` | `:31` | device helper |
| `install(self, values, gain, track_grad=False)` | `:111` | sets `self.rotary.inv_freq = v`, `self.rotary.attention_scaling = float(gain)`; guards 64 finite non-negative frequencies |
| `install_table(self, table)` | `:123` | consumes `table["values_float32"]`, `table["gain"]`, optional `table["track_grad"]` |
| `_patch_grad_rotary(self)` | `:127` | swaps in a grad-capable `Qwen2RotaryEmbedding.forward` twin (HF's is `@torch.no_grad()`) |
| `enable_grad_path()` | `:143` | gate for the above |
| `logits(self, ids, keep, want_grad=False)` | `:147` | per-position logits |
| `nll(self, ids, keep=512)` | `:156` | scalar |
| `nll_per_token(self, ids, keep=512)` | `:162` | **per-position loss vector** |
| `log_probs(self, ids, keep, want_grad=False)` | `:167` | `torch.log_softmax(lg, dim=-1)` |
| `grad_wrt_freq(self, ids, keep)` | `:171` | frequency gradient |

A differentiable twin (64 frequencies **and** gain as leaves) exists separately at `experiments/joint_kkt_20260910/joint_grad.py:87`, built on the same class.

### 4.2 Native-output KL

**`experiments/curvature_20260910/model.py:188`**:

```python
@torch.no_grad()
def output_kl(model, ids, keep, base_logp, table):
```

Docstring: *"E_u[ KL( p_base(.|u) || p_table(.|u) ) ] over the last `keep` positions. This is the native-preservation metric…"* Evaluated as `float((p * (b - lp)).sum(-1).mean())` in float64. `base_logp` is the **native** model's log-softmax, passed in; `table` is installed via `model.install_table(table)` before the call. Exported in `__all__` at `:320`. Supporting Fisher helpers in the same module: `fisher_diagonal` `:219`, `fisher_cross` `:236`, `fisher_scaling` `:256`, `fisher_all_at_once` `:300`.

### 4.3 Probe / eval entry points

- `experiments/curvature_20260910/local_probe.py` — `load_ids(args, model)` `:55`, `mc_fisher(model, ids, keep, base_table, n_samples, seed=20260910, batch_report=16)` `:70`, `main()` `:109`; instantiates `FrozenRoPE(args.model, dtype=args.dtype)` at `:138`.
- `experiments/curvature_20260910/long_grad.py` — `load_nll_ids(args, model)` `:63`, `load_bind_rows(args, model)` `:71` (reads prepared `screen.jsonl` rows with `prompt_ids` + gold refs), `mean_answer_nll(model, row)` `:95` (**teacher-forced NLL of the answer tokens only** — the closest existing "answer-span objective"), `main()` `:104`.
- `experiments/curvature_20260910/forward_check.py` — `main()` `:147`; imports `FrozenRoPE, output_kl` at `:67`; calls `output_kl(model, nid, args.keep, base_logp, table)` at `:204`.

### 4.4 RULER row loaders and corpus

- **Generator**: `experiments/evq_recovery/prepare_ruler.py:16 main()` — shells out to the pinned upstream `scripts.experiments.scale_transport.ruler_full_prepare` for `TASKS=('niah_single_1','niah_multikey_3','vt')` at lengths (4096, 8192, 16384, 32768), 24 rows each; writes `a.out/ruler_{split}.jsonl` `:52` + `ruler_manifest.json` `:60`.
- **Row schema** (`prepare_ruler.py:47-50`): `{id, source_id, split, task, prompt_ids, references, generation_budget, input_tokens, length_bucket}`.
- **Reader**: `experiments/evq_recovery/data.py:62 class JsonlIndex` (byte-offset jsonl index); `experiments/evq_recovery/evaluate.py:20 select_rows(root, split, panel)` reads `root/'data'/f'ruler_{split}.jsonl'` at `:22`.
- **Scoring**: `experiments/evq_recovery/evaluate.py:38 score_output(row, text)` — `official_recall = sum(ref.lower() in text.lower() …)/len(refs)`.
- Parallel frozen-line generator: `experiments/native_sparse_position/generate_primary_ruler.py:8 main()`, splits `('compact_dev', 2048, 8)` and `('frozen_long', 32768, 32)`, manifest SHA256 `046d3e90b399c6084d688aba96b19140a6667fe866b94180e86acd38c709eaee`; consumers (`support_ruler.py`, `mixture_ruler.py`, `ruler_primary_baseline.py`) all take `--inputs <jsonl>`.

**CORPUS NOT IN REPO.** `find . -name "*ruler*.jsonl"` → **empty**. `data/` contains only `curated/`, `evq_phase9_L2048_50M_tau0_tau1.5/`, `fineweb_val_cache/`, `video_temporal/`. The prepared corpus lives at the server root `/root/autodl-tmp/evq_recovery_20260910` (`experiments/evq_recovery/README.md:61`), from which `select_rows` reads `…/data/ruler_dev.jsonl` and `ruler_test.jsonl`. Required-file list at `experiments/evq_recovery/validate.py:22` = `['cpt_train.npy','lm_validation.npy','lm_test.npy','ruler_manifest.json']`. The probe corpus for the curvature package (`experiments/curvature_20260910/RUNBOOK.md:33`) is `long_inputs/pg19_test_37702.npy`, 131073 tokens, under working root `/root/autodl-tmp/nongeometric_screen_20260909`.

**Consequence**: any frozen-Qwen eval must either restore a server or regenerate the corpus. Nothing runs from this checkout as-is.

---

## Q5. Per-distance-bin / per-position MARGIN measurements — do they exist?

### Verdict: NOT FOUND as a reusable implementation. The right ingredients exist piecemeal.

**(c) NOT FOUND**: no implementation of `margin = logit(correct) − logit(best incorrect)`, and no softmax-probability gap, binned by token distance or answer position, anywhere in `experiments/`, `scripts/`, `analysis/`, `docs/`, `results/`.

### The one genuine correct-vs-competitor margin — measured, single case

**Code**: `experiments/nongeometric_screen/binding_swap.py:12-15`:

```python
def digit_margin(record,a,b):
    if record['generated_ids'][:1]!=[220]:return None
    step=record['token_trace'][1];scores=dict(zip(step['top_ids'],step['top_logits']))
    return scores[a]-scores[b] if a in scores and b in scores else None
```

Positions **are** joined to the score: `:31` records `value_token_positions=[pa[0], pb[0]]`, and `:44-45` store `first_digit_D_plus`, `first_digit_D_minus`, `first_digit_evidence_response`. **But it is hardcoded to one row** (`:21`, `key='niah_multikey_2_131072_2'`, `name='E1_s28_less'`, digits `'6683176'` vs `'9424151'`) — a one-off diagnostic, not a curve tool.

**The measured numbers** — `docs/research/NONGEOMETRIC_MECHANISM_TRANSFER_20260910.md:129-142`, verbatim:

> "At the first differing digit, all four arms have generated the same preceding space. The correct digit is `6` (token 21), and the competing digit is `9` (token 24). Both appear in every top-five trace. Their correct-minus-incorrect logit margins in the four column orders above are **-2.125, -1.125, -1.000, +0.250**.
>
> Thus readout alone improves the margin by 1.000 nat and prefix formation alone by 1.125 nats. Together they cross the greedy decision boundary; the continuous interaction remainder is only +0.250 nat, at the resolution of BF16 logits. It is incorrect to infer a large nonadditive internal mechanism from the binary success pattern alone. **A transferable method should improve this signed margin on new inputs**, rather than merely fit the two observed decision crossings."

That last sentence is the closest thing in the corpus to a mandate for the new experiment. Status: `[部分证据]` — single case, BF16 logits, **not binned by distance**. Corroborated in `digest_thread-0910-batch.md:58`, `digest_theory-0910.md:166`, `digest_failure-records.md:95`, and cross-checked at `analysis/kkt_20260910/mine/A4_sol06_10.md:305` ("逐值一致 ✓ [已验证]").

### (a)/(b) PARTIAL — reusable shapes

| what | where | why it is only partial |
|---|---|---|
| per-step **EOS** margin, top-5 logits retained | `scripts/experiments/olmo_fast_screen/diagnose.py:21 greedy(...)`, trace built `:43-46` (`eos_margin=best_eos-float(non_eos.max())`, `top_ids`, `top_logits`) | wrong quantity (EOS, not correct-answer); nothing bins by distance |
| **NLL-gap**, stratified per `(length, depth)` | `scripts/supporting_eval/eval_passkey_scratch.py:295 eval_passkey_nll_gap`, `gap = nll_wrong - nll_correct` `:486`, per-cell `mean_nll_gap` `:406`, per-trial `trial_gaps` `:368-396` | closest existing stratified curve; quantity is teacher-forced NLL gap, not logit margin |
| per-layer/head attention scores joined to Q/K positions | `experiments/nongeometric_screen/capture.py:49 capture_row(worker, folder, row_id, ids, query_positions, targets, metadata)` | attention scores, not output-logit margins |
| per-position loss vector, unbinned | `experiments/curvature_20260910/model.py:162 nll_per_token` | raw material; no binning helper on top |
| needle retention with position + binary outcome | `scripts/experiments/niah_retention_canary.py:65` (`source_block` = `[position, len(needle)]`) | binary exact-match, no logits |
| position bins (512-token) | `experiments/native_sparse_position/prepare_support_oracle.py:37,42` | used to match controls, not to bin a score |

**Checked and rejected as a source**: `results/pc2_ten_directions_20260910/per_example.jsonl` — 744 rows, key set enumerated; contains no margin, no per-position logits, and no distance field (`topk` is a block-selector config, `length_cap` a cap, `query_ordinal` an ordinal).

**A plan, not code**: `docs/exp/2026-07/2026-07-15_lora_readout_conversion_plan.md:293` — "first-token logit margin `m = z(g_1) - max_{t != g_1} z(t)`", with `:295` "at least 8 position buckets" and `:319` "bucketed by gold-query distance". Never implemented.

---

## Q6. Known failure modes / vetoes on frequency-table search

### 6.1 The six standing vetoes

`AGENTS.md` carries only 4 general rules (test the requested outcome; reason from mechanism; run decision-sufficient experiments; use prior results critically) — **no veto list**. The vetoes live in `analysis/unify_20260910/digests/digest_failure-records.md:200-204`:

| id | verbatim | cited origin |
|---|---|---|
| V-E1 | 自动恢复旧 Cosh 搜索、对手微调、seed42 权重恢复 → "禁止自动恢复"——除非作者显式重新授权 | REVIEW-0907 行 48 |
| **V-E2** | 重启 18 样本/64 自由度的 **margin-gradient 能力优化路线** → 已失败，"不能据此重新启动"；共享频率响应工具只作局部诊断 | **REVIEW-0907 行 178** |
| V-E3 | 把正确局部 Jacobian/Fisher 接成"下一个能力优化器" → 只在正则性与 trust region 内保留 | REVIEW-0907 行 236；SYNTHESIS §6 行 206-209 |
| **V-E4** | 用频率置换重新发现同一个 multiset 限制 → 限制已确立；保留槽位身份，不重复置换 | SYNTHESIS §5 行 101 |
| V-E5 | 在没有可区分预测时继续科学 GPU 工作；把"更稳定/更平滑/局部改善"当完成目标 | REVIEW-0907 行 266-269 |

### 6.2 V-E2 — exact wording and origin, as requested

**The label** (`digest_failure-records.md:201`) is a *later* addition. **The primary source is `docs/research/ROPE_RESEARCH_FAILURE_REVIEW_20260907.md:178`**, and its exact wording is:

> 这避免逐频率调用模型，也不需要完整L×L attention。输入可复用已经保存的选定query与所有可见keys。该量只描述冻结hidden-state框架中的局部块响应；**它不是整网LM风险的梯度，不能据此重新启动历史上已失败的18样本/64自由度margin-gradient路线。**

The immediately preceding section (`:167-169`) names the tool it is attached to: *"已完成的具体修复：正确的共享频率响应，不把它冒充能力优化器"* → `scripts/analysis/shared_frequency_response.py` (64×64 response Gram; algebra verified on small arrays at `:182`, CPU finite-difference max abs error `8.44e−10`; **never run against real model data** — the real Q/K/V cache was only on the then-shut-down server, `:184`).

**The origin of the "18 samples" failure itself** — `paper-2027/research/attention-aware-retrofit/theory/FIRST_PRINCIPLES_RETROFIT_THEORY_MEMO_20260902.md:385-386`:

> "the one independent margin-like measurement attempted in this project (18 samples → 64-D behavioral gradient) **failed its unopened holdout** — **64 DOF on 18 samples** — so the measurement that would have supplied margins has already empirically collapsed."

Also voided at `COMMON_DIRECTION_FEASIBILITY_AND_BASIN_BARRIER_THEORY_20260902.md:15,23` ("**Voided:** the framing of the §7.3 margin-gradient common-direction candidate") and summarized at `results/zero-training-deployment/ZERO_TRAINING_TWO_DAY_EXPERIMENT_SUMMARY_20260902.md:13,254`.

**Note on citation chains**: `analysis/kkt_20260910/mine/R5_vetoes.md:362` cites V-E2 as `D:digest_failure-records.md:L201` — i.e. the KKT workspace cites the *digest*, which cites REVIEW-0907:178. When quoting V-E2 verbatim, cite **REVIEW-0907:178** as the primary.

### 6.3 How the new packages already handle V-E2

Both untracked experiment packages carry a pre-emptive defence, which is the standing precedent a new ordering experiment should follow:

- `experiments/curvature_20260910/README.md:262-311` — §"The vetoed route this package is adjacent to, and why it is not that route", with the verbatim V-E2 quote at `:267-268` and a 4-row distinction table at `:278-283` (what is differentiated / what the objective is / the 64-dof quantity / holdout), plus a "**What would make this package the vetoed route**" failure list at `:303-311` ("treating `F_N` as the objective…; taking an unbounded step and reporting it without the replay gate; re-running until `G > 0` appears. `G ≈ 0` is a result; measuring the gradient on a document the panel scores").
- `experiments/joint_kkt_20260910/risk.py:36-37` — *"WHY THIS IS NOT THE FAILED 18-SAMPLE / 64-DOF ROUTE. V-E2 vetoed '18 samples / 64 degrees of freedom margin-gradient' as a capability-optimisation route"*; `:361` returns the "V-E2 ratio" in the receipt; and `experiments/joint_kkt_20260910/selftest.py:978` mechanically **fails the run** if "the V-E2 acknowledgement is missing from the receipt." That is the strongest existing pattern: make the acknowledgement a **test gate**, not a paragraph.

### 6.4 Other recorded failure modes relevant to table search

- **V-D19** (`digest_failure-records.md:191`): "二元任务分数的 crossover 顺序 ⇒ 强非线性隐藏态交互" — refuted by the near-additive margins; "先按 margin-跨阈值机制解释，非交互不可证但不得预设".
- **Static geometric / response proxies, the whole family** — `analysis/kkt_20260910/mine/T7_tools_b.md:176`, quoting the authoritative verdict: *"Gram、曲率、重构残差、phase risk、覆盖、平滑性变好 ⇒ 生成变好 —— 已有直接反例；18 样本 64 维行为梯度也已失败，不重新包装成'功能需求恢复'"*.
- **The 6-item "corpse" exclusion list** — `docs/research/PARALLEL_NONGEOMETRIC_20X10_PLAN_20260910.md:145`: 18 样本行为梯度外推、Q/K 范数选槽（无符号）、raw logit MSE（平移污染）、head-selective（F21 全零）、Jacobian 表外推（71–468% 误差）、C2 残差拟合（MAE 小、功能距离大）.
- **`docs/research/ROPE_LOCAL_FAILURE_SYNTHESIS_20260908.md:211-218`** — §7 "下一轮如何使用历史，而不是再写一张全否决矩阵"; `:175` explicitly bounds the veto scope: *"它不等于'任何只依赖频率的指标都不可能有用'，也不禁止简单跨模型规则"*. Only 4 items listed in the visible section (C4 etc.); the file is longer than the excerpt I read.
- **`paper-2027/claude_code_workspace/LESSONS.md`** is only **52 lines** and contains **no** ordering, permutation, V-E2, or margin entry (single hit is an unrelated "negative margin" phrase at `:27`). It is not a veto source for this question.
- **`docs/research/ROPE_ALLOCATION_THEORY_CORE_20260910.md`** (170 lines; read in full) contains **no veto list** — it is a problem-statement and materials doc. Its §5 (`:127-145`) is the KKT framing, and `:143` states the free-variable question the new experiment must answer: *"下一轮须明确：变量是有限频率、间隔、密度还是整数通道数；哪些端点／跨度固定；是否允许重复频率；使用整数距离还是连续近似。"* Its §2 (`:44-49`) gives the increment algebra that makes an ordering experiment well-posed: `a_i = a_i(native) + (m_i − m_{i−1})·log S`.
- **`analysis/unify_20260910/digests_codex/digest_calibration.md:67`** — the only *rigorous* comparative principle found: *"`m_q^Uni − m_q^Pro = q(N−q)/(N(N+1)) > 0` for `0<q<N` ⇒ Pro strictly delays all internal compression at equal endpoints — the only rigorous allocation principle, and it is **comparative (vs MrUni), not optimality**."*

---

## (c) GAPS — what does NOT exist and must be written from scratch

Ordered by whether the new ordering-causality experiment is blocked on it.

**G1 — The ordering arm itself. BLOCKING.**
No arm, table constructor, contract, or naming convention exists for "same `eps` multiset, permuted order". Need: a constructor producing the regressive reversal `eps_q = 2(N+1−q)/(N(N+1))`, a random permutation family with a frozen seed, and ideally a family that interpolates order continuously (e.g. via a permutation-symmetric parametrization) rather than three isolated points. Note the well-posedness conditions the corpus already fixed (`ROPE_ALLOCATION_THEORY_CORE_20260910.md:143`): state whether endpoints, span, `Σm`, gain, and slot identity are held fixed. A permutation of increments preserves `Σm` and endpoints **automatically** — that is its main virtue and should be stated as the design's core invariant.

**G2 — A V-E4 / V-E2 acknowledgement gate. BLOCKING (procedural).**
V-E4 forbids "用频率置换重新发现同一个 multiset 限制". Any ordering experiment must state in its receipt *why it is not V-E4* (V-E4 establishes the multiset limit; this holds the multiset fixed and asks about arrangement) and *why it is not V-E2* (not a margin-over-18-samples capability optimiser). The existing best practice is a **machine-checked gate**, not prose: `experiments/joint_kkt_20260910/selftest.py:978` fails the run when the acknowledgement is absent. Nothing comparable exists for an ordering experiment.

**G3 — Per-distance-bin / per-position margin measurement. BLOCKING for the stated goal.**
Does not exist. Must be written: (i) the margin quantity `z(correct) − max_{t≠correct} z(t)` at the first answer token; (ii) a position/distance join — the row schemas carry `value_token_positions` (`binding_swap.py:31`), `source_block` (`niah_retention_canary.py:65`), and `length_bucket` (`prepare_ruler.py:47-50`), so the raw material is there, but **no binning utility exists**; (iii) aggregation into ≥8 position buckets (the count proposed in the unimplemented plan `docs/exp/2026-07/2026-07-15_lora_readout_conversion_plan.md:295`). `FrozenRoPE.log_probs` (`:167`) is the natural substrate.

**G4 — A runnable frozen-Qwen RULER corpus. BLOCKING for any task metric.**
`find . -name "*ruler*.jsonl"` is empty; `results/nongeometric_screen_20260909/` does not exist locally; the prepared corpus is on a shut-down server (`/root/autodl-tmp/evq_recovery_20260910`). Must either restore the server or regenerate via `scripts/experiments/scale_transport/ruler_full_prepare.py:28` / `experiments/native_sparse_position/generate_primary_ruler.py:8`.

**G5 — A measured YaRN baseline on frozen Qwen. BLOCKING for the "beat un-tuned YaRN" success criterion.**
`GROUND_README.md:74` has YaRN_linear bit-exact as a *table* but with no scores. Without it, "each model beats its own un-tuned YaRN at 2×/4×" cannot be evaluated on Qwen — the criterion exists in project memory but has no Qwen receipt.

**G6 — `FrozenRoPE` is untracked, and the whole eval stack is uncommitted.**
`experiments/curvature_20260910/` and `analysis/kkt_20260910/` both show as `??` in git status; `git ls-files` is empty for both. Any new experiment that imports from them inherits an unpinned dependency. Decide explicitly whether to build on it or freeze it first.

**G7 — No `Σm`-matched control exists for shape comparisons.**
MrPro (`Σm = 29.333`) vs MrUni (`Σm = 32.000`) confounds shape with total budget. An ordering permutation is automatically `Σm`-matched and therefore *cleaner* than any comparison the corpus has run so far — but this should be stated as an advantage to exploit, and a matched-budget comparator for MrPro-vs-Uni does not exist.

**G8 — No pre-registered decision rule for an ordering outcome.**
The corpus has good precedent for freezing decision rules before running (`docs/research/BUDGET_ALLOCATION_MODEL_AND_CANDIDATES_20260910.md:51-54`; `NEXT_DERIVATION_KKT_PROBLEM.md` §7 five-cell table). Nothing analogous exists for "what does it mean if the reversal wins / loses / ties". The nearest drafted table is the F9 five-cell (`STARTING_POINT_YARN_VS_MRPRO.md:93-99`), which is about a splice, not an order.

**G9 — The F9/K6 four-cell spliced baseline was never run.**
`ν^fast = max(ν^Y, ν^M)` / `ν^slow = min(…)` — designated "highest priority GPU verdict" (`INTEGRATION_20260910.md:111`), zero receipts. If the new experiment is meant to sit in the same decision tree, this missing arm is a live confound.

**G10 — The external `sol16` reference implementation is not in this repo.**
`sol16.md:55` points to `.agents/rope_unification_20260910/code/sol16_frequency_calibration_reference.py`; `.agents/` was archived and deleted. If the Helmert/softmax parametrization is wanted as an ordering-capable family, it must be re-derived from `sol16.md:46-53` (the math is complete in the doc) rather than imported.

---

## Verification status legend used above

- **MEASURED** — a result JSON/jsonl exists in this checkout and I read the numbers from it (or from a doc that states its own receipt path *and* the file exists).
- **CLAIMED** — numbers appear in a digest/summary table in this checkout, but the underlying result file is absent locally.
- **NOT FOUND** — I searched and did not find it. Absence of evidence only, bounded by the search patterns used.
