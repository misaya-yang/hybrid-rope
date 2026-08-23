# Experiment plan

- **Date:** 2026-08-23
- **Status:** **PLAN. Nothing below has been run.** No GPU work is authorised by
  this file. Costs are measured, not guessed (§1).
- Every arm is hash-bound before GPU time, per `AGENTS.md` §4.

## 0. Implementation state observed at review time

`scripts/analysis/rope_transport/same_support_controls.py`
(SHA-256 `9b77ac431a7f6be8d31e70034b0902f234e5c947773afca3cc7f557d38fd0049`) and
the updated `scripts/eval/target_free_ruler_smoke.py`
(SHA-256 `c2612c0c111c509d20e6db8831ecab0a628f5bbcca964452fd2cf7d1d0c8d010`)
already implement the three same-support controls. This plan is written against
that implementation, not against a proposal.

Audited and **correct**:

- All three controls pin both endpoints exactly (`table[0] = native[0]`,
  `table[-1] = native[-1]/factor`) and assert strict monotonicity, so the
  fixed-support contract holds by construction and no order crossings survive.
- All three use `matched_attention_scaling(4.0)` = `1 + 0.1 ln 4`, identical to
  both the frozen method and official YaRN. Amplitude is held fixed.
- All three are gated on `--expected-active-sha256`; a table drift aborts.
- `nearest_linear_ramp_control` fits the ramp to the **movement profile**, never
  to task scores (`selection_uses_task_labels: False`). No leakage path.
- Dispatch for OLMo is correctly guarded at
  `target_free_ruler_smoke.py:335` (`is_olmo and args.method not in
  CONTROL_METHODS`), so controls reach `load_cross_model_method` on OLMo too and
  are **not** silently run as Native.
- dtype matches: `load_cross_model_method` uses bf16, and the OLMo path's
  `load_model` (from `olmo2_lora_conversion`) also uses bf16. **Verified** — the
  float32 `load_model` in `evaluate_ruler.py` is a different function and is not
  on this path.

One limitation to note before running: `allowed_lengths` still restricts
evaluation to $\{2L_{\text{native}}, 4L_{\text{native}}\}$, so none of these
controls can produce an in-window cell (see E4).

## 1. Measured cost (RTX 4090 48GB, generation wall time, excludes ~2 min load/validate)

| Protocol | rows | min per arm |
| --- | ---: | ---: |
| OLMo core-4, both 8K+16K | 160 | **3.3** |
| OLMo unseen-9, both 8K+16K | 360 | 23.8 |
| OLMo 2Wiki full-200 @16K | 200 | 1.3 |
| OLMo Qasper full-200 @16K | 200 | 1.2 |
| OLMo formal matrix, both shards | 386 | 9.3 |
| Qwen core-4 @64K, n=20 | 80 | 5.6 |
| Qwen core-4 @128K, n=20 | 80 | 17.1 |

## 2. Frozen table identities

Compute with `build_same_support_controls(native, native_context_length=L, factor=4.0)`.
Pass the matching value as `--expected-active-sha256`.

**OLMo-2-0425-1B-Instruct** ($L=4096$)

| Arm | float32 SHA-256 |
| --- | --- |
| `converged_budgeted_s4` | `a435d75441444bcea39b73d9cf530005249dc5afdc3cfb5a60fda10ef33312d3` |
| `same_support_geometric_s4` | `2754c9c233fe6f65686e86189c4cee94b75efac0bd8df19f140f3e575ffb723f` |
| `nearest_yarn_ramp_s4` | `be76ee4cfb8524ef5d52660b4dd63e7331f53ac9f788305c4fc3dcface7817f1` |

`converged_budgeted_s4` is **bitwise identical** to the deployed frozen table, so
on OLMo it is a same-code replication, not a new operator.

**Qwen2.5-1.5B-Instruct** ($L=32768$)

| Arm | float32 SHA-256 |
| --- | --- |
| `converged_budgeted_s4` | `3512335e408c279896a84b5b555b8be5ca3b62a70eb300f3ec866c5b5aa141d0` |
| `same_support_geometric_s4` | `4046a441284959f8d6f635a690b0f3e9ea2b4c18cd8b43eae20ce57463714db8` |
| `nearest_yarn_ramp_s4` | `c5e4576ac4fee4b60b6a887cecdb1c3f4cc33971b83645bc8fb95ce96658b9c9` |

On Qwen `converged_budgeted_s4` is a **different table** from the deployed
`15754e60…`. Existing Qwen numbers belong to the frozen table and must never be
merged with converged-table numbers.

---

## E1 — OLMo same-support identification triple, core-4 · **highest value**

**Question.** At fixed support, fixed amplitude, zero training, on a mature
checkpoint: does the interior allocation decide the outcome, and does the
redundancy measure contribute more than the split location?

**Arms** (3): `same_support_geometric_s4`, `nearest_yarn_ramp_s4`,
`converged_budgeted_s4`. Existing arms for context, already run: official YaRN f4
(0.2225 / 0.0125), deployed binary (0.7175 / 0.4075), Native (0.0000 / 0.0000).

**Protocol.** Identical to the completed core-4 runs: dataset
`6f64aa6bb44821f5…`, 20 rows per task, 8192 and 16384, greedy, official scoring.

**Cost.** 3 × 3.3 min ≈ **10 min GPU** (~20 min wall).

**Pre-registered readings.**

| Outcome | Reading |
| --- | --- |
| geometric ≈ floor, budgeted ≫ geometric | Interior allocation is decisive at fixed support on a mature checkpoint, training-free. This is the paper's central claim demonstrated on a released model — promote. |
| geometric ≈ YaRN, budgeted ≫ both | Same conclusion, plus: "non-geometric" is not sufficient; placement is what matters. Strengthens §6 of `02`. |
| geometric ≈ budgeted | **The line dies.** The effect is support stretching, not allocation. Do not put any of it in the paper. |
| ramp ≈ budgeted | Contribution = the derived split location only. Legitimate, but the manuscript must say exactly that and drop all operator-novelty language. |
| ramp ≪ budgeted | Ramp *shape* matters beyond the split. Surprising; needs its own explanation before any claim. |

**Stop rule.** If `geometric ≈ budgeted` within the core-4 resolution, stop the
entire retrofit line and record the negative. Do not proceed to E3.

**Replication check.** `converged_budgeted_s4` must reproduce 0.7175 / 0.4075
exactly. If it does not, the discrepancy is a code-version defect and everything
downstream is on hold until it is explained.

---

## E2 — Close the Qwen 128K sample-asymmetry hole

**Question.** None — this is hole-closing. The 128K comparison is currently
unusable (`01` §5).

**Arms** (2): `native`, `official_yarn` at 128K, `--limit-per-cell 20`, script
`c2612c0c…`. Binary at n=20 (0.6175) already exists.

**Cost.** 2 × 17.1 min ≈ **35 min GPU**.

**Stop rule.** None; run to completion. Record the three-arm n=20 macro and its
bootstrap CI. Delete the n=5 three-arm comparison from all downstream material —
it is now known to be biased by +0.0725 on the binary arm.

---

## E3 — Extend the winning triple to breadth · **conditional on E1**

Only if E1 shows geometric ≪ budgeted.

**Arms.** The same three controls on: unseen-9 RULER (both lengths), the 386-row
formal matrix, full-200 2Wiki, full-200 Qasper.

**Cost.** 3 × (23.8 + 9.3 + 1.3 + 1.2) ≈ **106 min GPU**.

**Why it matters.** Without this the identification exists only on the selection
task set. With it, the fixed-support triple is demonstrated on a hash-bound
confirmation set and on real documents, which is what makes it manuscript-grade
rather than a probe.

---

## E4 — Qwen in-window parity · requires a code change

**Question.** Does the *routing* half of the binary policy hold on Qwen? Today it
is untested there: the Qwen path installs s4 once at load time and every
evaluated length exceeds $L_{\text{native}}$, so the Native branch never
executes (`05` DEFECT-1).

**Change required.** Allow `1 × L_native` in `allowed_lengths` and make the
non-OLMo path select per row via `select_observed_session_factor`, recording
`active_profile` exactly as the OLMo path does.

**Deliverable.** One line: "Qwen binary 1x outputs are bitwise identical to Native
on N/N rows." Without it the word *binary* is not earned on Qwen and the
cross-model claim covers only the long branch.

**Cost.** Small (32K rows are cheap) plus the code change and its tests.

---

## E5 — Third model, different $d$ or $K$ · **only if E1 and E3 pass**

**Question.** Is the derived $\approx7$-rotation threshold a property of the
measure or a coincidence of two 128-dim / 64-pair models?

**Requirement.** A released checkpoint with a **different head dimension or pair
count**, not merely a different native window. Two more 128/64 models would add
almost nothing.

Do not schedule this before E1. If E1 kills the line, E5 is meaningless.

---

## Do not run

These are closed by completed negatives or by the stop-list in `HANDOFF.md` §8,
and re-running them consumes GPU without changing any score ceiling:

- Any further sweep of **our own** $p$, $s$, $c$, `max_points`, rank, gain,
  learning rate, or steps. The frozen-transfer property is the entire value of
  the Qwen result; retuning on Qwen destroys it.
- Any 128K run at n=5.
- The stateless boundary-slope operator (core-4 `0.0000/0.0000`).
- The smallest-covering Native/s2/s4 router (full 2Wiki 0.2473, below both binary
  and fixed s4; coverage is not a performance selector).
- The far-pass chord CE-only retrofit route (final physical-8K arm: 1/64
  first-token top-1, 0/64 answer-plus-EOS).
- The headwise rank-16 Q/K adapter (0.3875 vs 0.4000 for its zero-training parent).
- CPU table-axis searches ($D^\*$, coverage, phase-risk) — falsified by the
  one-turn floor counterexample.
- RULER-13 as a *selection* instrument. It is the frozen confirmation set; using
  it to choose anything destroys the only clean breadth evidence in the line.
- The mature function-morph audit — deprioritised and unrelated to this route.
- 8B multi-seed by default.

### A distinction that matters

Sweeping **our** hyperparameters is forbidden. Running additional **baseline**
configurations (a $\beta$-shifted YaRN, NTK-aware scaling, position
interpolation) is not the same act: it can only make our claim harder to
sustain, and it touches no method selection. E1's `nearest_yarn_ramp_s4` is
exactly such a baseline, and it is derived from the movement profile rather than
fitted to scores. Keep that line sharp in every writeup.
