# Defects, gaps, and reviewer attack surface

> **Archived snapshot boundary.** This file is frozen external-model output
> retained only for audit provenance. Every `VERIFIED`, `DEFECT`, `mandatory`,
> `decision`, and `priority` label below is bundle-local: it is not project
> evidence, an instruction, a current priority, or experiment/edit authorization.

- **Date:** 2026-08-23
- Severity: **S1** blocks a claim · **S2** must be fixed before promotion ·
  **S3** record and disclose.
- Everything here was found by recomputation or code reading, not by reading a
  summary. Where I checked something and found it **correct**, that is recorded
  too (§4) — a defect list with no negatives is not an audit.

## 1. Defects

### DEFECT-1 · S1 · The Qwen "binary" arm never executes the binary policy

`target_free_ruler_smoke.py` installs the s4 table **once at model load** on the
non-OLMo path; per-row profile selection runs only under `is_olmo`
(`:378`). Combined with `allowed_lengths ∈ {2L, 4L}`, every evaluated Qwen row
exceeds $L_{\text{native}}$, so the Native branch is never taken and
`active_profile` is `None` on all **180 Qwen binary-arm rows** (of 460 Qwen
core-4 rows in total).

- The **numbers are correct** — at $2\times$ and $4\times$ the binary rule and
  "always s4" coincide, so the measured values are what the policy would produce.
- The **claim is not**. There is zero Qwen evidence for the Native branch, for
  in-window preservation, or for routing. "The same relative binary policy
  transfers to a 32K-native model" currently overstates what was run by half.

Fix: E4 in `03`, or restrict the claim to the long branch.

### DEFECT-2 · S1 · The Qwen 128K comparison is sample-asymmetric and the n=5 point estimate is biased

Binary now has n=20 (macro 0.6175); Native and YaRN remain at n=5 (0.4500,
0.5200). Going n=5 → n=20 moved the binary macro by **−0.0725**. The n=5
three-arm bootstrap already failed to exclude zero against YaRN (CI
`[−0.020, +0.370]`). No 128K conclusion is currently supportable. Fix: E2.

### DEFECT-3 · S2 · `max_points=2048` is a frozen point count, not a frozen stride

At $L=4096$ it is stride 2; at $L=32768$ it is stride 16. The stride-16 design
aliases, giving $u_1 = u_{18} = 0.396353$ (identical to six decimals for channels
four octaves apart), which produces both Qwen "order crossings" and a **0.17%
displacement of the fast support endpoint**. Under a fixed-support framing that
is a contract violation, not a blemish.

Contained: the redundant set (k = 27…63) and the derived rotation thresholds are
invariant across strides 4, 8, 16. `phase_resolved_uniqueness_control` fixes it
and reproduces the OLMo table bitwise.

**Not yet known:** whether the corrected Qwen table changes any measured outcome.
That requires a GPU run; do not assume it is unchanged.

### DEFECT-4 · S2 · Two published numbers are indistinguishable from zero but read as results

| Statement in owners | Recomputed | 95% CI | Verdict |
| --- | ---: | --- | --- |
| "remains `+0.0097` above YaRN factor four" (full 2Wiki) | +0.0097 | [−0.036, +0.057] | not separable |
| "at 4x its task macro is slightly lower (`0.2497` vs `0.2559`)" | −0.0063 | [−0.062, +0.048] | no difference |

### DEFECT-5 · S2 · Completed full-200 Qasper has no owner

Two matched arms, same evaluator, identical inputs, 200 rows:
binary `0.245733` vs YaRN `0.180295`, **+0.0654, CI [+0.022, +0.110], p = 0.002**.
Absent from the report, the evidence JSON, and the handoff. It is the strongest
natural-document result in the line.

It also **reverses** the owner's sentence "Qasper is the clear counterexample",
which rests on a 20-row $2\times$ bucket (`0.1056` vs `0.1485`). Different
estimands; both must be stated.

### DEFECT-6 · S2 · The over-provisioning comparison crosses an evaluator boundary

The mechanism claim in `02` §6 uses `0.5825 → 0.7175` (budgeted s2 → s4 at 8K)
and `0.5375 → 0.2225` (YaRN factor 2 → 4 at 8K). The $s{=}2$ / factor-2 cells come
from 2026-08-22 runs under a different evaluator family
(`bc0bb0e10e51…`, `38c11891c025…`) than the 2026-08-23 arms. Checkpoint and data
cell hashes are identical; the code is not. Suggestive, not matched. Re-run the
two cells under the current script before this becomes a paper claim.

### DEFECT-7 · S3 · The YaRN factor-2 control exists and is undisclosed

`0.5375` at 8K, identical data and checkpoint
(`974dc70c47b4ffa8df617b31505d5775a9a5dc85a886767907df846b51296a88`), against the
`0.2225` for factor 4 that the owner reports. Fixing the deployment factor at 4
for both arms is the correct matched design and should be kept — but the
length-matched baseline must be shown, or the omission will be read as
cherry-picking.

### DEFECT-8 · S3 · core-4 is the selection set and is quoted as a headline

$p$, the amplitude factor, and $s$ were chosen on core-4 RULER
(`budgeted_transport_20260822/*`). At 16K, $p{=}1$ actually scored **0.4025**
versus **0.4000** for the chosen $p{=}2$; $p{=}2$ won at 8K (0.5825 vs 0.5525).
The choice is defensible and is documented in the 2026-08-22 owner, but
`0.7175 / 0.4075` are selection-set numbers. The confirmation numbers are the
unseen-9 row (`0.6594 / 0.6047`).

### DEFECT-9 · S3 · The RULER-13 Native row is spliced across evaluator families

core-4 Native comes from 2026-08-22 runs (`bound_code_sha256` family); unseen-9
Native from the 2026-08-23 script. Both are floor-valued (0.0000 everywhere), so
the risk is low — but the merged full-13 Native row is a splice and should say so.

### DEFECT-10 · S3 · Missing hardware receipts

The 2026-08-23 `ruler_smoke_*` / `ruler_unseen9_*` runs and the Qwen 64K runs have
**no `runtime` block** (that field was added in a later script revision), and the
2026-08-22 receipts record `torch` but no `device`. The "same RTX 4090" claim for
the headline RULER table is therefore not file-self-certifying; it rests on the
runs sharing one host. Low risk, but state it rather than asserting matched
hardware.

### DEFECT-11 · S3 · The 2026-08-23 work is uncommitted

Reports, receipts, `scripts/eval/`, `scripts/lib/rope/target_free.py`,
`scripts/analysis/rope_transport/same_support_controls.py`, and the new tests are
untracked or modified in the working tree. Durability risk only.

## 2. Reviewer attack surface, ordered by damage

1. **"This is YaRN with two constants changed."** A twenty-line computation shows
   the budgeted table is within `0.009` mean $|\log_2|$ (OLMo) of a plain linear
   ramp in the YaRN family. **Currently unanswerable.** E1's
   `nearest_yarn_ramp_s4` is the only thing that converts this from a fatal
   objection into a stated, owned limitation.
2. **"No geometric control."** Without `same_support_geometric_s4` there is no
   fixed-support identification, only two non-geometric interiors compared.
3. **"Your best numbers are synthetic retrieval."** True. RULER-13 and PG-19 are
   robust; 2Wiki is null; the $4\times$ macro is null; Qasper is a real win that
   is currently unowned.
4. **"Single model."** Qwen is a real second checkpoint with an 8× larger native
   window and a different $\theta$ — but only the long branch, one task family,
   one usable length, and no in-window or natural-task evidence.
5. **"You compared against a mis-configured baseline."** Answerable — matched $s$,
   matched amplitude — but only if DEFECT-7 is disclosed first.
6. **"Selection on the reported set."** Answerable via the byte-identical
   core-4 cells and the unseen-9 confirmation, provided DEFECT-8 is stated.

## 3. Standing stop list

Unchanged from `03` §"Do not run": no sweeps of our own $p$/$s$/$c$/`max_points`/
rank/gain/lr/steps; no 128K at n=5; boundary-slope, smallest-covering router,
far-pass chord CE-only, headwise rank-16 adapter, and CPU table-axis searches are
all closed by completed negatives; RULER-13 is never a selection instrument; 8B
multi-seed is not scheduled by default.

## 4. Checked and found correct

Recording these so the next reader does not re-spend the effort:

- Both checkpoint weight hashes match **huggingface.co**, not just the mirror.
- Every frequency table rebuilds bit-exactly from the model config; both YaRN
  tables reproduce under a different transformers major version.
- No math-attention fallback is possible: the harness crashes instead
  (`probe_binary_64k_v3.log`), and `enable_gqa` is set correctly for Qwen's
  12Q/2KV and correctly absent for OLMo's 16Q/16KV.
- All four formal-matrix arms use byte-identical rows; zero duplicates; zero
  cross-cell reuse; 2Wiki/Qasper inputs identical at row level including the
  truncated rows.
- Routing reads only `prefill_tokens + max_new_tokens` and `L_native` — no task
  label, no reference, no external `L_target`.
- Both code-version seams in the formal matrix and the 2Wiki run are **proven
  inert** by bitwise output agreement across the boundary (120/120, 138/138,
  176/176).
- All 24 `raw_result_sha256` entries in the evidence receipt match the host files.
- The new control dispatch is correctly guarded for OLMo and dtype-matched (bf16
  on both paths) — I checked this specifically because it looked like a bug.
- Recomputed vs stored aggregates: **0 mismatches**, everywhere.
