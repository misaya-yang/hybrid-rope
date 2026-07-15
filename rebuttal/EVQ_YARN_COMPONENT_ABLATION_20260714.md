# EVQ vs official-YaRN component ablation: abundant (MHA) vs scarce (MLA) channels

Date: 2026-07-14
Status: single-seed, mechanistic, supporting. Prepared as conditional evidence for
the P1 "Official YaRN / native endpoint" concern in `rebuttal_playbook.md`.

## 1. Status and claim boundary

This note does not modify or promote any primary paper claim. It does **not**
resurrect official-YaRN complementarity; that interpretation remains
**WITHDRAWN** (playbook §1.3). The ablation instead does two honest things:

1. It **confirms the withdrawal** of official-YaRN complementarity in the
   abundant-channel regime and identifies its mechanism (frequency correction,
   not mscale, is what erases the substrate gap).
2. It provides **single-seed mechanistic support** for the direction of the
   scarce-channel Primary III result: when rotary channels are scarce and the
   extrapolation is aggressive, the same official frequency correction does not
   erase the EVQ substrate advantage.

Both findings are single-seed. Neither is a new primary claim. Deploy only if a
real reviewer/AC triggers the official-YaRN / native-endpoint question.

## 2. Question

The submitted Primary I "YaRN" is the repository fixed-index smooth-ramp scaler,
already relabeled/withdrawn in the playbook. The still-open question the P1 row
recorded as "unknown" was: under the **official** YaRN equations
(`jquesnelle/yarn@995db5b`: wavelength-driven correction range + attention
mscale), (a) does any EVQ substrate advantage survive, (b) is any erasure caused
by the frequency correction or by the mscale attention temperature, and (c) does
channel scarcity change the answer?

## 3. Operator decomposition

Official YaRN is split into four inference-only operators applied to the same
raw-trained checkpoints:

- `raw` — no scaling.
- `freq_only` — official correction-range frequency blend, `attention_scaling = 1`
  (mscale disabled).
- `mscale_only` — raw frequencies, `attention_scaling = mscale`.
- `full` — frequency blend + mscale (this equals the submitted "official/derived
  YaRN" cell).

Native substrates use the official native-grid equations; EVQ substrates use the
same equations through the virtual-coordinate derived-YaRN path and are labeled
"YaRN-derived on EVQ", never official native-grid YaRN. Operator source of truth:
`scripts/lib/rope/official_yarn.py`.

Substrate gap below is `NLL(Native) - NLL(EVQ)`; positive favors EVQ.

## 4. Result A — abundant channels (MHA, K=32)

Source: `data/curated/native_rope_evq_150m_s42_500m_20260713.json` (151.9M, 500M
tokens, seed 42) and its four-operator companion decomposition on the same
checkpoints.

- Under `full` official YaRN, Native and EVQ converge to almost the same PPL. At
  16K, EVQ 32.94 vs Native 33.16; the raw substrate gap of 0.177 NLL collapses to
  ~0.007 NLL.
- The four-operator decomposition attributes this collapse to the **frequency
  correction**: `freq_only` compresses the substrate gap to
  `0.0055 / 0.0124 / 0.0330` at 4K/8K/16K, while `mscale_only` leaves it intact at
  `0.091 / 0.177 / 0.206`.

Reading: with abundant channels, the inference-time frequency correction
re-allocates enough spectrum to reproduce most of what EVQ did at training time,
so the two substrates converge. mscale is a common-mode attention-temperature
gain that does not touch the substrate difference. This is exactly why official
YaRN shows no complementarity here — the frequency parts are substitutes, not
complements.

## 5. Result B — scarce channels (MLA, K=16)

Source: `results/mla_yarn_short_s42_20260714/` (432M MLA, `d_rope=32`, `K=16`,
seed 42, `L_train=512`, 100M tokens, base 500K). Report:
`docs/exp/2026-07-14_mla_k16_short_context_yarn_ablation.md`. Eval lengths
1K/2K/4K/8K correspond to YaRN scales 2/4/8/16.

Substrate gap `NLL(Native) - NLL(EVQ)` (positive favors EVQ):

| Operator | 1K (s2) | 2K (s4) | 4K (s8) | 8K (s16) |
| --- | ---: | ---: | ---: | ---: |
| raw | +0.2266 | +0.1803 | +0.1480 | +0.1332 |
| freq_only | -0.0193 | -0.0009 | **+0.0492** | **+0.2389** |
| mscale_only | +0.2384 | +0.1990 | +0.1901 | +0.1754 |
| full | -0.0211 | -0.0022 | +0.0259 | +0.1776 |

- At modest extrapolation (1K/2K = scale 2/4) the frequency correction **does**
  erase the gap (−0.02 to 0.00), as in MHA.
- At aggressive extrapolation (4K/8K = scale 8/16) the gap **reopens** to +0.049
  and +0.239 under `freq_only`; the 8K difference-in-differences is −0.106
  (operator helps EVQ more than Native).
- Best absolute 8K result is EVQ + full derived-YaRN at PPL 71.550 vs Native +
  full official YaRN at 85.455 (a ~16% PPL advantage that survives the full
  correction).

Reading: with only 16 rotary channels, the inference-time correction cannot
re-allocate enough spectrum to substitute for EVQ's training-time allocation at
large scales. This is the scarce-channel stress mechanism behind Primary III,
now isolated as a matched component ablation.

## 6. What is refuted (proactive disclosure)

The scarce-channel run pre-registered and then **refuted** two hopeful
predictions. We disclose them rather than let a reviewer find them:

- **EVQ + mscale is not a standalone recipe.** Without frequency correction,
  raw-substrate extrapolation is broken, so mscale alone is catastrophic: EVQ
  `mscale_only` 8K PPL is 519.139 vs EVQ `full` 71.550. mscale is a useful
  *add-on after* frequency correction (`freq_only` 92.9 → `full` 71.6), not a
  substitute for it.
- **EVQ raw does not beat Native full.** EVQ `raw` beats Native `raw` at every
  length but loses to Native `full` (8K: 412.0 vs 85.5). EVQ is not a replacement
  for inference-time range scaling.

Therefore the only surviving statement is narrow: **under matched full official
YaRN, EVQ retains a measurable advantage in the scarce-channel + aggressive-
extrapolation regime.** It is not "EVQ replaces YaRN," not "official-YaRN
complementarity," and not "EVQ needs no inference scaling."

## 7. Boundaries

- Single-seed, mechanistic; does not replace the three-seed Primary III result.
- The MLA run trains at `L_train=512` (short) on 100M tokens; the advantage only
  appears at scale 8/16 extrapolation, and the frequency correction still
  substitutes for EVQ at scale 2/4.
- `tau=1.414` is kept at the historical Primary III `d_eff=128` convention and is
  **not** re-derived for the 512-token training length (by the paper's own
  `tau*=d_eff/sqrt(L)` this under-warps EVQ, so the observed advantage is likely
  conservative — but the mismatch is a methodological loose end a reviewer can
  raise; a `tau=5.66` re-derived run would close it).
- PPL-only. The 8K Passkey teacher-forced NLL-gap is ≈0 for both substrates, so
  this run makes **no** retrieval or task-capability claim; at the lengths where
  Passkey is non-trivial (1K/2K), Native `full` is comparable-to-better than EVQ
  `full`.
- Native uses official native-grid equations; EVQ uses virtual-coordinate
  derived-YaRN (labeled). This is not a canonical operator published by the YaRN
  authors and must not be called official native-grid YaRN on the EVQ substrate.

## 8. Disposition (playbook alignment)

- Trigger: real reviewer/AC raises official-YaRN, native-endpoint, or "does EVQ
  survive a faithful range scaler" (playbook P1 row; §9 branch "DAPE / YaRN / Geo
  fidelity" and "novelty / significance").
- Use to: (a) confirm the WITHDRAWN official-YaRN complementarity with its
  mechanism; (b) give single-seed mechanistic support for the scarce-channel
  Primary III direction; (c) proactively disclose the P2/P3 refutations.
- Do **not**: present as a new primary claim, upgrade Primary III, restore
  complementarity, or generalize beyond the audited single-seed short-training
  protocol. Subject to the §8 experiment gate.

## 9. Provenance

- MLA raw + attribution: `results/mla_yarn_short_s42_20260714/evaluation/raw_results.json`,
  `.../analysis.json`; report `docs/exp/2026-07-14_mla_k16_short_context_yarn_ablation.md`;
  checkpoints on the experiment machine (SHA256 in the report).
- MHA six-cell: `data/curated/native_rope_evq_150m_s42_500m_20260713.json`.
- MHA four-operator (`freq_only`/`mscale_only`) decomposition: attach the curated
  artifact + SHA256 before any reviewer-facing use; numbers above are the
  reported run on the same 151.9M seed-42 checkpoints.
- Operator implementation: `scripts/lib/rope/official_yarn.py` (pinned
  `jquesnelle/yarn@995db5b`).
