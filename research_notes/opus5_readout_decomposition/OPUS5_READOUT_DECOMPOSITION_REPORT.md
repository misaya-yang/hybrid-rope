# Layer-Resolved Causal Readout Decomposition — EVQ vs Geo, Llama-3-8B, 16K

Date: 2026-08-06
Scope: closes link 4 of `research_notes/FABLE5_EVQ_MECHANISM_AUDIT.md` §4 using only data already on this machine.
Status: **mechanism diagnostic, single seed, 5 matched cases** — inherits `measurement_label: "oracle-diagnostic"` and `single_seed_supporting: true` from both source manifests. Not promotable to a paper-level claim without expansion.

No model was loaded, no GPU used, nothing downloaded, no training. Nothing outside this directory was modified.

Reproduce with `./run_all.sh` (CPU, ~1 minute, reads ~540 MB of frozen tensors).

---

## 0. One-paragraph answer

The gold block's causal signal becomes visible in the output basis at **layers 22–24**, roughly **5–10 layers after** the QK addressing advantage is established (layers 11–21). Later layers **neither amplify nor destroy it**: the gold-specific delta is retained at a median 58% of its peak all the way to the true final logits. The failure is therefore not weak propagation and not late entry. At the first answer token the gold block is worth a median **0.78 logits against a 10.09-logit deficit — it closes 6.9% of what winning requires**, and the tokens ahead of it are essentially untouched by the ablation. Those tokens are identifiable: the top-1 final-layer competitor is exactly the first token the model actually generates, and the EVQ arm generates **one identical, prompt-independent haystack sentence for all five cases** ("The archive contains ordinary records about weather, roads, gardens, and public meetings"), unchanged under chat wrapping. The remote evidence is real, specific, causally used and retained; it loses to a generic continuation the model was going to emit regardless of the needle. **The bottleneck is that the model never enters answer mode at the first answer token, not that the retrieved evidence decays.**

---

## 1. Control gate

`controls.py` — all hard controls pass (`controls.json`, `gate_passed: true`). Every number below is downstream of this gate.

| Control | Result |
| --- | --- |
| C1 record integrity | 11/11 locally present records byte-match manifest SHA-256 and size |
| C2 tensor contract | all `[3, 32, 128256]` bf16, finite, `layer_indices == 0..31` |
| C3 gold ids across arms | identical for all 5 matched prompts |
| C4 final-lens parity | `final_logit_parity_max_abs == 0.0` in every record |
| C5 pairing | 5 prompt-SHA-256-matched pairs, depths 10/10/25/25/50 |
| C6 provenance | shared `passkey`/`code`/`script` SHA; distinct adapters (EVQ `8ea04234…`, Geo `0e7efa6e…`) |
| C7 phase0 cross-match | 5/5 matched prompts resolve to phase0 16K entries **by SHA, not by position** |

**Declared limitation (warning, not failure).** Both manifests declare 10 records and report `status: complete`; the remote run was complete but this machine holds only 11 of 20 files. Depths **75% and 90% are entirely absent**, and one EVQ depth-50 record is missing. The analysis therefore covers the **first half of the context only** (depths 10–50). The single Geo-only record (`aab0eff7`, depth 50) is used as an extra unpaired control and excluded from paired statistics.

### 1.1 Independent validation of the whole pipeline

`validate.py`. The `rank_16k_v1` run computed the same output-level quantities online from the live model on the *same adapter SHAs*. Re-deriving them from the stored bf16 lens tensors reproduces that run:

- **gold rank at layer 31 — exact integer match on all 11 records** (e.g. 22569, 938, 1862, 3035, 543 for EVQ; 64300, 36721, 42164, 50790, 15312, 18088 for Geo)
- **answer NLL — reproduced to < 5×10⁻⁴** (bf16 storage rounding)

This confirms pairing, gold-id gathers, layer ordering, sign convention and the layer-31 identity end to end. Layer 31 is the model's true output, not a lens estimate (`run_readout_trace` overwrites it with `final_logits`); layers 0–30 are lens estimates.

---

## 2. Method

Sign convention: `δ_ℓ = z_ℓ^full − z_ℓ^gold-drop`. Positive on the gold token = the gold block supports it.

| Metric | Definition | Basis behaviour |
| --- | --- | --- |
| `delta_gold_centred` δ̃_ℓ | `δ_ℓ(gold) − median_v δ_ℓ(v)` | removes the additive shift; magnitude still lens-dependent |
| `delta_gold_z` | `δ̃_ℓ / (1.4826·MAD_v δ_ℓ)` | scale-free within a layer |
| `causal_delta_rank` | `1 + count(δ_ℓ > δ_ℓ(gold))` — registered contract rule | invariant to shift and positive scaling |
| `kl_full_abl` | `KL(softmax z^full ‖ softmax z^drop)` | shift-invariant |
| `l2_centred` | `‖δ_ℓ − median‖₂` | lens-dependent, reported for completeness |
| `gap_to_best_competitor` | `max_{v≠gold} z(v) − z(gold)`, dense and ablated | output-space, exact at ℓ=31 |
| `entropy`, `top1_prob`, `gold_prob` | distribution shape | needed to separate sharpening from attenuation |

Entry layer is reported under **three** rules, because they disagree in an informative way: the audit's history rule (`δ̃_ℓ > 3·MAD` of layers `0..ℓ−1`, min history 4), a history-free vocabulary rule (`delta_gold_z > 3`), and a 10%-of-peak onset rule.

All 5 cases × 3 answer positions × 32 layers are reported individually in `per_layer_metrics.csv` (1056 rows). No result below rests on an average over the five cases alone.

---

## 3. Findings

### Q1 — Where does the gold-block causal signal first become visible in the output space?

**Layers 22–24**, at answer position 0.

| case | depth | audit rule | vocab rule | 10%-of-peak | peak layer |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6cd995c6 | 10 | 16 | 24 | 16 | 24 |
| 06800fde | 10 | 5 | 29 | 16 | 28 |
| 933bec9a | 25 | 22 | 26 | 22 | 28 |
| c4b929f0 | 25 | 17 | 23 | 17 | 24 |
| 8c28ff35 | 50 | 19 | 22 | 19 | 23 |

The basis-robust `causal_delta_rank` corroborates: it falls below 1000 at layers 19–28 and below 100 at layers 22–31, from a baseline of 10⁴–10⁵ (fig 3, left). The mean curve is flat within noise through layer 21 and rises 0.26 → 0.61 → 0.87 across layers 22, 23, 24 (fig 1, fig 2).

**Boundary, stated explicitly.** This is the first layer at which the effect is *decodable in the final-norm + unembedding basis*, which is **not** the layer at which the information enters the residual stream. The audit's caveat applies in full; the rank statistic removes shift and scale sensitivity but not the choice of readout basis. The audit's own history rule fires earlier (median 17) at magnitudes that do not clear the vocabulary noise floor at those layers, which is exactly the failure mode a plain logit lens is prone to — I report it but do not build on it.

### Q2 — Do later layers amplify, maintain or attenuate it?

**They maintain it.** Retention `R = δ̃₃₁ / max_ℓ δ̃_ℓ` has median **0.58** (range 0.42–0.88) at position 0. No amplification; mild, not catastrophic, attenuation.

Two things that look like decay are not:

1. **KL collapses ~8.6× from its layer-28 peak to layer 31 — this is sharpening, not attenuation.** Over the same layers entropy falls 9.5 → 3.5 nats and top-1 probability rises to 0.57. A fixed ~0.6-logit perturbation on a token 10 logits behind simply stops moving a distribution that has concentrated on a competitor. The gold-specific δ̃ is flat over exactly those layers (0.67 at L28 → 0.61 at L31). Fig 5.
2. **The gold token's absolute readout rank degrades ~22× from its best late layer (median L27) to layer 31** — but **Geo degrades the same way (median 8.6×, range 1.9–104×)**. This is a property of the shared base model's readout path at the first answer position, not something EVQ causes or fails at. It is also position-specific: at answer positions 1 and 2 the degradation factor is a flat **1.0** (fig 3, right).

### Q3 — Is the failure weak remote evidence, or being out-competed?

**Both are true and the decomposition separates them quantitatively — the competition term dominates.** At answer position 0 (`final_layer_competition.csv`, fig 4):

| case | depth | deficit without block | closed by block | residual deficit | closure |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6cd995c6 | 10 | 14.01 | 0.41 | 13.60 | 2.9% |
| 06800fde | 10 | 10.69 | 1.06 | 9.63 | 9.9% |
| 933bec9a | 25 | 11.13 | 1.03 | 10.09 | 9.3% |
| c4b929f0 | 25 | 11.41 | 0.78 | 10.63 | 6.9% |
| 8c28ff35 | 50 | 9.47 | 0.28 | 9.19 | 3.0% |

Median: the gold block supplies **0.78 of the 10.87 logits** required — **6.9%**. Roughly **9–33 more gold blocks' worth** of evidence would be needed to win.

The competitors are not being helped by the gold block: the median `|δ̃|` over the top-32 competitors is 0.15 versus 0.78 for gold. They are supported by something the ablation does not touch.

**What the competitors actually are.** No Llama-3 tokenizer exists on this machine, so I identified them from the run's own recorded generations instead (`validate.py`, V2):

- The **top-1 final-layer competitor equals the first freely generated token in all 11 records** (EVQ: id 220; Geo: id 791).
- **24 of the top-32 competitors are shared by all five cases** (75%), and **none of them is a gold answer token of any case** (`competitor_structure.json`).
- The EVQ arm emits **one single distinct prediction across all five cases**: *"The archive contains ordinary records about weather, roads, gardens, and public meetings"* — the haystack filler, independent of needle content and depth. Geo emits degenerate output (`"The\n://://://…"`).
- Unchanged under the chat-wrapped control run.

So the winning tokens are neither distractor digits nor local copies of nearby content. They are the model's default continuation of the document. **The model is not losing a retrieval competition; it is not attempting retrieval.**

Corroborating this: when teacher-forcing imposes the answer format, the block's contribution triples — median closure 6.9% (pos 0) → 18.7% (pos 1) → 23.3% (pos 2) — and the late-layer demotion vanishes (fig 4, right). The leading reading is that the bottleneck is entering the answer state at token 1. A weaker alternative I cannot exclude with n=5 is that continuing a partially emitted number is simply an easier prediction than starting one; both readings place the bottleneck at the first answer token.

This also explains the previously puzzling **oracle near-null**: forcing gold-block inclusion is worth 0.034 NLL versus score-sparse (9.0244 vs 9.0586, verified in `oracle_include_16k_v2/summary.json`), with exact match 0.00 in every mode. Attention routing is already delivering the block (EVQ sparse/dense answer-mass ratio 1.03), and the block's full causal worth at position 0 is ~1 logit against a ~10-logit deficit. Better routing cannot fix a readout that is answering a different question.

### Q4 — Does the layer-wise readout change track the per-layer QK metrics?

**Not instantaneously — cumulatively, with a 5–10 layer lag.**

| correlation across the 32 layers | Spearman |
| --- | ---: |
| readout δ̃_ℓ vs per-layer QK hit@16 advantage (all heads) | **−0.06** |
| readout δ̃_ℓ vs per-layer QK hit@16 advantage (frozen 32 heads) | −0.33 |
| readout δ̃_ℓ vs per-layer QK answer-mass advantage | +0.31 |
| readout δ̃_ℓ vs **cumulative** QK hit@16 advantage | **+0.63** |
| readout δ̃_ℓ vs cumulative QK answer-mass advantage | +0.66 |
| KL vs **cumulative** QK hit@16 advantage | **+0.90** |

The EVQ−Geo QK advantage is concentrated in layers 11–21 (summed advantage 2.75) versus 1.37 over layers 22–31, while the readout onset is at layers 22–24 (fig 2). This is mechanistically the expected shape: causal evidence readable at layer ℓ reflects attention that has *already happened* at layers ≤ ℓ, so the cumulative statistic is the correct comparand and the near-zero instantaneous correlation is not evidence of a missing link.

The alignment is measured at answer position 0 specifically, because that is the position the phase0 probe's query contract measures (`last_prompt_token_predicting_first_answer_token`) — the two instruments are compared where they measure the same query.

The 32K phase0 slice shows the decay side: summed QK advantage falls from 2.75 to 0.32 over layers 11–21. There are no 32K readout traces, so the corresponding readout collapse is *not* measured here.

### Q5 — Is Geo a stable near-zero negative control at every layer?

**At the generation-critical position, yes. Across all positions, not quite — and this is worth recording.**

| answer position | Geo peak \|δ̃\| median | Geo peak \|δ̃\| max | EVQ peak median | ratio |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.078 | **0.168** | 1.039 | 13.3× |
| 1 | 0.125 | 0.214 | 1.617 | 12.9× |
| 2 | 0.152 | **0.561** | 2.266 | 14.9× |

- **Position 0: a clean near-zero control at all 32 layers, all 5 cases.** Max 0.168 logits; final-layer closure fractions are ±1%; `causal_delta_rank` never localises — it stays above 1000 at every layer in 5 of the 6 Geo records, and the single exception is a rank of 651 at **layer 0**, before any attention has accumulated. EVQ reaches rank 1 on the same statistic.
- **Positions 1–2: not uniformly near-zero.** Two depth-10 cases show structured positive bumps of 0.43–0.56 logits over layers 25–30 and around layer 17. These are 5–13× below EVQ but they are not noise — the vocabulary MAD at those rows is ordinary (~0.04), so this is not a normalisation artefact.
- The unpaired sixth Geo record (`aab0eff7`, depth 50) behaves identically to the paired Geo cases, which strengthens the control.

The most likely source of the position-1/2 leak is the confound flagged at the outset: `gold_drop_all` removes 128 keys and the softmax renormalises over what remains, so *every* head's output shifts slightly even with no evidence use. The audit's control criterion ("δ̃ ≈ 0 at every layer") is met where the analysis's conclusions are drawn (position 0) and is met only approximately elsewhere. I did not treat this as a gate failure, but it caps how strongly the EVQ/Geo contrast can be called "binary" at later answer positions.

---

## 4. Failure mode best supported by the evidence

Against the audit's pre-specified decision rules: entry at L22–24 rules out **H-late-entry** (which required ℓ* ≥ 28); retention R = 0.58 ≥ 0.5 rules out **H-decay**. The evidence selects **H-competition**, and the decomposition sharpens it beyond the original formulation:

> **The 8B arm fails at the first answer token because the readout is committed to a content-independent continuation of the haystack, not because the remote evidence is absent, late, or degraded.** The gold block's causal contribution is real (deletion costs +1.51 NLL), specific (gold reaches rank 1–100 in the causal-delta vocabulary), and retained to the true output logits (R ≈ 0.58) — but it is worth ~7% of the logit deficit against a competitor set that the ablation barely moves and that the model emits verbatim regardless of the needle. Attention routing is not the binding constraint: perfect routing (oracle) is worth +0.034 NLL because the quantity it perfects is small relative to the gap.

This reframes "readout bottleneck" from *evidence integration* to *task-state entry*, and it accounts for three otherwise separate repository observations: the oracle near-null (T2), the repo-wide TF/NLL ↔ AR-generation dissociation (confounder C5 — teacher-forcing supplies the answer state that free generation never enters), and why OLMo routing-conversion training converts capability at all while a frequency swap alone does not (C1) — the conversion data teaches the answer state, and EVQ supplies a substrate on which the remote evidence is already available once that state is entered.

---

## 5. What now holds, and what still does not

**Holds (within 5 matched cases, single seed, adapted 8B, 16K, depths 10–50):**
1. Link 4 of the chain is no longer unmeasured. Causal gold-block evidence is output-visible from layers 22–24 and retained to the final logits at ~58% of peak.
2. Later layers do not destroy the remote evidence. Claims that "the signal decays across depth" are not supported.
3. The KL collapse in the last three layers is distribution sharpening, not attenuation — established by the entropy/top-1-probability trajectory, not assumed.
4. The first-token deficit is ~10 logits and the gold block closes ~7% of it; the competing tokens are causally independent of the gold block.
5. The competitors are the model's default haystack continuation, identified without a tokenizer via exact agreement between the top-1 final competitor and the first generated token in 11/11 records.
6. QK addressing (L11–21) precedes readout visibility (L22–24); the correct comparand is the cumulative QK advantage (ρ = +0.63; KL ρ = +0.90), not the per-layer one (ρ = −0.06).
7. The Geo arm is a valid near-zero control at answer position 0 at every layer.
8. The pipeline reproduces an independent live-model run exactly (rank) and to bf16 rounding (NLL).

**Does not hold / still unsupported:**
1. **"Layer 22–24 is where remote information enters the residual stream."** Not established. It is where it becomes decodable in the final-norm/unembedding basis. Basis sensitivity is mitigated by the rank statistics, not eliminated.
2. **"gold-drop isolates the A@V contribution of the gold block."** It does not. Removing 128 keys renormalises the softmax across all heads; the position-1/2 Geo leak (up to 0.56 logits) is the visible footprint of that. All numbers here are deletion effects, not vector decompositions.
3. **Anything about depths 75–90%, or about 32K.** Those records are not on this machine; the 32K QK decay is cited from phase0 only, with no readout counterpart.
4. **Statistical generality.** n = 5 matched cases, 1 LoRA seed, 1 model, 1 task family. The per-case spread is large where it matters (closure 2.9–9.9%; retention 0.42–0.88; one case, 6cd995c6, peaks at L24 and then collapses, unlike the other four). No significance testing is appropriate at this n and none is reported.
5. **That the same mechanism explains the OLMo/750M conversions.** Untested here — those are different models, and the arms that actually generate were never traced. This is the whole point of §6.
6. **Confounder C6 (Q/K norm / attention temperature) is untouched.** It is not recoverable from saved logit traces.
7. **The "never enters answer mode" reading versus "continuing a number is easier than starting one."** Both fit the position-0 vs position-1/2 contrast at n = 5.

---

## 6. Minimal next GPU probe — on the OLMo matched-capability arms

The single decisive missing cell: every mechanism measurement in this repository is on an 8B arm that **generates 0%**, while the arms that **do** generate (OLMo-2-1B matched routing conversion, Native 0/100 vs EVQ 69/100 at 8K) have never been traced. This analysis makes that gap sharp and gives it a falsifiable prediction.

**Probe.** Paired dense / `gold_drop_all` readout traces at the first answer token on the frozen OLMo-2-1B matched arms. Inference only, no training, no new data.

- **Arms:** Native-LoRA (0/100) and EVQ-LoRA seed 20260725 (69/100); adapter SHA-256s recorded in `OLMO2_1B_4K_ONLY_ROUTING_CONVERSION_20260726.md`. Add the EVQ arm on a held-out task where it scores 0% (UUID retrieval or variable tracking) to dissociate substrate from answer-state.
- **Data:** the same 8K NIAH set, n ≥ 30 (n = 100 preferred — this analysis is n = 5 and the per-case spread is wide).
- **Reuse:** `run_readout_trace` in `experiments/lora_evq_v2/eval_sparse_conversion.py` already implements exactly this contract; only the arm loader changes.
- **Storage:** do **not** save full `[pos, layer, vocab]` tensors. Per position/layer, store the gold logit, the top-64 ids + logits, and the vocabulary median/MAD of `δ`. That supports every metric in this report except full-vocabulary KL, at ~KB per record instead of ~50 MB.
- **Primary readout:** the position-0 **closure fraction** and `causal_delta_rank`, plus the free-running generation for each case.
- **Bundled C6 control (retires a standing confounder for free):** per-arm Q/K norms and attention entropy at the answer query on the same cases.

**Pre-registered prediction under the failure mode above.** In the converting EVQ arm the position-0 closure fraction should be **near 1.0** (the gold block wins the first-token competition) with an entry layer at a similar relative depth; in the Native arm it should be ≈ 0 with the gold block contributing nothing; the EVQ arm on the held-out 0% task should show EVQ-like entry and retention but 8B-like closure (~0.1).

**Falsifier.** If the converting EVQ arm generates correctly while its position-0 closure fraction is also ~0.07, the logit-gap framing in §4 is wrong and the "readout competition" reading must be discarded rather than patched.

---

## Appendix — artifacts

| File | Contents |
| --- | --- |
| `run_all.sh` | end-to-end reproduction (gate → decomposition → validation → figures) |
| `controls.py` / `controls.json` | control gate; exits non-zero on failure, blocking `decompose.py` |
| `decompose.py` | metrics M1–M5 |
| `validate.py` / `validation.json` | parity against `rank_16k_v1`; competitor identification |
| `figures.py` / `figures/*.png` | figs 1–5, every case drawn individually |
| `per_layer_metrics.csv` | 1056 rows — 11 records × 3 positions × 32 layers, 23 metric columns |
| `per_case_summary.csv` | entry layers (3 rules), peak, retention, final ranks/gaps |
| `readout_trajectory.csv` | best-vs-final rank, probability and KL peaks, entropy |
| `final_layer_competition.csv` | gap decomposition + top-32 competitor ids and deltas |
| `phase0_layer_profile.csv` | 960 rows — per-layer QK metrics, both arms, 16K and 32K |
| `qk_readout_alignment.json` | correlations + all layer curves |
| `competitor_structure.json` | cross-case competitor overlap |
| `summary.json` | headline aggregates |

Sources consumed (read-only): `results/readout_conversion_s42_20260715/raw/causal_{evq_cosh,native_geo}/`, `results/lora_sparse_conversion_s42_20260714/{phase0_evq,phase0_geo,phase0_summary}.json`, `results/lora_sparse_conversion_s42_20260714/server_26521/{rank_16k_v1,chatwrap_dense_16k_v1}/`.

Environment: CPU only; torch 2.9.1 / numpy 2.2.6 / scipy 1.17.0 / matplotlib 3.10.8 in the `ai_gateway` conda env (the system `python3` has no torch, as the audit noted).
