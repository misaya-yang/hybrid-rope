# OLMo-2 Fresh All-Long-Gap 8K NIAH n=100

Date: 2026-07-26  
Run label: `s20260727`  
Status: `RAW_BACKED_DUAL_COPY_FROZEN_CONFIRMATORY_FOLLOWUP`

Concern mapping: `R27bE.2`, `R27bE.5`, `AC.2`

## Experiment contract

1. **Reviewer/AC concern.** Determine whether the mature-model EVQ-LoRA
   result survives a larger 8K autoregressive evaluation in which every
   source-to-generation distance is outside the support observed during 4K
   routing training.
2. **Existing evidence.** On the original 8K n=100 set, Native and two EVQ
   training seeds score `0/100`, `69/100`, and `67/100` strict exact.
   A post-hoc split shows `0/50`, `21/50`, and `19/50` on the rows beyond the
   maximum training gap.
3. **Smallest missing evidence.** A fresh n=100 set containing only
   beyond-training-gap rows, disjoint from routing train, calibration, and the
   original n=100 evaluation.
4. **Smallest executable plan.** Pure inference with the three already frozen
   final adapters. Use the official RULER `niah_single_1` generator, 8K only,
   greedy autoregressive decoding, and strict first-number exact.
5. **Stop condition.** Stop after Native and the two existing EVQ training
   seeds complete. Do not train or introduce another mechanism.

No optimizer step was taken in this follow-up.

## Registered data constraint

Across the 1,024 frozen 4K routing rows, the largest source-answer to answer
start gap is 3,933 tokens. The new screen generated 512 official RULER
candidate rows and retained the first 100 satisfying

\[
g_i^{\mathrm{eval}}
=L_i^{\mathrm{input}}-p_i^{\mathrm{source\ answer}}>3933.
\]

The selected rows have gaps from 3,959 to 8,013 tokens. Selection rejected 83
earlier candidates for failing the gap condition; there were zero
query/source/value/answer overlaps, selected-identity duplicates, or
prompt-length failures before completion.

The test set is disjoint from:

- all 1,024 routing-training rows;
- all 128 routing-calibration rows;
- all 100 rows in the earlier 8K screen.

For each comparison, query, source key, source value, and answer overlap is
exactly zero. This is still the same official `niah_single_1` task family; it
is held-out identity and distance evidence, not unseen-task evidence.

## Frozen method identity

| Arm | Training seed | Adapter SHA-256 | Active `inv_freq` SHA-256 |
| --- | ---: | --- | --- |
| Native-LoRA | 20260725 | `6570ab94aec68431dd4e261eb3ef342ef72253357aa65a0d37b0a018df2f3f8d` | `dde15c31724177356ae954d6e11fb337e6fccef56e4520a905cac3f0d9885b34` |
| EVQ-LoRA A | 20260725 | `95ceeb70117c73233915760a9756b9b2a98416ec188b125054da8ced75cad16a` | `917a52426b4ac986545c8ec73b115daae3c6515d6b9047f09d30c972ea1a4607` |
| EVQ-LoRA B | 20260726 | `fdf6dfc249cb216c3effe22a2ee96fe439a5aff9a11a99007f022c5fe7c3b085` | `917a52426b4ac986545c8ec73b115daae3c6515d6b9047f09d30c972ea1a4607` |

The parent evidence archive freezes both matched final adapters, the second
EVQ adapter, optimizer configuration, seeds, raw training/calibration rows,
deterministically reconstructed per-step sample order, original 4K/8K/16K
predictions, NLL/rank/exact results, and file-level hashes. The completed
training did not persist Adam moments; only the actual optimizer protocol is
claimed as frozen.

## Primary result

| Arm | Strict first-number exact | Official substring | Wilson 95% interval |
| --- | ---: | ---: | ---: |
| Native-LoRA, seed 20260725 | **0/100** | 0/100 | `[0.00%, 3.70%]` |
| EVQ-LoRA, seed 20260725 | **49/100** | 49/100 | `[39.42%, 58.65%]` |
| EVQ-LoRA, seed 20260726 | **48/100** | 48/100 | `[38.46%, 57.68%]` |

Native emits no number on all 100 rows. For EVQ seed 20260725, 49 rows are
strictly correct, 50 emit a wrong first number, and one emits no number. For
seed 20260726, 48 are strictly correct and 52 emit a wrong first number.
Strict and official scores are identical in this evaluation.

The paired Native-versus-EVQ discordant counts are `0/49` and `0/48`.
Two-sided exact McNemar values are \(3.55\times10^{-15}\) and
\(7.11\times10^{-15}\), respectively. These quantify this fixed paired test
set; the two EVQ rows must not be pooled as 97 independent successes because
the same examples and related trained models are used.

## Training-seed stability

The two EVQ adapters agree on correctness for 81/100 rows:

| Both correct | Seed 20260725 only | Seed 20260726 only | Both wrong |
| ---: | ---: | ---: | ---: |
| 39 | 10 | 9 | 42 |

The one-point aggregate difference is not detectable by paired exact McNemar
(`p=1.0`). The replicated result is the large positive gap relative to Native
on the same all-long-gap rows, not a claim that either EVQ seed is superior.

## Distance description

The all-long-gap set is not uniform in distance:

| Gap band | Rows | EVQ seed 20260725 | EVQ seed 20260726 |
| --- | ---: | ---: | ---: |
| 3,934–4,095 | 3 | 3 | 2 |
| 4,096–5,119 | 33 | 18 | 17 |
| 5,120–6,143 | 28 | 12 | 13 |
| 6,144–7,167 | 17 | 7 | 8 |
| at least 7,168 | 19 | 9 | 8 |

The prior beyond-support subset had a larger mean gap (`6,155` versus
`5,781` tokens) and a larger fraction at or above 6,144 tokens (`58%` versus
`36%`). Its EVQ rates were 42% and 38%, compared with 49% and 48% here; the
Wilson intervals overlap. The defensible replicated fact is nonzero,
large Native-relative retrieval beyond all training gaps. The exact success
percentage remains sample- and distance-distribution dependent.

## Metric boundary

This follow-up measures greedy autoregressive generation and does not compute
teacher-forced NLL or token rank. The frozen parent experiment owns the
matched 4K/8K/16K natural-text NLL, 128-pair routing NLL/rank, and original
4K/8K/16K per-example generation outputs. Do not relabel the present exact
scores as NLL/rank evidence, full RULER, or general downstream transfer.

## Frozen evidence and hashes

Core inputs and outputs:

| Artifact | SHA-256 |
| --- | --- |
| Candidate pool | `b0fab7b87c3b3aaf367d69b36aafc089f85f8eb9b0420c62d9088c3e9beccf0b` |
| Selected 8K n=100 data | `2fa503f8d2b4007db17b771afe669be18db4486cb84ec38657c55cf75c9c2b87` |
| Data manifest | `9987db15abd90c66477f612e5ebd90a6b50dada5d1cd1076bacedbec4dbb1925` |
| READY receipt | `383586b59b2dea17a33d0031a793ce2d7ddef6b73fdef4eb374950985eb16674` |
| Native results / examples | `253624361ee544ce54f1d28271a74c67f2609ff4befd005fff66441f6541616c` / `28f7c47ae2b19401a65e56dda90a9d0acb05fac7361d2486c8b1b90f20eb3709` |
| EVQ-20260725 results / examples | `5aa214d9c41da821328a2e1daab357904c0cb4def6390c6c4d044542122c3f28` / `18ecdacecfabf58bcc51b8e59cb3b04afdaee5bb78d2d8116bf9147d0808b37f` |
| EVQ-20260726 results / examples | `9efc47600738c62c8aa0bbe916a0a701ef8357162c68ddd69399ad779840dc5a` / `3056ff3451e8ba81c608d4d721f689b65d1db0ec6b7981c20856b7b575e0d014` |
| Analysis JSON | `6ea79704807bc7cff6165e9efad58afc818a07785238b0c42335cf22522a6b41` |
| Frozen inventory | `fa49290be2675ce07eb7bc698cf7a4e521d964dba55c2e67c1fa5811f952463b` |
| Delta archive, local and remote | `e91e67ce1bc316a562037930e7a375ee4eb711b72ea5961da33b74afab06d0de` |
| Parent adapter/training archive, local and remote | `a48505d5c51be4f60d3c87bb118924b935d17cda26eb31416acf4b024d07d383` |

The delta archive contains all 300 predictions, all three formal result JSONs,
the selected data and candidate pool, manifest, READY receipt, analysis JSON,
exact code/test snapshots, parent optimizer/order/overlap receipts, and an
immutable inventory. The large adapter tensors remain in the separately
frozen parent archive and are referenced by verified byte hash.

Local archive:
`/tmp/olmo2_maturity_evidence_20260725/frozen/instruct_fresh_all_long_gap_n100_s20260727_v1.tar.gz`.

Remote archive:
`/root/autodl-tmp/olmo2_1b_longalign_assets/evidence_frozen/instruct_fresh_all_long_gap_n100_s20260727_v1.tar.gz`.

Local verification passed 29 focused tests. The RTX 5090 returned to
`0%` utilization and `0 MiB` allocated after the three inference arms.

## Reviewer-safe conclusion

> On a fresh set of 100 official 8K RULER `niah_single_1` examples, every
> source-to-generation gap exceeds all gaps used for 4K routing adaptation.
> Matched Native-LoRA scores 0/100 strict autoregressive exact, whereas two
> independently trained EVQ-LoRA adapters score 49/100 and 48/100. Training,
> calibration, prior-test, and fresh-test query/value identities are disjoint.
> This supports reproducible same-task capability conversion beyond observed
> training-distance support in a mature 1B Instruct model. It does not
> establish full RULER, unseen-task transfer, pure Cosh attribution, or solved
> 2x retrieval.
