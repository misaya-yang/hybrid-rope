# K32 normalized-index full-RULER confirmation (2026-09-01)

## Decision

**Status: `COMPLETE / CLEAR_ADVANCE`.**

On a new seed and the complete 13-task RULER suite, one frozen
normalized-index s2 table passes the 32K Native capability gate and improves
64K macro over both Native and official-equation YaRN. The 64K paired advantage
over YaRN is `+.060897`, with 95% interval `[.027627,.095835]`.

This confirms the earlier core-4 engineering signal on a broader untouched
task matrix. It makes normalized-index the current static-table candidate for
work-machine natural-likelihood and natural-QA confirmation. It does **not**
establish uniform per-task dominance, a physical law, K causality, or SOTA.

Owners:

- [Preregistration](../preflights/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_PREFLIGHT_20260901.md)
- [Hash-bound receipt](../evidence/K32_NORMALIZED_INDEX_FULL13_CONFIRMATION_RECEIPT_20260901.json)
- [Reproducer](../../../../scripts/analysis/summarize_k32_full13_confirmation.py)

## Frozen protocol

- exact Qwen2.5-0.5B-Instruct K32 checkpoint and Native table;
- seed `202609027`, 20 rows per task and length;
- all 13 official synthetic tasks at 32768 and 65536 tokens;
- exactly three arms fixed before data generation: Native, frozen
  normalized-index s2, and installed-HF official-equation YaRN-s2;
- one static table from prefill through generation, standard KV cache, no
  runtime routing, model update, table refit, gain search, or fourth arm;
- 520 generations per arm, 1,560 terminal generations total.

The data manifest SHA-256 is
`808eb4c2935bdcdc27b94b44f9e417dd4280c75cda01196ad187576d29f1f93e`.
All compared arms bind the same checkpoint, tokenizer, task-cell hashes,
references, scorer, prompt lengths, generation budgets, runner, Native tensor,
and complete decoded/token/EOS rows.

The official generator initially stopped before evaluation because its NLTK
search path omitted an already-present `punkt_tab` asset. After adding the
asset path, data generation was CPU-parallelized only across disjoint
task-length cells. Seven cells completed by both serial and parallel paths
match SHA-256 exactly (`7/7`). The failed partial directories are retained
externally; this was an input-preparation event, not a method arm.

## Length curve and primary comparisons

| Frozen profile | 32K macro | 64K macro | 32K retention vs Native |
| --- | ---: | ---: | ---: |
| Native | `.547821` | `.220513` | `1.000000` |
| normalized-index | `.559167` | `.514551` | `1.020711` |
| official YaRN-s2 | `.559423` | `.453654` | `1.021181` |

At 32K, index-minus-Native is `+.011346` with paired 95% interval
`[-.021987,.044103]`; index-minus-YaRN is effectively zero
(`-.000256 [-.032630,.032179]`). Thus the Native gate passes without a claim
of short-context improvement or index/YaRN separation.

At 64K:

| Contrast | Delta | Paired 95% CI |
| --- | ---: | --- |
| index − Native | `+.294038` | `[.250192,.338271]` |
| index − YaRN | `+.060897` | `[.027627,.095835]` |

Both intervals are wholly positive. Under the preregistered rule this is
`CLEAR_ADVANCE`, not merely non-inferiority.

The bootstrap uses 10,000 replicates, seed `202609028`, resampling rows within
each of the 13 fixed task strata and averaging task means equally. It estimates
row uncertainty conditional on this checkpoint, task suite, decoder and
scorer; it is not checkpoint or training-seed uncertainty.

## Complete per-task results

| Task | Native 32K | Index 32K | YaRN 32K | Native 64K | Index 64K | YaRN 64K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| niah_single_1 | 1.000 | 1.000 | 1.000 | .750 | 1.000 | 1.000 |
| niah_single_2 | 1.000 | 1.000 | 1.000 | .150 | 1.000 | .750 |
| niah_single_3 | 1.000 | 1.000 | 1.000 | .300 | 1.000 | .950 |
| niah_multikey_1 | .900 | .950 | .900 | .500 | .900 | .650 |
| niah_multikey_2 | .500 | .600 | .350 | .000 | .400 | .250 |
| niah_multikey_3 | .050 | .000 | .100 | .000 | .050 | .050 |
| niah_multivalue | .625 | .8125 | .8125 | .225 | .6375 | .5875 |
| niah_multiquery | .625 | .800 | .775 | .225 | .650 | .500 |
| variable tracking | .480 | .420 | .460 | .030 | .280 | .360 |
| common words | .025 | .020 | .025 | .070 | .005 | .000 |
| frequent words | .4667 | .3167 | .3500 | .3167 | .4667 | .4000 |
| SQuAD QA | .200 | .100 | .300 | .050 | .150 | .200 |
| HotpotQA | .250 | .250 | .200 | .250 | .150 | .200 |

The 64K index advantage is broad across the NIAH family but not uniform:
YaRN is better on variable tracking and both QA rows. Therefore the macro
advance does not license a natural-QA or task-universal claim. Qasper, 2Wiki,
Hotpot natural-document F1 and packed-natural NLL remain work-machine gates.

## Research update

Together with the earlier owners:

1. K32 N80 finds physical/index long parity but better Native retention for
   index; index also beats matched YaRN on core-4.
2. K64 cannot distinguish coordinates because the constructions coincide.
3. K128 N80 independently favors index over physical.
4. This new K32 full-13 panel favors index over YaRN at 64K while preserving
   the 32K macro.

The evidence therefore supports a simple engineering method: a checkpoint,
target scale `s`, and the frozen normalized-index profile yield one request-wide
static table. It does not yet support the stronger claim that a universal
two-parameter law has been identified across checkpoints or scales.

## Stop and handoff

Per user instruction, no further GPU experiment is launched on this instance.
The model-free packed-natural 32K/64K input artifact is prepared and hash-bound
but remains `NOT_RUN`. Its evaluator and summarizer are ready for the work
machine, conditional on the separate preregistration. The next stage must not
reopen physical-x, Native-Q/K P3, `G(x;K)`, residual correction, or gain search.
