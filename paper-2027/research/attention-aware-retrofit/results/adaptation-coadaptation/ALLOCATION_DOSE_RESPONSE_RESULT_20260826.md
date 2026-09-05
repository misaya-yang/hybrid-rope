# Fixed-support allocation dose response on released OLMo-2

- **Run:** 2026-08-25/26, single released OLMo-2-0425-1B-Instruct checkpoint
- **Status:** complete; registered primary construction does not pass the joint
  in-window/long-tail gate
- **Owner:** this report plus
  `../evidence/ALLOCATION_DOSE_RESPONSE_RESULTS_20260826.json`
- **Protocol:** frozen weights, amplitude `1.0`, Native endpoints held bitwise,
  128 fresh FineWeb-Edu documents at 4K/8K/16K, 20,000 paired document
  bootstrap resamples

## Result

Interior allocation is a behavioural dose axis, but the tested analytic Cosh
path does not provide a no-cost mature-checkpoint retrofit. Its smallest
non-zero dose (`lambda=0.02`) improves 16K final-1024 tail NLL by `-0.1153`
(`[-0.1338,-0.0966]`) while increasing 4K full NLL by `+0.0169`
(`[+0.0153,+0.0185]`). This misses the registered `+0.01` 4K guard. Larger
doses increase 4K cost sharply; the best tested 16K tail point is the interior
`lambda=0.35`, but its 4K full-NLL cost is `+3.2769`.

The independently learned oracle direction shows a more favourable local
trade-off. Scaling that displacement by `1` and `4` changes 4K full NLL by
only `+0.0008` and `+0.0061`, while changing 16K tail NLL by `-0.0450` and
`-0.1174`. The 16K tail effect grows through scale `16` (`-0.1572`), satisfying
the registered Path-B dose check. Long full-sequence NLL nevertheless worsens
at every non-zero point. Thus this is a continuous full-versus-tail
redistribution, not evidence that the analytic table or learned direction is a
finished method.

## Registered predictions

| Prediction | Outcome |
| --- | --- |
| P1 causal-measure `r2` increases along analytic `lambda` | passed by the frozen static manifest |
| P2 4K full NLL increases along analytic `lambda` | passed at every tested point |
| P3 16K tail NLL has an interior minimum | passed; tested minimum at `lambda=0.35` |
| P4 the minimum lies closer to static-`r2` maximiser `1` than to Native `0` | failed; `0.35` is closer to `0` |
| Primary joint gate: 16K tail gain and 4K full-NLL cost at most `+0.01` | failed for every analytic Path-A arm |
| Path-B magnitude grows through scales `1/4/16` | passed for 16K tail NLL |

The preregistration was not weakened after seeing these results. Capability
stage E2 was not launched because Path A produced no point meeting its joint
gate.

## Complete mean deltas from bitwise Native

Negative tail deltas favour the treatment. Positive full deltas are costs.

| arm | 4K full | 8K full | 8K tail | 16K full | 16K tail |
| --- | ---: | ---: | ---: | ---: | ---: |
| `coshA_tau4_lam0.02` | +0.0169 | +0.1899 | -0.0344 | +0.0935 | -0.1153 |
| `coshA_tau4_lam0.05` | +0.4347 | +0.4494 | -0.0074 | +0.2127 | -0.1918 |
| `coshA_tau4_lam0.1` | +1.1882 | +0.8323 | +0.0652 | +0.3952 | -0.0778 |
| `coshA_tau4_lam0.2` | +2.3256 | +1.4203 | -0.0548 | +0.6410 | -0.2014 |
| `coshA_tau4_lam0.35` | +3.2769 | +1.8168 | -0.1303 | +0.7937 | -0.3185 |
| `coshA_tau4_lam0.5` | +3.7043 | +2.0181 | -0.0963 | +0.8930 | -0.3004 |
| `coshA_tau4_lam0.7` | +4.0814 | +2.2581 | -0.0267 | +1.0744 | -0.1668 |
| `coshA_tau4_lam1` | +4.3630 | +2.4354 | +0.0692 | +1.1961 | -0.1190 |
| `coshA_tau2_lam0.1` | +0.4875 | +0.4740 | -0.0073 | +0.2257 | -0.2006 |
| `coshA_tau2_lam0.35` | +2.1696 | +1.3336 | -0.0692 | +0.5927 | -0.2534 |
| `coshA_tau2_lam0.7` | +3.3749 | +1.8863 | -0.0570 | +0.8474 | -0.2075 |
| `learnedB_lam1` | +0.0008 | +0.0332 | -0.0168 | +0.0145 | -0.0450 |
| `learnedB_lam4` | +0.0061 | +0.1381 | -0.0329 | +0.0666 | -0.1174 |
| `learnedB_lam16` | +0.5294 | +0.4958 | -0.0112 | +0.2321 | -0.1572 |
| `learnedB_lam64` | +1.9964 | +1.2658 | +0.1611 | +0.5781 | -0.1746 |

Native full/tail NLL is `2.7567/2.7539` at 4K, `4.6500/7.0021` at 8K,
and `5.8976/7.2702` at 16K.

## Interpretation and claim ceiling

The experiment answers a narrow reviewer objection: the frozen-checkpoint
effect is not confined to two isolated tables. Small moves along two frozen
directions produce graded changes, and the learned direction preserves the
same 16K-tail sign through scales `1/4/16`. It does not validate static `r2` as
a behavioural selector: the registered location prediction fails. It also
does not show a universal Cosh benefit, checkpoint-population generalisation,
or a completed zero-training method. Path B inherits one learned oracle and is
mechanism evidence only.

## Receipts

- full result SHA-256:
  `42c1c5ea09dc3c59b9682272cb2aae0854da204d41938a2e3924383e0f5d6279`
- per-row JSONL SHA-256:
  `a8210bd8f41988a74cbfcabaf6fca0f7ae2903f0da59322454046d60c1d6bd21`
- frozen prediction SHA-256:
  `99a0e8f83e1d05534a5e2b8d62b24b4ebef21e96e0ab5e13adfe79c9cf7d5a1a`
- frozen grid manifest SHA-256:
  `96ba385ebda82d353760df597f0b3db50d9ff229f228699cf82983f472b51384`
- runtime: `1368.84` s; peak allocated GPU memory: `4,002,026,496` bytes
- raw rows and checkpoint paths remain outside the anonymous repository

