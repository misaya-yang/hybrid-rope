# NTS2 frozen OLMo decision

## Result

`native_tailspline_s2_midgain_v1` is a strong Native task-performance result.
On the same frozen OLMo-2-0425-1B-Instruct checkpoint and the same inputs used
by the existing Native/NCP confirmation:

| Evaluation | Native | NCP | NTS2 | NTS2 − Native | NTS2 − NCP |
|---|---:|---:|---:|---:|---:|
| Full-13×10, task-equal | 73.53% | 76.78% | **76.14%** | **+2.62pp** | −0.64pp |
| Natural-QA99, task-equal | 39.41% | 38.46% | **47.45%** | **+8.03pp** | **+8.99pp** |
| Same-target full-context NLL | 2.91527 | **2.90248** | 2.91984 | +0.00457 | +0.01736 |

The NLL delta versus Native corresponds to approximately `+0.458%` PPL. Thus
NTS2 improves both complete RULER and all three Natural-QA task point estimates
while paying a small language-modeling cost.

Natural-QA gains versus Native are `+9.49pp` on 2WikiMQA, `+13.33pp` on
HotpotQA, and `+1.28pp` on Qasper. On Full-13 the largest positive changes are
VT (`+56pp`), FWE (`+10pp`), NIAH multikey-1 (`+10pp`) and multiquery
(`+7.5pp`); the largest negative change is multikey-3 (`−40pp`).

The preregistered all-interface gate did not pass: it required at least `+5pp`
over Native and `+2pp` over NCP on Full-13, plus full-context NLL within `0.001`
of NCP. Per the frozen plan, this result does not authorize changing scale,
gain, band, or mixing NTS2 with NCP.

## Frozen method identity

- OLMo canonical band: `[14, 32]`
- Scale: `S=2`
- Allocation: exact finite-grid TailSpline
- Rotary gain: `sqrt(1 + 0.1 ln 2) = 1.0340767466953285`
- FP32 frequency-table SHA256:
  `0ccaecb736d94072579e76caa3ae12fa65ac565a2f1b7a4e7f10287c51938485`
- No weight updates, model-output fitting, task fitting, or runtime switching

## Evidence

- [Decision report](reports/server_20260917/decision.json)
- [Frozen table receipt](reports/server_20260917/nts2_table.json)
- RULER: [raw](reports/server_20260917/ruler_generations.jsonl) ·
  [contract](reports/server_20260917/ruler_contract.json) ·
  [summary](reports/server_20260917/ruler_summary.json)
- Natural-QA: [raw](reports/server_20260917/qa_generations.jsonl) ·
  [contract](reports/server_20260917/qa_contract.json) ·
  [summary](reports/server_20260917/qa_summary.json)
- LM: [raw](reports/server_20260917/lm_scores_candidate.jsonl) ·
  [contract](reports/server_20260917/lm_contract.json)

The decision report reuses the already frozen Native/NCP raw rows and verifies
identical prompt/target identities before scoring. The server data remain under
`native_tailspline_s2_midgain_20260917`; the compact positive-result evidence
above is also Git-distributed.
