# CA-NCP OLMo Native-4K result

The frozen five-arm Full-13x10 experiment is complete. Full-strength coordinate
alignment fails decisively on this checkpoint and panel; the original CA-NCP
method must not be transferred to Llama.

| Arm | Task-equal Full-13 score |
|---|---:|
| N0 Native | 73.53% |
| C0 NCP | 76.78% |
| P0 carrier-NCP | 76.01% |
| N1 Native + alignment | 60.40% |
| P1 carrier-NCP + alignment | 64.67% |

The controlled alignment effects are `N1-N0 = -13.13pp`, with paired stability
interval `[-18.76,-7.62]pp`, and `P1-P0 = -11.35pp`, interval
`[-16.46,-6.33]pp`. Carrier-only `P0-C0` is `-0.77pp`, interval
`[-2.62,+0.62]pp`; the carrier change does not explain or repair the failure.
All arms have zero empty outputs, so this is not a formatting or empty-generation
artifact. The largest losses occur on the three multikey tasks.

The implementation checks passed: identity runtime parity is token exact, all
five arms contain the same 130 row IDs, and the nonidentity Torch operator was
checked against its dense CPU reference. The failure is structural. All 256
layer-by-KV-group planes were active; the median carrier overlap was `0.0644`,
equivalent to a median plane angle of about `86.3` degrees. Rank two was low
rank, not a small perturbation.

The formal report is
[paired_report.json](reports/server_20260917/paired_report.json), SHA256
`5eb443875306a0cba1b1b1536d7714f183045e29a1195082ce9ece67f842eca6`.
It records the five raw-generation hashes, panel identity, task breakdown,
output health, and common paired stability analysis. The raw JSONL files remain
on the experiment server under the paths represented by those hashes.

This result supports a narrow causal conclusion: full-strength CA alignment
damages OLMo Native-4K task quality under the frozen construction. It does not
claim that every bounded or sparse coordinate intervention fails. The
[safety follow-up](../ca_ncp_safe_followup_20260917/README.md) is explicitly a
post-failure development diagnostic on the same panel and requires independent
confirmation if any arm becomes positive.
