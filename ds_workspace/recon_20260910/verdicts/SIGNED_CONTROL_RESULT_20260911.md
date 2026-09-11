# Signed frequency control: a real OLMo effect concentrated in UUID completion

This is a CPU readout of previously generated outputs. No GPU/model jobs were
started. It does not establish a method that reliably exceeds MrRoPE on Qwen.

## Complete selection panel

The existing OLMo BM frequency table was changed at slots 28–31 by a common
additive shift of plus or minus `1/16384` radians per token, at unchanged gain.
Both arms have all 350 rows, with 350 unique source prompt hashes, seven task
types and 50 rows per task. Recomputing the original scorer yields no mismatches.
Legacy output records omit prompt hashes themselves; identity is joined through
the original source panel and row IDs, not a full cryptographic replay receipt.

| Arm | Accuracy |
|---|---:|
| Slower, minus shift | 47.1714% |
| Faster, plus shift | 35.3429% |
| Paired difference | +11.8286 pp |

Descriptive within-task paired bootstrap interval: **[+8.4282, +15.2571] pp**.
This is not adjusted for all earlier candidate selection.

| Task | Slower minus faster |
|---|---:|
| UUID single needle (`niah_single_3`) | +54 pp |
| `niah_multikey_1` | +18 pp |
| `qa_2` | +8 pp |
| `niah_single_1` | +6 pp |
| `niah_multikey_3` | 0 pp, both at floor |
| `cwe` | -0.2 pp |
| `niah_multivalue` | -3 pp |

The earlier forecast of a 0–4 pp signed effect was wrong on this panel. Preserve
the observed effect instead of erasing it because the overall research goal is
unfinished. Equally, this signed difference is not the effect against MrRoPE:
both interventions start from BM, and this is an OLMo-specific comparison.

## What the UUID outputs show

| Correct reference prefix present in output | Slower / 50 | Faster / 50 |
|---|---:|---:|
| First 8 characters | 46 | 44 |
| First 13 characters | 44 | 39 |
| First 18 characters | 42 | 23 |
| First 23 characters | 42 | 18 |
| Full 36-character UUID | 38 | 11 |

There are 28 slower-arm wins in this task; in 25 of them the faster output
already contains the correct first eight characters. Scoring the **first full
canonical UUID** instead of any reference substring still gives 38 versus 11.
Thus extra text eventually mentioning the reference is not responsible for this
task's measured gap.

The task supplies about 65.2% of the total signed score difference. The output
evidence is more consistent with a large difference in accurate completion of
the target string than a large difference in emitting any correct target prefix.
It does not prove an attention-addressing, binding, or decoding mechanism:
repeated retrieval during copying and downstream state changes remain alternatives.
The healthy Qwen six-task panel does not include this UUID task, so that panel
cannot directly test transfer of the dominant observed OLMo effect.

## Holdout is incomplete

The faster arm has 180 saved rows and the slower arm 60. The full source panel
contains only 120 unique prompts, as previously audited. The current common 60
rows happen to be unique: 24 at 4K, 36 at 16K.

- At 4K, six-task common-subset difference: -0.4861 pp, descriptive interval
  [-4.7917, +4.1667].
- At 16K, the available five-task common-subset difference is +5.875 pp,
  interval [-3.125, +15.875]. QA is absent and FWE is only partly covered.
- VT worsens by 17.5 pp on the eight available long rows (0 wins, 5 losses).

These numbers do not complete the registered confirmation. Sign pairing does
not cancel sample selection or task dependence. Do not resume these runs while
the author's no-GPU restriction is active.

## An exact mechanism constraint

For fixed raw complex Q/K and relative distance d, let the selected band's
original contribution be `C(d) = sum_j q_j conj(k_j) exp(i nu_j d)`.
The signed intervention obeys

    z_minus(d) - z_plus(d) = 2 sin(delta d) Im C(d).

The shared logit gain/normalization multiplies both sides. Here `delta d` lies
in [0,1] over the tested causal distances, so its sine does not change sign.
The direction of the fixed-state effect is therefore determined by the learned
complex contribution, not by an intrinsically favorable sign of frequency shift.
Full prefill changes those raw states too; this identity is not an end-to-end
prediction.

Within the shifted band, all pairwise frequency differences are unchanged.
For complex Fourier features Phi[d,j], a common shift left-multiplies Phi by a
diagonal unitary matrix, so Phi* Phi and its spectrum are unchanged. The CPU
check gives a Gram difference below 2e-15 and signed-identity error below 4e-15.
This excludes improvement of that *within-band complex Gram conditioning* as
an explanation for this intervention. It does not assert invariance of cross-band
relations, the full real sine/cosine Gram, or the actual network.

## Other completed controls

BM gain 1.05 scored 25.5857% on 350 rows; gain 1.10 scored 41.0429% on 350 rows.
Gain 1.20 has only 83 rows; its observed 63.8554% is not comparable to a complete
350-row score. No further gain sweep was launched.

## Reproduction and decision

Run `experiments.rope_decision_20260911.read_completed_controls` with root
`results/rope_decision_20260911/completed_legacy` and an output JSON path.
Saved readout: `results/rope_decision_20260911/completed_controls_readout.json`.
The algebra receipt is `results/rope_decision_20260911/common_carrier_identity.json`.

Retain accurate target-string completion as the specific phenomenon to explain.
Do not generalize the signed winner into a globally better table or infer that
uniformly slowing these frequencies will help VT or a different checkpoint.
The reliable-over-MrRoPE objective remains unresolved.
