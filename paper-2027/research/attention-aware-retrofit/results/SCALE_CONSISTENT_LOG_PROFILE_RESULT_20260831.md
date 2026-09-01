# Scale-consistent maximum-profile retrofit

- **Date:** 2026-08-31
- **Status:** current zero-training single-table exponent-space result
- **Decision:** retain the log-frequency law at `s=4` with the 1x-selected
  gain coefficient `c=0.074`; reject gain-free `L0` and do not promote the
  frozen movement mask to an `s=8` scale-general method
- **Model:** released OLMo-2-0425-1B-Instruct, 1.485B parameters
- **Training:** zero learned parameters and zero training tokens

## 1. Question and frozen construction

The experiment asks whether the already-frozen `s4` movement mask can define
one maximum profile from only the checkpoint and a requested maximum factor
`s`, without per-factor table fitting.  The mask is not refit:

```text
m_k = (1 - normalized_uniqueness_k)^2
```

The current arithmetic construction and the proposed log-frequency
construction are

```text
arithmetic: omega'_k = omega_k ((1-m_k) + m_k/s)
log law:    omega'_k = omega_k s^(-m_k)
```

The initial study evaluated `L0` with gain `1` and `L1` with the fixed analytic
schedule `1 + 0.1 log(s)`.  The 2026-09-01 follow-up kept the `s4` table fixed
and selected `c=0.074` using only the registered 1x PG-19 retention boundary;
no 2x/4x result selected the table or gain.  Every profile was installed before
prefill and remained fixed for the full KV-cache lifetime.

## 2. Geometry audit

Both laws preserve strict frequency ordering and the exact fast/slow endpoints
for `s=2,4,8`.  They differ in the meaning of the frozen movement mask.  Define

```text
q_k(s) = -log(omega'_k / omega_k) / log(s).
```

For the log law, `q_k(s)=m_k` exactly (up to floating-point error) at all three
factors.  For the arithmetic law, the RMS error `RMS(q-m)` is
`0.01099/0.02195/0.03200` at `s=2/4/8`; the maximum errors are
`0.08135/0.16630/0.24559`, all at pair `k=21`.  At that pair,
`m=0.67092`, while arithmetic `q` falls from `0.58957` to `0.50462` to
`0.42532` as the factor grows.  Thus the arithmetic law has a real, quantified
movement-saturation defect if `m` is intended to be a scale-independent
log-frequency generator.

The log law also satisfies the composition identity
`omega(s1*s2)=omega(s1)*s2^(-m)`.  This establishes scale consistency of the
formula only; it does not predict frozen-checkpoint language-model quality.

The reproducible audit is
[`scripts/analysis/audit_scale_consistent_log_interpolation.py`](../../../../scripts/analysis/audit_scale_consistent_log_interpolation.py).
The raw audit JSON and three-panel plot have SHA-256
`2d5a080bf363bcec3199a9570391a36e3b3c6943a7c0ad2b1d4c3584cfbe4510`
and `677bb3134362f1e6e2b2582820b6bc610d26052fa7115d51ad42c1bbf98766d1`.

## 3. Core-4 length curves

The table reports official task-macro over single-key, multikey-2,
multikey-3, and variable tracking, with 20 rows per task-length cell.

| Maximum profile | Law | 1x | 2x | 4x | 8x |
| --- | --- | ---: | ---: | ---: | ---: |
| s2 | arithmetic + fixed gain | 0.7875 | 0.5600 | -- | -- |
| s2 | log + fixed gain (`L1`) | 0.7750 | 0.5550 | -- | -- |
| s4 | arithmetic + fixed gain | 0.7425 | 0.6925 | 0.4025 | -- |
| s4 | log + fixed gain (`L1`) | **0.7550** | **0.7025** | **0.4275** | -- |
| s8 | arithmetic + fixed gain | 0.3700 | 0.3850 | 0.3075 | 0.2100 |
| s8 | log + fixed gain (`L1`) | **0.5925** | **0.4575** | **0.4625** | **0.3025** |

The log law is neutral at `s=2`, consistently improves the current arithmetic
profile at `s=4`, and materially improves it at every measured point for
`s=8`.  It therefore fixes a real part of the scaling defect.  It does not by
itself make the frozen mask a satisfactory `s=8` method.

## 4. Gain ablation and likelihood guard

`L0` cannot replace a gained system.  At `s=4`, gain-free log interpolation reduced the
16K single-key score to `0.4` in the five-row guard.  At `s=8`, its lower-range
core-4 macro was `0.0167`, and its 32K macro was exactly `0`.  The fixed
`1+0.1 log(s)` gain is therefore necessary in this checkpoint/protocol.

Eight FineWeb-Edu development documents gave the following prefix/dense/tail
NLL values:

| Profile | current arithmetic | log `L1` | log minus current |
| --- | --- | --- | --- |
| s2 at 2x | 2.94428 / 2.90189 / 2.87224 | 2.94453 / 2.90211 / 2.87249 | +0.00025 / +0.00022 / +0.00025 |
| s4 at 4x | 3.02282 / 2.99223 / 3.01011 | 3.02120 / 2.99270 / 3.01001 | -0.00162 / +0.00047 / -0.00010 |
| s8 profile at 4x | 3.37872 / 3.38158 / 3.40666 | 3.28390 / 3.28943 / 3.31757 | -0.09482 / -0.09215 / -0.08909 |

All losses are finite.  The guard supports formula-level improvement over the
current `s8` profile at 16K, not a 32K natural-text claim.  Raw summary hashes
for `s2/s4/s8` are respectively
`c72a7747d050c38b9dcb283875b410fe25e17c5e6bb06e839a32357dd53865f4`,
`deaa2d6e8d3a3d8db6e4cc7981bbaab67e099d4511dc51ca6c3f73dfc43f54cf`,
and `3ce005b2fc0a1b50c55b32a73d6ee2ee5a7c7d3bf695159e22e8b7dc654ca8b6`.

## 5. RULER-13 confirmation and stopping decision

On all 13 tasks and 20 rows per cell, the `s4` log-L1 maximum profile scored
`0.7101/0.6671/0.5486` at 4K/8K/16K.  The matched current arithmetic profile
scored `0.7036/0.6662/0.5442`; Native scored `0.7131/0.0000/0.0038`, and
official YaRN-4 scored `0.4314/0.2431/0.1056`.  The log-law result JSON SHA-256
is `ff8ebb9488da4ebdbc3b1442093c21b3226a901704e11aa17521e381a773357e`.

The `s8` full-task run triggered its registered early stop.  Completed rows
gave single-key-1 `0.95/0.95/1.00`, single-key-2 `0.50/0.55/0.65`, and
single-key-3 `0.00/0.20/0.25` at 4K/8K/16K.  The single-key-3 collapse means
the profile is not uniformly useful through `[1,8]`; the remaining tasks and
32K RULER-13 were not opened.  The preserved partial-row artifact has SHA-256
`f29f48bfd5eff538cff615bb6c97ada056937a1621b3bfb364607b23653259db`.

## 6. Formal single-table double gate (2026-09-01)

The follow-up froze the `s4` log-law table at float32 SHA-256
`56ddfae2800d4bbf9e6bd2d20bae751edc9865dcbaf641c7c8c4f1d7f1c15e5b`
and selected `c=0.074`, hence attention scaling `1.102585782722872`, using
only the 20-row 1x PG-19 retention boundary.  The same table and gain were then
used at every evaluated length, with zero routing and zero parameter updates.

The registered 1x gate is a conjunction, not an average:

| 1x endpoint | Candidate | Native | Retention | Gate |
| --- | ---: | ---: | ---: | ---: |
| PG-19 tail NLL / PPL retention | 3.104234 | 2.971047 | **0.875302** | >=0.875 |
| five-task natural macro | 0.315833 | 0.345134 | **0.915103** | >=0.875 |

The frozen candidate then produced:

| Natural endpoint | Native | YaRN-4 | arithmetic `s4`, `c=.074` | log `s4`, `c=.074` |
| --- | ---: | ---: | ---: | ---: |
| 2x PG-19 NLL | 7.100855 | 3.441096 | 3.088825 | **3.083278** |
| 2x six-task macro | 0.058883 | 0.218527 | 0.269405 | **0.307614** |
| 4x PG-19 NLL | 7.205538 | 3.793878 | 3.089382 | **3.081946** |
| 4x six-task macro | 0.021517 | **0.263677** | 0.250473 | 0.260055 |

The natural arithmetic rows above are the matched `c=.074` control.  The log
law is directionally better at all four natural long endpoints.  The matched
arithmetic result SHA-256 is
`0a57e4e3ef20a132fb971eb2403933065bce96e40d9b9b1cf16c6c668c19aa62`.
The separate same-gain 1x PG-19 control below determines the retention gate.

On the four-task RULER subset, the same candidate scored `0.7000/0.3375` at
8K/16K.  The matched Native values were `0/0`, YaRN-4 was `0.2200/0.0650`,
and the arithmetic `s4` profile was `0.6925/0.4025`.  Thus the candidate is not
uniformly best: it improves the arithmetic profile at 8K but gives back 0.0650
at 16K.

The subsequent untouched 13-task matrix scored `0.71397/0.66705/0.49859` at
4K/8K/16K.  Native scored `0.71308/0/0.00385`, and YaRN-4 scored
`0.43141/0.24308/0.10564` on the same rows.  The full-matrix result SHA-256 is
`86c042fdc55df15564b79b641f3d1ac55ea724e08690013821639e8eca90d1d0`.

At matched `c=.074`, the arithmetic profile scored
`0.69731/0.65429/0.50481`.  The log law improves 4K and 8K but reverses by
`-0.00622` at 16K.  Therefore the broad claim that exponent interpolation
uniformly improves long capability is closed: its natural endpoints are all
better, while RULER remains task- and length-dependent.  The matched arithmetic
RULER result SHA-256 is
`30b617935921ba22551e827b2c09a14e545c7a014f94c750302ad676c69bba81`.

### Qwen construction transfer

Without reading Qwen outcomes, the same construction recomputed `m_i` from the
Qwen Native `L=32768`, `b=10^6`, and `K=64`, retained `s=4` and the OLMo-selected
`c=0.074`, and produced table hash
`ed8abbb27a1a87beab68d63655acee1285f2a82cae196b92f43b78f9da58ba86`.
It scored `0.7000/0.5875` on 64K/128K core-4.  Native scored
`0.5450/0.4350`, YaRN-4 `0.6025/0.4650`, and the prior valid corrected-derived
control `0.6650/0.5400`.  The new result SHA-256 is
`2809395183b65f84d9a389dae3034dbd85004ee017fa22eef6851e4612e61e49`.

This is construction-algorithm transfer, not literal OLMo-vector transfer:
Qwen recomputes its own profile and the realized table has crossings at pairs
`1` and `18`.  The result has no matched Qwen 1x PPL gate and therefore supports
long capability transfer only.

A pre-registered Qwen control then preserved this final frequency multiset,
endpoints, gain, and 64K rows while permuting only the interior slot assignment.
All four 20-row task cells scored zero, versus macro `0.7000` for the ordered
reference.  The result SHA-256 is
`056938bce1e859398fa48e5771f274a20866cff9861484445e34289298636135`.
Together with the OLMo 1x collapse, this establishes cross-checkpoint
slot–frequency non-exchangeability.  It still does not identify a unique
coupling law.

The next pre-registered transfer used the dimensionless adjacent-lattice cycle
coordinate

```text
u_i = L_native (omega_i - omega_{i+1}) / (2 pi)
```

and piecewise-linearly transported the frozen OLMo `m(u)` onto Qwen without
reading the Qwen profile or outcomes.  At 64K it scored `0.6725`, within 0.0275
of the Qwen-recomputed profile's `0.7000` and inside the pre-registered 0.05
parity band.  The result SHA-256 is
`b3e523bf9a8f0bda03342238bee47dc98195aabd48012856f2757d0805756a22`.
The saved table is reproduced by linear interpolation in `u`; Qwen fast slots
`0--9` are endpoint-clamped. This supports one no-refit transport
parameterization, not a uniquely identified physical coordinate.

The CPU-only follow-up defines `x=ln(u)` and fits the OLMo movement with a
two-parameter clipped-affine `G_4(x)`. It reaches OLMo movement MAE `0.001223`
and differs from the existing Qwen transport by MAE `0.002174` without refit.
This is geometry compression only; its first LM evaluation is pending. See
[`CPU_LOW_DIM_COUPLING_LAW_20260901.md`](CPU_LOW_DIM_COUPLING_LAW_20260901.md).

### Weights-by-table readout fork

A pre-registered 2x2 changed only weights and table at the same OLMo geometry,
gain, and PG-19 rows.  The second weight state was a rank-64 Q/K LoRA trained
for 300 steps on Native-table 4K natural LM; it never used long lengths or the
log-s4 table.

| Weights | Table | 1x NLL | 2x NLL | 4x NLL |
| --- | --- | ---: | ---: | ---: |
| Original | Native | 3.085932 | 7.219371 | 7.221997 |
| Original | log-s4 | 3.104234 | 3.083278 | 3.081946 |
| Native-4K Q/K-LoRA | Native | 2.927003 | 7.114380 | 7.091656 |
| Native-4K Q/K-LoRA | log-s4 | 2.913996 | 2.900898 | 2.915873 |

Within both weight states, log-s4 satisfies the 0.875 1x retention condition
relative to the same-gain Native-table arm and improves both long NLLs.  The
pre-registered geometry-only H1 therefore survives with no readout-induced
sign reversal.  One Q/K-LoRA perturbation does not prove universal readout
independence.  The compact result SHA-256 is
`c53fd58a45f3824f9e2fc4734bbc53476da969aeb84a70bc6b4726fe4f50e7b5`.

The matched same-gain causal control is decision-relevant.  At `c=0.074`, the
arithmetic frequency interpolation reached PG-19 NLL `3.110788` and PPL
retention `0.869584`, failing the 0.875 gate, while exponent-space interpolation
passed.  This isolates a small but gate-changing benefit from implementing the
same frozen coupling as

```text
omega'_k = omega_k s^(-m_k)
z'_k = z_k - m_k log_b(s)
```

rather than linearly interpolating frequencies.  It does not establish that the
movement mask is unique or universal.

The compact remote receipt has SHA-256
`ec088c338c2b29f9dc3ce23d6e0741519fa72af770689ff9acc86aa8a3c2f1cf`.
Its candidate PG-19, 1x-task, natural-long, and RULER result hashes are
`e350de4d6dfaac1644d951057f9c6cf6e98c1739d688cef8b4e2e708dcf25595`,
`d8651c9864a78c5a6d81b08b7a11a9f1b7c64a0d18a6c21feb1a5e3810093390`,
`2145c5140ddcccc878767738615b1cb0f82afb59b17a0da3ade16c814780fd66`,
and `851477924e551d00c932a429eec8fac521f6463aa9b83d7abfcdcd28373298ea`.

### Dilation-distribution fork

One endpoint-preserving random permutation of the successful arithmetic
profile's interior dilation factors kept the same interior dilation multiset
but changed its slot pairing.  It produced 1x PG-19 NLL `4.068625` and core-4
scores `0/0` at 8K/16K.  Endpoint-matched Haar and MaxEnt-`lambda=1` ordered
allocations passed the PG-19 retention threshold but scored `0.6125/0` and
`0.6650/0` at 8K/16K.  The permutation also changed the final frequency
multiset and introduced 14 order crossings.  It therefore rejects unrestricted
random dilation reassignment but does not by itself isolate slot coupling from
physical-spectrum geometry.

The pre-registered discriminator permuted the successful log table's interior
**final frequencies**, preserving their multiset, endpoints, gain, and the same
seed.  H1 predicted slot/readout collapse; H2 predicted near parity from the
unchanged unordered spectrum.  The candidate's 1x PG-19 NLL was `6.864926`,
versus `3.104234` for the reference, so `Delta NLL=+3.760692` decisively crossed
the pre-registered H1 threshold `+0.10`.  Per protocol, 16K was not opened.
The result SHA-256 is
`ee03b6dae6345f9bb82b0be5f7cdc525f2ed65cb915901cfaff8458b4c578da0`.
This establishes slot–frequency non-exchangeability for the frozen checkpoint;
it does not identify a unique or optimal ordered law.

## 7. Zero-refit s=2 behavioral confirmation (2026-09-01)

The scale follow-up kept the two C2 boundaries frozen from s=4 and changed only
the analytic scale input:

\[
\omega_i'(2)=\omega_i2^{-G(x_i)}.
\]

It also reused the single coefficient `c=0.074`, giving attention scaling
`1.0512928913614359`; there was no s=2 fit or search. The float32 table SHA-256
is `5a467a9ef538431b8aba6f93b314bd7926149365ec54ea58985a0122beef3634`.
The table was fixed before prefill at 1x and 2x.

The same table passed both Native gates:

| OLMo 1x endpoint | s2 profile | Native | Retention |
| --- | ---: | ---: | ---: |
| PG-19 tail NLL / PPL retention | `2.987191` | `2.971047` | `0.983986` |
| five-task macro | `0.359361` | `0.345134` | `1.041220` |

At its 2x horizon it remained useful rather than merely Native-compatible:

| OLMo 2x endpoint | s2 profile | Native reference |
| --- | ---: | ---: |
| PG-19 tail NLL | `2.970433` | `7.100855` |
| six-task natural macro | `0.260341` | `0.058883` |
| RULER core-4 | `0.5150` | `0.0000` |

The core-4 task vector was `1.00 / 0.70 / 0.30 / 0.06`. On the same 8K rows,
the s4 C2 profile scored `0.7150`; hence zero-refit s2 supports a useful smooth
scale reduction, not the stronger claim that the smallest maximum profile is
optimal at each target length. This is OLMo-only behavioral scale evidence;
Qwen s2 and a full RULER-13 s2 matrix were not run.

The compact machine-path-free receipt is
[`../evidence/SCALE_INDEPENDENT_G_S2_RECEIPT_20260901.json`](../evidence/SCALE_INDEPENDENT_G_S2_RECEIPT_20260901.json).

## 8. Answers and boundary

1. **Movement saturation exists.** It is localized mainly to the transition
   band and grows monotonically from `s2` to `s8`.
2. **The log law solves formula-level scale consistency.** It also improves the
   measured `s4` and `s8` arithmetic profiles, but scale consistency is not
   sufficient for frozen-checkpoint task retention.
3. **Gain cannot be deleted.** `L0` fails the registered RULER guard.  The
   current `c=0.074` operating point was selected on 1x PG-19 only and passes
   the separate five-task 1x gate.
4. **Useful `s2` and `s4` static profiles were obtained without scale-specific
   table fitting.** The zero-refit s2 profile passes the 1x double gate and is
   useful at 2x; the s4 profile passes the same gate and strongly improves 2x/4x
   over Native. This is not universal dominance or per-target optimality.
5. **Slot identity is causal for the frozen readout.** The
   frequency-multiset-preserving permutation decisively rejects unordered
   physical-spectrum sufficiency.  This establishes non-exchangeability, not
   the uniqueness or optimality of the current ordered law, and does not
   authorize a 64-dimensional learned or outcome-searched table.
6. **Scope:** the s2 behavioral confirmation is a single-checkpoint,
   zero-training, support-changing exponent intervention. It supports
   `s=4 -> s=2` behavior-scale consistency on OLMo, not a universal optimum or
   arbitrary-s theorem.
