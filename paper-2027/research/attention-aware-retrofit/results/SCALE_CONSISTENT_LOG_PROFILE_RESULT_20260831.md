# Scale-consistent maximum-profile retrofit

- **Date:** 2026-08-31
- **Status:** bounded method-development result
- **Decision:** retain the log-frequency law at `s=4`; reject gain-free `L0`;
  do not promote the frozen movement mask to an `s=8` scale-general method
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

Only two log-law systems were evaluated: `L0` with gain `1`, and `L1` with the
fixed analytic schedule `1 + 0.1 log(s)`.  No coefficient, exponent, ramp,
cutoff, or band was selected from these results.  Every profile was installed
before prefill and remained fixed for the full KV-cache lifetime.

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

`L0` cannot replace `L1`.  At `s=4`, gain-free log interpolation reduced the
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

## 6. Answers and boundary

1. **Movement saturation exists.** It is localized mainly to the transition
   band and grows monotonically from `s2` to `s8`.
2. **The log law solves formula-level scale consistency.** It also improves the
   measured `s4` and `s8` arithmetic profiles, but scale consistency is not
   sufficient for frozen-checkpoint task retention.
3. **Gain cannot be deleted.** `L0` fails the registered RULER guard; `L1` is
   the only retained system.
4. **A genuine all-scale one-input profile was not obtained.** `s4` is a strong
   positive result, but the same frozen mask fails the full-task `s8` gate.
   No extra parameter was introduced to rescue it.
5. **Next theory target:** estimate the checkpoint-induced Q/K metric and ask
   whether a frequency-deformation direction is weakly observable on
   `[0,L]` but increases retrieval margin on `[L,sL]`.  This is a new
   measurement/theory program, not an unrun result and not part of the present
   candidate family.
