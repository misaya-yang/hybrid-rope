# Counterexample search — full-RoPE collision theory (falsification-first)

Deterministic run (seed 20260819), numpy float64, L = 64/128/256/512 as noted.

## Part A — canonical-correlation decomposition {s1, s2} vs {sqrt(c), sqrt(s)}

200 random (wi, wj) pairs per length, wi, wj ~ logU[2e-6, 1]; c, s are the squared normalized cos/sin overlaps; s1 >= s2 the canonical correlations (QR-based canon_corr).

| grid | L | max |s_i - sqrt(overlap^2)| | mean | max (well-cond.) |
|---|---|---|---|---|
| sym | 64 | 6.061e-09 | 1.698e-10 | 3.220e-15 |
| sym | 256 | 2.995e-10 | 9.492e-12 | 4.663e-15 |
| causal | 64 | 8.657e-01 | 1.379e-01 | 7.692e-01 |
| causal | 256 | 8.653e-01 | 1.508e-01 | 7.717e-01 |

'well-cond.' = both self-block diagonals within 1% of D(0); on the symmetric grid the residual deviation lives entirely in near-degenerate pairs (|w_i - w_j| L << 1, where D(w_i-w_j) +/- D(w_i+w_j) suffers float64 cancellation) and is ~1e-9, far below the O(1/L) ~ 1e-2 scale.

Interpretation: on the symmetric grid the cos-sin cross terms vanish exactly by odd symmetry, so the decomposition holds up to float64 cancellation in the Dirichlet differences (max ~1e-9, see note below the table); on the causal-weighted grid the cos-sin cross terms do NOT vanish (e.g. the constant part of cos(w d) correlates with the ramp part of sin(w d)), so the per-channel overlaps are only an approximation to the canonical correlations there (deviations up to ~0.87 for low-frequency pairs).

## Part B — Type-1 counterexamples (C_cos order vs effrank order), L = 128

Violation = pair (A, B) with C_cos(A) < C_cos(B) AND effrank(A) < effrank(B), i.e. the cosine-only collision total orders two tables the opposite way from the whitened-Gram effective rank.

### B.i/ii symmetric grid — 20000 random tables per family, K = 3 and 4

| K | family | n valid | viol. frac | viol. frac (Cc_B > Cc_A) | Spearman(C_cos, er) | best er-margin | near-miss margin |
|---|---|---|---|---|---|---|---|
| 3 | loguni | 20000 | 2.22% | 4.44% | -0.979 | 1.147 | -1.296e-12 |
| 3 | uni | 20000 | 3.73% | 7.44% | -0.957 | 0.9448 | -6.25e-07 |
| 3 | pic | 20000 | 3.08% | 6.16% | -0.972 | 1.134 | -4.277e-09 |
| 3 | mix | 20000 | 5.34% | 10.66% | -0.898 | 1.012 | -3.083e-09 |
| 3 | all+targ | 80026 | 2.73% | 5.48% | -0.973 | 1.147 | -1.296e-12 |
| 4 | loguni | 20000 | 2.22% | 4.44% | -0.980 | 1.346 | -3.68e-11 |
| 4 | uni | 20000 | 2.42% | 4.85% | -0.978 | 1.005 | -3.059e-09 |
| 4 | pic | 20000 | 2.55% | 5.10% | -0.981 | 1.725 | -5.156e-09 |
| 4 | mix | 20000 | 3.93% | 7.88% | -0.940 | 1.103 | -3.616e-08 |
| 4 | all+targ | 80014 | 1.87% | 3.76% | -0.987 | 1.725 | -3.68e-11 |

#### Best symmetric K=3 counterexample (largest effrank reversal)

- A = [0.00052403, 0.014597, 0.025767]: C_cos = 0.928, C_full = 1.554, effrank = 3.2944
- B = [0.02463, 0.032509, 0.065416]: C_cos = 0.93129, C_full = 0.78582, effrank = 4.4411
- margins: C_cos(B) - C_cos(A) = 0.0032928;  effrank(B) - effrank(A) = 1.1466  (violation: C_cos(A) < C_cos(B) yet effrank(A) < effrank(B))

Largest C_cos gap among violations with both tables non-degenerate: C_cos(B) - C_cos(A) = 1.1487 (A = [0.00014078, 0.0007724, 0.020622], B = [0.023778, 0.026778, 0.035966])

#### Best symmetric K=4 counterexample (largest effrank reversal)

- A = [0.00060202, 0.001626, 0.022113, 0.022927]: C_cos = 2.0801, C_full = 3.5307, effrank = 3.1887
- B = [0.021845, 0.025593, 0.036266, 0.13612]: C_cos = 2.1012, C_full = 1.7759, effrank = 4.914
- margins: C_cos(B) - C_cos(A) = 0.021088;  effrank(B) - effrank(A) = 1.7254  (violation: C_cos(A) < C_cos(B) yet effrank(A) < effrank(B))

Largest C_cos gap among violations with both tables non-degenerate: C_cos(B) - C_cos(A) = 1.8819 (A = [0.0016994, 0.0054146, 0.020037, 0.020954], B = [0.027806, 0.038252, 0.039292, 0.042421])

#### Hand-designed pairs on the symmetric grid

| pair | C_cos(A) | C_full(A) | er(A) | C_cos(B) | C_full(B) | er(B) | violates |
|---|---|---|---|---|---|---|---|
| hand A3 vs B3 | 0.00014553 | 0.3846 | 5.1435 | 0.14631 | 0.15112 | 5.6975 | YES |
| hand A4 vs B4 | 0.000302 | 0.41878 | 7.0291 | 0.14666 | 0.15133 | 7.6952 | YES |
| hand A6 vs B6 | 0.000799 | 0.45046 | 10.905 | 0.14717 | 0.15188 | 11.692 | YES |

### B.iii causal-weighted grid — 20000 random tables per family, K = 3 and 4

| K | family | n valid | viol. frac | viol. frac (Cc_B > Cc_A) | Spearman(C_cos, er) | best er-margin | near-miss margin |
|---|---|---|---|---|---|---|---|
| 3 | loguni | 20000 | 1.84% | 3.70% | -0.983 | 1.669 | -8.686e-07 |
| 3 | uni | 20000 | 10.04% | 20.14% | -0.737 | 1.381 | -1.682e-06 |
| 3 | pic | 20000 | 3.46% | 6.92% | -0.967 | 1.64 | -7.034e-05 |
| 3 | mix | 20000 | 12.48% | 24.94% | -0.631 | 1.492 | -4.369e-06 |
| 3 | all+targ | 80026 | 5.21% | 10.46% | -0.920 | 1.682 | -8.686e-07 |
| 4 | loguni | 20000 | 2.07% | 4.13% | -0.980 | 2.204 | -1.537e-07 |
| 4 | uni | 20000 | 7.30% | 14.60% | -0.854 | 1.559 | -3.351e-07 |
| 4 | pic | 20000 | 3.10% | 6.20% | -0.974 | 2.27 | -1.898e-09 |
| 4 | mix | 20000 | 8.39% | 16.82% | -0.831 | 1.914 | -1.683e-05 |
| 4 | all+targ | 80014 | 3.44% | 6.87% | -0.967 | 2.332 | -1.898e-09 |

#### Best causal-weighted K=3 counterexample (largest effrank reversal)

- A = [3.5991e-05, 0.043522, 0.086199]: C_cos = 0.0044013, C_full = 0.8275, effrank = 4.2183
- B = [0.20052, 0.2706, 2.0139]: C_cos = 0.0044299, C_full = 0.048548, effrank = 5.9006
- margins: C_cos(B) - C_cos(A) = 2.8573e-05;  effrank(B) - effrank(A) = 1.6823  (violation: C_cos(A) < C_cos(B) yet effrank(A) < effrank(B))

Largest C_cos gap among violations with both tables non-degenerate: C_cos(B) - C_cos(A) = 1.7849 (A = [5.0588e-05, 0.00064737, 0.030585], B = [0.34492, 0.34498, 0.3464])

#### Best causal-weighted K=4 counterexample (largest effrank reversal)

- A = [0.0019248, 0.030004, 0.062535, 0.088106]: C_cos = 0.30477, C_full = 2.1661, effrank = 4.2221
- B = [0.30066, 0.60274, 0.62319, 0.99176]: C_cos = 0.30559, C_full = 0.68651, effrank = 6.5544
- margins: C_cos(B) - C_cos(A) = 0.00082335;  effrank(B) - effrank(A) = 2.3323  (violation: C_cos(A) < C_cos(B) yet effrank(A) < effrank(B))

Largest C_cos gap among violations with both tables non-degenerate: C_cos(B) - C_cos(A) = 2.9609 (A = [0.00097437, 0.0019396, 0.02684, 0.028158], B = [0.34403, 0.34404, 0.34474, 0.34978])

#### Hand-designed pairs on the causal-weighted grid

| pair | C_cos(A) | C_full(A) | er(A) | C_cos(B) | C_full(B) | er(B) | violates |
|---|---|---|---|---|---|---|---|
| hand A3 vs B3 | 0.54502 | 1.9777 | 2.8084 | 0.45202 | 0.7681 | 4.412 | no |
| hand A4 vs B4 | 0.73598 | 2.8022 | 3.6733 | 0.45275 | 0.76951 | 6.3504 | no |
| hand A6 vs B6 | 1.1028 | 4.4525 | 5.3832 | 0.45494 | 0.77486 | 10.243 | no |

## Part C — Type-2 length reversals (C(A) vs C(B) at L = 128 / 256 / 512)

10000 random table pairs (K = 3, 'mix' family; frequencies fixed at draw time, tables re-evaluated at each length). d = C(A) - C(B); pattern records the signs at 128/256/512.

| metric | pattern counts (128/256/512) |
|---|---|
| C_cos | ---: 3779 | --+: 483 | -+-: 286 | -++: 483 | +--: 459 | +-+: 319 | ++-: 538 | +++: 3653 |
   strongest 128<->256 reversals for C_cos (1547 found):
   - A = [0.0032311, 0.011139, 0.36452], B = [0.34414, 0.35208, 0.66831]: d = (+0.18426, -0.13563, -0.032741)
   - A = [0.28042, 0.29733, 0.52128], B = [0.0087429, 0.15358, 0.16607]: d = (-0.26058, +0.043699, +0.0063633)
   - A = [0.19707, 0.21526, 2.5332], B = [0.049544, 0.060882, 0.39023]: d = (-0.45307, +0.042394, -0.0066883)
   strongest full 3-length reversal for C_cos (605 found): A = [0.19785, 0.21556, 0.75328], B = [0.30132, 0.35101, 0.3615], d = (-0.42226, +0.02314, -0.02122)
| C_full | ---: 3957 | --+: 425 | -+-: 260 | -++: 409 | +--: 367 | +-+: 278 | ++-: 449 | +++: 3855 |
   strongest 128<->256 reversals for C_full (1314 found):
   - A = [0.14744, 0.15273, 1.3028], B = [0.0015725, 0.010862, 0.16572]: d = (-0.092466, +0.10028, -0.049577)
   - A = [0.14789, 0.15329, 1.8229], B = [0.0018114, 0.010852, 0.35909]: d = (-0.093131, +0.082618, -0.052082)
   - A = [0.00040294, 0.014704, 0.24631], B = [0.14962, 0.15768, 0.26409]: d = (+0.092295, -0.057779, -0.023806)
   strongest full 3-length reversal for C_full (538 found): A = [0.14789, 0.15329, 1.8229], B = [0.0018114, 0.010852, 0.35909], d = (-0.093131, +0.082618, -0.052082)

Pairs where BOTH C_cos and C_full reverse in the same direction between 128 and 256: 406
   - A = [0.28042, 0.29733, 0.52128], B = [0.0087429, 0.15358, 0.16607]
     dC_cos = (-0.26058, +0.043699, +0.0063633), dC_full = (-0.25116, +0.044768, +0.0057287)
   - A = [0.27622, 0.29504, 0.40778], B = [0.34759, 0.35954, 0.8614]
     dC_cos = (-0.35694, +0.040271, +0.00023293), dC_full = (-0.34269, +0.04297, +0.00011762)
   - A = [0.19707, 0.21526, 2.5332], B = [0.049544, 0.060882, 0.39023]
     dC_cos = (-0.45307, +0.042394, -0.0066883), dC_full = (-0.37507, +0.038978, -0.0063542)

### Targeted Type-2 constructions

| construction | w0/c | dC_cos(128) | dC_cos(256) | dC_cos(512) | dC_full(128) | dC_full(256) | dC_full(512) |
|---|---|---|---|---|---|---|---|
| wi - wj = 2pi/128 vs pi/128 | 0.3 | -6.7308e-06 | -2.9945e-07 | -5.5581e-07 | -1.0891e-05 | -6.251e-07 | -4.0283e-07 |
| wi - wj = 2pi/128 vs pi/128 | 0.8 | -2.7797e-07 | +8.1683e-08 | -1.1126e-08 | +1.2179e-07 | -8.2789e-08 | -4.1496e-08 |
| wi - wj = 2pi/128 vs pi/128 | 1.4 | +1.9317e-07 | +8.5834e-08 | +2.8516e-08 | +9.3031e-08 | +3.8021e-08 | +7.8611e-09 |
| wi - wj = 2pi/128 vs pi/128 | 2.1 | +3.0308e-08 | +1.9495e-07 | +5.2582e-08 | +2.2998e-07 | +1.0381e-07 | +2.0343e-08 |
| cluster width 2/128 vs 1/128 | 0.2 | -0.93149 | -1.1578 | -0.36114 | -0.91979 | -1.1743 | -0.36398 |
| cluster width 2/128 vs 1/128 | 0.5 | -0.93424 | -1.1781 | -0.36454 | -0.92011 | -1.1744 | -0.36398 |
| cluster width 2/128 vs 1/128 | 0.9 | -0.9221 | -1.1713 | -0.36344 | -0.92009 | -1.1744 | -0.36398 |

## Part D — independent cross-checks of reported counterexamples

gram_direct (symmetric) / loop-summation (causal) recomputes the Gram from scratch; QR-based canon_corr recomputes canonical correlations from basis vectors. min self-eig = smallest eigenvalue over the 2x2 self-blocks (non-degeneracy).

| label | table | errGram | C_cos exact/direct | C_full exact/QR | effrank exact/direct | min self-eig |
|---|---|---|---|---|---|---|
| B-symmetric-K3-er | A: [0.00052403, 0.014597, 0.025767] | 2.84e-14 | 0.928/0.928 | 1.554/1.554 | 3.2944/3.2944 | 3.791e-01 |
| B-symmetric-K3-er | B: [0.02463, 0.032509, 0.065416] | 2.84e-14 | 0.93129/0.93129 | 0.78582/0.78582 | 4.4411/4.4411 | 1.136e+02 |
| B-symmetric-K3-cc | A: [0.00014078, 0.0007724, 0.020622] | 2.84e-14 | 1.1668/1.1668 | 1.9194/1.9194 | 2.9225/2.9225 | 2.738e-02 |
| B-symmetric-K3-cc | B: [0.023778, 0.026778, 0.035966] | 5.68e-14 | 2.3156/2.3156 | 2.0062/2.0062 | 2.9237/2.9237 | 1.178e+02 |
| B-symmetric-K4-er | A: [0.00060202, 0.001626, 0.022113, 0.022927] | 1.14e-13 | 2.0801/2.0801 | 3.5307/3.5307 | 3.1887/3.1887 | 5.002e-01 |
| B-symmetric-K4-er | B: [0.021845, 0.025593, 0.036266, 0.13612] | 1.14e-13 | 2.1012/2.1012 | 1.7759/1.7759 | 4.914/4.914 | 1.125e+02 |
| B-symmetric-K4-cc | A: [0.0016994, 0.0054146, 0.020037, 0.020954] | 1.99e-13 | 2.4531/2.4531 | 3.9218/3.9218 | 2.9878/2.9878 | 3.953e+00 |
| B-symmetric-K4-cc | B: [0.027806, 0.038252, 0.039292, 0.042421] | 1.28e-13 | 4.335/4.335 | 4.122/4.122 | 3.0003/3.0003 | 1.145e+02 |
| hand-K3 | A: [0.00078125, 0.024544, 0.049087] | 1.95e-14 | 0.00014553/0.00014553 | 0.3846/0.3846 | 5.1435/5.1435 | 8.417e-01 |
| hand-K3 | B: [0.8, 0.81688, 1.9] | 1.02e-13 | 0.14631/0.14631 | 0.15112/0.15112 | 5.6975/5.6975 | 1.269e+02 |
| hand-K4 | A: [0.00078125, 0.024544, 0.049087, 0.073631] | 5.68e-14 | 0.000302/0.000302 | 0.41878/0.41878 | 7.0291/7.0291 | 8.417e-01 |
| hand-K4 | B: [0.8, 0.81688, 1.9, 2.4] | 1.56e-13 | 0.14666/0.14666 | 0.15133/0.15133 | 7.6952/7.6952 | 1.269e+02 |
| hand-K6 | A: [0.00078125, 0.024544, 0.049087, 0.073631, 0.098175, 0.12272] | 5.68e-14 | 0.000799/0.000799 | 0.45046/0.45046 | 10.905/10.905 | 8.417e-01 |
| hand-K6 | B: [0.4, 0.8, 0.81688, 1.9, 2.4, 3.1] | 4.72e-13 | 0.14717/0.14717 | 0.15188/0.15188 | 11.692/11.692 | 1.164e+02 |
| B-causal-weighted-K3-er | A: [3.5991e-05, 0.043522, 0.086199] | 3.18e-12 | 0.0044013/0.0044013 | 0.8275/0.8275 | 4.2183/4.2183 | 9.809e-03 |
| B-causal-weighted-K3-er | B: [0.20052, 0.2706, 2.0139] | 1.82e-12 | 0.0044299/0.0044299 | 0.048548/0.048548 | 5.9006/5.9006 | 3.970e+03 |
| B-causal-weighted-K3-cc | A: [5.0588e-05, 0.00064737, 0.030585] | 1.82e-12 | 1.2038/1.2038 | 2.6932/2.6932 | 2.0054/2.0054 | 1.938e-02 |
| B-causal-weighted-K3-cc | B: [0.34492, 0.34498, 0.3464] | 1.82e-12 | 2.9887/2.9887 | 2.9961/2.9961 | 2.0071/2.0071 | 4.034e+03 |
| B-causal-weighted-K4-er | A: [0.0019248, 0.030004, 0.062535, 0.088106] | 9.09e-13 | 0.30477/0.30477 | 2.1661/2.1661 | 4.2221/4.2221 | 2.798e+01 |
| B-causal-weighted-K4-er | B: [0.30066, 0.60274, 0.62319, 0.99176] | 4.55e-13 | 0.30559/0.30559 | 0.68651/0.68651 | 6.5544/6.5544 | 4.021e+03 |
| B-causal-weighted-K4-cc | A: [0.00097437, 0.0019396, 0.02684, 0.028158] | 9.09e-13 | 2.7993/2.7993 | 5.595/5.595 | 2.0652/2.0652 | 7.184e+00 |
| B-causal-weighted-K4-cc | B: [0.34403, 0.34404, 0.34474, 0.34978] | 9.09e-13 | 5.7601/5.7601 | 5.9162/5.9162 | 2.0663/2.0663 | 4.033e+03 |
| C-rev-cos | A: [0.0032311, 0.011139, 0.36452] | 4.62e-14 | 0.89429/0.89429 | 0.94473/0.94473 | 3.9577/3.9577 | 1.394e+01 |
| C-rev-cos | B: [0.34414, 0.35208, 0.66831] | 9.10e-14 | 0.71002/0.71002 | 0.70235/0.70235 | 4.5586/4.5586 | 1.261e+02 |
| C-rev-cos | A: [0.0032311, 0.011139, 0.36452] | 1.14e-13 | 0.066114/0.066114 | 0.4243/0.4243 | 5.0688/5.0688 | 1.013e+02 |
| C-rev-cos | B: [0.34414, 0.35208, 0.66831] | 1.42e-13 | 0.20175/0.20175 | 0.19606/0.19606 | 5.6073/5.6073 | 2.544e+02 |
| C-rev-cos | A: [0.0032311, 0.011139, 0.36452] | 1.14e-13 | 0.0065092/0.0065092 | 0.046539/0.046539 | 5.9064/5.9064 | 4.701e+02 |
| C-rev-cos | B: [0.34414, 0.35208, 0.66831] | 3.98e-13 | 0.03925/0.03925 | 0.038363/0.038363 | 5.9233/5.9233 | 5.102e+02 |
| C-rev-cos-full | A: [0.19785, 0.21556, 0.75328] | 6.73e-14 | 0.12828/0.12828 | 0.11783/0.11783 | 5.7641/5.7641 | 1.252e+02 |
| C-rev-cos-full | B: [0.30132, 0.35101, 0.3615] | 2.84e-14 | 0.55054/0.55054 | 0.54608/0.54608 | 4.8953/4.8953 | 1.258e+02 |
| C-rev-cos-full | A: [0.19785, 0.21556, 0.75328] | 1.09e-13 | 0.050943/0.050943 | 0.047274/0.047274 | 5.9054/5.9054 | 2.541e+02 |
| C-rev-cos-full | B: [0.30132, 0.35101, 0.3615] | 8.53e-14 | 0.027803/0.027803 | 0.028255/0.028255 | 5.9436/5.9436 | 2.547e+02 |
| C-rev-cos-full | A: [0.19785, 0.21556, 0.75328] | 6.12e-13 | 0.0012964/0.0012964 | 0.0016019/0.0016019 | 5.9968/5.9968 | 5.090e+02 |
| C-rev-cos-full | B: [0.30132, 0.35101, 0.3615] | 1.39e-13 | 0.022516/0.022516 | 0.022467/0.022467 | 5.9551/5.9551 | 5.103e+02 |
| C-rev-full | A: [0.14744, 0.15273, 1.3028] | 1.14e-13 | 0.86495/0.86495 | 0.85725/0.85725 | 4.2009/4.2009 | 1.244e+02 |
| C-rev-full | B: [0.0015725, 0.010862, 0.16572] | 2.84e-14 | 0.89698/0.89698 | 0.94971/0.94971 | 3.9522/3.9522 | 3.390e+00 |
| C-rev-full | A: [0.14744, 0.15273, 1.3028] | 1.99e-13 | 0.53656/0.53656 | 0.52117/0.52117 | 4.9453/4.9453 | 2.539e+02 |
| C-rev-full | B: [0.0015725, 0.010862, 0.16572] | 2.89e-14 | 0.050793/0.050793 | 0.42089/0.42089 | 5.0703/5.0703 | 2.662e+01 |
| C-rev-full | A: [0.14744, 0.15273, 1.3028] | 4.50e-13 | 0.025044/0.025044 | 0.024145/0.024145 | 5.9517/5.9517 | 5.091e+02 |
| C-rev-full | B: [0.0015725, 0.010862, 0.16572] | 1.14e-13 | 0.026615/0.026615 | 0.073722/0.073722 | 5.8519/5.8519 | 1.938e+02 |

Canonical-correlation decomposition vs QR path on reported tables: max |s1^2 + s2^2 - (c + s)| = 1.27e-13 (confirms the Part-A identity on every reported counterexample).

## Verdict

Type-1 counterexamples EXIST, on both grids, and are not rare: on the symmetric grid 1.9-5.3% of random table pairs (K = 3, 4; L = 128; 20000 tables per family) have C_cos and effrank ordering each other backwards, rising to 20-25% (conditional on C_cos(B) > C_cos(A)) for the uniform and mixed families on the causal-weighted grid. The most extreme effrank reversals found are 1.73 units at K = 4 on the symmetric grid and 2.33 units at K = 4 on the causal grid (effrank range 0-2K); the largest C_cos gaps among well-conditioned violations are 1.15-2.96, i.e. up to roughly half the K(K-1)/2 collision budget. The cleanest single demonstration is the hand-designed pair: A = {0.1/L, pi/L, 2pi/L, 3pi/L} vs B = {0.8, 0.8+2.16/L, 1.9, 2.4} at L = 128 has C_cos(A) = 3.02e-4 vs C_cos(B) = 0.1467 (a 486x gap) while effrank(A) = 7.029 < 7.695 = effrank(B); in every reported violation C_full(A) > C_full(B), i.e. the full-RoPE metric orders the pair correctly and C_cos gets it backwards because it is blind to the sin channel (the sin components of low-frequency ramps and pi-grid harmonics are strongly correlated while their cos overlaps sit at Dirichlet nodes). Implications: 'lowering C_cos implies better extrapolation' holds as a strong statistical tendency (Spearman(C_cos, effrank) ranges from -0.63 on the uniform/mixed causal-grid families to -0.99 on the log-uniform/pi-grid families) but is NOT a monotone law; any allocation optimized on C_cos alone can hide ~1.7-2.3 units of effrank behind sin-channel redundancy, so the paper's claim should be stated for C_full (or the whitened-Gram effrank directly), with C_cos kept only as a cheap heuristic. Type-2 (length) reversals also exist: 15.5% of random table pairs reverse the C_cos ordering between L = 128 and 256 (13.1% for C_full; 4.1% both; 6.1%/5.4% reverse again at 512) with flip margins up to 0.14 (C_cos) / 0.09 (C_full); however the prescribed targeted constructions barely reverse (2pi/L-vs-pi/L pairs flip only at the 1e-7 scale) and low-frequency cluster-width orderings (1/L vs 1/(2L)) are stable across lengths — so length-monotonicity fails in general but the dominant low-frequency collisions keep their ordering.

## Files created

- counterexamples.py (this search script; numpy only, deterministic, seed 20260819)
- counterexamples_results.md (this report)
