"""The prediction the regime taxonomy actually makes -- and it is already in your data.

regime II (L < lambda <= M) is the band that gives MONOTONE, UNAMBIGUOUS distance
resolution.  Distinguishing ONE source needs little of it.  Distinguishing SEVERAL
sources at several distances simultaneously needs a lot of it.

EVQ tau=2 thins regime II from 10 to 8 channels and its density to 0.77x.
=> prediction: in-window damage should concentrate on MULTI-key / MULTI-value tasks
   and spare single-needle tasks.

Data: OLMO2_1B_SELECTIVE_QK_PHASE_ADAPTATION_20260729.md, 13-family RULER official
score at 4K, both arms Q/K-phase-adapted from their own matched Stage-A parent,
identical 1:1:2 phase exposure, 300 steps.  Only variable = the frequency table.
"""
fams = [   # family, native_4K, evq_4K,  n_keys_to_disambiguate
    ("niah_single_1", 100.0,  95.0, 1), ("niah_single_2", 100.0, 100.0, 1),
    ("niah_single_3", 100.0,  40.0, 1), ("niah_multikey_1", 90.0, 25.0, 2),
    ("niah_multikey_2", 95.0,  0.0, 3), ("niah_multikey_3", 50.0,  0.0, 4),
    ("niah_multivalue", 93.75, 67.5, 2), ("niah_multiquery", 90.0, 65.0, 2),
    ("vt",             64.0,  59.0, 2), ("cwe",             29.0,  8.5, 3),
    ("fwe",            46.67, 56.67, 1), ("qa_1",           50.0, 15.0, 1),
    ("qa_2",           30.0,  20.0, 1),
]
print(f"{'family':<18s} {'Native 4K':>10s} {'EVQ 4K':>8s} {'delta':>8s} {'multiplicity':>13s}")
print("-" * 62)
single, multi = [], []
for f, n, e, k in sorted(fams, key=lambda r: r[2] - r[1]):
    d = e - n
    print(f"{f:<18s} {n:>10.2f} {e:>8.2f} {d:>+8.2f} {k:>13d}")
    (multi if k >= 2 else single).append(d)
print("-" * 62)
m = lambda x: sum(x) / len(x)
print(f"mean delta, single-source families (k=1, n={len(single)}): {m(single):+7.2f} pp")
print(f"mean delta, multi-source  families (k>=2, n={len(multi)}): {m(multi):+7.2f} pp")
print(f"gap: multi-source families are hurt {m(multi)-m(single):+.2f} pp more")
print()
print("Same arms at 8K (where reach, not resolution, binds):")
print("  Native non-zero in 4/13 families;  EVQ non-zero in 10/13.")
print()
print("=> the in-window cost is NOT uniform.  It concentrates exactly where simultaneous")
print("   multi-distance discrimination is required -- i.e. on regime II.  2Wiki 4K exact")
print("   is 22.0% Native vs 21.5% EVQ (-0.5pp, near parity) in the SAME experiment.")
print("   'EVQ destroys in-window' is false as stated; 'EVQ trades multi-key resolution")
print("   for single-source reach' is what the data says.")
