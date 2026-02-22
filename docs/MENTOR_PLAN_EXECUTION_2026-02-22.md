# Mentor Plan Execution Summary (2026-02-22)

## Task 1: Ediag exact vs affine
- d=128, L=16384
- b=1e4: B_fit=0.474639, B_theory=0.474561, R2_mid=0.9942
- b=5e5: B_fit=0.672376, B_theory=0.676127, R2_mid=0.9954
- outputs: `results\theory_2026-02-22\ediag_exact_vs_affine.pdf`, `results\theory_2026-02-22\ediag_exact_vs_affine.png`

## Task 2: Phase transition family (L/b in {1.6,10,100,1000})
- L/b=1.6: p*=0.000000, exact_crossing=False, delta(p=0)=-2.303741e-03, delta(p=1)=-0.055607
- L/b=10.0: p*=0.000000, exact_crossing=False, delta(p=0)=-9.052562e-05, delta(p=1)=-0.047139
- L/b=100.0: p*=0.000755, exact_crossing=True, delta(p=0)=2.992713e-05, delta(p=1)=-0.039589
- L/b=1000.0: p*=0.000008, exact_crossing=True, delta(p=0)=2.640617e-07, delta(p=1)=-0.034131
- outputs: `results\theory_2026-02-22\phase_transition_lb_scan_delta.pdf`, `results\theory_2026-02-22\phase_transition_lb_scan_scores.pdf`

## Task 3: 124M Geo(base=100k), 4K train, target 3K steps
- train_hours=0.868, best_step=1900, best_val=0.018840, best_ppl=1.019019
- PPL by length (Standard): L1024=1.021, L2048=1.023, L4096=1.018, L8192=1.037, L16384=1.052, L32768=1.233

## Task 4: 50M base ablation add-on (base=200k/300k, standard vs sigmoid)
- base200k: L16K standard=1.214, sigmoid=1.177; L32K standard=2.071, sigmoid=1.493
- base300k: L16K standard=1.035, sigmoid=1.129; L32K standard=1.177, sigmoid=1.741

## Notes
- Task 3 crash root cause was Windows overwrite-lock on same best checkpoint filename; fixed by unique best checkpoint filenames and robust loader.
- These add-on runs used current local Phase-4 pipeline settings; keep protocol tags when merging with historical A100 tables.
