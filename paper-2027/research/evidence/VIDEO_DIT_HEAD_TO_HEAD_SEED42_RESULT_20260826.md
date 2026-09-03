# Video-DiT seed-42 head-to-head result

- **Status:** `COMPLETE`
- **Canonical machine-readable owner:**
  [`../../../data/curated/video_dit_seed42_head_to_head_20260826.json`](../../../data/curated/video_dit_seed42_head_to_head_20260826.json)
- **Role:** supporting cross-modal evidence for finite temporal-frequency
  allocation; not a fourth lifecycle route or a standalone video claim.

## Scientific contract

The experiment compares geometric allocation (`tau=0`) with EVQ-Cosh
(`tau=1.5`) in one seed-42 head-to-head training protocol for a 129.6M video
diffusion transformer with bidirectional attention and 3D RoPE. Both arms use
Oscillating Moving MNIST, 32 training frames, 128 evaluation frames, 15,000
steps, temporal base 10,000, and the same temporal range transform. The
denoising endpoint uses noise level 0.5 and 256 evaluation videos.

## Result

Geo/EVQ-Cosh denoising MSE is `0.00911327/0.00720361` on training frames,
`0.00724399/0.00606377` over all extrapolated frames, and
`0.00989095/0.00638799` on far extrapolated frames. The corresponding
EVQ-minus-Geo relative changes are `-20.95%`, `-16.29%`, and `-35.42%`.

## Raw receipt

The curated JSON records the three repository-relative raw paths and SHA-256
hashes used to extract these numbers. It contains no second-seed result. The
maximum manuscript claim is therefore one matched seed-42 comparison that
supports modality breadth; it does not quantify training-seed uncertainty or
establish a universal operating rule.
