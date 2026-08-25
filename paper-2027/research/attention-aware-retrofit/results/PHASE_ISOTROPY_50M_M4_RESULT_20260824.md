# Phase-isotropy 50M M4 screen (2026-08-24)

- Canonical verdict: **SCREEN_UNRESOLVED**
- Generated raw gate string: `FAILED_50M_GATE`
- Scope: internal method screening only; no manuscript claim.
- Model: existing 50M M4 harness, `L_train=256`, `d_head=64`, `K=32`, seed `137`.
- Main metric: final-128-token teacher-forced NLL; full NLL retained below.

## NLL by length

| Arm | 256 tail/full | 512 tail/full | 1024 tail/full | 2048 tail/full |
|---|---:|---:|---:|---:|
| FMRoPE | 5.952067/6.001411 | 6.184568/6.088442 | 6.560358/6.310329 | 6.343818/6.222811 |
| anchored EVQ-Cosh | 5.961192/6.010739 | 6.171763/6.084532 | 6.593057/6.323509 | 6.347251/6.249514 |
| phase-isotropy | 6.034230/6.074980 | 6.243165/6.148720 | 6.625052/6.367433 | 6.399968/6.286090 |

## Paired deltas (negative means the named arm is better)

```json
{
  "256": {
    "anchored EVQ-Cosh_minus_FMRoPE_tail_nll": 0.009124517440795898,
    "phase-isotropy_minus_FMRoPE_tail_nll": 0.08216285705566406,
    "anchored EVQ-Cosh_minus_FMRoPE_full_nll": 0.00932776927947998,
    "phase-isotropy_minus_FMRoPE_full_nll": 0.07356882095336914
  },
  "512": {
    "anchored EVQ-Cosh_minus_FMRoPE_tail_nll": -0.012804985046386719,
    "phase-isotropy_minus_FMRoPE_tail_nll": 0.05859708786010742,
    "anchored EVQ-Cosh_minus_FMRoPE_full_nll": -0.003910183906555176,
    "phase-isotropy_minus_FMRoPE_full_nll": 0.06027793884277344
  },
  "1024": {
    "anchored EVQ-Cosh_minus_FMRoPE_tail_nll": 0.03269851207733154,
    "phase-isotropy_minus_FMRoPE_tail_nll": 0.06469404697418213,
    "anchored EVQ-Cosh_minus_FMRoPE_full_nll": 0.01318049430847168,
    "phase-isotropy_minus_FMRoPE_full_nll": 0.05710458755493164
  },
  "2048": {
    "anchored EVQ-Cosh_minus_FMRoPE_tail_nll": 0.0034328699111938477,
    "phase-isotropy_minus_FMRoPE_tail_nll": 0.05615043640136719,
    "anchored EVQ-Cosh_minus_FMRoPE_full_nll": 0.02670300006866455,
    "phase-isotropy_minus_FMRoPE_full_nll": 0.0632789134979248
  }
}
```

## Weighted OOD and gate

- Weighted tail OOD FMRoPE: `6.379198900933716`; anchored EVQ-Cosh gain: `-0.00939831086152143`; phase-isotropy gain: `-0.059501590590705256`; retention: `None`.
- Checks: `{"all_three_complete": true, "phase_256_tail_delta_le_0.01": false, "phase_ood_tail_all_better": false, "phase_retains_80pct_cosh_gain_when_cosh_gain_positive": true}`
- Frequency float32 hashes: `{"FMRoPE": "06adcd404637282636289e77fa01c56630cb4945332c96d5fa2d9239fe0d6b04", "anchored EVQ-Cosh": "e0b201711857c01b85933b4f8618929f5124eee1827b014c4d9b03a47e48a0eb", "phase-isotropy": "dd6b63780239b9b9d0876a2e938217c64189a663d8efea099cbb376cd24d94c9"}`
- Training/data/init/anchor matching: `{"anchors": {"eval_chunks": 4, "offsets": {"1024": [750073, 941884, 950595, 1014394], "2048": [276752, 368333, 460830, 910230], "256": [54579, 150165, 429431, 1006598], "512": [167091, 224328, 305775, 667755]}, "offsets_sha256": "ac5d58952337de48f457e8fb620aba0899fb70d00ee552520f03a89790f3cb9b", "seed": 9999, "val_prefix_tokens": 1048576}, "data": {"dataset": "local_wikitext", "row_order": "deterministic randint(seed*1000003 + global_micro_step) over fixed sequential train rows", "train_prefix_sha256": "014ef2645a7bff02fff9d1f9d60f575d380d63a9086294a59d289bb863878ce6", "train_prefix_tokens": 8388608, "val_prefix_sha256": "0d57f15a0896aa65cec0be53dfedbe0ad683c9f59ec95c4185411901d8aa3c87", "val_prefix_tokens": 1048576}, "device": "mps", "dtype": "torch.float32", "frequency_float32_sha256": {"FMRoPE": "06adcd404637282636289e77fa01c56630cb4945332c96d5fa2d9239fe0d6b04", "anchored EVQ-Cosh": "e0b201711857c01b85933b4f8618929f5124eee1827b014c4d9b03a47e48a0eb", "phase-isotropy": "dd6b63780239b9b9d0876a2e938217c64189a663d8efea099cbb376cd24d94c9"}, "global_batch_size": 256, "grad_accum": 8, "matched_initialization_sha256": {"FMRoPE": "20fd1d517a0a70e4b1a99865c49e115f6bc00aa2245821bbee6353d015d5210b", "anchored EVQ-Cosh": "20fd1d517a0a70e4b1a99865c49e115f6bc00aa2245821bbee6353d015d5210b", "phase-isotropy": "20fd1d517a0a70e4b1a99865c49e115f6bc00aa2245821bbee6353d015d5210b"}, "micro_batch_size": 32, "model": {"K": 32, "head_dim": 64, "hidden_size": 512, "num_heads": 8, "num_layers": 6, "tier": "50m"}, "optimizer": "AdamW(lr=6e-4, betas=(0.9,0.95), weight_decay=0.1), cosine 0.1 floor, 5% warmup", "seed": 137, "train_tokens": 8388608}`
- Runtime seconds: `{"FMRoPE": 608.74, "anchored EVQ-Cosh": 639.19, "phase-isotropy": 648.71}`
- Peak MPS memory: `{"FMRoPE": {"peak_current_allocated_bytes": 6548781568, "peak_current_allocated_mib": 6245.40478515625, "peak_driver_allocated_bytes": 14639316992, "peak_driver_allocated_mib": 13961.140625, "sample_count": 1201}, "anchored EVQ-Cosh": {"peak_current_allocated_bytes": 9201905408, "peak_current_allocated_mib": 8775.620849609375, "peak_driver_allocated_bytes": 14646099968, "peak_driver_allocated_mib": 13967.609375, "sample_count": 1256}, "phase-isotropy": {"peak_current_allocated_bytes": 5786993920, "peak_current_allocated_mib": 5518.907470703125, "peak_driver_allocated_bytes": 14646099968, "peak_driver_allocated_mib": 13967.609375, "sample_count": 1274}}`
- Anomalies: `[]`

The finite one-seed screen is not a pooled effect, not a continuous optimum, and
not evidence for deployment or paper promotion. Its preflight routed a neutral
or negative anchored-Cosh reference to an unresolved regime, so the generated
gate failure does not reject phase-isotropy beyond this tested protocol.
