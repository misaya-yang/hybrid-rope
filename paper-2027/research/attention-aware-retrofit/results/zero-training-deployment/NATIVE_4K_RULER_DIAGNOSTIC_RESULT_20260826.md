# Native 4K RULER core-four diagnostic

- **Run:** 2026-08-25/26
- **Status:** complete; descriptive diagnostic, not a paper result
- **Protocol:** released OLMo-2-0425-1B-Instruct, Native RoPE, 20 rows per
  task, official task scorer, nominal length 4096

## Result

| task | 4K score |
| --- | ---: |
| single-key retrieval | 1.00 |
| multikey-2 | 0.85 |
| multikey-3 | 0.60 |
| variable tracking | 0.03 |

The four-task macro is `0.62`. The first three cells were run together; the VT
cell was run separately on the same frozen data root and checkpoint.

## Decision

The proposed reading “low VT at 4K proves a model-capability ceiling” is not
identified by this experiment. RULER cells at different nominal lengths use
different generated inputs, and an existing session-binary policy scores
`0.62` on VT at 8K while another budgeted policy scores `0.03/0.05` at
8K/16K. The 4K value therefore cannot separate task-instance difficulty,
position coding, and policy effects.

The multikey-3 sequence `0.60` at Native 4K, `0.50` at 8K, and `0.00` at 16K
under the existing budgeted policy is consistent with a length-sensitive
failure, but it is likewise descriptive because rows and active tables differ.
A decision experiment must reuse identical token content and change only the
position-ID/phase exposure. No such follow-up was launched in this submission
window.

## Receipts

- three-task result SHA-256:
  `7007bfbb035938e355bed8d9e238bcaa71a0a816bec165f15c59d800a86e3316`
- three-task examples SHA-256:
  `8fb50b2ccd7d36570d8eb7a8af36b8cf13c5e84b9f61dd580155991bfca02599`
- VT result SHA-256:
  `08a2d4e394bba75486749b5175b8d5f05f37a0614bf89f16e8ccf4f5a48ba9b6`
- VT examples SHA-256:
  `23f905daba6a90da11c08c66f5125779f29e16385d4a23170bed96e59db3846a`

Raw rows and machine paths remain outside the anonymous repository.
