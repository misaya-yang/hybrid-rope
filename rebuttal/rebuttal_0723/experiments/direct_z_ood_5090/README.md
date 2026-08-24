# Direct fixed-support `z` retrofit on OLMo-2 1B

This is the current GPU candidate for the third-axis question. It does not
train model weights. It calibrates 63 positive-gap logits (62 effective simplex
degrees of freedom) for one RoPE table while keeping the released Native
fast/slow endpoints and log span exactly fixed.

## Scientific question

For a frozen mature checkpoint, can one fixed table improve actual 2x
natural-text OOD NLL while keeping 1x NLL within a predeclared no-harm margin?

The optimized object is the model loss, not an attention prior, a collision
score, or a Cosh surrogate. Collision/alias measurements may explain a result
afterward, but cannot select the table. This isolates the allocation coordinate
`z` within Native support; it does not claim that support and allocation are
globally separable in language-model quality.

## Identification and limits

- The same table and attention scaling `1.0` are used at 1x and 2x. There is no
  length-conditioned branch and no request-time route.
- Rows 0--1 of the four-document 8K FineWeb tensor are design rows. Rows 2--3
  are untouched held-out gates. This is a cheap candidate screen, not paper
  evidence by itself.
- Passing requires a non-Native table, no held-out 1x row worse than Native by
  more than `0.05` on final-64-token teacher-forced tail NLL, lower mean
  held-out 2x tail NLL, and no held-out 2x row worse by more than `0.05`.
- "Zero-weight" means zero model-weight updates. The table has 63 calibrated
  gap logits and 62 effective allocation degrees of freedom, so this is neither
  zero-search nor a zero-learned-parameter construction.
- The pilot is model-relative at ratio 2. It avoids an absolute `L_target`, but
  it does not establish one finite table as optimal for all future horizons.
- Novelty is the explicit fixed-support interior-allocation intervention and
  retrofit method. It is not a claim that non-geometric RoPE tables themselves
  are new.

## GPU sequence and stopping

Every command requires a fresh explicit run authorization through
`DIRECT_Z_RUN_AUTHORIZED=YES`; the optimizer additionally enforces its own
authorization flag. Use the 5090 32GB instance unless the one-step smoke shows
that the frozen-weight backward pass does not fit.

1. `optimize-smoke`: one optimizer step; runtime/identity only, never
   promotable.
2. `optimize`: ten steps. Stop immediately if its held-out scientific gates
   fail.
3. `pg19-screen`: five rows per 1x/2x cell with the exact winning table.
4. `ruler-screen`: five rows per 8K task. Inspect both screens before setting
   `DIRECT_Z_PROMOTE=YES`.
5. `formal-natural`: held-out Qasper/2Wiki at 1x and 2x, 20 rows per cell.
6. `qasper-full`, then `2wiki-full`: only after the formal screen is worth the
   cost.

The launch order deliberately places branch-local LoRA outside this package.
Such a follow-up is justified only if this calibrated direct-`z` candidate
survives 2x OOD evaluation and the remaining 1x gap plausibly requires
co-adaptation. This condition does not govern the repository's completed
zero-training Native/s4 policy or separate adaptation owners. No result file
may be interpreted without its table hash, checkpoint receipt, and method
receipt.
