# Mature OLMo fixed-support co-adaptation oracle

- **Date:** 2026-08-25
- **Status:** completed internal mechanism study
- **Model:** released OLMo-2-0425-1B-Instruct (`1.485B`)
- **Decision:** the learned allocation produces a reproducible long-tail
  redistribution after matched natural-LM adaptation, but it does not dominate
  the Native table on full-sequence NLL or full-200 2Wiki. Stop this oracle and
  do not add shells, steps, seeds, or an allocation-LR sweep.

## Question and registered run

The run starts exactly at the Native RoPE table, fixes its sampled support
endpoints, and jointly learns one layer/head-shared interior allocation and
rank-64 Q/K LoRA. Physical inputs remain 4K. Two of every three optimizer steps
use natural-span answer supervision with deterministic phase-gap shells; the
third uses sparse 4K natural-LM replay. No target table or single target length
is used by the construction.

The registered 32-row result passes the 4K retention and zero-offset gates but
fails the all-shell gate:

| Endpoint | Final minus Native NLL |
| --- | ---: |
| 4K raw retention | `+0.03313` |
| phase offset 0 | `-0.44101` |
| phase offset `L` | `-0.15835` |
| phase offset `3L` | `+0.00837` |
| phase offset `15L` | `-0.04881` |

Both Q/K and allocation gradients are present, finite, and nonzero. The table
keeps exact Native endpoints and moves by at most `0.001298` of the sampled
log-support span. The registered gate is therefore a negative result; the
larger analyses below are post-run attribution, not replacements for it.

## Frozen 128-row attribution

The final table and adapter were recombined without retraining. Deltas below
are relative to Native table / LoRA off on all 128 held-out phase rows.

| Frozen cell | offset 0 | `L` | `3L` | `15L` | dense 4K |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned table / LoRA off | `+0.0367` | `+0.0779` | `-0.2133` | `+0.2732` | `+0.0012` |
| Native table / LoRA on | `-0.4570` | `-0.1494` | `-0.0125` | `-0.2136` | `+0.0324` |
| learned table / LoRA on | `-0.4495` | `-0.1033` | `-0.2605` | `-0.0372` | `+0.0337` |

Thus the phase-task bundle is mainly explained by Q/K adaptation. The table is
not null, however: it reallocates the shell response toward `3L`, at a cost at
`L` and `15L`. The 32-row `3L` sign and the 128-row sign differ, so no
uncertainty estimate or mechanism claim is inferred from the smaller/larger
mean discrepancy without per-row paired statistics.

## Physical continuous natural text

On four hash-bound FineWeb-Edu documents at each physical length, frozen
table-only deltas versus Native are:

| Length | full NLL | final-512 NLL |
| --- | ---: | ---: |
| 8K | `+0.04256` | `-0.05446` |
| 16K | `+0.01988` | `-0.05868` |

The small fixed-support movement therefore has a measurable effect outside the
phase proxy: it improves both long-tail endpoints while degrading mean
full-sequence NLL. The phase-trained Q/K LoRA does not transfer to natural LM;
it worsens full and tail NLL under the Native table.

## Dense-natural recovery and matched control

To separate a narrow proxy objective from coordinate co-adaptation, two new
Q/K-only runs use identical initialization, data order, 300-step budget, and
dense 4K next-token loss. One freezes the learned table; the other freezes the
Native table. The 896 training and 120 held-out FineWeb-Edu documents are
disjoint, and all eight physical 8K/16K evaluation documents are excluded from
training. Neither run uses a target extrapolation length.

The self-consistent learned-table model minus the self-consistent Native-table
model is:

| Endpoint | Delta NLL |
| --- | ---: |
| held-out 4K full | `+0.00098` |
| physical 8K full | `+0.03844` |
| physical 8K final-512 | `-0.03866` |
| physical 16K full | `+0.02190` |
| physical 16K final-512 | `-0.08767` |

Both self-consistent runs improve all five NLL endpoints versus the unadapted
Native checkpoint; that shared gain belongs to the matched dense-LM continued
adaptation, not allocation. Allocation's identified marginal effect is a
near-zero 4K cost and a long-sequence full/tail redistribution. This is direct
evidence that mature weights can co-adapt to the table without retaining its
short-window hard-swap cost, but it is not a jointly dominating allocation.

## Capability boundary

A 20-row-per-length 2Wiki screen initially favored the learned-table model.
The frozen comparison was therefore extended, without changing either model,
to all 200 official LongBench 2Wiki rows. Learned minus Native self-consistent
results are:

| Budget | token-F1 | normalized exact |
| --- | ---: | ---: |
| 4K | `-0.00654` | `-0.005` |
| 8K | `+0.00129` | `+0.005` |
| 16K | `+0.00089` | `+0.005` |

These are effectively tied at this endpoint. The tail-NLL gain has not been
shown to improve downstream capability.

## Scientific conclusion

This study establishes three bounded facts:

1. the original answer-only phase objective assigns most bundle improvement
   to Q/K LoRA and is not an adequate method objective;
2. a very small target-free fixed-support allocation change transfers to
   physical 8K/16K tail NLL and survives matched dense-natural co-adaptation;
3. the same change trades full-sequence NLL for tail NLL and does not separate
   from the Native control on full-200 2Wiki.

The result does not validate a universal learned schedule, a zero-training
retrofit, or the paper's current EVQ-Cosh construction. It also does not show
that allocation is noise. The next method must explain and improve this
full/tail redistribution rather than rerunning the present phase proxy or
optimizing a known factor-specific/routed table.

