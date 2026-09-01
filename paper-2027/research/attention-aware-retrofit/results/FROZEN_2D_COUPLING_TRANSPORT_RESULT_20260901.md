# Frozen 2D coupling transport: matched scale and K128 holdout (2026-09-01)

## Material passport

- **Status:** `P2_COMPLETE / P1_K128_SCREEN_UNRESOLVED_LONG_NEGATIVE /
  P3_REJECTED_BEFORE_GPU / STRETCH_NOT_OPENED`
- **Method:** frozen two-parameter clipped-affine `G(x)`; zero model updates,
  zero boundary refits, one static table for the full request
- **Frozen parameters:** `x_H=0.7382780681078285`,
  `x_L=0.366403835112904`, `c=0.074`
- **Scale tested:** matched `s=2`, with attention scaling
  `1.0512928913614359`
- **K32 owner:**
  [`K32_FINITE_K_COUPLING_ANALYSIS_20260901.md`](K32_FINITE_K_COUPLING_ANALYSIS_20260901.md)
- **Compact receipt:**
  [`../evidence/FROZEN_2D_COUPLING_TRANSPORT_RECEIPT_20260901.json`](../evidence/FROZEN_2D_COUPLING_TRANSPORT_RECEIPT_20260901.json)

## 1. Decision

P2 resolves the scale confound but does not rescue K32 Native compatibility.
Changing only `s=4 -> s=2` moves physical `x` from `.5125/.4350` to
`.5225/.5050` at 32K/64K. Correct scale therefore strengthens the long
backbone by `.0700`, while 32K retention remains only `.80695`, below the
`.875` gate.

The matched normalized-index control gives the complementary operating point:
`.5775/.4375` at 32K/64K. It passes Native while physical `x` wins long by
`.0675`. The result is a real Pareto crossing, not uniform physical-coordinate
dominance:

> frozen physical `x` is the stronger K32 long backbone; normalized index is
> the more Native-compatible representative.

P1 reaches a bounded mixed result on K128. The exact public
`unsloth/gemma-2b-it` artifact has standard full attention, `K=128`, Native 8K,
and no built-in RoPE scaling. Native scores zero on every registered core-four
cell. All three frozen tables recover nonzero 8K task behavior, but every arm
remains zero at 16K. Physical `x`, normalized index, and wrong-`c_orth` are
within `.0275` at 8K; the wrong normalization is nominally best. This screen
does not support a privileged physical coordinate at K128 and does not support
2x long transport on this checkpoint. Because model family changes with K, it
is not a causal rejection of K128 or of the continuous law.

The same-architecture Gemma-1.1 follow-up strengthens the screen rather than
rescuing long transport. Physical/index reach `.8075/.8225` at 8K, proving the
rows and checkpoint can resolve the registered tasks under a static-table
intervention, but both remain exactly zero at 16K. Physical and index are again
inside the `.05` parity band. The replicated 16K negative is material; the
checkpoint-family confounding still caps it at a screen, not a universal-law
verdict. A completed control audit shows Native reaches `.9050` at 4K, Native
frequencies with matched gain remain zero at 8K/16K, and physical `x` without
gain reaches `.8350` at 8K but remains zero at 16K. The model and harness can do
the tasks; table geometry—not gain-only—causes the 8K recovery.

P3 is rejected before GPU: the only geometry-defined cell-average direction
fails its CPU entrance condition and is not a long-null direction. No `alpha`,
shifted control, or Native calibration sweep was created. Stretch s8 was not
opened because P1 did not satisfy its long-side entrance gate.

## 2. Frozen construction and reusable code

For every admitted checkpoint,

\[
x_i=\log\!\frac{L_{\rm Native}\omega_i}{2\pi c_{\rm orth}},
\qquad
c_{\rm orth}=\frac{1}{1-b^{-1/K}},
\]

\[
G(x)=\operatorname{clip}\!\left(
\frac{x_H-x}{x_H-x_L},0,1
\right),qquad
\omega_i'=\omega_i s^{-G(x_i)}.
\]

[`scripts/analysis/export_frozen_coupling_transport.py`](../../../../scripts/analysis/export_frozen_coupling_transport.py)
now exports three zero-refit arms from a checkpoint config and a runtime-bound
Native tensor:

1. `dimensionless_x`: the physical coordinate above;
2. `normalized_raw_index`: frozen `G` point samples on the counterfactual K64
   grid with the target checkpoint's `b,L`, transported by `i/(K-1)`;
3. `wrong_source_c_orth`: the target K table evaluated with its counterfactual
   K64 `c_orth`, an asymmetric negative control.

The exporter rejects partial-head RoPE, pre-scaled checkpoints, invalid or
crossed tables, and records config, runtime Native, tensor, file, and movement
hashes. [`scripts/eval/run_frozen_coupling_k_transport.sh`](../../../../scripts/eval/run_frozen_coupling_k_transport.sh)
binds the checkpoint weight set, runtime Native tensor, data manifest, model
type, K, static-table hash, gain, and full request lifetime before evaluation.

The first K128 preflight correctly failed because the config-formula float32
Native tensor differed from Transformers' realized Gemma tensor in 42 slots by
at most `5.96e-8`. No benchmark row ran under that identity. The final tables
were rebuilt from the runtime initializer and bound to Native tensor SHA-256
`cc63341a0ac42a60b986ed638fd0d45b838b72fabeffec059c463eac4ed9ea15`.

## 3. P2 — K32 matched-s2 scale consistency

All rows reuse the earlier paired Qwen2.5-0.5B-Instruct K32 data. Neither the
physical nor index table read a K32 score during construction.

| Static profile | 32K macro | Native retention | 64K macro | 64K minus Native |
| --- | ---: | ---: | ---: | ---: |
| Native | `.6475` | `1.0000` | `.2775` | -- |
| physical `x`, s4 | `.5125` | `.7915` | `.4350` | `+.1575` |
| physical `x`, matched s2 | `.5225` | `.80695` | **`.5050`** | **`+.2275`** |
| normalized index, s4 | `.5350` | `.8263` | `.4275` | `+.1500` |
| normalized index, matched s2 | **`.5775`** | **`.89189`** | `.4375` | `+.1600` |

The result supports scale-sensitive behavior generated by one frozen `G`, not
an s4-only `G_4`. It does not establish a single uniformly best K32 table:
physical `x` trades `.0550` Native macro for `.0675` long macro relative to
index.

## 4. P1 — K128 transport screen

### 4.1 Checkpoint and geometry

The primary artifact is exactly
`unsloth/gemma-2b-it@5ae754dee0b5cbf7f7fb39a7731aefa6b7987c2d`,
weight SHA-256
`8fdf067bdfd010c75d8c4c0508fe45f4567c6097f3193928501d63a05020e0e6`.
It is a public unquantized redistribution; this study does **not** claim tensor
identity with the gated Google canonical repository or redistribute weights.

Its audited config is Gemma-1 full attention, `head_dim=256`, `K=128`, one KV
head, `L_Native=8192`, `b=10000`, no partial RoPE, no sliding window, and no
RoPE scaling. The frozen transition has `eta=5.168`, with physical non-binary
slots 53--57; normalized index uses 61--68 and wrong-`c_orth` uses 62--66.
The data manifest SHA-256 is
`45c77d00778d59f83d75dfc25fd0e14d38a3a98f0b31083230b9ff53c47b3542`;
all cells use official RULER commit
`c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`, seed `20260822`, and 20 rows.

### 4.2 Primary screen

| Table | 8K single / mk2 / mk3 / VT | 8K macro | 16K single / mk2 / mk3 / VT | 16K macro |
| --- | --- | ---: | --- | ---: |
| Native | `.00/.00/.00/.00` | `.0000` | `.00/.00/.00/.00` | `.0000` |
| physical `x` | `.45/.30/.15/.67` | `.3925` | `.00/.00/.00/.00` | `.0000` |
| normalized index | `.45/.15/.15/.72` | `.3675` | `.00/.00/.00/.00` | `.0000` |
| wrong `c_orth` | `.45/.20/.15/.78` | **`.3950`** | `.00/.00/.00/.00` | `.0000` |

At 8K, physical minus index is only `+.0250`, inside the predeclared `.05`
parity band; wrong-`c_orth` minus physical is `+.0025`. At 16K every arm is
zero. Therefore:

- the tables have a real checkpoint effect—the Native row is not simply copied;
- the physical coordinate is not distinguished from ordinal or wrong-normalized
  controls;
- no arm establishes useful exact-2x behavior;
- the screen is `SCREEN_UNRESOLVED` for cross-K coordinate identification,
  with a material negative 16K endpoint.

### 4.3 Same-architecture instruction checkpoint follow-up

To determine whether the zero Native control was specific to the initial
instruction tuning, the same frozen K128 tables and tokenizer-identical paired
rows were prepared for exact
`unsloth/gemma-1.1-2b-it@619e546640669a280738627cc623f4bd74a7b069`.
The exact weight SHA-256 is
`584d0f7d939d235ee14a4ba307b40dbc3f03d5483181b9381e9f10636b618933`.
Its config differs in the instruction-tuning activation but preserves the same
Gemma-1 architecture, K128 geometry, tokenizer, and table tensors. Independently
prepared data have byte-identical cell hashes to the first screen.

| Table | 8K single / mk2 / mk3 / VT | 8K macro | 16K single / mk2 / mk3 / VT | 16K macro |
| --- | --- | ---: | --- | ---: |
| Native | `.00/.00/.00/.00` | `.0000` | `.00/.00/.00/.00` | `.0000` |
| physical `x` | `1.00/1.00/.55/.68` | `.8075` | `.00/.00/.00/.00` | `.0000` |
| normalized index | `1.00/1.00/.55/.74` | **`.8225`** | `.00/.00/.00/.00` | `.0000` |

Index exceeds physical by only `.0150`, again inside parity. The stronger 8K
response rules out “the first instruction checkpoint alone was too weak to
express table effects.” Neither K128 screen supports exact-2x transport.

### 4.4 Download, gain, and loader-path controls

`-it` is the instruction-tuned artifact and was the intended checkpoint type;
there was no base/instruct mix-up. The following Gemma-1.1 controls use the
same weights, tokenizer, Flash backend, scorer, and paired rows:

| Frequencies | Gain | 4K macro | 8K macro | 16K macro | Role |
| --- | ---: | ---: | ---: | ---: | --- |
| Native | `1.0` | **`.9050`** | `.0000` | `.0000` | exact checkpoint baseline |
| Native | `1.0513` | -- | `.0000` | `.0000` | gain-only |
| physical `x` | `1.0` | -- | **`.8350`** | `.0000` | table-only |
| physical `x` | `1.0513` | -- | `.8075` | `.0000` | registered combined arm |

The 4K Native vector is `1.00/.95/.85/.82`; the model plainly performs the
downstream tasks. Gain-only is insufficient, while the physical table alone is
sufficient for strong 8K recovery and is `.0275` better than the gained table.
Gain is mildly harmful at this Gemma operating point.

A `Native frequencies + gain=1` identity table was also passed through
`external_table_static` on one fixed row per task/length. All eight generated
strings and scores exactly equal the original `method=native` outputs. This
closes the Native-versus-external loader-path confound at the tested canaries.
The remaining pattern is behavioral: the frozen table extends strong
core-four behavior from 4K to 8K but does not extend it to 16K.

This pattern is consistent with config `max_position_embeddings=8192` being an
implementation allowance rather than this checkpoint's task-level operating
length. It does not prove that pretraining used 4K or that one task-independent
`L_eff` exists. A future reference length must be frozen by Native-only,
task-family-diverse calibration before any long holdout; choosing it to make
RULER pass would be target tuning.

## 5. P3 — Native boundary correction precheck

The only direction uniquely fixed by existing geometry is

\[
B_{32}=m^{\rm cell}-P_{32}[G],
\]

with nonzero components `B_16=+0.0204871` and `B_17=-0.00140593`.
It preserves table validity, but fails every scientific entrance condition:

- it leaves the OLMo fast-side slot 19 shoulder exactly zero;
- OLMo movement RMSE/MAE worsen from `.006118/.001223` to
  `.006706/.001531`;
- it preserves neither discrete total movement nor first moment, so there is
  no basis to call it a long-null direction;
- OLMo and Qwen residual maxima do not define one cross-checkpoint geometric
  direction.

Any dipole, shifted boundary derivative, or neighboring-slot direction would
therefore be chosen after outcomes. P3 is `REJECT_P3`: no scalar `alpha`, no
Native sweep, no shifted GPU control, and no long holdout was opened.

## 6. Stretch and SOTA boundary

The s8 stretch gate required P1 physical `x` to improve the matched-s K128 long
endpoint without losing to index by more than `.05`. The K128 16K rows are all
zero, so the gate fails. No s8 C2 table was launched and no hierarchical or
32/32 composite law is authorized.

The current method cannot be called global, engineering, or RULER SOTA. The
strongest supportable claim is bounded:

> On the tested K64 and K32 checkpoints, one frozen two-parameter physical
> coordinate transports a strong long-context backbone; matched scale improves
> it, but Native compatibility is checkpoint-dependent. A K128 screen did not
> confirm privileged-coordinate or 2x generalization.

The next matched static baseline is deterministic
[Resonance RoPE](https://aclanthology.org/2024.findings-acl.32/), whose
published wavelength rounding adds no online cost. It belongs in a later
same-checkpoint baseline panel, not in this already mixed K128 identification
matrix. No claim is made against search-based or custom-kernel methods.

## 7. Stopping decision

1. Keep frozen `G(x)` as a supported K64/K32 long-backbone hypothesis, not a
   universal law.
2. Do not introduce `G(x;K)`: K128 changes checkpoint, base, and behavioral
   length together, while K32 matched-s long behavior is strong.
3. Do not open P3 until a benchmark-independent invariant uniquely derives a
   long-null Native correction direction.
4. Do not open s8 or hierarchical composition from this run.
5. Future SOTA language must be “best among tested deterministic static
   baselines” and requires a matched Resonance-YaRN panel first.

The highest-ROI universal-method hypothesis is now an offline, Native-only
selector with abstention, not one coordinate forced onto every checkpoint:

```text
freeze {Native, physical-x, normalized-index}
select using only preregistered Native calibration
emit exactly one static table for the checkpoint+s request family
abstain if no candidate passes the Native gate or a separate method-class
resolver says exact-2x is unmeasurable
```

This preserves the engineering contract—one static table, standard attention,
normal KV cache, no runtime routing—and does not read long outcomes to select a
profile. It is a future protocol, not a completed method result.

The Gemma controls also require separating
`L_config`, documented `L_train`, and a calibration-contract-specific
`L_eff`. If no trustworthy training owner exists, a candidate `L_ref` may be
frozen only on a preregistered power-of-two grid using both fresh natural-text
likelihood and fresh capability calibration, before any long holdout. If those
families disagree, no universal scalar `L_eff` may be used. Selecting 4K merely
because the present RULER rows favor it would be task tuning.

For future K evidence, arbitrary new model families are lower priority than a
same-family K32/K64 triangulation with a resolver gate. No K trend or
difference-in-differences is permitted until Native 1x and deterministic YaRN2
2x are both nonzero under the same checkpoint/data protocol.

## Claim ceiling

This study establishes K32 matched-scale Pareto structure and a hash-bound K128
negative/unresolved screen. It does not prove cross-K universality, identify K
causally, privilege `x` uniquely, solve Native compatibility, or establish
SOTA.
