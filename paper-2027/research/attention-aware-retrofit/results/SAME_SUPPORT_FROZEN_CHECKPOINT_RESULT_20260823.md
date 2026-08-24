# Fixed-support exponent allocation in frozen checkpoints

- **Date:** 2026-08-23
- **Status:** complete
- **Evidence role:** internal paper-upgrade owner, not manuscript text
- **Supersedes:** the decision gates in
  `../preflights/SAME_SUPPORT_CONTROL_PREFLIGHT_20260823.md` and the planned
  status in `../preflights/SMALL_MODEL_RETROFIT_CAUSAL_PREFLIGHT_20260823.md`

## Conclusion first

This study strengthens the paper's central claim, but not by establishing a
new inference-time operator family.

1. At fixed frequency endpoints, amplitude, checkpoint, data, and decoder,
   the normalized exponent allocation `z` has a large causal effect in two
   mature checkpoints. The same-support geometric control nearly collapses on
   OLMo at 16K (`0.0056`), whereas the non-geometric derived table scores
   `0.6047`. On Qwen at 64K, the corresponding scores are `0.5775` and
   `0.6650`.
2. The detailed uniqueness profile is not identified as necessary. A
   label-free nearest movement-profile ramp scores `0.6104` on OLMo and
   `0.6400` on Qwen; its paired evaluation-row interval against the full
   profile includes zero on both models. The frozen-checkpoint contribution is
   therefore the model-relative derivation of the useful split location, not
   a claim that the profile defines a new operator family.
3. A separate 151.9M two-training-seed crossing shows strong
   weights-by-runtime-table co-adaptation. At 1K, FMRoPE-trained weights prefer
   their FMRoPE-derived table (`3.426` versus `5.776` tail NLL), while
   anchored-Cosh-trained weights prefer their Cosh-derived table (`3.479`
   versus `4.455`). The crossover interaction is `3.400/3.251` NLL in seeds
   137/256. This is a mechanism replication of the existing 50M crossing, not
   a replacement for the three-seed exact-range training result.

The defensible paper story is consequently:

> normalized exponent allocation is an independent training-time variable;
> weights co-adapt to that coordinate system; and the same fixed-support axis
> remains causally consequential when installed in two frozen mature
> checkpoints.

It is not defensible to say merely that no earlier method ever produced a
non-geometric frequency table. Several scaling methods do so implicitly. The
novel object is the explicit `x_k = a + R z_k` decomposition, fixed-support
causal identification of `z`, and its connection to finite spectral
redundancy.

This statement distinguishes **restricted interventions**, not two uniquely
recoverable textual parameters. A realised tensor can be re-expressed with a
different base/exponent pair if both are free. The scalar-base control is still
causally distinct because it is restricted to the geometric exponent family;
the same-support experiment fixes everything that family can set at the two
endpoints and changes the interior curve. The authoritative global grammar is
[`../../ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md`](../../ROPE_CAUSAL_VARIABLES_AND_ZERO_TRAINING_RETROFIT_20260823.md).

This distinction matters against recent work. MrRoPE constructs a
training-free mixed-radix/progressive spectrum, CoPE soft-clips selected slow
frequencies, and Jet-Long uses a tuning-free dynamic bifocal transform with an
exact short-context path. These methods make “the output frequencies are
non-geometric” too broad to carry novelty. None of that erases the paper's
controlled object: interior exponent allocation at fixed sampled support,
identified during training rather than inferred from a baseline comparison.
This related-work check narrows the claim; it is not an exhaustive proof that
no prior paper uses equivalent notation.

## 1. What was held fixed

For each mature checkpoint, write

\[
x_k=-\log\omega_k=a+Rz_k.
\]

The control arms hold `(a,R)`, pair count, factor-four attention amplitude,
checkpoint, evaluation rows, decoding, precision, and hardware fixed. They
change only the interior `z`:

- **geometric:** `z_k=k/(K-1)`, a pure log-linear table between the endpoints;
- **nearest movement-profile ramp:** the discrete YaRN-family linear ramp that
  minimises movement-profile MSE to the derived table, without task labels;
- **derived:** phase-resolved conditional uniqueness with model-relative
  resolution and exact endpoint pinning.

This makes the geometric-versus-non-geometric comparison a clean
frozen-checkpoint intervention. It remains a different estimand from training
two models with different `z`.

The complete zero-training deployment owner is broader than this control: it
combines one deterministic long table, a fixed long attention amplitude, and a
Native/long session route. This section isolates the table's interior
coordinate; it does not redefine the practical method as three sequential
modules.

The factor-four amplitude is fixed in every arm. The earlier mature-model
frequency-by-amplitude 2x2 remains the owner of softmax interaction; this
study does not reuse its amplitude effect as an allocation effect.

## 2. Numerical aliasing was removed before comparison

The earlier Qwen construction sampled maximum distance at stride 16. It made
two unrelated pairs share the same numerical uniqueness value to six decimal
places, moved the fast endpoint by `0.17%`, and introduced two order
crossings. Recomputing at the model-relative maximum stride two removes both
crossings and pins both endpoints exactly. The substantively redundant block
`k=27..63` is unchanged.

At 64K, the corrected Qwen table scores `0.6650` versus `0.6700` for the old
aliased table. The `-0.0050` change has a paired row-bootstrap interval
`[-0.0700, 0.0575]`. The positive cross-model direction therefore did not
come from the artefact; only the corrected table is retained as a valid
identity.

## 3. Frozen mature-checkpoint result

### 3.1 Qwen2.5-1.5B, Native 32K, core-4 RULER at 64K

All cells contain 20 identical evaluation rows.

| Installed table | Macro score |
| --- | ---: |
| Native | 0.5450 |
| official YaRN factor four | 0.6025 |
| same-support geometric | 0.5775 |
| nearest movement-profile ramp | 0.6400 |
| **corrected derived allocation** | **0.6650** |

The paired row bootstrap uses 20,000 within-task resamples with seed
`20260823`. It conditions on this checkpoint and task set; it is not
checkpoint, model, or task-population uncertainty.

| Contrast | Difference | 95% interval | Bootstrap mass at or below zero |
| --- | ---: | --- | ---: |
| derived - geometric | +0.0875 | [+0.0025, +0.1750] | 0.0233 |
| derived - movement-profile ramp | +0.0250 | [-0.0525, +0.1000] | 0.2676 |
| derived - official YaRN | +0.0625 | [-0.0050, +0.1325] | 0.0349 |
| movement-profile ramp - official YaRN | +0.0375 | [-0.0325, +0.1075] | not used for a claim |

The decision gate passes for fixed-support `z` sensitivity and fails for
profile-detail novelty. The full profile and its nearest movement-profile ramp are practically
equivalent under the preregistered `0.03` gate.

### 3.2 OLMo-2-0425-1B, Native 4K, unseen-nine RULER at 16K

These nine tasks were confirmation-only: the table, amplitude, split rule,
checkpoint, and metric were frozen before this task subset was evaluated.
Every task contains 20 rows.

| Installed table | Macro score |
| --- | ---: |
| official YaRN factor four | 0.0794 |
| same-support geometric | 0.0056 |
| **derived allocation** | **0.6047** |
| nearest movement-profile ramp | 0.6104 |

| Contrast | Difference | 95% paired row-bootstrap interval |
| --- | ---: | --- |
| derived - geometric | +0.5992 | [+0.5488, +0.6480] |
| movement-profile ramp - geometric | +0.6048 | [+0.5559, +0.6542] |
| derived - movement-profile ramp | -0.0056 | [-0.0464, +0.0345] |
| derived - official YaRN | +0.5254 | [+0.4699, +0.5795] |

The OLMo result makes the fixed-support causal claim much stronger: a scalar
base change that only produces geometric spacing cannot reproduce the useful
interior allocation. It simultaneously narrows the inference-time method
claim, because the simple projected ramp captures the full profile.

### 3.3 Qwen 128K symmetric baseline

The completed n=20 baseline matrix is:

| Installed table | Macro score |
| --- | ---: |
| Native | 0.4350 |
| official YaRN factor four | 0.4650 |
| old aliased derived table | 0.6175 |
| same-support geometric | 0.4550 |
| **corrected derived allocation** | **0.5400** |

The corrected table remains above Native by `+0.1050`, with a paired
evaluation-row interval `[+0.0275,+0.1850]`, and above YaRN by `+0.0750`, with
interval `[-0.0125,+0.1625]`. It is `-0.0775` below the old aliased table, with
interval `[-0.1500,-0.0050]`: unlike at 64K, the numerical artefact materially
inflated the 128K score. Only `0.5400` is a valid corrected-profile result.

At fixed endpoints and amplitude, corrected derived exceeds geometric by
`+0.0850`; the paired evaluation-row interval is `[-0.0100,+0.1800]`. This
replicates the positive interior-allocation direction at 4x, with weaker
row-sensitivity evidence than at Qwen 64K or OLMo 16K. The n=5 probe is
superseded by this n=20 matrix and must not be cited.

## 4. 151.9M weights-by-table causal crossing

Two independently trained seeds contain paired FMRoPE and anchored-Cosh
weights. Their training tables share exact endpoints and differ only in
interior allocation. For each frozen checkpoint, two factor-four tables were
derived separately from those training coordinates and installed on the same
32 FineWeb-Edu validation anchors. The primary endpoint is final-128-token NLL;
lower is better.

### 4.1 Two-seed mean at length 1024

| Frozen weights | FMRoPE-derived table | Cosh-derived table | Same-support geometric |
| --- | ---: | ---: | ---: |
| FMRoPE-trained | **3.426** | 5.776 | 3.429 |
| anchored-Cosh-trained | 4.455 | **3.479** | 4.177 |

The table assignment reverses with the trained weights. Define the crossover
interaction as

\[
[L(W_F,T_C)-L(W_F,T_F)]-[L(W_C,T_C)-L(W_C,T_F)].
\]

It is `3.400` for seed 137 and `3.251` for seed 256 at 1K. At 512 it is
`2.663/2.564`. No row-bootstrap interval is presented as training uncertainty:
the training seed is the replication unit and only two checkpoint pairs are
available on this server.

The result establishes compatibility/co-adaptation, not global optimality of
either derived table. In particular, geometric is competitive on FMRoPE
weights but poor on Cosh weights. The outcome is exactly why a frozen mature
checkpoint retrofit and a from-scratch allocation experiment must not be
merged into one estimand.

## 5. Novelty and reviewer-facing interpretation

The experiment answers the likely reviewer objection in layers:

1. **Is the result only a larger support/base?** No. Geometric and derived
   tables have identical endpoints; OLMo differs by `+0.599` macro.
2. **Is it only YaRN amplitude?** No. Amplitude is identical in every control.
3. **Is the detailed uniqueness curve itself the invention?** Not supported.
   A nearest movement-profile ramp matches it on both models.
4. **Is the split task-tuned?** No. The ramp boundaries are a label-free
   projection of the frozen derived table: pairs `20->22` for OLMo and
   `28->31` for Qwen.
5. **Does changing `z` matter only after the fact?** No. The existing
   three-seed exact-range result identifies training-time `z`; the new 151.9M
   crossing shows that learned weights are strongly coordinate-dependent.

The strongest manuscript use is a compact mature-checkpoint corollary to the
main fixed-support theorem/experiment, not a standalone ``better YaRN''
section. If promoted, the text should say that the redundancy analysis derives
a model-relative split inside a standard interpolation family and that
fixed-support exponent allocation remains consequential after pretraining.
The related-work paragraph must also include Jet-Long rather than presenting
binary Native/long routing as unique systems machinery.

For a reviewer, the causal decomposition should be visible in one compact
table rather than reconstructed from method names:

| Question | Held fixed | Intervention | Owner |
| --- | --- | --- | --- |
| Does interior allocation matter in training? | support, recipe, seed pairing | FMRoPE versus anchored Cosh `z` | 151.9M three-seed exact-range |
| Do weights learn that coordinate system? | frozen weights/table factorial | cross weights and runtime table | 50M 2x2 plus this 151.9M replication |
| Does `z` still matter after pretraining? | support, amplitude, checkpoint, rows | geometric versus non-geometric `z` | this OLMo/Qwen study |
| Is the detailed new curve necessary? | same controls | nearest movement-profile ramp versus full profile | this study; answer is no |

The introduction should not rely on the phrase “non-geometric is new.” It
should define `z` before naming EVQ-Cosh, say that prior scaling rules can also
produce non-geometric spectra, and claim the narrower contribution: direct
fixed-support identification and explanation of the allocation axis.

## 6. Practical scope

The method installs one static frequency tensor for a session and uses the
standard rotary operation. It therefore preserves FlashAttention and normal
KV-cache semantics; a session must not change its table after cached keys have
been created. The exact Native branch is selected only when the observed
prefill plus generation budget fits the model's own Native context window.
Long requests use one deployment-frozen long profile.

This is better described as a **single-long-profile, budget-gated session
policy** than as universally target-free: it needs no externally supplied
`L_target`, but the serving stack still knows the model's Native window and the
request's generation budget.

## 7. Claim ceiling and remaining gap

Supported:

- fixed-support interior exponent allocation is causally consequential in two
  mature checkpoints under the stated RULER protocols;
- a model-relative, task-label-free split transfers across OLMo and Qwen;
- the 151.9M crossing replicates strong weights-table co-adaptation in two
  training seeds;
- the implementation is zero-training and compatible with standard rotary
  attention/KV caching when the table is session-static.

Not supported:

- a new interpolation operator family or universal best profile;
- superiority on all natural long-document tasks;
- task-population, model-population, or training-seed statistical claims from
  RULER row bootstraps;
- a claim that the frozen-checkpoint result is the same causal estimand as the
  exact-range from-training experiment;
- dynamic within-session switching after KV-cache creation.

The highest-value remaining external validation is one same-support
geometric/ramp/derived comparison on a natural OLMo long-document task and one
natural Qwen task. It should be run only if manuscript promotion is authorised;
the present GPU window was spent first on the causal controls and the 151.9M
mechanism crossing.

## 8. Evidence receipt

The compact machine-path-free JSON companion records table identities,
per-arm result hashes, paired-row sensitivity intervals, the complete
two-seed 151.9M summary, and the final Qwen 128K status. Raw outputs are stored
outside the repository and were copied with remote/local SHA-256 parity.

Focused validation passed `38/38` tests covering same-support construction,
target-free/session routing, context building, and downstream helpers. All
three changed Python entrypoints pass bytecode compilation; both evidence JSON
files parse; `git diff --check` passes. Local Conda was unavailable, so these
checks used system Python 3.9. The immutable `paper/` baseline was not changed
or compiled.
