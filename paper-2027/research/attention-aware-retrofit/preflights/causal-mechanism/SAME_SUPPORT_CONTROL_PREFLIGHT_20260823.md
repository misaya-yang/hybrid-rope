# Same-support mature-checkpoint controls

- **Registered:** 2026-08-23, before any same-support control GPU arm
- **Status at registration:** Qwen 128K Native n=20 still running; no control arm launched
- **Current status:** executed; superseded by
  `../../results/causal-mechanism/SAME_SUPPORT_FROZEN_CHECKPOINT_RESULT_20260823.md`. The registered
  order below is retained as historical protocol evidence, not an action queue.
- **Purpose:** discriminate numerical aliasing, support expansion, interior
  allocation, and YaRN-family split location without task-label tuning
- **Evidence role:** internal preregistration; a completed result owner must
  supersede it

## 1. Scientific boundary

For a frozen mature checkpoint, write

\[
x_k=-\log\omega_k=a+Rz_k.
\]

Holding `(a,R)`, pair count, attention amplitude, checkpoint, rows, decoder,
precision, and hardware fixed makes changes in output causally attributable to
the installed interior table `z`.  This is an inference-time operator
sensitivity/compatibility experiment.  It is **not** the same estimand as the
151.9M from-training exact-range experiment and cannot establish that one `z`
would train a better model.

## 2. Frozen CPU constructions

All long controls use factor four and attention scaling `1 + 0.1 ln(4)`.
Every new table pins the fast endpoint exactly to Native and the slow endpoint
exactly to Native divided by four.

| Model / method | float32 table SHA-256 | Construction |
| --- | --- | --- |
| OLMo converged budgeted | `a435d75441444bcea39b73d9cf530005249dc5afdc3cfb5a60fda10ef33312d3` | bitwise the already evaluated table |
| OLMo same-support geometric | `2754c9c233fe6f65686e86189c4cee94b75efac0bd8df19f140f3e575ffb723f` | log-linear between fixed endpoints |
| OLMo nearest YaRN ramp | `be76ee4cfb8524ef5d52660b4dd63e7331f53ac9f788305c4fc3dcface7817f1` | label-free projection, pairs 20 to 22 |
| Qwen converged budgeted | `3512335e408c279896a84b5b555b8be5ca3b62a70eb300f3ec866c5b5aa141d0` | 16,384 support points, maximum stride two |
| Qwen same-support geometric | `4046a441284959f8d6f635a690b0f3e9ea2b4c18cd8b43eae20ce57463714db8` | log-linear between fixed endpoints |
| Qwen nearest YaRN ramp | `c5e4576ac4fee4b60b6a887cecdb1c3f4cc33971b83645bc8fb95ce96658b9c9` | label-free projection, pairs 28 to 31 |

The model-relative uniqueness resolution keeps the OLMo construction's
original maximum distance stride of two.  It reproduces OLMo bitwise, removes
Qwen's stride-16 aliasing at pairs 1 and 18, removes both order crossings, and
restores the exact fast endpoint.  No task score selects its resolution.

Implementation SHA-256 at registration:

- evaluator: `c2612c0c111c509d20e6db8831ecab0a628f5bbcca964452fd2cf7d1d0c8d010`;
- control constructor: `9b77ac431a7f6be8d31e70034b0902f234e5c947773afca3cc7f557d38fd0049`.

Focused local tests passed `11/11`; remote CPU reconstruction matched all six
table hashes exactly.  No checkpoint was loaded for those CPU checks.

## 3. Frozen evaluation order

1. Finish the already queued Qwen 128K Native/official-YaRN/budgeted n=20
   matrix.
2. Qwen 64K core-4 n=20: converged budgeted, same-support geometric, nearest
   YaRN-family ramp.
3. OLMo 16K confirmation-only unseen-9 n=20: same-support geometric and nearest
   YaRN-family ramp.  The already evaluated budgeted table is bitwise the
   converged construction.
4. If the wall-clock guard permits, extend Qwen 128K n=20 first to converged
   budgeted and then to same-support geometric.  These are extensions, not
   method-selection cells.

Qwen uses data manifest
`88c250401040c68e935ef5bbdb8159351292f043a32b247c5de30916873ede45`.
OLMo uses the confirmation manifest
`431ad942eead911da4693a4ab7428085fd04e63d6914fc8831c3d5b0cc88c239`.

## 4. Decision gates

All outcomes are reported; no arm changes `p`, factor, amplitude, or another
arm's table.

### A. Numerical-convergence gate

- If converged Qwen 64K falls below official YaRN factor four (`0.6025`), stop
  using the old aliased Qwen score as positive cross-model evidence.
- If it remains above YaRN but below the old aliased score (`0.6700`), retain
  only the converged result and state that the numerical artefact contributed.
- If it matches or exceeds `0.6700`, the old result's direction did not depend
  on the artefact; the converged table becomes the only promotable identity.

### B. Interior-allocation gate

- If same-support geometric is not worse than converged budgeted, the current
  evidence does not support non-geometric interior allocation as the operative
  retrofit variable.
- If converged budgeted beats geometric on paired rows, fixed-support
  inference-time sensitivity to `z` is established for that checkpoint.  This
  still does not establish from-training superiority.

### C. YaRN-family gate

- If the nearest ramp is within `0.03` macro of converged budgeted and their
  paired row uncertainty includes zero, treat them as practically equivalent.
  The contribution is then automatic, label-free derivation of the split
  location, not a new operator family.
- Only a converged-budgeted advantage greater than `0.05` with paired row
  uncertainty excluding zero supports a claim that profile detail beyond the
  nearest linear ramp matters.  Intermediate outcomes remain descriptive.

Bootstrap intervals, if reported, condition on the frozen checkpoint and task
set.  They describe evaluation-row sensitivity only; they are not model,
training-seed, or task-population uncertainty and are not confirmatory p-values.

## 5. Stop conditions

- Do not tune YaRN betas, the geometric base, uniqueness exponent, amplitude,
  or support resolution on these results.
- Do not use the old Qwen crossing table as an exact-support arm.
- Do not promote RULER as natural-document capability evidence.
- Stop before the platform shutdown guard rather than launch an arm that cannot
  finish and write `results.json`.
