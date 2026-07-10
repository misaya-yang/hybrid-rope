# Local ignored core-asset promotion summary

## Decision

The ignored experiment tree was audited for evidence that could materially
change the paper or rebuttal. The repository should not absorb the complete
local result tree: it is about 86 GB and is dominated by checkpoints, logs,
caches, legacy runs, and supporting experiments. This pass promotes one
additional anonymous, hash-identified payload:

- `data/curated/phase11b_125m_l256_3seed.json`

This is the only newly recovered local family with direct scientific value for
a core paper boundary. It preserves 15 completed Phase11B runs: nine plain
Geo/EVQ scaling runs and six Geo+DAPE/EVQ+DAPE runs, all at three seeds.
No paper numbers, tables, figures, or claims were changed in this pass.

## What the promoted Phase11B evidence establishes

Protocol: 125M model, `L_train=256`, 100M training tokens, base 500K, seeds
42/137/256, and evaluation from 256 through 8K tokens.

| Method | PPL@256 | PPL@512 | PPL@1K | PPL@2K | PPL@4K | PPL@8K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Geo | 72.060 | 56.404 | 111.475 | 209.366 | 259.933 | 352.695 |
| EVQ tau=2 | 73.427 | 50.542 | 91.701 | 169.747 | 210.237 | 304.197 |
| EVQ tau=4 | 72.954 | 48.176 | 76.759 | 136.935 | 170.269 | 254.694 |
| Geo+DAPE | 68.501 | 40.588 | 47.867 | 56.354 | 43.639 | 55.922 |
| EVQ tau=4+DAPE | 69.898 | 41.248 | 48.523 | 57.431 | 44.435 | 56.844 |

The seed-paired EVQ tau=4 change relative to Geo is +1.24% at 256, then
-14.59%, -31.15%, -34.61%, -34.52%, and -27.78% from 512 through 8K. The
plain-EVQ scaling direction is consistent across all three seeds.

The DAPE result is equally important because it limits the claim: EVQ+DAPE is
1.38% to 2.04% worse than Geo+DAPE across the evaluated lengths. This protocol
therefore does not support EVQ-DAPE complementarity. Its defensible use is to
separate EVQ's zero-learned-parameter frequency allocation from DAPE's learned
attention-bias/refinement route.

## What this evidence does not establish

This Phase11B payload is not the submitted Primary II protocol. It uses
`L_train=256` and 100M tokens, whereas the retained Primary II Geo/DAPE/EVQ
diagnostic uses `L_train=128` and 15M tokens. It also compares Geo+DAPE with
EVQ+DAPE, not the exact bare Geo/DAPE/EVQ rows requested for replication.
Accordingly, it must not:

- upgrade Primary II from seed 42 to a three-seed claim;
- substitute for the two missing `L_train=128` seeds;
- be used to claim that EVQ and DAPE are complementary.

## Already tracked before this pass

The earlier evidence promotion on `main` already contains the reviewer-safe
artifacts that matter most:

- Primary I EVQ x YaRN Table 2/3 values and seedwise PK@8K values;
- Primary III three-seed MLA aggregate with exact source hash;
- Phase11 `L_train=256` Geo/EVQ/YaRN/NTK archival payloads;
- the 99-run Phase16 basin manifest;
- the full n=2,086 QuALITY aggregate;
- the base-10K/base-500K trained-text pilot;
- report-backed learnable-tau endpoints and scarce-channel pilot evidence.

The local raw sources for Phase11, QuALITY, the base pilot, and the new
Phase11B payload are present and match their recorded SHA256 identities. The
ignored source for the Primary III MLA snapshot is not present at its expected
local path, but the tracked snapshot retains its exact source hash and
per-seed values.

## Families not promoted

The following local families were inspected and intentionally not promoted:

| Family | Decision |
| --- | --- |
| Checkpoints, tensor caches, logs, bytecode | Reproducibility/storage noise; keep ignored. |
| 750M continuation | Single-seed supporting evidence, not a current rebuttal dependency. |
| Historical LLaMA/Qwen LoRA and LongBench | Old or affected control protocols; do not reuse for the corrected matched 8B claim. |
| Video DiT | Supporting two-seed modality evidence already summarized; no core rebuttal gap closed. |
| Weekend tau sweeps | The useful 99-run coverage is already represented by the sanitized Phase16 manifest. |
| Six zero-byte JSON files | Invalid artifacts; never treat them as completed evidence. |
| Broad copied audit workspaces and private manifests | Non-portable and potentially machine-specific; keep ignored. |

## Remaining decision-critical gaps

1. Two additional matched `L_train=128` seeds for Primary II Geo/DAPE/fixed
   EVQ remain the highest-value missing experiment.
2. The exact original Primary I raw run payload is not present locally; the
   tracked Table 2/3 artifact is a validated, transcribed evidence snapshot.
3. The exact ignored raw JSON behind the tracked Primary III MLA snapshot is
   absent locally, although its portable per-seed snapshot and source hash are
   tracked.
4. No reviewer-grade multi-seed 1B-token MLA completion artifact was found.
5. The corrected matched LLaMA-3-8B Geo/EVQ positional-distillation experiment
   has scripts and protocol only; it has not been run and has no result payload.

## Rebuild and validation

On a checkout containing the ignored Phase11B sources:

```bash
python scripts/build_rebuttal_evidence_bundle.py --only phase11b
python scripts/validate_rebuttal_evidence_bundle.py
python -m unittest tests.test_rebuttal_evidence_bundle -v
```

The builder refuses source files whose SHA256 identity differs from the two
audited raw JSON files.
