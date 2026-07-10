# Local core-asset recovery summary

## Corrected decision

The full local and branch audit found more decision-critical evidence than the
first pass reported. This update promotes three additional evidence families
and keeps the previously promoted Phase11B boundary result. No paper numbers,
tables, figures, PDFs, or claims were changed.

| Question | Corrected finding | Tracked reviewer path |
| --- | --- | --- |
| Primary I original raw payload | Found on `backup/2026-03-06`; all six Geo/EVQ seed records, raw/YaRN PPL, teacher-forced PK cells, and distinct AR-exact fields are present. | `data/results_5090b/evq_yarn_10pct_allseeds.json`; portable full-payload copy `data/curated/primary1_evq_yarn_10pct_raw.json`. |
| Primary II `L_train=128` extra seeds | Partially found: fixed EVQ tau=5 exists for seeds 42/137/256. Matched Geo and DAPE seeds 137/256 were not found. | Two exact archival source JSONs plus `data/curated/primary2_l128_fixed_tau5_3seed.json`. |
| Primary III MLA three-seed ignored JSON | The old physical path was not recovered, but the complete source object is preserved in the tracked snapshot. Replaying the evaluator's key order and `json.dumps(..., indent=2)` reproduces the old source byte-for-byte and matches its recorded SHA256. | `data/curated/eval_3seeds_full_results.json`, SHA256 `1e44d30...30953`; reconstruction source `data/curated/table18_mla_3seed_aggregate.json`. |
| 1B-token MLA multi-seed | Not found in current refs, the archival branch, local result trees, or recoverable Git objects. The retained 1B row remains single-seed supporting/schedule-sensitivity evidence. | No multi-seed artifact promoted. |

The earlier statement that the original Primary I raw payload was absent was
wrong and is superseded by this document. The earlier MLA statement also
needed precision: the original physical file was absent, but its exact JSON
bytes are deterministically recoverable from the portable raw-backed snapshot.

## Scientific value of the recovered assets

### Primary I is now raw-payload-backed

The restored 59,580-byte source has SHA256
`1dbec88e...511c` and contains the complete six-run payload for methods Geo and
EVQ at seeds 7, 42, and 123. Recomputing directly from the raw records gives:

| Method/eval | mean PPL@2K | mean PPL@8K | mean teacher-forced PK@8K |
| --- | ---: | ---: | ---: |
| Geo raw | 67.214 | 161.863 | 0.407 |
| Geo + YaRN(s=8) | 68.055 | 82.925 | 0.613 |
| EVQ raw | 67.869 | 150.279 | 0.533 |
| EVQ + YaRN(s=8) | 70.716 | 70.851 | 1.000 |

These recomputed means reproduce the rounded Table 2 values. The raw payload
also retains autoregressive exact match as a separate field; it does not change
the paper's definition of PK as teacher-forced NLL-gap retrieval.

### Primary II is partially recovered, not closed

The tau=5 fixed-EVQ arm is now raw-backed for seeds 42/137/256 under the exact
125M, `L_train=128`, 15M-token protocol. Its raw mean/sample-standard-deviation
is `182.605 +/- 1.144` at 128 tokens and `335.710 +/- 1.745` at 8K.

This is useful evidence that the fixed-EVQ row itself is stable across the two
supplementary seeds. It cannot upgrade the full Primary II comparison because
matched Geo and DAPE results for seeds 137/256 were not recovered. The honest
rebuttal statement remains: submitted Geo/DAPE/EVQ comparisons are seed 42,
with three-seed evidence available only for the fixed EVQ arm and learnable-tau
endpoints.

### Primary III exact JSON is recoverable

`table18_mla_3seed_aggregate.json` contains the evaluator's complete top-level
source keys in original order: `seeds`, `eval_lengths`, `progression`,
`extended`, and `summary`. Serializing those keys with the original evaluator
format produces 8,342 bytes, no trailing newline, and SHA256
`1e44d30...30953`, exactly matching the previously recorded ignored-source
identity. The builder and tests now enforce this byte-for-byte reconstruction.

This restores the original evaluation JSON, not the missing checkpoints,
training caches, or per-run training payloads.

### Phase11B remains supporting boundary evidence

`data/curated/phase11b_125m_l256_3seed.json` remains valuable for a separate
`L_train=256`, 100M-token protocol: plain EVQ improves long-range PPL across
three seeds, while EVQ+DAPE is 1.38% to 2.04% worse than Geo+DAPE. It is useful
to state a non-complementarity boundary, but it is not the submitted
`L_train=128` Primary II protocol and cannot substitute for its missing matched
Geo/DAPE seeds.

## What was deliberately not promoted

- Checkpoints, tensor caches, logs, build artifacts, and broad copied audit
  workspaces: large, machine-specific, or non-reviewer-grade.
- Historical LLaMA/Qwen controls affected by old protocol issues: they cannot
  validate the corrected matched 8B experiment.
- Single-seed 750M continuation and old 1B-token MLA rows: supporting evidence
  only; neither closes a current multi-seed reviewer question.
- Six zero-byte JSON files: invalid artifacts, never completed evidence.

## Remaining decision-critical gaps

1. Matched `L_train=128` Geo and DAPE seeds 137/256. Fixed EVQ tau=5 no longer
   needs recovery, but the comparison still needs those four matched runs.
2. Reviewer-grade 1B-token MLA multi-seed results. None were found.
3. Primary I and Primary III checkpoint/data hashes. Raw evaluation payloads
   are now tracked, but checkpoint-level provenance remains unavailable.
4. Corrected matched LLaMA-3-8B Geo/EVQ positional-distillation results. The
   protocol and scripts exist; no new completion is claimed here.

## Rebuild and validation

All newly recovered source JSONs are tracked, so a fresh checkout can rebuild
the portable assets without the broad ignored result tree:

```bash
python3 scripts/build_rebuttal_evidence_bundle.py \
  --only primary1 --only primary2_tau5 --only mla_raw
python3 scripts/validate_rebuttal_evidence_bundle.py
python3 -m unittest tests.test_rebuttal_evidence_bundle -v
```

The builder rejects any source whose SHA256 differs from the audited archival
identity, and the validator checks the raw source files and reconstructed MLA
file independently.
