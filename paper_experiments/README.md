# Paper experiment code workspace

This directory is a historical/supporting browsing view of code assembled for
the paper lineage. It covers older primary text/MLA anchors, theory diagnostics,
supporting text, video DiT, LoRA, data preparation, and figure code; it is not
the current ICLR evidence router.

## Evidence and scope boundary

- The checked-in schema-v1 `MANIFEST.json` contains 94 repository-relative,
  symlinked code entries. It establishes the identity and integrity of this code
  view only.
- The manifest does not prove that a run occurred, establish a result or
  checkpoint/data identity, demonstrate runtime readiness, or validate a current
  experiment preflight.
- This snapshot does not cover the current matched-content phase intervention,
  target-free research, or leave-one-band-out band-attribution route. Their
  current status and protocol boundaries must be read from
  [`../INDEX.md`](../INDEX.md) and the canonical owners/preflights it names.
- Canonical code remains under `scripts/` and `experiments/`; canonical result
  ownership is routed by [`../INDEX.md`](../INDEX.md) §2 and the corresponding
  research owners. [`../paper-2027/HANDOFF.md`](../paper-2027/HANDOFF.md)
  records volatile state rather than evidence ownership.

## Layout and source of truth

- `code/` mirrors repository-relative paths through symbolic links.
- `MANIFEST.json` records every source path, experiment family, and SHA-256 digest.
- The canonical implementation remains under `scripts/` and `experiments/`; this workspace is an organized live view, not a second editable codebase.
- Results, checkpoints, private audit notes, caches, and ignored server launch artifacts are deliberately excluded.

The workspace includes selected historical and supporting entrypoints plus their
direct local helpers. It excludes known obsolete alternatives such as
`phase21b_quality_eval.py` and the legacy core-text passkey evaluator. For
Primary I, the tracked Phase 14c runner is a supporting multiscale reproduction
path, not the complete historical 454M Table 2 run. For Primary II, the retained
L=128 protocol is documented in the repository, while the linked Phase 11
scripts are the available supporting runners; the workspace does not relabel an
L=256 runner as the original L=128 experiment.

## Rebuild and verify

Run only on the work machine, from the repository root, through its canonical
`aidemo` environment. On the low-configuration personal PC, leave this gate
skipped; do not install the work-machine environment merely to rebuild this
view.

```bash
conda run --no-capture-output -n aidemo python scripts/build_paper_experiment_workspace.py
conda run --no-capture-output -n aidemo python -m pytest tests/test_paper_experiment_workspace.py -q
```

On the work machine, create a standalone copy with links expanded only when a
separate review artifact is actually required:

```bash
cp -RL paper_experiments /tmp/evq-paper-experiments
```

Run linked entrypoints from the repository root so their existing package imports and relative configuration behavior remain unchanged.
