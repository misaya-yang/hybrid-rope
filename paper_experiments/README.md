# Paper experiment code workspace

This directory is the single browsing and reuse entry point for code behind the experiments reported in the paper. It covers the three primary text/MLA anchors, theory diagnostics, supporting text, video DiT, LoRA, data preparation, and paper figures.

## Layout and source of truth

- `code/` mirrors repository-relative paths through symbolic links.
- `MANIFEST.json` records every source path, experiment family, and SHA-256 digest.
- The canonical implementation remains under `scripts/` and `experiments/`; this workspace is an organized live view, not a second editable codebase.
- Results, checkpoints, private audit notes, caches, and ignored server launch artifacts are deliberately excluded.

The workspace includes current reviewer-grade entrypoints and their direct local helpers. It excludes known obsolete alternatives such as `phase21b_quality_eval.py` and the legacy core-text passkey evaluator. For Primary I, the tracked Phase 14c runner is a supporting multiscale reproduction path, not the complete historical 454M Table 2 run. For Primary II, the retained L=128 protocol is documented in the repository, while the linked Phase 11 scripts are the available current and supporting runners; the workspace does not relabel an L=256 runner as the original L=128 experiment.

## Rebuild and verify

Run from the repository root:

```bash
python3 scripts/build_paper_experiment_workspace.py
python3 -m pytest tests/test_paper_experiment_workspace.py -q
```

To create a standalone copy with links expanded:

```bash
cp -RL paper_experiments /tmp/evq-paper-experiments
```

Run linked entrypoints from the repository root so their existing package imports and relative configuration behavior remain unchanged.
