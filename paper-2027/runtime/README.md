# Frozen TailSpline/MrPro evaluation

The source archive puts the existing experimental entrypoints and their local
Python imports in this directory. Run the commands below from `runtime/`.
Python 3.10+, NumPy, PyTorch, Transformers, PyYAML and the pinned RULER generator
dependencies are required; model execution uses CUDA. Model paths refer to
locally obtained public checkpoints. Commands below describe reproduction;
building the manuscript does not run them.

## Inputs and preprocessing

Use NVIDIA/RULER revision `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`.
The Llama recipe uses Meta-Llama-3-8B-Instruct, native length 8192 and preparation
seed 20260924. OLMo uses OLMo-2-0425-1B-Instruct, native length 4096 and seed
20260925. Set `MODEL`, `RULER`, `SOURCES` and `OUT` to local directories.

For Llama:

```bash
python -m experiments.llama3_60dir_20260911.prepare_planb_panel \
  --model "$MODEL" --upstream "$RULER" --out "$OUT/full13" \
  --stage H --contract tailspline-classic --seed 20260924 \
  --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_1,qa_2 \
  --caps 8192,16384,32768 --counts-by-cap 8192:10,16384:10,32768:10 \
  --depth-targets 0.10,0.30,0.50,0.70,0.90

python -m experiments.fixed_rope_three_interfaces_20260913.prepare_llama_ppl46 \
  --model "$MODEL" --source-root "$SOURCES" \
  --source-manifest "$SOURCES/sources.json" --out "$OUT/ppl46"
```

For OLMo use `--model-contract generic --contract planb --seed 20260925`,
`--caps 4096,8192,16384 --counts-by-cap 4096:10,8192:10,16384:10`, and the
`prepare_olmo_ppl46` module. Other panel flags are unchanged.

PPL46 uses the recorded `sources.json` and its 32 Proof-Pile and 14 PG-19 files.
Public origins are the original `hoskinson-center/proof-pile` test archive
and `deepmind-gutenberg` test books. The source-selection helper is
`experiments/nongeometric_screen/prepare_long_sources.py`; the PPL46 preparers
consume the frozen 46-document manifest and preserve its document identities.
They tokenize each document separately and retain aligned prefixes. The paper's
appendix gives the selection and scoring rules.

## Construct and install both tables

Run once for `ARM=tailspline` and once for `ARM=mrpro`:

```bash
python -m experiments.fixed_rope_three_interfaces_20260913.tables analytic \
  --config "$MODEL/config.json" --method "$ARM" --scale 4 \
  --candidate-id "$ARM" --model-id "$MODEL_ID" --role "$ROLE" \
  --changed-variable exponent_allocation --out "$OUT/tables/$ARM.json"
```

Use `ROLE=candidate` for TailSpline and `ROLE=baseline` for MrPro. The public
configuration determines band `[18,35]` on Llama and `[14,32]` on OLMo; both use
gain `1 + 0.1 ln(4)`. The standalone `../figs/allocation_design.py` provides a
NumPy-only TailSpline constructor and algebra checks.

## Generate and score

For each Llama arm:

```bash
python -m experiments.olmo_recovery_20260912.recovery_v2_eval \
  --data "$OUT/ppl46/manifest.json" --model "$MODEL" --arm Native \
  --extra-panel "$OUT/full13/rows.jsonl" --only-extra-panels \
  --length-cap 8192 --length-cap 16384 --length-cap 32768 \
  --lm-length-cap 8192 --lm-length-cap 16384 --lm-length-cap 32768 \
  --prefill-chunk-size 8192 --batch-size 1 \
  --static-table-json "$OUT/tables/$ARM.json" --table-label "$ARM" \
  --out "$OUT/runs/$ARM" --execute
```

For OLMo use lengths `4096,8192,16384`, `--batch-size 4` and
`--prefill-chunk-size 0`. `--arm Native` selects the frozen loading path; the
explicit static-table receipt determines the installed experimental table.
Greedy decoding, official task scores, EOS/cap status and per-document LM loss
are written by this entrypoint. It applies no adapter or weight update.

## Paired report

```bash
python -m experiments.fixed_rope_three_interfaces_20260913.tailspline_llama_classic_report \
  --run tailspline="$OUT/runs/tailspline" --run mrpro="$OUT/runs/mrpro" \
  --receipt tailspline="$OUT/tables/tailspline.json" \
  --receipt mrpro="$OUT/tables/mrpro.json" \
  --ppl-manifest "$OUT/ppl46/manifest.json" \
  --candidate tailspline --baseline mrpro --out "$OUT/report.json"
```

Use `tailspline_olmo_classic_report` for OLMo. The report includes task/length
scores, combined and source-specific PPL, normalized log-length AUC and paired
bootstrap intervals. It consumes the original `generations.jsonl`,
`lm_rows.jsonl`, `contract.json`, status, table receipts and source manifest.
The manuscript's numerical summaries remain in `../figs/`.

## September 16 direct baseline and long-book panels

The archived reports in `figs/revision_evidence_inputs.json` contain the completed
three-method NIAH/PPL panels, GLM Full-13, Qwen/GLM natural QA, and higher-scale
evaluations. Run `python3 figs/make_revision_evidence.py` from the archive root
to check recorded aggregate arithmetic and regenerate the added tables. This
does not run models or rescore generated text. The script also checks the
fixed-state softmax identity and turn-boundary formulas.

The runtime includes `matched_three_method_quick_report.py`,
`official_yarn_naturalqa.py`, `run_natural_long.py`, and their local imports.
These retain the official task scorers, paired prompt selection, source-context
resampling and static YaRN installation. The Natural-QA panels use complete
books, a 40-token generation cap and the official English-QA F1 adapter.


## September 17 evidence and native-window deployments

The runtime sources are pinned by `../figs/runtime_source_snapshot.json`; the
packager reads that Git revision rather than ongoing working-tree experiments.
The bundle is a frozen reference implementation. Individual runs retain their
recorded runtime identities in the portable result inputs.

`four_model_yarn_full13.py` provides the matched three-arm Full-13 protocol;
`official_yarn_naturalqa.py` provides the natural-task comparison. The 70B
comparison uses the reported NF4 checkpoint with BF16 computation and the same
public Llama table and input manifests. It has no YaRN arm.

Native Contrastive Proximal allocation is provided by
`experiments/native_contrastive_proximal_20260915/tables.py`; the original FP32
receipt and public native grid are in `../figs/completed_evidence_inputs.json`.
The candidate preserves native support and unit gain. The source pack verifies
its exact FP32 reconstruction and the finite-series curvature certificate.

`lm_context.py` specifies same-target full/recent context pairs and scoring;
`run_native_lm.py` evaluates them. These target-only OLMo losses are separate
from whole-prefix PPL in the frozen extension experiments. The native QA primary
score weights questions within task and tasks equally; bootstrap resamples
source clusters while retaining that question weighting. The earlier
source-equal sensitivity report remains a separate estimand.

Recompute the paper tables, native score aggregates, paired intervals and
public table checks without model execution:

```bash
python3 figs/make_revision_evidence.py
```

Run this command from the source archive root, not from runtime/.
