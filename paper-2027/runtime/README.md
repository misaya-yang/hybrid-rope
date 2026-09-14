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
