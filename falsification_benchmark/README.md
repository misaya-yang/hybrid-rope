# RoPE theory falsification benchmark

This directory contains a frozen, CPU-only benchmark built exclusively from completed historical experiments. It contains no new RoPE theory, method, or experiment proposal and authorizes no model execution.

## Deliverables

- `experiment_registry.json` / `.md`: chronological 16-episode registry with provenance grades;
- `visible_packets/packets.json`: the only empirical packet supplied to a fresh theorist;
- `hidden_answers/answers.json`: coordinator-only empirical answers;
- `leakage_audit/`: cross-episode automated audit and frozen 0-violation report;
- `evaluator/`: standard-library deterministic scorer and synthetic unit tests;
- `fresh_theorist_guide.md`: blindness, schema, atomic submission, and hard-stop instructions.

## Verification

From this directory:

```bash
python3 leakage_audit/audit.py \
  --json-output leakage_audit/report.json \
  --markdown-output leakage_audit/report.md
python3 -m unittest -v evaluator.test_evaluator
python3 -m evaluator template --output /tmp/tfb_predictions.json
python3 -m evaluator validate --predictions /tmp/tfb_predictions.json
```

Do not give the registry, hidden answers, audit artifacts, evaluator source, or other repository files to a fresh theorist before its prediction file is atomically frozen. The current benchmark-building session is ineligible to predict.
