# Deterministic evaluator

Run from the benchmark directory with the Python standard library only:

```bash
python3 -m evaluator template --output /tmp/tfb_predictions.json
python3 -m evaluator validate --predictions /tmp/tfb_predictions.json
python3 -m evaluator score \
  --predictions /path/to/frozen_predictions.json \
  --answers hidden_answers/answers.json \
  --output /tmp/tfb_score.json
python3 -m unittest evaluator.test_evaluator
```

The generated template is intentionally a placeholder, not a prediction. A fresh theorist must replace every field before validation.

## Frozen scoring

For each episode:

- direction correctness: exact class match, score `0` or `1`;
- probability: five-class Brier score `sum_k (p_k-y_k)^2`, reported raw; the component linearly maps the minimum Brier permitted by the `1e-6` probability floor to `1` and the worst-case Brier `2` to `0`;
- magnitude: each target receives `exp(-0.5*(absolute_error/scoring_scale)^2)`, averaged within episode; raw absolute errors are also reported;
- qualitative pattern: set F1 over the offered pattern IDs.

The composite is fixed at `0.25 direction + 0.25 probability + 0.30 magnitude + 0.20 qualitative`. Macro scores are unweighted means over the 16 chronological episodes. Probabilities must include all five classes, each at least `1e-6`, and sum to one within `1e-9`. The declared direction must be an argmax; tied argmaxes are accepted.

Hidden answers are coordinator-only until the prediction file has been atomically frozen. The evaluator never calls a network service, model, GPU, or repository experiment code.
