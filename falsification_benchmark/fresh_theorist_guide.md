# Fresh-theorist instructions

You are evaluating a theory against a frozen historical benchmark. Your job is prediction only. Do not propose a new RoPE method, architecture, training recipe, or experiment.

## Blindness contract

You must be genuinely fresh to these hidden outcomes. If you have seen any hidden answer, source report, current-session discussion, or score from this benchmark, declare yourself ineligible and stop.

The benchmark coordinator gives you exactly two files:

1. `visible_packets/packets.json`
2. this guide

Do not inspect the repository, Git history, registry, hidden-answer directory, leakage audit, evaluator source, paper, reports, search engines, cached summaries, other agents, or external tools. Do not ask another model to retrieve the outcomes. Do not infer answers from filenames or modification times. The 16 predictions must be completed and frozen atomically before any hidden answer is revealed.

The visible packet is intentionally a minimal subset of facts known before each episode, not a complete historical state. Later packets omit earlier benchmark outcomes so the entire bundle can be predicted in one blind submission.

## Required prediction object

Start with a coordinator-generated template:

```bash
python3 -m evaluator template --output /tmp/theory_predictions.json
```

For every episode, replace all placeholders and submit:

```json
{
  "episode_id": "TFB-NNN",
  "predicted_direction": "A_BETTER",
  "direction_probabilities": {
    "A_BETTER": 0.40,
    "B_BETTER": 0.20,
    "PRACTICAL_TIE": 0.15,
    "CROSSOVER_OR_MIXED": 0.15,
    "INVALID_OR_UNRESOLVED": 0.10
  },
  "magnitude_predictions": {
    "the_packet_defined_target_id": 0.0
  },
  "qualitative_pattern_ids": ["ONE_OFFERED_PATTERN_ID"],
  "rationale": "A short precommitted derivation from the theory."
}
```

This snippet is schema-only and is not a prediction for any real episode.

Rules:

- `predicted_direction` must be one of the five classes and must be an argmax of the probability distribution.
- Include all five probabilities. Each must be at least `1e-6`; the sum must equal one within `1e-9`.
- Use every magnitude target ID exactly once. Values must be finite numbers in the packet's declared units.
- Choose only offered qualitative pattern IDs and obey each packet's min/max count.
- Keep episode order unchanged and include all 16 exactly once.
- Give a stable `theory_id` at the document top level. Do not alter it after outcome release.
- The rationale is unscored, but it should expose the theory's pre-result reasoning. Do not use it to hedge with mutually incompatible after-the-fact narratives.

Validate syntax before submission:

```bash
python3 -m evaluator validate --predictions /tmp/theory_predictions.json
```

Validation reads visible packets only. It does not access hidden answers.

## Atomic submission flow

1. Coordinator records the benchmark version and SHA-256 of both files supplied to you.
2. You produce and locally validate one complete prediction file.
3. Coordinator records its SHA-256 and timestamp; the file is now immutable.
4. Only then may the coordinator run the scorer with `hidden_answers/answers.json`.
5. Any edit after answer access is a new, ineligible submission.

Scoring has four frozen components: direction accuracy, five-class Brier calibration, magnitude accuracy under visible per-target scales, and qualitative set F1. The evaluator reports each component and a fixed composite. See the coordinator-only evaluator README after submission if an implementation audit is needed.

## Current-session exclusion

The agents that assembled, audited, or discussed this benchmark have already encountered historical outcomes and are prohibited from entering predictions. Their synthetic unit-test fixtures are not benchmark predictions. No score in the benchmark may be attributed to the benchmark-building session.

After scoring, stop. The benchmark does not authorize method recommendations, new theory construction, experiment planning, GPU use, or rerunning any historical arm.
