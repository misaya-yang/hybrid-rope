from __future__ import annotations

import copy
import unittest

from .core import BenchmarkError, DIRECTION_CLASSES, score_predictions, validate_predictions


def fixture() -> tuple[dict, dict]:
    packets = {
        "benchmark_id": "synthetic",
        "packet_count": 1,
        "direction_classes": list(DIRECTION_CLASSES),
        "probability_floor": 0.000001,
        "packets": [
            {
                "episode_id": "SYN-001",
                "execution_order": 1,
                "prediction_contract": {
                    "magnitude_targets": [{"id": "delta", "scoring_scale": 2.0}],
                    "qualitative_options": [{"id": "P1"}, {"id": "P2"}, {"id": "P3"}],
                    "qualitative_min": 1,
                    "qualitative_max": 2,
                },
            }
        ],
    }
    answers = {
        "answer_count": 1,
        "answers": [
            {
                "episode_id": "SYN-001",
                "observed_direction": "A_BETTER",
                "magnitude_answers": {"delta": -4.0},
                "qualitative_pattern_ids": ["P1", "P2"],
            }
        ],
    }
    return packets, answers


def prediction(probabilities: dict[str, float], direction: str = "A_BETTER") -> dict:
    return {
        "benchmark_id": "synthetic",
        "predictions": [
            {
                "episode_id": "SYN-001",
                "predicted_direction": direction,
                "direction_probabilities": probabilities,
                "magnitude_predictions": {"delta": -4.0},
                "qualitative_pattern_ids": ["P1", "P2"],
                "rationale": "synthetic fixture",
            }
        ],
    }


class EvaluatorTests(unittest.TestCase):
    def test_perfect_prediction_scores_one(self) -> None:
        packets, answers = fixture()
        probabilities = {key: 0.000001 for key in DIRECTION_CLASSES}
        probabilities["A_BETTER"] = 0.999996
        scored = score_predictions(packets, prediction(probabilities), answers)
        self.assertAlmostEqual(scored["macro"]["direction_score"], 1.0)
        self.assertAlmostEqual(scored["macro"]["magnitude_score"], 1.0)
        self.assertAlmostEqual(scored["macro"]["qualitative_score"], 1.0)
        self.assertAlmostEqual(scored["macro"]["probability_score"], 1.0)
        self.assertAlmostEqual(scored["macro"]["composite_score"], 1.0)

    def test_inverted_prediction_is_penalized(self) -> None:
        packets, answers = fixture()
        probabilities = {key: 0.000001 for key in DIRECTION_CLASSES}
        probabilities["B_BETTER"] = 0.999996
        value = prediction(probabilities, direction="B_BETTER")
        value["predictions"][0]["magnitude_predictions"]["delta"] = 8.0
        value["predictions"][0]["qualitative_pattern_ids"] = ["P3"]
        scored = score_predictions(packets, value, answers)
        self.assertEqual(scored["macro"]["direction_score"], 0.0)
        self.assertLess(scored["macro"]["composite_score"], 0.01)

    def test_uniform_probability_baseline_is_deterministic(self) -> None:
        packets, answers = fixture()
        probabilities = {key: 0.2 for key in DIRECTION_CLASSES}
        scored = score_predictions(packets, prediction(probabilities), answers)
        self.assertAlmostEqual(scored["macro"]["brier_score"], 0.8)
        self.assertAlmostEqual(scored["macro"]["probability_score"], 0.6, places=10)

    def test_partial_pattern_credit_uses_f1(self) -> None:
        packets, answers = fixture()
        probabilities = {key: 0.2 for key in DIRECTION_CLASSES}
        value = prediction(probabilities)
        value["predictions"][0]["qualitative_pattern_ids"] = ["P1"]
        scored = score_predictions(packets, value, answers)
        self.assertAlmostEqual(scored["macro"]["qualitative_score"], 2.0 / 3.0)

    def test_malformed_probability_sum_fails(self) -> None:
        packets, _ = fixture()
        probabilities = {key: 0.2 for key in DIRECTION_CLASSES}
        value = prediction(probabilities)
        value["predictions"][0]["direction_probabilities"]["A_BETTER"] = 0.3
        with self.assertRaises(BenchmarkError):
            validate_predictions(packets, value)

    def test_wrong_episode_order_fails(self) -> None:
        packets, _ = fixture()
        probabilities = {key: 0.2 for key in DIRECTION_CLASSES}
        value = prediction(probabilities)
        value["predictions"][0]["episode_id"] = "SYN-999"
        with self.assertRaises(BenchmarkError):
            validate_predictions(packets, value)


if __name__ == "__main__":
    unittest.main()
