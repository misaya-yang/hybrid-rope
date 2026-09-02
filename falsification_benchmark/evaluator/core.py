from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


DIRECTION_CLASSES = (
    "A_BETTER",
    "B_BETTER",
    "PRACTICAL_TIE",
    "CROSSOVER_OR_MIXED",
    "INVALID_OR_UNRESOLVED",
)

WEIGHTS = {
    "direction": 0.25,
    "probability": 0.25,
    "magnitude": 0.30,
    "qualitative": 0.20,
}


class BenchmarkError(ValueError):
    """Raised when an artifact or prediction violates the frozen contract."""


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise BenchmarkError(f"{path}: top-level JSON must be an object")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkError(f"{label}: expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise BenchmarkError(f"{label}: expected a finite number")
    return number


def _packet_index(packet_doc: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    packets = packet_doc.get("packets")
    if not isinstance(packets, list) or not packets:
        raise BenchmarkError("packet document must contain a non-empty packets list")
    declared = tuple(packet_doc.get("direction_classes", ()))
    if declared != DIRECTION_CLASSES:
        raise BenchmarkError("packet direction_classes do not match evaluator contract")
    expected_count = packet_doc.get("packet_count")
    if expected_count != len(packets):
        raise BenchmarkError("packet_count does not match packets length")

    index: dict[str, dict[str, Any]] = {}
    previous_order = 0
    for packet in packets:
        if not isinstance(packet, dict):
            raise BenchmarkError("each packet must be an object")
        episode_id = packet.get("episode_id")
        order = packet.get("execution_order")
        if not isinstance(episode_id, str) or not episode_id:
            raise BenchmarkError("packet episode_id must be a non-empty string")
        if episode_id in index:
            raise BenchmarkError(f"duplicate packet episode_id: {episode_id}")
        if not isinstance(order, int) or order != previous_order + 1:
            raise BenchmarkError("packets must use contiguous chronological execution_order")
        contract = packet.get("prediction_contract")
        if not isinstance(contract, dict):
            raise BenchmarkError(f"{episode_id}: missing prediction_contract")
        targets = contract.get("magnitude_targets")
        options = contract.get("qualitative_options")
        if not isinstance(targets, list) or not targets:
            raise BenchmarkError(f"{episode_id}: magnitude_targets must be non-empty")
        if not isinstance(options, list) or not options:
            raise BenchmarkError(f"{episode_id}: qualitative_options must be non-empty")
        target_ids: set[str] = set()
        for target in targets:
            target_id = target.get("id") if isinstance(target, dict) else None
            scale = target.get("scoring_scale") if isinstance(target, dict) else None
            if not isinstance(target_id, str) or not target_id or target_id in target_ids:
                raise BenchmarkError(f"{episode_id}: invalid or duplicate magnitude target")
            if _finite_number(scale, f"{episode_id}.{target_id}.scoring_scale") <= 0:
                raise BenchmarkError(f"{episode_id}.{target_id}: scoring_scale must be positive")
            target_ids.add(target_id)
        option_ids = [option.get("id") for option in options if isinstance(option, dict)]
        if len(option_ids) != len(options) or any(not isinstance(item, str) or not item for item in option_ids):
            raise BenchmarkError(f"{episode_id}: invalid qualitative option")
        if len(set(option_ids)) != len(option_ids):
            raise BenchmarkError(f"{episode_id}: duplicate qualitative option")
        qmin = contract.get("qualitative_min")
        qmax = contract.get("qualitative_max")
        if not isinstance(qmin, int) or not isinstance(qmax, int) or not 0 <= qmin <= qmax <= len(options):
            raise BenchmarkError(f"{episode_id}: invalid qualitative selection bounds")
        index[episode_id] = packet
        previous_order = order
    return packets, index


def _answer_index(answer_doc: dict[str, Any], packet_ids: list[str]) -> dict[str, dict[str, Any]]:
    answers = answer_doc.get("answers")
    if not isinstance(answers, list):
        raise BenchmarkError("answer document must contain an answers list")
    if answer_doc.get("answer_count") != len(answers):
        raise BenchmarkError("answer_count does not match answers length")
    answer_ids = [answer.get("episode_id") for answer in answers if isinstance(answer, dict)]
    if answer_ids != packet_ids:
        raise BenchmarkError("answer order/set must exactly match chronological packet order")
    return {answer["episode_id"]: answer for answer in answers}


def validate_predictions(packet_doc: dict[str, Any], prediction_doc: dict[str, Any]) -> list[dict[str, Any]]:
    packets, packet_index = _packet_index(packet_doc)
    predictions = prediction_doc.get("predictions")
    if not isinstance(predictions, list):
        raise BenchmarkError("prediction document must contain a predictions list")
    if prediction_doc.get("benchmark_id") != packet_doc.get("benchmark_id"):
        raise BenchmarkError("prediction benchmark_id mismatch")
    prediction_ids = [item.get("episode_id") for item in predictions if isinstance(item, dict)]
    packet_ids = [packet["episode_id"] for packet in packets]
    if prediction_ids != packet_ids:
        raise BenchmarkError("predictions must exactly match chronological packet order")

    floor = _finite_number(packet_doc.get("probability_floor", 0.000001), "probability_floor")
    if not 0 <= floor < 0.2:
        raise BenchmarkError("probability_floor must be in [0, 0.2)")

    for item in predictions:
        if not isinstance(item, dict):
            raise BenchmarkError("each prediction must be an object")
        episode_id = item["episode_id"]
        packet = packet_index[episode_id]
        contract = packet["prediction_contract"]
        direction = item.get("predicted_direction")
        if direction not in DIRECTION_CLASSES:
            raise BenchmarkError(f"{episode_id}: invalid predicted_direction")
        probabilities = item.get("direction_probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(DIRECTION_CLASSES):
            raise BenchmarkError(f"{episode_id}: direction_probabilities must contain exactly five classes")
        numeric_probabilities = {
            key: _finite_number(value, f"{episode_id}.direction_probabilities.{key}")
            for key, value in probabilities.items()
        }
        if any(value < floor or value > 1 for value in numeric_probabilities.values()):
            raise BenchmarkError(f"{episode_id}: probabilities must be within [floor, 1]")
        if not math.isclose(sum(numeric_probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise BenchmarkError(f"{episode_id}: probabilities must sum to 1 within 1e-9")
        max_probability = max(numeric_probabilities.values())
        if numeric_probabilities[direction] < max_probability - 1e-12:
            raise BenchmarkError(f"{episode_id}: predicted_direction must be an argmax of probabilities")

        expected_targets = {target["id"] for target in contract["magnitude_targets"]}
        magnitudes = item.get("magnitude_predictions")
        if not isinstance(magnitudes, dict) or set(magnitudes) != expected_targets:
            raise BenchmarkError(f"{episode_id}: magnitude_predictions keys mismatch")
        for target_id, value in magnitudes.items():
            _finite_number(value, f"{episode_id}.magnitude_predictions.{target_id}")

        patterns = item.get("qualitative_pattern_ids")
        if not isinstance(patterns, list) or len(patterns) != len(set(patterns)):
            raise BenchmarkError(f"{episode_id}: qualitative_pattern_ids must be a unique list")
        allowed = {option["id"] for option in contract["qualitative_options"]}
        if any(pattern not in allowed for pattern in patterns):
            raise BenchmarkError(f"{episode_id}: unknown qualitative pattern")
        if not contract["qualitative_min"] <= len(patterns) <= contract["qualitative_max"]:
            raise BenchmarkError(f"{episode_id}: qualitative selection count outside bounds")
        rationale = item.get("rationale", "")
        if not isinstance(rationale, str):
            raise BenchmarkError(f"{episode_id}: rationale must be a string when supplied")
    return predictions


def _f1(predicted: set[str], observed: set[str]) -> float:
    if not predicted and not observed:
        return 1.0
    if not predicted or not observed:
        return 0.0
    precision = len(predicted & observed) / len(predicted)
    recall = len(predicted & observed) / len(observed)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def score_predictions(
    packet_doc: dict[str, Any], prediction_doc: dict[str, Any], answer_doc: dict[str, Any]
) -> dict[str, Any]:
    predictions = validate_predictions(packet_doc, prediction_doc)
    packets, packet_index = _packet_index(packet_doc)
    packet_ids = [packet["episode_id"] for packet in packets]
    answer_index = _answer_index(answer_doc, packet_ids)

    rows: list[dict[str, Any]] = []
    probability_floor = float(packet_doc.get("probability_floor", 0.000001))
    minimum_valid_brier = len(DIRECTION_CLASSES) * (len(DIRECTION_CLASSES) - 1) * probability_floor**2
    for prediction in predictions:
        episode_id = prediction["episode_id"]
        packet = packet_index[episode_id]
        answer = answer_index[episode_id]
        observed_direction = answer.get("observed_direction")
        if observed_direction not in DIRECTION_CLASSES:
            raise BenchmarkError(f"{episode_id}: invalid observed_direction")
        direction_score = float(prediction["predicted_direction"] == observed_direction)

        probabilities = {key: float(value) for key, value in prediction["direction_probabilities"].items()}
        brier = sum(
            (probabilities[key] - (1.0 if key == observed_direction else 0.0)) ** 2
            for key in DIRECTION_CLASSES
        )
        probability_score = max(0.0, min(1.0, (2.0 - brier) / (2.0 - minimum_valid_brier)))
        log_loss = -math.log(probabilities[observed_direction])

        answer_magnitudes = answer.get("magnitude_answers")
        if not isinstance(answer_magnitudes, dict):
            raise BenchmarkError(f"{episode_id}: magnitude_answers must be an object")
        target_map = {target["id"]: target for target in packet["prediction_contract"]["magnitude_targets"]}
        if set(answer_magnitudes) != set(target_map):
            raise BenchmarkError(f"{episode_id}: hidden magnitude keys mismatch packet")
        magnitude_rows: dict[str, Any] = {}
        magnitude_scores: list[float] = []
        absolute_errors: list[float] = []
        for target_id, target in target_map.items():
            observed = _finite_number(answer_magnitudes[target_id], f"{episode_id}.{target_id}.observed")
            predicted = _finite_number(
                prediction["magnitude_predictions"][target_id], f"{episode_id}.{target_id}.predicted"
            )
            scale = float(target["scoring_scale"])
            absolute_error = abs(predicted - observed)
            target_score = math.exp(-0.5 * (absolute_error / scale) ** 2)
            absolute_errors.append(absolute_error)
            magnitude_scores.append(target_score)
            magnitude_rows[target_id] = {
                "predicted": predicted,
                "observed": observed,
                "absolute_error": absolute_error,
                "scoring_scale": scale,
                "score": target_score,
            }
        magnitude_score = sum(magnitude_scores) / len(magnitude_scores)

        observed_patterns = answer.get("qualitative_pattern_ids")
        if not isinstance(observed_patterns, list):
            raise BenchmarkError(f"{episode_id}: qualitative_pattern_ids must be a list")
        allowed_patterns = {
            option["id"] for option in packet["prediction_contract"]["qualitative_options"]
        }
        if any(pattern not in allowed_patterns for pattern in observed_patterns):
            raise BenchmarkError(f"{episode_id}: hidden qualitative pattern not offered in visible packet")
        qualitative_score = _f1(set(prediction["qualitative_pattern_ids"]), set(observed_patterns))

        composite = (
            WEIGHTS["direction"] * direction_score
            + WEIGHTS["probability"] * probability_score
            + WEIGHTS["magnitude"] * magnitude_score
            + WEIGHTS["qualitative"] * qualitative_score
        )
        rows.append(
            {
                "episode_id": episode_id,
                "direction_score": direction_score,
                "probability_score": probability_score,
                "brier_score": brier,
                "log_loss": log_loss,
                "magnitude_score": magnitude_score,
                "magnitude_mean_absolute_error": sum(absolute_errors) / len(absolute_errors),
                "magnitude_details": magnitude_rows,
                "qualitative_score": qualitative_score,
                "composite_score": composite,
            }
        )

    def mean(field: str) -> float:
        return sum(float(row[field]) for row in rows) / len(rows)

    return {
        "schema_version": "1.0",
        "benchmark_id": packet_doc["benchmark_id"],
        "episode_count": len(rows),
        "weights": WEIGHTS,
        "probability_normalization": {
            "raw_brier_worst_case": 2.0,
            "minimum_valid_brier": minimum_valid_brier,
            "probability_floor": probability_floor,
        },
        "macro": {
            "direction_score": mean("direction_score"),
            "probability_score": mean("probability_score"),
            "brier_score": mean("brier_score"),
            "log_loss": mean("log_loss"),
            "magnitude_score": mean("magnitude_score"),
            "qualitative_score": mean("qualitative_score"),
            "composite_score": mean("composite_score"),
        },
        "episodes": rows,
    }


def prediction_template(packet_doc: dict[str, Any]) -> dict[str, Any]:
    packets, _ = _packet_index(packet_doc)
    return {
        "schema_version": "1.0",
        "benchmark_id": packet_doc["benchmark_id"],
        "theory_id": "REPLACE_WITH_STABLE_ID",
        "predictions": [
            {
                "episode_id": packet["episode_id"],
                "predicted_direction": "A_BETTER",
                "direction_probabilities": {key: 0.2 for key in DIRECTION_CLASSES},
                "magnitude_predictions": {
                    target["id"]: 0.0
                    for target in packet["prediction_contract"]["magnitude_targets"]
                },
                "qualitative_pattern_ids": [
                    packet["prediction_contract"]["qualitative_options"][0]["id"]
                ],
                "rationale": "REPLACE_WITH_PRECOMMITTED_REASONING",
            }
            for packet in packets
        ],
    }
