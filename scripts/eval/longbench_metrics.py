"""Dependency-light official-style metrics for the frozen LongBench subset."""

from __future__ import annotations

import re
import string
from typing import Iterable


TASK_METRIC_MAP = {
    "narrativeqa": "qa_f1",
    "qasper": "qa_f1",
    "multifieldqa_en": "qa_f1",
    "hotpotqa": "qa_f1",
    "2wikimqa": "qa_f1",
    "gov_report": "rouge_l_f1",
}


def normalize_text(value: str) -> str:
    value = str(value).lower()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = "".join(character for character in value if character not in string.punctuation)
    return " ".join(value.split())


def token_f1(prediction: list[str], reference: list[str]) -> float:
    if not prediction or not reference:
        return 0.0
    overlap = sum(
        min(prediction.count(token), reference.count(token))
        for token in set(prediction)
    )
    if overlap <= 0:
        return 0.0
    precision = overlap / len(prediction)
    recall = overlap / len(reference)
    return float(2 * precision * recall / (precision + recall))


def qa_f1_score(prediction: str, references: Iterable[str]) -> float:
    pred = normalize_text(prediction).split()
    return float(max(
        (token_f1(pred, normalize_text(reference).split()) for reference in references),
        default=0.0,
    ))


def lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    state = [0] * (len(right) + 1)
    for left_value in left:
        previous = 0
        for index, right_value in enumerate(right, start=1):
            current = state[index]
            if left_value == right_value:
                state[index] = previous + 1
            else:
                state[index] = max(state[index], state[index - 1])
            previous = current
    return state[-1]


def rouge_l_f1(prediction: str, references: Iterable[str]) -> float:
    pred = normalize_text(prediction).split()
    if not pred:
        return 0.0
    best = 0.0
    for reference in references:
        gold = normalize_text(reference).split()
        overlap = lcs_length(pred, gold)
        if overlap <= 0:
            continue
        precision = overlap / len(pred)
        recall = overlap / len(gold)
        best = max(best, 2 * precision * recall / (precision + recall))
    return float(best)


def post_process_prediction(_task: str, prediction: str) -> str:
    return str(prediction).strip()


def score_prediction(
    task: str,
    metric: str,
    prediction: str,
    references: list[str],
    _all_classes: list[str],
) -> float:
    if task not in TASK_METRIC_MAP or TASK_METRIC_MAP[task] != metric:
        raise ValueError(f"metric identity drift for {task}: {metric}")
    if metric == "qa_f1":
        return qa_f1_score(prediction, references)
    if metric == "rouge_l_f1":
        return rouge_l_f1(prediction, references)
    raise ValueError(f"unsupported metric: {metric}")
