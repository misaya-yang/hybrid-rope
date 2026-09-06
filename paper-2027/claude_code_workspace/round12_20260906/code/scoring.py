#!/usr/bin/env python3
"""Round 12 scorers: strict group scorer, RULER official substring scorer, QA EM/F1.

All raw generation text is preserved in per-row receipts; scorers never edit it.
Official scores and strict scores are BOTH recorded and reported side by side;
neither replaces the other.
"""
from __future__ import annotations

import re
import string
from collections import Counter


# --------------------------------------------------------------------------
# Text normalization (SQuAD-style, used by EM/F1 only)
# --------------------------------------------------------------------------
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    text = text.lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return _WS.sub(" ", text).strip()


def exact_match(pred: str, golds: list[str]) -> bool:
    p = normalize_answer(pred)
    return any(p == normalize_answer(g) for g in golds)


def f1_score(pred: str, golds: list[str]) -> float:
    p_tokens = normalize_answer(pred).split()
    if not p_tokens:
        return 0.0
    best = 0.0
    for g in golds:
        g_tokens = normalize_answer(g).split()
        if not g_tokens:
            continue
        common = Counter(p_tokens) & Counter(g_tokens)
        n_same = sum(common.values())
        if n_same == 0:
            continue
        prec = n_same / len(p_tokens)
        rec = n_same / len(g_tokens)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


# --------------------------------------------------------------------------
# RULER official scorer: every expected answer string must appear (substring)
# in the raw generated text. This is the published protocol; it is reported
# alongside the strict scorer, not instead of it.
# --------------------------------------------------------------------------
def ruler_official(pred_raw: str, expected: list[str]) -> bool:
    return all(e in pred_raw for e in expected)


# --------------------------------------------------------------------------
# Strict (iron-curtain) scorer, verbatim convention from rounds 10-11:
# success = full answer exact against accepted aliases AND ended with EOS.
# Group-level success for multi-world tasks requires ALL worlds to succeed.
# --------------------------------------------------------------------------
def strict_row(generated_text: str, ended_with_eos: bool, accepted_full_answers: list[str]) -> bool:
    if not ended_with_eos:
        return False
    gen = generated_text.strip()
    return any(gen == a or gen.startswith(a) for a in accepted_full_answers) \
        if accepted_full_answers else False


def strict_group(world_results: dict[str, bool]) -> bool:
    return len(world_results) >= 1 and all(world_results.values())


# --------------------------------------------------------------------------
# Per-row receipt
# --------------------------------------------------------------------------
def score_row(row: dict, generated_text: str, ended_with_eos: bool, stop_reason: str) -> dict:
    """Score one generation row with every declared scorer."""
    golds = row.get("accepted_full_answers") or []
    out = {
        "row_id": row.get("row_id") or row.get("semantic_id"),
        "group_id": row.get("group_id") or row.get("semantic_id"),
        "world": row.get("world", "0"),
        "family": row.get("family"),
        "layout": row.get("layout"),
        "length_cap": row.get("length_cap"),
        "unmodified_output_text": generated_text,
        "ended_with_eos": bool(ended_with_eos),
        "stop_reason": stop_reason,
        "strict_exact_eos": strict_row(generated_text, ended_with_eos, golds),
        "ruler_official_contains": ruler_official(generated_text, golds) if golds else False,
        "qa_em": exact_match(generated_text, golds) if golds else False,
        "qa_f1": round(f1_score(generated_text, golds), 4) if golds else 0.0,
        "lenient_contains_any_gold": any(g in generated_text for g in golds) if golds else False,
    }
    return out
