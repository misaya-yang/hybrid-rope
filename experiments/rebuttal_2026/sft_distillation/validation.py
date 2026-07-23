#!/usr/bin/env python3
"""Programmatic correctness, leakage, and near-duplicate gates."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .protocol import (
    BANNED_MARKERS,
    MAX_SFT_TOKENS,
    GenerationJob,
    TaskSkeleton,
    assemble_context,
    solve_skeleton,
)


REQUIRED_RAW_FIELDS = {
    "task_type",
    "structured_facts",
    "oracle_answer",
    "context",
    "instruction",
    "teacher_answer",
    "evidence",
    "difficulty",
    "token_count",
    "generator_version",
}


def normalize_answer(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def render_user_message(context: str, instruction: str) -> str:
    return (
        "Context:\n"
        + context.strip()
        + "\n\nQuestion:\n"
        + instruction.strip()
        + "\n\nRespond with only the requested answer."
    )


def _teacher_surface_text(teacher: Mapping[str, Any]) -> str:
    pieces = [
        teacher.get("title", ""),
        teacher.get("opening", ""),
        *teacher.get("transitions", []),
        *teacher.get("distractor_paragraphs", []),
    ]
    return "\n".join(str(piece) for piece in pieces)


def _surface_fragment_is_safe(
    fragment: str, skeleton: TaskSkeleton
) -> bool:
    folded = fragment.casefold()
    if any(term.casefold() in folded for term in skeleton.reserved_terms):
        return False
    if any(marker.casefold() in folded for marker in BANNED_MARKERS):
        return False
    if normalize_answer(skeleton.oracle_answer) in normalize_answer(fragment):
        return False
    if any(
        marker in folded
        for marker in (
            "the answer",
            "the query",
            "the question",
            "targeted query",
            "according to the context",
            "real-world",
            "worldwide",
        )
    ):
        return False
    # Teacher distractors are required to be generic fictional prose.  Reject
    # capitalized tokens after a sentence's first word; this filters named
    # organizations, people, cities, countries, and historical labels without
    # needing an incomplete real-world entity denylist.
    for sentence in re.split(r"(?<=[.!?])\s+", fragment.strip()):
        tokens = re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", sentence)
        for token in tokens[1:]:
            if token[0].isupper():
                return False
    return True


def _split_surface_paragraph(value: str, max_words: int = 110) -> List[str]:
    """Split teacher prose without inventing or rewriting any token."""
    words = str(value).strip().split()
    if not words:
        return []
    return [
        " ".join(words[index : index + max_words])
        for index in range(0, len(words), max_words)
    ]


def fit_teacher_surface(
    *,
    job: GenerationJob,
    skeleton: TaskSkeleton,
    teacher: Mapping[str, Any],
    tokenizer: Any,
) -> Tuple[Dict[str, Any], str]:
    """Select safe teacher-written fragments to fit the registered token bin."""
    title = str(teacher.get("title", "")).strip()
    opening = str(teacher.get("opening", "")).strip()
    if not _surface_fragment_is_safe(title, skeleton):
        title = ""
    if not _surface_fragment_is_safe(opening, skeleton):
        opening = ""

    transition_units: List[str] = []
    for paragraph in teacher.get("transitions", []):
        for fragment in _split_surface_paragraph(str(paragraph)):
            if _surface_fragment_is_safe(fragment, skeleton):
                transition_units.append(fragment)
    distractor_units: List[str] = []
    for paragraph in teacher.get("distractor_paragraphs", []):
        for fragment in _split_surface_paragraph(str(paragraph)):
            if _surface_fragment_is_safe(fragment, skeleton):
                distractor_units.append(fragment)

    selected_transitions: List[str] = []
    selected_distractors: List[str] = []
    fitted: Dict[str, Any] = {
        **dict(teacher),
        "title": title,
        "opening": opening,
        "transitions": selected_transitions,
        "distractor_paragraphs": selected_distractors,
    }
    context = assemble_context(skeleton, fitted, seed=job.seed)
    current_tokens = len(tokenizer.encode(context, add_special_tokens=False))
    target = min(job.target_context_tokens, job.max_context_tokens - 32)

    # Interleave transition and distractor fragments, but require distractors
    # to dominate the added text.  A fragment that would cross the hard upper
    # bound is skipped rather than truncated semantically.
    units: List[Tuple[str, str]] = []
    for index in range(max(len(transition_units), len(distractor_units))):
        if index < len(distractor_units):
            units.append(("distractor", distractor_units[index]))
        if index < len(transition_units):
            units.append(("transition", transition_units[index]))
    for kind, fragment in units:
        destination = (
            selected_distractors if kind == "distractor" else selected_transitions
        )
        destination.append(fragment)
        candidate_context = assemble_context(skeleton, fitted, seed=job.seed)
        candidate_tokens = len(
            tokenizer.encode(candidate_context, add_special_tokens=False)
        )
        if candidate_tokens >= job.max_context_tokens:
            destination.pop()
            continue
        context = candidate_context
        current_tokens = candidate_tokens
        if (
            current_tokens >= target
            and current_tokens >= job.min_context_tokens
            and len(selected_distractors) >= 4
        ):
            break
    return fitted, context


def _question_required_terms(skeleton: TaskSkeleton) -> Tuple[str, ...]:
    facts = skeleton.structured_facts
    if skeleton.task_type == "information_retrieval":
        return (
            str(facts["query"]["entity"]),
            "material",
        )
    if skeleton.task_type == "relation_state_tracking":
        return (
            str(facts["query"]["item"]),
            "where",
        )
    if skeleton.task_type == "aggregation_statistics":
        return (
            str(facts["query"]["site"]),
            "total",
        )
    if skeleton.task_type == "multi_hop_qa":
        return (
            str(facts["query"]["start_person"]),
            "status",
        )
    raise ValueError(f"unknown task type {skeleton.task_type!r}")


def validate_instruction_scope(
    skeleton: TaskSkeleton, instruction: str
) -> Tuple[str, ...]:
    """Reject questions that disclose facts the context is meant to resolve."""
    facts = skeleton.structured_facts
    if skeleton.task_type == "information_retrieval":
        allowed = {str(facts["query"]["entity"]).casefold()}
    elif skeleton.task_type == "relation_state_tracking":
        allowed = {str(facts["query"]["item"]).casefold()}
    elif skeleton.task_type == "aggregation_statistics":
        allowed = {str(facts["query"]["site"]).casefold()}
    elif skeleton.task_type == "multi_hop_qa":
        allowed = {str(facts["query"]["start_person"]).casefold()}
    else:
        raise ValueError(f"unknown task type {skeleton.task_type!r}")
    folded = instruction.casefold()
    leaked = sorted(
        {
            term
            for term in skeleton.reserved_terms
            if term.casefold() in folded and term.casefold() not in allowed
        },
        key=str.casefold,
    )
    if not leaked:
        return ()
    return (
        "instruction exposes non-query task terms: "
        + ",".join(leaked[:4]),
    )


@dataclass(frozen=True)
class CandidateValidation:
    errors: Tuple[str, ...]
    context: str
    instruction: str
    teacher_answer: str
    evidence: Tuple[str, ...]
    context_token_count: int
    sft_token_count: int

    @property
    def passed(self) -> bool:
        return not self.errors


def validate_teacher_candidate(
    *,
    job: GenerationJob,
    skeleton: TaskSkeleton,
    teacher: Mapping[str, Any],
    tokenizer: Any,
) -> CandidateValidation:
    errors: List[str] = []
    required_keys = {
        "title",
        "opening",
        "transitions",
        "distractor_paragraphs",
        "instruction",
        "teacher_answer",
        "evidence",
    }
    missing = sorted(required_keys - set(teacher))
    if missing:
        errors.append("teacher JSON missing keys: " + ",".join(missing))
    for key in ("title", "opening", "instruction", "teacher_answer"):
        if not isinstance(teacher.get(key), str) or not str(
            teacher.get(key, "")
        ).strip():
            errors.append(f"teacher field {key} must be a non-empty string")
    for key in ("transitions", "distractor_paragraphs", "evidence"):
        value = teacher.get(key)
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            errors.append(f"teacher field {key} must be a list of strings")

    fitted_teacher, context = fit_teacher_surface(
        job=job,
        skeleton=skeleton,
        teacher=teacher,
        tokenizer=tokenizer,
    )
    instruction = str(teacher.get("instruction", "")).strip()
    teacher_answer = str(teacher.get("teacher_answer", "")).strip()
    evidence_raw = teacher.get("evidence", [])
    evidence = tuple(
        str(value).strip() for value in evidence_raw
    ) if isinstance(evidence_raw, list) else ()

    if teacher_answer != skeleton.oracle_answer:
        errors.append(
            "teacher_answer differs from the byte-exact oracle_answer"
        )
    try:
        solved = solve_skeleton(skeleton)
    except ValueError as error:
        errors.append(f"oracle solver failed: {error}")
    else:
        if solved != skeleton.oracle_answer:
            errors.append("program oracle solver disagrees with oracle_answer")

    if sorted(evidence) != sorted(skeleton.required_evidence):
        errors.append("evidence does not exactly match required evidence")
    for item in evidence:
        if context.count(item) != 1:
            errors.append(
                "evidence is absent or non-unique in context: "
                + hashlib.sha256(item.encode("utf-8")).hexdigest()[:12]
            )
    for fact in skeleton.canonical_facts:
        if context.count(fact) != 1:
            errors.append(
                "canonical fact is absent or duplicated: "
                + hashlib.sha256(fact.encode("utf-8")).hexdigest()[:12]
            )

    surface = _teacher_surface_text(fitted_teacher)
    surface_folded = surface.casefold()
    leaked_reserved = [
        term
        for term in skeleton.reserved_terms
        if term.casefold() in surface_folded
    ]
    if leaked_reserved:
        # Teacher-written prose is forbidden from making any claim involving
        # canonical entities/values.  Only immutable program facts may do so.
        errors.append(
            "teacher prose mentions reserved task terms: "
            + ",".join(sorted(leaked_reserved)[:4])
        )
    if normalize_answer(skeleton.oracle_answer) in normalize_answer(surface):
        errors.append("teacher prose leaks the oracle answer")

    combined = (context + "\n" + instruction).casefold()
    for marker in BANNED_MARKERS:
        if marker.casefold() in combined:
            errors.append(f"forbidden benchmark/style marker: {marker}")

    if normalize_answer(skeleton.oracle_answer) in normalize_answer(instruction):
        errors.append("instruction leaks the oracle answer")
    instruction_folded = instruction.casefold()
    for required in _question_required_terms(skeleton):
        if required.casefold() not in instruction_folded:
            errors.append(f"instruction lost required query term: {required}")
    errors.extend(validate_instruction_scope(skeleton, instruction))

    token_ids = tokenizer.encode(context, add_special_tokens=False)
    context_tokens = len(token_ids)
    if not (job.min_context_tokens <= context_tokens < job.max_context_tokens):
        errors.append(
            "context token count outside target bin: "
            f"{context_tokens} not in [{job.min_context_tokens},"
            f"{job.max_context_tokens})"
        )
    user = render_user_message(context, instruction)
    sft_tokens = len(
        tokenizer.encode(user, add_special_tokens=False)
    ) + len(
        tokenizer.encode(teacher_answer, add_special_tokens=False)
    )
    if sft_tokens > MAX_SFT_TOKENS:
        errors.append(
            f"rendered SFT example exceeds {MAX_SFT_TOKENS} tokens: {sft_tokens}"
        )

    if len(fitted_teacher.get("distractor_paragraphs", [])) < 4:
        errors.append("fewer than four teacher distractor paragraphs")
    if len(context) < 100:
        errors.append("assembled context is implausibly short")
    return CandidateValidation(
        errors=tuple(dict.fromkeys(errors)),
        context=context,
        instruction=instruction,
        teacher_answer=teacher_answer,
        evidence=evidence,
        context_token_count=context_tokens,
        sft_token_count=sft_tokens,
    )


WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)


def word_shingles(text: str, width: int = 5) -> Set[str]:
    words = WORD_RE.findall(text.casefold())
    if len(words) < width:
        return {" ".join(words)} if words else set()
    return {
        " ".join(words[index : index + width])
        for index in range(len(words) - width + 1)
    }


def simhash64(shingles: Iterable[str]) -> int:
    weights = [0] * 64
    count = 0
    for shingle in shingles:
        count += 1
        value = int.from_bytes(
            hashlib.blake2b(
                shingle.encode("utf-8"), digest_size=8
            ).digest(),
            "big",
        )
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    if count == 0:
        return 0
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    # Python 3.9 compatibility (int.bit_count is unavailable here).
    return bin(left ^ right).count("1")


@dataclass
class _DedupRecord:
    sample_id: str
    simhash: int
    shingles: Set[str]


class DedupIndex:
    """Band-indexed near-duplicate detector for bounded-cost dataset checks."""

    def __init__(self, similarity_threshold: float = 0.82) -> None:
        self.similarity_threshold = float(similarity_threshold)
        self.records: Dict[str, _DedupRecord] = {}
        self.buckets: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
        self.exact_hashes: Dict[str, str] = {}

    @staticmethod
    def _bands(value: int) -> Iterable[Tuple[int, int]]:
        for band in range(4):
            yield band, (value >> (16 * band)) & 0xFFFF

    def check(self, sample_id: str, text: str) -> Optional[Dict[str, Any]]:
        exact = hashlib.sha256(
            re.sub(r"\s+", " ", text.strip()).encode("utf-8")
        ).hexdigest()
        if exact in self.exact_hashes:
            return {
                "kind": "exact",
                "other_sample_id": self.exact_hashes[exact],
                "similarity": 1.0,
            }
        shingles = word_shingles(text)
        signature = simhash64(shingles)
        candidates: Set[str] = set()
        for key in self._bands(signature):
            candidates.update(self.buckets.get(key, set()))
        for other_id in sorted(candidates):
            other = self.records[other_id]
            if hamming_distance(signature, other.simhash) > 16:
                continue
            union = shingles | other.shingles
            similarity = (
                len(shingles & other.shingles) / float(len(union))
                if union
                else 1.0
            )
            if similarity >= self.similarity_threshold:
                return {
                    "kind": "near",
                    "other_sample_id": other_id,
                    "similarity": similarity,
                }
        return None

    def add(self, sample_id: str, text: str) -> None:
        duplicate = self.check(sample_id, text)
        if duplicate is not None:
            raise ValueError(f"duplicate context: {duplicate}")
        exact = hashlib.sha256(
            re.sub(r"\s+", " ", text.strip()).encode("utf-8")
        ).hexdigest()
        shingles = word_shingles(text)
        signature = simhash64(shingles)
        record = _DedupRecord(sample_id, signature, shingles)
        self.records[sample_id] = record
        self.exact_hashes[exact] = sample_id
        for key in self._bands(signature):
            self.buckets[key].add(sample_id)


def validate_raw_record(record: Mapping[str, Any], tokenizer: Any) -> List[str]:
    errors: List[str] = []
    missing = sorted(REQUIRED_RAW_FIELDS - set(record))
    if missing:
        errors.append("missing raw fields: " + ",".join(missing))
        return errors
    if record["teacher_answer"] != record["oracle_answer"]:
        errors.append("raw teacher_answer differs from oracle_answer")
    context = str(record["context"])
    for evidence in record["evidence"]:
        if context.count(str(evidence)) != 1:
            errors.append("raw evidence not uniquely present in context")
    count = len(tokenizer.encode(context, add_special_tokens=False))
    if count != int(record["token_count"]):
        errors.append("raw token_count does not match tokenizer")
    return errors


def validate_split_isolation(
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Dict[str, Any]:
    worlds: Dict[str, Set[str]] = {}
    families: Dict[str, Set[str]] = {}
    for split, rows in records_by_split.items():
        worlds[split] = {str(row["world_id"]) for row in rows}
        families[split] = {str(row["template_family"]) for row in rows}
    collisions = []
    split_names = sorted(records_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            world_overlap = sorted(worlds[left] & worlds[right])
            family_overlap = sorted(families[left] & families[right])
            if world_overlap or family_overlap:
                collisions.append(
                    {
                        "left": left,
                        "right": right,
                        "world_overlap": world_overlap,
                        "template_family_overlap": family_overlap,
                    }
                )
    return {
        "status": "PASS" if not collisions else "FAIL",
        "collisions": collisions,
        "world_counts": {
            split: len(values) for split, values in worlds.items()
        },
        "template_family_counts": {
            split: len(values) for split, values in families.items()
        },
    }
