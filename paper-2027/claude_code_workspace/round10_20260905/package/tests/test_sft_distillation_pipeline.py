from __future__ import annotations

import json
import threading
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.rebuttal_2026.sft_distillation.deepseek_client import (
    CompletionResult,
    DeepSeekClient,
)
from experiments.rebuttal_2026.sft_distillation.generate import (
    StateStore,
    process_job,
)
from experiments.rebuttal_2026.sft_distillation.protocol import (
    GENERATOR_VERSION,
    PILOT_SPLIT_COUNTS,
    PROMPT_VERSION,
    TASK_TYPES,
    GenerationJob,
    build_jobs,
    build_skeleton,
    build_teacher_messages,
    prompt_hash,
    solve_skeleton,
)
from experiments.rebuttal_2026.sft_distillation.validation import (
    DedupIndex,
    validate_split_isolation,
    validate_teacher_candidate,
)


class WordTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return text.split()


def test_audit_distribution_is_exact() -> None:
    jobs = build_jobs(stage="audit", num_samples=100)
    assert Counter(job.task_type for job in jobs) == {
        "information_retrieval": 30,
        "relation_state_tracking": 25,
        "aggregation_statistics": 20,
        "multi_hop_qa": 25,
    }
    assert Counter(job.length_bin for job in jobs) == {
        "512_1k": 20,
        "1k_2k": 30,
        "2k_3k": 25,
        "3k_4k": 25,
    }


def test_pilot_distribution_is_exact_per_split() -> None:
    jobs = build_jobs(stage="pilot", num_samples=3800)
    assert Counter(job.split for job in jobs) == PILOT_SPLIT_COUNTS
    by_split = defaultdict(list)
    for job in jobs:
        by_split[job.split].append(job)
    for split, count in PILOT_SPLIT_COUNTS.items():
        assert Counter(job.task_type for job in by_split[split]) == {
            "information_retrieval": int(count * 0.30),
            "relation_state_tracking": int(count * 0.25),
            "aggregation_statistics": int(count * 0.20),
            "multi_hop_qa": int(count * 0.25),
        }
        assert Counter(job.length_bin for job in by_split[split]) == {
            "512_1k": int(count * 0.20),
            "1k_2k": int(count * 0.30),
            "2k_3k": int(count * 0.25),
            "3k_4k": int(count * 0.25),
        }


def test_worlds_and_template_families_do_not_cross_splits() -> None:
    jobs = build_jobs(stage="pilot", num_samples=3800)
    rows = defaultdict(list)
    for job in jobs:
        rows[job.split].append(
            {
                "world_id": job.world_id,
                "template_family": job.template_family,
            }
        )
    assert validate_split_isolation(rows)["status"] == "PASS"


@pytest.mark.parametrize("task_type", TASK_TYPES)
def test_program_owns_unique_oracle(task_type: str) -> None:
    base = next(
        job
        for job in build_jobs(stage="audit", num_samples=100)
        if job.task_type == task_type
    )
    skeleton = build_skeleton(base)
    assert solve_skeleton(skeleton) == skeleton.oracle_answer
    assert skeleton.required_evidence
    assert set(skeleton.required_evidence).issubset(skeleton.canonical_facts)


def _valid_instruction(skeleton) -> str:
    facts = skeleton.structured_facts
    if skeleton.task_type == "information_retrieval":
        return (
            "What material is recorded for "
            f"{facts['query']['entity']}? Return only the material."
        )
    if skeleton.task_type == "relation_state_tracking":
        return (
            f"Where was the {facts['query']['item']} after all transfers? "
            "Return only the location."
        )
    if skeleton.task_type == "aggregation_statistics":
        return (
            f"What is the total for {facts['query']['site']}? "
            "Return the number followed by units."
        )
    return (
        "What is the access status reached from "
        f"{facts['query']['start_person']}? Return only the status."
    )


@pytest.mark.parametrize("task_type", TASK_TYPES)
def test_teacher_cannot_change_facts_or_answer(task_type: str) -> None:
    original = next(
        job
        for job in build_jobs(stage="audit", num_samples=100)
        if job.task_type == task_type
    )
    job = replace(
        original,
        min_context_tokens=180,
        max_context_tokens=900,
        target_context_tokens=300,
    )
    skeleton = build_skeleton(job)
    filler = " ".join(
        f"neutralword{index}" for index in range(260)
    )
    teacher = {
        "title": "Quarterly Operations Digest",
        "opening": "This fictional digest summarizes routine administrative work.",
        "transitions": [
            "The next section changes subject without modifying prior records.",
            "A separate office also reviewed ordinary procedural notes.",
            "Staff closed the period with a general planning discussion.",
        ],
        "distractor_paragraphs": [
            filler,
            "Routine teams discussed lighting, shelving, and meeting times.",
            "An unrelated committee reviewed stationery purchases.",
            "The cafeteria adjusted its seasonal menu.",
        ],
        "instruction": _valid_instruction(skeleton),
        "teacher_answer": skeleton.oracle_answer,
        "evidence": list(skeleton.required_evidence),
    }
    candidate = validate_teacher_candidate(
        job=job,
        skeleton=skeleton,
        teacher=teacher,
        tokenizer=WordTokenizer(),
    )
    assert candidate.passed, candidate.errors
    for fact in skeleton.canonical_facts:
        assert candidate.context.count(fact) == 1
    if task_type == "relation_state_tracking":
        fact_offsets = [
            candidate.context.index(fact)
            for fact in skeleton.canonical_facts
        ]
        assert fact_offsets == sorted(fact_offsets)

    changed = dict(teacher)
    changed["teacher_answer"] = "wrong"
    rejected = validate_teacher_candidate(
        job=job,
        skeleton=skeleton,
        teacher=changed,
        tokenizer=WordTokenizer(),
    )
    assert any("teacher_answer" in error for error in rejected.errors)


def test_real_named_entity_distractor_is_filtered() -> None:
    original = next(
        job
        for job in build_jobs(stage="audit", num_samples=100)
        if job.task_type == "information_retrieval"
    )
    job = replace(
        original,
        min_context_tokens=180,
        max_context_tokens=900,
        target_context_tokens=300,
    )
    skeleton = build_skeleton(job)
    filler = " ".join(f"neutralword{index}" for index in range(260))
    teacher = {
        "title": "fictional operations digest",
        "opening": "this digest describes generic administrative routines.",
        "transitions": [
            "the next section changes subject without altering prior records.",
            "another office reviewed ordinary procedural notes.",
            "staff closed the period with a general planning discussion.",
        ],
        "distractor_paragraphs": [
            "staff at the World Health Organization discussed routine filing.",
            filler,
            "routine teams discussed lighting shelving and meeting times.",
            "an unrelated committee reviewed stationery purchases.",
            "the cafeteria adjusted its seasonal menu.",
            "a generic office documented ordinary maintenance.",
        ],
        "instruction": _valid_instruction(skeleton),
        "teacher_answer": skeleton.oracle_answer,
        "evidence": list(skeleton.required_evidence),
    }
    candidate = validate_teacher_candidate(
        job=job,
        skeleton=skeleton,
        teacher=teacher,
        tokenizer=WordTokenizer(),
    )
    assert candidate.passed, candidate.errors
    assert "World Health Organization" not in candidate.context


def test_multi_hop_question_cannot_disclose_intermediate_chain() -> None:
    original = next(
        job
        for job in build_jobs(stage="audit", num_samples=100)
        if job.task_type == "multi_hop_qa"
    )
    job = replace(
        original,
        min_context_tokens=180,
        max_context_tokens=900,
        target_context_tokens=300,
    )
    skeleton = build_skeleton(job)
    facts = skeleton.structured_facts
    start = str(facts["query"]["start_person"])
    chain = list(skeleton.required_evidence)
    filler = " ".join(f"neutralword{index}" for index in range(260))
    teacher = {
        "title": "fictional operations digest",
        "opening": "this digest describes generic administrative routines.",
        "transitions": [
            "the next section changes subject without altering prior records.",
            "another office reviewed ordinary procedural notes.",
            "staff closed the period with a general planning discussion.",
        ],
        "distractor_paragraphs": [
            filler,
            "routine teams discussed lighting shelving and meeting times.",
            "an unrelated committee reviewed stationery purchases.",
            "the cafeteria adjusted its seasonal menu.",
        ],
        "instruction": (
            f"{chain[0]} {chain[1]} What is the status reached from {start}?"
        ),
        "teacher_answer": skeleton.oracle_answer,
        "evidence": chain,
    }
    candidate = validate_teacher_candidate(
        job=job,
        skeleton=skeleton,
        teacher=teacher,
        tokenizer=WordTokenizer(),
    )
    assert any(
        "instruction exposes non-query task terms" in error
        for error in candidate.errors
    )


def test_prompt_is_versioned_and_contains_no_secret() -> None:
    job = build_jobs(stage="audit", num_samples=100)[0]
    skeleton = build_skeleton(job)
    messages = build_teacher_messages(job, skeleton, attempt=1)
    payload = json.dumps(messages)
    assert PROMPT_VERSION in payload
    assert GENERATOR_VERSION in payload
    assert len(prompt_hash()) == 64
    assert "sk-" not in payload


def test_dedup_rejects_exact_and_near_copy() -> None:
    index = DedupIndex(similarity_threshold=0.80)
    original = " ".join(f"word{i}" for i in range(200))
    index.add("one", original)
    duplicate = index.check("two", original)
    assert duplicate is not None
    assert duplicate["kind"] == "exact"
    near = original + " a short unrelated ending"
    duplicate = index.check("three", near)
    assert duplicate is not None
    assert duplicate["similarity"] >= 0.80


def test_client_reads_model_only_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-placeholder")
    monkeypatch.setenv("DEEPSEEK_MODEL", "environment-model")
    client = DeepSeekClient(cache_dir=tmp_path)
    assert client.model == "environment-model"


def test_client_fails_closed_without_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        DeepSeekClient(cache_dir=tmp_path)


def test_state_store_resume_and_accepted_slot(
    tmp_path: Path,
) -> None:
    job = build_jobs(stage="audit", num_samples=1)[0]
    path = tmp_path / "state.sqlite3"
    state = StateStore(path, resume=False)
    state.register_job(job, slot_id=job.sample_id, replacement_index=0)
    state.record_result(
        job=job,
        raw_record={"sample_id": job.sample_id},
        attempt_records=[
            {
                "attempt_index": 0,
                "status": "accepted",
                "errors": [],
                "request_sha256": "a" * 64,
                "response_sha256": "b" * 64,
                "cache_hit": False,
                "usage": {"total_tokens": 10},
                "response_model": "environment-model",
                "created_at": "2026-07-24T00:00:00+00:00",
            }
        ],
        final_errors=[],
    )
    assert state.accepted_by_slot("audit")[job.sample_id]["sample_id"] == job.sample_id
    assert state.usage_summary("audit")["job_outcomes"] == {"accepted": 1}
    state.close()
    resumed = StateStore(path, resume=True)
    assert resumed.accepted_by_slot("audit")
    resumed.close()


def test_process_job_builds_required_raw_record() -> None:
    original = next(
        job
        for job in build_jobs(stage="audit", num_samples=100)
        if job.task_type == "information_retrieval"
    )
    job = replace(
        original,
        min_context_tokens=180,
        max_context_tokens=900,
        target_context_tokens=300,
    )
    skeleton = build_skeleton(job)
    content = {
        "title": "Quarterly Operations Digest",
        "opening": "This fictional digest covers ordinary administrative work.",
        "transitions": [
            "The report next turns to routine office planning.",
            "Another team reviewed ordinary schedules.",
            "The final section records general maintenance.",
        ],
        "distractor_paragraphs": [
            " ".join(f"neutralword{index}" for index in range(260)),
            "Staff discussed lighting and shelving.",
            "A committee reviewed stationery purchases.",
            "The cafeteria adjusted a seasonal menu.",
        ],
        "instruction": _valid_instruction(skeleton),
        "teacher_answer": skeleton.oracle_answer,
        "evidence": list(skeleton.required_evidence),
    }

    class FakeClient:
        def complete_json(self, messages, *, max_tokens):
            assert messages
            assert max_tokens > 0
            return CompletionResult(
                content=content,
                response_id="response",
                response_model="environment-model",
                system_fingerprint="fingerprint",
                finish_reason="stop",
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 300,
                    "total_tokens": 400,
                },
                request_sha256="a" * 64,
                response_sha256="b" * 64,
                cache_hit=False,
                created_at="2026-07-24T00:00:00+00:00",
            )

    record, attempts, errors = process_job(
        job=job,
        tokenizer=WordTokenizer(),
        client=FakeClient(),
        dedup=DedupIndex(),
        dedup_lock=threading.Lock(),
        prior_attempt_count=0,
        prior_errors=(),
        prior_token_count=None,
    )
    assert errors == ()
    assert record is not None
    assert record["teacher_answer"] == record["oracle_answer"]
    assert record["source_policy"]["external_text_sources"] == []
    assert attempts[-1]["status"] == "accepted"
