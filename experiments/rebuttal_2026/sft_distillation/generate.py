#!/usr/bin/env python3
"""Generate, validate, audit, and export verified SFT distillation data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from transformers import AutoTokenizer

from .deepseek_client import DeepSeekClient, Pricing, utc_now, write_json_atomic
from .protocol import (
    AUDIT_COUNT,
    DEFAULT_SEED,
    GENERATOR_VERSION,
    LENGTH_BINS,
    PILOT_SPLIT_COUNTS,
    PROMPT_VERSION,
    TASK_TYPES,
    TASK_WEIGHTS,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    GenerationJob,
    TaskSkeleton,
    build_jobs,
    build_skeleton,
    build_teacher_messages,
    canonical_json,
    length_distribution,
    prompt_hash,
    sha256_text,
    task_distribution,
)
from .validation import (
    DedupIndex,
    render_user_message,
    validate_instruction_scope,
    validate_raw_record,
    validate_split_isolation,
    validate_teacher_candidate,
)


SCHEMA_VERSION = 1
MAX_SAMPLE_ATTEMPTS = 3
MAX_REPLACEMENTS_PER_SLOT = 4
DB_SCHEMA_VERSION = 1


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary.replace(path)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def load_tokenizer(
    name_or_path: str,
    *,
    revision: str,
    local_files_only: bool,
) -> Tuple[Any, Dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        revision=revision or None,
        local_files_only=bool(local_files_only),
        use_fast=True,
        trust_remote_code=False,
    )
    tokenizer.model_max_length = 1 << 60
    backend = tokenizer.backend_tokenizer.to_str()
    record = {
        "name_or_path": name_or_path,
        "revision": revision,
        "class": tokenizer.__class__.__name__,
        "vocab_size": int(len(tokenizer)),
        "backend_sha256": sha256_text(backend),
        "special_tokens_map": tokenizer.special_tokens_map,
        "all_special_ids": [int(value) for value in tokenizer.all_special_ids],
    }
    return tokenizer, record


class StateStore:
    def __init__(self, path: Path, *, resume: bool) -> None:
        self.path = path.resolve()
        existed = self.path.exists()
        if existed and not resume:
            raise FileExistsError(
                f"state database exists; pass --resume: {self.path}"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._create_schema()
        if not existed:
            self.connection.execute(
                "INSERT INTO metadata(key,value) VALUES(?,?)",
                ("db_schema_version", str(DB_SCHEMA_VERSION)),
            )
            self.connection.commit()
        else:
            version = self.connection.execute(
                "SELECT value FROM metadata WHERE key='db_schema_version'"
            ).fetchone()
            if version is None or int(version[0]) != DB_SCHEMA_VERSION:
                raise RuntimeError("state database schema mismatch")

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs(
                sample_id TEXT PRIMARY KEY,
                slot_id TEXT NOT NULL,
                replacement_index INTEGER NOT NULL,
                job_json TEXT NOT NULL,
                job_sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                raw_json TEXT,
                last_errors_json TEXT,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempts(
                sample_id TEXT NOT NULL,
                attempt_index INTEGER NOT NULL,
                status TEXT NOT NULL,
                errors_json TEXT NOT NULL,
                request_sha256 TEXT,
                response_sha256 TEXT,
                cache_hit INTEGER NOT NULL,
                usage_json TEXT NOT NULL,
                response_model TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY(sample_id, attempt_index)
            );
            """
        )
        self.connection.commit()

    def register_job(
        self,
        job: GenerationJob,
        *,
        slot_id: str,
        replacement_index: int,
    ) -> None:
        payload = canonical_json(asdict(job))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        existing = self.connection.execute(
            "SELECT job_sha256 FROM jobs WHERE sample_id=?", (job.sample_id,)
        ).fetchone()
        if existing is not None:
            if existing[0] != digest:
                raise RuntimeError(f"job definition drift for {job.sample_id}")
            return
        self.connection.execute(
            """
            INSERT INTO jobs(
                sample_id,slot_id,replacement_index,job_json,job_sha256,
                status,attempts,updated_at
            ) VALUES(?,?,?,?,?,'pending',0,?)
            """,
            (
                job.sample_id,
                slot_id,
                replacement_index,
                payload,
                digest,
                utc_now(),
            ),
        )
        self.connection.commit()

    def prior_attempts(
        self, sample_id: str
    ) -> Tuple[int, Tuple[str, ...], Optional[int]]:
        row = self.connection.execute(
            "SELECT attempts,last_errors_json FROM jobs WHERE sample_id=?",
            (sample_id,),
        ).fetchone()
        if row is None:
            return 0, (), None
        errors = tuple(json.loads(row[1])) if row[1] else ()
        token_count: Optional[int] = None
        for error in errors:
            if "context token count outside target bin:" in error:
                try:
                    token_count = int(error.split(":", 1)[1].split()[0])
                except (IndexError, ValueError):
                    pass
        return int(row[0]), errors, token_count

    def record_result(
        self,
        *,
        job: GenerationJob,
        raw_record: Optional[Mapping[str, Any]],
        attempt_records: Sequence[Mapping[str, Any]],
        final_errors: Sequence[str],
    ) -> None:
        for record in attempt_records:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO attempts(
                    sample_id,attempt_index,status,errors_json,request_sha256,
                    response_sha256,cache_hit,usage_json,response_model,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    job.sample_id,
                    int(record["attempt_index"]),
                    str(record["status"]),
                    canonical_json(record.get("errors", [])),
                    record.get("request_sha256"),
                    record.get("response_sha256"),
                    1 if record.get("cache_hit") else 0,
                    canonical_json(record.get("usage", {})),
                    record.get("response_model"),
                    str(record.get("created_at", utc_now())),
                ),
            )
        self.connection.execute(
            """
            UPDATE jobs
            SET status=?,attempts=?,raw_json=?,last_errors_json=?,updated_at=?
            WHERE sample_id=?
            """,
            (
                "accepted" if raw_record is not None else "discarded",
                max(
                    [
                        int(record["attempt_index"]) + 1
                        for record in attempt_records
                    ]
                    or [0]
                ),
                canonical_json(raw_record) if raw_record is not None else None,
                canonical_json(list(final_errors)),
                utc_now(),
                job.sample_id,
            ),
        )
        self.connection.commit()

    def accepted_by_slot(self, stage: str) -> Dict[str, Dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT slot_id,raw_json FROM jobs
            WHERE status='accepted' AND json_extract(job_json,'$.stage')=?
            ORDER BY slot_id,replacement_index
            """,
            (stage,),
        ).fetchall()
        result: Dict[str, Dict[str, Any]] = {}
        for slot_id, raw_json in rows:
            if slot_id in result:
                raise RuntimeError(f"multiple accepted rows for slot {slot_id}")
            result[str(slot_id)] = json.loads(raw_json)
        return result

    def job(self, sample_id: str) -> GenerationJob:
        row = self.connection.execute(
            "SELECT job_json FROM jobs WHERE sample_id=?", (sample_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown state-store job {sample_id}")
        return GenerationJob(**json.loads(row[0]))

    def invalidate_accepted_record(
        self, sample_id: str, errors: Sequence[str]
    ) -> None:
        row = self.connection.execute(
            "SELECT status FROM jobs WHERE sample_id=?", (sample_id,)
        ).fetchone()
        if row is None or row[0] != "accepted":
            raise RuntimeError(
                f"cannot invalidate non-accepted job {sample_id}"
            )
        self.connection.execute(
            """
            UPDATE jobs
            SET status='discarded',raw_json=NULL,last_errors_json=?,updated_at=?
            WHERE sample_id=?
            """,
            (canonical_json(list(errors)), utc_now(), sample_id),
        )
        self.connection.commit()

    def latest_replacement_index(self, slot_id: str) -> int:
        row = self.connection.execute(
            "SELECT MAX(replacement_index) FROM jobs WHERE slot_id=?",
            (slot_id,),
        ).fetchone()
        return -1 if row is None or row[0] is None else int(row[0])

    def usage_summary(self, stage: str) -> Dict[str, Any]:
        rows = self.connection.execute(
            """
            SELECT a.status,a.errors_json,a.cache_hit,a.usage_json
            FROM attempts a JOIN jobs j ON a.sample_id=j.sample_id
            WHERE json_extract(j.job_json,'$.stage')=?
            """,
            (stage,),
        ).fetchall()
        usage = Counter()
        failures = Counter()
        cache_hits = 0
        for status, errors_json, cache_hit, usage_json in rows:
            cache_hits += int(cache_hit)
            for key, value in json.loads(usage_json).items():
                usage[key] += int(value)
            if status != "accepted":
                errors = json.loads(errors_json)
                if errors:
                    for error in errors:
                        failures[str(error).split(":", 1)[0]] += 1
                else:
                    failures["unknown"] += 1
        job_rows = self.connection.execute(
            """
            SELECT status,COUNT(*) FROM jobs
            WHERE json_extract(job_json,'$.stage')=?
            GROUP BY status
            """,
            (stage,),
        ).fetchall()
        job_counts = {str(status): int(count) for status, count in job_rows}
        discarded_rows = self.connection.execute(
            """
            SELECT last_errors_json FROM jobs
            WHERE status='discarded'
              AND json_extract(job_json,'$.stage')=?
            """,
            (stage,),
        ).fetchall()
        discarded_reasons = Counter()
        for (errors_json,) in discarded_rows:
            for error in json.loads(errors_json or "[]"):
                discarded_reasons[str(error).split(":", 1)[0]] += 1
        return {
            "attempt_count": len(rows),
            "cache_hit_count": cache_hits,
            "uncached_attempt_count": len(rows) - cache_hits,
            "job_outcomes": job_counts,
            "usage": dict(usage),
            "failure_reasons": dict(sorted(failures.items())),
            "discarded_job_reasons": dict(sorted(discarded_reasons.items())),
        }

    def close(self) -> None:
        self.connection.close()


def _slot_id(job: GenerationJob) -> str:
    return job.sample_id.split("-r", 1)[0]


def replacement_job(
    original: GenerationJob, replacement_index: int
) -> GenerationJob:
    if replacement_index <= 0:
        return original
    return replace(
        original,
        sample_id=f"{_slot_id(original)}-r{replacement_index}",
        world_id=f"{original.world_id}:replacement-{replacement_index}",
        seed=original.seed + replacement_index * 97_409,
    )


def _record_from_candidate(
    *,
    job: GenerationJob,
    skeleton: TaskSkeleton,
    candidate: Any,
    completion: Any,
    attempt_index: int,
) -> Dict[str, Any]:
    return {
        "sample_id": job.sample_id,
        "slot_id": _slot_id(job),
        "stage": job.stage,
        "split": job.split,
        "task_type": job.task_type,
        "template_family": job.template_family,
        "world_id": job.world_id,
        "structured_facts": skeleton.structured_facts,
        "oracle_answer": skeleton.oracle_answer,
        "context": candidate.context,
        "instruction": candidate.instruction,
        "teacher_answer": candidate.teacher_answer,
        "evidence": list(candidate.evidence),
        "difficulty": skeleton.difficulty,
        "token_count": candidate.context_token_count,
        "sft_token_count": candidate.sft_token_count,
        "length_bin": job.length_bin,
        "target_context_tokens": job.target_context_tokens,
        "generator_version": GENERATOR_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash(),
        "job_sha256": job.fingerprint(),
        "oracle_validation": {
            "solver": "programmatic",
            "unique": True,
            "status": "PASS",
        },
        "source_policy": {
            "external_text_sources": [],
            "benchmark_templates_used": [],
            "canonical_facts_owned_by_program": True,
            "teacher_scope": "surface_form_and_unrelated_distractors_only",
        },
        "teacher_metadata": {
            "model": completion.response_model,
            "system_fingerprint": completion.system_fingerprint,
            "request_sha256": completion.request_sha256,
            "response_sha256": completion.response_sha256,
            "response_id_sha256": hashlib.sha256(
                completion.response_id.encode("utf-8")
            ).hexdigest(),
            "finish_reason": completion.finish_reason,
            "usage": completion.usage,
            "cache_hit": completion.cache_hit,
            "created_at": completion.created_at,
            "sample_attempt": attempt_index + 1,
        },
    }


def process_job(
    *,
    job: GenerationJob,
    tokenizer: Any,
    client: DeepSeekClient,
    dedup: DedupIndex,
    dedup_lock: threading.Lock,
    prior_attempt_count: int,
    prior_errors: Sequence[str],
    prior_token_count: Optional[int],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Tuple[str, ...]]:
    skeleton = build_skeleton(job)
    attempts: List[Dict[str, Any]] = []
    errors = tuple(prior_errors)
    token_count = prior_token_count
    teacher_history: List[Mapping[str, Any]] = []
    for attempt_index in range(prior_attempt_count, MAX_SAMPLE_ATTEMPTS):
        messages = build_teacher_messages(
            job,
            skeleton,
            attempt=attempt_index + 1,
            prior_errors=errors,
            prior_token_count=token_count,
        )
        attempt_record: Dict[str, Any] = {
            "attempt_index": attempt_index,
            "status": "failed",
            "errors": [],
            "request_sha256": None,
            "response_sha256": None,
            "cache_hit": False,
            "usage": {},
            "response_model": None,
            "created_at": utc_now(),
        }
        try:
            completion = client.complete_json(
                messages,
                max_tokens=min(10000, 2 * job.max_context_tokens + 1000),
            )
            attempt_record.update(
                {
                    "request_sha256": completion.request_sha256,
                    "response_sha256": completion.response_sha256,
                    "cache_hit": completion.cache_hit,
                    "usage": completion.usage,
                    "response_model": completion.response_model,
                    "created_at": completion.created_at,
                }
            )
            teacher_history.append(completion.content)
            merged_teacher = {
                **completion.content,
                "transitions": [
                    value
                    for response in teacher_history
                    for value in response.get("transitions", [])
                ],
                "distractor_paragraphs": [
                    value
                    for response in teacher_history
                    for value in response.get("distractor_paragraphs", [])
                ],
            }
            candidate = validate_teacher_candidate(
                job=job,
                skeleton=skeleton,
                teacher=merged_teacher,
                tokenizer=tokenizer,
            )
            errors = candidate.errors
            token_count = candidate.context_token_count
            if candidate.passed:
                record = _record_from_candidate(
                    job=job,
                    skeleton=skeleton,
                    candidate=candidate,
                    completion=completion,
                    attempt_index=attempt_index,
                )
                with dedup_lock:
                    duplicate = dedup.check(
                        job.sample_id,
                        record["context"] + "\n" + record["instruction"],
                    )
                    if duplicate is None:
                        dedup.add(
                            job.sample_id,
                            record["context"] + "\n" + record["instruction"],
                        )
                    else:
                        errors = (
                            "sample is highly similar to "
                            f"{duplicate['other_sample_id']} "
                            f"(similarity={duplicate['similarity']:.4f})",
                        )
                if not errors:
                    attempt_record["status"] = "accepted"
                    attempts.append(attempt_record)
                    return record, attempts, ()
            attempt_record["errors"] = list(errors)
        except Exception as error:
            # Request/prompt payloads and credentials are never included.
            errors = (f"api_or_parse_failure: {type(error).__name__}: {error}",)
            attempt_record["errors"] = list(errors)
        attempts.append(attempt_record)
    return None, attempts, tuple(errors)


def _expected_percentages(counts: Mapping[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    return {
        key: (100.0 * value / total if total else 0.0)
        for key, value in counts.items()
    }


def _records_by_split(
    records: Sequence[Mapping[str, Any]]
) -> Dict[str, List[Mapping[str, Any]]]:
    result: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        result[str(record["split"])].append(record)
    return dict(result)


def materialize(
    *,
    output_dir: Path,
    stage: str,
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    tokenizer_record: Mapping[str, Any],
    state: StateStore,
    pricing: Pricing,
    requested_samples: int,
    api_verification: Mapping[str, Any],
) -> Dict[str, Any]:
    sorted_records = sorted(records, key=lambda row: str(row["slot_id"]))
    raw_files: Dict[str, Dict[str, Any]] = {}
    message_files: Dict[str, Dict[str, Any]] = {}
    for split, rows in sorted(_records_by_split(sorted_records).items()):
        raw_path = output_dir / "raw" / f"{split}.jsonl"
        messages_path = output_dir / "messages" / f"{split}.jsonl"
        write_jsonl_atomic(raw_path, rows)
        messages = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": render_user_message(
                            str(row["context"]), str(row["instruction"])
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": str(row["teacher_answer"]),
                    },
                ]
            }
            for row in rows
        ]
        write_jsonl_atomic(messages_path, messages)
        raw_files[split] = {
            "path": str(raw_path.resolve()),
            "rows": len(rows),
            "sha256": sha256_file(raw_path),
            "bytes": raw_path.stat().st_size,
        }
        message_files[split] = {
            "path": str(messages_path.resolve()),
            "rows": len(messages),
            "sha256": sha256_file(messages_path),
            "bytes": messages_path.stat().st_size,
        }

    raw_errors = []
    for record in sorted_records:
        for error in validate_raw_record(record, tokenizer):
            raw_errors.append({"sample_id": record["sample_id"], "error": error})
    isolation = validate_split_isolation(_records_by_split(sorted_records))
    duplicate_index = DedupIndex()
    duplicate_errors = []
    for record in sorted_records:
        text = str(record["context"]) + "\n" + str(record["instruction"])
        duplicate = duplicate_index.check(str(record["sample_id"]), text)
        if duplicate is not None:
            duplicate_errors.append(
                {"sample_id": record["sample_id"], **duplicate}
            )
        else:
            duplicate_index.add(str(record["sample_id"]), text)

    usage = state.usage_summary(stage)
    estimated_cost = pricing.estimate(usage["usage"])
    task_counts = task_distribution(sorted_records)
    length_counts = length_distribution(sorted_records)
    output_completion_rate = (
        len(sorted_records) / float(requested_samples)
        if requested_samples
        else 0.0
    )
    finalized_jobs = (
        int(usage["job_outcomes"].get("accepted", 0))
        + int(usage["job_outcomes"].get("discarded", 0))
    )
    valid_rate = (
        int(usage["job_outcomes"].get("accepted", 0)) / float(finalized_jobs)
        if finalized_jobs
        else 0.0
    )
    oracle_pass_count = sum(
        1
        for row in sorted_records
        if row["oracle_validation"]["status"] == "PASS"
        and row["oracle_answer"] == row["teacher_answer"]
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "generated_at": utc_now(),
        "generator_version": GENERATOR_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash(),
        "model": api_verification.get("model"),
        "programmatic_validation_status": "PENDING",
        "manual_review_status": (
            "PENDING" if stage == "audit" else "RECORDED_BY_AUDIT_GATE"
        ),
        "requested_samples": requested_samples,
        "accepted_samples": len(sorted_records),
        "output_completion_rate": output_completion_rate,
        "effective_valid_rate": valid_rate,
        "minimum_valid_rate": 0.90,
        "oracle_pass_count": oracle_pass_count,
        "oracle_pass_rate": (
            oracle_pass_count / float(len(sorted_records))
            if sorted_records
            else 0.0
        ),
        "task_distribution": task_counts,
        "task_percentages": _expected_percentages(task_counts),
        "length_distribution": length_counts,
        "length_percentages": _expected_percentages(length_counts),
        "context_token_total": sum(
            int(row["token_count"]) for row in sorted_records
        ),
        "sft_token_total": sum(
            int(row["sft_token_count"]) for row in sorted_records
        ),
        "raw_validation_errors": raw_errors,
        "near_duplicate_errors": duplicate_errors,
        "split_isolation": isolation,
        "api": {
            "verification": dict(api_verification),
            **usage,
            "pricing": pricing.as_dict(),
            "estimated_cost_usd": estimated_cost,
        },
        "gate": {
            "sample_count_exact": len(sorted_records) == requested_samples,
            "effective_valid_rate_at_least_90_percent": valid_rate >= 0.90,
            "oracle_validation_100_percent": (
                oracle_pass_count == len(sorted_records)
            ),
            "raw_schema_and_evidence_valid": not raw_errors,
            "no_high_similarity_duplicates": not duplicate_errors,
            "world_and_template_split_isolation": isolation["status"] == "PASS",
        },
    }
    report["status"] = (
        "PASS" if all(report["gate"].values()) else "FAIL"
    )
    report["programmatic_validation_status"] = report["status"]
    report_path = output_dir / "reports" / (
        "audit_report.json" if stage == "audit" else "pilot_audit_report.json"
    )
    write_json_atomic(report_path, report)

    review_rows = []
    by_task: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in sorted_records:
        by_task[str(row["task_type"])].append(row)
    for task_type in TASK_TYPES:
        available = by_task.get(task_type, [])
        # The initial 100-row audit contains only 20 aggregation samples by
        # design.  The final pilot queue always selects 30 per task.
        take = min(30, len(available))
        stride = max(1, len(available) // max(take, 1))
        selected = available[::stride][:take]
        for row in selected:
            review_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "task_type": row["task_type"],
                    "split": row["split"],
                    "token_count": row["token_count"],
                    "context": row["context"],
                    "instruction": row["instruction"],
                    "oracle_answer": row["oracle_answer"],
                    "evidence": row["evidence"],
                    "structured_facts": row["structured_facts"],
                    "required_checks": [
                        "answer_correct",
                        "evidence_supports_answer",
                        "no_benchmark_imitation",
                        "natural_and_unambiguous",
                    ],
                }
            )
    review_path = (
        output_dir / "reports" / f"{stage}_manual_review_queue.jsonl"
    )
    write_jsonl_atomic(review_path, review_rows)
    if stage == "audit":
        annotation_template = [
            {
                "sample_id": row["sample_id"],
                "answer_correct": None,
                "evidence_supports_answer": None,
                "no_benchmark_imitation": None,
                "natural_and_unambiguous": None,
                "notes": "",
            }
            for row in sorted_records
        ]
        write_jsonl_atomic(
            output_dir / "reports" / "audit_annotations_template.jsonl",
            annotation_template,
        )

    files_payload = {"raw": raw_files, "messages": message_files}
    dataset_identity = hashlib.sha256(
        canonical_json(files_payload).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "status": report["status"],
        "created_at": utc_now(),
        "generator_version": GENERATOR_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash(),
        "teacher_model": api_verification.get("model"),
        "tokenizer": dict(tokenizer_record),
        "dataset_identity_sha256": dataset_identity,
        "files": files_payload,
        "counts": {
            "total": len(sorted_records),
            "by_split": Counter(str(row["split"]) for row in sorted_records),
            "by_task": task_counts,
            "by_length_bin": length_counts,
        },
        "token_counts": {
            "context": report["context_token_total"],
            "rendered_sft": report["sft_token_total"],
        },
        "split_policy": {
            "unit": "world_id_and_template_family",
            "random_sample_split_forbidden": True,
            "validation": isolation,
        },
        "shared_consumers": {
            "Paper-Geo": {
                "dataset_identity_sha256": dataset_identity,
                "train_order": (
                    message_files.get("train", {}).get("sha256")
                    if stage == "pilot"
                    else None
                ),
            },
            "EVQ-Cosh": {
                "dataset_identity_sha256": dataset_identity,
                "train_order": (
                    message_files.get("train", {}).get("sha256")
                    if stage == "pilot"
                    else None
                ),
            },
        },
        "source_exclusions": [
            "RULER templates and samples",
            "NIAH/passkey templates",
            "Paul Graham essays",
            "SQuAD",
            "HotpotQA",
            "A-points-to-B variable tracking",
        ],
        "audit_report": {
            "path": str(report_path.resolve()),
            "sha256": sha256_file(report_path),
        },
        "manual_review_queue": {
            "path": str(review_path.resolve()),
            "rows": len(review_rows),
            "sha256": sha256_file(review_path),
            "contains_full_review_payload": True,
        },
        "api_accounting": report["api"],
    }
    manifest_name = (
        "audit_dataset_manifest.json"
        if stage == "audit"
        else "dataset_manifest.json"
    )
    manifest_path = output_dir / manifest_name
    write_json_atomic(manifest_path, manifest)
    (manifest_path.with_suffix(manifest_path.suffix + ".sha256")).write_text(
        f"{sha256_file(manifest_path)}  {manifest_path.name}\n",
        encoding="utf-8",
    )
    return {
        "report": report,
        "report_path": str(report_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manual_review_queue": str(review_path.resolve()),
    }


def validate_audit_gate(output_dir: Path) -> Dict[str, Any]:
    gate_path = output_dir / "audit_gate.json"
    if not gate_path.is_file():
        raise RuntimeError(
            "pilot generation is blocked until audit_gate.json exists"
        )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    manifest_path = output_dir / "audit_dataset_manifest.json"
    report_path = output_dir / "reports" / "audit_report.json"
    if gate.get("status") != "APPROVED":
        raise RuntimeError("audit gate is not APPROVED")
    if gate.get("audit_manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("audit manifest changed after approval")
    if gate.get("audit_report_sha256") != sha256_file(report_path):
        raise RuntimeError("audit report changed after approval")
    return gate


def approve_audit(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "audit_dataset_manifest.json"
    report_path = output_dir / "reports" / "audit_report.json"
    annotations_path = args.annotations.resolve()
    if not manifest_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("audit manifest/report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError("programmatic audit report is not PASS")
    if int(report.get("accepted_samples", 0)) != AUDIT_COUNT:
        raise RuntimeError(
            f"manual approval requires exactly {AUDIT_COUNT} audit samples"
        )
    annotations = load_jsonl(annotations_path)
    raw_rows = load_jsonl(output_dir / "raw" / "audit.jsonl")
    expected_ids = {str(row["sample_id"]) for row in raw_rows}
    seen: Dict[str, Dict[str, Any]] = {}
    required = {
        "sample_id",
        "answer_correct",
        "evidence_supports_answer",
        "no_benchmark_imitation",
        "natural_and_unambiguous",
    }
    for row in annotations:
        missing = required - set(row)
        if missing:
            raise ValueError(
                "annotation missing fields: " + ",".join(sorted(missing))
            )
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate annotation for {sample_id}")
        seen[sample_id] = row
    if set(seen) != expected_ids:
        raise ValueError(
            "annotations must cover all audit rows exactly; "
            f"missing={len(expected_ids - set(seen))}, "
            f"extra={len(set(seen) - expected_ids)}"
        )
    checks = [
        "answer_correct",
        "evidence_supports_answer",
        "no_benchmark_imitation",
        "natural_and_unambiguous",
    ]
    rejected = [
        sample_id
        for sample_id, row in seen.items()
        if any(row.get(check) is not True for check in checks)
    ]
    if rejected:
        raise RuntimeError(
            f"manual audit rejected {len(rejected)} samples; pilot remains blocked"
        )
    if not args.reviewer_id.strip():
        raise ValueError("--reviewer-id must be non-empty")
    gate = {
        "schema_version": SCHEMA_VERSION,
        "status": "APPROVED",
        "approved_at": utc_now(),
        "reviewer_id": args.reviewer_id.strip(),
        "annotation_count": len(annotations),
        "annotations_sha256": sha256_file(annotations_path),
        "audit_manifest_sha256": sha256_file(manifest_path),
        "audit_report_sha256": sha256_file(report_path),
    }
    write_json_atomic(output_dir / "audit_gate.json", gate)
    print(
        json.dumps(
            {
                "status": "APPROVED",
                "gate": str((output_dir / "audit_gate.json").resolve()),
                "annotation_count": len(annotations),
            },
            indent=2,
            sort_keys=True,
        )
    )


def verify_api(args: argparse.Namespace) -> Dict[str, Any]:
    client = DeepSeekClient(
        cache_dir=args.output_dir / "cache",
        timeout_seconds=args.timeout_seconds,
        transport_retries=args.transport_retries,
    )
    result = client.verify()
    path = args.output_dir / "api_verification.json"
    write_json_atomic(path, result)
    safe = {
        "status": result["status"],
        "model": result["model"],
        "model_list_contains_requested_model": True,
        "chat_json_output": True,
        "verification_path": str(path.resolve()),
    }
    print(json.dumps(safe, indent=2, sort_keys=True))
    return result


def dry_run(args: argparse.Namespace, tokenizer: Any, tokenizer_record: Mapping[str, Any]) -> None:
    jobs = build_jobs(stage=args.stage, num_samples=args.num_samples, seed=args.seed)
    skeletons = [build_skeleton(job) for job in jobs]
    task_counts = Counter(job.task_type for job in jobs)
    length_counts = Counter(job.length_bin for job in jobs)
    split_counts = Counter(job.split for job in jobs)
    template_sets: Dict[str, set] = defaultdict(set)
    world_sets: Dict[str, set] = defaultdict(set)
    for job in jobs:
        template_sets[job.split].add(job.template_family)
        world_sets[job.split].add(job.world_id)
    report = {
        "status": "PASS",
        "dry_run": True,
        "stage": args.stage,
        "num_samples": len(jobs),
        "task_distribution": dict(task_counts),
        "length_distribution": dict(length_counts),
        "split_distribution": dict(split_counts),
        "template_family_counts": {
            split: len(values) for split, values in template_sets.items()
        },
        "world_counts": {
            split: len(values) for split, values in world_sets.items()
        },
        "oracle_solver_pass_count": len(skeletons),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash(),
        "tokenizer": dict(tokenizer_record),
        "api_called": False,
    }
    path = args.output_dir / "reports" / f"{args.stage}_dry_run.json"
    write_json_atomic(path, report)
    print(json.dumps({**report, "path": str(path.resolve())}, indent=2, sort_keys=True))


def generate(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, tokenizer_record = load_tokenizer(
        args.tokenizer,
        revision=args.tokenizer_revision,
        local_files_only=args.local_files_only,
    )
    if args.dry_run:
        dry_run(args, tokenizer, tokenizer_record)
        return
    if args.stage == "pilot":
        validate_audit_gate(output_dir)
        expected = sum(PILOT_SPLIT_COUNTS.values())
        if args.num_samples != expected:
            raise ValueError(
                f"final pilot requires --num-samples {expected}; "
                "use --dry-run for smaller protocol checks"
            )
    api_verification = verify_api(args)
    client = DeepSeekClient(
        cache_dir=output_dir / "cache",
        timeout_seconds=args.timeout_seconds,
        transport_retries=args.transport_retries,
    )
    pricing = Pricing.from_environment()
    state = StateStore(
        output_dir / "work" / "state.sqlite3", resume=args.resume
    )
    try:
        base_jobs = build_jobs(
            stage=args.stage, num_samples=args.num_samples, seed=args.seed
        )
        accepted = state.accepted_by_slot(args.stage)
        # Resume must not preserve a row that a tightened semantic validator
        # can now prove has lost the intended task.  Invalidation keeps the
        # paid request accounting and creates a replacement world for only
        # the affected slot.
        for row in list(accepted.values()):
            stored_job = state.job(str(row["sample_id"]))
            scope_errors = validate_instruction_scope(
                build_skeleton(stored_job), str(row["instruction"])
            )
            if scope_errors:
                state.invalidate_accepted_record(
                    str(row["sample_id"]), scope_errors
                )
        accepted = state.accepted_by_slot(args.stage)
        dedup = DedupIndex()
        for row in accepted.values():
            dedup.add(
                str(row["sample_id"]),
                str(row["context"]) + "\n" + str(row["instruction"]),
            )
        dedup_lock = threading.Lock()
        base_by_slot = {_slot_id(job): job for job in base_jobs}
        missing_slots = [
            slot_id for slot_id in base_by_slot if slot_id not in accepted
        ]
        while missing_slots:
            jobs_to_run: List[GenerationJob] = []
            for slot_id in missing_slots:
                base = base_by_slot[slot_id]
                latest = state.latest_replacement_index(slot_id)
                replacement_index = max(0, latest + 1)
                if replacement_index > MAX_REPLACEMENTS_PER_SLOT:
                    raise RuntimeError(
                        f"replacement limit exhausted for slot {slot_id}"
                    )
                job = replacement_job(base, replacement_index)
                state.register_job(
                    job,
                    slot_id=slot_id,
                    replacement_index=replacement_index,
                )
                jobs_to_run.append(job)
            futures: Dict[Future, GenerationJob] = {}
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                for job in jobs_to_run:
                    prior_count, prior_errors, prior_token_count = (
                        state.prior_attempts(job.sample_id)
                    )
                    futures[
                        executor.submit(
                            process_job,
                            job=job,
                            tokenizer=tokenizer,
                            client=client,
                            dedup=dedup,
                            dedup_lock=dedup_lock,
                            prior_attempt_count=prior_count,
                            prior_errors=prior_errors,
                            prior_token_count=prior_token_count,
                        )
                    ] = job
                for future in as_completed(futures):
                    job = futures[future]
                    record, attempts, errors = future.result()
                    state.record_result(
                        job=job,
                        raw_record=record,
                        attempt_records=attempts,
                        final_errors=errors,
                    )
                    status = "accepted" if record is not None else "discarded"
                    print(
                        json.dumps(
                            {
                                "sample_id": job.sample_id,
                                "status": status,
                                "attempts_this_run": len(attempts),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            accepted = state.accepted_by_slot(args.stage)
            missing_slots = [
                slot_id for slot_id in base_by_slot if slot_id not in accepted
            ]

        result = materialize(
            output_dir=output_dir,
            stage=args.stage,
            records=list(accepted.values()),
            tokenizer=tokenizer,
            tokenizer_record=tokenizer_record,
            state=state,
            pricing=pricing,
            requested_samples=args.num_samples,
            api_verification=api_verification,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["report"]["status"] != "PASS":
            raise RuntimeError("dataset audit gate failed")
    finally:
        state.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verified DeepSeek SFT distillation dataset generator"
    )
    parser.add_argument(
        "--action",
        choices=("generate", "verify-api", "approve-audit"),
        default="generate",
    )
    parser.add_argument("--stage", choices=("audit", "pilot"), default="audit")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=AUDIT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--transport-retries", type=int, default=5)
    parser.add_argument("--tokenizer", default=TOKENIZER_REPO)
    parser.add_argument("--tokenizer-revision", default=TOKENIZER_REVISION)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--reviewer-id", default="")
    args = parser.parse_args(argv)
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.action == "approve-audit" and args.annotations is None:
        parser.error("--annotations is required for --action approve-audit")
    if args.action != "generate" and args.dry_run:
        parser.error("--dry-run is only valid with --action generate")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        if args.action == "verify-api":
            verify_api(args)
        elif args.action == "approve-audit":
            approve_audit(args)
        else:
            generate(args)
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
