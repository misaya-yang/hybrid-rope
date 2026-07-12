#!/usr/bin/env python3
"""Prepare post-release 2026 text packs for 8K/16K/32K PPL evaluation."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch


TEMPORAL_YEAR = 2026
TEMPORAL_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
FREEZE_AT = datetime(2026, 7, 12, 23, 59, 59, tzinfo=timezone.utc)
USER_AGENT = "EVQ-Cosh-temporal-holdout/1.0 (academic reproducibility audit)"


@dataclass(frozen=True)
class TemporalDocument:
    source: str
    doc_id: str
    published_at: str
    title: str
    text: str
    url: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TokenPacks:
    input_ids: np.ndarray
    score_mask: np.ndarray
    documents_by_pack: list[list[str]]
    doc_start_positions_by_pack: list[list[int]]
    document_spans_by_pack: list[list[dict[str, Any]]]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def request_bytes(url: str, *, attempts: int = 5, timeout: int = 30) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as error:  # pragma: no cover - exercised by live smoke tests
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"failed to fetch official source after {attempts} attempts: {url}") from last_error


def validate_document(document: TemporalDocument, *, min_chars: int) -> None:
    try:
        timestamp = datetime.fromisoformat(document.published_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"document {document.doc_id} has an invalid publication time") from error
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    if not TEMPORAL_START <= timestamp <= FREEZE_AT:
        raise ValueError(f"document {document.doc_id} is outside 2026 frozen window")
    if len(normalize_text(document.text)) < min_chars:
        raise ValueError(f"document {document.doc_id} is shorter than min_chars")
    if not document.doc_id or not document.source or not document.url:
        raise ValueError("temporal document identity is incomplete")


def deduplicate_documents(
    documents: Iterable[TemporalDocument],
) -> tuple[list[TemporalDocument], list[dict[str, str]]]:
    kept: list[TemporalDocument] = []
    rejected: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    seen_ids: set[tuple[str, str]] = set()
    for document in documents:
        identity = (document.source, document.doc_id)
        if identity in seen_ids:
            rejected.append(
                {
                    "doc_id": document.doc_id,
                    "reason": "duplicate_document_id",
                    "duplicate_of": document.doc_id,
                }
            )
            continue
        seen_ids.add(identity)
        key = normalize_text(document.text)
        duplicate_of = seen.get(key)
        if duplicate_of is not None:
            rejected.append(
                {
                    "doc_id": document.doc_id,
                    "reason": "duplicate_text",
                    "duplicate_of": duplicate_of,
                }
            )
            continue
        seen[key] = document.doc_id
        kept.append(document)
    return kept, rejected


def parse_arxiv_atom(payload: bytes) -> list[TemporalDocument]:
    root = ET.fromstring(payload)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    documents: list[TemporalDocument] = []
    for entry in root.findall("atom:entry", namespace):
        identifier = (entry.findtext("atom:id", default="", namespaces=namespace)).rsplit("/", 1)[-1]
        title = normalize_text(entry.findtext("atom:title", default="", namespaces=namespace))
        summary = normalize_text(entry.findtext("atom:summary", default="", namespaces=namespace))
        published = entry.findtext("atom:published", default="", namespaces=namespace)
        updated = entry.findtext("atom:updated", default="", namespaces=namespace)
        categories = [
            str(node.attrib.get("term", ""))
            for node in entry.findall("atom:category", namespace)
            if node.attrib.get("term")
        ]
        authors = [
            normalize_text(node.findtext("atom:name", default="", namespaces=namespace))
            for node in entry.findall("atom:author", namespace)
        ]
        documents.append(
            TemporalDocument(
                source="arxiv_2026",
                doc_id=identifier,
                published_at=published,
                title=title,
                text=summary,
                url=f"https://arxiv.org/abs/{identifier}",
                metadata={"updated_at": updated, "categories": categories, "authors": authors},
            )
        )
    return documents


def parse_wikipedia_pages(
    created_by_pageid: dict[int, dict[str, Any]],
    payload: dict[str, Any],
) -> list[TemporalDocument]:
    documents: list[TemporalDocument] = []
    for page in payload.get("query", {}).get("pages", []):
        page_id = int(page.get("pageid", -1))
        created = created_by_pageid.get(page_id)
        if created is None or page.get("missing"):
            continue
        revisions = page.get("revisions") or []
        current = revisions[0] if revisions else {}
        documents.append(
            TemporalDocument(
                source="wikipedia_new_2026",
                doc_id=str(page_id),
                published_at=str(created.get("timestamp", "")),
                title=normalize_text(str(page.get("title", ""))),
                text=normalize_text(str(page.get("extract", ""))),
                url=f"https://en.wikipedia.org/?curid={page_id}",
                metadata={
                    "creation_revid": created.get("revid"),
                    "current_revid": current.get("revid"),
                    "current_revision_at": current.get("timestamp"),
                    "current_revision_sha1": current.get("sha1"),
                },
            )
        )
    return documents


def parse_federal_register_document(
    metadata: dict[str, Any], raw_text: str
) -> TemporalDocument:
    document_number = str(metadata.get("document_number", ""))
    return TemporalDocument(
        source="federal_register_2026",
        doc_id=document_number,
        published_at=str(metadata.get("publication_date", "")),
        title=normalize_text(str(metadata.get("title", ""))),
        text=normalize_text(raw_text),
        url=str(metadata.get("html_url", "")),
        metadata={
            "type": metadata.get("type"),
            "agencies": [
                agency.get("name") or agency.get("raw_name")
                for agency in metadata.get("agencies", [])
                if agency.get("name") or agency.get("raw_name")
            ],
            "raw_text_url": metadata.get("raw_text_url"),
        },
    )


def parse_federal_register_abstracts(payload: dict[str, Any]) -> list[TemporalDocument]:
    documents = []
    for record in payload.get("results", []):
        document_number = str(record.get("document_number", ""))
        documents.append(
            TemporalDocument(
                source="federal_register_2026",
                doc_id=document_number,
                published_at=str(record.get("publication_date", "")),
                title=normalize_text(str(record.get("title", ""))),
                text=normalize_text(str(record.get("abstract", ""))),
                url=str(record.get("html_url", "")),
                metadata={
                    "type": record.get("type"),
                    "agencies": [
                        agency.get("name") or agency.get("raw_name")
                        for agency in record.get("agencies", [])
                        if agency.get("name") or agency.get("raw_name")
                    ],
                    "pdf_url": record.get("pdf_url"),
                },
            )
        )
    return documents


def fetch_federal_register_abstract_documents(
    *,
    max_documents: int,
    min_chars: int,
    fetch_bytes: Callable[[str], bytes] = request_bytes,
) -> tuple[list[TemporalDocument], dict[str, Any]]:
    endpoint = "https://www.federalregister.gov/api/v1/documents.json"
    params = urllib.parse.urlencode(
        {
            "per_page": min(max_documents, 1000),
            "order": "newest",
            "conditions[publication_date][gte]": "2026-01-01",
            "conditions[publication_date][lte]": "2026-07-12",
        }
    )
    next_page_url: str | None = f"{endpoint}?{params}"
    documents: list[TemporalDocument] = []
    while next_page_url and len(documents) < max_documents:
        payload = json.loads(fetch_bytes(next_page_url))
        documents.extend(parse_federal_register_abstracts(payload))
        next_page_url = payload.get("next_page_url")
    return _validated_documents(documents[:max_documents], min_chars=min_chars), {
        "api": endpoint,
        "publication_window": ["2026-01-01", "2026-07-12"],
        "publication_year": TEMPORAL_YEAR,
        "content": "official Federal Register title and abstract metadata",
        "license": "United States federal government work; verify per-document notices",
    }


def parse_stackoverflow_questions(payload: dict[str, Any]) -> list[TemporalDocument]:
    documents: list[TemporalDocument] = []
    for item in payload.get("items", []):
        question_id = str(item.get("question_id", ""))
        try:
            published = datetime.fromtimestamp(
                int(item.get("creation_date")), tz=timezone.utc
            ).isoformat().replace("+00:00", "Z")
        except (TypeError, ValueError, OSError):
            continue
        # Stack Exchange returns HTML with the `withbody` filter. Preserve code
        # contents while removing markup and normalizing HTML entities.
        body = html.unescape(re.sub(r"<[^>]+>", " ", str(item.get("body", ""))))
        documents.append(
            TemporalDocument(
                source="stackoverflow_2026",
                doc_id=question_id,
                published_at=published,
                title=html.unescape(normalize_text(str(item.get("title", "")))),
                text=normalize_text(body),
                url=str(item.get("link", "")),
                metadata={
                    "tags": item.get("tags", []),
                    "score": item.get("score"),
                    "answer_count": item.get("answer_count"),
                    "accepted_answer_id": item.get("accepted_answer_id"),
                    "last_activity_date": item.get("last_activity_date"),
                },
            )
        )
    return documents


def fetch_stackoverflow_documents(
    *,
    max_questions: int,
    min_chars: int,
    fetch_bytes: Callable[[str], bytes] = request_bytes,
) -> tuple[list[TemporalDocument], dict[str, Any]]:
    if max_questions > 2500:
        raise ValueError("anonymous Stack Exchange API access is capped at 25 pages / 2500 questions")
    endpoint = "https://api.stackexchange.com/2.3/questions"
    documents: list[TemporalDocument] = []
    page = 1
    while len(documents) < max_questions:
        params = urllib.parse.urlencode(
            {
                "site": "stackoverflow",
                "fromdate": 1767225600,
                "todate": 1783900799,
                "order": "desc",
                "sort": "creation",
                "filter": "withbody",
                "pagesize": min(100, max_questions - len(documents)),
                "page": page,
            }
        )
        payload = json.loads(fetch_bytes(f"{endpoint}?{params}"))
        documents.extend(parse_stackoverflow_questions(payload))
        backoff = int(payload.get("backoff", 0) or 0)
        if backoff:
            time.sleep(backoff)
        if not payload.get("has_more"):
            break
        page += 1
    return _validated_documents(documents[:max_questions], min_chars=min_chars), {
        "api": endpoint,
        "site": "stackoverflow",
        "selection": "questions created in 2026, newest first, official withbody filter",
        "publication_window": ["2026-01-01T00:00:00Z", "2026-07-12T23:59:59Z"],
        "publication_year": TEMPORAL_YEAR,
        "content": "question title and body; no answers",
        "license": "CC BY-SA per Stack Overflow content terms",
    }


def _validated_documents(
    documents: Iterable[TemporalDocument], *, min_chars: int
) -> list[TemporalDocument]:
    accepted: list[TemporalDocument] = []
    for document in documents:
        try:
            validate_document(document, min_chars=min_chars)
        except ValueError:
            continue
        accepted.append(document)
    accepted, _ = deduplicate_documents(accepted)
    return sorted(accepted, key=lambda document: (document.published_at, document.doc_id), reverse=True)


def fetch_arxiv_documents(
    *,
    max_results: int,
    min_chars: int,
    fetch_bytes: Callable[[str], bytes] = request_bytes,
) -> tuple[list[TemporalDocument], dict[str, Any]]:
    search_query = (
        "submittedDate:[202601010000 TO 202607122359] AND "
        "(cat:cs.CL OR cat:cs.LG OR cat:cs.AI)"
    )
    params = urllib.parse.urlencode(
        {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    url = f"https://export.arxiv.org/api/query?{params}"
    documents = _validated_documents(parse_arxiv_atom(fetch_bytes(url)), min_chars=min_chars)
    return documents, {
        "api": "https://export.arxiv.org/api/query",
        "query": search_query,
        "publication_year": TEMPORAL_YEAR,
        "content": "title and abstract from versioned arXiv Atom records",
        "license_note": "arXiv item licenses vary; raw text is retained only on the private evaluation host",
    }


def fetch_wikipedia_documents(
    *,
    max_candidates: int,
    max_pages: int | None = None,
    min_chars: int,
    fetch_bytes: Callable[[str], bytes] = request_bytes,
) -> tuple[list[TemporalDocument], dict[str, Any]]:
    endpoint = "https://en.wikipedia.org/w/api.php"
    created_by_pageid: dict[int, dict[str, Any]] = {}
    continuation: dict[str, Any] = {}
    while len(created_by_pageid) < max_candidates:
        params: dict[str, Any] = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "list": "recentchanges",
            "rcnamespace": 0,
            "rctype": "new",
            "rcstart": "2026-07-12T00:00:00Z",
            "rcend": "2026-06-01T00:00:00Z",
            "rclimit": min(500, max_candidates - len(created_by_pageid)),
            "rcprop": "title|ids|timestamp|sizes|flags",
        }
        params.update(continuation)
        payload = json.loads(fetch_bytes(f"{endpoint}?{urllib.parse.urlencode(params)}"))
        for record in payload.get("query", {}).get("recentchanges", []):
            page_id = int(record.get("pageid", -1))
            if page_id >= 0 and str(record.get("timestamp", "")).startswith("2026-"):
                created_by_pageid[page_id] = record
                if len(created_by_pageid) >= max_candidates:
                    break
        continuation = payload.get("continue") or {}
        if not continuation:
            break

    documents: list[TemporalDocument] = []
    # TextExtracts silently limits full-page extracts to one page. Fetch the
    # longest candidates individually so every accepted page has an auditable
    # full extract and revision identity.
    records = sorted(
        created_by_pageid.values(),
        key=lambda record: (int(record.get("newlen", 0)), int(record.get("pageid", -1))),
        reverse=True,
    )
    for record in records[: max_pages or max_candidates]:
        page_id = int(record["pageid"])
        params = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "pageids": str(page_id),
            "prop": "extracts|revisions|pageprops",
            "explaintext": 1,
            "exsectionformat": "plain",
            "rvprop": "ids|timestamp|sha1",
            "rvlimit": 1,
            "ppprop": "disambiguation",
        }
        payload = json.loads(fetch_bytes(f"{endpoint}?{urllib.parse.urlencode(params)}"))
        query = payload.get("query") or {}
        pages = query.get("pages", [])
        query["pages"] = [
            page
            for page in pages
            if "disambiguation" not in (page.get("pageprops") or {})
            and not str(page.get("title", "")).lower().startswith("list of ")
        ]
        payload["query"] = query
        documents.extend(parse_wikipedia_pages(created_by_pageid, payload))
    return _validated_documents(documents, min_chars=min_chars), {
        "api": endpoint,
        "selection": "namespace-0 pages created in 2026",
        "creation_window": ["2026-06-01T00:00:00Z", "2026-07-12T00:00:00Z"],
        "publication_year": TEMPORAL_YEAR,
        "license": "CC BY-SA 4.0 / GFDL per English Wikipedia terms",
    }


def fetch_federal_register_documents(
    *,
    max_documents: int,
    min_chars: int,
    fetch_bytes: Callable[[str], bytes] = request_bytes,
) -> tuple[list[TemporalDocument], dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            # Fetch a broad metadata page, then spend detail/raw-text requests
            # only on substantive long-form document classes.
            "per_page": 1000,
            "order": "newest",
            "conditions[publication_date][gte]": "2026-01-01",
            "conditions[publication_date][lte]": "2026-07-12",
        }
    )
    index_url = f"https://www.federalregister.gov/api/v1/documents.json?{params}"
    index = json.loads(fetch_bytes(index_url))
    records = [
        record
        for record in index.get("results", [])
        if record.get("type") in {"Rule", "Proposed Rule", "Presidential Document"}
    ]
    documents: list[TemporalDocument] = []
    for record in records[:max_documents]:
        document_number = str(record.get("document_number") or "")
        if not document_number:
            continue
        detail_url = str(
            record.get("json_url")
            or f"https://www.federalregister.gov/api/v1/documents/{document_number}.json"
        )
        detail = json.loads(fetch_bytes(detail_url))
        if detail.get("correction_of"):
            continue
        raw_text_url = str(detail.get("raw_text_url") or "")
        if not raw_text_url:
            continue
        raw_text = fetch_bytes(raw_text_url).decode("utf-8", errors="replace")
        documents.append(parse_federal_register_document(detail, raw_text))
    return _validated_documents(documents, min_chars=min_chars), {
        "api": "https://www.federalregister.gov/api/v1/documents.json",
        "publication_window": ["2026-01-01", "2026-07-12"],
        "publication_year": TEMPORAL_YEAR,
        "content": "official Federal Register raw text",
        "license": "United States federal government work; verify per-document notices",
    }


def _document_tokens(document: TemporalDocument, tokenizer: Any) -> list[int]:
    text = f"{normalize_text(document.title)}\n\n{normalize_text(document.text)}"
    return list(tokenizer.encode(text, add_special_tokens=False))


def build_token_packs(
    documents: Sequence[TemporalDocument],
    tokenizer: Any,
    *,
    pack_tokens: int,
    num_packs: int,
) -> TokenPacks:
    if pack_tokens <= 1 or num_packs <= 0:
        raise ValueError("pack_tokens and num_packs must be positive")
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")

    rows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    documents_by_pack: list[list[str]] = []
    starts_by_pack: list[list[int]] = []
    spans_by_pack: list[list[dict[str, Any]]] = []
    document_index = 0

    for _ in range(num_packs):
        tokens: list[int] = []
        score_mask: list[bool] = []
        pack_documents: list[str] = []
        pack_starts: list[int] = []
        pack_spans: list[dict[str, Any]] = []
        while len(tokens) < pack_tokens and document_index < len(documents):
            document = documents[document_index]
            document_index += 1
            encoded = _document_tokens(document, tokenizer)
            if not encoded:
                continue
            start = len(tokens)
            used_text_tokens = min(len(encoded), pack_tokens - start)
            pack_starts.append(start)
            pack_documents.append(document.doc_id)
            pack_spans.append(
                {
                    "doc_id": document.doc_id,
                    "start": start,
                    "end": min(start + len(encoded) + 1, pack_tokens),
                    "source_text_tokens": len(encoded),
                    "used_text_tokens": used_text_tokens,
                    "truncated": start + len(encoded) + 1 > pack_tokens,
                }
            )
            tokens.extend(encoded)
            score_mask.extend([False] + [True] * (len(encoded) - 1))
            tokens.append(int(eos_token_id))
            # Do not let domains with many short documents win by predicting
            # artificial packing-boundary EOS tokens.
            score_mask.append(False)
        if len(tokens) < pack_tokens:
            raise ValueError(
                f"insufficient temporal holdout tokens: built {len(rows)} full packs"
            )
        rows.append(np.asarray(tokens[:pack_tokens], dtype=np.int32))
        masks.append(np.asarray(score_mask[:pack_tokens], dtype=np.bool_))
        documents_by_pack.append(pack_documents)
        starts_by_pack.append([position for position in pack_starts if position < pack_tokens])
        spans_by_pack.append(pack_spans)

    return TokenPacks(
        input_ids=np.stack(rows),
        score_mask=np.stack(masks),
        documents_by_pack=documents_by_pack,
        doc_start_positions_by_pack=starts_by_pack,
        document_spans_by_pack=spans_by_pack,
    )


def _atomic_torch_save(value: torch.Tensor, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_domain_artifacts(
    *,
    domain: str,
    documents: Sequence[TemporalDocument],
    tokenizer: Any,
    tokenizer_record: dict[str, Any],
    output_dir: Path,
    pack_tokens: int,
    num_packs: int,
    source_record: dict[str, Any],
    protocol_lengths: Sequence[int] = (8192, 16384, 32768),
) -> dict[str, Any]:
    if not protocol_lengths or sorted(set(protocol_lengths)) != list(protocol_lengths):
        raise ValueError("protocol_lengths must be unique and increasing")
    if protocol_lengths[-1] != pack_tokens:
        raise ValueError("largest protocol length must equal pack_tokens")
    for document in documents:
        validate_document(document, min_chars=1)
    unique_documents, rejected = deduplicate_documents(documents)
    packs = build_token_packs(
        unique_documents,
        tokenizer,
        pack_tokens=pack_tokens,
        num_packs=num_packs,
    )
    flattened_ids = [
        doc_id for pack_documents in packs.documents_by_pack for doc_id in pack_documents
    ]
    if len(flattened_ids) != len(set(flattened_ids)):
        raise RuntimeError("packing reused a document across canonical 32K packs")
    used_ids = {
        doc_id for pack_documents in packs.documents_by_pack for doc_id in pack_documents
    }
    used_documents = [document for document in unique_documents if document.doc_id in used_ids]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    documents_path = output_dir / "documents.jsonl"
    input_ids_path = output_dir / "input_ids.pt"
    score_mask_path = output_dir / "target_score_mask.pt"
    manifest_path = output_dir / "manifest.json"

    documents_temporary = documents_path.with_suffix(".jsonl.incomplete")
    with documents_temporary.open("w", encoding="utf-8") as handle:
        for document in used_documents:
            record = {
                "source": document.source,
                "doc_id": document.doc_id,
                "published_at": document.published_at,
                "title": document.title,
                "text": document.text,
                "url": document.url,
                "metadata": document.metadata,
                "normalized_text_sha256": hashlib.sha256(
                    normalize_text(document.text).encode("utf-8")
                ).hexdigest(),
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(documents_temporary, documents_path)
    _atomic_torch_save(torch.from_numpy(packs.input_ids), input_ids_path)
    _atomic_torch_save(torch.from_numpy(packs.score_mask), score_mask_path)

    files = {}
    for key, path in (
        ("documents", documents_path),
        ("input_ids", input_ids_path),
        ("score_mask", score_mask_path),
    ):
        files[key] = {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema": "evq_cosh.temporal_holdout_2026.v1",
        "created_at_unix": int(time.time()),
        "domain": domain,
        "temporal_rule": "every source document has a verifiable 2026 publication or creation timestamp",
        "source": source_record,
        "tokenizer": tokenizer_record,
        "selection": {
            "accepted_before_packing": len(unique_documents),
            "used_documents": len(used_documents),
            "exact_duplicate_rejections": rejected,
            "model_outputs_observed_before_freeze": False,
        },
        "pack_contract": {
            "shape": [num_packs, pack_tokens],
            "input_dtype": "torch.int32",
            "mask_dtype": "torch.bool",
            "lengths": list(protocol_lengths),
            "length_rule": "each shorter length is an exact prefix of the canonical 32K pack",
            "position_rule": "global monotonically increasing positions; no reset at document boundaries",
            "target_rule": "score text tokens only; mask pack position zero, every document first token, and packing-boundary EOS tokens",
            "documents_by_pack": packs.documents_by_pack,
            "document_start_positions_by_pack": packs.doc_start_positions_by_pack,
            "document_spans_by_pack": packs.document_spans_by_pack,
            "pack_document_sets_disjoint": True,
        },
        "files": files,
    }
    _atomic_json(manifest, manifest_path)
    return manifest


def tokenizer_fingerprint(tokenizer: Any, model_name: str) -> dict[str, Any]:
    source = Path(model_name).expanduser()
    files: dict[str, str] = {}
    if source.is_dir():
        for name in (
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ):
            path = source / name
            if path.is_file():
                files[name] = sha256_file(path)
    return {
        "identifier": source.name if source.is_absolute() else str(model_name),
        "class": type(tokenizer).__name__,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "vocab_size": len(tokenizer),
        "files": files,
    }


def write_collection_manifest(
    output_dir: Path,
    *,
    tokenizer_record: dict[str, Any],
    domain_manifests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    domains: dict[str, Any] = {}
    for domain, manifest in sorted(domain_manifests.items()):
        manifest_path = output_dir / domain / "manifest.json"
        domains[domain] = {
            "manifest": str(Path(domain) / "manifest.json"),
            "manifest_sha256": sha256_file(manifest_path),
            "shape": manifest["pack_contract"]["shape"],
            "used_documents": manifest["selection"]["used_documents"],
        }
    collection = {
        "schema": "evq_cosh.temporal_holdout_2026.collection.v1",
        "frozen_through": FREEZE_AT.isoformat().replace("+00:00", "Z"),
        "claim_boundary": (
            "timestamp-selected 2026 temporal holdout; it is not asserted to be "
            "a proof of zero phrase-level pretraining overlap"
        ),
        "tokenizer": tokenizer_record,
        "preparation_code_sha256": sha256_file(Path(__file__).resolve()),
        "domains": domains,
    }
    _atomic_json(collection, output_dir / "collection_manifest.json")
    return collection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_packs", type=int, default=8)
    parser.add_argument("--pack_tokens", type=int, default=32768)
    parser.add_argument("--arxiv_max_results", type=int, default=1800)
    parser.add_argument("--wikipedia_max_candidates", type=int, default=3000)
    parser.add_argument("--wikipedia_max_pages", type=int, default=700)
    parser.add_argument("--stackoverflow_max_questions", type=int, default=2500)
    parser.add_argument("--federal_register_max_documents", type=int, default=4000)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to mutate an existing frozen dataset: {args.output_dir}"
        )
    if args.pack_tokens != 32768:
        raise ValueError("the registered temporal protocol requires pack_tokens=32768")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=Path(args.model_name).is_dir(),
    )
    tokenizer_record = tokenizer_fingerprint(tokenizer, args.model_name)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    fetch_plan = (
        (
            "federal_register_2026",
            lambda: fetch_federal_register_abstract_documents(
                max_documents=args.federal_register_max_documents,
                min_chars=100,
            ),
        ),
        (
            "arxiv_2026",
            lambda: fetch_arxiv_documents(
                max_results=args.arxiv_max_results,
                min_chars=500,
            ),
        ),
        (
            "stackoverflow_2026",
            lambda: fetch_stackoverflow_documents(
                max_questions=args.stackoverflow_max_questions,
                min_chars=200,
            ),
        ),
    )
    domain_manifests: dict[str, dict[str, Any]] = {}
    for index, (domain, fetcher) in enumerate(fetch_plan, start=1):
        print(f"[{index}/3] fetching official {domain} records", flush=True)
        documents, source_record = fetcher()
        print(
            f"[{index}/3] {domain}: {len(documents)} accepted documents; tokenizing",
            flush=True,
        )
        domain_manifests[domain] = write_domain_artifacts(
            domain=domain,
            documents=documents,
            tokenizer=tokenizer,
            tokenizer_record=tokenizer_record,
            output_dir=args.output_dir / domain,
            pack_tokens=args.pack_tokens,
            num_packs=args.num_packs,
            source_record=source_record,
        )
        print(f"[{index}/3] {domain}: frozen and hash-bound", flush=True)

    collection = write_collection_manifest(
        args.output_dir,
        tokenizer_record=tokenizer_record,
        domain_manifests=domain_manifests,
    )
    print(json.dumps(collection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
