#!/usr/bin/env python3
"""Contract tests for the post-release temporal holdout preparation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.data_prep.prepare_temporal_holdout_2026 import (
    TemporalDocument,
    build_token_packs,
    deduplicate_documents,
    fetch_arxiv_documents,
    fetch_federal_register_documents,
    fetch_federal_register_abstract_documents,
    fetch_stackoverflow_documents,
    fetch_wikipedia_documents,
    parse_arxiv_atom,
    parse_federal_register_document,
    parse_federal_register_abstracts,
    parse_stackoverflow_questions,
    parse_wikipedia_pages,
    validate_document,
    write_domain_artifacts,
)


class _ToyTokenizer:
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) % 251 + 1 for char in text]


def _document(doc_id: str, text: str, published_at: str = "2026-06-01") -> TemporalDocument:
    return TemporalDocument(
        source="test",
        doc_id=doc_id,
        published_at=published_at,
        title=f"Title {doc_id}",
        text=text,
        url=f"https://example.test/{doc_id}",
        metadata={},
    )


class TemporalHoldoutPreparationTests(unittest.TestCase):
    def test_rejects_documents_not_published_in_2026(self):
        with self.assertRaisesRegex(ValueError, "outside 2026"):
            validate_document(_document("old", "x" * 200, "2025-12-31"), min_chars=100)
        with self.assertRaisesRegex(ValueError, "outside 2026"):
            validate_document(_document("future", "x" * 200, "2026-12-31"), min_chars=100)

    def test_deduplicates_normalized_document_text(self):
        first = _document("a", "A sentence.\n\nAnother sentence.")
        duplicate = _document("b", "  A sentence.\nAnother   sentence.  ")
        distinct = _document("c", "Completely different text.")

        kept, rejected = deduplicate_documents([first, duplicate, distinct])

        self.assertEqual([document.doc_id for document in kept], ["a", "c"])
        self.assertEqual(rejected, [{"doc_id": "b", "reason": "duplicate_text", "duplicate_of": "a"}])

    def test_builds_exact_packs_and_masks_cross_document_targets(self):
        documents = [
            _document("a", "a" * 80),
            _document("b", "b" * 80),
            _document("c", "c" * 80),
            _document("d", "d" * 80),
        ]

        packs = build_token_packs(
            documents,
            _ToyTokenizer(),
            pack_tokens=64,
            num_packs=2,
        )

        self.assertEqual(tuple(packs.input_ids.shape), (2, 64))
        self.assertEqual(tuple(packs.score_mask.shape), (2, 64))
        self.assertEqual(packs.input_ids.dtype.name, "int32")
        self.assertEqual(packs.score_mask.dtype.name, "bool")
        self.assertFalse(bool(packs.score_mask[0, 0]))
        self.assertFalse(bool(packs.score_mask[1, 0]))
        for pack_index, starts in enumerate(packs.doc_start_positions_by_pack):
            for position in starts:
                self.assertFalse(bool(packs.score_mask[pack_index, position]))
        self.assertTrue(all(packs.documents_by_pack))
        self.assertTrue(set(packs.documents_by_pack[0]).isdisjoint(packs.documents_by_pack[1]))
        self.assertTrue(all(packs.document_spans_by_pack))

    def test_fails_closed_when_documents_cannot_fill_requested_packs(self):
        with self.assertRaisesRegex(ValueError, "insufficient temporal holdout tokens"):
            build_token_packs(
                [_document("short", "short text")],
                _ToyTokenizer(),
                pack_tokens=128,
                num_packs=2,
            )

    def test_parses_arxiv_v1_publication_time_and_abstract(self):
        payload = b"""<?xml version='1.0' encoding='UTF-8'?>
        <feed xmlns='http://www.w3.org/2005/Atom'>
          <entry><id>http://arxiv.org/abs/2607.00001v1</id>
          <title>A 2026 Paper</title><published>2026-07-01T12:00:00Z</published>
          <updated>2026-07-02T12:00:00Z</updated>
          <summary>New scientific findings with enough fixture text.</summary>
          <category term='cs.CL'/><author><name>A. Researcher</name></author></entry>
        </feed>"""

        documents = parse_arxiv_atom(payload)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].doc_id, "2607.00001v1")
        self.assertEqual(documents[0].published_at, "2026-07-01T12:00:00Z")
        self.assertEqual(documents[0].metadata["categories"], ["cs.CL"])

    def test_parses_wikipedia_pages_using_creation_timestamp(self):
        created = {
            123: {
                "pageid": 123,
                "revid": 456,
                "title": "New 2026 Topic",
                "timestamp": "2026-06-01T00:00:00Z",
            }
        }
        pages = {
            "query": {
                "pages": [
                    {
                        "pageid": 123,
                        "title": "New 2026 Topic",
                        "extract": "A newly created encyclopedia article.",
                        "revisions": [{"revid": 789, "timestamp": "2026-06-02T00:00:00Z"}],
                    }
                ]
            }
        }

        documents = parse_wikipedia_pages(created, pages)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].published_at, "2026-06-01T00:00:00Z")
        self.assertEqual(documents[0].metadata["current_revid"], 789)

    def test_parses_federal_register_raw_text_with_publication_date(self):
        metadata = {
            "document_number": "2026-12345",
            "publication_date": "2026-05-20",
            "title": "A New Rule",
            "html_url": "https://www.federalregister.gov/documents/2026/05/20/2026-12345/example",
            "type": "Rule",
            "agencies": [{"name": "Example Agency"}],
        }

        document = parse_federal_register_document(metadata, "Official rule text.")

        self.assertEqual(document.doc_id, "2026-12345")
        self.assertEqual(document.metadata["agencies"], ["Example Agency"])
        self.assertEqual(document.text, "Official rule text.")

    def test_parses_and_paginates_federal_register_abstracts(self):
        row = {
            "document_number": "2026-22222",
            "publication_date": "2026-06-01",
            "title": "Official 2026 Abstract",
            "abstract": "A" * 300,
            "html_url": "https://www.federalregister.gov/documents/example",
            "type": "Rule",
            "agencies": [{"name": "Agency"}],
        }
        parsed = parse_federal_register_abstracts({"results": [row]})
        self.assertEqual(parsed[0].text, "A" * 300)

        calls = []

        def fetch(url: str) -> bytes:
            calls.append(url)
            payload = {
                "results": [{**row, "document_number": str(len(calls)), "abstract": str(len(calls)) + "A" * 300}],
                "next_page_url": "https://example.test/page2" if len(calls) == 1 else None,
            }
            return json.dumps(payload).encode()

        documents, source = fetch_federal_register_abstract_documents(
            max_documents=2, min_chars=100, fetch_bytes=fetch
        )
        self.assertEqual(len(documents), 2)
        self.assertEqual(source["publication_year"], 2026)

    def test_parses_2026_stackoverflow_creation_time_and_html_body(self):
        payload = {
            "items": [
                {
                    "question_id": 123,
                    "creation_date": 1782864000,
                    "title": "A new 2026 systems question",
                    "body": "<p>How should I debug this?</p><pre><code>print('x')</code></pre>",
                    "link": "https://stackoverflow.com/questions/123/example",
                    "tags": ["python", "pytorch"],
                    "score": 2,
                    "answer_count": 1,
                }
            ]
        }

        documents = parse_stackoverflow_questions(payload)

        self.assertEqual(documents[0].doc_id, "123")
        self.assertTrue(documents[0].published_at.startswith("2026-"))
        self.assertIn("How should I debug this?", documents[0].text)
        self.assertIn("print('x')", documents[0].text)

    def test_fetches_paginated_stackoverflow_questions(self):
        calls = []

        def fetch(url: str) -> bytes:
            calls.append(url)
            return json.dumps(
                {
                    "items": [
                        {
                            "question_id": len(calls),
                            "creation_date": 1782864000,
                            "title": "Question",
                            "body": "<p>" + str(len(calls)) + "x" * 300 + "</p>",
                            "link": f"https://stackoverflow.com/questions/{len(calls)}",
                            "tags": ["python"],
                        }
                    ],
                    "has_more": len(calls) == 1,
                    "backoff": 0,
                }
            ).encode()

        documents, source = fetch_stackoverflow_documents(
            max_questions=2, min_chars=100, fetch_bytes=fetch
        )

        self.assertEqual(len(documents), 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(source["publication_year"], 2026)

    def test_writes_hash_bound_domain_artifacts(self):
        documents = [_document(str(index), chr(97 + index) * 100) for index in range(4)]
        with tempfile.TemporaryDirectory() as temporary:
            manifest = write_domain_artifacts(
                domain="fixture",
                documents=documents,
                tokenizer=_ToyTokenizer(),
                tokenizer_record={"identifier": "toy", "files": {}},
                output_dir=Path(temporary),
                pack_tokens=64,
                num_packs=2,
                protocol_lengths=(16, 32, 64),
                source_record={"temporal_rule": "published_at starts with 2026"},
            )

            saved = json.loads((Path(temporary) / "manifest.json").read_text())
            self.assertEqual(saved, manifest)
            self.assertEqual(manifest["schema"], "evq_cosh.temporal_holdout_2026.v1")
            self.assertEqual(manifest["pack_contract"]["shape"], [2, 64])
            self.assertEqual(manifest["pack_contract"]["lengths"], [16, 32, 64])
            self.assertEqual(set(manifest["files"]), {"documents", "input_ids", "score_mask"})
            self.assertTrue((Path(temporary) / "input_ids.pt").is_file())
            self.assertTrue((Path(temporary) / "target_score_mask.pt").is_file())

    def test_fetches_arxiv_documents_from_a_bounded_2026_query(self):
        payload = b"""<feed xmlns='http://www.w3.org/2005/Atom'><entry>
          <id>http://arxiv.org/abs/2607.00002v1</id><title>Recent Research</title>
          <published>2026-07-02T00:00:00Z</published><updated>2026-07-02T00:00:00Z</updated>
          <summary>""" + b"x" * 200 + b"""</summary><category term='cs.LG'/>
          <author><name>Researcher</name></author></entry></feed>"""
        requested = []

        def fetch(url: str) -> bytes:
            requested.append(url)
            return payload

        documents, source = fetch_arxiv_documents(
            max_results=10, min_chars=100, fetch_bytes=fetch
        )

        self.assertEqual([document.doc_id for document in documents], ["2607.00002v1"])
        self.assertIn("submittedDate", requested[0])
        self.assertEqual(source["publication_year"], 2026)

    def test_fetches_federal_register_detail_and_raw_text(self):
        index = {
            "results": [
                {
                    "document_number": "2026-11111",
                    "publication_date": "2026-04-01",
                    "title": "Official Notice",
                    "html_url": "https://www.federalregister.gov/documents/example",
                    "json_url": "https://www.federalregister.gov/api/v1/documents/2026-11111",
                    "type": "Rule",
                    "agencies": [{"name": "Agency"}],
                }
            ]
        }
        detail = {**index["results"][0], "raw_text_url": "https://example.test/raw.txt"}

        def fetch(url: str) -> bytes:
            if "documents.json" in url:
                return json.dumps(index).encode()
            if url.endswith("2026-11111"):
                return json.dumps(detail).encode()
            if url.endswith("raw.txt"):
                return b"z" * 300
            raise AssertionError(url)

        documents, source = fetch_federal_register_documents(
            max_documents=5, min_chars=100, fetch_bytes=fetch
        )

        self.assertEqual([document.doc_id for document in documents], ["2026-11111"])
        self.assertEqual(source["publication_year"], 2026)

    def test_fetches_new_wikipedia_pages_and_excludes_disambiguation(self):
        recent = {
            "batchcomplete": True,
            "query": {
                "recentchanges": [
                    {
                        "pageid": 10,
                        "revid": 20,
                        "title": "New Topic",
                        "timestamp": "2026-07-01T00:00:00Z",
                        "newlen": 1000,
                    }
                ]
            },
        }
        pages = {
            "query": {
                "pages": [
                    {
                        "pageid": 10,
                        "title": "New Topic",
                        "extract": "w" * 300,
                        "revisions": [{"revid": 21, "timestamp": "2026-07-02T00:00:00Z"}],
                    }
                ]
            }
        }

        def fetch(url: str) -> bytes:
            return json.dumps(recent if "recentchanges" in url else pages).encode()

        documents, source = fetch_wikipedia_documents(
            max_candidates=10, min_chars=100, fetch_bytes=fetch
        )

        self.assertEqual([document.doc_id for document in documents], ["10"])
        self.assertEqual(source["selection"], "namespace-0 pages created in 2026")


if __name__ == "__main__":
    unittest.main()
