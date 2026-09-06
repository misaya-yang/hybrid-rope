#!/usr/bin/env python3
"""CPU contract tests for the 2026 temporal-holdout evaluator."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.lora_evq_v2.eval_temporal_holdout_matched import (
    BUCKETS,
    chunked_masked_causal_sums,
    load_domain_artifacts,
    summarize_position_sums,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TemporalHoldoutEvalTests(unittest.TestCase):
    def test_masked_chunked_sums_match_full_cross_entropy(self):
        torch.manual_seed(7)
        hidden = torch.randn(1, 9, 5)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        target_mask = torch.tensor(
            [[False, True, True, False, True, True, True, False, True]]
        )
        lm_head = torch.nn.Linear(5, 11, bias=False)

        observed = chunked_masked_causal_sums(
            hidden,
            input_ids,
            target_mask,
            lm_head,
            buckets=((0, 4), (4, 9)),
            chunk_tokens=3,
        )

        logits = lm_head(hidden[:, :-1]).float()
        labels = input_ids[:, 1:]
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
        ).reshape_as(labels)
        valid = target_mask[:, 1:]
        self.assertAlmostEqual(
            observed["total"]["nll_sum"],
            float(losses[valid].detach().double().sum()),
            places=6,
        )
        self.assertEqual(observed["total"]["scored_tokens"], int(valid.sum()))
        first_valid = valid[:, :3]
        self.assertAlmostEqual(
            observed["0-4"]["nll_sum"],
            float(losses[:, :3][first_valid].detach().double().sum()),
            places=6,
        )

    def test_summarizes_nested_prefixes_from_one_32k_forward(self):
        sums = {
            "0-4K": {"nll_sum": 4.0, "scored_tokens": 2},
            "4-8K": {"nll_sum": 6.0, "scored_tokens": 2},
            "8-12K": {"nll_sum": 8.0, "scored_tokens": 2},
            "12-16K": {"nll_sum": 10.0, "scored_tokens": 2},
            "16-24K": {"nll_sum": 20.0, "scored_tokens": 4},
            "24-32K": {"nll_sum": 24.0, "scored_tokens": 4},
            "total": {"nll_sum": 72.0, "scored_tokens": 16},
        }

        summary = summarize_position_sums(sums)

        self.assertAlmostEqual(summary["prefixes"]["8K"]["nll"], 2.5)
        self.assertAlmostEqual(summary["prefixes"]["16K"]["nll"], 3.5)
        self.assertAlmostEqual(summary["prefixes"]["32K"]["nll"], 4.5)
        self.assertAlmostEqual(summary["prefixes"]["8K"]["ppl"], math.exp(2.5))

    def test_loads_only_hash_matched_canonical_32k_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids = torch.arange(2 * 32768, dtype=torch.int32).reshape(2, 32768)
            mask = torch.ones((2, 32768), dtype=torch.bool)
            mask[:, 0] = False
            torch.save(ids, root / "input_ids.pt")
            torch.save(mask, root / "target_score_mask.pt")
            (root / "documents.jsonl").write_text(
                '{"doc_id":"a","published_at":"2026-01-01"}\n'
                '{"doc_id":"b","published_at":"2026-01-02"}\n',
                encoding="utf-8",
            )
            manifest = {
                "schema": "evq_cosh.temporal_holdout_2026.v1",
                "domain": "fixture",
                "pack_contract": {
                    "shape": [2, 32768],
                    "lengths": [8192, 16384, 32768],
                    "documents_by_pack": [["a"], ["b"]],
                    "document_start_positions_by_pack": [[0], [0]],
                    "pack_document_sets_disjoint": True,
                },
                "files": {},
            }
            for key, name in (
                ("documents", "documents.jsonl"),
                ("input_ids", "input_ids.pt"),
                ("score_mask", "target_score_mask.pt"),
            ):
                path = root / name
                manifest["files"][key] = {
                    "name": name,
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            loaded_manifest, loaded_ids, loaded_mask = load_domain_artifacts(root)

            self.assertEqual(loaded_manifest["domain"], "fixture")
            self.assertTrue(torch.equal(loaded_ids, ids))
            self.assertTrue(torch.equal(loaded_mask, mask))

            with (root / "input_ids.pt").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "(size|hash) mismatch"):
                load_domain_artifacts(root)


if __name__ == "__main__":
    unittest.main()
