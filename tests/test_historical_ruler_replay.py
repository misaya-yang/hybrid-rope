from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.analysis.audit_historical_ruler_replay import (
    ARMS, COMPLETE, FROZEN, HISTORY, LENGTHS, TASKS, audit, main, sha256,
)


def hashed(text: str) -> str:
    return sha256(text.encode())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value) + "\n")


def write_run(path: Path, arm: str, lengths: tuple[int, ...], limit: int) -> None:
    path.mkdir(parents=True)
    native, active = hashed("native"), hashed(arm)
    gain = 1.0 if arm == "native" else 1.0512928913614359
    kind = "native" if arm == "native" else "external_table_static"
    factor = 4.0 if arm == "native" and limit == 20 else 2.0
    protocol = {
        "status": FROZEN, "checkpoint_sha256": hashed("weights"),
        "data_manifest_sha256": hashed("data"), "native_context_length": 32768,
        "method": kind, "model_type": "qwen2", "table_factor": factor,
        "table_sha256_float32": None if arm == "native" else active,
        "table_file_sha256": None if arm == "native" else hashed(arm + ".npy"),
        "expected_active_sha256": None if arm == "native" else active,
        "table_support": None if arm == "native" else "native_div_factor",
        "long_attention_scaling": None if arm == "native" else gain,
        "lengths": list(lengths), "tasks": list(TASKS), "limit_per_cell": limit,
        "script_sha256": hashed("old-runner" if limit == 20 else "new-runner"),
    }
    if limit == 1:
        protocol.update(profile_target_length=65536, expected_native_sha256=native)
    method = {"method": kind, "model_type": "qwen2", "native_context_length": 32768,
              "active_sha256_float32": active}
    if arm == "native":
        method["attention_scaling"] = gain
    else:
        method.update(native_sha256_float32=native, long_attention_scaling=gain,
                      table_factor=2.0, table_support="native_div_factor", initial_branch="external_long")
    rows, cells, scores = [], {}, {}
    for task in TASKS:
        cells[task], scores[task] = {}, {}
        for length in lengths:
            cells[task][str(length)] = {
                "sha256": hashed(f"input-{task}-{length}"), "rows": 20, "selected_rows": limit,
                "path": "/private/source/never-serialize-this/test.jsonl",
            }
            for index in range(limit):
                rows.append({"task": task, "nominal_length": length, "local_index": index,
                             "prediction": "sensitive-answer plus unchanged explanation",
                             "references": ["sensitive-answer"], "official_metric": "string_match_all",
                             "official_task_score": 1.0})
            scores[task][str(length)] = {"rows": limit, "official_task_score": 1.0}
    examples = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (path / "examples.jsonl").write_bytes(examples)
    write_json(path / "run_manifest.json", protocol)
    write_json(path / "results.json", {
        "status": COMPLETE, "protocol": protocol, "method": method,
        "data": {"cells": cells, "manifest_sha256": hashed("data"),
                 "tokenizer_sha256": hashed("tokenizer"), "ruler_commit": "c" * 40, "seed": 20260822},
        "runtime": {"torch": "2.8.0+cu128", "cuda": "12.8"},
        "results": {"cells": scores, "examples": len(rows), "examples_sha256": sha256(examples),
                    "macro_official_task_score": 1.0},
    })


def fixture(root: Path) -> None:
    for arm in ARMS:
        write_run(root / "k32_historical_replay" / arm, arm, LENGTHS, 1)
        for length in LENGTHS:
            path = root / HISTORY[arm].format(length=length, size=length // 1024)
            write_run(path, arm, (length,), 20)


def mutate_result(path: Path, mutation) -> None:
    result = json.loads((path / "results.json").read_text())
    mutation(result)
    write_json(path / "results.json", result)
    write_json(path / "run_manifest.json", result["protocol"])


def mutate_rows(path: Path, mutation, rehash: bool = True) -> None:
    rows = [json.loads(line) for line in (path / "examples.jsonl").read_text().splitlines()]
    mutation(rows)
    raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (path / "examples.jsonl").write_bytes(raw)
    if rehash:
        mutate_result(path, lambda result: result["results"].update(examples_sha256=sha256(raw)))


class HistoricalReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        fixture(self.root)
        self.replay = self.root / "k32_historical_replay" / "native"

    def test_match_is_24_decoded_canaries_not_token_parity(self):
        result = audit(self.root)
        self.assertEqual(result["status"], "MATCH")
        self.assertEqual(result["matched_cells"], 24)
        self.assertEqual(len(result["raw_receipts"]), 9)
        self.assertFalse(result["token_parity_established"])
        capacities = result["raw_receipts"]
        self.assertEqual(capacities["historical_native_32768"]["profile_capacity"], 131072)
        self.assertEqual(capacities["replay_native"]["profile_capacity"], 65536)
        encoded = json.dumps(result, allow_nan=False)
        for private in ("sensitive-answer", "/private/source", str(self.root)):
            self.assertNotIn(private, encoded)

    def test_same_score_but_different_full_prediction_is_drift(self):
        mutate_rows(self.replay, lambda rows: rows[0].update(prediction="sensitive-answer DIFFERENT"))
        result = audit(self.root)
        self.assertEqual(result["status"], "DRIFT")
        self.assertEqual(result["matched_cells"], 23)
        cell = result["cells"][0]
        self.assertTrue(cell["checks"]["official_score_match"])
        self.assertFalse(cell["checks"]["decoded_prediction_match"])

    def test_unreproducible_score_is_not_repaired(self):
        mutate_rows(self.replay, lambda rows: rows[0].update(official_task_score=0.0))
        before = (self.replay / "examples.jsonl").read_bytes()
        self.assertEqual(audit(self.root)["status"], "DRIFT")
        self.assertEqual((self.replay / "examples.jsonl").read_bytes(), before)

    def test_examples_hash_and_completion_are_required(self):
        mutate_rows(self.replay, lambda rows: rows[0].update(prediction="changed"), rehash=False)
        self.assertEqual(audit(self.root)["status"], "DRIFT")
        mutate_result(self.replay, lambda result: result.update(status="RUNNING"))
        self.assertEqual(audit(self.root)["status"], "DRIFT")

    def test_duplicate_or_missing_fixed_index_is_rejected(self):
        mutate_rows(self.replay, lambda rows: rows[0].update(local_index=1))
        self.assertEqual(audit(self.root)["status"], "DRIFT")

    def test_per_cell_input_hash_is_compared(self):
        mutate_result(self.replay, lambda result: result["data"]["cells"][TASKS[0]][str(LENGTHS[0])].update(
            sha256=hashed("different-input")))
        result = audit(self.root)
        self.assertEqual(result["status"], "DRIFT")
        self.assertFalse(result["cells"][0]["checks"]["input_cell_hash_match"])

    def test_same_prediction_with_changed_references_is_drift(self):
        mutate_rows(self.replay, lambda rows: rows[0].update(references=["unchanged explanation"]))
        result = audit(self.root)
        self.assertEqual(result["status"], "DRIFT")
        self.assertFalse(result["cells"][0]["checks"]["references_match"])

    def test_checkpoint_native_active_gain_and_runtime_are_bound(self):
        mutations = {
            "checkpoint": lambda result: result["protocol"].update(checkpoint_sha256=hashed("wrong")),
            "native": lambda result: result["method"].update(native_sha256_float32=hashed("wrong")),
            "active": lambda result: result["method"].update(active_sha256_float32=hashed("wrong")),
            "gain": lambda result: result["method"].update(long_attention_scaling=1.2),
            "tokenizer": lambda result: result["data"].update(tokenizer_sha256=hashed("wrong")),
            "runtime": lambda result: result["runtime"].update(torch="different"),
        }
        path = self.root / "k32_historical_replay" / "physical_x"
        original = (path / "results.json").read_bytes()
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                (path / "results.json").write_bytes(original)
                mutate_result(path, mutation)
                self.assertEqual(audit(self.root)["status"], "DRIFT")

    def test_terminal_protocol_and_cell_score_are_checked(self):
        protocol_path = self.replay / "run_manifest.json"
        original = protocol_path.read_bytes()
        protocol = json.loads(original)
        protocol["script_sha256"] = hashed("different")
        write_json(protocol_path, protocol)
        self.assertEqual(audit(self.root)["status"], "DRIFT")
        protocol_path.write_bytes(original)
        mutate_result(self.replay, lambda result: result["results"]["cells"][TASKS[0]][str(LENGTHS[0])].update(
            official_task_score=0.0))
        self.assertEqual(audit(self.root)["status"], "DRIFT")

    def test_missing_run_reports_drift_without_private_path(self):
        (self.replay / "results.json").unlink()
        result = audit(self.root)
        self.assertEqual(result["status"], "DRIFT")
        self.assertNotIn(str(self.root), json.dumps(result))

    def test_cli_writes_portable_json_and_nonzero_on_drift(self):
        output = self.root / "portable.json"
        argv = ["--root", str(self.root), "--output", str(output)]
        self.assertEqual(main(argv), 0)
        self.assertEqual(json.loads(output.read_text())["status"], "MATCH")
        mutate_rows(self.replay, lambda rows: rows[0].update(prediction="sensitive-answer changed"))
        self.assertEqual(main(argv), 1)
        self.assertEqual(json.loads(output.read_text())["status"], "DRIFT")


if __name__ == "__main__":
    unittest.main()
