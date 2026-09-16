"""Regression checks for the four-model official-static-YaRN QA supplement."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.iclr2027_strong_evidence_20260915.official_yarn_naturalqa import (
    Condition,
    METHOD_IDENTITY,
    _generation_rows,
    _recovery_runtime_signature,
    _runtime_matches,
    build_report,
    condition_from_name,
    validate_yarn_table,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values))


class OfficialYarnNaturalQATest(unittest.TestCase):
    def condition(self, root: Path, *, family: str, rows: int, clusters: int) -> Condition:
        model = root / "model"
        write_json(model / "config.json", {"model_type": "fixture"})
        panel = root / "assets/inputs.jsonl"
        manifest = root / "assets/manifest.json"
        write_json(manifest, {"status": "COMPLETE", "inputs_sha256": "fixture"})
        return Condition(
            name="fixture",
            public_model_name="Fixture",
            model_id="fixture",
            table_model_id="fixture",
            model=model,
            scale=4.0,
            family=family,
            panel=panel,
            asset_manifest=manifest,
            baseline_runs=root / "baseline/runs",
            baseline_tables=root / "baseline/tables",
            yarn_root=root / "yarn",
            yarn_table=root / "yarn/tables/yarn.json",
            data_manifest=manifest,
            length_cap=131072,
            expected_rows=rows,
            expected_clusters=clusters,
            rows_per_task=50,
        )

    def receipt(self, condition: Condition, *, updates: dict | None = None) -> dict:
        construction = {
            "mode": "official_yarn_native",
            "identity": "official static YaRN frequency map on a frozen checkpoint; no YaRN SFT",
            "model_weight_updates": 0,
            "same_table_all_layers_and_lengths": True,
        }
        if updates:
            construction.update(updates)
        return {
            "model_id": condition.table_model_id,
            "scale": condition.scale,
            "role": "baseline",
            "table": {
                "gain": 1.1,
                "values_float32": [1.0, 0.5],
                "construction": construction,
            },
        }

    def raw(self, panel: list[dict], output: str) -> list[dict]:
        return [{
            "row_id": row["row_id"],
            "task": row["task"],
            "prompt_sha256": row["prompt_sha256"],
            "input_tokens": row["input_tokens"],
            "references": row["references"],
            "generated_ids": [1],
            "output_text": output,
            "whole_response_f1": float(output == "answer"),
            "ended_eos": True,
            "hit_cap": False,
            "empty": False,
        } for row in panel]

    def materialize_report_inputs(
        self, condition: Condition, panel: list[dict], raw: dict[str, list[dict]],
    ) -> None:
        write_jsonl(condition.panel, panel)
        write_json(condition.yarn_table, self.receipt(condition))
        for arm, values in raw.items():
            run = (
                condition.yarn_root / "runs/yarn"
                if arm == "yarn" else condition.baseline_runs / arm
            )
            write_jsonl(run / "generations.jsonl", values)

    def test_allowlist_excludes_qwen_1p5b(self):
        with self.assertRaisesRegex(ValueError, "Qwen2.5-1.5B is excluded"):
            condition_from_name("qwen25_1p5b", Path("/tmp/plan"))

    def test_yarn_receipt_requires_zero_training_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            condition = self.condition(Path(directory), family="infinitebench_en_qa", rows=35, clusters=7)
            write_json(condition.yarn_table, self.receipt(condition))
            validate_yarn_table(condition.yarn_table, condition)
            write_json(
                condition.yarn_table,
                self.receipt(condition, updates={"model_weight_updates": 1}),
            )
            with self.assertRaisesRegex(ValueError, "zero-training installation"):
                validate_yarn_table(condition.yarn_table, condition)

    def test_runtime_signature_normalizes_missing_left_pad(self):
        base = {
            "base_arm": "Native",
            "unadapted": True,
            "row_ids": ["r"],
            "generation_length_caps": [16],
            "lm_enabled": False,
            "prefill_chunk_size": 8,
            "batch_size": 1,
            "runtime_versions": {"torch": "x"},
            "row_split": None,
        }
        explicit = dict(base, left_pad_batches=False)
        expected = _recovery_runtime_signature(explicit)
        self.assertEqual(_recovery_runtime_signature(base), expected)
        self.assertEqual(expected["generation_order"], "panel_order_v1")
        self.assertEqual(expected["generation_prefill_strategy"], "dynamic_cache_lower_right_v1")

    def test_legacy_runtime_matches_only_recorded_fields(self):
        expected = {"batch_size": 1, "runtime_versions": None}
        actual = {"batch_size": 1, "runtime_versions": {"torch": "new"}}
        self.assertTrue(_runtime_matches(expected, actual))
        self.assertFalse(_runtime_matches(expected, {"batch_size": 2, "runtime_versions": None}))
        self.assertFalse(_runtime_matches(
            {"batch_size": 1, "runtime_versions": {"torch": "old"}}, actual,
        ))

    def test_generation_validation_rejects_prompt_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            panel = [{
                "row_id": "r", "task": "longbook_qa_eng", "prompt_sha256": "p",
                "input_tokens": 3, "references": ["answer"],
            }]
            raw = self.raw(panel, "answer")
            raw[0]["prompt_sha256"] = "changed"
            write_json(run / "status.json", {"status": "COMPLETE", "rows": 1, "lm_rows": 0})
            write_jsonl(run / "generations.jsonl", raw)
            with self.assertRaisesRegex(ValueError, "prompt_sha256"):
                _generation_rows(run, panel, "yarn")

    def test_infinitebench_report_is_three_arm_official_and_not_pooled(self):
        with tempfile.TemporaryDirectory() as directory:
            condition = self.condition(Path(directory), family="infinitebench_en_qa", rows=35, clusters=7)
            panel = []
            for index in range(35):
                panel.append({
                    "row_id": f"row-{index}",
                    "task": "longbook_qa_eng",
                    "benchmark": "infinitebench",
                    "source_cluster_id": f"book-{index // 5}",
                    "source_id": str(index // 5),
                    "prompt_sha256": f"prompt-{index}",
                    "prompt_ids": [1, 2, index],
                    "input_tokens": 3,
                    "length_cap": 131072,
                    "length_bucket": "32769-131072",
                    "references": ["answer"],
                    "score_contract": "infinitebench_en_qa_rouge_f1_v1",
                })
            raw = {
                "tailspline": self.raw(panel, "answer"),
                "mrpro": self.raw(panel, "wrong"),
                "yarn": self.raw(panel, "wrong"),
            }
            self.materialize_report_inputs(condition, panel, raw)
            report = build_report(condition, panel, raw, {"runtime_versions": {}}, None)
            self.assertEqual(report["method_identity"], METHOD_IDENTITY)
            self.assertFalse(report["cross_model_pooling_allowed"])
            self.assertEqual(report["source_context_clusters"], 7)
            self.assertEqual(report["arms"]["tailspline"]["qa_f1"], 1.0)
            self.assertEqual(
                report["contrasts"]["tailspline_minus_yarn"]["comparison_family_size"], 2,
            )
            self.assertEqual(
                set(report["arms"]["tailspline"]["output_health"]["by_task"]),
                {"longbook_qa_eng"},
            )

    def test_longbench_report_uses_task_equal_family_two(self):
        with tempfile.TemporaryDirectory() as directory:
            condition = self.condition(Path(directory), family="longbench_naturalqa631", rows=5, clusters=5)
            panel = [{
                "row_id": task,
                "task": task,
                "document_cluster_id": f"doc-{task}",
                "prompt_sha256": f"prompt-{task}",
                "prompt_ids": [1],
                "input_tokens": 1,
                "length_cap": 131072,
                "references": ["answer"],
            } for task in ("hotpotqa", "2wikimqa", "qasper", "narrativeqa", "multifieldqa_en")]
            raw = {
                "tailspline": self.raw(panel, "answer"),
                "mrpro": self.raw(panel, "wrong"),
                "yarn": self.raw(panel, "wrong"),
            }
            self.materialize_report_inputs(condition, panel, raw)
            report = build_report(condition, panel, raw, {"runtime_versions": {}}, None)
            self.assertEqual(report["arms"]["tailspline"]["macro_f1"], 1.0)
            self.assertEqual(
                report["contrasts"]["tailspline_minus_mrpro"]["comparison_family_size"], 2,
            )
            self.assertFalse(report["cross_model_pooling_allowed"])


if __name__ == "__main__":
    unittest.main()
