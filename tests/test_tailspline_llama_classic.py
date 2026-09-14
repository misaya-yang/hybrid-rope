import hashlib
import json

import pytest

from experiments.fixed_rope_three_interfaces_20260913.build_mrrope_baseline_registry import (
    discover,
    summarize_source,
)
from experiments.fixed_rope_three_interfaces_20260913.prepare_llama_ppl46 import (
    validate_sources,
)
from experiments.fixed_rope_three_interfaces_20260913.tailspline_llama_classic_report import (
    COUNTS,
    LENGTHS,
    TASKS,
    ppl_bootstrap,
    ppl_summary,
)
from experiments.llama3_60dir_20260911.prepare_planb_panel import (
    REGISTERED_TASKS,
    select_depth_balanced,
)


def test_full13_contract_has_390_rows_and_all_public_tasks():
    assert len(TASKS) == 13
    assert TASKS == REGISTERED_TASKS
    assert COUNTS == {8192: 10, 16384: 10, 32768: 10}
    assert len(TASKS) * sum(COUNTS.values()) == 390


def test_depth_selector_freezes_10_30_50_70_90_without_model_outputs():
    cap = 10_000
    targets = (0.10, 0.30, 0.50, 0.70, 0.90)
    candidates = []
    for repeat in range(2):
        for depth in targets:
            candidates.append({
                "row_id": f"r-{repeat}-{depth}",
                "evidence_positions": [int(depth * cap)],
            })
    selected = select_depth_balanced(
        "niah_single_1", candidates, count=10, cap=cap, pilot=False,
        depth_targets=targets,
    )
    assert [tuple(row["depth_target"]) for row in selected] == [
        (targets[index % 5],) for index in range(10)
    ]
    assert all(row["depth_error_mean_abs"] == 0.0 for row in selected)


def test_ppl46_source_validation_preserves_32_proofpile_14_pg19(tmp_path):
    docs = []
    for index in range(46):
        dataset = "proofpile" if index < 32 else "pg19"
        path = tmp_path / f"{dataset}_{index}.txt"
        payload = f"held-out-{dataset}-{index}".encode()
        path.write_bytes(payload)
        docs.append({
            "dataset": dataset,
            "split": "test",
            "file": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"docs": docs}))
    validated = validate_sources(tmp_path, manifest)
    assert len(validated) == 46
    assert sum(row["dataset"] == "proofpile" for row in validated) == 32
    (tmp_path / docs[0]["file"]).write_text("drift")
    with pytest.raises(ValueError, match="hash drift"):
        validate_sources(tmp_path, manifest)


def test_ppl_report_is_token_weighted_and_document_paired():
    datasets = ["proofpile"] * 32 + ["pg19"] * 14
    candidate = []
    baseline = []
    for document in range(46):
        for length in LENGTHS:
            target_count = length - 1
            candidate.append({
                "document": document, "length": length,
                "whole_loss_sum": 2.0 * target_count,
                "whole_target_count": target_count,
            })
            baseline.append({
                "document": document, "length": length,
                "whole_loss_sum": 2.1 * target_count,
                "whole_target_count": target_count,
            })
    summary = ppl_summary(candidate, datasets)
    assert summary["combined"]["by_length"]["8192"]["documents"] == 46
    assert summary["proofpile"]["by_length"]["32768"]["documents"] == 32
    assert summary["pg19"]["by_length"]["32768"]["documents"] == 14
    bootstrap = ppl_bootstrap(candidate, baseline, draws=30, seed=914)
    delta = bootstrap["delta_log_length_ppl_auc"]
    assert delta["mean"] < 0.0
    assert delta["interval95"][1] < 0.0
    assert delta["negative_is_better"] is True


def test_mrrope_registry_discovers_complete_raw_without_moving_it(tmp_path):
    run = tmp_path / "experiment" / "run_MrPro_s4"
    run.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({
        "status": "COMPLETE", "rows": 1, "lm_rows": 0,
    }))
    (run / "generations.jsonl").write_text(json.dumps({
        "task": "niah_single_1", "length_cap": 8192,
        "prompt_sha256": "p", "ruler_official_score": 1.0,
    }) + "\n")
    found = discover(tmp_path)
    assert found == [run]
    record = summarize_source(tmp_path, run)
    assert record["source_path"] == str(run)
    assert record["rows_observed"] == 1
    assert record["tasks"] == ["niah_single_1"]
    assert record["lengths"] == [8192]
    assert record["reuse_class"] == "raw_reusable_with_exact_matching_contract"
