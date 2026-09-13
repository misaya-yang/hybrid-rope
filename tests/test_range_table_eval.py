import json
import math
from pathlib import Path
from types import SimpleNamespace

from experiments.olmo_recovery_20260912.recovery_v2_eval import collect_rows
from experiments.olmo_recovery_20260912.summarize_range_generation import log_auc, parse_arm


def test_explicit_panel_row_split_is_disjoint(tmp_path):
    panel = tmp_path / "rows.jsonl"
    rows = [
        {
            "row_id": split,
            "split": split,
            "task": "niah_single_1",
            "prompt_ids": [1, 2, 3],
            "references": ["answer"],
            "max_new_tokens": 4,
            "length_cap": 16,
        }
        for split in ("fit", "select", "internal_confirm")
    ]
    panel.write_text("".join(json.dumps(row) + "\n" for row in rows))
    args = SimpleNamespace(
        only_extra_panels=True,
        extra_panel=[panel],
        regression_data=None,
        split="dev",
        row_split="select",
        limit_per_cell=0,
    )
    selected = collect_rows(args, {})
    assert [row["row_id"] for row in selected] == ["select"]


def test_range_summary_arm_parser_and_log_auc():
    assert parse_arm("candidate=/tmp/run") == ("candidate", Path("/tmp/run"))
    assert math.isclose(log_auc({4: 0.5, 8: 1.0, 16: 0.5}), 0.75)
