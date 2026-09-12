"""Independent regression tests for the Plan B statistics contract.

These tests use synthetic rows only.  They deliberately cover failure cases
that the broad operator/tool test does not exercise: identity drift, partial
arms, long-only filtering, and bootstrap cell completeness.
Run with ``.venv/bin/python test_stats_contract.py`` from this directory.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile


HERE = __import__("pathlib").Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("paired_report", HERE / "paired_report.py")
P = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = P
spec.loader.exec_module(P)


def rows(tasks=("t0", "t1"), lengths=(4096, 16384, 32768), n=2):
    out = []
    for task in tasks:
        for length in lengths:
            for i in range(n):
                out.append({
                    "row_id": f"{task}-{length}-{i}",
                    "task": task,
                    "length_cap": length,
                    "source_id": f"{task}-source-{i}",
                    "semantic_group_id": f"{task}-group-{i}",
                    "prompt_input_ids_sha256": f"prompt-{task}-{length}-{i}",
                    "correct": 0,
                })
    return out


def expect_error(fn, label):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"{label}: did not reject")


def main():
    base = rows()

    changed = [dict(r) for r in base]
    changed[0]["source_id"] = "wrong-source"
    expect_error(lambda: P.require_aligned(
        {"base": P.index_arm(base), "changed": P.index_arm(changed)}),
        "identity drift")

    partial = base[:-1]
    expect_error(lambda: P.paired_stats(base, partial), "partial paired arm")

    # A difference confined to the 4K guard must not enter long-only N, q, or SE.
    guard_gain = [dict(r) for r in base]
    for r in guard_gain:
        if r["length_cap"] == 4096:
            r["correct"] = 1
    st = P.paired_stats(base, guard_gain, use_lengths={16384, 32768})
    assert st["n_pairs"] == 8, st
    assert st["discordance_q"] == 0.0, st
    assert st["se_paired"] == 0.0, st

    # A full-grid requirement is not allowed to silently average the cells
    # that happen to remain.
    missing = [r for r in base if not (r["task"] == "t1" and r["length_cap"] == 32768)]
    expect_error(lambda: P.macro_accuracy(
        missing, required_tasks=["t0", "t1"], required_lengths=[16384, 32768]),
        "missing task x length cell")

    arms = {"MR": P.index_arm(base), "M": P.index_arm(base), "YARN": P.index_arm(base)}
    contrasts = [{"name": "d", "terms": [("M", 1.0), ("MR", -1.0)],
                  "lengths": [16384, 32768]}]
    boot = P.cluster_bootstrap(
        arms, contrasts, n_boot=100, seed=7,
        required_tasks=["t0", "t1"], required_lengths=[16384, 32768])
    assert boot["n_boot_rejected_empty_cell"] > 0, boot
    assert boot["contrasts"]["d"]["point"] == 0.0, boot

    missing_arm = {k: dict(v) for k, v in arms.items()}
    missing_arm["M"] = {rid: r for rid, r in missing_arm["M"].items()
                         if not (r["task"] == "t1" and r["length_cap"] == 32768)}
    expect_error(lambda: P.cluster_bootstrap(
        missing_arm, contrasts, n_boot=20, required_tasks=["t0", "t1"],
        required_lengths=[16384, 32768]), "bootstrap partial arm")

    # Dedicated metrics must win over the legacy `correct`/partial field.
    metric_row = dict(base[0], correct=0.0, partial_score=0.0,
                      strict_score=1.0, qa_em=1.0, qa_f1=0.5)
    assert P.macro_accuracy([metric_row], required_tasks=["t0"],
                            required_lengths=[4096], score_key="partial")["macro"] == 0.0
    assert P.macro_accuracy([metric_row], required_tasks=["t0"],
                            required_lengths=[4096], score_key="strict_score")["macro"] == 1.0
    assert P.macro_accuracy([metric_row], required_tasks=["t0"],
                            required_lengths=[4096], score_key="qa_em")["macro"] == 1.0
    assert P.macro_accuracy([metric_row], required_tasks=["t0"],
                            required_lengths=[4096], score_key="qa_f1")["macro"] == 0.5
    partial_winner = dict(metric_row, partial_score=1.0, correct=0.0)
    partial_delta = P.paired_stats([metric_row], [partial_winner],
                                   use_lengths={4096})
    assert partial_delta["delta_macro"] == 1.0, partial_delta

    # The CLI's default native guard is 8K, and QA reports both EM and F1.
    with tempfile.TemporaryDirectory() as td:
        root = __import__("pathlib").Path(td)
        panel = []
        for task in ("t0", "t1"):
            for length in (8192, 16384, 32768):
                for i in range(2):
                    panel.append({"row_id": f"{task}-{length}-{i}", "task": task,
                                  "length_cap": length, "source_id": f"{task}-src-{i}",
                                  "semantic_group_id": f"{task}-grp-{i}",
                                  "correct": 0.0})
        def dump(name, data):
            path = root / name
            path.write_text("\n".join(json.dumps(x) for x in data), encoding="utf-8")
            return str(path)
        exp = dump("expected.jsonl", panel)
        mr = dump("mr.jsonl", panel)
        cand = dump("cand.jsonl", [dict(x, correct=1.0) for x in panel])
        yarn = dump("yarn.jsonl", panel)
        native = dump("native.jsonl", [dict(x, correct=1.0)
                                       for x in panel if x["length_cap"] == 8192])
        strict_c = [dict(x, strict_score=1.0) for x in panel if x["length_cap"] != 8192]
        strict_m = [dict(x, strict_score=0.0) for x in panel if x["length_cap"] != 8192]
        qa_c = [dict(x, qa_em=1.0, qa_f1=0.5) for x in panel if x["length_cap"] != 8192]
        qa_m = [dict(x, qa_em=0.0, qa_f1=0.0) for x in panel if x["length_cap"] != 8192]
        out = root / "report.json"
        cmd = [sys.executable, str(HERE / "upgrade_report.py"),
               "--mr", mr, "--candidate", cand, "--yarn", yarn,
               "--expected-data", exp, "--native", native,
               "--strict", dump("strict_c.jsonl", strict_c),
               "--strict-mr", dump("strict_m.jsonl", strict_m),
               "--qa", dump("qa_c.jsonl", qa_c), "--qa-mr", dump("qa_m.jsonl", qa_m),
               "--n-boot", "30", "--out", str(out)]
        run = subprocess.run(cmd, capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["retention_8K"]["native"] == 1.0
        assert report["strict_guard"]["score_keys"] == ["strict_score"]
        assert report["qa_guard"]["score_keys"] == ["qa_em", "qa_f1"]
        assert report["qa_guard"]["metrics"]["qa_em"]["delta"] == 1.0
        assert report["qa_guard"]["metrics"]["qa_f1"]["delta"] == 0.5

    print("stats contract tests pass")


if __name__ == "__main__":
    main()
