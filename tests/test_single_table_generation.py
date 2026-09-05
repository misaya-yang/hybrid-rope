import importlib.util
import json
import sys
from collections import Counter
from types import SimpleNamespace
from pathlib import Path

import pytest

from scripts.lib.rope.generation_contract import (
    matched_projection_rank, projection_parameter_count,
    retention_verdict, token_exact_eos,
    paired_retention_intervals,
)

SPEC = importlib.util.spec_from_file_location("single_table_generation", Path(__file__).parents[1] / "scripts/experiments/single_table_generation.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "USER\n" + messages[0]["content"] + "\nASSISTANT\n"


def pair(task="lookup", factor=1):
    tok = CharacterTokenizer()
    return [M.render_case(tok, group="a" * 64, task=task, length=4096*factor,
                          nonce=[123456, 234567, 345678, 456789, 567890], variant=v)
            for v in (0, 1)]


@pytest.mark.parametrize("task", ["lookup", "chain"])
@pytest.mark.parametrize("factor", [1, 2, 4, 8, 16, 32])
def test_physical_source_twins_and_oracle(task, factor):
    twins = pair(task, factor)
    M.validate_pair(twins)
    for row in twins:
        for condition in ("remote", "oracle", "deleted"):
            assert len(M.condition_prompt(row, condition)) + 64 == 4096*factor
            text = CharacterTokenizer().decode(M.condition_prompt(row, condition))
            assert M.reference_answer(text) == (None if condition == "deleted" else row["answer"])
        oracle = M.condition_prompt(row, "oracle")
        assert Counter(oracle)==Counter(row['input_ids'])
        assert len(M.condition_prompt(row,'compact'))+64 <= 2048
        assert M.reference_answer(CharacterTokenizer().decode(M.condition_prompt(row,'compact'))) == row['answer']
        assert oracle != row["input_ids"]
    assert M.condition_prompt(twins[0], "deleted") == M.condition_prompt(twins[1], "deleted")


def test_pair_rejects_query_leakage_and_missing_source():
    twins = pair()
    twins[1]["input_ids"][-5] += 1
    with pytest.raises(ValueError, match="outside the source"):
        M.validate_pair(twins)
    twins = pair()
    twins[0]["input_ids"][twins[0]["source_spans"][0]["start"]] += 1
    with pytest.raises(ValueError, match="source location"):
        M.validate_pair(twins)


@pytest.mark.parametrize("bad", [[1, 2], [1, 2, 0, 3], [1, 2, 0, 0], [9, 1, 2, 0], [1, 0]])
def test_complete_output_requires_exact_terminal_eos(bad):
    assert not token_exact_eos(bad, [1, 2], 0)
    assert token_exact_eos([1, 2, 0], [1, 2], 0)


def test_generation_diagnosis_does_not_award_substrings():
    tok = CharacterTokenizer()
    gold = tok.encode("123456|654321")
    good = M.target_score(gold + [0], gold, 0, tok.decode)
    assert good["exact_eos"] and good["route_exact"] and good["answer_exact"]
    bad = M.target_score(tok.encode("Answer: 123456|654321") + [0], gold, 0, tok.decode)
    assert not bad["exact_eos"] and not bad["route_exact"]
    no_eos = M.target_score(gold, gold, 0, tok.decode)
    assert no_eos["answer_exact"] and not no_eos["exact_eos"]
    assert not token_exact_eos([1, 0, 0], [1], 0)


def test_retention_conjunction_and_marginal_boundary():
    import math
    assert retention_verdict(3., 3.-math.log(.877), .5, .5)["decision"] == "MARGINAL"
    assert retention_verdict(3., 3., .5, .4)["decision"] == "STOP"
    assert retention_verdict(3., 3., .5, .5)["strict_088_pass"]
    with pytest.raises(ValueError):
        retention_verdict(3., 3., 0., .5)


def test_retention_bootstrap_preserves_pairs_and_group_unit():
    original = [{"task": task, "asset_sha256": f"{task}{i}", "group": f"source{i//2}",
                 **({"nll": 2.+i/10} if task == "pg19" else {"score": .5, "score_eos": .5})}
                for task in ("pg19", "qa") for i in range(8)]
    identical = paired_retention_intervals(original, original, resamples=100)
    assert identical["ci95"] == {"ppl": [1.,1.], "task": [1.,1.], "task_eos": [1.,1.]}
    assert identical["groups_per_task"] == {"pg19": 4, "qa": 4}
    with pytest.raises(ValueError, match="unpaired"):
        paired_retention_intervals(original, original[:-1])


def test_natural_metric_dependency_known_cases():
    from scripts.eval.longbench_metrics import qa_f1_score, rouge_l_f1
    assert qa_f1_score("red red", ["red blue"]) == .5
    assert qa_f1_score("The book.", ["book"]) == 1.
    assert qa_f1_score("different", ["answer"]) == 0.
    assert rouge_l_f1("x y x", ["x x z"]) == pytest.approx(2/3)


@pytest.mark.parametrize("kv", [2, 8])
def test_gqa_actual_parameter_matching(kv):
    config = {"hidden_size": 512, "num_attention_heads": 8, "num_key_value_heads": kv, "num_hidden_layers": 8}
    rank, actual = matched_projection_rank(config, M.QKVO, 64)
    assert rank == 32
    assert actual == projection_parameter_count(config, M.QK, 64)


def test_summary_recomputes_raw_and_requires_complete_twins(tmp_path):
    tok = CharacterTokenizer()
    records = []
    for row in pair():
        generated = row["answer_ids"] + [0]
        record = {k: row[k] for k in ("group", "task", "length", "variant")}
        record.update(condition="remote", generated_token_ids=generated, answer_token_ids=row["answer_ids"], eos_token_id=0,
                      **M.target_score(generated, row["answer_ids"], 0, tok.decode))
        records.append(record)
    def write():
        M.write_json(tmp_path / "run.json", {"expected_rows": len(records)})
        (tmp_path / "examples.jsonl").write_text("\n".join(json.dumps(r) for r in records))
        M.write_json(tmp_path / "complete.json", {"rows": len(records), "examples_sha256": M.sha(tmp_path / "examples.jsonl"), "run_sha256": M.sha(tmp_path / "run.json")})
    write()
    M.summarize(tmp_path, tok)
    assert json.loads((tmp_path / "summary.json").read_text())["cells"]["lookup:4096:remote"]["both_twins_exact_eos"] == 1
    records[0]["generated_token_ids"].pop()
    write()
    with pytest.raises(ValueError, match="raw tokens"):
        M.summarize(tmp_path, tok)
    records[0].update(M.target_score(records[0]["generated_token_ids"], records[0]["answer_token_ids"], 0, tok.decode))
    records.pop()
    write()
    with pytest.raises(ValueError, match="missing source twin"):
        M.summarize(tmp_path, tok)


def test_prepare_preflight_and_group_leak_rejection(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "tokenizer.json").write_text("{}")
    (checkpoint / "config.json").write_text("{}")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: CharacterTokenizer())))
    output = tmp_path / "data"
    M.prepare(SimpleNamespace(checkpoint=checkpoint, output=output, seed=42, train_pairs=2, eval_pairs=2))
    selected, manifest = M.load_data(output, "blind", (8,16,32))
    assert len(selected) == 12
    assert {r["task"] for r in selected} == {"lookup", "chain"}
    manifest["groups"]["blind"][0] = manifest["groups"]["train"][0]
    M.write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="split leakage"):
        M.load_data(output, "blind", (8,))


def make_retention_assets(root):
    manifest = {"tokenization_executed": True, "native_context_length": 4096,
                "tokenizer_sha256": "frozen", "longbench": {"cells": {}}}
    for task in ("pg19", *M.NATURAL_TASKS):
        data = []
        for i in range(8):
            row = {"input_ids": [1,2,3,4], "tokenizer_sha256": "frozen"}
            if task == "pg19":
                row.update(multiplier=1, anchor_source_sha256=f"book{i}", nll_target_start=2,
                           nll_target_tokens=2, nll_target_sha256=M.canonical([3,4]))
            else:
                row.update(task=task, source_id=f"doc{i}", bucket="retention",
                           generation_reserve=64, references=["answer"])
            data.append(row)
        path = root / f"{task}.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in data))
        entry = {"rows_path": path.name, "rows_sha256": M.sha(path)}
        if task == "pg19":
            manifest["pg19"] = entry
        else:
            manifest["longbench"]["cells"][f"longbench:{task}"] = entry
    path = root / "token_manifest.json"
    M.write_json(path, manifest)
    return path


def test_native_asset_split_hash_and_physical_budget(tmp_path):
    path = make_retention_assets(tmp_path)
    a = M.retention_rows(path, "selection")
    b = M.retention_rows(path, "confirmation")
    assert len(a) == len(b) == 24
    assert not {(r["task"], r["group"]) for r in a} & {(r["task"], r["group"]) for r in b}
    assert len(M.retention_rows(path, "selection", True)) == 4
    qasper = tmp_path / "qasper.jsonl"
    data = list(M.rows(qasper))
    data[0]["generation_reserve"] = 4096
    qasper.write_text("\n".join(json.dumps(row) for row in data))
    with pytest.raises(ValueError, match="hash drift"):
        M.retention_rows(path, "selection")
    manifest = json.loads(path.read_text())
    manifest["longbench"]["cells"]["longbench:qasper"]["rows_sha256"] = M.sha(qasper)
    M.write_json(path, manifest)
    with pytest.raises(ValueError, match="decode reserve"):
        M.retention_rows(path, "selection") + M.retention_rows(path, "confirmation")


def test_seal_refuses_unresolved_controls(tmp_path, monkeypatch):
    # Existing complete two-row panel is deliberately too small to resolve.
    selection = tmp_path / "selection"
    selection.mkdir()
    records = []
    for row in pair():
        record = {k: row[k] for k in ("group", "task", "length", "variant")}
        tokens = row["answer_ids"] + [0]
        record.update(condition="remote", generated_token_ids=tokens, answer_token_ids=row["answer_ids"], eos_token_id=0,
                      **M.target_score(tokens, row["answer_ids"], 0, CharacterTokenizer().decode))
        records.append(record)
    (selection / "examples.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    ident = {"table_sha256": "table", "gain": 1., "adapter_sha256": None,"adapter_config_sha256":None}
    M.write_json(selection / "run.json", {**ident, "expected_rows": 2, "split": "selection", "factors": [1,4]})
    M.write_json(selection / "complete.json", {"rows": 2, "examples_sha256": M.sha(selection / "examples.jsonl"), "run_sha256": M.sha(selection / "run.json")})
    retain = tmp_path / "retention"
    retain.mkdir()
    (retain / "examples.jsonl").write_text("{}")
    M.write_json(retain / "retention.json", {**ident, "strict_pass": True, "fold": "selection", "examples_sha256": M.sha(retain / "examples.jsonl")})
    original_summary = M.summarize
    monkeypatch.setattr(M, "summarize", lambda path: original_summary(path, CharacterTokenizer()))
    with pytest.raises(ValueError, match="not resolved"):
        M.seal(SimpleNamespace(selection_result=selection, retention_result=retain, output=tmp_path / "lock.json"))


def test_exact_string_accepts_equivalent_tokenization_but_no_extra_text():
    gold = [1, 2, 3]
    decode = lambda ids: "".join({1: "123", 2: "|", 3: "456", 4: "123|456", 5: "extra"}[i] for i in ids)
    score = M.target_score([4,0], gold, 0, decode)
    assert score["exact_eos"] and not score["canonical_token_exact_eos"]
    assert not M.target_score([4,5,0], gold, 0, decode)["exact_eos"]
