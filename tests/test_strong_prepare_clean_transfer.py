import hashlib
import json
from pathlib import Path

import pytest

from experiments.iclr2027_strong_evidence_20260915 import prepare_clean_transfer as subject


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, model_id: str = "olmo2_1b"):
    model = tmp_path / "model"
    model.mkdir()
    _write_json(model / "config.json", {
        "model_type": "olmo2" if model_id.startswith("olmo") else "qwen2",
        "architectures": ["Olmo2ForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "rope_theta": 500000.0,
        "max_position_embeddings": 4096,
    })
    _write_json(model / "tokenizer.json", {"version": "1.0"})
    data_root = tmp_path / f"RULER-{subject.RULER_REVISION}"
    data_root.mkdir()
    out = tmp_path / "prepared"
    return model, data_root, out


def _fake_cores(out: Path):
    planb_calls = []
    converter_calls = []

    def planb(argv):
        values = dict(zip(argv[::2], argv[1::2]))
        # Flags have no value, so parse the handful of required values directly.
        get = lambda name: argv[argv.index(name) + 1]
        task = get("--tasks")
        target = Path(get("--out"))
        lengths = [int(value) for value in get("--caps").split(",")]
        rows = int(get("--counts-by-cap").split(":", 1)[1].split(",", 1)[0])
        _write_json(target / "manifest.json", {
            "status": "COMPLETE", "model": get("--model"), "rows": rows * len(lengths),
        })
        _write_json(target / "rows.jsonl", {"max_new_tokens": 16})
        for length in lengths:
            source = target / "source" / str(length) / task / "validation.jsonl"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("\n".join(json.dumps({
                "input": f"{task}-{length}-{index}", "outputs": [str(index)],
            }) for index in range(rows)) + "\n")
        planb_calls.append(list(argv))
        return 0

    def converter(argv):
        get = lambda name: argv[argv.index(name) + 1]
        panel = Path(get("--out"))
        panel.mkdir(parents=True, exist_ok=True)
        length = int(get("--length"))
        count = int(get("--rows-per-task"))
        source_root = Path(get("--source-parts"))
        records = []
        sources = {}
        for task in subject.TASKS:
            source = source_root / task / "source" / str(length) / task / "validation.jsonl"
            sources[task] = [{"path": str(source.resolve()), "sha256": _sha(source)}]
            for index in range(count):
                prompt = [length, len(records) + 1]
                records.append({
                    "row_id": f"clean_{task}_{length}_{index:04d}",
                    "task": task,
                    "length_cap": length,
                    "prompt_ids": prompt,
                    "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                    "input_tokens": len(prompt),
                    "actual_length": len(prompt),
                    "references": [str(index)],
                    "max_new_tokens": 16,
                    "selection_mode": "source-order",
                    "selection_uses_model_outputs": False,
                    "irrelevant_padding_tokens": 0,
                })
        inputs = panel / "inputs.jsonl"
        inputs.write_text("\n".join(json.dumps(row) for row in records) + "\n")
        _write_json(panel / "manifest.json", {
            "status": "COMPLETE",
            "contract": "TAILSPLINE_LLAMA_HISTORICAL_CONVERTER",
            "sources": sources,
        })
        converter_calls.append(list(argv))
        return 0

    return planb, converter, planb_calls, converter_calls


def _argv(model: Path, data_root: Path, out: Path, model_id: str = "olmo2_1b"):
    return [
        "--model", str(model), "--model-id", model_id,
        "--data-root", str(data_root), "--out", str(out),
        "--scale", "4", "--lengths", "8192,16384",
        "--rows-per-task", "2", "--seed", "20261101", "--qa-offset", "7000",
    ]


def test_prepares_generic_full13_source_order_panels_without_llama_label(tmp_path):
    model, data_root, out = _fixture(tmp_path)
    planb, converter, planb_calls, converter_calls = _fake_cores(out)
    manifest = subject.prepare(
        _argv(model, data_root, out), planb_main=planb, converter_main=converter,
    )

    assert len(planb_calls) == 13
    assert len(converter_calls) == 2
    assert all("--model-contract" in call and call[call.index("--model-contract") + 1] == "generic"
               for call in planb_calls)
    assert all("--selection-mode" in call and call[call.index("--selection-mode") + 1] == "source-order"
               for call in planb_calls)
    assert all("--source-only" in call for call in planb_calls)
    assert manifest["model_id"] == "olmo2_1b"
    assert manifest["model_identity"]["model_type"] == "olmo2"
    assert manifest["rows"] == 13 * 2 * 2
    assert set(manifest["panels"]) == {"8192", "16384"}

    for length in (8192, 16384):
        panel_manifest = json.loads((out / "panels" / str(length) / "manifest.json").read_text())
        assert "LLAMA" not in panel_manifest["contract"].upper()
        assert panel_manifest["tasks"] == list(subject.TASKS)
        assert panel_manifest["rows"] == 26
        assert panel_manifest["content_padding"] is False
        assert panel_manifest["selection_uses_model_outputs"] is False
        assert all(not Path(source["artifact"]).is_absolute()
                   for sources in panel_manifest["sources"].values() for source in sources)

    # Re-freezing a completed first panel is the interrupted two-length resume
    # path: its source records already carry ``artifact`` rather than ``path``.
    first_path = out / "panels" / "8192" / "manifest.json"
    before = first_path.read_bytes()
    subject._freeze_panel_manifest(
        panel_dir=first_path.parent, out=out, model_id="olmo2_1b",
        identity=manifest["model_identity"], scale=4.0, length=8192,
        rows_per_task=2,
    )
    assert first_path.read_bytes() == before


def test_frozen_manifest_skips_equal_request_and_rejects_artifact_drift(tmp_path):
    model, data_root, out = _fixture(tmp_path, "qwen25_3b")
    planb, converter, planb_calls, converter_calls = _fake_cores(out)
    argv = _argv(model, data_root, out, "qwen25_3b")
    first = subject.prepare(argv, planb_main=planb, converter_main=converter)
    second = subject.prepare(
        argv,
        planb_main=lambda _: pytest.fail("frozen request should not regenerate sources"),
        converter_main=lambda _: pytest.fail("frozen request should not reconvert"),
    )
    assert second == first

    with (out / "panels" / "8192" / "inputs.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="drifted"):
        subject.prepare(argv, planb_main=planb, converter_main=converter)


def test_panel_validation_rejects_padding_or_incomplete_full13(tmp_path):
    path = tmp_path / "inputs.jsonl"
    rows = []
    for task in subject.TASKS:
        rows.append({
            "row_id": task,
            "task": task,
            "length_cap": 8192,
            "prompt_ids": [1, 2],
            "prompt_sha256": task,
            "input_tokens": 2,
            "actual_length": 2,
            "max_new_tokens": 16,
            "selection_mode": "source-order",
            "selection_uses_model_outputs": False,
            "irrelevant_padding_tokens": 0,
        })
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    rows[0]["irrelevant_padding_tokens"] = 1
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="unpadded"):
        subject._validate_panel_rows(path, length=8192, rows_per_task=1)

    path.write_text("\n".join(json.dumps(row) for row in rows[1:]) + "\n")
    with pytest.raises(ValueError, match="Full-13"):
        subject._validate_panel_rows(path, length=8192, rows_per_task=1)
