from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq import contract
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq import (
    prepare_official_stream as stream,
)
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq import prepare_eval_data
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq import (
    prepare_retrieval_data,
)
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq import prepare_assets
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq import train
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq import select_runtime
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq import preflight
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq import evaluate
from rebuttal.rebuttal_0723.experiments.olmo2_1b_evq import (
    prepare_ruler_data,
)


def test_endpoint_evq_tau_zero_is_native_geo() -> None:
    receipt = contract.assert_frequency_contract()
    assert receipt["grid"] == "endpoint"
    assert receipt["tau"] == 2.0
    assert receipt["evq"][0] == 1.0
    native_float32 = 1.0 / (
        torch.tensor(500_000.0, dtype=torch.float32)
        ** (
            torch.arange(0, 128, 2, dtype=torch.int64).to(torch.float32)
            / 128.0
        )
    )
    assert torch.equal(contract.endpoint_geo_inv_freq(), native_float32)
    assert torch.equal(
        contract.endpoint_geo_inv_freq(),
        contract.endpoint_evq_inv_freq(tau=0.0),
    )


def test_model_config_contract() -> None:
    config = SimpleNamespace(
        architectures=["Olmo2ForCausalLM"],
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=16,
        max_position_embeddings=4096,
        rope_theta=500000,
        vocab_size=100352,
        tie_word_embeddings=False,
    )
    contract.assert_model_config(config)
    config.rope_theta = 10_000
    try:
        contract.assert_model_config(config)
    except RuntimeError as error:
        assert "config drift" in str(error)
    else:
        raise AssertionError("config drift was not rejected")


def test_released_step2000_contract_is_pinned() -> None:
    assert contract.GEO2000_REVISION == (
        "07a0254be1449bcc7f78d05602d835464abf43c6"
    )
    assert set(contract.GEO2000_WEIGHT_FILES) == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert sum(
        row["size"] for row in contract.GEO2000_WEIGHT_FILES.values()
    ) == 5_939_687_552


def test_released_step5000_contract_is_pinned() -> None:
    assert contract.GEO5000_REVISION == (
        "c667e21af89dd476540c52a2e9912f75580c8271"
    )
    assert set(contract.GEO5000_WEIGHT_FILES) == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert sum(
        row["size"] for row in contract.GEO5000_WEIGHT_FILES.values()
    ) == 5_939_687_552


def test_32k_nll_summary_has_an_explicit_tail_bucket() -> None:
    row = evaluate.summarize_row(
        np.arange(32_767, dtype=np.float32),
        32_768,
    )
    assert "32768" in row
    assert "16-32K" in row["32768"]["position_bucket_nll"]


def test_ruler_fwe_preserves_and_records_bounded_upstream_overrun() -> None:
    rows = []
    for length in prepare_ruler_data.LENGTHS:
        for index in range(prepare_ruler_data.EXAMPLES_PER_CELL):
            rows.append(
                {
                    "index": index,
                    "input": "prompt",
                    "outputs": ["answer"],
                    "length": length + (50 if index == 0 else 0),
                    "max_length": length,
                    "gen_prefix": "Answer:",
                }
            )
    receipt = prepare_ruler_data.validate_rows(rows, "ruler_fwe")
    assert receipt["over_target_rows"] == len(prepare_ruler_data.LENGTHS)
    assert receipt["max_over_target_tokens"] == 50


def test_ruler_custom_length_sweep_is_validated() -> None:
    lengths = (4_096, 5_120)
    rows = []
    for length in lengths:
        for index in range(prepare_ruler_data.EXAMPLES_PER_CELL):
            rows.append(
                {
                    "index": index,
                    "input": "prompt",
                    "outputs": ["answer"],
                    "length": length,
                    "max_length": length,
                    "gen_prefix": "Answer:",
                }
            )
    receipt = prepare_ruler_data.validate_rows(
        rows,
        "niah_single_1",
        lengths=lengths,
    )
    assert receipt["per_length"] == {"4096": 500, "5120": 500}


def test_official_order_prefix_matches_numpy_shuffle() -> None:
    expected = np.arange(1000, dtype=np.uint32)
    np.random.Generator(np.random.PCG64(seed=contract.SEED)).shuffle(expected)
    actual = stream.generate_order_prefix(
        1000, 200, seed=contract.SEED
    )
    assert np.array_equal(actual, expected[:200])


def test_asset_download_retries_and_resumes_transport_failure(
    tmp_path, monkeypatch
) -> None:
    calls = []
    sleeps = []

    def flaky_snapshot_download(**kwargs: object) -> None:
        calls.append(kwargs)
        if len(calls) < 3:
            raise ConnectionError("transient partial read")

    monkeypatch.setattr(prepare_assets, "snapshot_download", flaky_snapshot_download)
    monkeypatch.setattr(prepare_assets.time, "sleep", sleeps.append)
    prepare_assets.download_snapshot(
        tmp_path,
        "pinned-revision",
        max_workers=4,
        attempts=4,
        retry_base_seconds=0.5,
    )
    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]
    assert all(call["local_dir"] == str(tmp_path) for call in calls)


def test_order_generation_obeys_cgroup_available_memory(
    monkeypatch,
) -> None:
    monkeypatch.setattr(stream, "memory_available_bytes", lambda: 100)
    with pytest.raises(MemoryError, match="exact OLMo permutation"):
        stream.generate_order_prefix(1_000, 10, seed=contract.SEED)


def test_source_mapping_preserves_duplicate_path_weight() -> None:
    rows = stream.source_rows(
        ["https://example/a", "https://example/a", "https://example/b"],
        {
            "https://example/a": stream.CHUNK_BYTES * 2,
            "https://example/b": stream.CHUNK_BYTES,
        },
    )
    prefix = np.asarray([0, 1, 2, 3, 4], dtype=np.uint32)
    sources, local = stream.map_indices(prefix, rows)
    assert sources.tolist() == [0, 0, 1, 1, 2]
    assert local.tolist() == [0, 1, 0, 1, 0]


def test_instance_filter_matches_registered_threshold() -> None:
    clean = np.arange(stream.SEQUENCE_LENGTH, dtype=np.uint32)
    assert stream.official_instance_valid(clean)

    repeated = clean.copy()
    repeated[: 32 * 3] = np.tile(
        np.asarray([11, 22, 33], dtype=np.uint32), 32
    )
    assert not stream.official_instance_valid(repeated)

    below_threshold = clean.copy()
    below_threshold[: 31 * 3] = np.tile(
        np.asarray([11, 22, 33], dtype=np.uint32), 31
    )
    assert stream.official_instance_valid(below_threshold)


def test_materializer_is_resumable_and_hashes_first_batch(
    tmp_path, monkeypatch
) -> None:
    rows = [
        {
            "url": "https://example/a",
            "global_start": 0,
            "global_end": contract.GLOBAL_BATCH_SEQUENCES,
        }
    ]
    source_ids = np.zeros(contract.GLOBAL_BATCH_SEQUENCES, dtype=np.uint16)
    local_indices = np.arange(
        contract.GLOBAL_BATCH_SEQUENCES, dtype=np.uint64
    )

    def fake_range_get(url: str, start: int, end: int, **_: object) -> bytes:
        assert url == "https://example/a"
        assert end - start + 1 == stream.CHUNK_BYTES
        index = start // stream.CHUNK_BYTES
        tokens = np.full(
            stream.SEQUENCE_LENGTH,
            index % contract.MODEL_CONTRACT["vocab_size"],
            dtype=np.uint32,
        )
        return tokens.tobytes()

    monkeypatch.setattr(stream, "range_get", fake_range_get)
    receipt = stream.materialize(
        tmp_path,
        rows,
        source_ids,
        local_indices,
        workers=4,
    )
    path = tmp_path / receipt["path"]
    assert path.stat().st_size == (
        contract.GLOBAL_BATCH_SEQUENCES * stream.CHUNK_BYTES
    )
    assert receipt["first_global_batch_sha256"] == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    second = stream.materialize(
        tmp_path,
        rows,
        source_ids,
        local_indices,
        workers=4,
    )
    assert second["sha256"] == receipt["sha256"]


def test_learning_rate_matches_official_token_warmup() -> None:
    assert train.learning_rate(1) == pytest.approx(1.0e-7)
    assert train.learning_rate(1000) == pytest.approx(1.0e-4)


def test_labels_mask_whole_filtered_instances() -> None:
    tokens = torch.arange(16).view(2, 8)
    labels = train.labels_for(tokens, torch.tensor([True, False]))
    assert labels[0].tolist() == list(range(1, 8))
    assert labels[1].tolist() == [-100] * 7


def test_native_loss_reports_ce_and_z_loss() -> None:
    torch.manual_seed(1)
    module = train.LanguageModelLoss("native")
    hidden = torch.randn(2, 5, 4, requires_grad=True)
    weight = torch.randn(7, 4, requires_grad=True)
    labels = torch.randint(0, 7, (2, 4))
    total, ce_sum, z_sum = module(weight, hidden, labels)
    assert total.ndim == ce_sum.ndim == z_sum.ndim == 0
    assert torch.isfinite(total)
    assert float(z_sum) > 0
    total.backward()
    assert hidden.grad is not None
    assert weight.grad is not None


def test_flash_only_sdpa_matches_explicit_causal_mask() -> None:
    torch.manual_seed(17)
    query = torch.randn(2, 3, 7, 8)
    key = torch.randn(2, 3, 7, 8)
    value = torch.randn(2, 3, 7, 8)
    actual, weights = train.flash_only_sdpa_forward(
        SimpleNamespace(),
        query,
        key,
        value,
        None,
        dropout=0.0,
        scaling=8**-0.5,
    )
    mask = torch.ones(7, 7, dtype=torch.bool).tril().view(1, 1, 7, 7)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=0.0,
        scale=8**-0.5,
        is_causal=False,
    ).transpose(1, 2).contiguous()
    torch.testing.assert_close(actual, expected)
    assert weights is None
    assert train.flash_only_causal_mask(attention_mask=None) is None
    with pytest.raises(RuntimeError, match="padding masks"):
        train.flash_only_causal_mask(attention_mask=mask)


def _probe_payload(backend: str, speed: float) -> dict:
    return {
        "status": "GPU_PROBE_PASS",
        "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "compute_capability": [12, 0],
        "torch_version": "2.8.0+cu128",
        "cuda_version": "12.8",
        "precision": "amp_bf16",
        "attention": {
            "implementation": "evq_flash_only_sdpa",
            "flash_enabled": True,
            "math_enabled": False,
            "memory_efficient_enabled": False,
            "cudnn_enabled": False,
        },
        "compile": {
            "enabled": True,
            "mode": "max-autotune-no-cudagraphs",
            "cache": "/tmp/torchinductor",
        },
        "schedule": "evq",
        "sequence_length": 4096,
        "global_batch_sequences": 512,
        "microbatch_sequences": 4,
        "gradient_accumulation": 128,
        "warmup_steps": 5,
        "measured_steps": 20,
        "loss_backend": backend,
        "total_memory_bytes": 96 * 1024**3,
        "peak_memory_bytes": 70 * 1024**3,
        "peak_memory_reserved_bytes": 72 * 1024**3,
        "optimizer_state_initialized": True,
        "tokens_per_second": speed,
        "first_measured_loss": 10.0,
        "last_measured_loss": 9.9,
        "first_measured_ce_loss": 9.99,
        "last_measured_ce_loss": 9.89,
        "first_measured_z_loss": 0.01,
        "last_measured_z_loss": 0.01,
        "first_measured_grad_norm": 2.0,
        "last_measured_grad_norm": 1.9,
    }


def test_probe_selector_chooses_liger_only_after_parity(tmp_path) -> None:
    probe_dir = tmp_path / "probes"
    probe_dir.mkdir()
    (probe_dir / "probe_native.json").write_text(
        json.dumps(_probe_payload("native", 25_000.0)), encoding="utf-8"
    )
    (probe_dir / "probe_liger.json").write_text(
        json.dumps(_probe_payload("liger", 30_000.0)), encoding="utf-8"
    )
    output = tmp_path / "selected.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.select_probe",
            "--probe-dir",
            str(probe_dir),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    selected = json.loads(output.read_text(encoding="utf-8"))
    assert selected["selected_loss_backend"] == "liger"

    bad_liger = _probe_payload("liger", 30_000.0)
    bad_liger["first_measured_loss"] = 11.0
    (probe_dir / "probe_liger.json").write_text(
        json.dumps(bad_liger), encoding="utf-8"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.select_probe",
            "--probe-dir",
            str(probe_dir),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    selected = json.loads(output.read_text(encoding="utf-8"))
    assert selected["selected_loss_backend"] == "native"


def test_probe_selector_uses_memory_for_near_tied_backends(tmp_path) -> None:
    probe_dir = tmp_path / "probes"
    probe_dir.mkdir()
    native = _probe_payload("native", 29_000.0)
    native["peak_memory_bytes"] = 77 * 1024**3
    native["peak_memory_reserved_bytes"] = 78 * 1024**3
    liger = _probe_payload("liger", 28_800.0)
    liger["peak_memory_bytes"] = 46 * 1024**3
    liger["peak_memory_reserved_bytes"] = 47 * 1024**3
    (probe_dir / "probe_native.json").write_text(
        json.dumps(native), encoding="utf-8"
    )
    (probe_dir / "probe_liger.json").write_text(
        json.dumps(liger), encoding="utf-8"
    )
    output = tmp_path / "selected.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rebuttal.rebuttal_0723.experiments.olmo2_1b_evq.select_probe",
            "--probe-dir",
            str(probe_dir),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    selected = json.loads(output.read_text(encoding="utf-8"))
    assert selected["selected_loss_backend"] == "liger"


def test_eval_long_candidates_never_cross_document_boundaries() -> None:
    eos = contract.TOKENIZER_MARKERS["eos_token_id"]
    long_a = np.arange(prepare_eval_data.LONG_LENGTH + 3, dtype=np.uint32)
    long_b = np.arange(
        10,
        10 + prepare_eval_data.LONG_LENGTH + 5,
        dtype=np.uint32,
    )
    tokens = np.concatenate(
        [long_a, np.asarray([eos], dtype=np.uint32), long_b, [eos]]
    ).astype(np.uint32)
    candidates = prepare_eval_data.build_long_candidates([("books", tokens)])
    assert len(candidates) == 2
    for _, _, window in candidates:
        assert len(window) == prepare_eval_data.LONG_LENGTH
        assert eos not in window[:-1]


def test_evq_smoke_requires_same_batches_and_comparable_dynamics(
    tmp_path,
) -> None:
    geo_log = tmp_path / "geo.jsonl"
    evq_log = tmp_path / "evq.jsonl"

    def rows(multiplier: float, *, mutate_hash: bool = False) -> list[dict]:
        return [
            {
                "step": step,
                "batch_sha256_uint32": (
                    "wrong" if mutate_hash and step == 7 else f"batch-{step}"
                ),
                "train_ce_loss": (10.0 - step * 0.01) * multiplier,
                "grad_norm_pre_clip": 2.0 * multiplier,
            }
            for step in range(1, 21)
        ]

    for path, payload in ((geo_log, rows(1.0)), (evq_log, rows(1.01))):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in payload),
            encoding="utf-8",
        )
    receipt = train.validate_smoke_against_geo(
        evq_log=evq_log,
        geo_log=geo_log,
        output=tmp_path / "smoke.json",
    )
    assert receipt["status"] == "EVQ_SMOKE_PASS"

    evq_log.write_text(
        "".join(
            json.dumps(row) + "\n" for row in rows(1.01, mutate_hash=True)
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="batches differ"):
        train.validate_smoke_against_geo(
            evq_log=evq_log,
            geo_log=geo_log,
            output=tmp_path / "bad-smoke.json",
        )


def test_retrieval_source_deletion_pair_is_exactly_length_matched() -> None:
    class FakeTokenizer:
        @staticmethod
        def encode(text: str, add_special_tokens: bool = False) -> list[int]:
            assert not add_special_tokens
            return [100 + (ord(char) % 50) for char in text]

    sourced, deleted, metadata = prepare_retrieval_data.build_example(
        tokenizer=FakeTokenizer(),
        filler=np.arange(30_000, dtype=np.uint32) % 10_000,
        length=4_096,
        source_fraction=0.5,
        distractor_count=8,
        gold=(" blue", 7),
        distractors=[
            (" red", 8),
            (" green", 9),
            (" black", 10),
            (" white", 11),
            (" orange", 12),
            (" purple", 13),
            (" yellow", 14),
            (" nine", 15),
        ],
        rng=random.Random(17),
    )
    assert sourced.shape == deleted.shape == (4_096,)
    start = metadata["source_start"]
    end = start + metadata["source_token_length"]
    assert np.array_equal(
        np.delete(sourced, np.s_[start:end]),
        np.delete(deleted, np.s_[start:end]),
    )
    assert metadata["distractor_token_ids"] == list(range(8, 16))


def test_runtime_selection_requires_sustained_100_plus_300_probe(
    tmp_path,
) -> None:
    backend = {
        "status": "GPU_RUNTIME_SELECTED",
        "selected_loss_backend": "liger",
        "runtime": _probe_payload("liger", 25_000.0),
    }
    backend_path = tmp_path / "backend.json"
    backend_path.write_text(json.dumps(backend), encoding="utf-8")
    mb8 = _probe_payload("liger", 31_000.0)
    mb8["microbatch_sequences"] = 8
    mb8["gradient_accumulation"] = 64
    mb8_path = tmp_path / "mb8.json"
    mb8_path.write_text(json.dumps(mb8), encoding="utf-8")
    candidate_path = tmp_path / "candidate.json"
    select_runtime.choose(
        SimpleNamespace(
            backend_receipt=backend_path,
            microbatch8_probe=mb8_path,
            output=candidate_path,
        )
    )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert candidate["selected_microbatch_sequences"] == 8

    sustained = dict(mb8)
    sustained["warmup_steps"] = 100
    sustained["measured_steps"] = 300
    sustained_path = tmp_path / "sustained.json"
    sustained_path.write_text(json.dumps(sustained), encoding="utf-8")
    final_path = tmp_path / "runtime.json"
    select_runtime.finalize(
        SimpleNamespace(
            candidate_receipt=candidate_path,
            sustained_probe=sustained_path,
            output=final_path,
        )
    )
    final = json.loads(final_path.read_text(encoding="utf-8"))
    assert final["status"] == "GPU_RUNTIME_SELECTED"
    assert final["tokens_per_second"] == 31_000.0


def test_compiled_cuda_arches_uses_compile_time_flags(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        torch._C,
        "_cuda_getArchFlags",
        lambda: "sm_80 sm_90 sm_120",
        raising=False,
    )
    assert preflight.compiled_cuda_arches() == [
        "sm_80",
        "sm_90",
        "sm_120",
    ]


def test_portable_model_evidence_is_bound_to_current_artifacts(
    tmp_path,
) -> None:
    checks = {
        "code": {
            "status": "PASS",
            "evidence": {"code_sha256": "code"},
        },
        "assets": {
            "status": "PASS",
            "evidence": {"asset_manifest_sha256": "assets"},
        },
        "dataset": {
            "status": "PASS",
            "evidence": {
                "manifest_sha256": "data",
                "stream_sha256": "stream",
                "first_global_batch_sha256": "batch",
            },
        },
        "evaluation_dataset": {
            "status": "PASS",
            "evidence": {"manifest_sha256": "eval"},
        },
        "retrieval_dataset": {
            "status": "PASS",
            "evidence": {"manifest_sha256": "retrieval"},
        },
        "frequency_formula": {
            "status": "PASS",
            "evidence": {"tau": 2.0},
        },
        "model_intervention": {
            "status": "PASS",
            "evidence": {
                "parameter_sha256_before": "parameters",
                "parameter_sha256_after": "parameters",
            },
        },
    }
    portable = {
        "status": "PORTABLE_PREFLIGHT_PASS",
        "checks": checks,
    }
    path = tmp_path / "portable.json"
    path.write_text(json.dumps(portable), encoding="utf-8")
    reused = preflight.reuse_portable_model_intervention(path, checks)
    assert reused["evidence_mode"] == "REUSED_PORTABLE_HASH_BOUND"

    drifted = json.loads(json.dumps(checks))
    drifted["dataset"]["evidence"]["stream_sha256"] = "different"
    with pytest.raises(RuntimeError, match="does not match"):
        preflight.reuse_portable_model_intervention(path, drifted)


def test_stream_validate_only_does_not_reseal_corruption(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(stream, "SEQUENCE_LENGTH", 8)
    monkeypatch.setattr(stream, "CHUNK_BYTES", 8 * 4)
    monkeypatch.setattr(stream, "GLOBAL_BATCH_SEQUENCES", 100)
    monkeypatch.setattr(stream, "EXPECTED_CONFIGURED_PATHS", 1)
    monkeypatch.setattr(stream, "EXPECTED_UNIQUE_URLS", 1)
    urls = ["https://example.test/data.npy"]
    monkeypatch.setattr(
        stream,
        "load_official_paths",
        lambda _: urls,
    )
    sizes = {urls[0]: stream.CHUNK_BYTES * 100}
    (tmp_path / "source_sizes.json").write_text(
        json.dumps(sizes), encoding="utf-8"
    )
    rows = stream.source_rows(urls, sizes)
    source_path = tmp_path / "source_manifest.json"
    source_path.write_text(json.dumps(rows), encoding="utf-8")
    prefix = np.arange(100, dtype=np.uint32)
    source_ids, local_indices = stream.map_indices(prefix, rows)
    order_path = tmp_path / "order_indices.uint32.npy"
    source_ids_path = tmp_path / "order_source_ids.uint16.npy"
    local_indices_path = tmp_path / "order_local_indices.uint64.npy"
    np.save(order_path, prefix, allow_pickle=False)
    np.save(source_ids_path, source_ids, allow_pickle=False)
    np.save(local_indices_path, local_indices, allow_pickle=False)
    token_path = tmp_path / "train_tokens.uint32.bin"
    np.arange(800, dtype=np.uint32).tofile(token_path)
    valid_path = tmp_path / "instance_valid.uint8.bin"
    np.ones(100, dtype=np.uint8).tofile(valid_path)
    np.ones(100, dtype=np.uint8).tofile(
        tmp_path / "download_complete.uint8.bin"
    )
    spotcheck_path = tmp_path / "decode_spotcheck.jsonl"
    spotcheck_path.write_text(
        "".join(
            json.dumps({"instance_index": index}) + "\n"
            for index in range(100)
        ),
        encoding="utf-8",
    )
    mapping_sha = contract.sha256_json(
        {
            "source_ids": contract.sha256_file(source_ids_path),
            "local_indices": contract.sha256_file(local_indices_path),
        }
    )
    source_spotchecks = [
        {
            "output_instance": index,
            "global_instance": index,
            "source_id": 0,
            "local_instance": index,
            "payload_sha256": f"{index:064x}",
            "instance_valid": True,
        }
        for index in range(100)
    ]
    proof = {
        "status": "ORDER_AND_SOURCE_PROOF_PASS",
        "official_config_sha256": contract.OFFICIAL_CONFIG_SHA256,
        "seed": contract.SEED,
        "total_instances": 100,
        "needed_instances": 100,
        "order_file_sha256": contract.sha256_file(order_path),
        "order_values_sha256": stream.array_values_sha256(prefix),
        "source_manifest_sha256": contract.sha256_file(source_path),
        "training_stream_sha256": contract.sha256_file(token_path),
        "instance_valid_sha256": contract.sha256_file(valid_path),
        "source_spotcheck_count": 100,
        "source_spotchecks": source_spotchecks,
        "source_spotchecks_sha256": contract.sha256_json(
            source_spotchecks
        ),
    }
    proof_path = tmp_path / "order_stream_proof.json"
    proof_path.write_text(json.dumps(proof), encoding="utf-8")
    manifest = {
        "status": "STREAM_VERIFIED",
        "contract": {
            "tokenizer": "allenai_dolma2",
            "dtype": "uint32",
            "sequence_length": 8,
            "seed": contract.SEED,
            "global_batch_sequences": 100,
            "steps": 1,
            "expected_tokens": 800,
            "official_config_sha256": contract.OFFICIAL_CONFIG_SHA256,
        },
        "source_manifest": {
            "path": source_path.name,
            "sha256": contract.sha256_file(source_path),
            "configured_paths": 1,
            "unique_urls": 1,
            "total_instances": 100,
        },
        "order": {
            "indices_path": order_path.name,
            "indices_sha256": contract.sha256_file(order_path),
            "source_ids_path": source_ids_path.name,
            "source_ids_sha256": contract.sha256_file(source_ids_path),
            "local_indices_path": local_indices_path.name,
            "local_indices_sha256": contract.sha256_file(local_indices_path),
            "mapping_sha256": mapping_sha,
        },
        "training_stream": {
            "path": token_path.name,
            "bytes": token_path.stat().st_size,
            "instances": 100,
            "tokens": 800,
            "sha256": contract.sha256_file(token_path),
            "first_global_batch_sha256": stream.first_global_batch_sha256(
                token_path
            ),
            "instance_valid_path": valid_path.name,
            "instance_valid_sha256": contract.sha256_file(valid_path),
            "invalid_instances": 0,
        },
        "decode_spotcheck": {
            "path": spotcheck_path.name,
            "samples": 100,
            "sha256": contract.sha256_file(spotcheck_path),
        },
        "order_stream_proof": {
            "path": proof_path.name,
            "sha256": contract.sha256_file(proof_path),
            "status": "ORDER_AND_SOURCE_PROOF_PASS",
        },
    }
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = manifest_path.read_bytes()
    stream.validate_existing(
        tmp_path,
        tmp_path / "official.yaml",
        tokenizer_path=None,
        steps=1,
    )
    assert manifest_path.read_bytes() == before

    with token_path.open("r+b") as handle:
        handle.seek(17)
        original = handle.read(1)
        handle.seek(17)
        handle.write(bytes([original[0] ^ 1]))
    with pytest.raises(RuntimeError, match="training stream"):
        stream.validate_existing(
            tmp_path,
            tmp_path / "official.yaml",
            tokenizer_path=None,
            steps=1,
        )
    assert manifest_path.read_bytes() == before


def test_preflight_rejects_empty_retrieval_matrix(tmp_path) -> None:
    eval_manifest = tmp_path / "eval_manifest.json"
    eval_manifest.write_text("{}", encoding="utf-8")
    retrieval_manifest = tmp_path / "retrieval_manifest.json"
    retrieval_manifest.write_text(
        json.dumps(
            {
                "status": "RETRIEVAL_DATA_VERIFIED",
                "lengths": [4_096, 8_192, 16_384],
                "source_fractions": [0.1, 0.5, 0.9],
                "distractor_counts": [0, 8],
                "examples_per_cell": 4,
                "natural_eval_manifest_sha256": contract.sha256_file(
                    eval_manifest
                ),
                "arrays": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="matrix"):
        preflight.validate_retrieval_dataset(
            retrieval_manifest,
            eval_manifest_path=eval_manifest,
        )


def test_resume_archives_interrupted_tail_and_restores_prefix(
    tmp_path,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    rows = [{"step": step, "value": step} for step in range(1, 8)]
    log_path = output / "train.jsonl"
    log_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    prefix_text = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows[:5]
    )
    (output / "completed.json").write_text(
        json.dumps(
            {
                "status": "TRAINING_COMPLETE",
                "stop_step": 5,
                "log_sha256": hashlib.sha256(
                    prefix_text.encode("utf-8")
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    resume = output / "step-000005-full"
    resume.mkdir()
    train.validate_output_for_phase(
        output,
        start_step=5,
        resume_from=resume,
    )
    restored = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step"] for row in restored] == [1, 2, 3, 4, 5]
    archives = list(output.glob("aborted_tail_after_000005_*.jsonl"))
    assert len(archives) == 1


def test_resume_rejects_completed_run_before_mutating_log(tmp_path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    rows = [{"step": step} for step in range(1, 7)]
    log_path = output / "train.jsonl"
    text = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows
    )
    log_path.write_text(text, encoding="utf-8")
    (output / "completed.json").write_text(
        json.dumps(
            {
                "status": "TRAINING_COMPLETE",
                "stop_step": 6,
                "log_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="completion receipt"):
        train.validate_output_for_phase(
            output,
            start_step=5,
            resume_from=output / "step-000005-full",
        )
    assert log_path.read_text(encoding="utf-8") == text


def test_terminal_checkpoint_recovery_finishes_without_replay(
    tmp_path,
) -> None:
    output = tmp_path / "evq"
    incomplete = output / ".step-001000-full.incomplete"
    incomplete.mkdir(parents=True)
    data_manifest = tmp_path / "dataset.json"
    data_manifest.write_text("{}", encoding="utf-8")
    artifact_hashes = {}
    for name, content in (
        ("model.safetensors", b"model"),
        ("optimizer.pt", b"optimizer"),
        ("rng.pt", b"rng"),
    ):
        path = incomplete / name
        path.write_bytes(content)
        artifact_hashes[name] = contract.sha256_file(path)
    frequency = contract.assert_frequency_contract()
    frequency["active_schedule"] = "evq"
    frequency["active_sha256_float32"] = frequency[
        "evq_sha256_float32"
    ]
    (incomplete / "trainer_state.json").write_text(
        json.dumps(
            {
                "step": contract.FIRST_GATE_STEPS,
                "tokens_seen": contract.FIRST_GATE_TOKENS,
                "data_manifest_sha256": contract.sha256_file(data_manifest),
                "model_sha256": artifact_hashes["model.safetensors"],
                "optimizer_sha256": artifact_hashes["optimizer.pt"],
                "rng_sha256": artifact_hashes["rng.pt"],
                "frequency_receipt": frequency,
                "run_config": {"schedule": "evq"},
            }
        ),
        encoding="utf-8",
    )
    (output / "train.jsonl").write_text(
        "".join(
            json.dumps({"step": step}, sort_keys=True) + "\n"
            for step in range(1, 1001)
        ),
        encoding="utf-8",
    )
    assert train.recover_terminal_checkpoint(
        output,
        start_step=500,
        stop_step=contract.FIRST_GATE_STEPS,
        resume_from=output / "step-000500-full",
        data_manifest=data_manifest,
        schedule="evq",
        frequency_receipt=frequency,
    )
    assert (output / "step-001000-full").is_dir()
    completed = json.loads(
        (output / "completed.json").read_text(encoding="utf-8")
    )
    assert completed["recovered_after_checkpoint_write"] is True


def test_trained_checkpoint_must_be_bound_to_step1000_receipts(
    tmp_path,
) -> None:
    output = tmp_path / "evq"
    checkpoint = output / "step-001000-full"
    checkpoint.mkdir(parents=True)
    model_path = checkpoint / "model.safetensors"
    model_path.write_bytes(b"model-state")
    optimizer_path = checkpoint / "optimizer.pt"
    optimizer_path.write_bytes(b"optimizer-state")
    rng_path = checkpoint / "rng.pt"
    rng_path.write_bytes(b"rng-state")
    data_manifest = tmp_path / "dataset_manifest.json"
    data_manifest.write_text("{}", encoding="utf-8")
    rows = [{"step": step} for step in range(1, 1001)]
    log_path = output / "train.jsonl"
    log_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    completion = {
        "status": "TRAINING_COMPLETE",
        "start_step": 500,
        "stop_step": contract.FIRST_GATE_STEPS,
        "tokens_seen": contract.FIRST_GATE_TOKENS,
        "log_sha256": contract.sha256_file(log_path),
    }
    for name in (
        "completed.json",
        "phase_000500_001000_completed.json",
    ):
        (output / name).write_text(
            json.dumps(completion), encoding="utf-8"
        )
    frequency = contract.assert_frequency_contract()
    frequency["active_schedule"] = "evq"
    frequency["active_sha256_float32"] = frequency[
        "evq_sha256_float32"
    ]
    state = {
        "step": contract.FIRST_GATE_STEPS,
        "tokens_seen": contract.FIRST_GATE_TOKENS,
        "data_manifest_sha256": contract.sha256_file(data_manifest),
        "model_sha256": contract.sha256_file(model_path),
        "optimizer_sha256": contract.sha256_file(optimizer_path),
        "rng_sha256": contract.sha256_file(rng_path),
        "frequency_receipt": frequency,
        "run_config": {
            "schedule": "evq",
            "start_step": 500,
            "stop_step": contract.FIRST_GATE_STEPS,
            "sequence_length": 4_096,
            "global_batch_sequences": 512,
            "precision": "amp_bf16",
        },
    }
    state_path = checkpoint / "trainer_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    receipt = evaluate.validate_trained_checkpoint(
        checkpoint,
        data_manifest=data_manifest,
    )
    assert receipt["model_sha256"] == contract.sha256_file(model_path)

    model_path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="model.safetensors"):
        evaluate.validate_trained_checkpoint(
            checkpoint,
            data_manifest=data_manifest,
        )
