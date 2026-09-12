"""CPU-only integration fixtures for the Llama RoPE adapter and C tap.

These tests deliberately exercise the contracts that a passing construction
self-test cannot see: HF 4.x/5.x call signatures, GQA head mapping, operator
surfaces, and the single-gain/D11 diagonal semantics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "llama3_60dir_20260911"))

import adapter  # noqa: E402
import c_collect  # noqa: E402
import core  # noqa: E402
import operators  # noqa: E402


def geometry():
    return core.Geometry.from_mapping({
        "hidden_size": 4096, "num_attention_heads": 32,
        "num_key_value_heads": 8, "num_hidden_layers": 32,
        "max_position_embeddings": 8192, "rope_theta": 500000.0,
        "rope_scaling": None,
    })


def test_all_extension_surfaces_match_cpu_reference_with_gqa():
    g = geometry()
    positions = np.array([0.0, 1.0, 8192.0, 32767.0])
    rng = np.random.default_rng(20260911)
    q = rng.normal(size=(1, 32, 4, 128)).astype(np.float32)
    k = rng.normal(size=(1, 8, 4, 128)).astype(np.float32)

    for d in range(7, 21):
        C = adapter.build_from_operator(g, operators.build(f"D{d:02d}a", g))
        qp, kp, _ = adapter.apply_pure(C, q, k, positions, g)
        qt, kt = adapter.apply_torch(C, torch.from_numpy(q), torch.from_numpy(k),
                                     torch.from_numpy(positions), torch)
        # CPU torch uses FP32 trig while the reference keeps FP64 until return.
        assert np.allclose(qt.numpy(), qp.astype(np.float32), atol=4e-3, rtol=4e-3), C.method
        assert np.allclose(kt.numpy(), kp.astype(np.float32), atol=4e-3, rtol=4e-3), C.method


def test_d11_keeps_independent_q_and_k_diagonals_without_double_application():
    g = geometry()
    op = operators.build("D11a", g)
    C = adapter.build_from_operator(g, op)
    assert C.metric is None
    assert np.allclose(C.q_diag, op.q_diag())
    assert np.allclose(C.k_diag, op.k_diag())

    rng = np.random.default_rng(7)
    q = rng.normal(size=(1, 32, 2, 128)).astype(np.float32)
    k = rng.normal(size=(1, 8, 2, 128)).astype(np.float32)
    p = np.array([123.0, 8192.0])
    got_q, got_k, _ = adapter.apply_pure(C, q, k, p, g)
    want_q = operators.apply_rotation(q, op.q_phase(p), op.q_amp(p), op.q_diag())
    want_k = operators.apply_rotation(k, op.k_phase(p), op.k_amp(p), op.k_diag())
    assert np.allclose(got_q, want_q)
    assert np.allclose(got_k, want_k)


def test_gqa_statistics_repeats_each_kv_group_for_four_q_heads():
    rng = np.random.default_rng(3)
    q = rng.normal(size=(32, 6, 128))
    k = rng.normal(size=(8, 6, 128))
    got = c_collect.per_head_statistics(q, k, 5, (1, 3), (3, 5))
    want = c_collect.per_head_statistics(q, np.repeat(k, 4, axis=0), 5, (1, 3), (3, 5))
    for key in ("c_evidence", "c_distractor", "delta_c"):
        assert np.array_equal(got[key], want[key])


def test_rotary_adapter_stashes_positions_and_accepts_hf_515_shape():
    g = geometry()
    C = adapter.build_from_operator(g, operators.build("D10a", g))

    class Rotary:
        def forward(self, x, position_ids):
            return x, x

    class Modeling:
        LlamaRotaryEmbedding = Rotary

        @staticmethod
        def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
            return q + 2, k + 2

    modeling = Modeling()
    q = torch.zeros((1, 32, 2, 128))
    k = torch.zeros((1, 8, 2, 128))
    with adapter.RoPEAdapter(torch, modeling, C, g, keep_stock_for={"D10a"}):
        modeling.LlamaRotaryEmbedding.forward(None, None, torch.tensor([[0, 1]]))
        out_q, out_k = modeling.apply_rotary_pos_emb(q, k, None, None)
    assert torch.equal(out_q, q + 2)
    assert torch.equal(out_k, k + 2)


class _ToyTokenizer:
    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        return {"input_ids": [0] + [ord(c) for c in text] if add_special_tokens else [ord(c) for c in text]}


def test_c_tokenizer_rejects_truncation_and_missing_markers():
    tok = _ToyTokenizer()
    with pytest.raises(ValueError, match="exceeds native C limit"):
        c_collect.tokenize_with_spans(tok, "abcdef", "a", "b", max_tokens=3)
    with pytest.raises(ValueError, match="marker not found"):
        c_collect.tokenize_with_spans(tok, "abc", "missing", "b")


def test_partial_c_collection_is_rejected_before_model_work():
    # The guard is intentionally before any model invocation; a smoke limit
    # must not leave behind an artifact that looks like the one-shot C file.
    with pytest.raises(ValueError, match="partial C collection refused"):
        c_collect.collect("/no/model", "/tmp/should-not-be-written.npz", limit=1)


def test_c_coverage_manifest_counts_auxiliary_blocks_and_blocks_unknown_specs(tmp_path):
    coverage = c_collect.collection_coverage()
    assert coverage["blocks"]["relations"]["records"] == 192 * 4
    assert coverage["blocks"]["natural"]["groups"] == 64
    assert coverage["blocks"]["carrier"]["groups"] == 32
    assert coverage["blocks"]["carrier"]["checkpoint_records_expected"] == 64
    assert coverage["blocks"]["gain"]["records_expected"] == 192
    assert coverage["status"] == "BLOCKED"

    out = tmp_path / "c_stats.npz"
    with pytest.raises(RuntimeError, match="C collection BLOCKED"):
        c_collect.collect("/no/model", str(out))
    manifest = out.with_suffix(".coverage.json")
    assert manifest.exists()
    assert not out.exists()
    assert __import__("json").loads(manifest.read_text())["status"] == "BLOCKED"
