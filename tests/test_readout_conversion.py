from collections import Counter
import json
from pathlib import Path

import pytest
import torch

from experiments.lora_evq_v2.eval_official_yarn_capability import sha256_file
from experiments.lora_evq_v2.eval_sparse_conversion import (
    ASSOCIATION_SWAP_MANIFEST_SCHEMA,
    ASSOCIATION_SWAP_TRACE_SCHEMA,
    READOUT_TRACE_MANIFEST_SCHEMA,
    READOUT_TRACE_SCHEMA,
    _association_answer_pairs,
    build_association_swap_pair,
    causal_delta_rank,
    per_layer_logit_lens_delta,
)
from scripts.analysis.readout_conversion import (
    SUMMARY_SCHEMA,
    analyze,
    scalar_alpha_interval,
)


class _LiteralTokenizer:
    def __call__(self, text, *, add_special_tokens=False, return_attention_mask=False):
        assert add_special_tokens is False
        assert return_attention_mask is False
        return {"input_ids": [1_000 + ord(character) for character in text]}


def test_causal_delta_rank_broadcasts_gold_positions_and_uses_strict_rank():
    full = torch.tensor(
        [
            [[2.0, 2.0, 1.0, 0.0], [0.0, 3.0, 2.0, 1.0]],
            [[4.0, 3.0, 2.0, 1.0], [0.0, 2.0, 2.0, 5.0]],
        ]
    )
    ranks = causal_delta_rank(full, torch.zeros_like(full), [0, 2])
    assert ranks.tolist() == [[1, 4], [3, 2]]


def test_association_swap_builder_matches_multiset_position_and_frequency_controls():
    pair = build_association_swap_pair(
        _LiteralTokenizer(),
        key_a_ids=[11, 12],
        key_b_ids=[13, 14],
        answer_a_ids=[21, 22],
        answer_b_ids=[23, 24],
        filler_ids=[31, 32, 33],
        first_token_frequency_buckets={21: 7, 23: 7},
        target_length=1_024,
        depth_percent=50.0,
        block_size=128,
        slot_gap_blocks=1,
    )

    prompt_a = pair["prompts"]["query_a"]
    prompt_b = pair["prompts"]["query_b"]
    assert len(prompt_a) == len(prompt_b) == 1_024
    assert Counter(prompt_a) == Counter(prompt_b)
    assert pair["slot_blocks"][0] != pair["slot_blocks"][1]
    assert pair["gold_spans"]["query_a"] == pair["answer_spans"]["a"]
    assert pair["gold_spans"]["query_b"] == pair["answer_spans"]["b"]
    assert pair["first_token_frequency_bucket"] == 7

    mirrored = build_association_swap_pair(
        _LiteralTokenizer(),
        key_a_ids=[11, 12],
        key_b_ids=[13, 14],
        answer_a_ids=[21, 22],
        answer_b_ids=[23, 24],
        filler_ids=[31, 32, 33],
        first_token_frequency_buckets={21: 7, 23: 7},
        target_length=1_024,
        depth_percent=50.0,
        block_size=128,
        slot_gap_blocks=1,
        mirror=True,
    )
    assert mirrored["answer_spans"]["a"] == pair["answer_spans"]["b"]
    assert mirrored["answer_spans"]["b"] == pair["answer_spans"]["a"]


def test_association_answer_pairs_are_seeded_and_bucket_matched():
    counts = torch.tensor([0, 1, 1, 2, 2, 3, 3, 7, 7, 15, 15])
    first = _association_answer_pairs(counts, excluded_ids={1}, seed=42)
    second = _association_answer_pairs(counts, excluded_ids={1}, seed=42)
    assert first == second
    assert first
    for token_a, token_b, bucket in first:
        assert token_a != token_b
        assert int(torch.floor(torch.log2(counts[token_a].float() + 1))) == bucket
        assert int(torch.floor(torch.log2(counts[token_b].float() + 1))) == bucket


def test_scalar_alpha_interval_detects_feasible_and_impossible_contrasts():
    assert scalar_alpha_interval(
        torch.tensor([0.0, 2.0, 1.0]),
        torch.tensor([2.0, 0.0, 1.0]),
        0,
    ) == (1.0, float("inf"))
    assert (
        scalar_alpha_interval(
            torch.tensor([0.0, 2.0]),
            torch.tensor([0.0, 1.0]),
            0,
        )
        is None
    )


def test_per_layer_logit_lens_delta_applies_norm_before_unembedding():
    torch.manual_seed(42)
    norm = torch.nn.LayerNorm(3)
    head = torch.nn.Linear(3, 5, bias=False)
    full = torch.randn(2, 3)
    ablated = torch.randn(2, 3)

    observed = per_layer_logit_lens_delta(full, ablated, norm=norm, lm_head=head)
    expected = head(norm(full)).float() - head(norm(ablated)).float()
    assert torch.allclose(observed, expected)


def _write_record(root: Path, name: str, payload: dict) -> dict:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {"file": name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _write_manifest(root: Path, payload: dict) -> Path:
    path = root / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_analysis_emits_figure_and_sanitized_summary(tmp_path):
    causal_manifests = []
    swap_manifests = []
    for substrate, scale in (("native_geo", 1.0), ("evq_cosh", 1.5)):
        causal_root = tmp_path / f"causal_{substrate}"
        full = scale * torch.tensor(
            [
                [[3.0, 1.0, 0.0, -1.0], [2.0, 1.0, 0.0, -1.0]],
                [[0.0, 1.0, 3.0, -1.0], [0.0, 2.0, 3.0, -1.0]],
            ]
        )
        causal_item = _write_record(
            causal_root,
            "records/case.pt",
            {
                "schema": READOUT_TRACE_SCHEMA,
                "substrate": substrate,
                "prompt_sha256": "a" * 64,
                "target_length": 16_384,
                "depth_percent": 50.0,
                "gold_token_ids": torch.tensor([0, 2]),
                "layer_indices": torch.tensor([0, 1]),
                "full_logits": full,
                "ablated_logits": torch.zeros_like(full),
                "private_passkey_text": "SECRET_PASSKEY",
            },
        )
        causal_manifests.append(
            _write_manifest(
                causal_root,
                {
                    "schema": READOUT_TRACE_MANIFEST_SCHEMA,
                    "status": "complete",
                    "measurement_label": "oracle-diagnostic",
                    "substrate": substrate,
                    "records": [causal_item],
                },
            )
        )

        swap_root = tmp_path / f"swap_{substrate}"
        swap_full = scale * torch.tensor(
            [
                [[3.0, 1.0, 0.0, 0.0], [4.0, 1.0, 0.0, 0.0]],
                [[1.0, 3.0, 0.0, 0.0], [1.0, 4.0, 0.0, 0.0]],
            ]
        )
        swap_item = _write_record(
            swap_root,
            "records/pair.pt",
            {
                "schema": ASSOCIATION_SWAP_TRACE_SCHEMA,
                "substrate": substrate,
                "pair_sha256": "b" * 64,
                "split": "test",
                "depth_percent": 50.0,
                "candidate_first_token_ids": torch.tensor([0, 1]),
                "layer_indices": torch.tensor([0, 1]),
                "full_logits": swap_full,
                "ablated_logits": torch.zeros_like(swap_full),
                "private_passkey_text": "SECRET_PASSKEY",
            },
        )
        swap_manifests.append(
            _write_manifest(
                swap_root,
                {
                    "schema": ASSOCIATION_SWAP_MANIFEST_SCHEMA,
                    "status": "complete",
                    "measurement_label": "oracle-diagnostic",
                    "substrate": substrate,
                    "records": [swap_item],
                },
            )
        )

    with pytest.raises(ValueError, match="ten registered causal cases"):
        analyze(
            causal_manifests,
            swap_manifests,
            tmp_path / "blocked_analysis",
            bootstrap_samples=4,
        )

    output = tmp_path / "analysis"
    summary = analyze(
        causal_manifests,
        swap_manifests,
        output,
        bootstrap_samples=64,
        enforce_planned_counts=False,
    )

    assert summary["schema"] == SUMMARY_SCHEMA
    assert summary["matched_case_count"] == 1
    assert summary["matched_swap_pair_count"] == 1
    assert summary["association_swap_final_layer"]
    assert all(
        row["feasible_fraction"] == 1.0
        for row in summary["oracle_scalar_feasibility"]
    )
    assert (output / "linchpin_causal_delta_rank.png").stat().st_size > 0
    text = (output / "summary.json").read_text(encoding="utf-8")
    assert "SECRET_PASSKEY" not in text
    assert "gold_token_ids" not in text
    assert "candidate_first_token_ids" not in text
    assert "prompt_sha256" not in text
