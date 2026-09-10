"""Boundary, source isolation, reference mapping and metric integration checks."""
import copy
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
import torch

from experiments.pm_keep.balanced_queries import BalancedConfig, BalancedValueSession
from .prepare import (EXACT, MRCR, QA, attach, excluded_sources, interleave,
                      record_bank, render_messages, render_split, validate)
from .scoring import score, score_nosa
from .report import sampling_comparison


@pytest.fixture(params=["results/pm_keep_20260909/source/tokenizer", "results/nosa_position_20260909/source"])
def tokenizer(request):
    return AutoTokenizer.from_pretrained(request.param, local_files_only=True)


def natural_row(tokenizer, question="Which person moved away?"):
    context = "A natural document with a nonbreaking space\u00a0and Unicode punctuation—Alice left.\n" * 20
    view = render_split(tokenizer, context, question, "Give the answer only.")
    return attach(dict(row_id="x", task="natural_fixture", split="dev", doc_id="document-x",
                       material_cluster_id="document-x", constituent_doc_ids=["document-x"],
                       expected=None, references=["Alice"], score_contract=QA, max_new_tokens=32), view, 32768)


def test_native_boundary_and_source_text_survive_normalization(tokenizer):
    first = natural_row(tokenizer)
    second = natural_row(tokenizer, "Where is the departure described?")
    assert first["prefix_ids"] == second["prefix_ids"]
    assert first["raw_context"] in first["full_prompt"]
    assert first["prefix_ids"] + first["suffix_ids"] == first["prompt_ids"]
    assert validate([first], tokenizer)["rows"] == 1


def test_actual_future_token_in_prefix_is_rejected(tokenizer):
    row = natural_row(tokenizer)
    row["prefix_ids"].append(row["suffix_ids"].pop(0))
    row["prefix_length"] += 1
    with pytest.raises(ValueError, match="actual prefix token offsets"):
        validate([row], tokenizer)


def test_existing_and_cross_split_document_reuse_rejected(tokenizer):
    row = natural_row(tokenizer)
    with pytest.raises(ValueError, match="old DEV/TEST"):
        validate([row], tokenizer, [row])
    other = copy.deepcopy(row)
    other.update(row_id="y", split="test")
    with pytest.raises(ValueError, match="share source material"):
        validate([row, other], tokenizer)


def test_competing_records_preserve_target_and_do_not_reuse_splits():
    a, b = record_bank("dev", 2), record_bank("test", 2)
    assert len(a) == 256 and len(set(a)) == 256
    assert not ({v for pair in a for v in pair} & {v for pair in b for v in pair})
    text = interleave("Natural prose. " * 100, a, 900)
    assert text.count("Record key=") == 256
    assert all(f"Record key={key}; value={value}." in text for key, value in a)


def test_mrcr_boundary_and_metrics_use_complete_output(tokenizer):
    messages = [{"role": "user", "content": "Tell a story."},
                {"role": "assistant", "content": "First complete story."},
                {"role": "user", "content": "Tell a story."},
                {"role": "assistant", "content": "Second complete story."},
                {"role": "user", "content": "Repeat the first story and prepend MARK."}]
    row = dict(messages=messages, references=["MARKFirst complete story."], random_string_to_prepend="MARK",
               occurrence_message_indices=[1, 3], score_contract=MRCR, material_cluster_id="m", family_id="m",
               max_new_tokens=512)
    view = render_messages(tokenizer, row)
    assert view["prefix_ids"] + view["suffix_ids"] == view["prompt_ids"]
    eos = tokenizer.eos_token_id
    ids = tokenizer.encode(row["references"][0], add_special_tokens=False)
    assert score(row, ids + [eos], tokenizer, {eos})["exact_plus_eos"]
    assert not score(row, ids, tokenizer, {eos})["exact_plus_eos"]
    wrong = tokenizer.encode("MARKSecond complete story.", add_special_tokens=False) + [eos]
    assert score(row, wrong, tokenizer, {eos})["wrong_occurrence_whole_string"]
    missing = tokenizer.encode("First complete story.", add_special_tokens=False) + [eos]
    assert score(row, missing, tokenizer, {eos})["official_sequence_ratio"] == 0


def test_balanced_real_tiny_qwen_keeps_query_blind_branch_semantics():
    torch.manual_seed(11)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                        eos_token_id=2)
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).eval()
    cfg = BalancedConfig(samples_per_head=8, horizon=8, sink_tokens=1, recent_tokens=1, keep_fraction=.5)
    prefix = [4, 5, 6, 4, 5, 6, 7, 8, 4, 5, 6, 9]
    session = BalancedValueSession(model, prefix, cfg).prefill()
    before = session.score("P")[0].clone()
    branch = session.branch("P")
    branch.consume([11, 12])
    generated = branch.generate(3, {2})
    assert 1 <= len(generated["generated_ids"]) <= 3
    assert session.cache.layers[0].get_seq_length() == len(prefix)
    torch.testing.assert_close(session.score("P")[0], before)


def test_nosa_short_prefill_is_not_pooled_as_active_routing(tokenizer):
    row = natural_row(tokenizer)
    ids = tokenizer.encode("Alice", add_special_tokens=False) + [tokenizer.eos_token_id]
    result = score_nosa(row, ids, tokenizer, {tokenizer.eos_token_id}, topk=64)
    assert not result["routing_active_during_prefill"]
    assert result["task"].endswith("__short_prefill_control")


def test_sampling_report_clusters_questions_and_rejects_model_confound(tmp_path):
    left, right = tmp_path / "balanced", tmp_path / "uniform"
    config = dict(samples_per_head=256, horizon=128, seed=1, keep_fraction=.25,
                  sink_tokens=4, recent_tokens=256, value_objective="raw_norm")
    manifest = dict(model={"revision": "same"}, dtype="bfloat16", backend="same", config=config)
    for path, values in ((left, [.8, .8, .3, .3]), (right, [.6, .6, .4, .4])):
        path.mkdir()
        (path / "broad_execution.json").write_text(json.dumps(dict(data_sha256="same", scoring_version="same")))
        (path / "manifest.json").write_text(json.dumps(manifest))
        rows = [dict(row_id=f"q{i}", arm="P", task="mrcr_long", material_cluster_id=f"conversation{i//2}",
                     official_sequence_ratio=value) for i, value in enumerate(values)]
        (path / "per_example.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    result = sampling_comparison(left, right)[0]
    assert result["rows"] == 4 and result["independent_units"] == 2
    assert result["paired_difference_pp"] == pytest.approx(5)
    manifest["model"] = {"revision": "different"}
    (right / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="model"):
        sampling_comparison(left, right)
