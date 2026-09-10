"""Budget, source-span, sham, native branch isolation, and complete P reuse."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from .balanced_queries import BalancedValueSession
from .run import BASELINE_VERSION, digest
from .target_record_oracle import (OracleConfig, ROW_IDS, VERSION, force_record_sets,
                                   frozen_variants, p_input_key, reusable_p, target_spans)


def test_fixed_budget_lowest_removals_and_matched_sham():
    p = torch.tensor([[0, 1, 4, 5, 14, 15], [0, 1, 2, 3, 14, 15]])
    before = p.clone()
    scores = torch.arange(16, dtype=torch.float32).expand(2, -1)
    oracle, sham, rows = force_record_sets(p, scores, [7, 8], 16, sink_tokens=1, recent_tokens=2)
    assert torch.equal(p, before)
    assert oracle.shape == sham.shape == p.shape
    assert oracle.tolist() == [[0, 5, 7, 8, 14, 15], [0, 3, 7, 8, 14, 15]]
    for i in range(2):
        assert {0, 14, 15} <= set(sham[i].tolist())
        assert {7, 8}.isdisjoint(set(sham[i].tolist()) - set(p[i].tolist()))
        assert set(p[i].tolist()) - set(oracle[i].tolist()) == set(p[i].tolist()) - set(sham[i].tolist())
        assert rows[i]["swaps"] == 2
    with pytest.raises(ValueError, match="cannot fit"):
        force_record_sets(p, scores, list(range(1, 10)), 16, sink_tokens=1, recent_tokens=2)
    with pytest.raises(RuntimeError, match="truncated"):
        force_record_sets(torch.arange(6)[None], torch.arange(8).float()[None], [6, 7], 8,
                          sink_tokens=0, recent_tokens=0)


def test_target_record_boundary_and_fixed_input_scope():
    text = "Record key=abc; value=first.\nRecord key=abc; value=second."
    row = {"row_id": ROW_IDS[0], "split": "dev", "prefix_text": text,
           "prefix_ids": list(range(len(text))), "suffix_ids": [200],
           "prompt_ids": list(range(len(text))) + [200],
           "query": {"keys": ["abc"], "ordinals": [2]},
           "records": [{"key": "abc", "value": "first"}, {"key": "abc", "value": "second"}]}
    def tokenizer(value, **kwargs):
        assert value == text
        return {"input_ids": row["prefix_ids"], "offset_mapping": [(i, i + 1) for i in range(len(text))]}
    spans, selected = target_spans(row, tokenizer)
    assert len(spans) == 1 and spans[0]["value"] == "second"
    assert selected == list(range(text.index("Record", 1), len(text)))
    with pytest.raises(ValueError, match="two preselected"):
        target_spans(dict(row, row_id="unselected"), tokenizer)


@torch.inference_mode()
def test_tiny_native_paths_keep_prefix_and_reuse_complete_P(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(319)
    c = Qwen2Config(vocab_size=101, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, eos_token_id=2, attention_dropout=0.0)
    c._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(c).eval()
    cfg = OracleConfig(samples_per_head=8, horizon=8, sink_tokens=1, recent_tokens=2, keep_fraction=.5)
    prefix = list(range(1, 25))
    run = BalancedValueSession(model, prefix, cfg).prefill()
    old_p = run.branch("P").consume([60, 61]).generate(4, [])
    snapshot = [(l.keys.clone(), l.values.clone()) for l in run.cache.layers]
    # One target location absent from at least one P head/layer; no labels used.
    target = [i for i in range(1, 22) if i not in run.keep_indices("P")[0][0].tolist()][:2]
    variants, receipts = frozen_variants(run, target)
    outputs = {}
    for arm, indices in variants.items():
        branch = run.branch("P", indices)
        assert branch.cache.get_seq_length() == 12
        outputs[arm] = branch.consume([60, 61]).generate(4, [])
        assert outputs[arm]["initial_prefix_kv_bytes"] == old_p["initial_prefix_kv_bytes"]
        assert len(outputs[arm]["generated_ids"]) == 4
    assert outputs["P"]["generated_ids"] == old_p["generated_ids"]
    for old, current in zip(snapshot, run.cache.layers):
        torch.testing.assert_close(old[0], current.keys, rtol=0, atol=0)
        torch.testing.assert_close(old[1], current.values, rtol=0, atol=0)
    assert run.cache.get_seq_length() == len(prefix)
    # Exercise the actual existing candidate reuse loader, not a simulated flag.
    identity = {"path": "/same-model", "config_sha256": "x", "weights": {"w": 1}}
    row = {"row_id": ROW_IDS[0], "prefix_ids": prefix, "suffix_ids": [60, 61],
           "prompt_ids": prefix + [60, 61], "prefix_length": len(prefix), "max_new_tokens": 4,
           "score_contract": "literal_full_string_plus_terminal_eos_v1", "expected": "", "references": [""]}
    version = BASELINE_VERSION + "_" + VERSION
    source = {"model": identity, "backend": "native_HF_Qwen2_SDPA_original_position_cache_v1",
              "dtype": "float32", "config": asdict(cfg), "decode": "raw_greedy_argmax",
              "baseline_version": version,
              "task_protocol": "context first, question unseen by compression; not original LongBench prompt order",
              "sources": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("adapter.py", "ops.py")}}
    keep_sha = digest([x.cpu().tolist() for x in variants["P"]])
    old_record = {"row_id": row["row_id"], "arm": "P", "generated_token_ids": old_p["generated_ids"],
                  "keep_indices_sha256": keep_sha,
                  "baseline_cache_key": p_input_key(identity, row, cfg, "float32", version)}
    (tmp_path / "manifest.json").write_text(json.dumps(source))
    (tmp_path / "per_example.jsonl").write_text(json.dumps(old_record) + "\n")
    reused = reusable_p(tmp_path, identity, row, cfg, "float32", keep_sha)
    assert reused["generated_token_ids"] == outputs["P"]["generated_ids"]
    assert reused["reused_P_control"]
    with pytest.raises(ValueError, match="P set differs"):
        reusable_p(tmp_path, identity, row, cfg, "float32", "changed_keep")
    changed = dict(row, prompt_ids=prefix + [60, 62])
    with pytest.raises(ValueError, match="input/scoring"):
        reusable_p(tmp_path, identity, changed, cfg, "float32", keep_sha)
