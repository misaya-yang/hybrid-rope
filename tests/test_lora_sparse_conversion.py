from types import SimpleNamespace

import torch

from experiments.lora_evq_v2.eval_sparse_conversion import (
    ATTENTION_IMPL,
    _apply_rotary,
    _counterfactual_prompts,
    _phase0_gate_checks,
    _probe_metrics,
    _register_attention,
    _set_attention_mode,
    _sparse_keep_mask,
    _tokenizer_identity_matches,
    _validate_config,
    causal_backbone,
    exact_block_attention_forward,
)
from experiments.lora_evq_v2.eval_ruler import task_kv_retrieval


def test_counterfactual_prompts_are_equal_length_and_remove_the_source():
    original_answer = [91, 92]
    swapped_answer = [81, 82]
    prompt = list(range(30))
    prompt[12:14] = original_answer
    prompts = _counterfactual_prompts(
        prompt,
        needle_span=(10, 16),
        answer_span=(12, 14),
        original_answer_ids=original_answer,
        swapped_answer_ids=swapped_answer,
    )

    assert {len(value) for value in prompts.values()} == {len(prompt)}
    assert prompts["swapped"][12:14] == swapped_answer
    assert original_answer not in [
        prompts["source_removed"][i : i + 2]
        for i in range(len(prompt) - 1)
    ]
    assert swapped_answer not in [
        prompts["source_removed"][i : i + 2]
        for i in range(len(prompt) - 1)
    ]


def test_rotary_and_gqa_full_budget_parity():
    torch.manual_seed(42)
    value = torch.randn(1, 4, 3, 8)
    angles = torch.randn(1, 3, 8)
    cos, sin = angles.cos(), angles.sin()
    first, second = value.chunk(2, dim=-1)
    expected = value * cos.unsqueeze(1) + torch.cat((-second, first), dim=-1) * sin.unsqueeze(1)
    assert torch.allclose(_apply_rotary(value, cos, sin), expected)

    query = torch.randn(1, 4, 1, 8)
    key = torch.randn(1, 2, 16, 8)
    values = torch.randn(1, 2, 16, 8)
    module = SimpleNamespace(training=False)
    module._evq_sparse_config = {
        "mode": "dense",
        "block_size": 4,
        "top_blocks": 1,
        "local_window": 4,
        "sink_tokens": 1,
    }
    dense, dense_weights = exact_block_attention_forward(
        module, query, key, values, None, 8**-0.5
    )
    module._evq_sparse_config["mode"] = "full"
    full, full_weights = exact_block_attention_forward(
        module, query, key, values, None, 8**-0.5
    )
    assert dense.shape == (1, 1, 4, 8)
    assert torch.equal(dense, full)
    assert torch.equal(dense_weights, full_weights)


def test_score_and_fixed_masks_share_budget_and_probe_gqa_heads():
    scores = torch.zeros(1, 4, 1, 16)
    scores[..., 8:12] = 10
    config = {
        "block_size": 4,
        "top_blocks": 1,
        "local_window": 4,
        "sink_tokens": 1,
    }
    score_mask = _sparse_keep_mask(scores, mode="score", config=config)
    fixed_mask = _sparse_keep_mask(scores, mode="fixed", config=config)
    assert score_mask.shape == fixed_mask.shape == scores.shape
    assert score_mask.sum(dim=-1).tolist() == fixed_mask.sum(dim=-1).tolist()
    assert score_mask[..., 8:12].all()

    metrics = _probe_metrics(
        scores[0, :, 0], needle_start=8, needle_end=10, config=config
    )
    assert len(metrics["block_rank"]) == 4
    assert metrics["block_rank"] == [1, 1, 1, 1]
    assert metrics["hit_at"]["16"] == [True, True, True, True]


def test_exploratory_16k_gate_requires_evq_topk_separation():
    checks = _phase0_gate_checks(
        {
            "16384": {
                "evq_wins": 10,
                "median_hit_delta_evq_minus_geo": 0.48,
                "median_evq_hit_at_16": 0.64,
                "median_geo_hit_at_16": 0.19,
            }
        },
        (16384,),
    )
    assert all(checks.values())


def test_registered_decode_attention_matches_full_budget_on_tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(42)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    ).eval()
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]])
    _register_attention()

    def first_logits(mode):
        _set_attention_mode(model, "sdpa")
        prefill = causal_backbone(model)(prompt[:, :-1], use_cache=True, return_dict=True)
        _set_attention_mode(
            model,
            ATTENTION_IMPL,
            mode=mode,
            config={
                "block_size": 4,
                "top_blocks": 1,
                "local_window": 2,
                "sink_tokens": 1,
            },
        )
        return model(
            prompt[:, -1:],
            attention_mask=torch.ones(1, prompt.shape[1], dtype=torch.long),
            past_key_values=prefill.past_key_values,
            use_cache=True,
            return_dict=True,
        ).logits

    assert torch.equal(first_logits("dense"), first_logits("full"))


def test_transformers5_default_rope_config_is_still_raw(monkeypatch):
    import transformers

    config = SimpleNamespace(
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=8192,
        rope_theta=None,
        rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},
        rope_scaling={"rope_theta": 500000.0, "rope_type": "default"},
    )
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    assert _validate_config("unused") == {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 8192,
        "rope_theta": 500000.0,
    }


def test_v1_and_v2_tokenizer_identity_fields_are_equivalent():
    files = {"tokenizer.json": "abc"}
    expected = {"identifier": "Meta-Llama-3-8B-Instruct", "files": files}
    assert _tokenizer_identity_matches(
        {"requested": "Meta-Llama-3-8B-Instruct", "files": files}, expected
    )
    assert _tokenizer_identity_matches(
        {"identifier": "Meta-Llama-3-8B-Instruct", "files": files}, expected
    )


def test_kv_retrieval_rejects_underfilled_context(monkeypatch):
    monkeypatch.setattr(
        "experiments.lora_evq_v2.eval_ruler.build_prompt",
        lambda *args, **kwargs: torch.zeros((1, 512), dtype=torch.long),
    )
    try:
        task_kv_retrieval(None, None, 8192, n_trials=1)
    except RuntimeError as exc:
        assert "not a real 8192-token case" in str(exc)
    else:
        raise AssertionError("underfilled KV context was accepted")
