"""Only the new splice interface: native tiny-model self-replay."""
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from .balanced_queries import BalancedValueSession
from .target_record_oracle import OracleConfig
from .question_state_cross import append_question_state


@torch.inference_mode()
def test_native_self_replay():
    torch.set_num_threads(1)
    torch.manual_seed(319)
    config = Qwen2Config(vocab_size=101, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, eos_token_id=2, attention_dropout=0.)
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).eval()
    cfg = OracleConfig(samples_per_head=8, horizon=8, sink_tokens=1, recent_tokens=2, keep_fraction=.5)
    session = BalancedValueSession(model, list(range(1, 25)), cfg).prefill()
    prefix = [(x.keys.clone(), x.values.clone()) for x in session.cache.layers]
    for arm in ("F", "P"):
        indices = session.keep_indices(arm)
        donor = session.branch(arm, indices).consume([60, 61])
        branch = session.branch(arm, indices)
        b = indices[0].shape[1]
        receipt = append_question_state(branch, donor, donor_prefix_slots=b, question_tokens=2)
        assert branch.last_logits is None and branch.logical_position == 26
        assert receipt["physical_length_before_final"] == b + 2
        branch.consume([62]); donor.consume([62])
        torch.testing.assert_close(branch.last_logits, donor.last_logits, rtol=0, atol=0)
        assert branch.generate(4, [])['generated_ids'] == donor.generate(4, [])['generated_ids']
    for original, layer in zip(prefix, session.cache.layers):
        assert torch.equal(original[0], layer.keys) and torch.equal(original[1], layer.values)


if __name__ == "__main__":
    test_native_self_replay()
    print('PASS: Full and compressed self-replay logits/tokens exact; original prefix intact')
