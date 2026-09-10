"""Verify actual single-token reader mask, replay equivalence and fixed states."""
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from .balanced_queries import BalancedValueSession
from .target_record_oracle import OracleConfig
from .question_state_cross import append_question_state, tensor_hash
from .question_key_read_mask import deny_question_reads

@torch.inference_mode()
def test_mask():
    torch.set_num_threads(1)
    torch.manual_seed(319)
    cfg = Qwen2Config(vocab_size=101, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
        max_position_embeddings=128,eos_token_id=2,attention_dropout=0.)
    cfg._attn_implementation='sdpa'
    model=Qwen2ForCausalLM(cfg).eval()
    session=BalancedValueSession(model,list(range(1,25)),OracleConfig(
        samples_per_head=8,horizon=8,sink_tokens=1,recent_tokens=2,keep_fraction=.5)).prefill()
    indices=session.keep_indices('P'); b=indices[0].shape[1]
    donor=session.branch('P',indices).consume([60,61])
    before=tensor_hash([t for l in donor.cache.layers for t in (l.keys,l.values)])
    def branch():
        x=session.branch('P',indices)
        append_question_state(x,donor,donor_prefix_slots=b,question_tokens=2)
        return x
    native, replay=branch(),branch()
    native.consume([62])
    native_logits=native.last_logits.clone()
    native_ids=native.generate(4,[])['generated_ids']
    with deny_question_reads(replay,[]) as receipt:
        replay.consume([62])
        torch.testing.assert_close(native_logits,replay.last_logits,rtol=1e-5,atol=1e-7)
        assert replay.generate(4,[])['generated_ids']==native_ids
    masked=branch()
    with deny_question_reads(masked,[b]) as receipt:
        masked.consume([62])
        first=masked.last_logits.clone()
        assert receipt['mask_lengths']==[b+3]
        assert not torch.equal(first, native_logits)
        assert len(masked.generate(4,[])['generated_ids'])==4
    assert receipt['calls']==4 and receipt['denied_physical_slots']==(b,)
    assert before==tensor_hash([t for l in donor.cache.layers for t in (l.keys,l.values)])
    assert not model._forward_pre_hooks

if __name__=='__main__':
    test_mask()
    print('PASS: all-visible logits within 1e-7 absolute/1e-5 relative; tokens exact; actual masked decode; donor intact; hook removed')
