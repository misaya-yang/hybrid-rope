"""CPU integration: teacher is a separate original model, not the adapted student.

These tiny eager-attention checks do not qualify physical-64K GPU memory or speed.
"""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM

from scripts.experiments.cross_audit.training import causal_loss
from scripts.experiments.scale_transport.carrier_ruler_run import (
    attach_decision_trace, digest, install_signed, load_frozen_adapter, save_decision_trace,
)
from scripts.experiments.scale_transport.long_lora import MODULES, native_teacher_kl, original_teacher


def setup_pair(rank=4):
    torch.manual_seed(20260907)
    config = Qwen2Config(vocab_size=32, hidden_size=128, intermediate_size=256,
        num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
        max_position_embeddings=32768, rope_theta=1e6,
        attention_dropout=0., bos_token_id=1, eos_token_id=2, pad_token_id=0)
    config._attn_implementation = 'eager'
    original = Qwen2ForCausalLM(config).eval()
    student = copy.deepcopy(original)
    native = original.model.rotary_emb.inv_freq.numpy().copy()
    values = native*.5
    values[-1] = 0
    table = {'values_float32': values.tolist(), 'gain': 1.125}
    install_signed(student, table['values_float32'], table['gain'])
    wrapped = get_peft_model(student, LoraConfig(r=rank, lora_alpha=rank, target_modules=list(MODULES),
        lora_dropout=0., bias='none', task_type='CAUSAL_LM'))
    student = wrapped.get_base_model()
    # A nonzero adapter makes failure to disable it observable.
    with torch.no_grad():
        for name, parameter in student.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(0, .03)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    student.enable_input_require_grads()
    generation = GenerationConfig(do_sample=False, num_beams=1, use_cache=True,
        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    return original, student, wrapped, native, table, generation


def original_probabilities(model, ids, positions):
    with torch.no_grad():
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, positions]
        return F.linear(h, model.lm_head.weight).float().softmax(-1)


def test_text_teacher_ignores_adapter_and_restores_student_clock():
    original, student, wrapped, native, table, generation = setup_pair()
    row = {'id': 'text', 'source_id': 'source0', 'group': 'text',
        'input_ids': [1, 3, 4, 6, 7, 8, 9], 'prediction_positions': [0, 3, 5]}
    ids = torch.tensor([row['input_ids']])
    pos = torch.tensor(row['prediction_positions'])
    student.eval()
    before = original_probabilities(student, ids, pos)
    expected = original_probabilities(original, ids, pos)
    assert not torch.allclose(before, expected, atol=1e-7, rtol=1e-6)
    replay_ids, replay_pos, teacher_logits, trace = original_teacher(student, wrapped,
        SimpleNamespace(eos_token_id=2), row, native, table, generation)
    probabilities = teacher_logits.softmax(-1)
    torch.testing.assert_close(probabilities, expected, atol=1e-7, rtol=1e-6)
    assert not probabilities.requires_grad
    np.testing.assert_array_equal(student.model.rotary_emb.inv_freq.numpy(), table['values_float32'])
    assert student.model.rotary_emb.attention_scaling == table['gain'] and student.training
    after = original_probabilities(student, ids, pos)
    torch.testing.assert_close(after, before, atol=1e-7, rtol=1e-6)
    ce, count = causal_loss(student, ids, ids, chunk_size=2)
    kl, positions = native_teacher_kl(student, replay_ids, replay_pos, teacher_logits)
    (ce+kl).backward()
    assert count == ids.shape[1]-1 and positions == 3
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)
               for name, p in student.named_parameters() if 'lora_' in name)


@pytest.mark.parametrize('use_alternative_eos', [False, True])
def test_generation_teacher_uses_original_greedy_prefixes(use_alternative_eos):
    original, student, wrapped, native, table, generation = setup_pair()
    row = {'id': 'instruction', 'source_id': 'source1', 'group': 'instruction',
        'prompt_ids': [1, 4, 6, 9], 'generation_budget': 4}
    with torch.no_grad():
        result = original.generate(torch.tensor([row['prompt_ids']]), generation_config=generation,
            max_new_tokens=4, logits_to_keep=1)
    expected_tokens = result[0, len(row['prompt_ids']):].tolist()
    tokenizer = SimpleNamespace(eos_token_id=2)
    if use_alternative_eos:
        # The first genuine greedy token becomes an additional valid stop token.
        # A distinct tokenizer EOS makes using only that singleton observable.
        tokenizer.eos_token_id = (expected_tokens[0]+1) % 32
        generation.eos_token_id = [tokenizer.eos_token_id, expected_tokens[0]]
        expected_tokens = expected_tokens[:1]
    ids, pos, teacher_logits, trace = original_teacher(student, wrapped,
        tokenizer, row, native, table, generation)
    assert trace['generated_ids'] == expected_tokens
    assert ids[0].tolist() == row['prompt_ids']+expected_tokens[:-1]
    assert pos[-1] == ids.shape[1]-1
    torch.testing.assert_close(teacher_logits.softmax(-1), original_probabilities(original, ids, pos), atol=1e-7, rtol=1e-6)
    assert student.training
    if use_alternative_eos:
        assert trace['eos'] and len(trace['generated_ids']) == 1


def test_frozen_adapter_roundtrip_preserves_unmerged_bf16_update(tmp_path):
    original, student, wrapped, native, table, generation = setup_pair(rank=16)
    folder = tmp_path/'adapter'
    wrapped.save_pretrained(folder, safe_serialization=True)
    (folder/'deployment.json').write_text(json.dumps(dict(table=table,
        model_revision='tiny-independent-base', optimizer_steps=128)))
    files = {str(p): digest(p) for p in folder.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match='deployment table'):
        load_frozen_adapter(original, folder, dict(table, gain=1.), 'tiny-independent-base', files)
    with torch.no_grad():
        for name, parameter in student.named_parameters():
            if 'lora_' in name:
                parameter.data = parameter.data.to(torch.bfloat16)
    student.eval()
    ids = torch.tensor([[1, 3, 5, 9]])
    positions = torch.tensor([0, 3])
    expected = original_probabilities(student, ids, positions)
    install_signed(original, table['values_float32'], table['gain'])
    before = original_probabilities(original, ids, positions)
    loaded, receipt = load_frozen_adapter(original, folder, table, 'tiny-independent-base', files)
    actual = original_probabilities(loaded, ids, positions)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    assert not torch.allclose(actual, before, atol=1e-7, rtol=1e-6)
    assert not receipt['merged'] and receipt['dtype'] == 'torch.bfloat16'
    assert all(p.dtype == torch.bfloat16 for name, p in loaded.named_parameters() if 'lora_' in name)


def test_trace_keeps_each_actual_decoding_query_without_changing_generation(tmp_path):
    original, _, _, _, _, generation = setup_pair()
    prompt = torch.tensor([[1, 4, 6, 9]])
    with torch.inference_mode():
        expected = original.generate(prompt, generation_config=generation, max_new_tokens=4, logits_to_keep=1)
        buffers, handles = attach_decision_trace(original)
        try:
            output = original.generate(prompt, generation_config=generation, max_new_tokens=4,
                logits_to_keep=1, return_dict_in_generate=True, output_scores=True)
        finally:
            for handle in handles:
                handle.remove()
        torch.testing.assert_close(output.sequences, expected)
        generated = output.sequences[0, len(prompt[0]):].tolist()
        folder = tmp_path/'trace'
        save_decision_trace(original, buffers, output.scores, generated,
            dict(row_id='tiny', input_tokens=4, prompt_sha256='fixture'), folder)
        saved = torch.load(folder/'layer_0.pt', map_location='cpu', weights_only=True)
        assert saved['q'].shape == (1, len(generated), 128)
        assert saved['k'].shape == (1, 4+len(generated)-1, 128)
        assert saved['pos'].tolist() == list(range(3, 3+len(generated)))
        teacher_input = output.sequences[:, :-1]
        hidden = original.model.layers[0].input_layernorm(original.model.embed_tokens(teacher_input))
        q = original.model.layers[0].self_attn.q_proj(hidden)[0, saved['pos']]
        torch.testing.assert_close(saved['q'][0], q, atol=1e-6, rtol=1e-6)
        scores = torch.load(folder/'generation_scores.pt', map_location='cpu', weights_only=True)
        assert scores.argmax(-1).tolist() == generated
