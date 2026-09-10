import copy
import json

import numpy as np
import pytest
import torch
from transformers import Olmo2Config,Olmo2ForCausalLM
from transformers.models.olmo2.modeling_olmo2 import Olmo2RotaryEmbedding

from .data import JsonlIndex,qa_scores,supervised_chat
from .runtime import configure,training_step
from .tables import construct
from .train import order_index
from .evaluate import language_scores,score_output
from .prepare import shingles
from scripts.experiments.cross_audit.training import causal_loss
from scripts.experiments.cross_audit.tables import install_static


def tiny():
    torch.manual_seed(23)
    return Olmo2ForCausalLM(Olmo2Config(vocab_size=67,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,
        max_position_embeddings=16,rope_theta=500000,pad_token_id=0,eos_token_id=2))


def test_actual_olmo_grids_and_matched_deformation():
    config=Olmo2Config(hidden_size=2048,num_attention_heads=16,num_key_value_heads=16,
        max_position_embeddings=4096,rope_theta=500000)
    native=Olmo2RotaryEmbedding(config).inv_freq.numpy()
    arms=construct(config.to_dict(),native)
    rms=[]
    for name in ('Cosh','Exponential','Hybrid'):
        values=np.asarray(arms[name]['values'],dtype=np.float32)
        assert np.array_equal(values[[0,-1]],native[[0,-1]])
        assert np.all(values[:-1]>values[1:])
        rms.append(arms[name]['rms_normalized_deformation'])
    assert max(rms)-min(rms)<1e-10
    assert arms['YaRN']['gain']>1
    assert np.array_equal(np.asarray(arms['Native']['values'],dtype=np.float32),native)


def test_chunked_dense_ce_matches_full_value_and_gradient():
    model=tiny();reference=copy.deepcopy(model)
    ids=torch.tensor([[3,4,5,6,7,8,2]])
    actual,count=causal_loss(model,ids,ids,chunk_size=2)
    logits=reference(input_ids=ids[:,:-1],use_cache=False).logits
    expected=torch.nn.functional.cross_entropy(logits.flatten(0,1),ids[:,1:].flatten())
    actual.backward();expected.backward()
    assert count==6
    torch.testing.assert_close(actual,expected)
    for a,b in zip(model.parameters(),reference.parameters()):
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=3e-5,atol=1e-6)


def test_all_linear_update_has_ffn_signal_and_correct_prediction_counts(tmp_path):
    base=tiny();initial=copy.deepcopy(base)
    model,wrapper=configure(base,rank=4)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    row=dict(input_ids=[3,4,5,6,7,2],target_start=4)
    record=training_step(model,optimizer,np.asarray([3,4,5,6,7,8,2]),row,row,chunk_size=2)
    assert record['cpt_prediction_tokens']==6
    assert record['sft_prediction_tokens']==record['native_prediction_tokens']==2
    assert all(record['module_grad_norms'][name]>0 for name in ('q_proj','v_proj','gate_proj','up_proj','down_proj'))
    assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
    wrapper.save_pretrained(tmp_path)
    from peft import PeftModel
    restored=PeftModel.from_pretrained(initial,tmp_path).get_base_model()
    ids=torch.tensor([[3,4,5,6]])
    model.eval();restored.eval()
    torch.testing.assert_close(model(ids).logits,restored(ids).logits)


def test_fixed_rotary_survives_cached_decoding_beyond_configured_window():
    model=tiny().eval()
    native=model.model.rotary_emb.inv_freq.detach().numpy().copy()
    values=(native*.9).astype(np.float32)
    install_static(model,values,1.)
    ids=torch.tensor([[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]])
    with torch.no_grad():
        full=model(ids,use_cache=False).logits[:,-1]
        pre=model(ids[:,:-1],use_cache=True)
        cached=model(ids[:,-1:],past_key_values=pre.past_key_values,use_cache=True).logits[:,-1]
    torch.testing.assert_close(full,cached,rtol=1e-4,atol=1e-6)
    np.testing.assert_array_equal(model.model.rotary_emb.inv_freq.numpy(),values)


def test_source_index_and_full_answer_metrics(tmp_path):
    file=tmp_path/'rows.jsonl'
    file.write_text(json.dumps({'text':'答案'})+'\n'+json.dumps({'input_ids':[1,2,3]})+'\n')
    rows=JsonlIndex(file)
    assert len(rows)==2 and rows[0]['text']=='答案' and rows[1]['input_ids']==[1,2,3]
    assert qa_scores('Paris',['Paris'])==dict(f1=1.,exact=1.)
    assert qa_scores('The answer might be Paris or London',['Paris'])['exact']==0
    assert qa_scores('London',['Paris'])['f1']==0


def test_resume_order_is_fixed_and_visits_every_row():
    order=[order_index(i,17,42) for i in range(34)]
    assert set(order[:17])==set(range(17))==set(order[17:])
    assert order[13:]==[order_index(i,17,42) for i in range(13,34)]


def test_native_template_supervision_boundary():
    class Tokenizer:
        eos_token_id=2
        def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=False):
            return [3,4,5] if add_generation_prompt else [3,4,5,6,7,2]
    row=supervised_chat(Tokenizer(),'question','answer')
    assert row['target_start']==3 and row['answer_tokens']==3


def test_multiple_required_ruler_items_are_not_alternative_exact_answers():
    row=dict(suite='ruler',references=['alpha','beta'])
    scores=score_output(row,'alpha')
    assert scores['official_recall']==.5 and scores['all_answers_found']==0
    assert scores['exact'] is None


def test_single_forward_lm_readout_matches_training_ce():
    model=tiny().eval();ids=torch.tensor([[3,4,5,6,7,8,2]])
    with torch.no_grad():
        full,tail,n=language_scores(model,ids,chunk_size=2)
        ce,_=causal_loss(model,ids,ids,chunk_size=2)
    assert full==pytest.approx(float(ce),abs=1e-6)
    assert full==tail and n==6


def test_document_overlap_screen_detects_offset_shift():
    text=' '.join(f'word{i}' for i in range(150))
    denied=shingles(text)
    shifted='a different document wrapper starts here '+text
    assert len(shingles(shifted,stride=1)&denied)>=3
