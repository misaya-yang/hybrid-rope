import json
import sys
from pathlib import Path
import pytest
from scripts.experiments.cross_audit.contracts import sha_file
from scripts.experiments.cross_audit.jobs import run
from scripts.experiments.cross_audit.train import paired_sft,scheduled_lr,save_final_checkpoint
from scripts.experiments.cross_audit.teacher import prediction_positions


def test_native_pool_hidden_positions_are_not_shifted_twice():
    row=dict(input_ids=[1,2,3,4,5],prompt_ids=[1,2,3],prediction_positions=[2,3,4])
    ids,positions=prediction_positions(row)
    assert positions==[2,3,4]
    assert ids[positions[0]+1]==4
    assert prediction_positions(row,require_target=True)[1]==[2,3]


def test_paired_training_order_is_seeded_and_complete():
    rows=[dict(group_id=g,world=w,layout=l) for g in ('a','b','c') for w in ('0','1') for l in ('near','far')]
    assert paired_sft(rows,137)==paired_sft(list(reversed(rows)),137)
    with pytest.raises(ValueError,match='incomplete'):
        paired_sft(rows[:-1],137)
    assert scheduled_lr(100,100,2e-5)==pytest.approx(2e-6)


def test_supervisor_refuses_changed_plan_or_failed_dependency(tmp_path):
    path=tmp_path/'plan.json'
    path.write_text(json.dumps(dict(state_dir=str(tmp_path/'state'),jobs=[dict(id='run',dependencies=['previous'])])))
    with pytest.raises(ValueError,match='changed'):
        run(path,'run','not-the-digest')
    with pytest.raises(ValueError,match='dependency'):
        run(path,'run',sha_file(path))


def test_final_checkpoint_reloads_updated_weights_without_resume(tmp_path):
    import torch
    from transformers import Olmo2Config,Olmo2ForCausalLM,PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    cfg=Olmo2Config(hidden_size=16,intermediate_size=24,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,vocab_size=23)
    cfg._attn_implementation='eager';model=Olmo2ForCausalLM(cfg)
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'[UNK]':0},unk_token='[UNK]')),
        unk_token='[UNK]')
    original={n:p.detach().clone() for n,p in model.named_parameters()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    ids=torch.tensor([[1,2,3,4]])
    model(ids,labels=ids,use_cache=False).loss.backward();optimizer.step()
    assert any(not torch.equal(p,original[n]) for n,p in model.named_parameters())
    dest=tmp_path/'step_1528'
    save_final_checkpoint(model,tokenizer,dest,arm='Z',entry={'tensor_sha256':'table-id','amplitude':1.1},
        contract_sha256='contract-id',step=1528,input_tokens=99)
    loaded=Olmo2ForCausalLM.from_pretrained(dest,local_files_only=True)
    for name,p in loaded.named_parameters():torch.testing.assert_close(p,dict(model.named_parameters())[name])
    assert list(tmp_path.iterdir())==[dest]
    assert not list(tmp_path.rglob('*.pt'))
    manifest=json.loads((dest/'checkpoint_manifest.json').read_text())
    assert manifest['step']==1528 and manifest['input_tokens']==99
    for name,digest in manifest['files'].items():assert sha_file(dest/name)==digest
