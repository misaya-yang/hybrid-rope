from types import SimpleNamespace
import torch

from scripts.experiments.olmo_fast_screen.diagnose import greedy


class Fake:
    device=torch.device('cpu')
    def __init__(self):self.calls=[]
    def __call__(self,**kwargs):
        ids=kwargs['input_ids'][0].tolist();positions=kwargs['position_ids'][0].tolist()
        self.calls.append((ids,positions,kwargs['attention_mask'].shape[-1]))
        cache=kwargs['past_key_values'];past=0 if cache is None else cache.get_seq_length()
        total=past+len(ids)
        logits=torch.full((1,1,16),-10.);logits[0,0,ids[-1]+1]=10.
        return SimpleNamespace(logits=logits,past_key_values=SimpleNamespace(get_seq_length=lambda:total))


def test_first_prediction_and_noncontiguous_positions_and_eos_list():
    model=Fake()
    result,_=greedy(model,[4,5],[0,99],max_new_tokens=8,eos_token_id=[7,9])
    assert result['generated_ids']==[6,7] and result['ended_eos']
    assert model.calls==[([4,5],[0,99],2),([6],[100],3)]


def test_max_budget_does_not_append_or_forward_an_extra_token():
    model=Fake()
    result,cache=greedy(model,[4,5],[0,99],max_new_tokens=1,eos_token_id=7)
    assert result['generated_ids']==[6] and not result['ended_eos']
    assert len(model.calls)==1 and cache.get_seq_length()==2


def test_cached_query_uses_physical_length_but_original_position():
    model=Fake();cache=SimpleNamespace(get_seq_length=lambda:3)
    result,_=greedy(model,[5],[130999],max_new_tokens=2,eos_token_id=7,cache=cache)
    assert result['generated_ids']==[6,7]
    assert model.calls==[([5],[130999],4),([6],[131000],5)]
