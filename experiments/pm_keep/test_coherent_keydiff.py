import torch
from .coherent_keydiff import sentence_units,permuted_units,allocate_units

def test_units():
    text='A. BC!\nDEF? G'
    units=sentence_units(text,[(i,i+1) for i in range(len(text))])
    assert units==[(0,2),(2,6),(6,7),(7,11),(11,13)]
    sham=permuted_units(units)
    assert sorted(b-a for a,b in sham)==sorted(b-a for a,b in units)
    assert sham[0][0]==0 and sham[-1][1]==len(text)

def test_budget_and_coherence():
    scores=torch.tensor([[9.,0.,8.,0.,7.,0.,6.,0.]])
    sets,receipt=allocate_units(scores,[(0,2),(2,4),(4,6),(6,8)],4,sink_tokens=0,recent_tokens=0)
    assert sets.tolist()==[[0,1,2,3]] and receipt[0]['remainder_token_slots']==0
    for budget in range(3,9):
        sets,receipt=allocate_units(scores,[(0,2),(2,4),(4,6),(6,8)],budget,sink_tokens=1,recent_tokens=2)
        assert len(set(sets[0].tolist()))==budget
        assert {0,6,7}<=set(sets[0].tolist())
    singleton=[(i,i+1) for i in range(8)]
    sets,_=allocate_units(scores,singleton,4,sink_tokens=0,recent_tokens=0)
    assert sets.tolist()==[[0,2,4,6]]

if __name__=='__main__':
    test_units();test_budget_and_coherence();print('PASS: prefix partition, sham lengths, coherent units, exact budgets and protected positions')
