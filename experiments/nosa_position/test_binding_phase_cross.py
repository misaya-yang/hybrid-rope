import torch
from .binding_phase_cross import PhaseCrossSelector,make_selector
from .test_runtime import context

def test_phase_boundary():
    ctx=context(43,[31,32,33])
    cross=PhaseCrossSelector('full_to_cobs');cross.threshold=32
    seen=[]
    def left(c):
        assert c is ctx;seen.append('left');return torch.tensor([[[0,1],[0,1],[0,1]]]*2)
    def right(c):
        assert c is ctx;seen.append('right');return torch.tensor([[[0,2],[0,2],[0,2]]]*2)
    cross.history,cross.answer=left,right
    assert cross(ctx).tolist()==[[[0,1],[0,2],[0,2]]]*2 and seen==['left','right']
    assert cross.metrics['history_queries']==1 and cross.metrics['answer_queries']==2

def test_same_source_identity():
    torch.set_num_threads(1)
    ctx=context(43,[31,32,33])
    for name in ('full_covariance','cobs_rank2','pc2','native'):
        cross=PhaseCrossSelector(history=name,answer=name);cross.threshold=32
        assert torch.equal(cross(ctx),make_selector(name)(ctx))

if __name__=='__main__':test_phase_boundary();test_same_source_identity();print('PASS: final-token boundary, identical forwarded context, same-source native selection identity')
