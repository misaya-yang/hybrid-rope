from dataclasses import replace
from pathlib import Path
import torch
from .runtime import AttentionSettings,SelectionContext
from .selector_controls import BlockSummarySelector
from .full_covariance_probe import full_covariance_logmass
from .candidate_refinement import refine_candidate_scores,CandidateRefinementSelector,scientific_ast

@torch.no_grad()
def test_refinement():
    torch.manual_seed(417)
    torch.set_num_threads(1)
    settings=AttentionSettings(kernel_size=4,kernel_stride=2,block_size=4,
        init_blocks=1,local_blocks=1,select_blocks=1,topk=4,attention_query_chunk_size=2)
    k=torch.randn(2,40,8);q=torch.randn(4,3,8);cis=torch.randn(2,40)
    context=SelectionContext(q,k,torch.randn_like(k),cis,torch.tensor([13,29,38]),0,settings)
    coarse=BlockSummarySelector('pc2').logmass(context)
    grouped,updated,pool,_=refine_candidate_scores(context,coarse)
    reference=full_covariance_logmass(context)
    index=pool[:,None].expand(-1,2,-1,-1)
    torch.testing.assert_close(updated.gather(-1,index),reference.gather(-1,index),atol=2e-6,rtol=2e-6)
    allowed=torch.zeros_like(grouped,dtype=torch.bool).scatter_(-1,pool,True)
    assert torch.isneginf(grouped[~allowed]).all()
    full_context=replace(context,settings=replace(settings,topk=10))
    full_group,full_scores,_,_=refine_candidate_scores(full_context,coarse)
    torch.testing.assert_close(full_scores,reference,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(full_group,reference.softmax(-1).sum(1),atol=2e-6,rtol=2e-6)
    changed_k=k.clone();changed_cis=cis.clone();changed_k[:,39]=1000;changed_cis[:,39]=1000
    changed=replace(context,k=changed_k,cis=changed_cis)
    changed_coarse=BlockSummarySelector('pc2').logmass(changed)
    new_group,_,_,_=refine_candidate_scores(changed,changed_coarse)
    torch.testing.assert_close(grouped,new_group,atol=0,rtol=0)
    selected=CandidateRefinementSelector()(context)
    assert selected.shape==(2,3,4)
    assert (selected<=context.query_positions[None,:,None]//4).all()

def test_scientific_identity():
    source=Path(__file__).with_name('candidate_refinement.py').read_text()
    assert scientific_ast(source)==scientific_ast(source+'\n# wrapper comment\n')
    assert scientific_ast(source)!=scientific_ast(source.replace('0.5*variance','0.25*variance'))

if __name__=='__main__':
    test_refinement();test_scientific_identity();print('PASS: selected raw moments match full reference; all-pool identity; no future leakage; native quota output')
