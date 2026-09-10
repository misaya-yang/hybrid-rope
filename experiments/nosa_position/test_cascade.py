import unittest
import torch
from .cascade import exact_candidates, replace_candidate_mass
from .runtime import AttentionSettings, SelectionContext
from .exact_probe import ExactBlockSelector


class CascadeTests(unittest.TestCase):
    def test_candidate_scoring_matches_causal_exact_and_padding_is_inert(self):
        torch.manual_seed(7)
        c=SelectionContext(q=torch.randn(4,2,8),k=torch.randn(2,7,8),v=torch.zeros(2,7,8),cis=torch.randn(2,7),query_positions=torch.tensor([5,6]),layer_idx=0,settings=AttentionSettings(block_size=2,kernel_stride=1,kernel_size=2))
        ids=torch.tensor([[[0,2,-1],[1,3,-1]],[[1,2,-1],[0,3,-1]]])
        candidate,_=exact_candidates(c,ids)
        oracle=ExactBlockSelector('exact_mass').logmass(c)
        for h in range(2):
            for q in range(2):
                valid=ids[h,q]>=0
                torch.testing.assert_close(candidate[h,:,q,valid],oracle[h,:,q,ids[h,q,valid]],atol=1e-6,rtol=1e-6)
        mixed=replace_candidate_mass(torch.zeros_like(oracle),ids,candidate)
        self.assertEqual(float(mixed[0,:,1,0].sum()),0.)

    def test_candidate_only_denominator_reverses_gqa_ranking(self):
        mass=torch.tensor([[[[.4,.1,.5]],[[.001,.009,.99]]]])
        ids=torch.tensor([[[0,1]]])
        mixed=replace_candidate_mass(mass.log(),ids,mass.log()[...,:2])
        self.assertEqual(int(mixed.softmax(-1).sum(1)[...,:2].argmax()),0)
        self.assertEqual(int(mixed[...,:2].softmax(-1).sum(1).argmax()),1)

if __name__=='__main__':unittest.main()
