"""Sampling boundaries and first-order objective identity, not GPU qualification."""
import itertools
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('prefix_trainer',Path(__file__).parents[1]/'scripts/train/train_single_table_native_constrained.py')
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
prefix_lm_positions=trainer.prefix_lm_positions


def test_prefix_sample_is_frozen_disjoint_from_answer_and_spans_long_prompt():
    row={'prompt_ids':list(range(16000)),'semantic_id':'fixed','world':0,'length_cap':16384,'layout':'far'}
    a=prefix_lm_positions(row,42)
    assert a==prefix_lm_positions(row,42)
    assert a!=prefix_lm_positions(row,43)
    assert len(a)==len(set(a))==128
    assert min(a)<4000 and max(a)>12000
    assert min(a)>=0 and max(a)<len(row['prompt_ids'])-1
    row['prompt_ids']=[7,8,9]
    assert prefix_lm_positions(row,42)==[0,1]


def test_uniform_subset_ce_and_gradient_average_equal_dense_objective():
    import pytest
    torch=pytest.importorskip('torch')
    torch.manual_seed(4)
    logits=torch.randn(4,7,dtype=torch.float64,requires_grad=True)
    labels=torch.tensor([1,2,3,4]);F=torch.nn.functional
    dense=F.cross_entropy(logits,labels)
    samples=[F.cross_entropy(logits[list(indices)],labels[list(indices)]) for indices in itertools.combinations(range(4),2)]
    averaged=torch.stack(samples).mean()
    torch.testing.assert_close(averaged,dense,atol=1e-14,rtol=1e-14)
    a=torch.autograd.grad(averaged,logits,retain_graph=True)[0]
    b=torch.autograd.grad(dense,logits)[0]
    torch.testing.assert_close(a,b,atol=1e-14,rtol=1e-14)
