import numpy as np
import pytest
from experiments.rope_decision_20260911.qwen_readout import clustered_se, summarize


def test_lower_nll_wins():
    d = [-.02,-.019,-.021,-.02,-.018,-.022]
    r = summarize(d,[0.]*6,list(range(6)))
    assert r['lower_nll_arm']=='BM' and r['criterion_met']
    r = summarize([0.]*6,d,list(range(6)))
    assert r['lower_nll_arm']=='MrRoPE' and r['criterion_met']


def test_unequal_cluster_sizes_match_piece_mean():
    values, groups = [0.]*10+[1.,2.], [0]*10+[1,2]
    assert clustered_se(values,groups) == pytest.approx(.32072508996543025)


def test_singleton_clusters_reduce_to_paired_se():
    d = np.array([.01,.02,.04,.08])
    assert clustered_se(d,range(4))==pytest.approx(d.std(ddof=1)/2)


def test_mismatched_pairs_are_not_truncated():
    with pytest.raises(ValueError):
        summarize([0.,1.],[1.],[0,1])
