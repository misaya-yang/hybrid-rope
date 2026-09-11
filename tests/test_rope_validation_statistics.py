import pytest
from experiments.rope_decision_20260911.read_validation import contrast


def test_equal_task_weights_are_not_row_weights():
    cases={str(i):dict(task='a' if i<2 else 'b',length_cap=4096) for i in range(8)}
    a={str(i):dict(correct=1. if i<2 else 0.) for i in range(8)}
    b={str(i):dict(correct=0.) for i in range(8)}
    r=contrast(a,b,cases,4096)
    assert r['delta_pp']==pytest.approx(50.)
    assert r['se_pp']==0
    assert r['wins']==2 and r['ties']==6
