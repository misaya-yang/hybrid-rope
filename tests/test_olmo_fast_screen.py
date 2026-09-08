"""Projection, source dependence, scoring and spend-boundary regressions; no GPU."""
import json
import math

import pytest

from scripts.analysis.project_mrpro_transition import apply, basis, project
from scripts.experiments.olmo_fast_screen.bench import FAMILIES, instance, normalize_answer, score, verdict
from scripts.experiments.olmo_fast_screen.run import eligible_queue
from scripts.experiments.olmo_fast_screen.supervise import phase_state
from scripts.analysis.build_boundary_matched_mrpro import increments, cumulative, roughness, independent_minimum


def test_nontrivial_projection_matches_independent_linear_solution():
    _, psi = basis(7,0,6)
    # A nonzero constant component is orthogonal to this symmetric odd basis.
    residual = [2.5*v+.2 for v in psi]
    a = project(residual,psi)
    assert a == pytest.approx(2.5)
    err = lambda coefficient: sum((d-coefficient*v)**2 for d,v in zip(residual,psi))
    assert err(a) < err(0)
    assert err(a) < err(a-.1) and err(a) < err(a+.1)


def test_actual_projection_retains_negative_exponents_and_outer_bands():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root/'docs/research/ROPE_MRPRO_TRANSITION_PROJECTION_20260908.json').read_text())
    assert data['a_star'] == pytest.approx(2.1477690111869907,abs=1e-12)
    for key in ('source_qwen','target_olmo'):
        checks = data[key]['checks']
        assert checks['finite_positive'] and checks['strictly_decreasing']
        assert checks['fast_band_bitwise_equal'] and checks['slow_band_bitwise_equal']
        assert checks['support_endpoints_bitwise_equal']
        assert checks['exponent_min'] < -.16  # No silent clipping or a_star shrinkage.
        assert abs(checks['sum_exponent_delta_intended']) < 1e-12
    assert data['review']['main_opposition_slots'] == [30,31]


@pytest.mark.parametrize('family',FAMILIES)
def test_counterfactuals_need_the_source_and_fit_capacity(family):
    encode = lambda text:list(text.encode())
    pair = instance(family,768,20260908,encode,compact=True)
    first,second = pair
    assert first['question'] == second['question']
    assert first['answer'] != second['answer']
    assert first['prompt_ids'] != second['prompt_ids']
    for row in pair:
        assert len(row['prompt_ids'])+row['max_new_tokens'] <= row['length_cap']
        assert row['answer'] not in row['question']
        assert all(line in row['prompt_text'] for line in row['evidence_lines'])
        assert score(row,row['answer']) == 1
        assert score(row,'The answer might be '+row['answer']+' or something else.') == 0
    assert score(second,first['answer']) == 0


def test_scoring_does_not_reward_target_substrings_or_require_eos():
    row = {'answer':'jade'}
    assert score(row,'Answer: JADE.') == 1
    assert score(row,'jade\ncoral') == 0
    assert score(row,'not jade') == 0


def examples(correct=0):
    return [dict(row_id=f+'_'+str(i),family=f,length_cap=(4096,8192)[i],
                 correct=correct,ended_eos=True) for f in FAMILIES for i in (0,1)]


def test_a_long_gain_with_native_regression_is_a_tradeoff():
    baseline = examples()
    baseline[0]['correct'] = 1
    candidate = examples()
    candidate[1]['correct'] = candidate[3]['correct'] = 1
    assert verdict(candidate,baseline)['status'] == 'TRADEOFF'
    candidate[0]['correct'] = 1
    assert verdict(candidate,baseline)['status'] == 'DEVELOPMENT_WIN'


def test_partial_or_duplicate_results_are_not_ranked():
    baseline = examples()
    with pytest.raises(ValueError):verdict(baseline[:-1],baseline)
    with pytest.raises(ValueError):verdict(baseline+[baseline[0]],baseline)


def test_unreviewed_candidates_never_enter_the_gpu_queue():
    q = {'ordered_candidates':[{'id':'C1','eligible':False,'review_status':'NOT_RECOMMENDED'}]}
    assert eligible_queue(q) == []
    q['ordered_candidates'][0]['eligible'] = True
    with pytest.raises(ValueError):eligible_queue(q)


def test_resume_preserves_cumulative_budget_and_deadline(tmp_path):
    path = tmp_path/'phase.json'
    original = phase_state(path,300,False,1000.)
    resumed = phase_state(path,300,True,1100.)
    assert resumed['deadline_unix'] == original['deadline_unix'] == 1300.
    with pytest.raises(ValueError):phase_state(path,600,True,1100.)
    with pytest.raises(TimeoutError):phase_state(path,300,True,1301.)
    with pytest.raises(FileExistsError):phase_state(path,300,False,1100.)


def test_five_minute_estimate_does_not_create_a_deadline(tmp_path):
    path=tmp_path/'phase.json'
    state=phase_state(path,None,False,1000.)
    assert state['deadline_unix'] is None and state['budget_seconds'] is None
    # A long elapsed duration alone cannot turn a soft estimate into a kill.
    assert phase_state(path,None,True,100000.)['deadline_unix'] is None


@pytest.mark.parametrize('n',(3,17,18))
def test_boundary_matched_is_the_independent_strict_minimum(n):
    from fractions import Fraction
    candidate=increments(n)
    assert candidate==independent_minimum(n)
    assert sum(candidate)==1 and min(candidate)>0
    for q in range(n+1):assert sum(candidate[:q])==cumulative(n,q)
    mr=[Fraction(2*i,n*(n+1)) for i in range(1,n+1)]
    assert roughness(candidate)/roughness(mr)==Fraction(3,n+2)
    # A feasible nonzero perturbation must strictly increase this objective.
    moved=candidate.copy();delta=min(moved[0],moved[1])/2
    moved[0]+=delta;moved[1]-=delta
    assert sum(moved)==1 and min(moved)>0
    assert roughness(moved)>roughness(candidate)
