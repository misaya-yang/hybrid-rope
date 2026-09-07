import pytest
from scripts.experiments.cross_audit.contracts import score, summarize, validate_rows, select_groups


def row(world='0', **kwargs):
    return dict(row_id=world, group_id='g', world=world, family='single_evidence',
                layout='compact', length_cap=2048, prompt_ids=[1, 2],
                generation_budget=8, accepted_full_answers=['Paris'], **kwargs)


def test_full_answer_rejects_prefix_credit_and_missing_eos():
    assert score(row(), 'Paris', True)['full_answer_exact_eos']
    assert not score(row(), 'Paris but actually London', True)['full_answer_exact_eos']
    assert not score(row(), 'Paris', False)['full_answer_exact_eos']


def test_multi_key_requires_all_values_without_extra_text():
    r = row(expected_ruler_answers=['abc', 'def'])
    r['family'] = 'ruler_multi_key'
    assert score(r, 'abc, def', True)['full_answer_exact_eos']
    assert not score(r, 'abc', True)['full_answer_exact_eos']
    assert not score(r, 'abc def explanation', True)['full_answer_exact_eos']
    assert score(r, 'Here are abc and def', True)['all_required_contains']


def test_worlds_and_outputs_cannot_silently_disappear():
    with pytest.raises(ValueError, match='worlds'):
        validate_rows([row()])
    rs = [row(), row('1')]
    with pytest.raises(ValueError, match='output rows'):
        summarize([dict(row_id='0', full_answer_exact_eos=True)], rs)
    assert len(select_groups(rs, 1)) == 2
    with pytest.raises(ValueError, match='empty'):
        validate_rows([])


def test_layouts_do_not_merge_group_worlds():
    rs = [row(), row('1')]
    far = [{**r, 'row_id': r['row_id']+'f', 'layout': 'far', 'length_cap': 16384} for r in rs]
    outputs = [dict(row_id=r['row_id'], full_answer_exact_eos='f' not in r['row_id']) for r in rs+far]
    summary = summarize(outputs, rs+far)
    assert summary['single_evidence:compact:2048']['complete_group_successes'] == 1
    assert summary['single_evidence:far:16384']['complete_group_successes'] == 0
