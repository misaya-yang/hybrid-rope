import pytest

from scripts.experiments.olmo_fast_screen.ruler_bench import TASKS, score, summarize, verdict


def panel(long_score, short_score):
    return [dict(row_id=f'{task}_{cap}', task=task, length_cap=cap, correct=value,
                 references=['answer'], prompt_sha256=f'{task}_{cap}', ended_eos=True)
            for cap, value in ((4096, short_score), (16384, long_score)) for task in TASKS]


def test_official_multi_answer_and_qa_alias_semantics():
    refs = ['amber', 'birch', 'cedar']
    assert score(dict(task='fwe', references=refs), 'AMBER, cedar') == 2/3
    assert score(dict(task='qa_1', references=refs), 'The answer is BIRCH.') == 1
    assert score(dict(task='vt', references=['AB', 'CD']), 'AB\tCD') == 1
    assert score(dict(task='fwe', references=refs), '') == 0
    with pytest.raises(ValueError):
        score(dict(task='qa_1', references=[]), 'anything')


def test_length_and_task_weighting_not_row_count_weighting():
    records = panel(0, 1)
    records[6]['correct'] = 1
    records.extend([dict(records[6], row_id=f'extra_{n}') for n in range(3)])
    result = summarize(records)
    assert result['by_length']['4096']['macro_accuracy'] == 1
    assert result['by_length']['16384']['macro_accuracy'] == pytest.approx(1/6)


def test_long_gain_with_short_loss_is_tradeoff():
    result = verdict(panel(.75, .25), panel(.5, .5))
    assert result['status'] == 'TRADEOFF'
    assert result['macro_delta_by_length'] == {'4096': -.25, '16384': .25}
    assert (result['paired_wins'], result['paired_losses']) == (6, 6)


def test_mismatched_prompt_cannot_reuse_baseline():
    candidate, baseline = panel(.75, .5), panel(.5, .5)
    candidate[0]['prompt_sha256'] = 'different'
    with pytest.raises(ValueError, match='identity'):
        verdict(candidate, baseline)


def test_duplicate_or_missing_task_rejected():
    baseline = panel(.5, .5)
    with pytest.raises(ValueError):
        verdict(baseline+[baseline[0]], baseline)
    with pytest.raises(ValueError, match='incomplete'):
        verdict(baseline[1:], baseline[1:])


def test_predeclared_longer_cap_uses_its_actual_primary_endpoint():
    candidate, baseline = panel(.75, .5), panel(.5, .5)
    for rows in (candidate, baseline):
        for row in rows:
            if row['length_cap'] == 16384:row['length_cap'] = 32768
    result = verdict(candidate, baseline)
    assert result['status'] == 'DEVELOPMENT_WIN'
    assert result['macro_delta_by_length'] == {'4096': 0, '32768': .25}
