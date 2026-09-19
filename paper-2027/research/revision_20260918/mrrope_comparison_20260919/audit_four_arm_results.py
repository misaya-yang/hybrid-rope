"""Offline audit of retained four-arm outputs; never loads models or launches runs."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

OWNER = Path(__file__).resolve().parent
ROOT = OWNER / 'four_arm_results'
ARMS = ['llama31_mrpro', 'llama31_tailspline', 'llama3_mrpro', 'llama3_tailspline']
TASKS = ['niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1',
         'niah_multikey_2', 'niah_multikey_3', 'niah_multivalue', 'niah_multiquery',
         'vt', 'cwe', 'fwe', 'qa_1', 'qa_2']
PAIR_KEYS = ['task', 'source_index', 'source_order_index', 'row_id', 'prompt_sha256',
             'input_ids_sha256', 'input_tokens', 'references', 'generate_kwargs', 'seed']


def read(path):
    return json.loads(path.read_text())


def main():
    comparison = read(ROOT / 'comparison.json')
    assert comparison['status'] == read(ROOT / 'queue_status.json')['status'] == 'COMPLETE'
    records, protocols, runtimes, scores, audit = {}, {}, {}, {}, {}
    for name in ARMS:
        folder = ROOT / name
        rows = [json.loads(line) for line in (folder / 'generations.jsonl').read_text().splitlines()]
        records[name], protocols[name], runtimes[name] = rows, read(folder / 'protocol.json'), read(folder / 'runtime.json')
        p, rt, report = protocols[name], runtimes[name], read(folder / 'report.json')
        assert len(rows) == len({r['row_id'] for r in rows}) == 130
        assert Counter(r['task'] for r in rows) == Counter({t: 10 for t in TASKS})
        assert p['scale'] == 16 and p['reference_length'] == 8192 and p['context'] == 131072
        assert p['dtype'] == 'bfloat16' and p['quantization'] is None
        assert p['attention'] == rt['attention'] == 'flash_attention_2'
        assert rt['device_map'] == {'parameter_devices': ['cuda:0']}
        assert p['config']['max_position_embeddings'] == (131072 if name.startswith('llama31_') else 8192)
        values = {}
        for ti, task in enumerate(TASKS):
            subset = [r for r in rows if r['task'] == task]
            assert [r['source_index'] for r in subset] == list(range(10))
            assert [r['seed'] for r in subset] == [20260919 + ti * 10 + i for i in range(10)]
            values[task] = []
            for row in subset:
                assert row['references'] and 0 < row['input_tokens'] <= 131072
                assert 0 < len(row['output_ids']) <= 30
                pred = row['prediction'].lower()
                matched = [ref.lower() in pred for ref in row['references']]
                values[task].append(float(any(matched)) if task.startswith('qa_') else sum(matched) / len(matched))
        exact = {t: 100 * sum(values[t]) / 10 for t in TASKS}
        rounded = {t: round(exact[t], 2) for t in TASKS}
        for t in TASKS:
            assert report['tasks'][t] == {'n': 10, 'score': rounded[t]}
        macro = sum(rounded.values()) / 13
        assert abs(report['full13_task_equal_score'] - macro) < 1e-10
        assert report == comparison['reports'][name] and report['model_execution'] is True
        scores[name] = rounded
        audit[name] = {'rows': len(rows), 'unique_rows': 130, 'tasks': 13,
                       'report_macro_percent': macro, 'unrounded_macro_percent': sum(exact.values()) / 13,
                       'sum_generation_seconds': sum(r['seconds'] for r in rows),
                       'output_tokens': sum(len(r['output_ids']) for r in rows),
                       'stop_eos_or_newline_rows': sum(r['output_ids'][-1] in [128009, 198] for r in rows),
                       'at_output_budget_rows': sum(len(r['output_ids']) == 30 for r in rows)}
    for name in ARMS:
        for a, b in zip(records[ARMS[0]], records[name]):
            assert all(a[k] == b[k] for k in PAIR_KEYS), name
        for key in ['versions', 'generation_config', 'runner_sha256', 'tokenizer']:
            assert protocols[name][key] == protocols[ARMS[0]][key], key
        assert runtimes[name]['effective_generation_config'] == runtimes[ARMS[0]]['effective_generation_config']
        assert runtimes[name]['gain'] == runtimes[ARMS[0]]['gain']
    for method in ['mrpro', 'tailspline']:
        assert runtimes['llama31_' + method]['inv_freq'] == runtimes['llama3_' + method]['inv_freq']
    assert hashlib.sha256((OWNER / 'official_single_arm.py').read_bytes()).hexdigest() == protocols[ARMS[0]]['runner_sha256']
    output = {'status': 'PASS', 'scope': 'Offline raw prediction rescoring and recorded identity audit; no model rerun',
              'paired_keys_verified': PAIR_KEYS, 'same_method_tables_equal_across_checkpoints': True,
              'shared_gain': runtimes[ARMS[0]]['gain'],
              'input_tokens_min': min(r['input_tokens'] for r in records[ARMS[0]]),
              'input_tokens_max': max(r['input_tokens'] for r in records[ARMS[0]]), 'arms': audit,
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name not in ['audit.json', 'per_task.csv']}}
    (ROOT / 'audit.json').write_text(json.dumps(output, indent=2) + '\n')
    with (ROOT / 'per_task.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['task'] + ARMS + ['llama31_T_minus_P_pp', 'llama3_T_minus_P_pp'])
        for t in TASKS:
            writer.writerow([t] + [scores[a][t] for a in ARMS] +
                            [round(scores[f+'_tailspline'][t] - scores[f+'_mrpro'][t], 2) for f in ['llama31', 'llama3']])
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
