"""Verify and combine disjoint natural-QA batches without rerunning references."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from scripts.eval.longbench_metrics import qa_f1_score, normalize_text
from scripts.experiments.olmo_fast_screen.prepare import sha_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    methods = ('MrPro', 'MrProBM')
    rows = {m: {} for m in methods}
    provenance = {}
    contract = None
    for run in args.runs:
        status = json.loads((run/'status.json').read_text())
        if status['status'] != 'COMPLETE':
            raise ValueError('incomplete run')
        runtime = json.loads((run/'runtime.json').read_text())
        current = {key: runtime[key] for key in ('model_id','revision','generation_config','table_identity')}
        if contract is None:
            contract = current
        if contract != current:
            raise ValueError('batch scientific contract differs')
        provenance[run.name] = dict(status=status, files={})
        for method in methods:
            raw = run/(method+'.jsonl')
            complete = json.loads((run/(method+'.json')).read_text())
            if complete['status'] != 'COMPLETE' or sha_file(raw) != complete['raw_sha256']:
                raise ValueError('raw/receipt mismatch')
            data = list(map(json.loads, raw.read_text().splitlines()))
            if [r['row_id'] for r in data] != complete['row_ids']:
                raise ValueError('row order mismatch')
            for r in data:
                if r['row_id'] in rows[method]:
                    raise ValueError('duplicate sample across batches')
                if abs(qa_f1_score(r['output_text'], r['references'])-r['correct']) > 1e-12:
                    raise ValueError('score recomputation failed')
                eos = runtime['generation_config']['eos_token_id']
                eos = {eos} if isinstance(eos, int) else set(eos)
                if r['ended_eos'] != (r['generated_ids'][-1] in eos) or len(r['generated_ids']) > r['max_new_tokens']:
                    raise ValueError('EOS/budget mismatch')
                rows[method][r['row_id']] = r
            provenance[run.name]['files'][method] = sha_file(raw)
    if rows['MrPro'].keys() != rows['MrProBM'].keys():
        raise ValueError('unmatched rows')
    cells = defaultdict(list)
    details = []
    for key, base in rows['MrPro'].items():
        candidate = rows['MrProBM'][key]
        for field in ('prompt_sha256','task','input_tokens','references','max_new_tokens'):
            if candidate[field] != base[field]:
                raise ValueError('paired input/decoder differs')
        stratum = 'extended' if base['input_tokens'] > 4096 else 'within_native_length'
        cells[(base['task'],stratum)].append((base,candidate))
        details.append(dict(row_id=key, prompt_sha256=base['prompt_sha256'],
            input_tokens=base['input_tokens'], baseline=base['correct'], candidate=candidate['correct'],
            baseline_output=base['output_text'], candidate_output=candidate['output_text'],
            baseline_ids=base['generated_ids'], candidate_ids=candidate['generated_ids'],
            baseline_eos=base['ended_eos'], candidate_eos=candidate['ended_eos']))
    summary = {}
    for (task,stratum), pairs in cells.items():
        b = sum(x['correct'] for x,y in pairs)/len(pairs)
        c = sum(y['correct'] for x,y in pairs)/len(pairs)
        summary.setdefault(stratum,{})[task] = dict(n=len(pairs), MrPro=b, BM=c, delta=c-b,
            wins=sum(y['correct']>x['correct'] for x,y in pairs),
            losses=sum(y['correct']<x['correct'] for x,y in pairs),
            MrPro_eos=sum(x['ended_eos'] for x,y in pairs), BM_eos=sum(y['ended_eos'] for x,y in pairs))
    macro = {s:{'MrPro':sum(c['MrPro'] for c in tasks.values())/len(tasks),
                'BM':sum(c['BM'] for c in tasks.values())/len(tasks)} for s,tasks in summary.items()}
    for cell in macro.values():
        cell['delta'] = cell['BM']-cell['MrPro']
    rng = np.random.default_rng(20260908)
    for stratum, tasks in summary.items():
        draws = []
        for task in tasks:
            differences = np.asarray([y['correct']-x['correct'] for x,y in cells[(task,stratum)]])
            draws.append(differences[rng.integers(len(differences), size=(10000,len(differences)))].mean(axis=1))
        interval = np.quantile(np.mean(draws, axis=0), [.025,.975]).tolist()
        macro[stratum]['paired_bootstrap_95_interval'] = interval
        macro[stratum]['interval_scope'] = 'Resample paired rows within each fixed task; exploratory uncertainty within the selected pool, not unseen-model generalization.'
    output = dict(status='COMPLETE_TASK_TRANSFER', contract=contract, runs=provenance,
        rows_per_arm=len(details), by_stratum=summary, task_equal_macro=macro, rows=details,
        scope='Frozen untruncated in-range LongBench subset. Natural token-F1, not exact-string/EOS capability or all LongBench. Within-native-length compares two S4 deployments, not Native original-table retention.')
    args.out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(rows_per_arm=len(details), task_equal_macro=macro, by_stratum=summary),indent=2))


if __name__ == '__main__':
    main()
