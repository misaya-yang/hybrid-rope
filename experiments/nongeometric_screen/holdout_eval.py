"""Evaluate a frozen candidate and one missing reference on a new RULER cohort."""
import json
from pathlib import Path

from .worker import digest,read_rows,save,sha


def run(worker,job):
    from scripts.experiments.olmo_fast_screen.ruler_bench import summarize
    prepared=Path(job['prepared']);manifest=json.loads((prepared/'manifest.json').read_text())
    if manifest['model_id']!=worker.manifest['model_id'] or manifest['revision']!=worker.manifest['revision']:
        raise ValueError('holdout model identity differs')
    if sha(prepared/'generation_config.json')!=sha(worker.prepared/'generation_config.json'):
        raise ValueError('holdout decoding differs from frozen development contract')
    screen=read_rows(prepared/'screen.jsonl')
    if sha(prepared/'screen.jsonl')!=manifest['prepared_files']['screen.jsonl']:
        raise ValueError('holdout input drift')
    old_prompts={r['prompt_sha256'] for r in worker.screen}
    for row in screen:
        if digest(row['prompt_ids'])!=row['prompt_sha256'] or row['prompt_sha256'] in old_prompts:
            raise ValueError('holdout input mismatch or exact development prompt reuse')
    if job.get('tasks'):screen=[r for r in screen if r['task'] in job['tasks']]
    if job.get('lengths'):screen=[r for r in screen if r['length_cap'] in job['lengths']]
    if job.get('per_cell') is not None:screen=[r for r in screen if int(r['row_id'].rsplit('_',1)[1])<job['per_cell']]
    folder=worker.root/'holdout_results'/job['cohort'];folder.mkdir(parents=True,exist_ok=True)
    contract=dict(input_manifest_sha256=sha(prepared/'manifest.json'),decoding_sha256=sha(prepared/'generation_config.json'))
    if (folder/'cohort.json').exists() and json.loads((folder/'cohort.json').read_text())!=contract:
        raise ValueError('cannot silently change a holdout cohort')
    save(folder/'cohort.json',contract)
    summaries={};specs={};all_records={}
    for name in job['methods']:
        spec={'table':worker.tables['MrPro']} if name=='MrPro' else json.loads((worker.root/'results'/name/'contract.json').read_text())['spec']
        path=folder/(name+'_contract.json')
        if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('frozen holdout method changed')
        save(path,spec);specs[name]=spec
        all_records[name]={r['row_id']:r for r in read_rows(folder/(name+'.jsonl'))}
    for row in screen:
        for name in job['methods']:
            existing=all_records[name]
            if row['row_id'] in existing:
                if existing[row['row_id']]['prompt_sha256']!=row['prompt_sha256']:raise ValueError('cached holdout row changed')
                continue
            worker.apply(specs[name])
            result=worker.generate(row)
            with (folder/(name+'.jsonl')).open('a') as stream:stream.write(json.dumps(result)+'\n')
            existing[row['row_id']]=result
            save(worker.root/'live.json',dict(job=job['id'],phase='holdout_ruler',method=name,row=row['row_id'],completed=len(existing),requested=len(screen)))
        print(json.dumps(dict(holdout=job['cohort'],row=row['row_id'],scores={name:all_records[name][row['row_id']]['correct'] for name in job['methods']})),flush=True)
    for name in job['methods']:
        summaries[name]=summarize([all_records[name][r['row_id']] for r in screen])
        save(folder/(job['id']+'__'+name+'_summary.json'),dict(status='COMPLETE',summary=summaries[name],raw_sha256=sha(folder/(name+'.jsonl')),source_sha256=sha(__file__),
            evaluated_row_ids=[r['row_id'] for r in screen],
            scope='Frozen method on new-seed official RULER inputs; task/length subsets are declared in the job'))
    return dict(status='COMPLETE',summaries=summaries)
