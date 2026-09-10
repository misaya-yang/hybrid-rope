"""Apply the declared permissive screen rule without erasing task tradeoffs."""
import json

from .worker import read_rows,save


def run(worker,job):
    decisions=[];eligible=[]
    retired_path=worker.root/'selection/retired_arms.json'
    retired=json.loads(retired_path.read_text()) if retired_path.exists() else {}
    for folder in sorted((worker.root/'results').iterdir()):
        if folder.name in retired:
            decisions.append(dict(method=folder.name,decision='RETIRED_SELECTION_ERROR',reason=retired[folder.name]));continue
        summary_path=folder/'summary.json'
        if not summary_path.exists():continue
        summary=json.loads(summary_path.read_text())
        if summary['candidate']['rows']>=36:
            decisions.append(dict(method=folder.name,decision='ALREADY_COMPLETE_36'));continue
        nll=read_rows(folder/'nll.jsonl')
        two_bad=all(summary['nll_by_length'].get(str(length),{}).get('delta',0)>.15 for length in (8192,32768))
        majority_bad=sum(r['nll']>r['baseline_nll'] for r in nll)>len(nll)/2
        rows=read_rows(folder/'ruler.jsonl')
        lost_tasks={r['task'] for r in rows if r['correct']<r['baseline_correct']}
        collapse=two_bad and majority_bad and summary['wins']==0 and len(lost_tasks)>=2
        if collapse:
            decisions.append(dict(method=folder.name,decision='STOP_FIXED_VERSION_BROAD_DEGRADATION',summary=summary));continue
        cap='131072';delta=summary['candidate']['by_length'][cap]['macro_accuracy']-summary['baseline']['by_length'][cap]['macro_accuracy']
        eligible.append((delta,folder.name,json.loads((folder/'contract.json').read_text())['spec']))
    for index,(delta,name,spec) in enumerate(sorted(eligible,reverse=True),60):
        save(worker.root/'queue'/f'{index:03d}_{name}_full.json',dict(id=name,spec=spec,panel='full',nll_docs=16,nll_lengths=[8192,16384,32768]))
        decisions.append(dict(method=name,decision='COMPLETE_REMAINING_HISTORICAL_ROWS',small_long_macro_delta=delta,
            reason='No broad-collapse criterion; mild NLL/passkey losses or score ties do not rule out mixed downstream gains.'))
    save(worker.root/'screen_decisions.json',dict(decisions=decisions,scope='Investment decisions, not claims of method success'))
    return dict(status='COMPLETE',decisions=decisions)
