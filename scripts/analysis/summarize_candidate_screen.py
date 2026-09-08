"""Compare one completed candidate-only run against an archived matched reference."""
import argparse
import json
from pathlib import Path

from scripts.experiments.olmo_fast_screen.prepare import sha_file
from scripts.experiments.olmo_fast_screen.ruler_bench import score, summarize, verdict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--method', required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--frozen-panel', action='store_true', help='Report all runtime-frozen tasks without the legacy six-task gate')
    args = p.parse_args()
    run = args.run
    status = json.loads((run/'status.json').read_text())
    receipt = json.loads((run/(args.method+'.json')).read_text())
    if status['status'] != 'COMPLETE' or receipt['status'] != 'COMPLETE':
        raise ValueError('candidate run is incomplete')
    raw = run/(args.method+'.jsonl')
    if sha_file(raw) != receipt['raw_sha256']:
        raise ValueError('candidate raw checksum differs')
    runtime = json.loads((run/'runtime.json').read_text())
    candidate = list(map(json.loads, raw.read_text().splitlines()))
    baseline_all = list(map(json.loads, args.baseline.read_text().splitlines()))
    reference = {r['row_id']: r for r in baseline_all}
    if len(reference) != len(baseline_all):
        raise ValueError('duplicate archived baseline rows')
    if [r['row_id'] for r in candidate] != receipt['row_ids'] or receipt['row_ids'] != runtime['row_ids']:
        raise ValueError('candidate row order differs from frozen runtime')
    baseline = [reference[r['row_id']] for r in candidate]
    eos = runtime['generation_config']['eos_token_id']
    eos = {eos} if isinstance(eos, int) else set(eos)
    for c, b in zip(candidate, baseline):
        for field in ('prompt_sha256', 'references', 'task', 'length_cap', 'input_tokens'):
            if c[field] != b[field]:
                raise ValueError('candidate and reference identity differs: '+field)
        if abs(score(c, c['output_text'])-c['correct']) > 1e-12:
            raise ValueError('candidate score differs on recomputation')
        if abs(score(b, b['output_text'])-b['correct']) > 1e-12:
            raise ValueError('baseline score differs on recomputation')
        if c['ended_eos'] != (c['generated_ids'][-1] in eos):
            raise ValueError('candidate termination mismatch')
        if len(c['generated_ids']) > c['max_new_tokens']:
            raise ValueError('candidate exceeded frozen budget')
    if args.frozen_panel:
        if len({r['row_id'] for r in candidate}) != len(candidate):
            raise ValueError('duplicate candidate rows')
        c, b = summarize(candidate), summarize(baseline)
        result = dict(status='COMPLETE_PAIRED_PANEL', candidate=c, baseline=b,
            macro_delta_by_length={cap:cell['macro_accuracy']-b['by_length'][cap]['macro_accuracy']
                for cap,cell in c['by_length'].items()},
            paired_wins=sum(x['correct']>y['correct'] for x,y in zip(candidate,baseline)),
            paired_losses=sum(x['correct']<y['correct'] for x,y in zip(candidate,baseline)))
    else:
        result = verdict(candidate, baseline)
    details = []
    for c, b in zip(candidate, baseline):
        details.append(dict(row_id=c['row_id'], prompt_sha256=c['prompt_sha256'],
            candidate_score=c['correct'], baseline_score=b['correct'],
            delta=c['correct']-b['correct'], candidate_output=c['output_text'],
            baseline_output=b['output_text'], candidate_ids=c['generated_ids'],
            baseline_ids=b['generated_ids'], candidate_eos=c['ended_eos'], baseline_eos=b['ended_eos']))
    report = dict(model_id=runtime['model_id'], revision=runtime['revision'],
        method=args.method, result=result, run_status=status,
        baseline_reused=True, files_sha256=dict(candidate=sha_file(raw), baseline=sha_file(args.baseline),
            runtime=sha_file(run/'runtime.json')), rows=details,
        scope='Matched frozen task panel, not full official RULER. Official recall and complete-string/EOS are distinct.')
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], delta=result['macro_delta_by_length'],
        candidate=summarize(candidate)['by_length'], baseline=summarize(baseline)['by_length']), indent=2))


if __name__ == '__main__':
    main()
