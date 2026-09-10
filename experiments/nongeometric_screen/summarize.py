"""Compact actual-output summaries; do not rank incomplete panels as victories."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def main(root):
    root=Path(root);records=[]
    retired_path=root/'selection/retired_arms.json'
    retired=json.loads(retired_path.read_text()) if retired_path.exists() else {}
    for folder in sorted((root/'results').iterdir()):
        rr=rows(folder/'ruler.jsonl');nn=rows(folder/'nll.jsonl')
        if not rr:continue
        changes=defaultdict(list)
        for r in rr:changes[(r['length_cap'],r['task'])].append(r['correct']-r['baseline_correct'])
        cells={str(cap):{task:statistics.mean(v) for (length,task),v in changes.items() if length==cap}
               for cap in sorted({k[0] for k in changes})}
        nll={str(length):dict(n=sum(r['length']==length for r in nn),
              delta=statistics.mean(r['nll']-r['baseline_nll'] for r in nn if r['length']==length))
             for length in sorted({r['length'] for r in nn})}
        changed=[dict(row_id=r['row_id'],delta=r['correct']-r['baseline_correct'],output=r['output_text'],ended_eos=r['ended_eos']) for r in rr if r['correct']!=r['baseline_correct']]
        records.append(dict(method=folder.name,rows=len(rr),complete_original_panel=len(rr)==36,
            retired_reason=retired.get(folder.name),
            macro_delta={cap:statistics.mean(values.values()) for cap,values in cells.items()},task_deltas=cells,nll=nll,
            wins=sum(r['correct']>r['baseline_correct'] for r in rr),losses=sum(r['correct']<r['baseline_correct'] for r in rr),
            changed_rows=changed))
    lines=['# Current development results','',
        'Each row is compared with its identical historical MrPro input. Partial rows are progress, not final comparisons. No independent confirmation is implied.','',
        '| Method | Rows | NLL delta 8K / 16K / 32K | RULER delta 32K / 128K (pp) | W / L |',
        '|---|---:|---|---|---|']
    for r in records:
        if r['retired_reason']:continue
        ns=' / '.join(f"{r['nll'][str(L)]['delta']:+.5f}" if str(L) in r['nll'] else '—' for L in (8192,16384,32768))
        rs=' / '.join(f"{100*r['macro_delta'][str(L)]:+.2f}" if str(L) in r['macro_delta'] else '—' for L in (32768,131072))
        lines.append(f"| {r['method']} | {r['rows']} | {ns} | {rs} | {r['wins']} / {r['losses']} |")
    if retired:
        lines.extend(['','Retired construction attempts (measured rows retained in JSON):',''])
        lines.extend(f'- {name}: {reason}' for name,reason in sorted(retired.items()))
    (root/'development_summary.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
    (root/'development_summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
