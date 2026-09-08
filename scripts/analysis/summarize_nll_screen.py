"""Verify paired teacher-forced NLL receipts and summarize document-level deltas."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.experiments.olmo_fast_screen.prepare import sha_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=a.run
    status=json.loads((r/'status.json').read_text())
    if status['status']!='COMPLETE':
        raise ValueError('incomplete NLL run')
    runtime=json.loads((r/'runtime.json').read_text())
    all_rows={};files={};summary={}
    for method in ('Native','MrPro','MrProBM'):
        raw=r/(method+'.jsonl');receipt=json.loads((r/(method+'.json')).read_text())
        files[method]=sha_file(raw)
        if receipt['status']!='COMPLETE' or files[method]!=receipt['raw_sha256']:
            raise ValueError('NLL receipt mismatch')
        rows={}
        for x in map(json.loads,raw.read_text().splitlines()):
            key=(x['doc'],x['length'])
            if key in rows or x['method']!=method:
                raise ValueError('duplicate or misassigned NLL row')
            if len(x['token_nll'])!=512 or len(x['target_ids'])!=512:
                raise ValueError('unexpected tail token count')
            if not np.isfinite(x['token_nll']).all() or abs(np.mean(x['token_nll'])-x['nll'])>1e-5:
                raise ValueError('invalid token NLL or inconsistent mean')
            rows[key]=x
        all_rows[method]=rows
        for length in sorted({k[1] for k in rows}):
            values=[x['nll'] for k,x in rows.items() if k[1]==length]
            summary.setdefault(str(length),{})[method]=dict(n=len(values),mean_nll=float(np.mean(values)))
    if all_rows['MrPro'].keys()!=all_rows['MrProBM'].keys():
        raise ValueError('unmatched paired NLL rows')
    for method,rows in all_rows.items():
        for key,x in rows.items():
            reference=all_rows['MrPro'][key]
            if x['input_sha256']!=reference['input_sha256'] or x['target_ids']!=reference['target_ids']:
                raise ValueError('NLL input or target identity differs')
    rng=np.random.default_rng(20260908)
    details=[]
    for length,cell in summary.items():
        keys=[k for k in all_rows['MrPro'] if k[1]==int(length)]
        differences=np.asarray([all_rows['MrProBM'][k]['nll']-all_rows['MrPro'][k]['nll'] for k in keys])
        bootstrap=differences[rng.integers(len(keys),size=(10000,len(keys)))].mean(axis=1)
        cell['BM_minus_MrPro']=dict(mean=float(differences.mean()),
            lower_nll_docs=int((differences<0).sum()),higher_nll_docs=int((differences>0).sum()),
            paired_bootstrap_95_interval=np.quantile(bootstrap,[.025,.975]).tolist())
        details.extend(dict(doc=k[0],length=k[1],MrPro=all_rows['MrPro'][k]['nll'],
            BM=all_rows['MrProBM'][k]['nll'],delta=float(d)) for k,d in zip(keys,differences))
    result=dict(status='COMPLETE',summary=summary,rows=details,runtime=runtime,cost=status,
        raw_files_sha256=files,scope='16 frozen natural prefixes, final 512 next-token NLL per length. Document-paired exploratory intervals; not full corpus, whole-string generation or long retrieval capability.')
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
