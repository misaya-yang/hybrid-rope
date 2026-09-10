"""Same-clock expanded kernel: isolate E10 arithmetic from frequency mixture."""
import json
import numpy as np
import torch
from .worker import read_rows,save,sha


def run(worker,job):
    table=worker.tables['MrPro'];spec=dict(operator='dual_frequency',table=table,second_table=table)
    folder=worker.root/'numerical_controls/E10_same_clock';folder.mkdir(parents=True,exist_ok=True)
    save(folder/'contract.json',dict(spec=spec,source_sha256=sha(__file__)))
    worker.apply(spec);records=read_rows(folder/'nll.jsonl');done={(r['doc'],r['length']) for r in records}
    candidate={(r['doc'],r['length']):r for r in read_rows(worker.root/'results/E10_dual_frequency/nll.jsonl')}
    with torch.inference_mode():
        for length in (8192,32768):
            for doc in worker.nll_manifest['docs'][:4]:
                key=(doc['file'],length)
                if key in done:continue
                data=np.load(worker.nll_inputs/doc['file'])
                ids=torch.tensor(data[:length].astype(np.int64),device='cuda')[None]
                target=torch.tensor(data[length-511:length+1].astype(np.int64),device='cuda')
                logits=worker.model(ids,use_cache=False,logits_to_keep=512).logits[0].float()
                loss=torch.nn.functional.cross_entropy(logits,target,reduction='none')
                r=dict(doc=doc['file'],length=length,nll=loss.mean().item(),token_nll=loss.tolist(),
                    baseline_nll=worker.nll_base[key]['nll'],E10_nll=candidate[key]['nll'],input_sha256=doc['sha256'])
                with (folder/'nll.jsonl').open('a') as stream:stream.write(json.dumps(r)+'\n')
                records.append(r);del ids,target,logits,loss
    summary={str(length):dict(n=sum(r['length']==length for r in records),
        expanded_Mr_minus_stock_Mr=float(np.mean([r['nll']-r['baseline_nll'] for r in records if r['length']==length])),
        E10_minus_expanded_Mr=float(np.mean([r['E10_nll']-r['nll'] for r in records if r['length']==length]))) for length in (8192,32768)}
    result=dict(status='COMPLETE',by_length=summary,scope='Same frequency function, expanded BF16 execution; four old development documents, not task generalization')
    save(folder/'summary.json',result);worker.apply({'table':table})
    return result
