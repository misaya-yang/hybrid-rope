"""E09 shared bases from pooled within-block K covariance in two frames."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from .exact_probe import ExactBlockSelector
from .runtime import AttentionSettings,NosaReferenceForCausalLM,apply_rope
from .run import write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    data=Path(args.data);rows=[json.loads(l) for l in data.read_text().splitlines()]
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.52)
    global_moments,local_moments,counts,done={},{},{},{}
    def observe(context,selected):
        h,length,d=context.k.shape;layer=context.layer_idx;full=length//64
        begin=done.get(layer,0)
        if full==begin:return
        x=context.k[:,begin*64:full*64].float().reshape(h,full-begin,64,d)
        w=context.cis[:,begin*64:full*64].float().reshape(h,full-begin,64).softmax(-1)
        mean=(w[...,None]*x).sum(-2)
        centered=(x-mean[:,:,None])*w.sqrt()[...,None]
        flat=centered.flatten(1,2)
        g=flat.transpose(1,2)@flat
        origins=torch.arange(begin,full,device=x.device).repeat_interleave(64)*64
        aligned=apply_rope(flat,-origins,context.rope_inv_freq,1.)
        l=aligned.transpose(1,2)@aligned
        global_moments[layer]=global_moments.get(layer,torch.zeros_like(g,dtype=torch.float64))+g.double()
        local_moments[layer]=local_moments.get(layer,torch.zeros_like(l,dtype=torch.float64))+l.double()
        counts[layer]=counts.get(layer,0)+full-begin;done[layer]=full
    status={'status':'LOADING','pid':os.getpid(),'completed_documents':0,'total_documents':len(rows)}
    write_json(root/'status.json',status);started=time.time()
    model=NosaReferenceForCausalLM.from_pretrained(args.model,device='cuda',dtype=torch.bfloat16,
        selector=ExactBlockSelector('exact_mass'),settings=AttentionSettings(topk=64,select_blocks=16,attention_query_chunk_size=2048),trace_callback=observe)
    try:
        for row in rows:
            done.clear();status.update(status='RUNNING',row_id=row['row_id']);write_json(root/'status.json',status)
            before=time.time();output=model.prefill(torch.tensor([row['prompt_ids']],device='cuda'),chunk_size=2048)
            if not torch.isfinite(output.logits).all():raise FloatingPointError('nonfinite key calibration')
            del output;torch.cuda.synchronize();status['completed_documents']+=1
            print(json.dumps({'row_id':row['row_id'],'tokens':len(row['prompt_ids']),'seconds':time.time()-before}),flush=True)
        bases,local_bases,eigenvalues,local_eigenvalues={},{},{},{}
        diagnostics=[]
        for layer in global_moments:
            g,gu=torch.linalg.eigh(global_moments[layer]/counts[layer])
            l,lu=torch.linalg.eigh(local_moments[layer]/counts[layer])
            bases[layer]=gu[:,:,-32:].float().cpu();local_bases[layer]=lu[:,:,-32:].float().cpu()
            eigenvalues[layer]=g.cpu();local_eigenvalues[layer]=l.cpu()
            diagnostics.append({'layer':layer,'global_fraction32':(g[:,-32:].sum(-1)/g.sum(-1)).cpu().tolist(),
                'local_fraction32':(l[:,-32:].sum(-1)/l.sum(-1)).cpu().tolist(),
                'relative_trace_difference':((l.sum(-1)-g.sum(-1)).abs()/g.sum(-1)).cpu().tolist()})
        torch.save({'bases':bases,'local_bases':local_bases,'eigenvalues':eigenvalues,'local_eigenvalues':local_eigenvalues,
                    'source_rows_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'counts':counts,
                    'method':'pooled CIS-weighted within-block K covariance; global versus R(-block_start) frame'},root/'key_basis.pt')
        write_json(root/'spectra.json',{'rows':diagnostics,'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
                   'scope':'same24 independent calibration sources; no answer labels; pooled key spectra, not task success',
                   'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        status.update(status='COMPLETE',seconds=time.time()-started)
    except BaseException as e:
        status.update(status='FAILED',error_type=type(e).__name__,error=str(e));raise
    finally:write_json(root/'status.json',status)


if __name__=='__main__':main()
