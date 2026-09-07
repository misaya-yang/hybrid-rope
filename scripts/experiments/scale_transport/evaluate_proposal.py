"""Actual-generation diagnostic of the already frozen unguarded proposal.

Separate from the REFERENCE_ONLY guarded constructor result. No retuning.
"""
import argparse,hashlib,json,time,math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from scripts.experiments.cross_audit.runtime import cuda_runtime
from scripts.experiments.cross_audit.tables import tensor_sha,transform
from scripts.experiments.scale_transport.run import install,write,GAIN
from scripts.eval.longbench_metrics import qa_f1_score


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--prepared',required=True);p.add_argument('--construction',required=True);p.add_argument('--out',required=True);p.add_argument('--budget-seconds',type=int,required=True);p.add_argument('--variant',choices=['full','tail-only'],default='full');p.add_argument('--native-receipt');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);root=Path(a.root);prepared=Path(a.prepared);source=Path(a.construction)
    started=time.monotonic();deadline=started+a.budget_seconds;hardware=cuda_runtime()
    manifest=json.loads((prepared/'manifest.json').read_text())
    for f,h in manifest['files'].items():
        if hashlib.sha256((prepared/f).read_bytes()).hexdigest()!=h:raise ValueError('data drift')
    proposal_data=json.loads((source/'proposal.json').read_text());arr=np.asarray(proposal_data['proposal'],dtype=np.float32)
    if json.loads((source/'deployment.json').read_text())['status']!='REFERENCE_ONLY':raise ValueError('diagnostic scope requires reference-only constructor')
    model=AutoModelForCausalLM.from_pretrained(root/'model',local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval()
    tok=AutoTokenizer.from_pretrained(root/'model',local_files_only=True);native=model.model.rotary_emb.inv_freq.float().cpu().numpy().copy()
    if tensor_sha(native)!=json.loads((source/'source_identity.json').read_text())['actual_native_sha256']:raise ValueError('source array drift')
    label='ScaleTransportUnguarded'
    if a.variant=='tail-only':
        mr,_,meta=transform(native,dim=128,base=1e6,reference_length=32768,scale=4,method='mrpro')
        assert meta['high']==40
        arr[:40]=mr[:40];label='MrMiddleScaleTail'
    rows=[json.loads(l) for l in (prepared/'eval.jsonl').read_text().splitlines()];eos=tok.eos_token_id
    gc=GenerationConfig(do_sample=False,num_beams=1,use_cache=True,eos_token_id=eos,pad_token_id=tok.pad_token_id or eos)
    native_jobs=[('Native',native,1.,r) for r in rows if r['bucket']=='native']
    native_identity=None
    if a.native_receipt:
        prior=Path(a.native_receipt);receipt=json.loads((prior/'manifest.json').read_text());native_identity=hashlib.sha256((prior/'examples.jsonl').read_bytes()).hexdigest()
        if receipt['status']!='COMPLETE' or native_identity!=receipt['examples_sha256']:raise ValueError('invalid prior Native receipt')
        native_rows=[json.loads(l) for l in (prior/'examples.jsonl').read_text().splitlines() if json.loads(l)['arm']=='Native']
        expected={r['row_id']:(r['context_sha256'],r['input_tokens'],r['references']) for r in rows if r['bucket']=='native'}
        observed={r['row_id']:(r['context_sha256'],r['input_tokens'],r['references']) for r in native_rows}
        if expected!=observed or any(r['table_sha256']!=tensor_sha(native) for r in native_rows):raise ValueError('prior Native row mismatch')
        native_jobs=[]
    jobs=native_jobs+[(label,arr,GAIN,r) for r in rows]
    write(out/'intervention.json',dict(label=label,variant=a.variant,native_receipt_sha256=native_identity,scope='post-construction capability diagnostic; not the guarded method',gain=GAIN,table_sha256=tensor_sha(arr),source_proposal_sha256=hashlib.sha256((source/'proposal.json').read_bytes()).hexdigest(),parameters_retuned=False))
    records=[]
    with (out/'examples.jsonl').open('x') as f,torch.inference_mode():
        for arm,values,gain,row in jobs:
            if time.monotonic()>deadline:raise TimeoutError('diagnostic budget')
            event=dict(stage='generation',arm=arm,row=row['row_id'],completed=len(records),elapsed_seconds=time.monotonic()-started);write(out/'progress.json',event);print(json.dumps(event),flush=True)
            install(model,values,gain);ids=torch.tensor(row['ids'],device='cuda')[None,:];before=time.monotonic();torch.cuda.reset_peak_memory_stats()
            generated=model.generate(ids,generation_config=gc,max_new_tokens=row['budget'],logits_to_keep=1)
            new=generated[0,ids.shape[1]:].tolist();text=tok.decode(new,skip_special_tokens=True)
            if tensor_sha(model.model.rotary_emb.inv_freq.cpu().numpy())!=tensor_sha(values):raise RuntimeError('table drift')
            rec={k:row[k] for k in ['row_id','task','bucket','references','input_tokens','context_sha256']}
            rec.update(arm=arm,output_text=text,generated_ids=new,eos=bool(new and new[-1]==eos),qa_f1=qa_f1_score(text,row['references']),seconds=time.monotonic()-before,peak_bytes=torch.cuda.max_memory_allocated(),table_sha256=tensor_sha(values))
            f.write(json.dumps(rec)+'\n');f.flush();records.append(rec)
    write(out/'manifest.json',dict(status='COMPLETE',scope='post-construction unguarded proposal diagnosis, unchanged data and parameters; not full benchmark or SOTA',rows=len(records),hardware=hardware,elapsed_seconds=time.monotonic()-started,examples_sha256=hashlib.sha256((out/'examples.jsonl').read_bytes()).hexdigest()))
if __name__=='__main__':main()
