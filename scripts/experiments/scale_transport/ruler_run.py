"""One frozen own-method RULER subset, with exact upstream metric semantics."""
import argparse,hashlib,importlib.util,json,time,math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
from scripts.experiments.cross_audit.runtime import cuda_runtime
from scripts.experiments.cross_audit.tables import tensor_sha
from scripts.experiments.scale_transport.run import install,write


def main():
    p=argparse.ArgumentParser()
    for k in ['root','prepared','table','upstream','out']:p.add_argument('--'+k,required=True)
    p.add_argument('--budget-seconds',type=int,required=True);a=p.parse_args()
    root=Path(a.root);data=Path(a.prepared);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();deadline=start+a.budget_seconds
    raw=(data/'rows.jsonl').read_bytes();dm=json.loads((data/'manifest.json').read_text());assert hashlib.sha256(raw).hexdigest()==dm['rows_sha256'];rows=[json.loads(l) for l in raw.splitlines()];assert len(rows)==dm['rows']==30
    table=json.loads(Path(a.table).read_text())['candidate'];values=np.asarray(table['values_float32'],dtype=np.float32);assert tensor_sha(values)==table['tensor_sha256'];gain=table['gain']
    ready=json.loads((root/'model_ready.json').read_text());assert ready['status']=='COMPLETE'
    spec=importlib.util.spec_from_file_location('ruler_eval_constants',Path(a.upstream)/'scripts/eval/synthetic/constants.py');metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    hardware=cuda_runtime();write(out/'progress.json',dict(stage='loading',elapsed_seconds=time.monotonic()-start))
    model=AutoModelForCausalLM.from_pretrained(root/'model',local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval();tok=AutoTokenizer.from_pretrained(root/'model',local_files_only=True)
    install(model,values,gain);eos=tok.eos_token_id;gc=GenerationConfig(do_sample=False,num_beams=1,use_cache=True,eos_token_id=eos,pad_token_id=tok.pad_token_id or eos)
    write(out/'deployment.json',dict(name=table['name'],tensor_sha256=tensor_sha(values),values_float32=values.tolist(),gain=gain,source_revision=ready['revision'],source_table_json_sha256=hashlib.sha256(Path(a.table).read_bytes()).hexdigest(),upstream=dm['upstream_commit'],scope='3-task development subset, not full RULER macro'))
    records=[];skipped=[];qualified={}
    with (out/'examples.jsonl').open('x') as f,torch.inference_mode():
        for row in rows:
            if time.monotonic()>deadline:raise TimeoutError('bounded RULER run')
            if row['length_cap']>4096 and not qualified.get(row['task'],False):skipped.append(row['row_id']);continue
            if not .95*row['length_cap']<=len(row['ids']) or len(row['ids'])+row['budget']>row['length_cap']:raise ValueError('prompt/reserve length contract')
            event=dict(stage='generation',task=row['task'],row=row['row_id'],input_tokens=len(row['ids']),completed=len(records),elapsed_seconds=time.monotonic()-start);write(out/'progress.json',event);print(json.dumps(event),flush=True)
            ids=torch.tensor(row['ids'],device='cuda')[None,:];before=time.monotonic();torch.cuda.reset_peak_memory_stats()
            result=model.generate(ids,generation_config=gc,max_new_tokens=row['budget'],logits_to_keep=1)
            new=result[0,ids.shape[1]:].tolist();text=tok.decode(new,skip_special_tokens=True);ended=bool(new and new[-1]==eos)
            if tensor_sha(model.model.rotary_emb.inv_freq.cpu().numpy())!=table['tensor_sha256'] or model.model.rotary_emb.attention_scaling!=gain:raise RuntimeError('static operator drift')
            score=metrics.string_match_all([text],[row['references']]);complete=all(ref.lower() in text.lower() for ref in row['references'])
            rec={k:row[k] for k in ['row_id','task','length_cap','input_tokens','budget','references','prompt_sha256']};rec.update(output_text=text,generated_ids=new,eos=ended,official_score=score,all_answers=complete,seconds=time.monotonic()-before,peak_bytes=torch.cuda.max_memory_allocated())
            f.write(json.dumps(rec)+'\n');f.flush();records.append(rec)
            if row['length_cap']==4096:qualified[row['task']]=qualified.get(row['task'],False) or (complete and ended)
    summary={}
    for task in dm['tasks']:
        summary[task]={}
        for L in [4096,131072]:
            rs=[r for r in records if r['task']==task and r['length_cap']==L]
            summary[task][str(L)]=dict(rows=len(rs),official_score=metrics.string_match_all([r['output_text'] for r in rs],[r['references'] for r in rs]) if rs else None,eos=sum(r['eos'] for r in rs),all_answers=sum(r['all_answers'] for r in rs),status='COMPLETE' if rs else 'SKIPPED_UNRESOLVED_SHORT_CONTROL')
    status='COMPLETE' if not skipped else 'PARTIAL_UNRESOLVED_CONTROLS'
    write(out/'manifest.json',dict(status=status,scope='one own-method 3-task RULER subset; not a 13-task leaderboard score',summary=summary,qualified=qualified,skipped_rows=skipped,rows=len(records),hardware=hardware,elapsed_seconds=time.monotonic()-start,prepared_manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),examples_sha256=hashlib.sha256((out/'examples.jsonl').read_bytes()).hexdigest()))
if __name__=='__main__':main()
