"""Freeze a small subset using pinned upstream RULER generators, no GPU."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
import yaml
from transformers import AutoTokenizer

TASKS=('niah_single_1','niah_multikey_3','vt')

def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=False);up=Path(a.upstream);tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    spec=importlib.util.spec_from_file_location('ruler_data_constants',up/'scripts/data/synthetic/constants.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    custom=yaml.safe_load((up/'scripts/synthetic.yaml').read_text());rows=[]
    for length,count in [(4096,2),(131072,8)]:
        for task in TASKS:
            config=custom[task];base=module.TASKS[config['task']]
            template=tok.apply_chat_template([{'role':'user','content':base['template']}],tokenize=False,add_generation_prompt=True)+base.get('answer_prefix','')
            argv=[sys.executable,str(up/f"scripts/data/synthetic/{config['task']}.py"),'--save_dir',str(root/str(length)),'--save_name',task,'--subset','validation','--tokenizer_path',a.model,'--tokenizer_type','hf','--max_seq_length',str(length),'--tokens_to_generate',str(base['tokens_to_generate']),'--num_samples',str(count),'--random_seed','137','--template',template]
            for k,v in config['args'].items():argv+=['--'+k,str(v)]
            with (root/f'{length}_{task}.log').open('x') as log:subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600,env={**os.environ,'TOKENIZERS_PARALLELISM':'false','NLTK_DATA':'/root/autodl-tmp/nltk_data'})
            path=root/str(length)/task/'validation.jsonl';raw=[json.loads(l) for l in path.read_text().splitlines()];assert len(raw)==count
            for i,r in enumerate(raw):
                text=r['input']+r.get('answer_prefix','');ids=tok.encode(text,add_special_tokens=False)
                if len(ids)+base['tokens_to_generate']>length:raise ValueError('token/reserve overflow')
                if len(ids)<.95*length:raise ValueError('unexpectedly underfilled length bucket')
                rows.append(dict(row_id=f'{task}_{length}_{i}',task=task,upstream_index=r['index'],ids=ids,references=r['outputs'],length_cap=length,input_tokens=len(ids),budget=base['tokens_to_generate'],prompt_sha256=hashlib.sha256(text.encode()).hexdigest()))
            print(json.dumps({'prepared':task,'cap':length,'rows':count}),flush=True)
    with (root/'rows.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    receipt={'status':'COMPLETE','scope':'3-task development subset; not 13-task RULER macro','upstream_commit':'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a','tasks':TASKS,'rows':len(rows),'rows_sha256':hashlib.sha256((root/'rows.jsonl').read_bytes()).hexdigest(),'source_files':{str(p.relative_to(up)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (up/'scripts').rglob('*') if p.is_file() and p.suffix in ('.py','.yaml')},'min_max_tokens':[min(r['input_tokens'] for r in rows),max(r['input_tokens'] for r in rows)]}
    (root/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
