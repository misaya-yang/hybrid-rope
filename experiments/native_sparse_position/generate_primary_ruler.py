"""Frozen RULER multiquery assay using existing upstream/data and native Qwen tokenizer."""
import argparse,hashlib,json,os,runpy,subprocess,sys
from pathlib import Path
from transformers import AutoTokenizer

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);t=AutoTokenizer.from_pretrained(a.model)
    base=runpy.run_path(str(a.upstream/'scripts/data/synthetic/constants.py'))['TASKS']['niah']['template']
    base=base.replace('\nWhat are all', '\nReturn exactly the requested numbers, in query order, separated by single spaces. Do not add labels, punctuation or explanations.\nWhat are all')
    template=t.apply_chat_template([{'role':'user','content':base}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    receipts=[]
    for split,cap,n,seed in [('compact_dev',2048,8,20260910),('frozen_long',32768,32,20260911)]:
        command=[sys.executable,str(a.upstream/'scripts/data/synthetic/niah.py'),'--save_dir',str(a.output/split),'--save_name','niah_multiquery','--subset','validation',
          '--tokenizer_path',a.model,'--tokenizer_type','hf','--max_seq_length',str(cap),'--tokens_to_generate','128','--num_samples',str(n),'--random_seed',str(seed),'--template',template,
          '--type_haystack','essay','--type_needle_k','words','--type_needle_v','numbers','--num_needle_k','1','--num_needle_v','1','--num_needle_q','4']
        subprocess.run(command,check=True,cwd=a.upstream)
        files=list((a.output/split).rglob('validation.jsonl'))
        if len(files)!=1:raise ValueError(f'Unexpected source output: {files}')
        raw=files[0].read_text();rows=[]
        for i,r in enumerate(map(json.loads,raw.splitlines())):
            # Upstream strips the completed template, including the native
            # assistant-header newline. Restore that structural delimiter.
            prompt=r['input']
            if prompt.endswith('<|im_start|>assistant'):
                prompt+='\n'
            if not prompt.endswith('<|im_start|>assistant\n'):
                raise ValueError('Incomplete native assistant header')
            ids=t(prompt,add_special_tokens=False)['input_ids']
            if len(ids)+128>cap:raise ValueError('Native cap contract changed')
            rows.append({'row_id':f'{split}_multiquery_{i}','task':'niah_multiquery','family':i,'prompt_ids':ids,'input_tokens':len(ids),
              'references':r['outputs'],'expected':' '.join(r['outputs']),'max_new_tokens':128,'length_cap':cap,'upstream_index':r.get('index',i),
              'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
        (a.output/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        receipts.append({'split':split,'seed':seed,'cap':cap,'rows':len(rows),'source_sha256':hashlib.sha256(raw.encode()).hexdigest()})
    (a.output/'manifest.json').write_text(json.dumps({'source_revision':'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a','model':a.model,'format':'Exactly four numerical strings in query order, single spaces, then EOS; no answer prefix or constrained decoding',
      'scope':'RULER multiquery generator with declared output-format adaptation; not full official RULER','receipts':receipts},indent=2));print(json.dumps(receipts),flush=True)
if __name__=='__main__':main()
