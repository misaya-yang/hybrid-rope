"""Complete the shared48 DEV from source-preserving existing and new material."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .prepare import family_material, counterfactual_user, render_chat, NOISE
from .run import write_json


def rows(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
def sha(value):return hashlib.sha256(value.encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for name in ('model','public','broad','calibration','qa-candidates','hop-candidates','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    result=[]
    for r in rows(args.public):
        if r['split']=='dev' and r['length_cap']==16384 and r['task'] in ('niah_multiquery','niah_multikey_1') and int(r['row_id'].split('_')[-1])<6:
            result.append({**r,'ten_family':'retrieval','source_unit':r['family_id']})
    broad=rows(args.broad)
    for r in broad:
        if r['split']!='dev' or len(r['prompt_ids'])<=4096:continue
        if r['task']=='natural_qasper':category='natural_qa'
        elif r['task'] in ('natural_hotpotqa','natural_2wikimqa'):category='multi_hop'
        else:continue
        result.append({**r,'ten_family':category,'source_unit':r['material_cluster_id']})
    used={r.get('context_sha256') for r in broad}|{r['context_sha256'] for r in rows(args.calibration)}
    used_docs={d for r in broad for d in r.get('constituent_doc_ids',[])}
    for path,category,task in ((args.qa_candidates,'natural_qa','natural_qasper'),(args.hop_candidates,'multi_hop','natural_2wikimqa')):
        count=sum(r['ten_family']==category for r in result)
        for r in rows(path):
            if count>=12:break
            context_hash=r['context_sha256']
            docs=set(r.get('constituent_doc_ids',[]))
            if context_hash in used or docs&used_docs:continue
            text='Read and remember the complete context below.\n\nCONTEXT\n'+r['context']+'\nEND CONTEXT\n\nQuestion: '+r['input']+'\nAnswer concisely using only the context. Output only the answer, then stop.'
            prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
            ids=tok.encode(prompt,add_special_tokens=False)
            if not 4096<len(ids)<=16128:continue
            row_id='pc2ten_'+task+'_dev_'+context_hash[:16]
            result.append({'row_id':row_id,'task':task,'suite':'pc2_ten_directions_v1','split':'dev',
                           'family_id':'context:'+context_hash,'material_cluster_id':'context:'+context_hash,
                           'source_unit':'context:'+context_hash,'context_sha256':context_hash,'constituent_doc_ids':sorted(docs),
                           'raw_context':r['context'],'raw_question':r['input'],'prompt':prompt,'prompt_ids':ids,
                           'input_tokens':len(ids),'max_new_tokens':256,'length_cap':16384,'expected':None,
                           'references':r['answers'],'score_contract':'longbench_qa_f1_context_first_v1','ten_family':category,
                           'source':{'source_id':r.get('_id'),'context_sha256':context_hash,'unedited_complete_context':True},
                           'background':'unaltered_natural_document'})
            used.add(context_hash);used_docs|=docs;count+=1
        if count<12:raise ValueError(f'{category}: only {count} independent intact documents; do not pad/truncate')
    for i in range(12):
        material=family_material('dev',i)
        length=8192 if i<6 else 16384
        query=0 if i%2==0 else 3
        def candidate(n):
            text,expected,records=counterfactual_user(material,query,0,n)
            prompt=render_chat(tok,text)
            return prompt,expected,records,tok.encode(prompt,add_special_tokens=False)
        low,high,best=0,length//len(tok.encode(NOISE+'\n',add_special_tokens=False))+2,0
        while low<=high:
            mid=(low+high)//2
            if len(candidate(mid)[3])+32<=length:best=mid;low=mid+1
            else:high=mid-1
        prompt,expected,records,ids=candidate(best)
        row_id=f'pc2ten_occurrence_dev_{i:03d}'
        result.append({'row_id':row_id,'task':'repeat_key_first_latest','suite':'pc2_ten_directions_v1',
                       'split':'dev','family_id':f'counterfactual_dev_{i:03d}',
                       'material_cluster_id':f"dev:{material['seed']}",'source_unit':f"dev:{material['seed']}",
                       'prompt':prompt,'prompt_ids':ids,'input_tokens':len(ids),'length_cap':length,'max_new_tokens':32,
                       'expected':expected,'references':[expected],'score_contract':'literal_full_string_plus_terminal_eos_v1',
                       'query_ordinal':query+1,'records':records,'seed':material['seed'],'ten_family':'occurrence',
                       'background':'repeated_sentences','source':{'generator':'existing counterfactual_user/family_material',
                       'adaptation':'one predetermined first/latest member per independent family; no factorial-interaction claim'}})
    counts=Counter(r['ten_family'] for r in result)
    if len(result)!=48 or set(counts.values())!={12}:raise ValueError(f'bad common panel counts: {counts}')
    if len({r['source_unit'] for r in result})!=48:raise ValueError('duplicate source units in common panel')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result)
    path=root/'rows.jsonl'
    if path.exists() and path.read_text()!=data:raise ValueError('refuse to change frozen common panel')
    path.write_text(data)
    write_json(root/'manifest.json',{'rows':48,'counts':dict(counts),'rows_sha256':sha(data),
        'sources':{name:hashlib.sha256(Path(getattr(args,name.replace('-','_'))).read_bytes()).hexdigest()
                   for name in ('public','broad','calibration','qa-candidates','hop-candidates')},
        'independence':'synthetic family units conditional on fixed generator/background; natural documents/constituent titles deduplicated',
        'lengths':'actual full texts; controlled occurrence has6 at8K and6 at16K; natural text never padded/truncated',
        'status':'DEV; includes earlier observed retrieval/natural rows; never a fresh confirmation set',
        'row_ids':[r['row_id'] for r in result]})
    print(json.dumps({'status':'PREPARED','rows':48,'counts':dict(counts),'tokens':[len(r['prompt_ids']) for r in result]}),flush=True)


if __name__=='__main__':main()
