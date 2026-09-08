"""Native-checkpoint task qualification: access, chronology, complete answer + EOS.
No frequency change, sparse mask insertion, adaptation, or causal mechanism claim.
"""
import argparse,hashlib,json,os,random,time
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText,AutoModelForCausalLM,AutoConfig,AutoTokenizer

COLORS=['amber','violet','silver','orange','green','purple','yellow','blue']
def make_rows():
    rng=random.Random(20260908);rows=[]
    for family in range(4):
        c1,c2=rng.sample(COLORS,2);unit=f'ZX{family+31}'
        events=[f'[M7] Unit {unit} was assigned code {c1}.',f'[P2] Unit {unit} was assigned code {c2}.']
        distractors=[f'[D{i}] Unit AUX{i} was assigned code {rng.choice(COLORS)}.' for i in range(12)]
        for reverse in (False,True):
            pair=events[::-1] if reverse else events
            codes=[c2,c1] if reverse else [c1,c2]
            body='\n'.join(distractors[:3]+[pair[0]]+distractors[3:9]+[pair[1]]+distractors[9:])
            questions=[('marker_M7','What code is stated in record [M7]?',c1),
                ('marker_P2','What code is stated in record [P2]?',c2),
                ('first',f'What was the first code assigned to unit {unit}?',codes[0]),
                ('current',f'What is the current code for unit {unit}?',codes[1]),
                ('history',f'List the two codes assigned to unit {unit} in chronological order, separated by a comma.',', '.join(codes))]
            for task,q,answer in questions:
                prompt=('The event log below is in chronological order, from earliest to latest. A new assignment replaces the previous code of the same unit. '
                  'Record labels are identifiers, not timestamps. Use only the log. Answer with the code alone, or the requested comma-separated list, and then stop.\n\n'
                  +body+'\n\nQuestion: '+q+'\nAnswer:')
                rows.append(dict(row_id=f'f{family}_r{int(reverse)}_{task}',family=family,reverse=reverse,task=task,
                    prompt=prompt,expected=answer,body_sha256=hashlib.sha256(body.encode()).hexdigest()))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inputs-jsonl',type=Path);p.add_argument('--background-shard',type=Path);p.add_argument('--context-tokens',type=int,default=16384)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'outputs.jsonl').exists():raise FileExistsError('Do not overwrite result rows')
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    config=AutoConfig.from_pretrained(a.model)
    loader=AutoModelForImageTextToText if config.model_type=='qwen3_5' else AutoModelForCausalLM
    model=loader.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False)
    if a.inputs_jsonl:
        rows=[json.loads(x) for x in a.inputs_jsonl.read_text().splitlines()]
    elif a.background_shard:
        from long_cases import prepare
        rows=prepare(tokenizer,a.background_shard,a.context_tokens)
    else:rows=make_rows()
    eos=model.generation_config.eos_token_id
    if eos is None:eos=tokenizer.eos_token_id
    eos_ids=set(eos if isinstance(eos,list) else [eos])
    meta=dict(status='RUNNING',pid=os.getpid(),model=str(a.model),planned_rows=len(rows),eos_ids=sorted(eos_ids),
        parameters=sum(x.numel() for x in model.parameters()),torch=str(torch.__version__),
        model_revision=getattr(config,'_commit_hash',None),model_config_sha256=hashlib.sha256((a.model/'config.json').read_bytes()).hexdigest(),max_new_tokens=48,do_sample=False,enable_thinking=False,
        evidence_scope='Native checkpoint task qualification only; no positional ablation')
    (a.output/'status.json').write_text(json.dumps(meta,indent=2))
    (a.output/'inputs.jsonl').write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
    started=time.monotonic()
    with open(a.output/'outputs.jsonl','x') as f,torch.inference_mode():
        for row in rows:
            prompt=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs=tokenizer(prompt,return_tensors='pt').to('cuda')
            if inputs['input_ids'].shape[-1]>(a.context_tokens if (a.background_shard or a.inputs_jsonl) else 1024):raise ValueError('Registered input budget exceeded')
            tick=time.monotonic()
            ids=model.generate(**inputs,max_new_tokens=48,do_sample=False,eos_token_id=eos,pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)[0,inputs['input_ids'].shape[-1]:].tolist()
            decoded=tokenizer.decode(ids,skip_special_tokens=True)
            strict=decoded.strip()==row['expected']; canonical=decoded.strip().strip('"\'').rstrip('.!').strip().lower()==row['expected']
            result={**row,'input_tokens':inputs['input_ids'].shape[-1],
                'input_sha256':hashlib.sha256(inputs['input_ids'].cpu().numpy().astype('<i8').tobytes()).hexdigest(),
                'generated_ids':ids,'output_text':decoded,'text_exact':strict,'canonical_answer_match':canonical,
                'ended_eos':bool(ids and ids[-1] in eos_ids),'seconds':time.monotonic()-tick}
            result['full_exact_and_eos']=strict and result['ended_eos']
            # Whole output grammar, frozen after short-format qualification; no substring matching.
            import re
            word='(?:'+ '|'.join(COLORS)+')'
            grammar=r'\s*(?:code\s+)?('+word+r')(?:\s*,\s*(?:code\s+)?('+word+r'))?\s*[.!]?\s*'
            match=re.fullmatch(grammar,decoded.lower())
            result['whole_answer_payload_match']=bool(match and ', '.join(x for x in match.groups() if x)==row['expected'])
            f.write(json.dumps(result)+'\n');f.flush()
            print(json.dumps({k:result[k] for k in ('row_id','output_text','text_exact','ended_eos','seconds')}),flush=True)
    meta.update(status='COMPLETE',seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated())
    (a.output/'status.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
