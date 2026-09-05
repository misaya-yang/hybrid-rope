#!/usr/bin/env python3
"""Fixed-table, all-linear LoRA with an ORIGINAL-Native full-vocabulary teacher.

Consumes qualified natural transport views and independent Native replay assets.
It deliberately does not manufacture natural counterfactual truth from synthetic
lookup rows. Missing qualified assets block preflight before CUDA is loaded.
"""
from __future__ import annotations
import argparse
import copy
import gc
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.single_table_generation import (
    WEIGHT_SHA, NATIVE, sha, canonical, rows, write_json, tokenizer_identity,
    load_runtime, guard_resources, greedy,checkpoint_contract,
)
from scripts.lib.rope.generation_contract import projection_parameter_count,stable_teacher_kl
from scripts.analysis.export_single_table_controls import P2_SHA

MODULES = ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')
GROUPS = ('text','instruction','reasoning','position_format')


def token_ids(value):
    return isinstance(value,list) and bool(value) and all(type(v) is int and v >= 0 for v in value)


def learning_rate_factor(local_step, count):
    """First warmup step is nonzero; even a one-step smoke actually updates."""
    warmup=max(1,math.ceil(.1*count))
    if local_step < warmup: return (local_step+1)/warmup
    return .5*(1+math.cos(math.pi*(local_step-warmup)/max(1,count-warmup)))


def prefix_lm_positions(row, seed):
    """One frozen uniform subset; predict prompt tokens only, never answer labels."""
    count=len(row['prompt_ids'])-1
    if count<1: raise ValueError('prefix LM needs at least two prompt tokens')
    key=(20260904128,seed,row['semantic_id'],row['world'],row['length_cap'],row['layout'])
    rng=random.Random(int(canonical(key)[:16],16))
    return sorted(rng.sample(range(count),min(128,count)))


def evaluation_rows(data, split, lengths, expected_groups):
    """Check the entire requested C/N/F matrix before any GPU work."""
    selected=[r for r in data if r['split']==split and r['length_cap'] in lengths]
    groups={r['semantic_id'] for r in selected}
    if len(groups)!=expected_groups: raise ValueError('missing evaluation semantic groups')
    expected={(world,layout,length) for length in lengths for world in (0,1)
              for layout in (('compact',) if length==2048 else ('near','far'))}
    for sid in groups:
        actual=[(r['world'],r['layout'],r['length_cap']) for r in selected if r['semantic_id']==sid]
        if len(actual)!=len(expected) or set(actual)!=expected:
            raise ValueError('missing/duplicate evaluation world-layout-length cell')
    return selected


def validate_answer_worlds(data,tokenizer):
    """Different tokenizations of one answer are not different lawful worlds."""
    worlds={}
    for row in data:
        text=tokenizer.decode(row['target_ids'][:-1],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        aliases=set(row.get('accepted_full_answers',[text]))
        if text not in aliases: raise ValueError('canonical truth absent from accepted complete answers')
        key=(row['split'],row['semantic_id'],row['world'])
        if key in worlds and worlds[key]!=aliases: raise ValueError('world truth changes across lengths/layouts')
        worlds[key]=aliases
    for split,sid,_ in worlds:
        if (split,sid,0) not in worlds or (split,sid,1) not in worlds:
            raise ValueError('missing lawful answer world')
        if worlds[(split,sid,0)] & worlds[(split,sid,1)]:
            raise ValueError('counterfactual worlds share a complete accepted answer')


def validate_teacher_cache(args, native):
    if not args.teacher_cache or not (args.teacher_cache/'manifest.json').is_file():
        raise ValueError('BLOCKED_TEACHER_CACHE: complete original-Native cache required')
    cache=json.loads((args.teacher_cache/'manifest.json').read_text())
    if (cache['status']!='ORIGINAL_NATIVE_FULL_VOCAB_CACHE_V1' or cache['pool_sha256']!=sha(args.native_pool)
            or not cache['teacher']['table_is_native'] or cache['teacher']['gain']!=1
            or cache['teacher']['adapter_sha256'] is not None or cache['teacher']['checkpoint_sha256']!=checkpoint_contract(args)['weight_sha256']
            or cache['teacher']['tokenizer_files']!=tokenizer_identity(args.checkpoint)):
        raise ValueError('teacher cache identity drift')
    entries={e['id']:e for e in cache['entries']}
    required={r['id'] for r in native if r['split']!='test'}
    if len(entries)!=len(cache['entries']) or set(entries)!=required:
        raise ValueError('teacher cache incomplete, duplicate, or includes test data')
    vocab=json.loads((args.checkpoint/'config.json').read_text())['vocab_size']
    for row in native:
        if row['split']=='test': continue
        e=entries[row['id']]
        if (e['row_sha256']!=canonical(row) or sha(args.teacher_cache/e['path'])!=e['sha256']
                or e['shape']!=[len(row['prediction_positions']),vocab] or e['dtype']!='float32'):
            raise ValueError('teacher row/cache bytes or shape drift')
    return entries


def read_native_pool(path):
    if not path.is_file(): raise ValueError('BLOCKED_NATIVE_DATA: missing independent replay manifest')
    manifest = json.loads(path.read_text())
    if manifest.get('status') != 'NATIVE_REPLAY_POOL_V1':
        raise ValueError('BLOCKED_NATIVE_DATA: require NATIVE_REPLAY_POOL_V1')
    source = path.parent / manifest['rows_path']
    if sha(source) != manifest['rows_sha256']:
        raise ValueError('Native pool hash drift')
    result = list(rows(source))
    seen, ownership = set(), {}
    for row in result:
        if row['id'] in seen or row['group'] not in GROUPS:
            raise ValueError('duplicate Native row or unknown stratum')
        seen.add(row['id'])
        if row['split'] not in ('train','calibration','validation','test'):
            raise ValueError('unknown Native split')
        source_id = row['source_id']
        if source_id in ownership and ownership[source_id] != row['split']:
            raise ValueError('Native source leakage across splits')
        ownership[source_id] = row['split']
        positions = row['prediction_positions']
        if (not token_ids(row['input_ids']) or not 1 <= len(row['input_ids']) <= NATIVE or not positions
                or len(set(positions)) != len(positions)
                or any(type(p) is not int or p < 0 or p >= len(row['input_ids']) for p in positions)):
            raise ValueError('invalid Native physical sequence/prediction positions')
        if row['split'] in ('validation','test'):
            if row['group']=='text':
                if not row.get('text_domain') or any(p+1>=len(row['input_ids']) for p in positions):
                    raise ValueError('Native text endpoint requires domain and observed next-token labels')
            elif (not token_ids(row.get('prompt_ids')) or row.get('truth_verified') is not True
                  or not row.get('accepted_full_answers')
                  or any(not isinstance(a,str) or not a for a in row['accepted_full_answers'])
                  or type(row.get('generation_budget')) is not int or row['generation_budget']<=0
                  or len(row['prompt_ids'])+row['generation_budget']>NATIVE):
                raise ValueError('Native endpoint requires independently verified full answers and real generation reserve')
    return result, manifest


def read_tasks(path,expected_weight_sha=WEIGHT_SHA):
    if not path.is_file(): raise ValueError('BLOCKED_DATA_QUALIFICATION: missing qualified natural-world manifest')
    manifest = json.loads(path.read_text())
    if manifest.get('status') != 'QUALIFIED_NATURAL_TRANSPORT_V1':
        raise ValueError('BLOCKED_DATA_QUALIFICATION: require qualified natural C/N/F worlds')
    source = path.parent / manifest['views_path']
    qualification_path = path.parent / manifest['qualification_path']
    if sha(source) != manifest['views_sha256'] or sha(qualification_path) != manifest['qualification_sha256']:
        raise ValueError('transport/qualification hash drift')
    qualifications = json.loads(qualification_path.read_text())
    if (qualifications.get('selection_rule')!='fixed_order_native_compact_both_worlds'
            or not 128 <= int(qualifications.get('screened_candidates',0)) <= 2000
            or len(str(qualifications.get('candidate_pool_sha256','')))!=64
            or not isinstance(qualifications.get('rejections'),list)):
        raise ValueError('qualification requires fixed candidate-pool identity, count and rejection ledger')
    candidate_pool=path.parent/qualifications['candidate_pool_path']
    if sha(candidate_pool)!=qualifications['candidate_pool_sha256']:
        raise ValueError('qualification candidate-pool bytes drift')
    if (qualifications['teacher_weight_sha256'] != expected_weight_sha
            or qualifications['teacher_table'] != 'native' or qualifications['teacher_gain'] != 1):
        raise ValueError('qualification must use original Native, not Z-short-correct')
    qmap = {}
    for q in qualifications['rows']:
        key = (q['semantic_id'], q['world'])
        if key in qmap or q.get('truth_verified') is not True:
            raise ValueError('duplicate or unverified truth qualification')
        if (q['generated_ids'] != q['target_ids'] or len(q['target_ids'])<2
                or q['target_ids'][-1] != manifest['eos_token_id']
                or manifest['eos_token_id'] in q['target_ids'][:-1]
                or len(q['compact_prompt_ids']) + len(q['target_ids']) > 2048):
            raise ValueError('Native compact complete-output/EOS qualification failed')
        if len(q['gold_margins']) != len(q['target_ids']) or any(not math.isfinite(v) or v <= 0 for v in q['gold_margins']):
            raise ValueError('qualification needs positive full-trajectory teacher margins')
        qmap[key] = q
    data = list(rows(source))
    ownership = {name: {} for name in ('semantic_id','source_id','template_lineage')}
    train = []
    for row in data:
        if row['split'] not in ('train','validation','test'):
            raise ValueError('unknown task split')
        for field, owner in ownership.items():
            key = row[field]
            if key in owner and owner[key] != row['split']:
                raise ValueError(f'task {field} crosses splits')
            owner[key] = row['split']
        if row['world'] not in (0,1) or row['layout'] not in ('compact','near','far'):
            raise ValueError('invalid world/layout')
        if (not token_ids(row['prompt_ids']) or not token_ids(row['target_ids'])
                or row['target_ids'][-1]!=manifest['eos_token_id']
                or manifest['eos_token_id'] in row['target_ids'][:-1]
                or len(row['prompt_ids'])+len(row['target_ids'])>row['length_cap']):
            raise ValueError('invalid task tokens, EOS or physical cap')
        if row.get('position_ids') is not None:
            raise ValueError('explicit/virtual positions forbidden; use contiguous default positions')
        if row['split'] != 'train':
            budget=row.get('generation_budget')
            if (type(budget) is not int or budget<len(row['target_ids'])
                    or len(row['prompt_ids'])+budget>row['length_cap']):
                raise ValueError('evaluation requires sufficient physical generation reserve')
            if row['length_cap']>2048 and len(row['prompt_ids'])+budget<row['length_cap']-64:
                raise ValueError('evaluation view is not physically long')
            if ('accepted_full_answers' in row and (not row['accepted_full_answers']
                    or any(not isinstance(a,str) or not a for a in row['accepted_full_answers']))):
                raise ValueError('invalid lawful full-answer aliases')
            continue
        q = qmap[(row['semantic_id'],row['world'])]
        if row['target_ids'] != q['target_ids']:
            raise ValueError('training target differs from verified Native compact completion')
        if row['length_cap'] not in (2048,8192,16384):
            raise ValueError('training exceeds the declared physical length domain')
        if len(row['prompt_ids']) + len(row['target_ids']) > row['length_cap']:
            raise ValueError('training prompt+answer exceeds physical cap')
        if row['length_cap'] > 2048 and len(row['prompt_ids']) + len(row['target_ids']) < row['length_cap']-64:
            raise ValueError('short/virtual view mislabeled as physical 8K/16K')
        if (row['length_cap']==2048 and (row['layout']!='compact' or row['prompt_ids']!=q['compact_prompt_ids'])):
            raise ValueError('training compact view differs from qualified Native prompt')
        row['teacher_margin_targets'] = [min(v,1.) for v in q['gold_margins']]
        row['native_compact_prompt_ids'] = q['compact_prompt_ids']
        train.append(row)
    ids = sorted({r['semantic_id'] for r in train})
    if len(ids) != 128 or len(train) != 768:
        raise ValueError('BLOCKED_DATA_QUALIFICATION: fixed 128 groups / 768 views required; do not fill by cherry-picking')
    for semantic_id in ids:
        cells = [(r['world'],r['length_cap']) for r in train if r['semantic_id']==semantic_id]
        if sorted(cells) != [(w,l) for w in (0,1) for l in (2048,8192,16384)]:
            raise ValueError('missing/duplicate task world-length cell')
        if qmap[(semantic_id,0)]['target_ids'] == qmap[(semantic_id,1)]['target_ids']:
            raise ValueError('counterfactual worlds must require different answers')
    families={sid:{r['family'] for r in train if r['semantic_id']==sid} for sid in ids}
    if any(len(f)!=1 for f in families.values()): raise ValueError('semantic family changed between views')
    counts={f:sum(v=={f} for v in families.values()) for f in ('single_evidence','double_evidence','binding')}
    if counts!={'single_evidence':64,'double_evidence':32,'binding':32}:
        raise ValueError('fixed natural-task family quotas not met')
    specs=manifest['evaluation_splits']
    if specs['validation']!={'groups':64,'lengths':[2048,16384]}:
        raise ValueError('require frozen 64-group compact/near/far <=16K validation')
    if specs['test']['groups']<256 or specs['test']['lengths'] not in ([2048,16384,32768,65536],[2048,16384,32768,65536,131072]):
        raise ValueError('require frozen >=256-group test matrix before training')
    for split,spec in specs.items():
        if split not in ('validation','test'): raise ValueError('unknown evaluation split')
        evaluation_rows(data,split,spec['lengths'],spec['groups'])
    return train, manifest


def task_order(data, seed):
    rng = random.Random(seed)
    buckets = [[r for r in data if r['length_cap']==length] for length in (2048,8192,16384)]
    for bucket in buckets:
        rng.shuffle(bucket)
    if len({len(b) for b in buckets}) != 1:
        raise ValueError('length exposure is unbalanced')
    return [row for i in range(len(buckets[0])) for bucket in buckets for row in (bucket[i],)]


def native_replay_order(native,seed):
    rng=random.Random(seed)
    pools={g:[r for r in native if r['split']=='train' and r['group']==g] for g in GROUPS}
    for group in GROUPS: rng.shuffle(pools[group])
    return pools


def check_assets(args, need_tasks=True):
    native, native_manifest = read_native_pool(args.native_pool)
    if native_manifest['tokenizer_files'] != tokenizer_identity(args.checkpoint):
        raise ValueError('Native tokenizer drift')
    vocab=json.loads((args.checkpoint/'config.json').read_text())['vocab_size']
    if any(max(r['input_ids'])>=vocab or (r.get('prompt_ids') and max(r['prompt_ids'])>=vocab) for r in native):
        raise ValueError('Native token outside checkpoint vocabulary')
    for group in GROUPS:
        for split,count in (('train',128),('calibration',32),('validation',64)):
            if sum(r['split']==split and r['group']==group for r in native) != count:
                raise ValueError(f'BLOCKED_NATIVE_DATA: require {count} {split} rows per stratum')
    if len({r['text_domain'] for r in native if r['split']=='validation' and r['group']=='text'})<2:
        raise ValueError('Native validation requires at least two text domains')
    if not need_tasks:
        return native, native_manifest
    task, task_manifest = read_tasks(args.tasks,checkpoint_contract(args)['weight_sha256'])
    if task_manifest['tokenizer_files'] != native_manifest['tokenizer_files']:
        raise ValueError('task/Native tokenization mismatch')
    all_task=list(rows(args.tasks.parent/task_manifest['views_path']))
    if {r['source_id'] for r in native} & {r['source_id'] for r in all_task}:
        raise ValueError('Native replay and task sources overlap')
    if any(max(r['prompt_ids'])>=vocab or max(r['target_ids'])>=vocab for r in all_task):
        raise ValueError('task token outside checkpoint vocabulary')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True,trust_remote_code=False)
    if tokenizer.eos_token_id!=task_manifest['eos_token_id']: raise ValueError('task tokenizer EOS drift')
    validate_answer_worlds(all_task,tokenizer)
    return task, native, task_manifest


def cache_native(args):
    import numpy as np
    import torch
    native, manifest = check_assets(args, need_tasks=False)
    if args.table is not None or args.gain != 1 or args.adapter is not None:
        raise ValueError('teacher must be ORIGINAL Native, never the deployment table')
    if args.output.exists() and not args.resume_cache:
        raise FileExistsError('use --resume-cache only for this interrupted cache')
    args.output.mkdir(parents=True,exist_ok=args.resume_cache)
    if (args.output/'manifest.json').exists():
        raise FileExistsError('teacher cache already complete; reuse it without recomputing')
    config=json.loads((args.checkpoint/'config.json').read_text())
    contract={'pool_sha256':sha(args.native_pool),'checkpoint_config_sha256':sha(args.checkpoint/'config.json'),
              'tokenizer_files':tokenizer_identity(args.checkpoint),'engine_sha256':sha(__file__)}
    contract_path=args.output/'cache_run.json'
    ledger=args.output/'cache_rows.jsonl'
    entries=[]
    if args.resume_cache:
        if json.loads(contract_path.read_text())!=contract: raise ValueError('cache resume identity drift')
        entries=list(rows(ledger)) if ledger.exists() else []
    else: write_json(contract_path,contract)
    native_by_id={r['id']:r for r in native if r['split']!='test'}
    cached=set()
    for entry in entries:
        if (entry['id'] in cached or entry['id'] not in native_by_id
                or entry['row_sha256']!=canonical(native_by_id[entry['id']])
                or sha(args.output/entry['path'])!=entry['sha256']):
            raise ValueError('partial teacher cache ledger drift')
        cached.add(entry['id'])
    required=sum(len(r['prediction_positions']) for r in native if r['split']!='test' and r['id'] not in cached)*int(config['vocab_size'])*4
    if shutil.disk_usage(args.output).free < required + 2*2**30:
        raise RuntimeError('insufficient disk for full-vocabulary FP32 teacher cache')
    model, _, identity = load_runtime(args)
    start = time.monotonic()
    for index,row in enumerate(native):
        if row['split']=='test' or row['id'] in cached:
            continue  # final test cannot become a cached training asset
        guard_resources(model, identity, start, args)
        inputs = torch.tensor([row['input_ids']], device='cuda')
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
            hidden = model.model(input_ids=inputs,use_cache=False,return_dict=True).last_hidden_state
            logits = model.lm_head(hidden[:,row['prediction_positions']]).float()[0]
        if not torch.isfinite(logits).all():
            raise RuntimeError('nonfinite teacher logits')
        path = args.output / f'{index:05d}.npy'
        np.save(path, logits.cpu().numpy(), allow_pickle=False)
        entries.append({'id':row['id'],'path':path.name,'sha256':sha(path),'row_sha256':canonical(row),
                        'shape':list(logits.shape),'dtype':'float32'})
        with ledger.open('a') as handle: handle.write(json.dumps(entries[-1])+'\n')
        if index % 32 == 0:
            print(json.dumps({'cached':len(entries),'seconds':time.monotonic()-start}),flush=True)
    write_json(args.output/'manifest.json',{'status':'ORIGINAL_NATIVE_FULL_VOCAB_CACHE_V1',
               'teacher':identity,'pool_sha256':sha(args.native_pool),'entries':entries,
               'numerics':'BF16 model/head forward, full-vocabulary logits stored FP32; not top-k',
               'seconds':time.monotonic()-start})


def train(args, smoke=False):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig,get_peft_model,PeftModel
    task,native,manifest = check_assets(args)
    if args.arm=='N' and (args.table is not None or args.gain!=1.):
        raise ValueError('N arm must deploy original Native table/gain')
    if args.arm in ('Z','Y'):
        if args.table is None: raise ValueError('Z/Y require a frozen control-table file')
        control=json.loads((args.table.parent/'manifest.json').read_text())
        expected_arm=control['arms'][args.arm]
        if (control['status']!='FIXED_NZGY_CONTROLS_FROZEN_V1' or expected_arm['path']!=args.table.name
                or expected_arm['file_sha256']!=sha(args.table) or expected_arm['rotary_amplitude']!=args.gain):
            raise ValueError('declared N/Z/Y arm identity mismatch')
        if args.arm=='Z' and expected_arm['float32_sha256']!=P2_SHA and control.get('profile_source_Z_sha256')!=P2_SHA:
            raise ValueError('Z is not the retained full-p2 witness')
    if args.compact_only:
        task=[{**r,'prompt_ids':r['native_compact_prompt_ids']} for r in task]
    entries=validate_teacher_cache(args,native)
    recipe={key:getattr(args,key) for key in ('arm','gain','seed','placement','compact_only','kl_budget','prefix_lm')}
    recipe.update(tasks_sha256=sha(args.tasks),native_pool_sha256=sha(args.native_pool),
                  checkpoint_contract_sha256=sha(args.checkpoint_contract) if args.checkpoint_contract else None,
                  table_file_sha256=sha(args.table) if args.table else None,
                  teacher_cache_manifest_sha256=sha(args.teacher_cache/'manifest.json'),
                  engine_sha256=sha(__file__),runtime_sha256=sha(ROOT/'scripts/experiments/single_table_generation.py'),
                  contract_sha256=sha(ROOT/'scripts/lib/rope/generation_contract.py'),
                  smoke_only=smoke)
    restored=None; start_step=0
    if args.resume:
        receipt=json.loads((args.resume/'checkpoint.json').read_text())
        if (receipt['recipe']!=recipe or receipt['state_sha256']!=sha(args.resume/'trainer_state.pt')
                or receipt['adapter_sha256']!=sha(args.resume/'adapter_model.safetensors')
                or receipt['adapter_config_sha256']!=sha(args.resume/'adapter_config.json')):
            raise ValueError('resume recipe or checkpoint bytes drift')
        restored=torch.load(args.resume/'trainer_state.pt',map_location='cpu',weights_only=True)
        start_step=restored['step']
        if start_step!=receipt['step'] or not 0 < start_step < args.stop_after_step:
            raise ValueError('resume requires a later stop step within the same frozen schedule')
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    model,tok,identity = load_runtime(args,training=True)
    if restored:
        for field,value in receipt['runtime'].items():
            if identity[field]!=value: raise ValueError('resume runtime drift; do not silently change numeric backend')
    if tok.eos_token_id!=manifest['eos_token_id']: raise ValueError('task/model termination token mismatch')
    target_modules = MODULES if args.placement=='all_linear' else MODULES[:4]
    all_count = projection_parameter_count(model.config.to_dict(), MODULES,16)
    rank = 16 if args.placement=='all_linear' else round(all_count/projection_parameter_count(model.config.to_dict(),MODULES[:4],1))
    expected = projection_parameter_count(model.config.to_dict(),target_modules,rank)
    if abs(expected/all_count-1)>.01: raise ValueError('attention-only parameter budget mismatch')
    model = (PeftModel.from_pretrained(model,args.resume,is_trainable=True) if args.resume else
             get_peft_model(model,LoraConfig(r=rank,lora_alpha=rank,lora_dropout=0.,bias='none',
                           target_modules=list(target_modules),task_type='CAUSAL_LM')))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads(); model.train()
    named_params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    params=[p for _,p in named_params]
    module_params={group:[p for n,p in named_params if ('.mlp.' in n)==(group=='ffn')]
                   for group in ('attention','ffn')}
    if any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad):
        raise RuntimeError('base/norm/embedding/head parameter escaped freeze')
    if sum(p.numel() for p in params)!=expected: raise ValueError('actual trainable parameter count drift')
    if not args.resume:
        if any(bool(torch.count_nonzero(p)) for n,p in named_params if 'lora_B' in n):
            raise RuntimeError('fresh adapter must have zero-output B initialization')
        if any(not bool(torch.count_nonzero(p)) for n,p in named_params if 'lora_A' in n):
            raise RuntimeError('A/B both zero would disconnect initial adapter learning')
    base=model.get_base_model()
    def project(ids,positions):
        inputs=torch.tensor([ids],device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            hidden=base.model(input_ids=inputs,use_cache=False,return_dict=True).last_hidden_state
            return base.lm_head(hidden[:,positions]).float()
    def native_kl(row,details=False):
        student=project(row['input_ids'],row['prediction_positions'])
        teacher=torch.from_numpy(np.array(np.load(args.teacher_cache/entries[row['id']]['path'],allow_pickle=False))).to('cuda')[None]
        if student.shape!=teacher.shape: raise ValueError('full-vocabulary teacher/student shape mismatch')
        logp=teacher.log_softmax(-1); logq=student.log_softmax(-1)
        values=(logp.exp()*(logp-logq)).sum(-1)
        if details:
            return {'KL':float(values.mean()),'max_prefix_KL':float(values.max()),
                    'teacher_argmax_agreement':float((teacher.argmax(-1)==student.argmax(-1)).float().mean())}
        return stable_teacher_kl(student,teacher)
    def task_loss(row):
        gold=row['target_ids']; prompt=row['prompt_ids']
        prefix_positions=prefix_lm_positions(row,args.seed) if args.prefix_lm else []
        positions=list(range(len(prompt)-1,len(prompt)+len(gold)-1))
        all_logits=project([*prompt,*gold[:-1]],positions+prefix_positions)
        logits=all_logits[:,:len(gold)]
        target=torch.tensor([gold],device='cuda')
        ce=F.cross_entropy(logits.flatten(0,1),target.flatten())
        correct=logits.gather(-1,target.unsqueeze(-1)).squeeze(-1)
        wrong=logits.scatter(-1,target.unsqueeze(-1),-torch.inf).amax(-1)
        tau=torch.tensor([row['teacher_margin_targets']],device='cuda')
        margin=correct-wrong
        loss=ce+.25*F.relu(tau-margin).amax()
        metrics={'answer_EOS_CE':float(ce.detach()),
                 'minimum_gold_margin':float(margin.detach().amin()),'EOS_margin':float(margin.detach()[0,-1])}
        if prefix_positions:
            labels=torch.tensor([prompt[p+1] for p in prefix_positions],device='cuda')
            prefix_ce=F.cross_entropy(all_logits[:,len(gold):].flatten(0,1),labels)
            loss=loss+.1*prefix_ce
            metrics.update(prefix_LM_CE=float(prefix_ce.detach()),prefix_LM_positions=len(prefix_positions))
        return loss,metrics
    pools=native_replay_order(native,args.seed)
    native_cursors={g:0 for g in GROUPS};native_exposures=[]
    rng=random.Random(args.seed); ordered=task_order(task,args.seed); dual={g:1. for g in GROUPS}
    start=time.monotonic(); exposures=[]
    restoration_steps,transfer_steps=(1,1) if smoke else (32,96)
    stop_step=2 if smoke else args.stop_after_step
    def calibration_probe():
        # Fixed calibration, never replay or model-selection validation data.
        result={}; model.eval()
        with torch.inference_mode():
            for group in GROUPS:
                probe=sorted((r for r in native if r['split']=='calibration' and r['group']==group),key=lambda r:r['id'])[:8]
                values=[native_kl(r,details=True) for r in probe]
                result[group]={key:(max(v[key] for v in values) if key=='max_prefix_KL' else
                                   sum(v[key] for v in values)/len(values)) for key in values[0]}
                guard_resources(model,identity,start,args)
        model.train(); return result
    initial_kl=restored['initial_kl'] if restored else calibration_probe()
    if restored:
        dual=restored['dual']; exposures=restored['exposures']; rng.setstate(restored['python_rng'])
        native_cursors=restored['native_cursors'];native_exposures=restored['native_exposures']
        torch.set_rng_state(restored['torch_rng']); torch.cuda.set_rng_state_all(restored['cuda_rng'])
    write_json(args.output/'run.json',{**identity,'protocol':'NATIVE_CONSTRAINED_TRANSFER_V3',
        'recipe':recipe,'start_step':start_step,'requested_stop_step':stop_step,
        'parent_checkpoint_sha256':sha(args.resume/'checkpoint.json') if args.resume else None,
        'arm':args.arm,'compact_only':args.compact_only,
        'tasks_sha256':sha(args.tasks),'native_pool_sha256':sha(args.native_pool),
        'teacher_cache_manifest_sha256':sha(args.teacher_cache/'manifest.json'),
        'rank':rank,'alpha':rank,'trainable_parameters':expected,'seed':args.seed,
        'native_kl_target':args.kl_budget,'endpoint_retention_primary':.88,
        'native_sampling':'stratified shuffled without replacement;112 unique rows per stratum in full recipe',
        'loss':'full_answer_EOS_CE + .25 worst_teacher_capped_margin'+(' + .1 sampled_prefix_LM_CE' if args.prefix_lm else ''),
        'prefix_LM_contract':{'enabled':args.prefix_lm,'coefficient':.1,'positions_per_view_max':128,
            'sampling':'uniform without replacement, separate keyed seed, prompt next-token positions only',
            'all_training_position_sets_sha256':canonical([prefix_lm_positions(r,args.seed) for r in task]) if args.prefix_lm else None},
        'gradient_probe_task_scope':'answer/EOS plus sampled prefix LM' if args.prefix_lm else 'answer/EOS only',
        'restoration_steps':restoration_steps,'transfer_steps':transfer_steps,'task_view_count':8*transfer_steps,
        'initial_deployed_native_fixed_calibration_subset':initial_kl,
        'smoke_only':smoke,
        'engine_sha256':sha(__file__)})
    optimizer=None
    def save(step,stage,probe=False):
        dest=args.output/f'step_{step:03d}'
        model.save_pretrained(dest,safe_serialization=True)
        state={'step':step,'stage':stage,'optimizer':optimizer.state_dict() if optimizer else None,
               'dual':dual,'exposures':exposures,'initial_kl':initial_kl,'python_rng':rng.getstate(),
               'native_cursors':native_cursors,'native_exposures':native_exposures,
               'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
        torch.save(state,dest/'trainer_state.pt')
        receipt={'status':'CHECKPOINT_NOT_NATIVE_FEASIBILITY','step':step,'recipe':recipe,
                   'runtime':{key:identity[key] for key in ('torch','cuda','transformers','peft','attention_implementation')},
                   'deployment':{key:identity[key] for key in ('table_sha256','gain','checkpoint_sha256','checkpoint_config_sha256','tokenizer_files')},
                   'adapter_sha256':sha(dest/'adapter_model.safetensors'),'state_sha256':sha(dest/'trainer_state.pt'),
                   'adapter_config_sha256':sha(dest/'adapter_config.json'),
                   'calibration_diagnostics_not_endpoint_gate':None}
        write_json(dest/'checkpoint.json',receipt)
        if probe:
            try: receipt['calibration_diagnostics_not_endpoint_gate']=calibration_probe()
            except RuntimeError as error:
                if 'wall-clock budget exhausted' not in str(error): raise
                receipt['calibration_status']='INCOMPLETE_TIME_CAP_CHECKPOINT_PRESERVED'
            write_json(dest/'checkpoint.json',receipt)
        return dest
    if not restored: save(0,'initial')
    final_step=start_step; stage='initial'; paused_for_budget=False
    with (args.output/'training.jsonl').open('w',buffering=1) as handle:
        for stage,count in (('restoration',restoration_steps),('transfer',transfer_steps)):
            offset=restoration_steps if stage=='transfer' else 0
            if start_step>=offset+count or final_step>=stop_step: continue
            optimizer=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=0.,fused=True)
            if restored and restored['stage']==stage and start_step>offset:
                optimizer.load_state_dict(restored['optimizer'])
            for local_step in range(max(0,start_step-offset),count):
                if final_step>=stop_step: break
                if time.monotonic()-start >= args.max_seconds:
                    paused_for_budget=True; break
                guard_resources(model,identity,start,args); optimizer.zero_grad(set_to_none=True)
                native_values={}; task_value=0.; task_metrics=[]
                gradient_probe=stage=='transfer' and (local_step==0 or (local_step+1)%32==0 or smoke)
                task_gradients=None
                if stage=='transfer':
                    for row in ordered[local_step*8:(local_step+1)*8]:
                        loss,metrics=task_loss(row); task_metrics.append(metrics)
                        if not torch.isfinite(loss): raise RuntimeError('nonfinite task loss')
                        (loss/8).backward(); task_value+=float(loss.detach())/8
                        exposures.append((row['semantic_id'],row['world'],row['length_cap']))
                    if gradient_probe:
                        task_gradients={id(p):p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                                        for p in params}
                group_ids=range(4) if stage=='restoration' else (local_step%4,(local_step+1)%4)
                for gi in group_ids:
                    group=GROUPS[gi]; values=[]
                    for _ in range(2 if stage=='restoration' else 1):
                        if native_cursors[group]>=len(pools[group]): raise RuntimeError('registered Native replay pool exhausted')
                        row=pools[group][native_cursors[group]];native_cursors[group]+=1
                        native_exposures.append(row['id']);kl=native_kl(row)
                        if not torch.isfinite(kl) or float(kl.detach()) < -1e-5: raise RuntimeError('invalid Native KL')
                        weight=1/8 if stage=='restoration' else dual[group]/args.kl_budget
                        (weight*kl).backward(); values.append(float(kl.detach()))
                    native_values[group]=sum(values)/len(values)
                gradient_diagnostics={}
                if task_gradients is not None:
                    for group,subset in module_params.items():
                        t2=n2=dot=0.
                        for param in subset:
                            task_grad=task_gradients[id(param)].float()
                            total_grad=param.grad.detach().float() if param.grad is not None else torch.zeros_like(param).float()
                            native_grad=total_grad-task_grad
                            t2+=float(task_grad.square().sum()); n2+=float(native_grad.square().sum())
                            dot+=float((task_grad*native_grad).sum())
                        gradient_diagnostics[group]={'task_norm':math.sqrt(t2),'weighted_native_norm':math.sqrt(n2),
                              'cosine':dot/math.sqrt(t2*n2) if t2*n2>0 else None,
                              'weighted_native_to_task':math.sqrt(n2/t2) if t2>0 else None}
                    del task_gradients
                norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
                if args.arm=='N' and stage=='restoration' and (norm!=0. or any(abs(v)>1e-7 for v in native_values.values())):
                    raise RuntimeError('Native identity restoration must have zero KL and zero gradient; inspect numeric/runtime parity')
                # Zero restoration gradient on original Native is legitimate.
                if not math.isfinite(norm) or (stage=='transfer' and norm<=0): raise RuntimeError('nonfinite or disconnected transfer gradient')
                rate=learning_rate_factor(local_step,count)
                for pg in optimizer.param_groups: pg['lr']=1e-4*rate
                before_update={id(p):p.detach().clone() for p in params} if gradient_probe else None
                optimizer.step()
                updates={}
                if before_update is not None:
                    for group,subset in module_params.items():
                        updates[group]=math.sqrt(sum(float((p.detach().float()-before_update[id(p)].float()).square().sum()) for p in subset))
                        if smoke and subset and (not math.isfinite(updates[group]) or updates[group]<=0):
                            raise RuntimeError(f'smoke did not update {group} LoRA factors')
                    del before_update
                if stage=='transfer':
                    for group,value in native_values.items(): dual[group]=max(0.,dual[group]+.05*(value/args.kl_budget-1))
                step=local_step+1+(restoration_steps if stage=='transfer' else 0)
                final_step=step
                record={'step':step,'stage':stage,'task_loss':task_value,'native_kl':native_values,
                        'task_components':{k:sum(v[k] for v in task_metrics)/len(task_metrics) for k in task_metrics[0]} if task_metrics else {},
                        'gradient_diagnostics':gradient_diagnostics,
                        'LoRA_factor_update_norms':updates,
                        'dual':dual.copy(),'grad_norm':norm,'learning_rate':1e-4*rate,'seconds':time.monotonic()-start}
                handle.write(json.dumps(record)+'\n')
                if step%32==0 or local_step+1==count: save(step,stage,probe=not smoke)
                if step%8==0 or gradient_probe: print(json.dumps(record),flush=True)
            if paused_for_budget: break
    final_dir=args.output/f'step_{final_step:03d}'
    if not (final_dir/'checkpoint.json').exists(): save(final_step,stage)
    paused_for_budget=paused_for_budget or time.monotonic()-start>=args.max_seconds
    if paused_for_budget:
        write_json(args.output/'paused.json',{'status':'BUDGET_PAUSED_NOT_FAILURE','step':final_step,
                   'checkpoint_sha256':sha(final_dir/'checkpoint.json'),'next_action':'resume same recipe after reviewing receipts'})
        return
    model.eval()
    probe_prompt=task[0]['native_compact_prompt_ids']
    before_reload=greedy(base,probe_prompt,tok.eos_token_id,8)
    reload_args=copy.copy(args); reload_args.adapter=args.output/f'step_{final_step:03d}'
    peak_allocated=torch.cuda.max_memory_allocated()
    # Release training graph/optimizer before loading a second base model.
    optimizer.zero_grad(set_to_none=True)
    del optimizer,params,named_params,module_params,base,model
    if 'loss' in locals(): del loss
    if 'kl' in locals(): del kl
    if 'param' in locals(): del param,total_grad,native_grad,task_grad
    gc.collect(); torch.cuda.empty_cache()
    reloaded,_,reload_identity=load_runtime(reload_args)
    if greedy(reloaded,probe_prompt,tok.eos_token_id,8)!=before_reload:
        raise RuntimeError('saved/merged adapter changes cached greedy probe; no complete receipt')
    guard_resources(reloaded,reload_identity,start,args)
    write_json(args.output/'complete.json',{'status':'SMOKE_COMPLETE_NOT_SCIENCE' if smoke else
        ('TRAINING_COMPLETE_NOT_FEASIBILITY_OR_CAPABILITY' if final_step==128 else 'SEGMENT_COMPLETE_REVIEW_BEFORE_RESUME'),
        'run_sha256':sha(args.output/'run.json'),'training_sha256':sha(args.output/'training.jsonl'),
        'task_exposure_sha256':canonical(exposures),'cumulative_task_forward_count':len(exposures),
        'native_exposure_sha256':canonical(native_exposures),'unique_native_replay_rows':len(set(native_exposures)),
        'cumulative_native_training_forwards':8*min(final_step,restoration_steps)+2*max(0,final_step-restoration_steps),
        'peak_allocated_bytes':max(peak_allocated,torch.cuda.max_memory_allocated()),
        'actual_cumulative_task_tokens':sum(len(r['prompt_ids'])+len(r['target_ids'])-1 for r in ordered[:len(exposures)]),
        'task_sequence_including_terminal_EOS_tokens':sum(len(r['prompt_ids'])+len(r['target_ids']) for r in ordered[:len(exposures)]),
        'answer_EOS_supervised_positions':sum(len(r['target_ids']) for r in ordered[:len(exposures)]),
        'prefix_LM_supervised_positions':sum(len(prefix_lm_positions(r,args.seed)) for r in ordered[:len(exposures)]) if args.prefix_lm else 0,
        'input_count_definition':'prompt plus answer excluding terminal EOS, which is a target but not a student input',
        'calibration_forwards_per_probe':32,
        'final_adapter_reload_greedy_exact':True,
        'seconds':time.monotonic()-start,'completed_step':final_step,
        'saved_steps':[int(p.name.split('_')[-1]) for p in sorted(args.output.glob('step_*'))],
        'next_action':'independent Native retention then C/N/F validation; training loss/KL do not select a model'})


def evaluate_tasks(args):
    """Independent complete-output reader for the qualified natural-view schema."""
    _,_,manifest=check_assets(args)
    if not set(args.lengths)<=set(manifest['evaluation_splits'][args.split]['lengths']):
        raise ValueError('unregistered evaluation lengths')
    data=evaluation_rows(list(rows(args.tasks.parent/manifest['views_path'])),args.split,args.lengths,
                         manifest['evaluation_splits'][args.split]['groups'])
    if args.split=='test' and not args.frozen_selection:
        raise ValueError('test outcomes require a sealed method/adapter/task-manifest receipt')
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    model,tok,identity=load_runtime(args)
    if tok.eos_token_id!=manifest['eos_token_id']: raise ValueError('evaluation termination token mismatch')
    if args.frozen_selection:
        lock=json.loads(args.frozen_selection.read_text())
        for field in ('table_sha256','gain','adapter_sha256','adapter_config_sha256'):
            if lock[field]!=identity[field]: raise ValueError('sealed method identity drift')
        if lock['task_manifest_sha256']!=sha(args.tasks): raise ValueError('sealed task data drift')
    start=time.monotonic(); groups={}
    with (args.output/'examples.jsonl').open('w',buffering=1) as handle:
        for row in data:
            guard_resources(model,identity,start,args)
            gold=row['target_ids']
            if gold[-1]!=manifest['eos_token_id'] or len(row['prompt_ids'])+len(gold)>row['length_cap']:
                raise ValueError('evaluation answer/EOS/physical cap drift')
            budget=int(row['generation_budget'])
            if len(row['prompt_ids'])+budget>row['length_cap']:
                raise ValueError('evaluation generation reserve exceeds declared physical cap')
            output=greedy(model,row['prompt_ids'],tok.eos_token_id,budget)
            ended=bool(output and output[-1]==tok.eos_token_id and tok.eos_token_id not in output[:-1])
            decode=lambda ids:tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            text=decode(output[:-1] if ended else output)
            aliases=row.get('accepted_full_answers',[decode(gold[:-1])])
            success=ended and text in aliases
            record={key:row[key] for key in ('semantic_id','world','layout','length_cap','family')}
            record.update(generated_ids=output,unmodified_output_text=text,accepted_full_answers=aliases,
                          eos_token_id=tok.eos_token_id,ended_with_eos=ended,full_exact_eos=success,
                          prompt_sha256=canonical(row['prompt_ids']))
            handle.write(json.dumps(record)+'\n')
            key=(row['family'],row['layout'],row['length_cap'],row['semantic_id'])
            pair=groups.setdefault(key,{})
            if row['world'] in pair: raise ValueError('duplicate natural world/layout result')
            pair[row['world']]=success
    cells={}
    for (family,layout,length,semantic_id),pair in groups.items():
        if set(pair)!={0,1}: raise ValueError('missing natural counterfactual world')
        cells.setdefault(f'{family}:{layout}:{length}',[]).append(all(pair.values()))
    write_json(args.output/'evaluation.json',{**identity,'task_manifest_sha256':sha(args.tasks),
        'evaluation_engine_sha256':sha(__file__),
        'split':args.split,'rows':len(data),'examples_sha256':sha(args.output/'examples.jsonl'),
        'cells':{key:{'semantic_groups':len(values),'both_worlds_exact_eos':sum(values)/len(values)} for key,values in cells.items()},
        'scope':'full generated answers with lawful aliases and EOS; Native feasibility is a separate gate'})


def evaluate_native(args):
    """Fresh held-out Native endpoints, independent of the replay KL objective."""
    import torch
    import torch.nn.functional as F
    native,manifest=check_assets(args,need_tasks=False)
    selected=[r for r in native if r['split']==args.split]
    if set(r['group'] for r in selected)!=set(GROUPS): raise ValueError('incomplete Native endpoint strata')
    if args.split=='test' and not args.frozen_selection: raise ValueError('Native test needs frozen selection')
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    model,tok,identity=load_runtime(args)
    if args.frozen_selection:
        lock=json.loads(args.frozen_selection.read_text())
        for field in ('table_sha256','gain','adapter_sha256','adapter_config_sha256'):
            if identity[field]!=lock[field]: raise ValueError('sealed Native deployment identity drift')
        if lock['native_pool_sha256']!=sha(args.native_pool): raise ValueError('sealed Native pool drift')
    start=time.monotonic()
    with (args.output/'examples.jsonl').open('w',buffering=1) as handle:
        for row in selected:
            guard_resources(model,identity,start,args)
            record={'task':row['group'],'group':row['source_id'],'asset_sha256':canonical(row),'row_id':row['id']}
            if row['group']=='text':
                positions=row['prediction_positions']
                inputs=torch.tensor([row['input_ids']],device='cuda')
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    hidden=model.model(input_ids=inputs,use_cache=False,return_dict=True).last_hidden_state
                    logits=model.lm_head(hidden[:,positions]).float()
                    labels=inputs[:,[p+1 for p in positions]]
                    nll=F.cross_entropy(logits.flatten(0,1),labels.flatten())
                record.update(nll=float(nll),text_domain=row['text_domain'],scored_tokens=len(positions))
            else:
                output=greedy(model,row['prompt_ids'],tok.eos_token_id,row['generation_budget'])
                ended=bool(output and output[-1]==tok.eos_token_id and tok.eos_token_id not in output[:-1])
                text=tok.decode(output[:-1] if ended else output,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                correct=text in row['accepted_full_answers']
                record.update(generated_ids=output,eos_token_id=tok.eos_token_id,unmodified_output_text=text,
                              accepted_full_answers=row['accepted_full_answers'],ended_with_eos=ended,
                              score=float(correct),score_eos=float(correct and ended))
            if any(isinstance(v,float) and not math.isfinite(v) for v in record.values()): raise RuntimeError('nonfinite Native endpoint')
            handle.write(json.dumps(record)+'\n')
    write_json(args.output/'native_evaluation.json',{**identity,'status':'FRESH_STRATIFIED_NATIVE_ENDPOINTS_V1',
               'native_pool_sha256':sha(args.native_pool),'fold':'selection' if args.split=='validation' else 'confirmation',
               'data_sha256':canonical([canonical(r) for r in selected]),'examples_sha256':sha(args.output/'examples.jsonl'),
               'evaluation_engine_sha256':sha(__file__),'rows':len(selected),
               'scope':'held-out observed-token text NLL and full-answer/EOS generation; no KL-as-retention substitution'})


def compare_smoke(args):
    """Work-machine CPU comparison of uninterrupted versus resumed GPU smoke."""
    import torch
    from safetensors.torch import load_file
    if not args.reference_run or not args.resumed_run: raise ValueError('both smoke run directories required')
    original=json.loads((args.reference_run/'complete.json').read_text())
    resumed=json.loads((args.resumed_run/'complete.json').read_text())
    if any(r['status']!='SMOKE_COMPLETE_NOT_SCIENCE' for r in (original,resumed)):
        raise ValueError('both smokes must have completed including merged greedy parity')
    for directory,result in ((args.reference_run,original),(args.resumed_run,resumed)):
        cp=json.loads((directory/'step_002/checkpoint.json').read_text())
        if (result['run_sha256']!=sha(directory/'run.json') or result['training_sha256']!=sha(directory/'training.jsonl')
                or cp['adapter_sha256']!=sha(directory/'step_002/adapter_model.safetensors')
                or cp['adapter_config_sha256']!=sha(directory/'step_002/adapter_config.json')):
            raise ValueError('smoke receipt or adapter bytes changed')
    a=json.loads((args.reference_run/'run.json').read_text()); b=json.loads((args.resumed_run/'run.json').read_text())
    if a['recipe']!=b['recipe'] or b['parent_checkpoint_sha256']!=sha(args.reference_run/'step_001/checkpoint.json'):
        raise ValueError('resume smoke is not the same frozen first-stage checkpoint')
    left=load_file(str(args.reference_run/'step_002/adapter_model.safetensors'))
    right=load_file(str(args.resumed_run/'step_002/adapter_model.safetensors'))
    if left.keys()!=right.keys(): raise ValueError('resumed adapter structure changed')
    maximum=max(float((left[k]-right[k]).abs().max()) for k in left)
    ok=all(torch.allclose(left[k],right[k],atol=1e-6,rtol=1e-5) for k in left)
    if args.output.exists(): raise FileExistsError(args.output)
    write_json(args.output,{'status':'RESUME_SMOKE_NUMERIC_PARITY_NOT_SCIENCE' if ok else 'UNRESOLVED_RESUME_NUMERIC_DRIFT',
               'atol':1e-6,'rtol':1e-5,'max_parameter_abs_difference':maximum,
               'reference_complete_sha256':sha(args.reference_run/'complete.json'),
               'resumed_complete_sha256':sha(args.resumed_run/'complete.json'),
               'scope':'one restoration/transfer boundary only; does not prove all future resumptions or capability'})
    if not ok: raise RuntimeError('resume smoke parameter drift; do not start the training matrix')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('preflight','cache-native','smoke','train','evaluate','native-evaluate','compare-smoke'))
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--checkpoint-contract',type=Path)
    p.add_argument('--native-pool',type=Path,required=True)
    p.add_argument('--tasks',type=Path)
    p.add_argument('--teacher-cache',type=Path)
    p.add_argument('--resume-cache',action='store_true')
    p.add_argument('--resume',type=Path,help='verified step directory; preserves optimizer, RNG and exposure order')
    p.add_argument('--reference-run',type=Path)
    p.add_argument('--resumed-run',type=Path)
    p.add_argument('--stop-after-step',type=int,choices=(32,64,96,128),default=64,
                   help='first run defaults to 32 restoration + 32 transfer; review before continuing')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--table',type=Path)
    p.add_argument('--gain',type=float,default=1.)
    p.add_argument('--seed',type=int,choices=(42,43,44),default=42)
    p.add_argument('--placement',choices=('all_linear','attention_matched'),default='all_linear')
    p.add_argument('--arm',choices=('N','Z','Y'))
    p.add_argument('--compact-only',action='store_true')
    p.add_argument('--adapter',type=Path)
    p.add_argument('--split',choices=('validation','test'),default='validation')
    p.add_argument('--lengths',nargs='+',type=int,default=[2048,16384])
    p.add_argument('--frozen-selection',type=Path)
    p.add_argument('--kl-budget',type=float,default=.02)
    p.add_argument('--prefix-lm',action='store_true',help='prospective fixed .1-weight/128-position prefix-LM companion; separate run and review required')
    p.add_argument('--authorized',action='store_true')
    p.add_argument('--max-seconds',type=float,default=3600)
    args=p.parse_args(); args.data=None; args.min_headroom_gib=1.
    if not all(math.isfinite(v) and v>0 for v in (args.gain,args.kl_budget,args.max_seconds)):
        p.error('finite positive gain/KL target/time cap required')
    if args.action not in ('cache-native','native-evaluate','compare-smoke') and not args.tasks: p.error('--tasks is required')
    if args.action in ('train','smoke') and not args.arm: p.error('--arm N/Z/Y is required')
    if args.action in ('train','smoke') and args.adapter: p.error('all arms start fresh; no inherited adapter')
    if args.resume and args.action not in ('train','smoke'): p.error('--resume is only for train/smoke')
    if args.resume_cache and args.action!='cache-native': p.error('--resume-cache is only for cache-native')
    if args.prefix_lm and (args.action not in ('train','smoke','preflight') or args.resume):
        p.error('prefix-LM companion requires a fresh train/smoke/preflight; no unqualified resume')
    if (args.compact_only or args.placement!='all_linear') and (args.seed!=42 or args.arm!='Z'):
        p.error('explanatory ablations are fixed to Z / seed 42, after main feasibility')
    if args.action=='preflight':
        task,native,_=check_assets(args)
        if args.teacher_cache: validate_teacher_cache(args,native)
        config=json.loads((args.checkpoint/'config.json').read_text())
        needed=sum(len(r['prediction_positions']) for r in native if r['split']!='test')*config['vocab_size']*4
        report={'status':'CPU_ASSETS_VALIDATED_RUNTIME_PENDING','task_views':len(task),'native_rows':len(native),
                'teacher_cache_validated':bool(args.teacher_cache),'full_teacher_cache_bytes':needed,
                'task_forward_count_full_recipe':768,'native_train_forward_count_full_recipe':448,
                'first_segment_steps':64,'next_action':'cache-native then actual-gradient smoke; do not launch matrix'}
        if args.output.exists(): raise FileExistsError(args.output)
        args.output.mkdir(parents=True); write_json(args.output/'readiness.json',report)
        print(json.dumps(report))
    elif args.action=='cache-native': cache_native(args)
    elif args.action=='evaluate': evaluate_tasks(args)
    elif args.action=='native-evaluate': evaluate_native(args)
    elif args.action=='compare-smoke': compare_smoke(args)
    else:
        if not args.teacher_cache: p.error('--teacher-cache is required')
        train(args,smoke=args.action=='smoke')

if __name__=='__main__': main()
