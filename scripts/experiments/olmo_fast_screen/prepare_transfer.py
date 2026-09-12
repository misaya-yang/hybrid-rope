"""Prepare cached checkpoint identities and fixed MrPro/BM/UNI tables for transfer."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, GenerationConfig

from scripts.experiments.cross_audit.tables import native_table, transform, tensor_sha
from scripts.lib.rope.boundary_matched import boundary_matched_inv_freq
from .bench import digest
from .prepare import sha_file, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',required=True,type=Path)
    p.add_argument('--key',required=True)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--reuse-inputs',type=Path)
    p.add_argument('--reuse-assets',type=Path)
    p.add_argument('--scale',type=float,default=4.0,
                   help='deployment extension factor (default preserves the original 4x protocol)')
    p.add_argument('--include-uni',action='store_true',
                   help='also prepare the endpoint-and-area-matched MrRoPE-Uni control')
    args=p.parse_args();spec=json.loads(args.spec.read_text())[args.key]
    model=Path(spec['model_path']);out=args.out.resolve()
    config=json.loads((model/'config.json').read_text())
    if config.get('rope_scaling') or config.get('use_sliding_window'):
        raise ValueError('requires an unscaled native checkpoint with full attention')
    weights={};stats={};count=0
    previous=json.loads((args.reuse_assets/'manifest.json').read_text()) if args.reuse_assets else None
    for name,expected in spec['weight_files_sha256'].items():
        path=model/name;stat=path.stat();stats[name]=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
        if previous and previous['model_path']==str(model) and previous['weight_stats'][name]==stats[name] and previous['weight_files_sha256'][name]==expected:
            actual=expected
        else:actual=sha_file(path)
        if actual!=expected:raise ValueError('cached weight differs from recorded identity: '+name)
        weights[name]=actual
        with safe_open(path,framework='pt',device='cpu') as f:
            count+=sum(math.prod(f.get_slice(k).get_shape()) for k in f.keys())
    if (model/'model.safetensors.index.json').exists():
        index=json.loads((model/'model.safetensors.index.json').read_text())
        if set(index['weight_map'].values())!=set(weights):raise ValueError('incomplete shard inventory')
    for name,expected in spec.get('metadata_sha256',{}).items():
        if sha_file(model/name)!=expected:raise ValueError('cached model metadata differs: '+name)
    for name,expected in spec.get('metadata_blob_sha1',{}).items():
        data=(model/name).read_bytes()
        actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if actual!=expected:raise ValueError('cached upstream metadata blob differs: '+name)
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    contract=dict(tokenizer_json=sha_file(model/'tokenizer.json'),
                  template_sha256=hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest(),
                  eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
    dim=config.get('head_dim',config['hidden_size']//config['num_attention_heads'])
    base=config['rope_theta'];length=config['max_position_embeddings'];scale=args.scale
    native=native_table(dim,base)
    mr,gain,meta=transform(native,dim=dim,base=base,reference_length=length,scale=scale,method='mrpro')
    bm,bm_gain,bm_meta=boundary_matched_inv_freq(torch.from_numpy(native),base=base,reference_length=length,scale=scale)
    assert gain==bm_gain
    prepared_tables=[('Native',native,1.),('MrPro',mr,gain),('MrProBM',bm.numpy(),gain)]
    uni_meta=None
    if args.include_uni:
        uni,uni_gain,uni_meta=transform(
            native,dim=dim,base=base,reference_length=length,scale=scale,method='mruni')
        assert gain==uni_gain
        prepared_tables.append(('MrProUni',uni,uni_gain))
    tables={name:dict(values_float32=value.tolist(),tensor_sha256=tensor_sha(value),gain=a)
            for name,value,a in prepared_tables}
    decoding=GenerationConfig.from_pretrained(model,local_files_only=True)
    for name,value in dict(do_sample=False,temperature=None,top_p=None,top_k=None,num_beams=1,
        num_return_sequences=1,repetition_penalty=1.,no_repeat_ngram_size=0,use_cache=True,
        min_new_tokens=0,min_length=0,pad_token_id=tokenizer.pad_token_id).items():setattr(decoding,name,value)
    generation=decoding.to_dict();generation.pop('max_new_tokens',None)
    out.mkdir(parents=True,exist_ok=False)
    write(out/'tables.json',tables);write(out/'generation_config.json',generation)
    candidates=[dict(id='MrProBM',eligible=True,
        review_status='REVIEWED_FOR_GPU',definition=bm_meta['formula'],
        hypothesis=('At the declared extension scale, BM smooths both transition-band joins '
                    'while preserving the native and fully-scaled endpoints.'),
        failure_rule=('No positive long macro gain over MrPro does not establish transfer; '
                      'report all tasks and paired rows.'))]
    if args.include_uni:
        candidates.append(dict(id='MrProUni',eligible=True,
            review_status='REVIEWED_FOR_GPU',definition='m_q=q/N (MrRoPE-Uni Eq.13)',
            hypothesis=('UNI matches BM endpoints and total exponent area but has slope shocks '
                        'at both band joins, isolating the BM taper.'),
            failure_rule=('BM must beat both MrPro and this exact area-and-endpoint control; '
                          'otherwise the boundary-taper mechanism is not supported.')))
    write(out/'queue.json',dict(max_candidates=10,ordered_candidates=candidates))
    manifest={}
    if args.reuse_inputs:
        old=args.reuse_inputs;source=json.loads((old/'manifest.json').read_text())
        if source['tokenizer_contract']!=contract:raise ValueError('tokenizer/template contract differs; regenerate inputs')
        if source['prepared_files']['generation_config.json']!=sha_file(out/'generation_config.json'):
            raise ValueError('decoding contract differs between checkpoints')
        for name,h in source['prepared_files'].items():
            if sha_file(old/name)!=h:raise ValueError('source preparation drift: '+name)
        manifest=source.copy()
        for name in ('screen.jsonl','qualification.jsonl','prompts.jsonl'):shutil.copyfile(old/name,out/name)
        manifest['shared_input_manifest_sha256']=sha_file(old/'manifest.json')
    root=Path(__file__).resolve().parents[3]
    dependencies=['scripts/experiments/olmo_fast_screen/'+name for name in
        ('prepare.py','prepare_transfer.py','prepare_ruler.py','bench.py','ruler_bench.py','runtime.py','run.py','supervise.py')]
    dependencies+=['scripts/experiments/cross_audit/tables.py','scripts/lib/rope/official_yarn.py','scripts/lib/rope/boundary_matched.py']
    manifest.update(status='PREPARED_TRANSFER_INPUTS_READY' if args.reuse_inputs else 'MODEL_TABLES_READY',
        model_id=spec['model_id'],revision=spec['revision'],model_path=str(model),actual_parameters=count,
        weight_files_sha256=weights,weight_sha256=digest(weights),weight_stats=stats,
        asset_spec_sha256=sha_file(args.spec),asset_provenance=spec['provenance'],
        model_files_sha256={p.name:sha_file(p) for p in model.iterdir() if p.is_file() and p.suffix in ('.json','.txt')},
        tokenizer_contract=contract,static_scale=scale,native_length=length,base=base,head_dim=dim,
        construction=dict(mr=meta,bm=bm_meta,uni=uni_meta),reference_arm='MrPro',
        complete_candidate_queue=bool(args.include_uni),
        software={name:importlib.metadata.version(name) for name in ('torch','transformers','numpy')},
        code_files={name:sha_file(root/name) for name in dependencies})
    manifest.pop('weight_stat',None)
    names=['tables.json','generation_config.json','queue.json']
    if args.reuse_inputs:names+=['screen.jsonl','qualification.jsonl','prompts.jsonl']
    manifest['prepared_files']={name:sha_file(out/name) for name in names}
    write(out/'manifest.json',manifest)
    print(json.dumps(dict(model=spec['model_id'],parameters=count,native_length=length,scale=scale,
                         low=meta['low'],high=meta['high'],status=manifest['status'])),flush=True)


if __name__=='__main__':main()
