"""CPU-only freeze/rescore. Never loads model weights or opens a GPU context."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil

import numpy as np

from .contracts import read_rows, score, sha_file, sha_json, validate_rows, write_json
from .tables import check_table, native_table, tensor_sha, transform


def user_content(tokenizer, ids):
    text = tokenizer.decode(ids, skip_special_tokens=False)
    # Old OLMo transport rows already carry a chat template. Do not wrap twice.
    prefix = (tokenizer.bos_token or '') + '<|user|>\n'
    suffix = '\n<|assistant|>\n'
    if text.startswith(prefix) and text.endswith(suffix):
        return text[len(prefix):-len(suffix)]
    if '<|user|>' in text or '<|assistant|>' in text:
        raise ValueError('unrecognized preformatted chat row')
    return text


def chat_ids(tokenizer, content):
    result = tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                          add_generation_prompt=True, tokenize=True)
    if hasattr(result, 'input_ids'):
        result = result['input_ids']
    if result and isinstance(result[0], list):
        result = result[0]
    return list(result)


def freeze(args):
    from transformers import AutoTokenizer
    out, model = Path(args.out), Path(args.model)
    out.mkdir(parents=True, exist_ok=False)
    cfg_path = model / 'config.json'
    cfg = json.loads(cfg_path.read_text())
    if cfg['model_type'] != 'olmo2' or cfg.get('rope_scaling'):
        raise ValueError('this preparation requires unscaled OLMo2')
    dim = cfg.get('head_dim') or cfg['hidden_size'] // cfg['num_attention_heads']
    base, length = cfg['rope_theta'], cfg['max_position_embeddings']
    if (dim, base, length, cfg['hidden_size'], cfg['num_hidden_layers']) != (128, 500000, 4096, 2048, 16):
        raise ValueError('expected OLMo-2-0425-1B-Instruct geometry')
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    table_dir = out / 'tables'
    table_dir.mkdir()
    source = native_table(dim, base)
    arms = {}
    for name, method in [('Native', 'identity'), ('YaRN', 'yarn'), ('MrUni', 'mruni'), ('MrPro', 'mrpro')]:
        values, gain, meta = transform(source, dim=dim, base=base, reference_length=length,
                                       scale=4, method=method)
        np.save(table_dir / f'{name}.npy', values)
        arms[name] = dict(path=f'tables/{name}.npy', tensor_sha256=tensor_sha(values),
                          amplitude=gain, logit_multiplier=gain**2, construction=meta)
    old = Path(args.old_tables)
    old_manifest = json.loads((old / 'manifest_round12.json').read_text())
    z = old_manifest['arms']['Z']
    z_path = old / z['path']
    values = np.load(z_path, allow_pickle=False)
    check_table(values, dim)
    if tensor_sha(values) != '56ddfae2800d4bbf9e6bd2d20bae751edc9865dcbaf641c7c8c4f1d7f1c15e5b':
        raise ValueError('Z is not the archived candidate')
    if float(z['rotary_amplitude']) != 1.102585782722872:
        raise ValueError('Z amplitude differs from archived candidate')
    shutil.copyfile(z_path, table_dir / 'Z.npy')
    arms['Z'] = dict(path='tables/Z.npy', tensor_sha256=tensor_sha(values),
                     amplitude=z['rotary_amplitude'], logit_multiplier=z['rotary_amplitude']**2,
                     construction='unchanged Round11/12 archived tensor; no regeneration')
    equivalence = {}
    for old_name, new_name in [('M', 'MrPro'), ('Y', 'YaRN'), ('Y2', 'YaRN')]:
        if old_name in old_manifest['arms']:
            entry = old_manifest['arms'][old_name]
            old_a = np.load(old / entry['path'], allow_pickle=False)
            new_a = np.load(out / arms[new_name]['path'], allow_pickle=False)
            equivalence[old_name] = dict(new_arm=new_name,
                old_tensor_sha256=tensor_sha(old_a), new_tensor_sha256=tensor_sha(new_a),
                max_absolute_difference=float(np.max(np.abs(old_a-new_a))),
                old_amplitude=entry['rotary_amplitude'], new_amplitude=arms[new_name]['amplitude'])
    write_json(out/'table_manifest.json', dict(status='FROZEN', model_id='allenai/OLMo-2-0425-1B-Instruct',
        head_dim=dim, base=base, reference_length=length, scale=4, arms=arms,
        policy='one static table and amplitude for prefill, decode and Native'))
    write_json(out/'baseline_equivalence.json', equivalence)

    # Reuse supplied-source tasks as development only, respecting existing splits.
    train_ids, val_ids, train_sources, val_sources = set(), set(), set(), set()
    eval_rows, sft_rows = [], []
    lengths = {2048, 8192, 16384}
    for r in read_rows(args.transport):
        if r['split'] not in ('train', 'validation') or r['family'] not in ('single_evidence', 'double_evidence', 'binding'):
            continue
        if int(r['length_cap']) not in lengths:
            continue
        content = user_content(tokenizer, r['prompt_ids'])
        ids = chat_ids(tokenizer, content)
        if len(ids) + int(r['generation_budget']) > int(r['length_cap']):
            raise ValueError(f'chat prompt plus generation reserve exceeds declared window: {r["semantic_id"]}')
        item = {**r, 'group_id': r['semantic_id'], 'world': str(r['world']), 'prompt_ids': ids,
                'row_id': sha_json([r['semantic_id'],r['world'],r['layout'],r['length_cap']]),
                'input_tokens': len(ids), 'prompt_sha256': sha_json(ids),
                'L_native': length, 'L_physical': len(ids), 'L_test': int(r['length_cap']),
                'L_phase': None, 'evidence_distance': None, 'distractor_count': None,
                'metric_scope': 'constructed source-grounded development tasks; not full official benchmark'}
        if r['split'] == 'train':
            train_ids.add(r['semantic_id']); train_sources.update(r.get('source_lineages', []))
            if r['layout'] in ('near', 'far'):
                if not r['target_ids'] or r['target_ids'][-1] != tokenizer.eos_token_id:
                    raise ValueError('training answer lacks EOS')
                sft_rows.append(item)
        else:
            val_ids.add(r['semantic_id']); val_sources.update(r.get('source_lineages', []))
            eval_rows.append(item)
            if r['layout'] == 'compact':
                if '\n\nQuestion:' not in content:
                    raise ValueError('cannot locate question for deleted-source control')
                question = content.rsplit('\n\nQuestion:', 1)[1]
                deleted = chat_ids(tokenizer, 'Use only the supplied passages. No passages are supplied.\n\nQuestion:'+question)
                eval_rows.append({**item, 'layout':'deleted', 'prompt_ids':deleted,
                    'row_id':sha_json([item['row_id'],'deleted']), 'prompt_sha256':sha_json(deleted),
                    'input_tokens':len(deleted), 'L_physical':len(deleted), 'source_block':None,
                    'control':'same question with all source passages removed; original targets retained for leakage diagnosis'})
    if train_ids & val_ids or train_sources & val_sources:
        raise ValueError('training/validation semantic or source-lineage overlap')
    validate_rows(eval_rows)
    # Near/far/compact must refer to the same worlds. Deleted prompts must hide world.
    by_group = defaultdict(list)
    for r in eval_rows:
        by_group[r['group_id']].append(r)
    for gid, rs in by_group.items():
        if {r['layout'] for r in rs} != {'near','far','compact','deleted'}:
            raise ValueError(f'incomplete control layouts: {gid}')
        deleted = [r['prompt_ids'] for r in rs if r['layout']=='deleted']
        if len(deleted)!=2 or deleted[0]!=deleted[1]:
            raise ValueError('deleted-source control reveals world identity')
    for name, rs in [('eval_rows.jsonl',eval_rows),('sft_rows.jsonl',sft_rows)]:
        with (out/name).open('x') as f:
            for r in rs: f.write(json.dumps(r)+'\n')
    assets = {str(p.resolve()): sha_file(p) for p in [cfg_path, Path(args.transport),
        *(p for p in model.iterdir() if p.suffix in ('.json','.safetensors') and p.is_file())]}
    outputs = {str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file()}
    write_json(out/'eval_manifest.json', dict(status='CPU_FROZEN_NOT_QUALIFIED',
        role='development; earlier exposure not excluded; not independent confirmation',
        eval_rows=len(eval_rows), train_rows=len(sft_rows), train_groups=len(train_ids),
        validation_groups=len(val_ids), source_lineage_overlap=0,
        template_sha256=sha_json(tokenizer.chat_template), eos_token_id=tokenizer.eos_token_id,
        tokenizer_files={p.name:sha_file(p) for p in model.glob('*token*.json')},
        length_fields='actual prompt+reserve checked; unknown evidence/phase fields remain null'))
    outputs['eval_manifest.json'] = sha_file(out/'eval_manifest.json')
    write_json(out/'assets_and_missing.json', dict(assets=assets, prepared_files=outputs,
        missing=['independent E5 confirmation data/model',
                 'external verify_theory.py/select_allocation.py package'],
        scratch_scope='author selected existing seeds 137/256 for new overlays; historical seed42 results reused without weight recovery',
        gpu_checks='not run; no-card preparation', model=str(model.resolve())))
    print(json.dumps(dict(status='PREPARED', eval_rows=len(eval_rows), sft_rows=len(sft_rows),
                         prepared_files=outputs), indent=2))


def rescore(args):
    tasks = list(read_rows(args.tasks))
    # Legacy task ids collide across near/far, so always join the complete key.
    def key(r): return (r['row_id'], r['family'], r['layout'], r['length_cap'], str(r['world']))
    indexed = {key(r):r for r in tasks}
    if len(indexed)!=len(tasks): raise ValueError('ambiguous task identities')
    result=[]
    for path in args.outputs:
        counts=Counter()
        for r in read_rows(path):
            if key(r) not in indexed: raise ValueError('output cannot be joined to task')
            new = score(indexed[key(r)], r['unmodified_output_text'], r['ended_with_eos'])
            counts['rows']+=1
            counts['legacy_strict']+=bool(r['strict_exact_eos'])
            counts['corrected_full_eos']+=new['full_answer_exact_eos']
            counts['changed']+=bool(r['strict_exact_eos'])!=new['full_answer_exact_eos']
        result.append(dict(source=str(path),sha256=sha_file(path),counts=dict(counts)))
    write_json(args.out,dict(status='SAVED_TEXT_RESCORE_ONLY',tasks_sha256=sha_file(args.tasks),results=result,
        limits='No model rerun; legacy missing tokens/EOS metadata cannot be reconstructed; no scientific promotion'))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze')
    for a in ('model','old-tables','transport','out'): f.add_argument('--'+a,required=True)
    r=sub.add_parser('rescore');r.add_argument('--tasks',required=True);r.add_argument('--outputs',nargs='+',required=True);r.add_argument('--out',required=True)
    a=p.parse_args(); freeze(a) if a.command=='freeze' else rescore(a)


if __name__=='__main__': main()
