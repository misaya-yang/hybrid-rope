"""Fixed oracle-support DEV diagnostic: deny direct reads of question-key slots."""
from contextlib import contextmanager
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from .balanced_queries import BalancedValueSession
from .question_state_cross import append_question_state
from .run import digest, model_identity, records, write_json
from .target_record_oracle import OracleConfig, ROW_IDS, score


@contextmanager
def deny_question_reads(branch, physical_slots):
    slots = tuple(sorted(set(physical_slots)))
    initial = branch.cache.get_seq_length()
    if any(i < 0 or i >= initial for i in slots):
        raise ValueError('mask must address existing physical slots')
    receipt = {'denied_physical_slots': slots, 'calls': 0, 'mask_lengths': []}
    def hook(model, args, kwargs):
        if kwargs.get('past_key_values') is not branch.cache or args[0].shape[1] != 1:
            raise RuntimeError('mask used outside its single-token branch')
        length = branch.cache.get_seq_length() + 1
        mask = torch.ones((1, 1, 1, length), dtype=torch.bool, device=branch.device)
        mask[..., list(slots)] = False
        kwargs['attention_mask'] = {'full_attention': mask}
        receipt['calls'] += 1
        receipt['mask_lengths'].append(length)
        return args, kwargs
    handle = branch.model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield receipt
    finally:
        handle.remove()


def question_masks(row, tokenizer):
    text = row['prefix_text'] + row['suffix_text']
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if enc['input_ids'] != row['prompt_ids']:
        raise ValueError('prompt retokenization mismatch')
    boundary, t = len(row['prefix_text']), len(row['prefix_ids'])
    key = row['query']['keys'][0]
    suffix = row['suffix_text']
    if suffix.count(key) != 1:
        raise ValueError('require unique key occurrence in question')
    lo = boundary + suffix.index(key)
    hi = lo + len(key)
    key_slots = [i-t for i,(s,e) in enumerate(enc['offset_mapping']) if e > lo and s < hi]
    n = len(row['suffix_ids']) - 1
    if not key_slots or not all(0 <= i < n for i in key_slots):
        raise ValueError('key must be entirely within prior question slots')
    choices = [i for i in range(n) if i not in key_slots and row['suffix_ids'][i] not in tokenizer.all_special_ids]
    sham = sorted(random.Random(20260910).sample(choices, len(key_slots)))
    return {'key': key_slots, 'sham': sham, 'visible': [],
            'key_token_ids': [row['suffix_ids'][i] for i in key_slots],
            'sham_token_ids': [row['suffix_ids'][i] for i in sham]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'model', 'data', 'oracle-run', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--panel', choices=('initial','sham_split','punctuation_split','punctuation_combined'), default='initial')
    p.add_argument('--support', choices=('P','oracle_target_record'), default='oracle_target_record')
    a = p.parse_args()
    root, old, out = Path(a.root), Path(a.oracle_run), Path(a.output)
    original = json.loads((old/'contract.json').read_text())
    identity = model_identity(a.model)
    for k in ('config_sha256', 'tokenizer_sha256', 'weights'):
        if identity[k] != original['model'][k]:
            raise ValueError('model mismatch '+k)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    rows_by_id = {r['row_id']: r for r in records(a.data)}
    rows = [rows_by_id[r] for r in ROW_IDS]
    baseline = {r['row_id']:r for r in records(old/'per_example.jsonl') if r['arm']==a.support}
    supports, masks = {}, {}
    for row in rows:
        rid = row['row_id']
        if row['split'] != 'dev' or row['prefix_ids']+row['suffix_ids'] != row['prompt_ids']:
            raise ValueError('input contract')
        target = original['targets'][rid]
        for k in ('prefix_ids', 'prompt_ids'):
            if digest(row[k]) != target[k+'_sha256']:
                raise ValueError('input identity')
        indices = torch.load(old/(rid+'.keep_sets.pt'), map_location='cpu', weights_only=True)[a.support]
        if digest([x.tolist() for x in indices]) != baseline[rid]['keep_indices_sha256']:
            raise ValueError('support identity')
        supports[rid], masks[rid] = indices, question_masks(row, tokenizer)
    arms = {'initial':('visible','key','sham'), 'sham_split':('sham_words','sham_punctuation'), 'punctuation_split':('sham_colon','sham_newline'), 'punctuation_combined':('sham_punctuation',)}[a.panel]
    for row in rows:
        m = masks[row['row_id']]
        m['sham_words'] = [i for i in m['sham'] if tokenizer.decode([row['suffix_ids'][i]]).strip().isalpha()]
        m['sham_punctuation'] = [i for i in m['sham'] if i not in m['sham_words']]
        m['sham_colon'] = [i for i in m['sham_punctuation'] if tokenizer.decode([row['suffix_ids'][i]]) == ':']
        m['sham_newline'] = [i for i in m['sham_punctuation'] if tokenizer.decode([row['suffix_ids'][i]]) == '\n']
        m['sham_decoded_tokens'] = [tokenizer.decode([row['suffix_ids'][i]]) for i in m['sham']]
    contract = {'probe':'question_key_direct_read_mask_v1', 'panel':a.panel, 'support':a.support, 'model':identity, 'masks':masks,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'old_run':str(old), 'new_generations':len(rows)*len(arms), 'dtype':original['dtype'],
        'scope':'privileged oracle-support DEV diagnostic; changed reader attention; no deployment claim',
        'mask_phase':'final prompt token and answer generation only; own question state frozen',
        'sham_seed':20260910, 'prefix_keep_hashes':{rid:baseline[rid]['keep_indices_sha256'] for rid in ROW_IDS}}
    if not a.execute:
        print(json.dumps(contract)); return
    if (root/'STOP').exists() or out.exists() or not torch.cuda.is_available():
        raise RuntimeError('STOP, existing output, or missing CUDA')
    torch.set_num_threads(4)
    write_json(out/'contract.json', contract)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    started, completed = time.perf_counter(), []
    write_json(out/'status.json', {'status':'LOADING'})
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
            torch_dtype=getattr(torch,original['dtype']), attn_implementation='sdpa').cuda().eval()
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos,list) else [eos])
        with (out/'per_example.jsonl').open('w') as stream:
            for row in rows:
                if (root/'STOP').exists(): break
                rid, indices = row['row_id'], supports[row['row_id']]
                write_json(out/'status.json', {'status':'RUNNING','row_id':rid})
                session = BalancedValueSession(model,row['prefix_ids'],OracleConfig(**original['config'])).prefill()
                donor = session.branch(a.support, indices).consume(row['suffix_ids'][:-1])
                b, n = indices[0].shape[1], len(row['suffix_ids'])-1
                stream.write(json.dumps({**baseline[rid], 'reused_cell':True})+'\n'); stream.flush()
                for arm in arms:
                    branch = session.branch(a.support, indices)
                    splice = append_question_state(branch,donor,donor_prefix_slots=b,question_tokens=n)
                    with deny_question_reads(branch,[b+i for i in masks[rid][arm]]) as mask_receipt:
                        branch.consume(row['suffix_ids'][-1:])
                        generated = branch.generate(row['max_new_tokens'],eos)
                    result = {k:row.get(k) for k in ('row_id','task','split','family_id','material_cluster_id','expected','score_contract')}
                    result.update(arm=a.support+'_mask_question_'+arm, reused_cell=False,
                        keep_indices_sha256=baseline[rid]['keep_indices_sha256'],
                        generated_token_ids=generated['generated_ids'], generation=generated,
                        mask=mask_receipt, splice=splice, diagnostic_only=True,
                        **score(row,generated['generated_ids'],tokenizer,eos))
                    stream.write(json.dumps(result)+'\n'); stream.flush()
                    del branch
                completed.append(rid)
                del donor, session
        write_json(out/'status.json',{'status':'COMPLETE' if len(completed)==2 else 'STOPPED',
            'elapsed_seconds':time.perf_counter()-started,'completed_rows':completed})
    except BaseException as exc:
        write_json(out/'status.json',{'status':'FAILED','error':str(exc)}); raise

if __name__ == '__main__':
    main()
