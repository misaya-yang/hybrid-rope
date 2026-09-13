#!/usr/bin/env python3
"""OLMo BM/MrPro layer-group interventions on a fixed E2 subset, without training."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import time

TASKS = ('hotpotqa', '2wikimqa', 'qasper', 'narrativeqa', 'multifieldqa_en')
GROUPS = {'early': (0, 5), 'middle': (5, 11), 'late': (11, 16)}


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.incomplete')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def assignments():
    result = {name: [name] * 16 for name in ('bm_g4', 'mrpro_g4')}
    for base, replacement in (('bm_g4', 'mrpro_g4'), ('mrpro_g4', 'bm_g4')):
        for group, (start, stop) in GROUPS.items():
            values = [base] * 16
            values[start:stop] = [replacement] * (stop - start)
            result[f'{base}_with_{replacement}_{group}'] = values
    return result


def prepare(args):
    parent = json.loads((args.e2_prepared / 'manifest.json').read_text())
    tables = json.loads((args.e2_prepared / 'tables.json').read_text())
    selected = []
    counts = defaultdict(int)
    for row in read_rows(args.e2_prepared / 'inputs.jsonl'):
        if row['task'] in TASKS and row['input_tokens'] > 4096 and counts[row['task']] < 12:
            selected.append(row)
            counts[row['task']] += 1
    if any(counts[task] != 12 for task in TASKS):
        raise ValueError('need 12 long E2 inputs per task')
    if tables['bm_g4']['gain'] != tables['mrpro_g4']['gain']:
        raise ValueError('layer comparison requires matched gain')
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / 'inputs.jsonl').open('w') as stream:
        for row in selected:
            stream.write(json.dumps(row) + '\n')
    write(args.out / 'tables.json', {name: tables[name] for name in ('bm_g4', 'mrpro_g4')})
    generation = json.loads((args.e2_prepared / 'generation_config.json').read_text())
    write(args.out / 'generation_config.json', generation)
    manifest = {'status': 'PREPARED_NOT_RUN', 'model': parent['model'], 'rows': 60,
                'policies': assignments(), 'groups_zero_based_half_open': GROUPS,
                'new_generations_with_reused_baselines': 360, 'all_generations': 480,
                'selection': 'first 12 long inputs per E2 task, before reading model scores',
                'asset_identity_policy': 'user_attested_clone/no_sha_validation',
                'scope': 'E2 development mechanism subset; not independent confirmation or the full E7'}
    write(args.out / 'manifest.json', manifest)
    print(json.dumps(manifest))


class OLMoLayerTables:
    """Choose precomputed rotary embeddings at each attention call, including decode."""
    def __init__(self, model, tables, policy):
        from scripts.experiments.olmo_fast_screen.runtime import install
        if model.config.model_type != 'olmo2' or len(model.model.layers) != 16:
            raise ValueError('this intervention is for the retained 16-layer OLMo2')
        self.handles = []
        self.embeddings = {}
        self.counts = [0] * 16
        self.rotaries = {}
        self.base = policy[0]
        for name in dict.fromkeys(policy):
            install(model, tables[name])
            self.rotaries[name] = deepcopy(model.model.rotary_emb)
        install(model, tables[self.base])

        def cache(module, positional, keywords, output):
            self.embeddings = {self.base: output}
            for name, rotary in self.rotaries.items():
                if name != self.base:
                    self.embeddings[name] = rotary(*positional, **keywords)
        self.handles.append(model.model.rotary_emb.register_forward_hook(cache, with_kwargs=True))
        for index, (layer, name) in enumerate(zip(model.model.layers, policy)):
            def choose(module, positional, keywords, table_name=name, layer_index=index):
                if 'position_embeddings' not in keywords or table_name not in self.embeddings:
                    raise ValueError('unexpected OLMo rotary attention interface')
                replacement = dict(keywords)
                replacement['position_embeddings'] = self.embeddings[table_name]
                self.counts[layer_index] += 1
                return positional, replacement
            self.handles.append(layer.self_attn.register_forward_pre_hook(choose, with_kwargs=True))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.embeddings.clear()
        self.rotaries.clear()


def run(args):
    manifest = json.loads((args.prepared / 'manifest.json').read_text())
    if not args.execute:
        print(json.dumps({'status': 'PLAN_ONLY', 'rows': manifest['rows'], 'policies': list(manifest['policies']),
                          'generations': [360, 480], 'gpu_execution': False})); return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from experiments.olmo_recovery_20260912.runtime import validate_cuda
    from scripts.experiments.olmo_fast_screen.runtime import install
    from scripts.eval.longbench_metrics import qa_f1_score
    validate_cuda()
    data = read_rows(args.prepared / 'inputs.jsonl')
    tables = json.loads((args.prepared / 'tables.json').read_text())
    decoding_dict = json.loads((args.prepared / 'generation_config.json').read_text())
    decoding = GenerationConfig.from_dict(decoding_dict)
    model_path = args.model or Path(manifest['model']['path'])
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16,
                                                device_map={'': 'cuda'}, attn_implementation='sdpa').eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    eos = decoding.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    args.out.mkdir(parents=True, exist_ok=True)
    contract = {'policies': manifest['policies'], 'generation_config': decoding_dict,
                'rows': [row['row_id'] for row in data],
                'tables': {name: {'values': table['values_float32'], 'gain': table['gain']} for name, table in tables.items()}}
    contract_path = args.out / 'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('output contains a different layer/decoder/panel experiment')
    write(contract_path, contract)
    reused = {}
    if args.reuse_e2:
        previous_runtime = json.loads((args.reuse_e2 / 'runtime.json').read_text())
        previous_decoding = previous_runtime.get('generation_config')
        if previous_decoding is None:
            parent_prepared = Path(previous_runtime.get('prepared', args.e2_prepared or args.prepared))
            config_path = parent_prepared / 'generation_config.json'
            previous_decoding = json.loads(config_path.read_text())
        if previous_decoding != decoding_dict:
            raise ValueError('E2 generation settings differ; omit reuse to evaluate these baselines')
        for arm in ('bm_g4', 'mrpro_g4'):
            path = args.reuse_e2 / f'{arm}.jsonl'
            if path.exists():
                reused[arm] = {row['row_id']: row for row in read_rows(path)}
    started = time.monotonic()
    for name, policy in manifest['policies'].items():
        path = args.out / f'{name}.jsonl'
        saved = read_rows(path) if path.exists() else []
        if len(saved) > len(data) or any(r['row_id'] != data[i]['row_id'] or r['policy'] != name for i, r in enumerate(saved)):
            raise ValueError('saved rows are not the matching policy prefix')
        hook = None
        install(model, tables[policy[0]])
        if len(set(policy)) > 1:
            hook = OLMoLayerTables(model, tables, policy)
        try:
            with path.open('a') as stream, torch.inference_mode():
                for row in data[len(saved):]:
                    source = reused.get(name, {}).get(row['row_id'])
                    if source is not None:
                        if any(source[k] != row[k] for k in ('references', 'input_tokens', 'max_new_tokens', 'task')):
                            raise ValueError('E2 reuse row differs')
                        generated = source['generated_ids']
                        origin = 'reused_E2'
                    else:
                        ids = torch.tensor([row['prompt_ids']], device='cuda', dtype=torch.long)
                        generated = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=decoding,
                                                   max_new_tokens=row['max_new_tokens'])[0, ids.shape[1]:].tolist()
                        if hook and not all(hook.counts):
                            raise RuntimeError('one or more layer interventions did not execute')
                        origin = 'generated'
                    ended = bool(generated and generated[-1] in eos)
                    text = tokenizer.decode(generated[:-1] if ended else generated, skip_special_tokens=False)
                    result = {key: row[key] for key in ('row_id', 'task', 'references', 'input_tokens', 'max_new_tokens')}
                    result.update(document_cluster_id=row.get('document_cluster_id', row['row_id']), policy=name,
                                  generated_ids=generated, output_text=text, ended_eos=ended, origin=origin,
                                  whole_response_f1=qa_f1_score(text, row['references']))
                    stream.write(json.dumps(result) + '\n'); stream.flush()
                    write(args.out / 'live.json', {'policy': name, 'row_id': row['row_id']})
        finally:
            if hook:
                hook.close()
    write(args.out / 'status.json', {'status': 'COMPLETE', 'rows_per_policy': len(data), 'policies': list(manifest['policies']),
                                     'elapsed_seconds': time.monotonic() - started})


def score(args):
    manifest = json.loads((args.prepared / 'manifest.json').read_text())
    data = read_rows(args.prepared / 'inputs.jsonl')
    ids = {row['row_id'] for row in data}
    results = {}
    for policy in manifest['policies']:
        rows = read_rows(args.run / f'{policy}.jsonl')
        if len(rows) != len(ids) or {r['row_id'] for r in rows} != ids:
            raise ValueError(f'incomplete policy: {policy}')
        results[policy] = {r['row_id']: r for r in rows}
    summaries, contrasts = {}, {}
    for policy, values in results.items():
        cells = {task: [r['whole_response_f1'] for r in values.values() if r['task'] == task] for task in TASKS}
        by_task = {task: sum(v)/len(v) for task, v in cells.items()}
        summaries[policy] = {'five_task_macro_f1': sum(by_task.values())/len(TASKS), 'by_task': by_task}
        if '_with_' not in policy:
            continue
        base = policy.split('_with_')[0]
        clusters = {}
        for task in TASKS:
            groups = defaultdict(list)
            for row in data:
                if row['task'] == task:
                    groups[row.get('document_cluster_id', row['row_id'])].append(
                        values[row['row_id']]['whole_response_f1'] - results[base][row['row_id']]['whole_response_f1'])
            clusters[task] = [sum(v)/len(v) for v in groups.values()]
        rng = random.Random(20260912)
        draws = sorted(sum(sum(rng.choices(v, k=len(v)))/len(v) for v in clusters.values())/len(TASKS) for _ in range(2000))
        contrasts[policy + '_minus_' + base] = {
            'macro_delta': summaries[policy]['five_task_macro_f1'] - summaries[base]['five_task_macro_f1'],
            'cluster_bootstrap_ci95': [draws[50], draws[1949]]}
    write(args.out, {'status': 'COMPLETE', 'scope': 'E2 mechanism subset, not independent confirmation',
                     'summaries': summaries, 'contrasts': contrasts})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--e2-prepared', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('run'); p.add_argument('--prepared', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', type=Path); p.add_argument('--reuse-e2', type=Path); p.add_argument('--e2-prepared', type=Path); p.add_argument('--execute', action='store_true')
    p = sub.add_parser('score'); p.add_argument('--prepared', type=Path, required=True); p.add_argument('--run', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); {'prepare': prepare, 'run': run, 'score': score}[args.command](args)


if __name__ == '__main__':
    main()
