"""Frozen-model full-generation screening of explicit static per-layer tables."""
import argparse
from contextlib import nullcontext, ExitStack
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .bench import digest
from .diagnose import greedy
from .layer_policy import LayerTablePolicy
from .prepare import sha_file, verify_weight_stats
from .run import atomic
from .ruler_bench import score
from .runtime import install, verify
from .causal_gain import CausalGain
from .chunked_mlp import tokenwise_mlp_chunks, qualify_chunking
from .bias_position import BiasPositionTerm
from scripts.eval.longbench_metrics import qa_f1_score
from scripts.experiments.scale_transport.position_visibility import check_decoder


def work(args):
    started = time.monotonic()
    prepared, out = args.prepared, args.out
    manifest = json.loads((prepared/'manifest.json').read_text())
    natural = manifest.get('benchmark') == 'longbench_natural_v1'
    for name, expected in manifest['prepared_files'].items():
        if sha_file(prepared/name) != expected:
            raise ValueError('frozen prepared inputs changed')
    verify_weight_stats(manifest)
    for name, expected in manifest['model_files_sha256'].items():
        if sha_file(Path(manifest['model_path'])/name) != expected:
            raise ValueError('model metadata changed')
    params = json.loads((prepared/'generation_config.json').read_text())
    checked = dict(params)
    for name, default in dict(encoder_no_repeat_ngram_size=0,
            encoder_repetition_penalty=1., remove_invalid_values=False).items():
        if checked.get(name) is None:
            checked[name] = default
    check_decoder(checked)
    if params['repetition_penalty'] != 1.:
        raise ValueError('requires unit repetition penalty')
    rows = list(map(json.loads, (prepared/'screen.jsonl').read_text().splitlines()))
    if args.rows:
        by_id = {r['row_id']: r for r in rows}
        rows = [by_id[key] for key in args.rows]
    if args.exclude_rows:
        rows = [r for r in rows if r['row_id'] not in args.exclude_rows]
    if not rows or len({r['row_id'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate rows')
    tables = json.loads((prepared/'tables.json').read_text())
    if args.extra_tables:
        extra = json.loads(args.extra_tables.read_text())
        if tables.keys() & extra.keys():
            raise ValueError('extra tables cannot overwrite frozen references')
        tables.update(extra)
    spec = json.loads(args.spec.read_text())
    methods = spec['methods']
    if len({digest(v) for v in methods.values()}) != len(methods):
        raise ValueError('duplicate policies')
    cached = {}
    reused_files = {}
    if args.reuse_generations:
        previous = json.loads((args.reuse_generations/'runtime.json').read_text())
        expected = dict(model_id=manifest['model_id'], revision=manifest['revision'],
            generation_config=params, spec=spec, mlp_chunk_size=args.mlp_chunk_size,
            table_identity={name:digest(table) for name,table in tables.items()})
        for key,value in expected.items():
            if previous.get(key) != value:
                raise ValueError('reused generation scientific contract differs: '+key)
        by_id = {r['row_id']:r for r in rows}
        eos = params['eos_token_id']
        eos = {eos} if isinstance(eos,int) else set(eos)
        for name in methods:
            path = args.reuse_generations/(name+'.jsonl')
            if not path.exists():
                continue
            reused_files[name] = sha_file(path)
            cached[name] = {}
            for old in map(json.loads,path.read_text().splitlines()):
                row = by_id[old['row_id']]
                if old['row_id'] in cached[name] or old['method'] != name:
                    raise ValueError('duplicate or wrong reused generation')
                for key in ('prompt_sha256','input_tokens','task','length_cap','max_new_tokens','references'):
                    if old[key] != row[key]:
                        raise ValueError('reused generation input differs: '+key)
                score_fn = qa_f1_score if natural else score
                actual_score = score_fn(old['output_text'],row['references']) if natural else score_fn(row,old['output_text'])
                if abs(actual_score-old['correct']) > 1e-12:
                    raise ValueError('reused generation score differs')
                if (not old['generated_ids'] or len(old['generated_ids']) > row['max_new_tokens']
                    or old['ended_eos'] != (old['generated_ids'][-1] in eos)):
                    raise ValueError('reused generation termination differs')
                cached[name][old['row_id']] = dict(old,reused_from=str(args.reuse_generations))
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader'], text=True)
    if any(x.strip().isdigit() for x in active.splitlines()):
        raise RuntimeError('another GPU process is active')
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    model = AutoModelForCausalLM.from_pretrained(manifest['model_path'], local_files_only=True,
        dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='sdpa').eval()
    if sum(p.numel() for p in model.parameters()) != manifest['actual_parameters']:
        raise ValueError('model count mismatch')
    for assignment in methods.values():
        if len(assignment) != len(model.model.layers) or not set(assignment) <= tables.keys():
            raise ValueError('invalid per-layer policy')
    tokenizer = AutoTokenizer.from_pretrained(manifest['model_path'], local_files_only=True)
    chunk_qualification = qualify_chunking(model, rows[0]['prompt_ids'], args.mlp_chunk_size)
    import transformers
    root = Path(__file__).resolve().parents[3]
    deps = set(manifest['code_files']) | {
        'scripts/experiments/olmo_fast_screen/layer_screen.py',
        'scripts/experiments/olmo_fast_screen/layer_policy.py',
        'scripts/experiments/olmo_fast_screen/causal_gain.py',
        'scripts/experiments/olmo_fast_screen/chunked_mlp.py',
        'scripts/experiments/olmo_fast_screen/bias_position.py',
        'scripts/lib/rope/gap_capped.py',
        'scripts/experiments/olmo_fast_screen/diagnose.py',
        'scripts/experiments/scale_transport/position_visibility.py'}
    if natural:
        deps.add('scripts/eval/longbench_metrics.py')
    atomic(out/'runtime.json', dict(parent_manifest_sha256=sha_file(prepared/'manifest.json'),
        model_id=manifest['model_id'], revision=manifest['revision'],
        code_files={name: sha_file(root/name) for name in deps},
        spec=spec, spec_sha256=sha_file(args.spec), torch=torch.__version__,
        extra_tables_sha256=sha_file(args.extra_tables) if args.extra_tables else None,
        transformers=transformers.__version__, backend='Flash SDPA only',
        mlp_chunk_size=args.mlp_chunk_size,
        mlp_chunk_qualification=chunk_qualification,
        reused_generations_sha256=reused_files,
        allocator_configuration=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
        row_ids=[r['row_id'] for r in rows], generation_config=params,
        table_identity={name: digest(table) for name, table in tables.items()}))
    count = 0

    def generate(name, row):
        nonlocal count
        if (out/'STOP').exists():
            raise RuntimeError('operator stop')
        atomic(out/'live.json', dict(method=name, row_id=row['row_id'], completed=count))
        ids = row['prompt_ids']; begin = time.monotonic()
        if natural or manifest.get('use_stock_generation'):
            inputs = torch.tensor([ids], device=model.device)
            generated = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                generation_config=GenerationConfig.from_dict(params),
                max_new_tokens=row['max_new_tokens'])[0, len(ids):].tolist()
            eos = params['eos_token_id']
            eos = {eos} if isinstance(eos, int) else set(eos)
            data = dict(generated_ids=generated, ended_eos=generated[-1] in eos)
        else:
            data, cache = greedy(model, ids, list(range(len(ids))),
                max_new_tokens=row['max_new_tokens'], eos_token_id=params['eos_token_id'])
            del cache
        tokens = data['generated_ids']
        text = tokenizer.decode(tokens[:-1] if data['ended_eos'] else tokens, skip_special_tokens=False)
        data.update(method=name, row_id=row['row_id'], prompt_sha256=row['prompt_sha256'],
            input_tokens=len(ids), task=row['task'], length_cap=row['length_cap'],
            max_new_tokens=row['max_new_tokens'], references=row['references'],
            correct=qa_f1_score(text, row['references']) if natural else score(row, text),
            output_text=text, elapsed_seconds=time.monotonic()-begin)
        count += 1
        print(json.dumps({k: data[k] for k in ('method','row_id','correct','elapsed_seconds')}), flush=True)
        return data

    with torch.inference_mode(), tokenwise_mlp_chunks(model, args.mlp_chunk_size):
        if args.qualify_baseline:
            qualification = []
            for arm in ('MrPro', 'MrProBM'):
                saved = {r['row_id']: r for r in map(json.loads,
                    (args.qualify_baseline/(arm+'.jsonl')).read_text().splitlines())}
                row = rows[0]
                with LayerTablePolicy(model, tables, [arm]*len(model.model.layers)):
                    data = generate('qualify_'+arm, row)
                if data['generated_ids'] != saved[row['row_id']]['generated_ids']:
                    raise ValueError('uniform layer policy does not reproduce saved full output')
                qualification.append(data)
            atomic(out/'qualification.json', dict(status='PASS', results=qualification))
        for name, assignment in methods.items():
            records = []
            bias_policy = None
            if len(set(assignment)) == 1:
                # A single global table uses stock attention without layer hooks.
                install(model, tables[assignment[0]]); verify(model, tables[assignment[0]])
                policy = nullcontext()
            else:
                policy = LayerTablePolicy(model, tables, assignment)
            with ExitStack() as stack:
                stack.enter_context(policy)
                gain_spec = spec.get('causal_gain', {}).get(name)
                bias_spec = spec.get('bias_position', {}).get(name)
                if bias_spec:
                    if len(set(assignment)) != 1 or gain_spec:
                        raise ValueError('bias-position comparison uses one fixed table and gain')
                    bias_policy = stack.enter_context(BiasPositionTerm(model,read_table=tables[assignment[0]],
                        prior_table=tables[bias_spec['prior_table']],mode=bias_spec.get('mode','bias')))
                if gain_spec:
                    if len(set(assignment)) != 1:
                        raise ValueError('causal gain comparison uses one fixed table')
                    stack.enter_context(CausalGain(model, installed_gain=tables[assignment[0]]['gain'],
                        **gain_spec))
                with (out/(name+'.jsonl')).open('x') as stream:
                    for row in rows:
                        data = cached.get(name,{}).get(row['row_id'])
                        if data is None:
                            data = generate(name, row)
                        stream.write(json.dumps(data)+'\n'); stream.flush()
                        records.append(data)
            atomic(out/(name+'.json'), dict(status='COMPLETE', rows=len(records),
                row_ids=[r['row_id'] for r in records], raw_sha256=sha_file(out/(name+'.jsonl')),
                assignments=assignment, score_sum=sum(r['correct'] for r in records),
                bias_position_receipt=bias_policy.receipt() if bias_policy else None))
    atomic(out/'status.json', dict(status='COMPLETE', generations=count,
        reused_generations=sum(len(v) for v in cached.values()),
        elapsed_seconds=time.monotonic()-started, peak_memory_bytes=torch.cuda.max_memory_allocated()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--extra-tables', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rows', nargs='+')
    p.add_argument('--exclude-rows', nargs='+')
    p.add_argument('--qualify-baseline', type=Path)
    p.add_argument('--mlp-chunk-size', type=int, default=0)
    p.add_argument('--reuse-generations', type=Path)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    atomic(args.out/'status.json', dict(status='RUNNING', pid=os.getpid(), started_unix=time.time()))
    try:
        work(args)
    except BaseException as exc:
        atomic(args.out/'status.json', dict(status='FAILED', error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
