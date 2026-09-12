"""Prepare official RULER tasks while reusing the existing model/table preparation."""
import argparse
import contextlib
import runpy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .bench import digest
from .prepare import sha_file, write, verify_weight_stats
from .ruler_bench import TASKS, FAMILIES, score

UPSTREAM_REVISION = 'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a'
SEED = 20260909


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-prepared', required=True, type=Path)
    parser.add_argument('--upstream', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--short-count', type=int, default=2)
    parser.add_argument('--long-count', type=int, default=4)
    parser.add_argument('--long-cap', type=int, default=16384)
    parser.add_argument('--short-cap', type=int, default=4096)
    parser.add_argument('--qa-offset', type=int, default=0)
    parser.add_argument('--reuse-nonqa', type=Path)
    parser.add_argument('--tasks', nargs='+', default=list(TASKS))
    args = parser.parse_args()
    if args.short_count < 0 or args.long_count < 1:
        parser.error('short count must be nonnegative and long count positive')
    if args.short_cap <= 0 or args.long_cap <= args.short_cap:
        parser.error('positive short cap and larger long cap required')
    cells = tuple((cap, count) for cap, count in
        ((args.short_cap, args.short_count), (args.long_cap, args.long_count)) if count)
    if args.qa_offset < 0:
        parser.error('QA offset must be nonnegative')
    if args.reuse_nonqa:
        reuse_manifest = json.loads((args.reuse_nonqa/'manifest.json').read_text())
        if reuse_manifest['seed'] != args.seed or reuse_manifest['samples_per_task_by_cap'] != {str(k):v for k,v in cells}:
            raise ValueError('reused generator seed/count mismatch')
    old, upstream, out = (p.resolve() for p in (args.reuse_prepared, args.upstream, args.out))
    manifest = json.loads((old/'manifest.json').read_text())
    model_path = Path(manifest['model_path'])
    verify_weight_stats(manifest)
    for name, expected in manifest['model_files_sha256'].items():
        if sha_file(model_path/name) != expected:
            raise ValueError('model metadata changed: ' + name)
    for name in ('tables.json', 'queue.json', 'generation_config.json'):
        if sha_file(old/name) != manifest['prepared_files'][name]:
            raise ValueError('existing preparation changed: ' + name)
    out.mkdir(parents=True, exist_ok=False)
    os.environ['USE_TORCH'] = '0'
    os.environ['USE_TF'] = '0'
    import yaml
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    definitions = yaml.safe_load((upstream/'scripts/synthetic.yaml').read_text())
    tasks = args.tasks
    if len(set(tasks)) != len(tasks) or not set(tasks) <= definitions.keys():
        raise ValueError('unknown or duplicate RULER tasks')
    families = {task: FAMILIES.get(task, 'retrieval' if task.startswith('niah_') else
        'aggregation' if task in ('cwe', 'fwe') else 'qa') for task in tasks}
    constants = load_module('ruler_data_constants', upstream/'scripts/data/synthetic/constants.py')
    metrics = load_module('ruler_metrics', upstream/'scripts/eval/synthetic/constants.py')
    all_rows = []
    source_rows = []
    for cap, count in cells:
        for task in tasks:
            config = definitions[task]
            base = constants.TASKS[config['task']]
            budget = base['tokens_to_generate']
            template = tokenizer.apply_chat_template(
                [{'role':'user', 'content':base['template']}], tokenize=False,
                add_generation_prompt=True) + base.get('answer_prefix', '')
            command = [sys.executable, str(upstream/f"scripts/data/synthetic/{config['task']}.py"),
                '--save_dir', str(out/'source'/str(cap)), '--save_name', task,
                '--subset', 'validation', '--tokenizer_path', str(model_path),
                '--tokenizer_type', 'hf', '--max_seq_length', str(cap),
                '--tokens_to_generate', str(budget), '--num_samples', str(count),
                '--random_seed', str(args.seed), '--template', template]
            for name, value in config['args'].items():
                command.extend(['--'+name, str(value)])
            if task.startswith('qa_'):
                command.extend(['--pre_samples', str(args.qa_offset)])
            # No-GPU instances have only 2 GB RAM. Run the unmodified generator
            # sequentially in this process instead of duplicating HF dependencies.
            previous_argv, previous_path, previous_cwd = sys.argv, sys.path[:], os.getcwd()
            reuse_source = args.reuse_nonqa/'source'/str(cap)/task/'validation.jsonl' if args.reuse_nonqa and not task.startswith('qa_') else None
            if reuse_source:
                destination = out/'source'/str(cap)/task/'validation.jsonl'
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(reuse_source, destination)
            with (out/f'{cap}_{task}.log').open('x') as log:
                try:
                    sys.argv = command[1:]
                    sys.path.insert(0, str(upstream/'scripts/data/synthetic'))
                    os.chdir(upstream)
                    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        if reuse_source:
                            print('Reused unchanged official source:', reuse_source)
                        else:
                            runpy.run_path(command[1], run_name='__main__')
                finally:
                    sys.argv, sys.path = previous_argv, previous_path
                    os.chdir(previous_cwd)
            path = out/'source'/str(cap)/task/'validation.jsonl'
            generated = [json.loads(line) for line in path.read_text().splitlines() if line]
            if len(generated) != count:
                raise ValueError('generator did not produce the requested count')
            for index, raw in enumerate(generated):
                text = raw['input'] + raw.get('answer_prefix', '')
                ids = tokenizer.encode(text, add_special_tokens=False)
                # Upstream reduces the haystack in coarse increments; valid
                # official rows can fall below 90% of the requested length cap.
                # Keep those rows unchanged and report their realized length.
                if len(ids) + budget > cap or not ids:
                    raise ValueError(f'{task}/{cap}: input length or generation reserve invalid: {len(ids)}')
                refs = raw['outputs']
                if not refs or any(not isinstance(ref, str) or not ref.strip() for ref in refs):
                    raise ValueError('invalid reference answers')
                row = dict(row_id=f'{task}_{cap}_{index}', task=task, family=families[task],
                    upstream_index=raw['index'], length_cap=cap, prompt_ids=ids,
                    prompt_sha256=digest(ids), input_tokens=len(ids), references=refs,
                    max_new_tokens=budget)
                metric = metrics.string_match_part if task.startswith('qa_') else metrics.string_match_all
                for prediction in ('', refs[0], ' '.join(refs), 'irrelevant answer'):
                    if round(score(row, prediction)*100, 2) != metric([prediction], [refs]):
                        raise AssertionError('scorer differs from upstream')
                all_rows.append(row)
                source_rows.append(dict(row_id=row['row_id'], prompt_text=text, references=refs,
                                        upstream_index=raw['index']))
            print(json.dumps(dict(task=task, length_cap=cap, rows=count)), flush=True)
    if len({r['row_id'] for r in all_rows}) != len(all_rows):
        raise ValueError('duplicate row identities')
    with (out/'screen.jsonl').open('x') as stream:
        for row in all_rows:
            stream.write(json.dumps(row)+'\n')
    with (out/'prompts.jsonl').open('x') as stream:
        for row in source_rows:
            stream.write(json.dumps(row)+'\n')
    (out/'qualification.jsonl').write_text('')
    for name in ('tables.json', 'queue.json', 'generation_config.json'):
        shutil.copyfile(old/name, out/name)
    # Each task uses its original RULER output budget in the runner.
    decoding = json.loads((out/'generation_config.json').read_text())
    decoding.pop('max_new_tokens', None)
    write(out/'generation_config.json', decoding)
    root = Path(__file__).resolve().parents[3]
    dependencies = list(manifest['code_files'])
    dependencies += ['scripts/experiments/olmo_fast_screen/'+name for name in ('ruler_bench.py', 'prepare_ruler.py', 'runtime.py')]
    sources = {str(path.relative_to(upstream)): sha_file(path) for path in (upstream/'scripts').rglob('*')
               if path.is_file() and path.suffix in ('.py', '.yaml')}
    source_assets = {name: sha_file(upstream/'scripts/data/synthetic/json'/name)
                     for name in ('PaulGrahamEssays.json', 'english_words.json', 'squad.json')}
    candidate_ids=[item['id'] for item in json.loads((out/'queue.json').read_text())['ordered_candidates']
                   if item['eligible']]
    manifest.update(benchmark='ruler_mixed_v1', status='PREPARED_GPU_NOT_RUN',
        physical_caps=[cap for cap, _ in cells], tasks=tasks, families=families,
        use_stock_generation=True,
        seed=args.seed, qa_offset=args.qa_offset, samples_per_task_by_cap=dict(cells), screen_rows=len(all_rows), qualification_rows=0,
        reused_nonqa_manifest_sha256=sha_file(args.reuse_nonqa/'manifest.json') if args.reuse_nonqa else None,
        screen_input_tokens=sum(r['input_tokens'] for r in all_rows),
        screen_min_max_tokens=[min(r['input_tokens'] for r in all_rows), max(r['input_tokens'] for r in all_rows)],
        row_order=[r['row_id'] for r in all_rows], prompt_collection_sha256=digest([r['prompt_ids'] for r in all_rows]),
        upstream_revision_label=UPSTREAM_REVISION, upstream_source_files=sources, upstream_assets=source_assets,
        inputs_reused_from_manifest_sha256=None,
        scoring='Official RULER match-all, QA match-any; fractional per-row scores; equal task weights within each length.',
        qualification='No self-made Native gate. Report MrPro floor/ceiling cells and all selected tasks without outcome-based removal.',
        selection=f'Primary: {args.long_cap}-token-cap six-task macro gain; report {args.short_cap}-token cap separately. Positive long gain with nonnegative short change is a development win; short loss is a tradeoff.',
        scope=f'RULER {len(tasks)}-task panel, {len(tasks)*args.short_count} rows at {args.short_cap}-token cap and {len(tasks)*args.long_count} at {args.long_cap}-token cap. Sample scope and confirmation role are declared in the experiment protocol.',
        gpu_execution=('NOT_RUN; one MrPro baseline and fixed candidates '
                       + ','.join(candidate_ids)),
        code_files={name:sha_file(root/name) for name in dependencies})
    manifest['prepared_files'] = {name:sha_file(out/name) for name in
        ('screen.jsonl','qualification.jsonl','tables.json','generation_config.json','queue.json','prompts.jsonl')}
    write(out/'manifest.json', manifest)
    print(json.dumps({k:manifest[k] for k in ('status','tasks','screen_rows','screen_input_tokens','screen_min_max_tokens')}, indent=2))


if __name__ == '__main__':
    main()
