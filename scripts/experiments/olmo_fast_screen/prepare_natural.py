"""Reuse source-verified, untruncated LongBench prompts for OLMo BM task transfer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from transformers import AutoTokenizer

from .bench import digest
from .prepare import sha_file, verify_weight_stats, write

TASKS = ('hotpotqa', '2wikimqa', 'qasper')
BUDGETS = {'hotpotqa': 32, '2wikimqa': 32, 'qasper': 128, 'narrativeqa': 128, 'multifieldqa_en': 64}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--tasks', nargs='+', choices=list(BUDGETS), default=list(TASKS))
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit-per-task', type=int, default=0)
    p.add_argument('--offset-per-task', type=int, default=0)
    args = p.parse_args()
    parent = args.parent
    manifest = json.loads((parent/'manifest.json').read_text())
    verify_weight_stats(manifest)
    tokenized = args.data_root/'tokenized/olmo2_target_free'
    source_manifest = json.loads((tokenized/'token_manifest.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(manifest['model_path'], local_files_only=True)
    rows = []
    sources = {}
    with zipfile.ZipFile(args.data_root/'longbench/data.zip') as archive:
        for task in args.tasks:
            cell = source_manifest['longbench']['cells']['longbench:'+task]
            path = tokenized/cell['rows_path']
            if sha_file(path) != cell['rows_sha256']:
                raise ValueError('frozen source tokens changed')
            raw = list(map(json.loads, archive.read('data/'+task+'.jsonl').decode().splitlines()))
            candidates = list(map(json.loads, path.read_text().splitlines()))
            selected = []
            for old in candidates:
                ids = old['input_ids']; original = raw[old['source_row_index']]
                if old['input_tokens'] != len(ids):
                    raise ValueError('token count differs')
                if hashlib.sha256(original['context'].encode()).hexdigest() != old['source_context_sha256']:
                    raise ValueError('source document differs')
                if old['references'] != original['answers']:
                    raise ValueError('answers differ from original source')
                rendered = tokenizer.decode(ids, skip_special_tokens=False)
                if (hashlib.sha256(rendered.encode()).hexdigest() != old['rendered_chat_prompt_sha256']
                        or original['context'] not in rendered or original['input'] not in rendered):
                    raise ValueError('chat/tokenizer roundtrip or complete source differs')
                if len(ids)+BUDGETS[task] > 16384:
                    continue
                selected.append(dict(row_id=task+'_'+str(old['source_row_index']),
                    source_id=old['source_id'], source_row_index=old['source_row_index'],
                    task=task, family='natural_qa', length_cap=16384,
                    native_stratum='native' if len(ids) <= 4096 else 'extended',
                    prompt_ids=ids, prompt_sha256=digest(ids), input_tokens=len(ids),
                    max_new_tokens=BUDGETS[task], references=old['references']))
            # Stable pre-existing source-hash order, no outcome-based selection.
            selected = selected[args.offset_per_task:]
            if args.limit_per_task:
                selected = selected[:args.limit_per_task]
            rows.extend(selected)
            sources[task] = dict(source_tokens_sha256=sha_file(path), original_rows=len(raw),
                frozen_available_rows=len(candidates), selected_rows=len(selected))
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ('tables.json', 'generation_config.json'):
        if sha_file(parent/name) != manifest['prepared_files'][name]:
            raise ValueError('parent table/decoder changed')
        shutil.copyfile(parent/name, args.out/name)
    with (args.out/'screen.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row)+'\n')
    code_root = Path(__file__).resolve().parents[3]
    dependencies = [
        'scripts/experiments/olmo_fast_screen/'+name for name in (
            'prepare.py','prepare_natural.py','bench.py','run.py','runtime.py','ruler_bench.py',
            'diagnose.py','layer_screen.py','layer_policy.py','causal_gain.py')]
    dependencies += ['scripts/experiments/cross_audit/tables.py',
        'scripts/experiments/scale_transport/position_visibility.py',
        'scripts/lib/rope/official_yarn.py','scripts/lib/rope/gap_capped.py',
        'scripts/eval/longbench_metrics.py']
    manifest.update(benchmark='longbench_natural_v1', status='PREPARED_GPU_NOT_RUN',
        source_manifest_sha256=sha_file(tokenized/'token_manifest.json'), source_tasks=sources,
        tasks=args.tasks, screen_rows=len(rows),
        source_scope='All selected frozen untruncated natural prompts fitting 16K with official task budgets. Not all LongBench tasks or exact-16K evidence distances.',
        limit_per_task=args.limit_per_task, offset_per_task=args.offset_per_task, task_budgets=BUDGETS,
        code_files={name: sha_file(code_root/name) for name in dependencies},
        prepared_files={name: sha_file(args.out/name) for name in ('tables.json','generation_config.json','screen.jsonl')})
    write(args.out/'manifest.json', manifest)
    write(args.out/'spec.json', dict(methods={'MrPro': ['MrPro']*16, 'MrProBM': ['MrProBM']*16},
        scope='New natural-task baseline once, archived; then the unchanged previously validated BM. No new frequency candidate.'))
    print(json.dumps(dict(rows=len(rows), tasks=sources), indent=2))


if __name__ == '__main__':
    main()
