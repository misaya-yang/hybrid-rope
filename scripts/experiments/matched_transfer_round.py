#!/usr/bin/env python3
"""Prepare or execute one frozen N_compact/Z/Y round; no test, resume or sweep.

Standard-library planner. Actual asset/tokenizer/cache preflight runs with the
work-machine Python and exact completed N128 release before a plan is sealed.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ENGINE = 'scripts/train/train_single_table_native_constrained.py'
RUNTIME = 'scripts/experiments/single_table_generation.py'
CONTRACT = 'scripts/lib/rope/generation_contract.py'
GROUPS = ('text', 'instruction', 'reasoning', 'position_format')
CASES = ('N_compact', 'Z', 'Y')
N128_TRAINING_SHA = '682da90ffe3e10256e1cce40741f0f5e061640cdd5b0981c198c40e19d9f1992'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    with Path(p).open('x') as f:
        json.dump(v, f, indent=2, sort_keys=True)
        f.write('\n')


def jsonl(p):
    with Path(p).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def source_functions(p, names):
    tree = ast.parse(Path(p).read_text())
    found = {n.name: ast.dump(n, include_attributes=False) for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in names}
    if set(found) != set(names):
        raise ValueError('missing schedule/asset function in frozen engine')
    return found


def exposure_contract(task, qualification, native, seed=42):
    """Reconstruct release008 order, then verify its saved exposure fingerprints."""
    q = {(r['semantic_id'], r['world']): r for r in qualification['rows']}
    train = [r for r in task if r['split'] == 'train']
    if len(train) != 768 or len({r['semantic_id'] for r in train}) != 128:
        raise ValueError('expected 128 groups / 768 views')
    buckets = [[r for r in train if r['length_cap'] == cap] for cap in (2048, 8192, 16384)]
    if [len(b) for b in buckets] != [256]*3:
        raise ValueError('unbalanced exposure cells')
    rng = random.Random(seed)
    for b in buckets:
        rng.shuffle(b)
    ordered = [b[i] for i in range(256) for b in buckets]
    targets, compact, long_inputs, exposure = [], [], [], []
    cells = set()
    for r in ordered:
        key = (r['semantic_id'], r['world'], r['length_cap'])
        if key in cells:
            raise ValueError('duplicate exposure cell')
        cells.add(key)
        ref = q[key[:2]]
        if r['target_ids'] != ref['target_ids']:
            raise ValueError('target or EOS changed under compact replacement')
        if len(ref['compact_prompt_ids']) + len(r['target_ids']) > 2048:
            raise ValueError('compact input exceeds 2K including answer')
        targets.append([*key, r['target_ids'], [min(v, 1.) for v in ref['gold_margins']]])
        compact.append(ref['compact_prompt_ids'])
        long_inputs.append(r['prompt_ids'])
        exposure.append(key)
    rng = random.Random(seed)
    pools = {g: [r for r in native if r['split'] == 'train' and r['group'] == g] for g in GROUPS}
    for pool in pools.values():
        rng.shuffle(pool)
    cursor = dict.fromkeys(GROUPS, 0)
    replay = []
    for stage, count in (('restoration', 32), ('transfer', 96)):
        for step in range(count):
            indices = range(4) if stage == 'restoration' else (step % 4, (step+1) % 4)
            for gi in indices:
                g = GROUPS[gi]
                for _ in range(2 if stage == 'restoration' else 1):
                    row = pools[g][cursor[g]]
                    cursor[g] += 1
                    replay.append(row['id'])
    return {
        'task_exposure_sha256': digest(exposure), 'native_exposure_sha256': digest(replay),
        'ordered_targets_and_margins_sha256': digest(targets),
        'long_prompt_sequence_sha256': digest(long_inputs), 'compact_prompt_sequence_sha256': digest(compact),
        'views': len(ordered), 'replay_rows_R': len(replay[:256]), 'replay_rows_T': len(replay[256:]),
        'answer_EOS_labels': sum(len(r['target_ids']) for r in ordered),
        'actual_long_input_tokens': sum(len(r['prompt_ids'])+len(r['target_ids'])-1 for r in ordered),
        'actual_compact_input_tokens': sum(len(p)+len(r['target_ids'])-1 for p, r in zip(compact, ordered)),
        'scope': 'Matched target/margin exposure, update count and replay order; input tokens/FLOPs differ. Dual values may diverge under the same rule.'}


def commands(c, case, controls):
    arm = 'N' if case == 'N_compact' else case
    out = Path(c['output_root'])/case
    common = ['--checkpoint', c['checkpoint'], '--checkpoint-contract', c['checkpoint_contract'],
              '--tasks', c['tasks'], '--native-pool', c['native_pool'], '--seed', '42']
    table = []
    if arm != 'N':
        control = controls['arms'][arm]
        if not math.isfinite(control['rotary_amplitude']) or control['rotary_amplitude'] <= 0:
            raise ValueError('invalid recorded amplitude')
        table = ['--table', str(Path(c['controls'])/control['path']), '--gain', str(control['rotary_amplitude'])]
    engine = str(Path(c['engine_root'])/ENGINE)
    tasks = []
    def add(name, action, args, seconds):
        tasks.append({'name': name, 'seconds': seconds,
                      'argv': [c['python'], engine, action, *common, *table, '--authorized',
                               '--max-seconds', str(seconds), *args]})
    add('train', 'train', ['--arm', arm, '--placement', 'all_linear', '--kl-budget', '.02',
        '--teacher-cache', c['teacher_cache'], '--stop-after-step', '128', '--output', str(out/'train')]
        + (['--compact-only'] if case == 'N_compact' else []), 3600)
    # All outputs remain selection/development. Never expose Native/task test here.
    for step in ([128] if arm == 'N' else [0, 32, 128]):
        adapter = [] if step == 0 else ['--adapter', str(out/'train'/f'step_{step:03d}')]
        add(f'native{step}', 'native-evaluate', [*adapter, '--split', 'validation',
            '--output', str(out/f'native{step}')], 900)
        if step != 32:
            add(f'task{step}', 'evaluate', [*adapter, '--split', 'validation', '--lengths', '2048', '16384',
                '--output', str(out/f'task{step}')], 1800)
    tasks.append({'name': 'review', 'seconds': 600, 'argv': [c['python'],
        str(ROOT/'scripts/analysis/review_native_constrained_transfer.py'),
        '--protocol', 'single_evidence_v4', '--checkpoint', c['checkpoint'], '--tasks', c['tasks'],
        '--adapter', str(out/'train/step_128'), '--native-baseline', c['native_baseline'],
        '--native-candidate', str(out/'native128'), '--task-baseline', c['task_baseline'],
        '--task-candidate', str(out/'task128'), '--baseline-engine-source', c['baseline_eval_engine'],
        '--candidate-engine-source', engine, '--output', str(out/'review.json')]})
    return tasks


def prepare(config):
    c = read(config)
    required = ('python', 'engine_root', 'baseline_run', 'checkpoint', 'checkpoint_contract',
                'tasks', 'native_pool', 'teacher_cache', 'controls', 'native_baseline',
                'task_baseline', 'baseline_eval_engine', 'output_root')
    for key in required:
        if key not in c or not Path(c[key]).is_absolute():
            raise ValueError(f'{key} must be an explicit absolute work-machine path')
    output = Path(c['output_root'])
    if output.exists():
        raise FileExistsError('use a fresh output root; existing evidence is never overwritten')
    base = Path(c['baseline_run'])
    run, complete = read(base/'run.json'), read(base/'complete.json')
    if (complete['status'] != 'TRAINING_COMPLETE_NOT_FEASIBILITY_OR_CAPABILITY'
            or complete['completed_step'] != 128 or run['arm'] != 'N' or run['seed'] != 42
            or run['compact_only'] or run['recipe']['placement'] != 'all_linear'
            or run['recipe'].get('prefix_lm', False) or run['native_kl_target'] != .02
            or run['restoration_steps'] != 32 or run['transfer_steps'] != 96):
        raise ValueError('reference is not the fixed corrected N128 recipe')
    if (complete['training_sha256'] != N128_TRAINING_SHA or
            sha(base/'run.json') != complete['run_sha256'] or sha(base/'training.jsonl') != complete['training_sha256']):
        raise ValueError('reference run/log receipt drift')
    eroot = Path(c['engine_root'])
    files = {eroot/ENGINE: run['engine_sha256'], eroot/RUNTIME: run['code_sha256'],
             eroot/CONTRACT: run['recipe']['contract_sha256']}
    for p, expected in files.items():
        if sha(p) != expected:
            raise ValueError('must use exact completed N128 engine/runtime/KL code')
    if '--compact-only' not in (eroot/ENGINE).read_text():
        raise ValueError('frozen engine lacks compact flag; do not silently use a different trainer')
    names = ('task_order', 'native_replay_order', 'read_tasks', 'read_native_pool')
    if source_functions(eroot/ENGINE, names) != source_functions(ROOT/ENGINE, names):
        raise ValueError('schedule/asset parser changed; inspect before using planner reconstruction')
    for key, recipe_key in [('tasks', 'tasks_sha256'), ('native_pool', 'native_pool_sha256'),
                            ('checkpoint_contract', 'checkpoint_contract_sha256')]:
        if sha(c[key]) != run['recipe'][recipe_key]:
            raise ValueError(f'{key} differs from N128')
    cache_manifest = Path(c['teacher_cache'])/'manifest.json'
    if sha(cache_manifest) != run['teacher_cache_manifest_sha256']:
        raise ValueError('teacher cache manifest drift')
    model_contract = read(c['checkpoint_contract'])
    if model_contract['model_type'] != 'qwen2' or model_contract['native_context_length'] != 32768:
        raise ValueError('this round is only the configured-Native32K Qwen branch')
    weight = Path(c['checkpoint'])/'model.safetensors'
    if sha(weight) != run['checkpoint_sha256'] or model_contract['weight_sha256'] != run['checkpoint_sha256']:
        raise ValueError('base checkpoint weight mismatch')
    tm, nm = read(c['tasks']), read(c['native_pool'])
    task_file = Path(c['tasks']).parent/tm['views_path']
    native_file = Path(c['native_pool']).parent/nm['rows_path']
    qfile = Path(c['tasks']).parent/tm['qualification_path']
    for p, h in [(task_file, tm['views_sha256']), (native_file, nm['rows_sha256']), (qfile, tm['qualification_sha256'])]:
        if sha(p) != h:
            raise ValueError('data/qualification hash drift')
    exposure = exposure_contract(jsonl(task_file), read(qfile), jsonl(native_file))
    for field in ('task_exposure_sha256', 'native_exposure_sha256'):
        if exposure[field] != complete[field]:
            raise ValueError('reconstructed schedule differs from completed N128')
    controls = read(Path(c['controls'])/'manifest.json')
    if (controls['status'] != 'FIXED_NZGY_CONTROLS_FROZEN_V1'
            or controls['checkpoint_config_sha256'] != run['checkpoint_config_sha256']
            or controls['actual_native_context_length'] != 32768
            or controls['native_sha256'] != run['table_sha256']):
        raise ValueError('wrong control manifest')
    # Freeze all external inputs and all launcher/reviewer dependencies.
    paths = set(files) | {Path(config).resolve(), weight, cache_manifest, task_file, native_file, qfile,
        Path(c['controls'])/'manifest.json', base/'run.json', base/'complete.json', base/'training.jsonl',
        Path(c['baseline_eval_engine']), Path(c['tasks']), Path(c['native_pool']), Path(c['checkpoint_contract'])}
    for p in Path(c['checkpoint']).iterdir():
        if p.name == 'config.json' or 'tokenizer' in p.name or p.name in ('special_tokens_map.json', 'chat_template.jinja'):
            paths.add(p)
    for arm in ('Z', 'Y'):
        p = Path(c['controls'])/controls['arms'][arm]['path']
        if sha(p) != controls['arms'][arm]['file_sha256']:
            raise ValueError('frozen table file drift')
        paths.add(p)
    for key, receipt in [('native_baseline', 'native_evaluation.json'), ('task_baseline', 'evaluation.json')]:
        p = Path(c[key]); meta = read(p/receipt)
        if sha(p/'examples.jsonl') != meta['examples_sha256']:
            raise ValueError('baseline raw rows drift')
        if not meta['table_is_native'] or meta['gain'] != 1 or meta['adapter_sha256'] is not None:
            raise ValueError('validation baseline is not original Native')
        if meta.get('fold', meta.get('split')) not in ('selection', 'validation'):
            raise ValueError('confirmation/test pool cannot be used by this round')
        paths.update((p/receipt, p/'examples.jsonl'))
    for rel in (ENGINE, RUNTIME, CONTRACT, 'scripts/analysis/export_single_table_controls.py',
                'scripts/analysis/review_native_constrained_transfer.py', 'scripts/experiments/matched_transfer_round.py'):
        paths.add(ROOT/rel)
    # Record the whole small archived Python dependency tree; no mutation there.
    paths.update((eroot/'scripts').rglob('*.py'))
    output.mkdir(parents=True)
    (output/'logs').mkdir()
    if shutil.disk_usage(output).free < 8*2**30:
        raise ValueError('need 8 GiB free for bounded saved adapters/optimizer states; relocate new output, preserve old evidence')
    preflight = [c['python'], str(eroot/ENGINE), 'preflight', '--checkpoint', c['checkpoint'],
        '--checkpoint-contract', c['checkpoint_contract'], '--tasks', c['tasks'], '--native-pool', c['native_pool'],
        '--teacher-cache', c['teacher_cache'], '--arm', 'N', '--output', str(output/'asset_preflight.json')]
    with (output/'logs/prepare.log').open('x') as log:
        subprocess.run(preflight, cwd=eroot, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
    plan = {'status': 'PREPARED_CPU_ONLY_GPU_PENDING', 'config': c, 'config_source': str(Path(config).resolve()), 'exposure': exposure,
            'files': {str(p): sha(p) for p in sorted(paths)},
            'cases': {case: commands(c, case, controls) for case in CASES},
            'limits': 'End step128 only; seed42; no prefix/resume/test/search. R32 Native is diagnostic. No host shutdown.'}
    write(output/'plan.json', plan)
    print(json.dumps({'status': plan['status'], 'plan': str(output/'plan.json'), 'exposure': exposure}, indent=2))


def call_stage(stage, cwd, log):
    with log.open('x') as f:
        proc = subprocess.Popen(stage['argv'], cwd=cwd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=stage['seconds']+120)
            if code:
                raise RuntimeError(f"stage {stage['name']} exited {code}; inspect its preserved log")
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            raise


def execute(plan_path, cases, authorized):
    if not authorized:
        raise ValueError('run requires the explicitly authorized machine/budget; preparation never starts CUDA')
    plan = read(plan_path)
    if plan['status'] != 'PREPARED_CPU_ONLY_GPU_PENDING' or len(cases) != len(set(cases)):
        raise ValueError('invalid plan or duplicate case')
    for p, h in plan['files'].items():
        if sha(p) != h:
            raise ValueError(f'frozen input/code changed: {Path(p).name}')
    c = plan['config']; out = Path(c['output_root'])
    if read(plan['config_source']) != c:
        raise ValueError('plan/config mismatch')
    control = read(Path(c['controls'])/'manifest.json')
    if plan['cases'] != {case: commands(c, case, control) for case in CASES}:
        raise ValueError('frozen stage commands changed')
    # Host observation only; actual Flash/BF16 checks remain the archived runtime's job.
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if gpu.strip():
        raise ValueError('GPU has a live compute process; inspect ownership before launching')
    if shutil.disk_usage(out).free < 8*2**30:
        raise ValueError('insufficient output disk before launch')
    lock = out/'running.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        for case in cases:
            target = out/case
            target.mkdir()  # Existing or partial runs are never resumed or overwritten.
            state = {'case': case, 'plan_sha256': sha(plan_path), 'completed_stages': [], 'started': time.time()}
            try:
                for stage in plan['cases'][case]:
                    print(f"{case}: {stage['name']}", flush=True)
                    call_stage(stage, c['engine_root'], out/'logs'/f"{case}_{stage['name']}.log")
                    if stage['name'] == 'train':
                        complete = read(target/'train/complete.json')
                        if (complete['completed_step'] != 128 or not complete['final_adapter_reload_greedy_exact']
                                or complete['status'] != 'TRAINING_COMPLETE_NOT_FEASIBILITY_OR_CAPABILITY'):
                            raise ValueError('incomplete training; no automatic resume or evaluation')
                        for field in ('task_exposure_sha256', 'native_exposure_sha256'):
                            if complete[field] != plan['exposure'][field]:
                                raise ValueError('executed exposure mismatch')
                    state['completed_stages'].append(stage['name'])
                state['status'] = 'BOUNDED_CASE_COMPLETE_READ_REVIEW'
            except BaseException as error:
                state.update(status='STOPPED_PRESERVE_PARTIAL', reason=str(error))
                raise
            finally:
                state['finished'] = time.time()
                write(target/'execution.json', state)
    finally:
        lock.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('template', 'prepare', 'run'))
    p.add_argument('--config', type=Path); p.add_argument('--plan', type=Path)
    p.add_argument('--cases', nargs='+', choices=CASES, default=['N_compact'])
    p.add_argument('--authorized', action='store_true')
    a = p.parse_args()
    if a.action == 'template':
        print(json.dumps({k: f'/ABSOLUTE/{k}' for k in ('python', 'engine_root', 'baseline_run', 'checkpoint',
            'checkpoint_contract', 'tasks', 'native_pool', 'teacher_cache', 'controls', 'native_baseline',
            'task_baseline', 'baseline_eval_engine', 'output_root')}, indent=2))
    elif a.action == 'prepare':
        if not a.config: p.error('--config required')
        prepare(a.config)
    else:
        if not a.plan: p.error('--plan required')
        execute(a.plan, a.cases, a.authorized)


if __name__ == '__main__':
    main()
