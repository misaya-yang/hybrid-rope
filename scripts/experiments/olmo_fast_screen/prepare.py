"""Prepare the tokenizer, frozen short benchmark and queue without model inference."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

from .bench import FAMILIES, MAX_NEW_TOKENS, digest, prepare_rows


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def reviewed_candidate(path, source):
    candidate = json.loads(Path(path).read_text())
    if (candidate['target_revision'],candidate['target_model']) != (source['revision'],source['model_id']):
        raise ValueError('candidate model identity differs')
    if candidate['candidate_id'] != 'MrProBM':
        raise ValueError('this phase contains only the author-selected MrPro-BM')
    target = candidate['target_olmo']
    if not all(target['checks'][key] for key in (
        'finite_positive','strictly_decreasing','fast_band_bitwise_equal',
        'slow_band_bitwise_equal','support_endpoints_bitwise_equal')):
        raise ValueError('BM table fails array constraints')
    queue = dict(max_candidates=10,ordered_candidates=[dict(
        id='MrProBM',eligible=True,review_status='REVIEWED_FOR_GPU',
        definition=candidate['definition'],source_candidate_sha256=sha_file(path),
        hypothesis=candidate['review']['task_hypothesis'],
        failure_rule=candidate['review']['prediction_falsifier'],
        adverse_review=candidate['review']['failure_modes'],
    )],note='Up to ten theoretically defined and reviewed candidates; only BM is ready so far. Earlier psi projection remains rejected. Five minutes is an estimate, not a timeout.')
    tables = {'Native':source['tables']['Native'],'MrPro':source['tables']['MrPro'],
              'MrProBM':{key:target[key] for key in ('values_float32','tensor_sha256','gain')}}
    return tables,queue


def reuse_prepared(args):
    old=args.reuse_prepared.resolve();manifest=json.loads((old/'manifest.json').read_text())
    source=json.loads(args.olmo_source.read_text())
    if args.model.resolve()!=Path(manifest['model_path']):
        raise ValueError('cannot change the model while reusing prepared inputs')
    for name,expected in manifest['prepared_files'].items():
        if sha_file(old/name)!=expected:raise ValueError('prior prepared inputs/configuration drifted')
    weight=(args.model/'model.safetensors').stat()
    if {'size':weight.st_size,'mtime_ns':weight.st_mtime_ns}!=manifest['weight_stat']:
        raise ValueError('model changed since its preparation-time digest check')
    tables,queue=reviewed_candidate(args.candidate,source)
    out=args.out.resolve();shutil.copytree(old,out)
    previous_sha=sha_file(old/'manifest.json')
    write(out/'tables.json',tables);write(out/'queue.json',queue)
    root=Path(__file__).resolve().parents[3]
    manifest.update(status='PREPARED_BM_FIRST_CANDIDATE_GPU_NOT_RUN',
        source_candidate_sha256=sha_file(args.candidate),
        inputs_reused_from_manifest_sha256=previous_sha,
        gpu_execution='NOT_RUN; BM ready, additional reviewed candidates unfinished',
        timing='Five minutes per main arm is an estimate. Complete all frozen rows and record actual cost; no automatic per-arm timeout or invented total deadline.')
    manifest['prepared_files']={name:sha_file(out/name) for name in manifest['prepared_files']}
    manifest['code_files']={name:sha_file(root/name) for name in manifest['code_files']}
    for name in ('screen.jsonl','qualification.jsonl','generation_config.json'):
        if sha_file(out/name)!=sha_file(old/name):raise AssertionError('reused input/decoder changed')
    write(out/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('status','screen_rows','qualification_rows',
                     'screen_input_tokens','screen_min_max_tokens','gpu_execution')},indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--olmo-source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reuse-prepared',type=Path)
    args = p.parse_args()
    if args.reuse_prepared:
        reuse_prepared(args)
        return
    from transformers import AutoTokenizer, GenerationConfig
    model = args.model.resolve()
    source = json.loads(args.olmo_source.read_text())
    config = json.loads((model/'config.json').read_text())
    if (config['model_type'], config['hidden_size'], config['num_hidden_layers'],
        config['num_attention_heads'], config['rope_theta'], config['max_position_embeddings']) != (
        'olmo2', 2048, 16, 16, 500000, 4096) or config.get('rope_scaling'):
        raise ValueError('requires the specified unscaled OLMo 1.485B checkpoint')
    # One preparation-time read, never rehash the 3 GB weights per candidate.
    weight = model/'model.safetensors'
    weight_sha = sha_file(weight)
    if weight_sha != source['model_sha256']:
        raise ValueError('model is not the retained canonical checkpoint')
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)

    def encode(text):
        result = tokenizer.apply_chat_template([{'role':'user','content':text}],
                                               tokenize=True, add_generation_prompt=True)
        if hasattr(result, 'keys'):
            result = result['input_ids']
        if result and isinstance(result[0], list):
            result = result[0]
        return list(result)

    screen, qualification = prepare_rows(encode)
    if len(screen) != 24 or len(qualification) != 8:
        raise AssertionError('unexpected benchmark size')
    if len({r['row_id'] for r in screen+qualification}) != 32:
        raise AssertionError('duplicate rows')
    decoding = GenerationConfig.from_pretrained(model, local_files_only=True)
    decoding.do_sample = False
    decoding.num_beams = 1
    decoding.num_return_sequences = 1
    decoding.repetition_penalty = 1.0
    decoding.no_repeat_ngram_size = 0
    decoding.use_cache = True
    decoding.eos_token_id = tokenizer.eos_token_id
    decoding.pad_token_id = tokenizer.pad_token_id
    decoding.max_new_tokens = MAX_NEW_TOKENS
    decoding.min_new_tokens = 0
    decoding.min_length = 0
    decoding.forced_bos_token_id = None
    decoding.forced_eos_token_id = None
    decoding.bad_words_ids = None
    decoding.suppress_tokens = None
    decoding.begin_suppress_tokens = None
    tables,queue=reviewed_candidate(args.candidate,source)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in [('screen',screen), ('qualification',qualification)]:
        with (out/(name+'.jsonl')).open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row)+'\n')
    write(out/'tables.json', tables)
    write(out/'generation_config.json', decoding.to_dict())
    write(out/'queue.json', queue)
    root = Path(__file__).resolve().parents[3]
    dependencies = [
        'scripts/experiments/olmo_fast_screen/'+name for name in ('bench.py','prepare.py','run.py','supervise.py')]
    dependencies += ['scripts/experiments/cross_audit/tables.py',
                     'scripts/lib/rope/official_yarn.py']
    stat = weight.stat()
    report = dict(
        status='PREPARED_BM_FIRST_CANDIDATE_GPU_NOT_RUN',
        model_id=source['model_id'], revision=source['revision'], model_path=str(model),
        actual_parameters=source['actual_parameters'], weight_sha256=weight_sha,
        weight_stat={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns},
        model_files_sha256={p.name:sha_file(p) for p in model.iterdir()
                            if p.is_file() and p.suffix in ('.json','.txt')},
        physical_caps=[4096,8192,16384], static_scale=4, native_length=4096,
        base=500000, head_dim=128, families=list(FAMILIES), screen_rows=len(screen),
        qualification_rows=len(qualification), screen_input_tokens=sum(r['input_tokens'] for r in screen),
        screen_min_max_tokens=[min(r['input_tokens'] for r in screen),max(r['input_tokens'] for r in screen)],
        seed=20260908, row_order=[r['row_id'] for r in screen],
        prompt_collection_sha256=digest([r['prompt_ids'] for r in screen]),
        prepared_files={name:sha_file(out/name) for name in
                        ('screen.jsonl','qualification.jsonl','tables.json','generation_config.json','queue.json')},
        code_files={name:sha_file(root/name) for name in dependencies},
        software={name:importlib.metadata.version(name) for name in ('torch','transformers','numpy')},
        scoring='Whole normalized one-word exact match; EOS reported separately. Four equally weighted task families.',
        qualification='Native compact: at least 6/8 correct and at least 1/2 in every family; otherwise stop before candidate evaluation.',
        selection='Any strictly positive macro gain with no decrease in 4K correct count is a development win; stop the queue for deeper analysis. Other complete results continue the fixed queue.',
        timing='Target estimate: roughly five minutes per main arm after a shared model load. Complete all frozen rows; no five-minute kill or invented total deadline. Record actual baseline and candidate cost.',
        scope='Constructed development benchmark; no RULER task is copied, no full public benchmark, no generalization claim.',
        source_candidate_sha256=sha_file(args.candidate),
        gpu_execution='NOT_RUN; Native qualification, one MrPro baseline, one MrProBM comparison when GPU phase starts',
    )
    write(out/'manifest.json', report)
    print(json.dumps({k:report[k] for k in ('status','screen_rows','qualification_rows',
                     'screen_input_tokens','screen_min_max_tokens','gpu_execution')},indent=2))


if __name__ == '__main__':
    main()
