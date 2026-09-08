"""Frozen retrospective cases: prefix formation x readout RoPE table.

Capture pre-RoPE K directly; never invert rounded BF16 cached keys. All prefix
tokens remain visible. V and pre-RoPE K retain their source-prefill provenance.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen2.modeling_qwen2 import rotate_half

from .diagnose import greedy
from .prepare import sha_file, verify_weight_stats
from .run import atomic
from .runtime import install, verify
from .ruler_bench import score
from scripts.experiments.scale_transport.position_visibility import check_decoder

CASES = ('niah_multikey_2_131072_1', 'vt_131072_1')
ARMS = ('MrPro', 'MrProBM')


def rotated_key(raw, cos, sin):
    # OLMo2 supplies FP32 cos/sin and casts the rotated result back to K dtype.
    # Qwen2 supplies K-dtype cos/sin, so this preserves its existing arithmetic.
    return (raw * cos.unsqueeze(1) + rotate_half(raw) * sin.unsqueeze(1)).to(raw.dtype)


def trim_prefix(cache, length):
    for layer in cache.layers:
        if type(layer).__name__ != 'DynamicLayer':
            raise ValueError('requires full-attention DynamicLayer')
        layer.keys = layer.keys[..., :length, :].contiguous()
        layer.values = layer.values[..., :length, :].contiguous()
    if cache.get_seq_length() != length:
        raise ValueError('cache length mismatch')


def work(prepared, baseline, out, cases=CASES, matched_cached_baseline=False, factor_reference=None,
         factor_read='MrPro'):
    start = time.monotonic()
    manifest = json.loads((prepared/'manifest.json').read_text())
    for name, expected in manifest['prepared_files'].items():
        if sha_file(prepared/name) != expected:
            raise ValueError('frozen input changed: '+name)
    verify_weight_stats(manifest)
    for name, expected in manifest['model_files_sha256'].items():
        if sha_file(Path(manifest['model_path'])/name) != expected:
            raise ValueError('model metadata changed: '+name)
    params = json.loads((prepared/'generation_config.json').read_text())
    checked = dict(params)
    for name, default in dict(encoder_no_repeat_ngram_size=0,
            encoder_repetition_penalty=1., remove_invalid_values=False).items():
        if checked.get(name) is None:
            checked[name] = default
    check_decoder(checked)
    if params['repetition_penalty'] != 1.:
        raise ValueError('requires unit repetition penalty')
    rows = {r['row_id']: r for r in map(json.loads, (prepared/'screen.jsonl').read_text().splitlines())}
    saved = {arm: {r['row_id']: r for r in map(json.loads,
        (baseline/(arm+'.jsonl')).read_text().splitlines())} for arm in ARMS}
    if not cases or len(set(cases)) != len(cases) or any(
            case not in rows or any(case not in saved[arm] for arm in ARMS) for case in cases):
        raise ValueError('cases must be unique prepared rows with both archived references')
    tables = json.loads((prepared/'tables.json').read_text())
    if tables[ARMS[0]]['gain'] != tables[ARMS[1]]['gain']:
        raise ValueError('this intervention fixes gain')
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
    if model.config.model_type not in ('qwen2','olmo2') or getattr(model.config, 'use_sliding_window', False):
        raise ValueError('requires reviewed full-attention Qwen2 or OLMo2')
    capture_point = 'k_norm' if model.config.model_type == 'olmo2' else 'k_proj'
    if sum(p.numel() for p in model.parameters()) != manifest['actual_parameters']:
        raise ValueError('model count mismatch')
    tokenizer = AutoTokenizer.from_pretrained(manifest['model_path'], local_files_only=True)
    import inspect
    import transformers
    root = Path(__file__).resolve().parents[3]
    deps = set(manifest['code_files']) | {
        'scripts/experiments/olmo_fast_screen/cross_cache.py',
        'scripts/experiments/olmo_fast_screen/diagnose.py',
        'scripts/experiments/scale_transport/position_visibility.py'}
    atomic(out/'runtime.json', dict(parent_manifest_sha256=sha_file(prepared/'manifest.json'),
        model_id=manifest['model_id'], revision=manifest['revision'],
        code_files={name: sha_file(root/name) for name in deps},
        baseline_sha256={arm: sha_file(baseline/(arm+'.jsonl')) for arm in ARMS},
        torch=torch.__version__, transformers=transformers.__version__,
        cases=cases, matched_cached_baseline=matched_cached_baseline,
        pre_rope_key_capture=capture_point,
        factor_reference_sha256=sha_file(factor_reference) if factor_reference else None,
        factor_read=factor_read if factor_reference else None,
        backend='Flash SDPA only', tables=tables,
        scope='Retrospective fixed-case causal intervention, not benchmark improvement'))
    (out/'attention_source.txt').write_text(inspect.getsource(type(model.model.layers[0].self_attn)))
    results = []
    qualifications = []

    def execute(row, source, read, mode, ids, positions, cache=None):
        if (out/'STOP').exists():
            raise RuntimeError('operator stop')
        atomic(out/'live.json', dict(row_id=row['row_id'], source=source, read=read,
            mode=mode, completed=len(results)))
        install(model, tables[read]); verify(model, tables[read])
        begin = time.monotonic()
        data, cache = greedy(model, ids, positions, max_new_tokens=row['max_new_tokens'],
            eos_token_id=params['eos_token_id'], cache=cache)
        tokens = data['generated_ids']
        text = tokenizer.decode(tokens[:-1] if data['ended_eos'] else tokens, skip_special_tokens=False)
        data.update(row_id=row['row_id'], source=source, read=read, mode=mode,
            correct=score(row, text), output_text=text, references=row['references'],
            elapsed_seconds=time.monotonic()-begin)
        results.append(data); atomic(out/'results.json', results)
        print(json.dumps({k: data[k] for k in ('row_id','source','read','mode','correct','elapsed_seconds')}), flush=True)
        return data, cache

    with torch.inference_mode():
        for case in cases:
            row = rows[case]; ids = row['prompt_ids']; n = len(ids)
            bank = {}
            for source in ARMS:
                raw = {}
                handles = []
                def capture(index, head_dim):
                    def hook(module, args, output):
                        if output.shape[1] == n:
                            raw[index] = output.view(1, n, -1, head_dim).transpose(1, 2)
                    return hook
                for j, layer in enumerate(model.model.layers):
                    module = getattr(layer.self_attn, capture_point)
                    handles.append(module.register_forward_hook(capture(j, layer.self_attn.head_dim)))
                try:
                    original, cache = execute(row, source, source, 'O', ids, list(range(n)))
                finally:
                    for handle in handles:
                        handle.remove()
                if original['generated_ids'] != saved[source][case]['generated_ids']:
                    raise ValueError('original output does not reproduce baseline')
                if len(raw) != len(model.model.layers):
                    raise ValueError('missing pre-rotation keys')
                trim_prefix(cache, n-1)
                # Same full position shape and dtype as original HF prefill.
                position_ids = torch.arange(n, device=model.device).unsqueeze(0)
                cos, sin = model.model.rotary_emb(raw[0], position_ids)
                for j, layer in enumerate(cache.layers):
                    rebuilt = rotated_key(raw[j], cos, sin)[..., :n-1, :]
                    if not torch.equal(rebuilt, layer.keys):
                        raise ValueError('same-table key reconstruction is not bitwise exact')
                del rebuilt, cos, sin
                if factor_reference:
                    # Fix every read to the declared table while preserving each source's
                    # content vectors. CPU storage bounds peak GPU memory.
                    install(model, tables[factor_read]); verify(model, tables[factor_read])
                    cos, sin = model.model.rotary_emb(raw[0], position_ids)
                    bank[source] = [
                        (rotated_key(raw[j], cos, sin)[..., :n-1, :].contiguous().cpu(),
                         layer.values.cpu())
                        for j, layer in enumerate(cache.layers)]
                    del cache, raw, cos, sin, position_ids
                    continue
                diagonal, cache = execute(row, source, source, 'same_table_cached_query',
                    ids[-1:], [n-1], cache)
                same_original = diagonal['generated_ids'] == original['generated_ids']
                if not same_original and not matched_cached_baseline:
                    raise ValueError('cached last-query replay differs from full prefill')
                # When the cached-query numerical path differs from full prefill,
                # establish its own complete-output baseline. Rebuild same-table K
                # through exactly the same mutation path as the cross intervention.
                if matched_cached_baseline:
                    trim_prefix(cache, n-1)
                    cos, sin = model.model.rotary_emb(raw[0], position_ids)
                    for j, layer in enumerate(cache.layers):
                        layer.keys = rotated_key(raw[j], cos, sin)[..., :n-1, :].contiguous()
                    del cos, sin
                    rebuilt_result, cache = execute(row, source, source, 'same_table_rebuilt_query',
                        ids[-1:], [n-1], cache)
                    if rebuilt_result['generated_ids'] != diagonal['generated_ids']:
                        raise ValueError('rebuilt same-table keys change matched cached baseline')
                qualifications.append(dict(row_id=case, source=source,
                    original_matches_saved=True, keys_bitwise_equal=True,
                    diagonal_matches_original=same_original,
                    rebuilt_matches_cached_baseline=True if matched_cached_baseline else None))
                atomic(out/'qualification.json', qualifications)
                trim_prefix(cache, n-1)
                read = next(arm for arm in ARMS if arm != source)
                install(model, tables[read]); verify(model, tables[read])
                cos, sin = model.model.rotary_emb(raw[0], position_ids)
                for j, layer in enumerate(cache.layers):
                    # Values are untouched; K is regenerated from the same captured projection.
                    layer.keys = rotated_key(raw[j], cos, sin)[..., :n-1, :].contiguous()
                del raw, cos, sin, position_ids
                crossed, cache = execute(row, source, read, 'cross_table_cached_query',
                    ids[-1:], [n-1], cache)
                del cache
            if factor_reference:
                reference = json.loads(factor_reference.read_text())['records']
                # Establish both diagonal controls before any mixed-content arm.
                for key_source, value_source in (
                    ('MrPro', 'MrPro'), ('MrProBM', 'MrProBM'),
                    ('MrPro', 'MrProBM'), ('MrProBM', 'MrPro')):
                    cache = DynamicCache(config=model.config)
                    for j in range(len(model.model.layers)):
                        cache.update(bank[key_source][j][0].to(model.device),
                            bank[value_source][j][1].to(model.device), j)
                    data, cache = execute(row, key_source+'/'+value_source, factor_read,
                        'factor_KV', ids[-1:], [n-1], cache)
                    data.update(key_source=key_source, value_source=value_source)
                    atomic(out/'results.json', results)
                    if key_source == value_source:
                        mode = ('same_table_cached_query' if key_source == factor_read
                                else 'cross_table_cached_query')
                        matches = [r for r in reference if r['row_id'] == case
                            and r['source'] == key_source and r['read'] == factor_read
                            and r['mode'] == mode]
                        if len(matches) != 1 or data['generated_ids'] != matches[0]['generated_ids']:
                            raise ValueError('factor diagonal differs from complete cached reference')
                        qualifications.append(dict(row_id=case, source=key_source,
                            original_matches_saved=True, keys_bitwise_equal=True,
                            factor_diagonal_matches_cached_reference=True))
                        atomic(out/'qualification.json', qualifications)
                    del cache
                del bank
    atomic(out/'status.json', dict(status='COMPLETE', generations=len(results),
        elapsed_seconds=time.monotonic()-start, peak_memory_bytes=torch.cuda.max_memory_allocated()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cases', nargs='+', default=list(CASES))
    p.add_argument('--matched-cached-baseline', action='store_true')
    p.add_argument('--factor-reference', type=Path)
    p.add_argument('--factor-read', choices=ARMS, default='MrPro')
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    atomic(args.out/'status.json', dict(status='RUNNING', pid=os.getpid(), started_unix=time.time()))
    try:
        work(args.prepared, args.baseline, args.out, args.cases, args.matched_cached_baseline,
            args.factor_reference,args.factor_read)
    except BaseException as exc:
        atomic(args.out/'status.json', dict(status='FAILED', error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
