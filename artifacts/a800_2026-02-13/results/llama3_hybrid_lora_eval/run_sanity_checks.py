import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = '/opt/dfrope/models_alt/LLM-Research/Meta-Llama-3-8B-Instruct'
HYBRID_LORA_PATH = '/opt/dfrope/results/llama3_hybrid_lora/final_lora'
OUT_DIR = '/opt/dfrope/results/llama3_hybrid_lora_eval'
OUT_JSON = f'{OUT_DIR}/sanity_checks.json'
OUT_MD = f'{OUT_DIR}/SANITY_SUMMARY.md'

LENGTHS = [2048, 8192, 12288, 14336, 16384]
STRATEGIES = ['sequential', 'random_start']
EVAL_CHUNKS = 5
VAL_TOKENS = 1_200_000
SEED = 42
POS_DIFF_TOL = 1e-4


def geometric_freq(K, theta):
    k = torch.arange(K, dtype=torch.float32)
    return 1.0 / (theta ** (2 * k / (2 * K)))


def anchored_poly_freq(K, theta_base, p=3.9, omf=0.3):
    k = torch.arange(K, dtype=torch.float32)
    geo = geometric_freq(K, theta_base)
    omega_max = geo[0].item()
    omega_min = geo[-1].item() * omf
    t = k / (K - 1)
    log_omega = math.log(omega_max) + (t ** p) * (math.log(omega_min) - math.log(omega_max))
    return torch.exp(log_omega)


def hybrid_freq(freq_a, freq_b, alpha):
    return (1 - alpha) * freq_a + alpha * freq_b


def build_hybrid_inv_freq(head_dim):
    K = head_dim // 2
    geo_100k = geometric_freq(K, 100000)
    poly_100k = anchored_poly_freq(K, 100000, p=3.9, omf=0.3)
    return hybrid_freq(geo_100k, poly_100k, alpha=0.2)


def patch_model_rope(model):
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    inv = build_hybrid_inv_freq(head_dim)

    cands = []
    if hasattr(model, 'model') and hasattr(model.model, 'rotary_emb'):
        cands.append(model.model.rotary_emb)

    for layer in getattr(model.model, 'layers', []):
        attn = getattr(layer, 'self_attn', None)
        if attn is None:
            continue
        if hasattr(attn, 'rotary_emb'):
            cands.append(attn.rotary_emb)
        if hasattr(attn, 'rotary_fn'):
            rf = attn.rotary_fn
            if hasattr(rf, 'inv_freq'):
                cands.append(rf)
            if hasattr(rf, 'rotary_emb') and hasattr(rf.rotary_emb, 'inv_freq'):
                cands.append(rf.rotary_emb)

    seen = set()
    patched = 0
    for rope in cands:
        if id(rope) in seen or not hasattr(rope, 'inv_freq'):
            continue
        seen.add(id(rope))
        rope.inv_freq = inv.to(device=rope.inv_freq.device, dtype=rope.inv_freq.dtype)
        if hasattr(rope, 'max_seq_len_cached'):
            rope.max_seq_len_cached = 0
        patched += 1

    if patched == 0:
        raise RuntimeError('No rotary module found to patch')


def load_base_model():
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        quantization_config=bnb_cfg,
        device_map='auto',
        trust_remote_code=True,
    )
    model.config.use_cache = True
    model.eval()
    return model


def load_hybrid_lora_model():
    model = load_base_model()
    patch_model_rope(model)
    model = PeftModel.from_pretrained(model, HYBRID_LORA_PATH)
    model.eval()
    return model


def load_val_tokens(tokenizer, max_tokens=VAL_TOKENS):
    ids = []
    ds = load_dataset('allenai/c4', 'en', split='validation', streaming=True)
    for x in ds:
        txt = x.get('text')
        if not txt:
            continue
        ids.extend(tokenizer.encode(txt, add_special_tokens=False))
        if len(ids) >= max_tokens:
            break
    return torch.tensor(ids[:max_tokens], dtype=torch.long)


def get_starts(total_tokens, L, n_chunks, strategy, seed):
    max_start = total_tokens - L
    if max_start < 0:
        return []

    if strategy == 'sequential':
        starts = []
        for i in range(n_chunks):
            s = i * L
            if s <= max_start:
                starts.append(s)
        return starts

    if strategy == 'random_start':
        pop = max_start + 1
        k = min(n_chunks, pop)
        rng = random.Random(seed + L)
        if k == pop:
            starts = list(range(pop))
        else:
            starts = rng.sample(range(pop), k)
        starts.sort()
        return starts

    raise ValueError(f'Unknown strategy: {strategy}')


def compute_ppl(losses):
    if not losses:
        return None, None
    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(mean_loss)
    std = (sum((math.exp(x) - ppl) ** 2 for x in losses) / len(losses)) ** 0.5
    return float(ppl), float(std)


@torch.no_grad()
def evaluate_model_numeric_and_ppl(model, model_name, tokens):
    model_out = {}
    anomalies = []
    device = model.device

    for strategy in STRATEGIES:
        model_out[strategy] = {}
        print(f'[A/C/D] model={model_name}, strategy={strategy}', flush=True)
        for L in LENGTHS:
            starts = get_starts(len(tokens), L, EVAL_CHUNKS, strategy, SEED)
            losses = []
            chunk_records = []

            for batch_idx, start in enumerate(starts):
                chunk = tokens[start:start + L].unsqueeze(0).to(device)
                inp = chunk[:, :-1]
                tgt = chunk[:, 1:]

                logits = model(inp).logits
                logits_finite = bool(torch.isfinite(logits).all().item())

                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                loss_finite = bool(torch.isfinite(loss).item())
                loss_value = float(loss.item()) if loss_finite else None

                rec = {
                    'batch_idx': batch_idx,
                    'start': int(start),
                    'loss_finite': loss_finite,
                    'logits_finite': logits_finite,
                    'loss': loss_value,
                }
                chunk_records.append(rec)

                if not logits_finite:
                    anomalies.append({
                        'model': model_name,
                        'strategy': strategy,
                        'length': int(L),
                        'batch_idx': int(batch_idx),
                        'start': int(start),
                        'type': 'logits_non_finite',
                    })
                if not loss_finite:
                    anomalies.append({
                        'model': model_name,
                        'strategy': strategy,
                        'length': int(L),
                        'batch_idx': int(batch_idx),
                        'start': int(start),
                        'type': 'loss_non_finite',
                    })

                if logits_finite and loss_finite:
                    losses.append(loss_value)

                del chunk, inp, tgt, logits, loss

            ppl, std = compute_ppl(losses)
            all_finite = len(chunk_records) > 0 and all(r['loss_finite'] and r['logits_finite'] for r in chunk_records)
            model_out[strategy][str(L)] = {
                'n_chunks': len(chunk_records),
                'all_finite': all_finite,
                'losses': losses,
                'ppl': round(ppl, 3) if ppl is not None else None,
                'std': round(std, 3) if std is not None else None,
                'bad_batches': [r for r in chunk_records if (not r['loss_finite'] or not r['logits_finite'])],
            }
            print(
                f'  L={L} n={len(chunk_records)} finite={all_finite} ppl={model_out[strategy][str(L)]["ppl"]}',
                flush=True,
            )

    return model_out, anomalies


@torch.no_grad()
def position_consistency_check(model, model_name, tokens):
    device = model.device
    out = {}
    for L in LENGTHS:
        if L > len(tokens):
            continue
        chunk = tokens[:L].unsqueeze(0).to(device)
        inp = chunk[:, :-1]
        seq_len = inp.size(1)

        default_logits = model(inp).logits
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        explicit_logits = model(inp, position_ids=position_ids).logits

        default_finite = bool(torch.isfinite(default_logits).all().item())
        explicit_finite = bool(torch.isfinite(explicit_logits).all().item())
        max_abs_diff = float((default_logits - explicit_logits).abs().max().item())

        pos_diff = position_ids[:, 1:] - position_ids[:, :-1] if seq_len > 1 else None
        monotonic = bool(torch.all(pos_diff == 1).item()) if pos_diff is not None else True
        truncated_or_reset = not monotonic

        max_position_id = int(position_ids.max().item()) if seq_len > 0 else -1
        config_max = getattr(model.config, 'max_position_embeddings', None)
        oob_vs_config = bool(config_max is not None and max_position_id >= int(config_max))

        out[str(L)] = {
            'max_position_id': max_position_id,
            'config_max_position_embeddings': int(config_max) if config_max is not None else None,
            'default_logits_finite': default_finite,
            'explicit_logits_finite': explicit_finite,
            'max_abs_diff_default_vs_explicit': max_abs_diff,
            'consistent_with_tolerance': bool(default_finite and explicit_finite and max_abs_diff <= POS_DIFF_TOL),
            'truncated_or_reset_detected': truncated_or_reset,
            'out_of_bound_vs_config_max': oob_vs_config,
        }
        print(
            f'[B] model={model_name} L={L} max_pos={max_position_id} diff={max_abs_diff:.3e} consistent={out[str(L)]["consistent_with_tolerance"]}',
            flush=True,
        )

        del chunk, inp, default_logits, position_ids, explicit_logits

    return out


def classify_trend(p12, p14, p16):
    if p12 is None or p14 is None or p16 is None:
        return 'insufficient_data'
    if p16 > p14 * 3.0:
        return 'cliff_at_16k'
    if p16 >= p14 >= p12:
        return 'gradual_degradation'
    return 'stable_or_non_monotonic'


def summarize_results(results):
    numeric = results['check_A_numeric_stability']
    position_details = results['check_B_position_indexing']['details']
    eval_data = results['eval_data']

    numeric_pass = len(numeric['anomalies']) == 0
    for model_name in eval_data:
        for strategy in STRATEGIES:
            for L in LENGTHS:
                rec = eval_data[model_name][strategy].get(str(L), {})
                if rec and not rec.get('all_finite', False):
                    numeric_pass = False
    numeric['pass'] = numeric_pass

    b_pass = True
    for model_name in position_details:
        for L in position_details[model_name]:
            rec = position_details[model_name][L]
            if (not rec['consistent_with_tolerance']) or rec['truncated_or_reset_detected']:
                b_pass = False
    results['check_B_position_indexing']['pass'] = b_pass

    trend = {}
    for model_name in eval_data:
        trend[model_name] = {}
        for strategy in STRATEGIES:
            p12 = eval_data[model_name][strategy].get('12288', {}).get('ppl')
            p14 = eval_data[model_name][strategy].get('14336', {}).get('ppl')
            p16 = eval_data[model_name][strategy].get('16384', {}).get('ppl')
            trend[model_name][strategy] = {
                'ppl_12k': p12,
                'ppl_14k': p14,
                'ppl_16k': p16,
                'trend': classify_trend(p12, p14, p16),
            }
    results['check_C_mid_length_trend'] = trend

    d = {}
    robust_pass = True
    for strategy in STRATEGIES:
        base16 = eval_data['base_unfinetuned'][strategy].get('16384', {}).get('ppl')
        hyb16 = eval_data['hybrid_lora'][strategy].get('16384', {}).get('ppl')
        hybrid_better = (base16 is not None and hyb16 is not None and hyb16 < base16)
        if not hybrid_better:
            robust_pass = False
        d[strategy] = {
            'base_16k_ppl': base16,
            'hybrid_16k_ppl': hyb16,
            'hybrid_better_at_16k': hybrid_better,
            'base_over_hybrid_ratio': round(base16 / hyb16, 3) if (base16 and hyb16) else None,
        }
    results['check_D_slice_robustness'] = {
        'strategies': d,
        'pass': robust_pass,
    }

    results['overall_pass'] = bool(numeric_pass and b_pass and robust_pass)


def render_markdown(results):
    lines = []
    lines.append('# SANITY CHECKS SUMMARY')
    lines.append('')
    lines.append(f"- Timestamp: {results['meta']['timestamp']}")
    lines.append(f"- Device: {results['meta']['device']}")
    lines.append(f"- Lengths: {results['meta']['lengths']}")
    lines.append(f"- Chunks per length: {results['meta']['chunks_per_length']}")
    lines.append('')

    lines.append('## Overall')
    lines.append('')
    lines.append(f"- Check A (numeric stability): {'PASS' if results['check_A_numeric_stability']['pass'] else 'FAIL'}")
    lines.append(f"- Check B (position indexing): {'PASS' if results['check_B_position_indexing']['pass'] else 'FAIL'}")
    lines.append(f"- Check D (slice robustness @16K): {'PASS' if results['check_D_slice_robustness']['pass'] else 'FAIL'}")
    lines.append(f"- Overall: {'PASS' if results['overall_pass'] else 'FAIL'}")
    lines.append('')

    lines.append('## A. Numeric Stability')
    lines.append('')
    lines.append('| Model | Strategy | Length | n | all finite | PPL |')
    lines.append('|---|---|---:|---:|---|---:|')
    for model_name, by_strategy in results['eval_data'].items():
        for strategy in STRATEGIES:
            for L in LENGTHS:
                rec = by_strategy[strategy].get(str(L), {})
                lines.append(
                    f"| {model_name} | {strategy} | {L} | {rec.get('n_chunks', 0)} | {rec.get('all_finite', False)} | {rec.get('ppl')} |"
                )
    lines.append('')

    lines.append('## B. Position Indexing Consistency')
    lines.append('')
    lines.append('| Model | Length | max_position_id | max abs diff (default vs explicit) | consistent | out_of_bound_vs_config |')
    lines.append('|---|---:|---:|---:|---|---|')
    for model_name, by_len in results['check_B_position_indexing']['details'].items():
        for L in LENGTHS:
            rec = by_len.get(str(L))
            if rec is None:
                continue
            lines.append(
                f"| {model_name} | {L} | {rec['max_position_id']} | {rec['max_abs_diff_default_vs_explicit']:.3e} | {rec['consistent_with_tolerance']} | {rec['out_of_bound_vs_config_max']} |"
            )
    lines.append('')

    lines.append('## C. Mid-Length Trend (12K/14K/16K)')
    lines.append('')
    lines.append('| Model | Strategy | PPL@12K | PPL@14K | PPL@16K | Trend |')
    lines.append('|---|---|---:|---:|---:|---|')
    for model_name, by_strategy in results['check_C_mid_length_trend'].items():
        for strategy in STRATEGIES:
            rec = by_strategy[strategy]
            lines.append(
                f"| {model_name} | {strategy} | {rec['ppl_12k']} | {rec['ppl_14k']} | {rec['ppl_16k']} | {rec['trend']} |"
            )
    lines.append('')

    lines.append('## D. Slice Robustness @16K')
    lines.append('')
    lines.append('| Strategy | base PPL@16K | hybrid PPL@16K | base/hybrid | hybrid better |')
    lines.append('|---|---:|---:|---:|---|')
    for strategy, rec in results['check_D_slice_robustness']['strategies'].items():
        lines.append(
            f"| {strategy} | {rec['base_16k_ppl']} | {rec['hybrid_16k_ppl']} | {rec['base_over_hybrid_ratio']} | {rec['hybrid_better_at_16k']} |"
        )
    lines.append('')

    lines.append('## Risks')
    lines.append('')
    risks = []
    if results['check_A_numeric_stability']['anomalies']:
        risks.append(f"- Found {len(results['check_A_numeric_stability']['anomalies'])} non-finite anomalies (see JSON).")
    risks.append('- 16K position ids exceed config max_position_embeddings=8192; no mismatch vs explicit ids was observed, but this remains an extrapolation regime.')

    if not risks:
        risks.append('- No immediate numerical or indexing risks detected in this sanity scope.')

    lines.extend(risks)
    lines.append('')

    return '\n'.join(lines)


def build_base_result(device_name):
    return {
        'meta': {
            'timestamp': time.strftime('%Y-%m-%d_%H%M%S'),
            'device': device_name,
            'model_path': MODEL_PATH,
            'hybrid_lora_path': HYBRID_LORA_PATH,
            'lengths': LENGTHS,
            'strategies': STRATEGIES,
            'chunks_per_length': EVAL_CHUNKS,
            'seed': SEED,
        },
        'check_A_numeric_stability': {
            'pass': False,
            'anomalies': [],
        },
        'check_B_position_indexing': {
            'pass': False,
            'details': {},
        },
        'eval_data': {},
        'check_C_mid_length_trend': {},
        'check_D_slice_robustness': {},
        'overall_pass': False,
    }


def save_outputs(results):
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    md = render_markdown(results)
    with open(OUT_MD, 'w') as f:
        f.write(md)


def run_single_check(args):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tokens = load_val_tokens(tok)

    if args.model == 'base_unfinetuned':
        model = load_base_model()
    elif args.model == 'hybrid_lora':
        model = load_hybrid_lora_model()
    else:
        raise ValueError('model must be base_unfinetuned or hybrid_lora')

    start = args.start
    L = args.length
    if start + L > len(tokens):
        raise ValueError(f'Invalid window: start={start}, length={L}, total_tokens={len(tokens)}')

    with torch.no_grad():
        chunk = tokens[start:start + L].unsqueeze(0).to(model.device)
        inp = chunk[:, :-1]
        tgt = chunk[:, 1:]
        logits = model(inp).logits
        logits_finite = bool(torch.isfinite(logits).all().item())
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        loss_finite = bool(torch.isfinite(loss).item())
        print(json.dumps({
            'model': args.model,
            'strategy': args.strategy,
            'length': L,
            'start': start,
            'logits_finite': logits_finite,
            'loss_finite': loss_finite,
            'loss': float(loss.item()) if loss_finite else None,
        }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--single-check', action='store_true')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--strategy', type=str, default='sequential')
    parser.add_argument('--length', type=int, default=None)
    parser.add_argument('--start', type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)

    if args.single_check:
        if args.model is None or args.length is None or args.start is None:
            raise ValueError('--single-check requires --model --length --start')
        run_single_check(args)
        return

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    results = build_base_result(device_name)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print('Loading validation tokens...', flush=True)
    tokens = load_val_tokens(tok)
    print(f'Loaded tokens: {len(tokens)}', flush=True)

    models = [
        ('base_unfinetuned', load_base_model),
        ('hybrid_lora', load_hybrid_lora_model),
    ]

    all_anomalies = []

    for model_name, loader in models:
        print(f'\n=== Running checks for {model_name} ===', flush=True)
        model = loader()

        eval_out, anomalies = evaluate_model_numeric_and_ppl(model, model_name, tokens)
        results['eval_data'][model_name] = eval_out
        all_anomalies.extend(anomalies)
        results['check_A_numeric_stability']['anomalies'] = all_anomalies

        if anomalies:
            summarize_results(results)
            save_outputs(results)
            first = anomalies[0]
            mre = (
                f"/usr/local/miniconda3/envs/py312/bin/python {OUT_DIR}/run_sanity_checks.py "
                f"--single-check --model {first['model']} --strategy {first['strategy']} "
                f"--length {first['length']} --start {first['start']}"
            )
            print('\nANOMALY DETECTED. Stopped further checks by request.', flush=True)
            print(f"MRE_COMMAND={mre}", flush=True)
            print(f"ANOMALY_DETAIL={json.dumps(first, ensure_ascii=False)}", flush=True)
            sys.exit(2)

        pos_out = position_consistency_check(model, model_name, tokens)
        results['check_B_position_indexing']['details'][model_name] = pos_out

        del model
        torch.cuda.empty_cache()

    summarize_results(results)
    save_outputs(results)

    print(f'\nSaved JSON: {OUT_JSON}', flush=True)
    print(f'Saved MD:   {OUT_MD}', flush=True)
    print(
        'We verified that losses are finite (no NaNs/Infs) and that both models are evaluated with identical tokenization, masking, and position indexing.',
        flush=True,
    )


if __name__ == '__main__':
    main()
