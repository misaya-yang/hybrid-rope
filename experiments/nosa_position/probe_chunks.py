"""Compare execution chunks on one existing 16K input; save full token parity."""
import argparse
import json
import time
from pathlib import Path

import torch

from .runtime import AttentionSettings, NosaReferenceForCausalLM, native_select
from .selector_controls import BlockSummarySelector


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    args = p.parse_args()
    root = Path(args.root)
    output = root / 'runs/chunk_probe_v1'
    output.mkdir(exist_ok=True)
    rows = [json.loads(x) for x in (root / 'data/pc2/rows.jsonl').read_text().splitlines()]
    row = next(r for r in rows if r['row_id'] == 'ruler_dev_niah_multikey_1_16384_000')
    torch.set_num_threads(4)
    model = NosaReferenceForCausalLM.from_pretrained('/root/autodl-tmp/NOSA-1B', device='cuda', dtype=torch.bfloat16)
    ids = torch.tensor([row['prompt_ids']], device='cuda')
    eos = model.config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    results = []
    for selector in ['native', 'pc2']:
        reference = None
        for chunk, query in [(128, 16), (128, 64), (512, 64)]:
            model.selector = native_select if selector == 'native' else BlockSummarySelector(selector)
            model.settings = AttentionSettings(attention_query_chunk_size=query)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            value = model.prefill(ids, chunk_size=chunk)
            torch.cuda.synchronize()
            prefill = time.perf_counter() - started
            logits = value.logits.cpu()
            tokens = []
            for _ in range(row['max_new_tokens']):
                token = int(value.logits[0, -1].argmax())
                tokens.append(token)
                if token in eos:
                    break
                value = model(torch.tensor([[token]], device='cuda'), past_key_values=value.past_key_values, num_logits_to_keep=1)
            torch.cuda.synchronize()
            result = dict(row_id=row['row_id'], selector=selector, chunk_size=chunk,
                          attention_query_chunk_size=query, prefill_seconds=prefill,
                          total_seconds=time.perf_counter()-started, tokens_per_second=ids.numel()/prefill,
                          peak_bytes=torch.cuda.max_memory_allocated(), generated_token_ids=tokens,
                          finite_logits=bool(torch.isfinite(logits).all()))
            if reference is None:
                reference = (logits, tokens)
            result.update(logits_max_abs=float((logits-reference[0]).abs().max()),
                          logits_equal=bool(torch.equal(logits, reference[0])), tokens_equal=tokens == reference[1])
            results.append(result)
            (output / 'results.json').write_text(json.dumps(results, indent=2)+'\n')
            print(json.dumps(result), flush=True)
            del value, logits


if __name__ == '__main__':
    main()
