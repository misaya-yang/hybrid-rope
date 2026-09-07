"""Capture all-token Native Q/K means on the existing eight C documents.

One zero-training construction pass for the author-selected carrier estimator;
no generation, teacher answers, gradients, candidate search, or validation data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from scripts.experiments.cross_audit.runtime import cuda_runtime
from scripts.experiments.cross_audit.tables import tensor_sha
from scripts.experiments.scale_transport.run import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic(); root = args.root
    args.out.mkdir(parents=True, exist_ok=False)
    manifest_path = root/'prepared_v2/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    docs = [d for d in manifest['docs'] if d['split'] == 'C']
    if len(docs) != 8:
        raise ValueError('reuse exactly the eight existing construction documents')
    for d in docs:
        path = root/'prepared_v2'/d['file']
        # docs[].sha256 identifies source text; files[] identifies NPY bytes.
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['files'][d['file']]:
            raise ValueError('frozen construction file changed')
    hardware = cuda_runtime()
    model = AutoModelForCausalLM.from_pretrained(root/'model', local_files_only=True,
        dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='sdpa').eval()
    native = model.model.rotary_emb.inv_freq.float().cpu().numpy()
    identity = json.loads((root/'runs/pilot_01/source_identity.json').read_text())
    if tensor_sha(native) != identity['actual_native_sha256'] or model.model.rotary_emb.attention_scaling != 1:
        raise ValueError('Native operator identity')
    layers = [5,11,17,23,29,35]; active = {}; means = {}; handles = []
    for layer in layers:
        for name, heads in [('q',16),('k',2)]:
            module = getattr(model.model.layers[layer].self_attn, name+'_proj')
            def hook(module, args, result, layer=layer, name=name, heads=heads):
                if result.shape[:2] != (1,32768) or result.shape[-1] != heads*128:
                    raise ValueError('all-token pre-RoPE Q/K layout')
                mean = result.detach().mean(dim=(0,1), dtype=torch.float32).reshape(heads,128)
                if not torch.isfinite(mean).all():
                    raise ValueError('nonfinite Native means')
                means[f'{active["doc"]}_{layer}_{name}'] = mean.cpu().numpy()
            handles.append(module.register_forward_hook(hook))
    with torch.inference_mode():
        for d in docs:
            if time.monotonic()-start > 300:
                raise TimeoutError('five-minute construction cap')
            path = root/'prepared_v2'/d['file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['files'][d['file']]:
                raise ValueError('frozen construction document changed')
            tokens = np.load(path, allow_pickle=False)
            if tokens.shape != (32769,):
                raise ValueError('original 32K context plus held-out next token')
            active['doc'] = path.stem; before = time.monotonic()
            result = model(torch.tensor(tokens[:-1].astype(np.int64),device='cuda')[None,:],
                           use_cache=False,logits_to_keep=1)
            if not torch.isfinite(result.logits).all():
                raise ValueError('nonfinite construction forward')
            print(json.dumps({'doc':active['doc'],'seconds':time.monotonic()-before}),flush=True)
    for handle in handles:
        handle.remove()
    np.savez(args.out/'means.npz', **means, native=native)
    write(args.out/'manifest.json', {'status':'COMPLETE_NATIVE_BACKGROUND_MEANS',
        'zero_training':True,'model_weight_updates':0,'generated_answers':0,
        'query_measure':'all 32768 pre-RoPE query vectors per C document, not the old eight-query subset',
        'key_measure':'all 32768 pre-RoPE key vectors per C document',
        'layers':layers,'query_heads':16,'kv_heads':2,'head_dim':128,
        'docs':[{'doc':Path(d['file']).stem,'source_text_sha256':d['sha256'],
                 'npy_file_sha256':manifest['files'][d['file']]} for d in docs],
        'native_tensor_sha256':tensor_sha(native),'model_revision':identity['revision'],
        'prepared_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'means_npz_sha256':hashlib.sha256((args.out/'means.npz').read_bytes()).hexdigest(),
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'hardware':hardware,'elapsed_seconds':time.monotonic()-start})


if __name__ == '__main__':
    main()
