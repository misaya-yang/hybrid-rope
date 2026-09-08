"""Paired tail-token NLL on frozen natural documents; no generation claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from .prepare import sha_file, verify_weight_stats, write


def prepare(a):
    from transformers import AutoTokenizer
    manifest = json.loads((a.prepared/'manifest.json').read_text())
    source = json.loads(a.source_manifest.read_text())
    tok = AutoTokenizer.from_pretrained(manifest['model_path'], local_files_only=True)
    source_tok = AutoTokenizer.from_pretrained(a.source_tokenizer, local_files_only=True)
    a.out.mkdir(exist_ok=False)
    docs = []
    for old in source['docs']:
        source_file = a.source_manifest.parent/old['file']
        if sha_file(source_file) != source['files'][old['file']]:
            raise ValueError('frozen source token file changed')
        text = source_tok.decode(np.load(source_file).tolist(), skip_special_tokens=False)
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) < max(a.lengths)+1:
            raise ValueError('selected prefix too short for frozen lengths')
        name = old['file']
        np.save(a.out/name, np.asarray(ids[:max(a.lengths)+1], dtype=np.int32))
        docs.append(dict(file=name, sha256=sha_file(a.out/name), source_row=old['source_row'],
            source_tokens_sha256=source['files'][name], decoded_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            original_tokens=len(ids)))
    write(a.out/'manifest.json', dict(docs=docs, lengths=a.lengths, tail_tokens=512,
        model_manifest_sha256=sha_file(a.prepared/'manifest.json'),
        source_manifest_sha256=sha_file(a.source_manifest),
        source_tokenizer_sha256=sha_file(Path(a.source_tokenizer)/'tokenizer.json'),
        scope='Existing 16 long FineWeb-Edu prefixes decoded from frozen Qwen tokens and retokenized for target. Original parquet unavailable. Tail 512 next-token NLL, not full-document NLL or generation capability.'))


def run(a):
    import torch
    from transformers import AutoModelForCausalLM
    from .runtime import install, verify
    from .run import atomic
    manifest = json.loads((a.prepared/'manifest.json').read_text())
    data = json.loads((a.inputs/'manifest.json').read_text())
    if data['model_manifest_sha256'] != sha_file(a.prepared/'manifest.json'):
        raise ValueError('model preparation changed')
    verify_weight_stats(manifest)
    for name, expected in manifest['model_files_sha256'].items():
        if sha_file(Path(manifest['model_path'])/name) != expected:
            raise ValueError('model metadata changed')
    if sha_file(a.prepared/'tables.json') != manifest['prepared_files']['tables.json']:
        raise ValueError('tables changed')
    for doc in data['docs']:
        if sha_file(a.inputs/doc['file']) != doc['sha256']:
            raise ValueError('input changed')
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if any(x.strip().isdigit() for x in active.splitlines()):
        raise RuntimeError('GPU busy')
    a.out.mkdir(exist_ok=False)
    atomic(a.out/'status.json',dict(status='RUNNING',pid=os.getpid()))
    started = time.monotonic()
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        model = AutoModelForCausalLM.from_pretrained(manifest['model_path'],local_files_only=True,
            dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa').eval()
        if sum(p.numel() for p in model.parameters()) != manifest['actual_parameters']:
            raise ValueError('parameter count mismatch')
        tables = json.loads((a.prepared/'tables.json').read_text())
        root = Path(__file__).resolve().parents[3]
        deps = ['scripts/experiments/olmo_fast_screen/'+x for x in
                ('natural_nll.py','runtime.py','prepare.py','run.py')]
        deps += ['scripts/experiments/cross_audit/tables.py']
        write(a.out/'runtime.json',dict(model_id=manifest['model_id'],revision=manifest['revision'],
            input_manifest_sha256=sha_file(a.inputs/'manifest.json'), tables=tables,
            code_files={p:sha_file(root/p) for p in deps}, torch=torch.__version__,
            attention='Flash SDPA', scoring='FP32 cross-entropy on final 512 next tokens'))
        for method in ('Native','MrPro','MrProBM'):
            install(model,tables[method]); verify(model,tables[method])
            raw = a.out/(method+'.jsonl')
            with raw.open('w') as f, torch.inference_mode():
                for length in data['lengths']:
                    if method == 'Native' and length > manifest['native_length']:
                        continue
                    for doc in data['docs']:
                        tokens = np.load(a.inputs/doc['file'])
                        ids = torch.tensor(tokens[:length].astype(np.int64),device='cuda')[None]
                        target = torch.tensor(tokens[length-512+1:length+1].astype(np.int64),device='cuda')
                        logits = model(ids,use_cache=False,logits_to_keep=512).logits[0].float()
                        losses = torch.nn.functional.cross_entropy(logits,target,reduction='none')
                        row = dict(method=method,doc=doc['file'],length=length,input_sha256=doc['sha256'],
                            target_ids=target.tolist(),nll=float(losses.mean()),token_nll=losses.tolist())
                        f.write(json.dumps(row)+'\n'); f.flush()
                        atomic(a.out/'live.json',dict(method=method,doc=doc['file'],length=length))
                        del logits,losses,ids,target
            write(a.out/(method+'.json'),dict(status='COMPLETE',raw_sha256=sha_file(raw)))
        atomic(a.out/'status.json',dict(status='COMPLETE',elapsed_seconds=time.monotonic()-started,
                                      peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    except Exception as exc:
        atomic(a.out/'status.json',dict(status='FAILED',error=repr(exc)))
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run'])
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--inputs',type=Path);p.add_argument('--source-manifest',type=Path)
    p.add_argument('--source-tokenizer',type=Path)
    p.add_argument('--lengths',type=int,nargs='+',default=[4096,8192,16384])
    a=p.parse_args()
    (prepare if a.phase=='prepare' else run)(a)
