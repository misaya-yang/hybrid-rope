"""Native full-row captures for the supplied Pro block-stretch hypothesis.

This module never consumes the historical selected-key replay as a full row.
Calibration uses natural text, not task references or forced answer prefixes.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import types

import numpy as np
import torch

from .capture import rotate
from .worker import Worker, digest, save


def capture(args):
    worker = Worker(args.root, args.history)
    common_gain = worker.tables['MrPro']['gain']
    source_table = dict(worker.tables['Native'], gain=common_gain)
    worker.apply({'table': source_table})
    root = Path(args.root)/'native_full_rows'
    length = 32768
    manifest = {'source_frequency_table': source_table,
        'source_description':'Native frequencies at the common MrPro comparison gain; not the separate operational Native gain=1 baseline',
        'common_gain':common_gain, 'length':length,'fit_docs':[0,1,2,3],
        'validation_docs':[4,5], 'block_size':4096,'scale':4,
        'source_kind':'Existing natural-text documents; no RULER references or answer trajectories',
        'key_support':'Every causally visible key for every saved row',
        'docs':[]}
    for index,doc in enumerate(worker.nll_manifest['docs'][:6]):
        folder = root/f'doc_{index:02d}'
        if (folder/'complete.json').exists():
            manifest['docs'].append(json.loads((folder/'complete.json').read_text()))
            continue
        folder.mkdir(parents=True,exist_ok=True)
        ids = np.load(worker.nll_inputs/doc['file'])[:length].tolist()
        if len(ids) != length:
            raise ValueError('Calibration requires the declared full source window')
        selected_heads = torch.tensor([index%4+4*i for i in range(4)],device='cuda')
        selected_kv = selected_heads//8
        originals, receipts = [], []
        qpos = length-1

        def make_forward(original, layer_index):
            def forward(module, hidden_states, position_embeddings, attention_mask,
                        past_key_values=None, **kwargs):
                if hidden_states.shape[:2] != (1,length) or past_key_values is not None:
                    raise ValueError('Expected one complete uncached native prefix')
                q = module.q_proj(hidden_states[:,qpos:qpos+1]).view(1,1,16,128)[0,0,selected_heads]
                k = module.k_proj(hidden_states).view(1,length,2,128)[0].transpose(0,1)
                v = module.v_proj(hidden_states).view(1,length,2,128)[0].transpose(0,1)
                cos,sin = position_embeddings
                qr = rotate(q,cos[0,qpos],sin[0,qpos])
                kr = rotate(k,cos[0][None],sin[0][None])
                actual_scores = torch.einsum('hd,hkd->hk',qr.float(),kr[selected_kv].float())*module.scaling
                d = torch.arange(length,device='cuda',dtype=torch.float32)-qpos
                phase = d[:,None]*worker.model.model.rotary_emb.inv_freq[None,:]
                q0,q1=q[:,:64].float(),q[:,64:].float()
                k0,k1=k[selected_kv,:,:64].float(),k[selected_kv,:,64:].float()
                c=q0[:,None,:]*k0+q1[:,None,:]*k1
                s=q1[:,None,:]*k0-q0[:,None,:]*k1
                exact_scores=(c*phase.cos()+s*phase.sin()).sum(-1)*module.scaling*common_gain**2
                actual_logp=actual_scores.log_softmax(-1)
                exact_logp=exact_scores.log_softmax(-1)
                replay_kl=(actual_logp.exp()*(actual_logp-exact_logp)).sum(-1)
                payload={'doc':index,'layer':layer_index,'query_position':qpos,
                    'selected_heads':selected_heads.cpu(),'selected_kv':selected_kv.cpu(),
                    'q_raw':q.cpu(),'k_raw':k.cpu(),'v_raw':v.cpu(),
                    'native_exact_logp':exact_logp.cpu(),
                    'native_runtime_logp':actual_logp.cpu(),
                    'runtime_to_exact_replay_KL':replay_kl.cpu(),
                    'attention_score_scale':module.scaling*common_gain**2,
                    'signed_separation':'key minus query','total_keys':length}
                torch.save(payload,folder/f'layer_{layer_index:02d}.pt')
                receipts.append({'layer':layer_index,'mean_runtime_to_exact_KL':float(replay_kl.mean()),
                    'maximum_runtime_to_exact_KL':float(replay_kl.max())})
                del q,k,v,qr,kr,c,s,k0,k1,q0,q1,payload,actual_scores,exact_scores
                return original(hidden_states,position_embeddings,attention_mask,
                    past_key_values=past_key_values,**kwargs)
            return forward

        for li,layer in enumerate(worker.model.model.layers):
            attn=layer.self_attn; originals.append((attn,attn.forward))
            attn.forward=types.MethodType(make_forward(attn.forward,li),attn)
        started=time.monotonic()
        try:
            with torch.inference_mode():
                worker.model(torch.tensor([ids],device='cuda'),use_cache=False,logits_to_keep=1)
        finally:
            for attn,original in originals:attn.forward=original
        record={'doc':index,'source':doc,'input_tokens_sha256':digest(ids),
            'query_position':qpos,'selected_heads':selected_heads.tolist(),
            'layers':receipts,'seconds':time.monotonic()-started,
            'status':'COMPLETE','all_keys_retained':True}
        save(folder/'complete.json',record)
        manifest['docs'].append(record)
        save(root/'manifest.json',manifest)
        print(json.dumps({'capture_doc':index,'seconds':record['seconds'],
            'max_runtime_replay_KL':max(r['maximum_runtime_to_exact_KL'] for r in receipts)}),flush=True)
    save(root/'manifest.json',manifest)
    del worker
    gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--history',default='/root/autodl-tmp/bm_transfer_20260908')
    parser.add_argument('--mode',choices=['capture'],default='capture')
    args=parser.parse_args()
    capture(args)
