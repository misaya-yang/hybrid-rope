"""Dense formula and cached/full-prefix checks for the relative clock operator."""
import math

import torch

from .distance_operator import attention, rotate
from .worker import save, sha


def check_kernel(tables):
    torch.manual_seed(20260910)
    n=97;w=13
    q=torch.randn(1,16,n,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,2,n,128,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    native=torch.tensor(tables['Native']['values_float32'],device='cuda')
    mr=torch.tensor(tables['MrPro']['values_float32'],device='cuda');gain=tables['MrPro']['gain']
    p=torch.arange(n,device='cuda');delta=p[:,None]-p[None]
    ql=rotate(q,p,native,gain).float();kl=rotate(k,p,native,gain).repeat_interleave(8,1).float()
    qr=rotate(q,p,mr,gain,w*(native-mr)).float();kr=rotate(k,p,mr,gain).repeat_interleave(8,1).float()
    z=torch.where((delta<=w)[None,None],ql@kl.transpose(-1,-2),qr@kr.transpose(-1,-2))/math.sqrt(128)
    z.masked_fill_((delta<0)[None,None],-torch.inf)
    expected=z.softmax(-1)@v.repeat_interleave(8,1).float()
    with torch.inference_mode():
        got=attention(q,k,v,w,native,mr,gain,1/math.sqrt(128)).float()
        err=float((got-expected).square().mean().sqrt()/expected.square().mean().sqrt())
        assert err<.02,err
        last=attention(q[:,:,-1:],k,v,w,native,mr,gain,1/math.sqrt(128)).float()
        decode_err=float((last-expected[:,:,-1:]).square().mean().sqrt()/expected[:,:,-1:].square().mean().sqrt())
        assert decode_err<.02,decode_err
    return dict(dense_relative_rms=err,decode_relative_rms=decode_err)


def run(worker,job):
    checked=check_kernel(worker.tables);w=13
    with torch.inference_mode():
        spec=dict(operator='distance',window=w,table=worker.tables['MrPro'])
        worker.apply(spec)
        tokens=torch.tensor([worker.screen[0]['prompt_ids'][:512]],device='cuda')
        whole=worker.model(tokens,use_cache=False,logits_to_keep=8).logits.float()
        prefix=worker.model(tokens[:,:256],use_cache=True,logits_to_keep=1)
        chunked=worker.model(tokens[:,256:],past_key_values=prefix.past_key_values,use_cache=True,logits_to_keep=8).logits.float()
        partition_rms=float((whole-chunked).square().mean().sqrt()/whole.square().mean().sqrt())
        assert partition_rms<.02,partition_rms
        worker.apply({'table':worker.tables['MrPro']})
    result=dict(status='PASS',**checked,
        model_partition_relative_rms=partition_rms,backend='PyTorch compiled FlexAttention, two block masks, joint logsumexp',
        source_sha256=sha(__file__),scope='Formula/partition checks, not a method outcome')
    save(worker.root/'distance_checks.json',result)
    return result


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--tables',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args()
    result=dict(status='PASS',**check_kernel(json.loads(Path(args.tables).read_text())),scope='Kernel formula only; no model loaded')
    save(args.out,result);print(json.dumps(result))
