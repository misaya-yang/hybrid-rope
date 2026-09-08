"""BF16 inference qualification on the actual GPU; no language-model claims."""
import argparse,json,math,time
from pathlib import Path
import torch
from flash_attn import flash_attn_varlen_func
from infllm_v2 import infllmv2_attn_varlen_func,infllmv2_attn_stage1

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.manual_seed(20260908);records=[]
    with torch.inference_mode():
        for N,Q in [(257,257),(513,1)]:
            q=torch.randn(Q,32,128,device='cuda',dtype=torch.bfloat16)
            k=torch.randn(N,2,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
            cq=torch.tensor([0,Q],device='cuda',dtype=torch.int32);ck=torch.tensor([0,N],device='cuda',dtype=torch.int32)
            pos=torch.arange(Q,device='cuda') if Q==N else torch.tensor([N-1],device='cuda')
            blocks=(N+63)//64;idx=torch.arange(blocks,device='cuda',dtype=torch.int32)[None,None].expand(2,Q,-1).clone()
            idx.masked_fill_(idx>pos[None,:,None]//64,-1)
            gold=flash_attn_varlen_func(q,k,v,cq,ck,Q,N,causal=Q!=1)
            out=infllmv2_attn_varlen_func(q,k,v,cq,ck,Q,N,causal=Q!=1,topk_idx=idx,dropout_p=0.,deterministic=False,return_attn_probs=False)
            diff=(out.float()-gold.float());rel=float(diff.norm()/gold.float().norm());mx=float(diff.abs().max())
            if not bool(torch.isfinite(out).all()) or rel>.005 or mx>.03:raise ValueError(f'All-key reader parity failed: {N,Q,rel,mx}')
            records.append({'test':'all-key causal reader','keys':N,'queries':Q,'relative_l2':rel,'max_absolute_error':mx,'bitwise_equal':bool(torch.equal(out,gold))})
            # A strict nontrivial shared-GQA subset, compared with FP64 masked attention.
            idx=torch.arange(0,blocks,2,device='cuda',dtype=torch.int32)[None,None].expand(2,Q,-1).clone()
            idx.masked_fill_(idx>pos[None,:,None]//64,-1)
            chosen=torch.zeros(2,Q,N,device='cuda',dtype=torch.bool)
            for b in range(blocks):chosen[:,:,b*64:min((b+1)*64,N)]=(idx==b).any(-1)[:,:,None]
            chosen &= torch.arange(N,device='cuda')[None,None]<=pos[None,:,None]
            scores=torch.einsum('qhd,khd->hqk',q.double(),k.double().repeat_interleave(16,1))/math.sqrt(128)
            scores.masked_fill_(~chosen.repeat_interleave(16,0),-torch.inf)
            ref=torch.einsum('hqk,khd->qhd',scores.softmax(-1),v.double().repeat_interleave(16,1))
            out=infllmv2_attn_varlen_func(q,k,v,cq,ck,Q,N,causal=Q!=1,topk_idx=idx,dropout_p=0.,deterministic=False,return_attn_probs=False)
            diff=out.double()-ref;rel=float(diff.norm()/ref.norm());mx=float(diff.abs().max())
            if not bool(torch.isfinite(out).all()) or rel>.005 or mx>.03:raise ValueError(f'Subset reader failed: {N,Q,rel,mx}')
            records.append({'test':'shared-GQA sparse reader vs FP64','keys':N,'queries':Q,'relative_l2':rel,'max_absolute_error':mx})
        # Decode selector scores: no causal ambiguity, compare exact mathematical denominator.
        q=(torch.randn(1,32,128,device='cuda')*2).bfloat16();keys=torch.randn(1536,2,128,device='cuda').bfloat16()
        fine=torch.stack([keys[i:i+32].float().mean(0).bfloat16() for i in range(0,1505,16)])
        coarse=torch.stack([keys[i:i+128].float().mean(0).bfloat16() for i in range(0,1409,64)])
        cu=lambda n:torch.tensor([0,n],device='cuda',dtype=torch.int32)
        actual=infllmv2_attn_stage1(q,fine,coarse,cu(1),cu(len(fine)),cu(len(coarse)),1,len(fine),causal=False)
        sf=torch.einsum('ghd,cgd->ghc',q[0].double().reshape(2,16,128),fine.double())/math.sqrt(128)
        sc=torch.einsum('ghd,cgd->ghc',q[0].double().reshape(2,16,128),coarse.double())/math.sqrt(128)
        expected=torch.exp(sf-sc.logsumexp(-1)[:,:,None]).sum(1)[:,None]
        rel=float((actual.double()-expected).norm()/expected.norm())
        records.append({'test':'coarse-LSE shared-head score vs FP64 expression','relative_l2':rel,'shape':list(actual.shape),'expected_shape':list(expected.shape)})
        if actual.shape!=expected.shape or not bool(torch.isfinite(actual).all()) or rel>.02:raise ValueError(f'Selector score expression mismatch: {rel}')
    a.output.write_text(json.dumps({'status':'PASS','device':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'records':records},indent=2));print(json.dumps(records,indent=2))
if __name__=='__main__':main()
