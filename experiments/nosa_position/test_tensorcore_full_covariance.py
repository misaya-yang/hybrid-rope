"""CUDA-only numerical checks and isolated projection timings; no LM claims."""
import json
import math
from dataclasses import replace
from pathlib import Path
import statistics
import torch
from .runtime import AttentionSettings,SelectionContext
from .full_covariance_probe import full_covariance_logmass
from .tensorcore_full_covariance import native_bf16_projection,tensorcore_full_covariance_logmass

@torch.inference_mode()
def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(4);torch.manual_seed(20260910)
    k=torch.randn(2,257,128,device='cuda',dtype=torch.bfloat16)
    q=torch.randn(8,3,128,device='cuda',dtype=torch.bfloat16)
    cis=torch.randn(2,257,device='cuda',dtype=torch.bfloat16)
    context=SelectionContext(q,k,torch.zeros_like(k),cis,torch.tensor([63,128,256],device='cuda'),0,AttentionSettings())
    old_flag=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    ref=full_covariance_logmass(context);new=tensorcore_full_covariance_logmass(context)
    torch.testing.assert_close(new,ref,atol=2e-5,rtol=2e-5)
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction==old_flag
    changed=k.clone();changed[:,129:]*=100
    later=tensorcore_full_covariance_logmass(replace(context,k=changed))
    torch.testing.assert_close(later[:,:,:2],new[:,:,:2],atol=0,rtol=0)
    root=Path('/root/autodl-tmp/position_overnight_20260909/runs/pc2_cutoff_capture_dev_v1')
    max_error=0.;count=0
    for path in sorted(root.glob('*.pt')):
        state=torch.load(path,map_location='cpu',weights_only=True)
        if not state['changed_raw_blocks']:continue
        query=state['query'].cuda();groups=query.shape[0]//2
        for block in state['changed_raw_blocks']:
            h=block['kv_head'];qq=query[h*groups:(h+1)*groups].unsqueeze(0)
            kk=block['keys'].cuda().unsqueeze(0)
            expected=(qq.float()/math.sqrt(qq.shape[-1])) @ kk.float()[:,None].transpose(-1,-2)
            actual=native_bf16_projection(qq,kk)
            max_error=max(max_error,float((actual-expected).abs().max()));count+=1
    assert count and max_error<2e-4
    query=torch.randn(2,16,64,128,device='cuda',dtype=torch.bfloat16)
    keys=torch.randn(2,16384,128,device='cuda',dtype=torch.bfloat16)
    def original():return (query.float()/math.sqrt(128)) @ keys.float()[:,None].transpose(-1,-2)
    def optimized():return native_bf16_projection(query,keys)
    for _ in range(3):original();optimized()
    timings={'FP32_projection':[],'native_BF16_FP32_output':[]}
    for repeat in range(10):
        order=[('FP32_projection',original),('native_BF16_FP32_output',optimized)]
        if repeat%2:order.reverse()
        for name,fn in order:
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();value=fn();end.record();end.synchronize()
            timings[name].append(start.elapsed_time(end));del value
    report={'synthetic_full_score_max_abs_error':float((new-ref)[torch.isfinite(ref)].abs().max()),
        'future_perturbation_unchanged':True,'precision_flag_restored':True,
        'actual_changed_blocks_checked':count,'actual_QK_max_abs_error':max_error,
        'projection_cuda_ms_median':{k:statistics.median(v) for k,v in timings.items()},
        'projection_timing_samples_ms':timings,'scope':'operator checks plus isolated QK timing; not full-model speed or selection identity'}
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
