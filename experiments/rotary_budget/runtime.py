"""CUDA qualification by compatible cubin and actual Flash forward/backward."""
import torch

def validate(old):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 CUDA required')
    props=torch.cuda.get_device_properties(0);major,minor=torch.cuda.get_device_capability(0)
    arches=list(torch.cuda.get_arch_list());native=f'sm_{major}{minor}' in arches
    compatible=[a for a in arches if a.startswith('sm_') and a[3:].isdigit() and int(a[3:])//10==major and int(a[3:])%10<=minor]
    if not compatible:raise RuntimeError(f'No compatible cubin: {arches} for {major}.{minor}')
    kernels=old.configure_cuda_kernels()
    q=torch.randn(1,12,128,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    k=torch.randn_like(q,requires_grad=True);v=torch.randn_like(q,requires_grad=True)
    y=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
    y.float().square().mean().backward();torch.cuda.synchronize()
    if not all(torch.isfinite(t).all() for t in (y,q.grad,k.grad,v.grad)):
        raise RuntimeError('Flash forward/backward finite check failed')
    return dict(name=props.name,capability=[major,minor],native_arch_listed=native,
      compatible_cubins=compatible,compiled_architectures=arches,total_memory_bytes=props.total_memory,
      torch=str(torch.__version__),cuda=torch.version.cuda,kernels=kernels,flash_forward_backward='PASS',
      compatibility_source='https://docs.nvidia.com/cuda/ada-compatibility-guide/index.html')
