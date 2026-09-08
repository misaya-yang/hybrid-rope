"""Parameter-free discrete boundary-matched allocation on a native RoPE grid.

Verified local benefit: OLMo-2-0425-1B-Instruct, static S=4, six-task RULER
subsets at up to 16K. This is not a guarantee for other models or scales.
The returned gain multiplies cos/sin amplitudes, not phase or position IDs.
"""
from __future__ import annotations

import math
import torch


def boundary_matched_inv_freq(
    native_inv_freq: torch.Tensor, *, base: float, reference_length: int, scale: float
) -> tuple[torch.Tensor, float, dict]:
    """Return an FP32 table, scalar cos/sin gain, and exact construction metadata.

    Supply the unscaled native FP32 frequencies. Install the returned table
    after converting the model to BF16/FP16, so phase frequencies stay FP32.
    This operator does not choose a deployment scale or alter KV caches.
    """
    if (native_inv_freq.ndim != 1 or native_inv_freq.dtype != torch.float32
        or native_inv_freq.numel() < 2):
        raise ValueError('native frequencies must be a one-dimensional FP32 tensor')
    if not math.isfinite(base) or base <= 1 or reference_length <= 0 or not math.isfinite(scale) or scale < 1:
        raise ValueError('invalid native geometry or extension scale')
    native = native_inv_freq.detach().cpu()
    dim = 2*native.numel()
    expected = 1/(base**(torch.arange(0,dim,2,dtype=torch.float32)/dim))
    if not torch.allclose(native,expected,rtol=3e-7,atol=0):
        raise ValueError('input is not the declared native endpoint grid')
    if scale == 1:
        return native_inv_freq.clone(),1.,dict(method='boundary_matched',scale=1.,identity=True)
    turns = [float(w)*reference_length/(2*math.pi) for w in native]
    fast = [j for j,t in enumerate(turns) if t>32]
    slow = [j for j,t in enumerate(turns) if t<1]
    if not fast or not slow or slow[0]<=fast[-1]:
        raise ValueError('native grid has no valid MrPro 32/1-turn transition')
    low,high = fast[-1],slow[0]
    n = high-low
    exponents = []
    values = []
    for j,w in enumerate(native.tolist()):
        q = min(n,max(0,j-low))
        m = q*(q+1)*(3*n+2-2*q)/(n*(n+1)*(n+2))
        exponents.append(m)
        values.append(w if j<=low else w/scale if j>=high else w*scale**(-m))
    result = torch.tensor(values,dtype=torch.float32,device=native_inv_freq.device)
    if not torch.isfinite(result).all() or not torch.all(result>0) or not torch.all(result[:-1]>result[1:]):
        raise ValueError('scale produces an invalid finite FP32 table')
    return result,1+.1*math.log(scale),dict(method='boundary_matched',low=low,high=high,
        N=n,scale=scale,reference_length=reference_length,base=base,exponents=exponents,
        formula='m_q=q*(q+1)*(3*N+2-2*q)/(N*(N+1)*(N+2))')
