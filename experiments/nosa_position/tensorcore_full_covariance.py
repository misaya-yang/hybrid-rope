"""Full second-order reference with native BF16 products and FP32 output.

Only the raw QK projection changes: use native Q/K input bits, FP32 accumulation
and output, then scale by sqrt(D). All cumulants, causal current-block mass,
GQA normalization, quotas and reader use the prior implementation. Different
GEMM/reassociation rounding is measured, not claimed bitwise-identical. This
still scores every key and does not provide sparse asymptotic complexity.
"""
import argparse
import fcntl
import hashlib
import math
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from . import run as base
from .runtime import SelectionContext
from .full_covariance_probe import FullCovarianceSelector


@torch.no_grad()
def native_bf16_projection(query,keys):
    h,g,n,d=query.shape
    if query.device.type!='cuda' or query.dtype!=torch.bfloat16 or keys.dtype!=torch.bfloat16:
        raise ValueError('native BF16 CUDA operands required')
    previous=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        dots=torch.bmm(query.reshape(h,g*n,d),keys.transpose(1,2),out_dtype=torch.float32)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=previous
    return dots.reshape(h,g,n,keys.shape[1])/math.sqrt(d)


@torch.no_grad()
def tensorcore_full_covariance_logmass(context: SelectionContext):
    """Full causal second-order score with native BF16 products and FP32 output."""
    q, k, cis, settings = context.q, context.k, context.cis, context.settings
    if q.ndim != 3 or k.ndim != 3 or cis.shape != k.shape[:2]:
        raise ValueError("expected Q[Hq,Q,D], K[KV,T,D], CIS[KV,T]")
    kvh, length, dim = k.shape
    heads, queries, qdim = q.shape
    if min(kvh, length, dim, heads, queries) < 1 or qdim != dim or heads % kvh:
        raise ValueError("invalid query/key dimensions or contiguous GQA grouping")
    if context.query_positions.shape != (queries,) or bool((context.query_positions < 0).any()) or bool((context.query_positions >= length).any()):
        raise ValueError("query positions must identify visible positions within K")
    if q.device.type != 'cuda' or q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError('native BF16 CUDA Q/K required; no precision fallback')
    dtype = torch.float32
    group, size = heads // kvh, settings.block_size
    blocks = math.ceil(length / size)
    padding = blocks * size - length
    parts = []
    with torch.autocast(device_type=q.device.type, enabled=False):
        query = q.reshape(kvh, group, queries, dim)
        block_cis = F.pad(cis.to(dtype), (0, padding), value=-torch.inf).reshape(kvh, blocks, size)
        weights = block_cis.softmax(-1)
        log_cis_weight = block_cis.logsumexp(-1)
        endpoints = (torch.arange(blocks, device=q.device) + 1) * size - 1
        offsets = torch.arange(size, device=q.device)
        for begin in range(0, queries, settings.attention_query_chunk_size):
            end = min(queries, begin + settings.attention_query_chunk_size)
            positions = context.query_positions[begin:end]
            scores = native_bf16_projection(query[:, :, begin:end], k)
            scores = F.pad(scores, (0, padding), value=0).reshape(kvh, group, end - begin, blocks, size)
            w = weights[:, None, None]
            mean = (w * scores).sum(-1)
            variance = (w * (scores - mean[..., None]).square()).sum(-1)
            logmass = log_cis_weight[:, None, None] + mean + 0.5 * variance
            logmass.masked_fill_(~(endpoints[None] <= positions[:, None])[None, None], -torch.inf)
            # Current blocks are exact even when this query ends a full block.
            current = positions // size
            index = current[None, None, :, None, None].expand(kvh, group, -1, 1, size)
            current_scores = scores.gather(-2, index).squeeze(-2)
            current_logits = current_scores + block_cis[:, current][:, None]
            visible = offsets[None] <= positions[:, None] % size
            exact_partial = current_logits.masked_fill(~visible[None, None], -torch.inf).logsumexp(-1)
            logmass.scatter_(-1, current[None, None, :, None].expand(kvh, group, -1, 1), exact_partial[..., None])
            parts.append(logmass)
    output = torch.cat(parts, dim=2)
    if bool(torch.isnan(output).any()) or bool(torch.isposinf(output).any()):
        raise FloatingPointError("invalid full second-order diagnostic score")
    return output


class TensorCoreFullCovarianceSelector(FullCovarianceSelector):
    def __init__(self,mode='full_covariance_tc',**kwargs):
        self.tensorcore=mode=='full_covariance_tc'
        super().__init__('full_covariance' if self.tensorcore else mode,**kwargs)
    @torch.no_grad()
    def logmass(self,context):
        if not self.tensorcore:return super().logmass(context)
        result=tensorcore_full_covariance_logmass(context)
        self.metrics['full_covariance_raw_key_scores']+=context.q.shape[0]*context.q.shape[1]*context.k.shape[1]
        return result


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--queue-root',default='/root/autodl-tmp/position_overnight_20260909')
    options,remaining=parser.parse_known_args()
    if not options.execute:
        print({'status':'DRY_RUN','runner_argv':remaining});return
    old_factory,old_modes,old_hashes,old_argv=base.BlockSummarySelector,base.MODES,base.source_hashes,sys.argv
    def hashes():
        files=(Path(__file__),Path(__file__).with_name('full_covariance_probe.py'),Path(__file__).with_name('exact_probe.py'))
        return {**old_hashes(),**{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    root=Path(options.queue_root)
    with (root/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'STOP').exists():raise RuntimeError('global STOP')
        try:
            base.BlockSummarySelector=TensorCoreFullCovarianceSelector
            base.MODES=(*old_modes,'full_covariance','full_covariance_tc');base.source_hashes=hashes
            sys.argv=[str(Path(__file__)),*remaining];base.main()
        finally:
            base.BlockSummarySelector,base.MODES,base.source_hashes,sys.argv=old_factory,old_modes,old_hashes,old_argv

if __name__=='__main__':main()
