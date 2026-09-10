"""E04 empirical projected distributions; E09 shared frame covariance controls."""
import math
import os
import hashlib
from pathlib import Path

import torch

from .exact_probe import ExactBlockSelector
from .runtime import apply_rope, select_with_scores


def exact_current_blocks(context, out, metrics=None):
    """Restore every current physical block's causal exact response in-place."""
    h, length, d = context.k.shape
    q = context.q.float().reshape(h, -1, context.q.shape[1], d) / math.sqrt(d)
    size = context.settings.block_size
    current = context.query_positions // size
    for block in current.unique().tolist():
        rows = torch.where(current == block)[0]
        start, stop = block*size, min((block+1)*size, length)
        logits = q[:, :, rows] @ context.k[:, None, start:stop].float().transpose(-1, -2)
        if metrics is not None:
            metrics['exact_raw_key_scores'] += logits.numel()
        logits += context.cis[:, None, None, start:stop].float()
        visible = torch.arange(start, stop, device=q.device)[None] <= context.query_positions[rows, None]
        out[:, :, rows, block] = logits.masked_fill(~visible[None, None], -torch.inf).logsumexp(-1)
    visible = torch.arange(out.shape[-1], device=q.device)[None] <= current[:, None]
    return out.masked_fill(~visible[None, None], -torch.inf)


class ProjectedDistributionSelector(ExactBlockSelector):
    basis_file = None
    basis_state = None
    modes = ('e04_empirical', 'e04_second', 'e09_global', 'e09_local')

    def __init__(self, mode='e04_empirical', **kwargs):
        self.variant = mode
        super().__init__('exact_mass', **kwargs)
        path = os.environ.get('PC2_QUERY_BASIS')
        if not path:
            raise ValueError('PC2_QUERY_BASIS must identify the frozen independent calibration')
        if self.variant.startswith('e09_'):
            path = os.environ.get('PC2_KEY_BASIS') or str(Path(path).parent.parent/'calibration_key_v1/key_basis.pt')
        if self.__class__.basis_file != path:
            self.__class__.basis_state = torch.load(path, map_location='cpu', weights_only=True)
            self.__class__.basis_file = path
        self.descriptors = {}
        self.bases = {}
        self.metrics['calibration_basis_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.metrics['calibration_rows_sha256'] = self.basis_state['source_rows_sha256']

    def _basis(self, context):
        layer = context.layer_idx
        if layer not in self.bases:
            field = 'local_bases' if self.variant == 'e09_local' else 'bases'
            self.bases[layer] = self.basis_state[field][layer].to(device=context.q.device)
        return self.bases[layer]

    def _build(self, context, basis):
        h, length, d = context.k.shape
        size = context.settings.block_size
        full = length // size
        if int(context.query_positions[0]) == 0:
            self.descriptors.pop(context.layer_idx, None)
        old = self.descriptors.get(context.layer_idx)
        done = old['mean'].shape[1] if old else 0
        if full > done:
            keys = context.k[:, done*size:full*size].float().reshape(h, full-done, size, d)
            bias = context.cis[:, done*size:full*size].float().reshape(h, full-done, size)
            logz = bias.logsumexp(-1)
            w = bias.softmax(-1)
            mean = (w[..., None] * keys).sum(-2)
            centered = keys - mean[:, :, None]
            if self.variant == 'e09_local':
                origins = torch.arange(done, full, device=keys.device).repeat_interleave(size) * size
                centered = apply_rope(centered.flatten(1, 2), -origins, context.rope_inv_freq, 1.).reshape_as(centered)
            projected = torch.einsum('hbtd,hds->hbts', centered, basis)
            fresh = {'mean':mean, 'logz':logz}
            if self.variant == 'e04_empirical':
                fresh.update(projected=projected, logw=bias-logz[..., None])
            else:
                fresh['covariance'] = torch.einsum('hbti,hbt,hbtj->hbij', projected, w, projected)
            old = {name:torch.cat((old[name], value),1) for name,value in fresh.items()} if old else fresh
            self.descriptors[context.layer_idx] = old
            self.metrics['projection_build_raw_k_elements'] = self.metrics.get('projection_build_raw_k_elements',0)+keys.numel()
        self.metrics['max_metadata_bytes'] = sum(t.nbytes for entry in self.descriptors.values() for t in entry.values()) + sum(t.nbytes for t in self.bases.values())
        return old, full

    @torch.no_grad()
    def logmass(self, context):
        h, length, d = context.k.shape
        size = context.settings.block_size
        queries, blocks = context.q.shape[1], math.ceil(length/size)
        basis = self._basis(context)
        cache, full = self._build(context, basis)
        q = context.q.float().reshape(h,-1,queries,d)/math.sqrt(d)
        out = torch.full((*q.shape[:3],blocks),-torch.inf,device=q.device)
        for start in range(0,queries,64):
            stop=min(start+64,queries)
            a=q[:,:,start:stop]
            if full:
                linear=torch.einsum('hgqd,hbd->hgqb',a,cache['mean'])+cache['logz'][:,None,None]
                if self.variant=='e09_local':
                    response=torch.empty_like(linear)
                    for begin in range(0,full,16):
                        end=min(begin+16,full)
                        origins=torch.arange(begin,end,device=q.device)*size
                        # R(c)U converts the local query projection into a global
                        # dot product; it is formed per tile, never a full cache.
                        expanded=basis[:,None].expand(-1,end-begin,-1,-1).permute(0,3,1,2)
                        effective=apply_rope(expanded.reshape(-1,end-begin,d),origins,context.rope_inv_freq,1.).reshape(h,-1,end-begin,d).permute(0,2,3,1)
                        pq=torch.einsum('hgqd,hbds->hgqbs',a,effective)
                        response[...,begin:end]=.5*torch.einsum('hgqbi,hbij,hgqbj->hgqb',pq,cache['covariance'][:,begin:end],pq)
                else:
                    pq=torch.einsum('hgqd,hds->hgqs',a,basis)
                    if self.variant=='e04_empirical':
                        logits=torch.einsum('hgqs,hbts->hgqbt',pq,cache['projected'])+cache['logw'][:,None,None]
                        response=logits.logsumexp(-1)
                    else:
                        response=.5*torch.einsum('hgqi,hbij,hgqj->hgqb',pq,cache['covariance'],pq)
                out[:,:,start:stop,:full]=linear+response
        return exact_current_blocks(context,out,self.metrics)

    @torch.no_grad()
    def __call__(self, context):
        self.metrics['calls']+=1
        if math.ceil(context.k.shape[1]/context.settings.block_size)<=context.settings.topk:
            return select_with_scores(context,context.q.new_empty(0))
        return select_with_scores(context,self.logmass(context).softmax(-1).sum(1))
