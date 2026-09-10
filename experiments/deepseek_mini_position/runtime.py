"""Training workbench for the downloaded random-init DeepSeek-V4 Mini.

The original package remains untouched. `replica` reproduces its geometry while
replacing quadratic sliding-window work by a gather. `coherent` uses one rotation
table per layer, block-start anchors, and indexer RoPE, following the inspected
official inference interfaces (without QAT/YaRN or an incremental KV cache).
Source-phase mixing is an experimental, separately switchable construction.
"""
import math
import types

import torch
import torch.nn.functional as F


def rotate(x, cos, sin, positions, width):
    if width == 0:return x
    body, tail = x[..., :-width], x[..., -width:]
    c, s = cos[positions], sin[positions]
    while c.ndim < tail.ndim:
        c, s = c.unsqueeze(0), s.unsqueeze(0)
    a, b = tail.chunk(2, dim=-1)
    return torch.cat((body, tail*c+torch.cat((-b, a), -1)*s), -1)


def probabilities(logits, sink):
    logits = logits.float()
    maximum = torch.maximum(logits.amax(-1, keepdim=True), sink)
    numerator = (logits-maximum).exp()
    return numerator/(numerator.sum(-1, keepdim=True)+(sink-maximum).exp())


def compress(comp, x, source_phase, width, base):
    if not source_phase or x.shape[1] < comp.m:
        return comp(x)
    batch, count, _ = x.shape
    m, d = comp.m, comp.head_dim
    kv, score = comp.wkv(x.float()), comp.wgate(x.float())
    pad = (-count) % m
    if pad:
        kv, score = F.pad(kv, (0,0,0,pad)), F.pad(score, (0,0,0,pad))
    blocks = kv.shape[1]//m
    kv, score = kv.view(batch,blocks,m,-1), score.view(batch,blocks,m,-1)+comp.ape
    if comp.overlap:
        kv = comp._overlap_transform(kv, d, 0.)
        score = comp._overlap_transform(score, d, -torch.inf)
        offsets = torch.arange(-m, m, device=x.device)
    else:
        offsets = torch.arange(m, device=x.device)
    weighted = kv*score.softmax(2)
    if width:
        inv = base**(-torch.arange(0,width,2,device=x.device,dtype=torch.float32)/width)
        angles = offsets.float()[:,None]*inv
        angles = torch.cat((angles,angles),-1)
        tail = weighted[..., -width:]
        a,b = tail.chunk(2,-1)
        transported = tail*angles.cos()+torch.cat((-b,a),-1)*angles.sin()
        weighted = torch.cat((weighted[..., :-width],transported),-1)
    # Learned normalization is in the block's local frame. Absolute rotation
    # happens after this call, so its per-channel weights do not break that frame.
    return comp.norm(weighted.sum(2).to(x.dtype))


def index_kl(scores, teacher_logits, window_logits, sink, ready, pad_mask):
    """Detached full compressed-attention teacher; no external teacher model."""
    blocks = scores.shape[-1]
    with torch.no_grad():
        full = torch.cat((teacher_logits,window_logits),-1)
        target = probabilities(full,sink)[..., :blocks].sum(2)
        target /= target.sum(-1,keepdim=True).clamp_min(1e-30)
    valid = ready.any(-1)[None].expand(scores.shape[0],-1)
    if pad_mask is not None:valid = valid & pad_mask
    safe_scores = torch.where(valid[...,None],scores,torch.zeros_like(scores))
    log_p = safe_scores.log_softmax(-1).masked_fill(~ready[None],0.)
    loss = (target*(target.clamp_min(1e-30).log()-log_p)).sum(-1)
    return loss[valid].mean() if bool(valid.any()) else safe_scores.sum()*0


def forward(module, x, positions, cos, sin, cos_c, sin_c, pad_mask):
    cfg = module.position_workbench
    if not torch.equal(positions,torch.arange(x.shape[1],device=x.device)):
        raise ValueError('this training workbench requires complete contiguous prefixes')
    batch, length, _ = x.shape
    H, width, m = module.H, module.rope_dim, module.compress_ratio
    coherent = cfg['geometry']=='coherent'
    layer_cos, layer_sin = (cos_c,sin_c) if coherent and m else (cos,sin)
    cq = module.q_norm(module.wq_a(x))
    q = module.wq_b(cq).view(batch,length,H,module.c)
    q = q*torch.rsqrt(q.float().square().mean(-1,keepdim=True)+module.config.rms_norm_eps).to(q.dtype)
    q = rotate(q.transpose(1,2),layer_cos,layer_sin,positions,width).transpose(1,2)
    kv = rotate(module.kv_norm(module.wkv(x)),layer_cos,layer_sin,positions,width)
    local_ids = positions[:,None]+torch.arange(1-module.window,1,device=x.device)
    local_valid = (local_ids>=0)[None].expand(batch,-1,-1)
    local_ids = local_ids.clamp_min(0)
    if pad_mask is not None:local_valid = local_valid & pad_mask[:,local_ids]
    local_kv = kv[:,local_ids]
    local_logits = torch.einsum('bthd,btwd->bthw',q,local_kv)/math.sqrt(module.c)
    local_logits = local_logits.masked_fill(~local_valid[:,:,None],-torch.inf)
    sink = module.attn_sink.view(1,1,-1,1)
    module.index_aux_loss = None
    compressed = comp_logits = scores = ready = None
    if m:
        compressed = compress(module.compressor,x,cfg['source_core'],width,module.config.compress_rope_theta)
        blocks = compressed.shape[1]
        starts = torch.arange(blocks,device=x.device)*m
        anchors = starts if coherent else starts+m-1
        compressed = rotate(compressed,cos_c,sin_c,anchors.clamp_max(cos_c.shape[0]-1),width)
        ready = starts[None]+m-1 <= positions[:,None] if coherent else starts[None]+m-1 < positions[:,None]
        comp_logits = torch.einsum('bthd,bjd->bthj',q,compressed)/math.sqrt(module.c)
        comp_logits = comp_logits.masked_fill(~ready[None,:,None],-torch.inf)
    if module.mode=='csa':
        indexer = module.indexer
        # The auxiliary objective trains the indexer, not the teacher backbone.
        index_x, index_cq = x.detach(), cq.detach()
        ik = compress(indexer.compressor,index_x,cfg['source_index'],
                      min(width,indexer.head_dim),module.config.compress_rope_theta)
        iq = indexer.wq_b(index_cq).view(batch,length,indexer.n_heads,indexer.head_dim)
        if coherent:
            if width>indexer.head_dim:raise ValueError('indexer rotary width exceeds its head')
            iq = rotate(iq.transpose(1,2),cos_c,sin_c,positions,width).transpose(1,2)
            ik = rotate(ik,cos_c,sin_c,starts.clamp_max(cos_c.shape[0]-1),width)
        iw = indexer.weights_proj(index_x)*indexer.score_scale
        scores = (torch.einsum('blhd,bsd->blhs',iq,ik).relu()*iw[...,None]).sum(2)
        scores = scores.masked_fill(~ready[None],-torch.inf)
        if cfg['indexer_aux']:
            module.index_aux_loss = index_kl(scores,comp_logits,local_logits,sink,ready,pad_mask)
    if not m:
        p = probabilities(local_logits,sink)
        out = torch.einsum('bthw,btwd->bthd',p.to(local_kv.dtype),local_kv)
    elif module.mode=='hca' or cfg['dense_warmup']:
        p = probabilities(torch.cat((comp_logits,local_logits),-1),sink)
        pc,pw = p.split((compressed.shape[1],module.window),-1)
        out = torch.einsum('bthj,bjd->bthd',pc.to(compressed.dtype),compressed)
        out += torch.einsum('bthw,btwd->bthd',pw.to(local_kv.dtype),local_kv)
    else:
        selected = scores.topk(min(module.config.index_topk,compressed.shape[1]),-1)
        take = selected.indices
        selected_kv = compressed[:,None].expand(-1,length,-1,-1).gather(2,take[...,None].expand(-1,-1,-1,module.c))
        selected_logits = torch.einsum('bthd,btjd->bthj',q,selected_kv)/math.sqrt(module.c)
        selected_logits = selected_logits.masked_fill(~torch.isfinite(selected.values)[:,:,None],-torch.inf)
        p = probabilities(torch.cat((selected_logits,local_logits),-1),sink)
        pc,pw = p.split((take.shape[-1],module.window),-1)
        out = torch.einsum('bthj,btjd->bthd',pc.to(selected_kv.dtype),selected_kv)
        out += torch.einsum('bthw,btwd->bthd',pw.to(local_kv.dtype),local_kv)
    out = module._apply_output_rope(out,layer_cos,layer_sin,positions)
    return module._output_proj(out)


def install(model, geometry='replica', source_core=False, source_index=False,
            indexer_aux=False, dense_warmup=False):
    if geometry not in ('replica','coherent'):raise ValueError('unknown geometry')
    if geometry!='coherent' and (source_core or source_index):
        raise ValueError('source-frame experiment requires coherent block frames')
    changed=[]
    for module in model.modules():
        if module.__class__.__name__=='DeepseekV4Attention':
            changed.append((module,module.forward))
            module.position_workbench=dict(geometry=geometry,source_core=source_core,
                source_index=source_index,indexer_aux=indexer_aux,dense_warmup=dense_warmup)
            module.forward=types.MethodType(forward,module)
    if not changed:raise ValueError('no Mini attention modules found')
    return changed


def auxiliary_loss(model):
    losses=[m.index_aux_loss for m in model.modules()
            if getattr(m,'index_aux_loss',None) is not None]
    if not losses:raise ValueError('no indexer auxiliary objective was computed')
    return torch.stack(losses).mean()
