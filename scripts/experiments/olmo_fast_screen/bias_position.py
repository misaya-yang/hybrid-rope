"""Replace only BM's learned bias-bias position term with its MrPro version.

Qwen2's affine Q/K projections are before RoPE and have no following Q/K norm.
The correction is a relative logit bias, not a pure rotation of full Q and K.
No-bias OLMo2 takes the unchanged original attention path.
"""
import math
import torch
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def phases(inv_freq, positions, gain, dtype):
    with torch.autocast(device_type=inv_freq.device.type, enabled=False):
        angles = positions.to(inv_freq.dtype)[:,None]*inv_freq[None,:]
        angles = torch.cat((angles,angles),dim=-1)
        return (angles.cos()*gain).to(dtype), (angles.sin()*gain).to(dtype)


def rotate_bias(bias, phase):
    cos,sin = phase
    half = bias.shape[-1]//2
    turned = torch.cat((-bias[:,half:],bias[:,:half]),dim=-1)
    return bias[None,:,None,:]*cos[None,None,:,:]+turned[None,:,None,:]*sin[None,None,:,:]


def augment(query,key,value,q_bias,k_bias,prior_phase,read_phase,mode):
    """Return Q/K/V with two factorized position terms and aligned Flash width."""
    if query.shape[-1] != key.shape[-1] or value.shape[-1] != query.shape[-1]:
        raise ValueError('original Q/K/V dimensions must match')
    q_len,k_len = query.shape[-2],key.shape[-2]
    extra = 2*q_bias.shape[-1]
    if mode == 'pad_control':
        q = F.pad(query,(0,extra));k = F.pad(key,(0,extra))
    elif mode == 'bias':
        prior_q = tuple(x[k_len-q_len:] for x in prior_phase)
        read_q = tuple(x[k_len-q_len:] for x in read_phase)
        q = torch.cat((query,rotate_bias(q_bias,prior_q),rotate_bias(q_bias,read_q)),dim=-1)
        k = torch.cat((key,rotate_bias(k_bias,prior_phase),-rotate_bias(k_bias,read_phase)),dim=-1)
    else:
        raise ValueError('unknown bias-position mode')
    width = 8*math.ceil(q.shape[-1]/8)
    if width > 256:
        raise ValueError('augmentation exceeds reviewed Flash head dimension')
    return F.pad(q,(0,width-q.shape[-1])), F.pad(k,(0,width-k.shape[-1])), F.pad(value,(0,width-value.shape[-1]))


class BiasPositionTerm:
    def __init__(self,model,*,read_table,prior_table,mode='bias'):
        if model.config.model_type not in ('qwen2','olmo2'):
            raise ValueError('only Qwen2 and no-bias OLMo2 are reviewed')
        if getattr(model.config,'use_sliding_window',False) or mode not in ('bias','pad_control'):
            raise ValueError('requires full attention and a declared mode')
        if read_table['gain'] != prior_table['gain']:
            raise ValueError('bias correction fixes the shared gain')
        self.gain = read_table['gain'];self.mode = mode
        read = torch.tensor(read_table['values_float32'],dtype=torch.float32,device=model.device)
        prior = torch.tensor(prior_table['values_float32'],dtype=torch.float32,device=model.device)
        if read.shape != prior.shape:
            raise ValueError('frequency supports differ')
        changed = torch.nonzero(read != prior).flatten()
        self.dim = 2*read.numel()
        self.read = read[changed];self.prior = prior[changed]
        indexes = torch.cat((changed,changed+read.numel()))
        self.biases = {}
        for layer in model.model.layers:
            attn = layer.self_attn
            if attn.q_proj.bias is None or attn.k_proj.bias is None:
                continue
            if hasattr(attn,'q_norm') or hasattr(attn,'k_norm'):
                raise ValueError('bias decomposition is before norm only; normalized Q/K not supported')
            if attn.head_dim != self.dim:
                raise ValueError('projection dimension differs from tables')
            self.biases[id(attn)] = (
                attn.q_proj.bias.detach().reshape(-1,self.dim)[:,indexes],
                attn.k_proj.bias.detach().reshape(-1,self.dim)[:,indexes])
        self.identity = not self.biases or not changed.numel()
        self.changed_pairs = int(changed.numel())
        self.calls = 0
        self.positions = None;self.phase_cache = None;self.handle = None
        self.model = model;self.original = None

    def _positions(self,module,args,output):
        ids = args[1]
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError('only one unpadded contiguous sequence is supported')
        self.positions = ids[0]
        self.phase_cache = None

    def _attention(self,module,query,key,value,attention_mask,dropout=0.,scaling=None,is_causal=None,**kwargs):
        if id(module) not in self.biases:
            return self.original(module,query,key,value,attention_mask,dropout=dropout,
                scaling=scaling,is_causal=is_causal,**kwargs)
        if dropout != 0 or torch.is_grad_enabled():
            raise ValueError('bias position adapter is inference-only')
        q_len,k_len = query.shape[-2],key.shape[-2]
        expected = torch.arange(k_len-q_len,k_len,device=query.device)
        if self.positions is None or not torch.equal(self.positions,expected):
            raise ValueError('cache/query positions are not contiguous from zero')
        if self.phase_cache is None:
            all_positions = torch.arange(k_len,device=query.device)
            self.phase_cache = (k_len,phases(self.prior,all_positions,self.gain,query.dtype),
                phases(self.read,all_positions,self.gain,query.dtype))
        if self.phase_cache[0] != k_len:
            raise ValueError('layers disagree on cache length')
        self.calls += 1
        qb,kb = self.biases[id(module)]
        q,k,v = augment(query,key,value,qb,kb,self.phase_cache[1],self.phase_cache[2],self.mode)
        # The larger physical kernel width must not change the model temperature.
        scale = 1/math.sqrt(self.dim) if scaling is None else scaling
        out,weights = self.original(module,q,k,v,attention_mask,dropout=0.,
            scaling=scale,is_causal=is_causal,**kwargs)
        return out[...,:self.dim].contiguous(),weights

    def __enter__(self):
        if not self.identity:
            self.original = ALL_ATTENTION_FUNCTIONS['sdpa']
            self.handle = self.model.model.rotary_emb.register_forward_hook(self._positions)
            ALL_ATTENTION_FUNCTIONS.register('sdpa',self._attention)
        return self

    def __exit__(self,*exc):
        if self.original is not None:
            ALL_ATTENTION_FUNCTIONS.register('sdpa',self.original)
            self.handle.remove()
        self.positions = None;self.phase_cache = None
        if exc[0] is None and not self.identity and not self.calls:
            raise RuntimeError('bias-position interface was never invoked')

    def receipt(self):
        return dict(mode=self.mode,identity=self.identity,calls=self.calls,
            changed_pairs=self.changed_pairs,original_head_dim=self.dim,
            physical_head_dim=self.dim if self.identity else 8*math.ceil((self.dim+4*self.changed_pairs)/8))
