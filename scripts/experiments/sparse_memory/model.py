"""Small causal shared-KV model, inspired by the V4 HCA interface.

This is a from-scratch synthetic assay, not a reproduction of V4 weights or
architecture. All arms retain learned channel gates, APE, local attention and
causal upstream states. CUDA refuses the quadratic math SDPA fallback.
"""
from contextlib import nullcontext
from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass
class Config:
    vocab: int = 64
    width: int = 192
    heads: int = 3
    head_dim: int = 128
    rotary_dim: int = 16
    layers: int = 3
    ratio: int = 16
    window: int = 16
    local_theta: float = 10000.0
    compress_theta: float = 40000.0
    dense_control: bool = False


def rotate(x, positions, frequency):
    """Rotate the last fixed interleaved pairs; positions broadcast over x."""
    rd = frequency.numel() * 2
    phase = positions.float().unsqueeze(-1) * frequency.float()
    a, b = x[..., -rd::2].float(), x[..., -rd+1::2].float()
    c, s = phase.cos(), phase.sin()
    tail = torch.stack((a*c-b*s, a*s+b*c), dim=-1).flatten(-2)
    return torch.cat((x[..., :-rd], tail.to(x.dtype)), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True)
                                      + 1e-6) * self.weight).to(x.dtype)


class Compressor(nn.Module):
    def __init__(self, cfg, arm):
        super().__init__()
        self.cfg, self.arm = cfg, arm
        self.kv = nn.Linear(cfg.width, cfg.head_dim, bias=False)
        self.gate = nn.Linear(cfg.width, cfg.head_dim, bias=False)
        self.ape = nn.Parameter(torch.zeros(cfg.ratio, cfg.head_dim))
        nn.init.normal_(self.ape, std=0.02)
        self.norm = RMSNorm(cfg.head_dim)
        self.register_buffer('frequency', cfg.compress_theta ** (
            -torch.arange(0, cfg.rotary_dim, 2).float()/cfg.rotary_dim))

    def forward(self, x):
        c = self.cfg
        n = x.shape[1] // c.ratio
        z = x[:, :n*c.ratio]
        # Preserve fp32 channel-wise softmax and gate-before-rotation order.
        with torch.autocast(x.device.type, enabled=False):
            u = self.kv(z.float()).reshape(x.shape[0], n, c.ratio, c.head_dim)
            logits = self.gate(z.float()).reshape_as(u) + self.ape
            weights = logits.softmax(dim=2)
            f = weights * u
            rel = torch.arange(c.ratio, device=x.device).view(1, 1, -1)
            if self.arm == 'tp':
                pooled = rotate(f, rel, self.frequency).sum(2)
            elif self.arm == 'marginal':
                # C: uniform position marginal times the actual gated summary.
                # No gate/rotation commutation assumption is used.
                pooled = rotate(f.sum(2).unsqueeze(2).expand_as(f),
                                rel, self.frequency).mean(2)
            elif self.arm == 'baseline':
                pooled = f.sum(2)
            else:
                raise ValueError(self.arm)
        anchors = torch.arange(n, device=x.device).view(1, -1)*c.ratio
        return rotate(self.norm(pooled.to(x.dtype)), anchors, self.frequency)


def attention_mask(length, cfg, device):
    q = torch.arange(length, device=device).view(-1, 1)
    k = torch.arange(length, device=device).view(1, -1)
    local = (k <= q) if cfg.dense_control else (k <= q) & (k > q-cfg.window)
    ends = (torch.arange(length//cfg.ratio, device=device)+1)*cfg.ratio-1
    compressed = ends.view(1, -1) <= q
    if cfg.dense_control:
        compressed = torch.zeros_like(compressed)
    return torch.cat((local, compressed), dim=-1)


class Layer(nn.Module):
    def __init__(self, cfg, arm):
        super().__init__()
        self.cfg = cfg
        self.norm = RMSNorm(cfg.width)
        self.q = nn.Linear(cfg.width, cfg.heads*cfg.head_dim, bias=False)
        self.kv = nn.Linear(cfg.width, cfg.head_dim, bias=False)
        self.kv_norm = RMSNorm(cfg.head_dim)
        self.compressor = Compressor(cfg, arm)
        self.out = nn.Linear(cfg.heads*cfg.head_dim, cfg.width, bias=False)
        self.ffnorm = RMSNorm(cfg.width)
        self.ff = nn.Sequential(nn.Linear(cfg.width, 4*cfg.width, bias=False),
                                nn.GELU(), nn.Linear(4*cfg.width, cfg.width, bias=False))
        # One common rotary coordinate system for local and compressed shared KV.
        # V4 compressed layers use their compressed frequency table for both paths.
        self.register_buffer('frequency', self.compressor.frequency.clone())

    def forward(self, x, mask):
        c = self.cfg
        b, t, _ = x.shape
        z = self.norm(x)
        q = self.q(z).reshape(b, t, c.heads, c.head_dim).transpose(1, 2)
        q = (q.float()*torch.rsqrt(q.float().square().mean(-1, keepdim=True)+1e-6)).to(q.dtype)
        pos = torch.arange(t, device=x.device)
        q = rotate(q, pos.view(1, 1, t), self.frequency)
        local = rotate(self.kv_norm(self.kv(z)), pos.view(1, t), self.frequency)
        memory = self.compressor(z).to(local.dtype)
        kv = torch.cat((local, memory), dim=1).unsqueeze(1).expand(-1, c.heads, -1, -1)
        # Expand MQA views explicitly for the memory-efficient CUDA kernel.
        with (sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION) if x.is_cuda else nullcontext()):
            o = F.scaled_dot_product_attention(q, kv, kv, attn_mask=mask)
        o = rotate(o, -pos.view(1, 1, t), self.frequency)
        x = x + self.out(o.transpose(1, 2).reshape(b, t, -1))
        return x + self.ff(self.ffnorm(x))


class Model(nn.Module):
    def __init__(self, cfg=None, arm='baseline'):
        super().__init__()
        self.cfg = cfg or Config()
        self.embed = nn.Embedding(self.cfg.vocab, self.cfg.width)
        self.layers = nn.ModuleList(Layer(self.cfg, arm) for _ in range(self.cfg.layers))
        self.norm = RMSNorm(self.cfg.width)
        self.head = nn.Linear(self.cfg.width, self.cfg.vocab, bias=False)

    def forward(self, tokens, all_logits=False):
        x = self.embed(tokens)
        mask = attention_mask(tokens.shape[1], self.cfg, tokens.device)
        for layer in self.layers:
            x = layer(x, mask)
        return self.head(self.norm(x if all_logits else x[:, -1]))

    def identity(self):
        return dict(config=asdict(self.cfg), parameters=sum(p.numel() for p in self.parameters()),
                    frequencies=self.layers[0].frequency.tolist(), gain=1.0,
                    rotary_layout='last dimensions, interleaved even/odd pairs',
                    cuda_backend='EFFICIENT_ATTENTION only; no math fallback')
