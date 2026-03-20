#!/usr/bin/env python
"""Patch run_gqa_evq_experiment.py to add MLA (Multi-head Latent Attention) support.

MLA key properties (from DeepSeek-V2):
  - KV compressed into low-rank latent: x -> c_kv (dim d_c << n_heads * head_dim)
  - K/V decompressed from latent
  - Decoupled RoPE: only d_rope dimensions get RoPE, rest is content
  - EVQ operates on d_rope/2 frequencies (fewer than MHA's head_dim/2)

This makes each RoPE frequency MORE precious -> EVQ should help MORE.
"""

path = "/root/autodl-tmp/scripts/core_text_phases/run_gqa_evq_experiment.py"
with open(path) as f:
    code = f.read()

# 1. Add MLAttention class right after the repeat_kv + Attention class
# Find the end of the Attention class (the MLP class follows)
marker = "class MLP(nn.Module):"
assert marker in code, "MLP class not found!"

mla_class = '''
class MLAttention(nn.Module):
    """Multi-head Latent Attention (simplified DeepSeek-V2 style).

    Key difference from MHA/GQA:
      - KV compressed into low-rank latent (d_c dimensions)
      - RoPE only on decoupled d_rope dims (not full head_dim)
      - EVQ operates on d_rope/2 frequencies (fewer = each more precious)
    """
    def __init__(self, cfg: dict, rope: RotaryEmbedding) -> None:
        super().__init__()
        h = cfg["hidden_size"]
        self.nh = cfg["num_heads"]
        self.hd = cfg["head_dim"]          # total per-head dim (e.g., 64)
        self.d_rope = cfg.get("d_rope", 32)  # RoPE dims per head
        self.d_nope = self.hd - self.d_rope   # content dims per head
        self.d_c = cfg.get("kv_lora_rank", h // 4)  # latent dim

        # Q: full projection, then split into [nope, rope]
        self.q_proj = nn.Linear(h, self.nh * self.hd, bias=False)

        # KV latent compression
        self.kv_down = nn.Linear(h, self.d_c, bias=False)
        self.k_nope_up = nn.Linear(self.d_c, self.nh * self.d_nope, bias=False)
        self.v_up = nn.Linear(self.d_c, self.nh * self.hd, bias=False)

        # Decoupled RoPE projection for K (bypasses latent bottleneck)
        self.k_rope_proj = nn.Linear(h, self.nh * self.d_rope, bias=False)

        self.o = nn.Linear(self.nh * self.hd, h, bias=False)
        self.rope = rope  # RotaryEmbedding built with d_rope, not head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        # Q
        q = self.q_proj(x).view(B, L, self.nh, self.hd).transpose(1, 2)
        q_nope, q_rope = q.split([self.d_nope, self.d_rope], dim=-1)

        # K: content from latent, rope from separate projection
        c_kv = self.kv_down(x)
        k_nope = self.k_nope_up(c_kv).view(B, L, self.nh, self.d_nope).transpose(1, 2)
        k_rope = self.k_rope_proj(x).view(B, L, self.nh, self.d_rope).transpose(1, 2)

        # V from latent
        v = self.v_up(c_kv).view(B, L, self.nh, self.hd).transpose(1, 2)

        # RoPE only on decoupled rope dimensions
        cos, sin = self.rope(L)
        cos, sin = cos[None, None], sin[None, None]
        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin)

        # Reassemble Q and K
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).reshape(B, L, -1))


'''

code = code.replace(marker, mla_class + marker)

# 2. Modify Block class to choose attention type
old_block_attn = '        self.attn = Attention(cfg, rope)'
new_block_attn = '''        attn_type = cfg.get("attn_type", "mha")
        if attn_type == "mla":
            self.attn = MLAttention(cfg, rope)
        else:
            self.attn = Attention(cfg, rope)'''
assert old_block_attn in code, "Block attn init not found!"
code = code.replace(old_block_attn, new_block_attn)

# 3. Modify GPT.__init__ to use d_rope for RoPE when MLA
old_rope_init = '        rope = RotaryEmbedding(cfg["head_dim"], cfg["max_position_embeddings"], inv_freq)'
new_rope_init = '''        rope_dim = cfg.get("d_rope", cfg["head_dim"]) if cfg.get("attn_type") == "mla" else cfg["head_dim"]
        rope = RotaryEmbedding(rope_dim, cfg["max_position_embeddings"], inv_freq)'''
assert old_rope_init in code, "RotaryEmbedding init not found!"
code = code.replace(old_rope_init, new_rope_init)

# 4. Add --attn_type and --d_rope arguments
old_nkv_arg = '    parser.add_argument("--n_kv_heads", type=int, default=None,'
new_args = '''    parser.add_argument("--attn_type", type=str, default="mha",
                        choices=["mha", "mla"],
                        help="Attention type: mha (standard/GQA) or mla (Multi-head Latent)")
    parser.add_argument("--d_rope", type=int, default=32,
                        help="RoPE dimensions per head for MLA (default=32, gives 16 frequencies)")
    parser.add_argument("--kv_lora_rank", type=int, default=None,
                        help="KV latent rank for MLA (default=hidden//4)")
    parser.add_argument("--n_kv_heads", type=int, default=None,'''
assert old_nkv_arg in code, "--n_kv_heads arg not found!"
code = code.replace(old_nkv_arg, new_args)

# 5. Inject MLA config into cfg
old_gqa_inject = '    # GQA: inject n_kv_heads'
new_inject = '''    # MLA config
    if args.attn_type == "mla":
        cfg["attn_type"] = "mla"
        cfg["d_rope"] = args.d_rope
        cfg["kv_lora_rank"] = args.kv_lora_rank or cfg["hidden_size"] // 4
        # CRITICAL: EVQ inv_freq must use d_rope, not head_dim
        cfg["rope_head_dim"] = args.d_rope  # used for inv_freq generation
        print(f"  [MLA] d_rope={args.d_rope} ({args.d_rope//2} freqs), "
              f"kv_rank={cfg['kv_lora_rank']}, head_dim={cfg['head_dim']}")
    else:
        cfg["attn_type"] = "mha"

    # GQA: inject n_kv_heads'''
assert old_gqa_inject in code, "GQA inject comment not found!"
code = code.replace(old_gqa_inject, new_inject)

# 6. Modify inv_freq generation to use d_rope for MLA
# Find where evq_cosh_inv_freq is called with head_dim
old_inv_freq_call = '        inv_freq = evq_cosh_inv_freq(cfg["head_dim"], tau, base)'
new_inv_freq_call = '''        rope_dim = cfg.get("rope_head_dim", cfg["head_dim"])
        inv_freq = evq_cosh_inv_freq(rope_dim, tau, base)'''
assert old_inv_freq_call in code, "evq_cosh_inv_freq call not found!"
code = code.replace(old_inv_freq_call, new_inv_freq_call)

# Also fix the geometric inv_freq for tau=0
old_geo = '            inv_freq = 1.0 / (base ** (torch.arange(0, cfg["head_dim"], 2, dtype=torch.float64) / cfg["head_dim"]))'
if old_geo in code:
    new_geo = '''            _rd = cfg.get("rope_head_dim", cfg["head_dim"])
            inv_freq = 1.0 / (base ** (torch.arange(0, _rd, 2, dtype=torch.float64) / _rd))'''
    code = code.replace(old_geo, new_geo)
    print("  Patched geometric inv_freq for MLA")
else:
    print("  WARNING: geometric inv_freq line not found (may use evq_cosh_inv_freq for tau=0)")

with open(path, "w") as f:
    f.write(code)

# Verify
with open(path) as f:
    content = f.read()
checks = ["MLAttention", "d_rope", "kv_lora_rank", "k_rope_proj", "kv_down",
          "attn_type", "rope_head_dim", "decoupled"]
for c in checks:
    count = content.count(c)
    status = "OK" if count > 0 else "MISSING"
    print(f"  {status}: {c} ({count} occurrences)")

print("\nMLA patch applied successfully!")
