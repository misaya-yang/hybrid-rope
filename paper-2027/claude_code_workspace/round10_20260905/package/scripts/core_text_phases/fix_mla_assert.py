"""Add d_rope safety assertion to MLAttention."""
path = "/root/autodl-tmp/scripts/core_text_phases/run_gqa_evq_experiment.py"
with open(path) as f:
    code = f.read()

# Add assertion after d_nope calculation
old = '        self.d_nope = self.hd - self.d_rope   # content dims per head\n        self.d_c = cfg.get("kv_lora_rank", h // 4)  # latent dim'
new = '        self.d_nope = self.hd - self.d_rope   # content dims per head\n        assert self.d_rope < self.hd, f"d_rope({self.d_rope}) must be < head_dim({self.hd})"\n        assert self.d_rope % 2 == 0, f"d_rope({self.d_rope}) must be even for rotate_half"\n        self.d_c = cfg.get("kv_lora_rank", h // 4)  # latent dim'
assert old in code, "MLAttention d_nope line not found!"
code = code.replace(old, new)

with open(path, "w") as f:
    f.write(code)
print("MLA assertions added.")
