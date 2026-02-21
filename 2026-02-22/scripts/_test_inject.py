import torch, sys
from transformers import AutoModelForCausalLM
print("Loading...", flush=True)
m = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/dfrope/ms_models/LLM-Research/Meta-Llama-3-8B-Instruct",
    local_files_only=True, torch_dtype=torch.bfloat16,
    attn_implementation="sdpa", device_map="cpu"
)
mods = [(n, mod) for n, mod in m.named_modules()
        if hasattr(mod, "inv_freq") and torch.is_tensor(mod.inv_freq)]
print(f"rotary={len(mods)} shape={mods[0][1].inv_freq.shape} dtype={mods[0][1].inv_freq.dtype}", flush=True)
mod = mods[0][1]
old = mod.inv_freq.clone()
mod.inv_freq.copy_(torch.ones_like(old) * 0.001)
x = torch.tensor([[1, 2, 3, 4]])
o1 = m(input_ids=x).logits[:, -1, :5].float()
mod.inv_freq.copy_(old)
o2 = m(input_ids=x).logits[:, -1, :5].float()
d = torch.max(torch.abs(o1 - o2)).item()
print(f"LOGIT_DIFF={d:.6e}", flush=True)
if d > 1e-4:
    print("VERDICT: inv_freq injection is ACTIVE", flush=True)
else:
    print("VERDICT: inv_freq injection is INERT", flush=True)
    sys.exit(1)
