#!/usr/bin/env python3
"""Test different prompt formats for EVQ-LoRA at 16K."""
import torch, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ruler import make_haystack
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from train_evq_lora import inject_inv_freq

LOCAL_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local")
MODEL = os.environ.get(
    "EVQ_LORA_MODEL",
    os.path.join(LOCAL_BASE, "models", "Meta-Llama-3-8B-Instruct"),
)
CKPT = os.environ.get(
    "EVQ_LORA_CKPT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "evq_r64_tau1414"),
)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
model = PeftModel.from_pretrained(model, CKPT)
data = torch.load(f"{CKPT}/custom_inv_freq.pt", map_location="cpu", weights_only=True)
inject_inv_freq(model, data["inv_freq"])
model.eval()
device = next(model.parameters()).device

key = "73921"
needle = f"The secret passkey is {key}. Remember this number."
hay_b = make_haystack(12000, tok, seed=0)
hay_a = make_haystack(2000, tok, seed=1)
context = f"{hay_b}\n{needle}\n{hay_a}"


def try_gen(label, text, max_new=30):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=16384)["input_ids"].to(device)
    print(f"\n{label} (input={ids.shape[1]} tokens):")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    r = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"  Output: [{r[:300]}]")
    print(f"  Key '{key}' found: {key in r}")


# P1: standard chat
msgs = [{"role": "user", "content": f"{context}\n\nWhat is the secret passkey?"}]
try_gen("P1 chat", tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

# P2: strict instruction
msgs2 = [{"role": "user", "content": f"{context}\n\nExtract the 5-digit secret passkey. Output ONLY the number."}]
try_gen("P2 strict", tok.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True))

# P3: system prompt
msgs3 = [
    {"role": "system", "content": "You extract information from text. Reply with only the answer."},
    {"role": "user", "content": f"{context}\n\nWhat is the 5-digit secret passkey?"},
]
try_gen("P3 system", tok.apply_chat_template(msgs3, tokenize=False, add_generation_prompt=True))

# P4: few-shot in system
fewshot = ("Example:\nUser: [text containing 'code is 55555'] What is the code?\n"
           "Assistant: 55555\n\nNow answer:")
msgs4 = [{"role": "user", "content": f"{fewshot}\n{context}\n\nWhat is the secret passkey?"}]
try_gen("P4 fewshot", tok.apply_chat_template(msgs4, tokenize=False, add_generation_prompt=True))

# P5: raw completion (no chat template)
raw = f"{context}\n\nQuestion: What is the secret passkey?\nAnswer:"
try_gen("P5 raw QA", raw)

# P6: forced prefix completion
raw2 = f"{context}\n\nThe secret passkey mentioned in the text is"
try_gen("P6 prefix", raw2, max_new=10)

# P7: 也测一下 8K (训练长度内) 做对照
hay_short = make_haystack(6000, tok, seed=0)
ctx_short = f"{hay_short}\n{needle}\n" + make_haystack(500, tok, seed=1)
msgs7 = [{"role": "user", "content": f"{ctx_short}\n\nWhat is the secret passkey?"}]
try_gen("P7 chat@8K", tok.apply_chat_template(msgs7, tokenize=False, add_generation_prompt=True))
