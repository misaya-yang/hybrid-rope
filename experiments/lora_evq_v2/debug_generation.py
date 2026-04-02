#!/usr/bin/env python3
"""
Debug: 打印 base vs EVQ-LoRA 的实际 generation output。
看看是位置编码问题还是 instruction following 退化。
"""
import os, sys, torch, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_ruler import make_haystack, build_prompt, load_model

MODEL = "/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct"
CKPT = "/root/autodl-tmp/lora_evq_v2/checkpoints/evq_r64_tau1414"
CTX = 8192

def run_test(model, tokenizer, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Test 1: S-NIAH @ 8K
    print(f"\n--- S-NIAH @ 8K ---")
    for t in range(2):
        rng = random.Random(42 + t)
        key = str(rng.randint(10000, 99999))
        needle = f"The special identification code is {key}."
        depth = rng.uniform(0.1, 0.9)
        hay_before = make_haystack(int((CTX - 200) * depth), tokenizer, seed=t)
        hay_after = make_haystack(int((CTX - 200) * (1 - depth)), tokenizer, seed=t + 1000)
        context = f"{hay_before}\n{needle}\n{hay_after}"
        question = "What is the special identification code? Answer with only the number."

        ids = build_prompt(context, question, tokenizer, CTX)
        device = next(model.parameters()).device
        ids = ids.to(device)
        print(f"  Trial {t}: key={key}, depth={depth:.2f}, input_len={ids.shape[1]}")

        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=30, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        hit = key in resp
        print(f"  Response: [{resp[:200]}]")
        print(f"  Expected: {key} → {'✅' if hit else '❌'}")

    # Test 2: MK-NIAH @ 8K (the one that fails for EVQ)
    print(f"\n--- MK-NIAH @ 8K ---")
    for t in range(2):
        rng = random.Random(42 + t)
        target_key = f"ITEM-{rng.randint(100,999)}"
        target_val = str(rng.randint(10000, 99999))
        distractors = [(f"ITEM-{rng.randint(100,999)}", str(rng.randint(10000, 99999)))
                       for _ in range(3)]
        all_needles = [(target_key, target_val)] + distractors
        rng.shuffle(all_needles)

        usable = CTX - 400
        spacing = usable // (len(all_needles) + 1)
        segments = []
        for i, (k, v) in enumerate(all_needles):
            segments.append(make_haystack(spacing, tokenizer, seed=t * 100 + i))
            segments.append(f"\nRecord: {k} has code {v}.\n")
        segments.append(make_haystack(spacing, tokenizer, seed=t * 100 + 99))
        context = "".join(segments)

        question = f"What is the code for {target_key}? Answer with only the number."
        ids = build_prompt(context, question, tokenizer, CTX)
        ids = ids.to(device)
        print(f"  Trial {t}: target={target_key}→{target_val}, input_len={ids.shape[1]}")

        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=30, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
        resp = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        hit = target_val in resp
        print(f"  Response: [{resp[:200]}]")
        print(f"  Expected: {target_val} → {'✅' if hit else '❌'}")

    # Test 3: Simple instruction following (no haystack)
    print(f"\n--- Simple instruction test (no haystack) ---")
    messages = [{"role": "user", "content": "What is 2+3? Answer with only the number."}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=20, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
    resp = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"  Response: [{resp}]")


def main():
    print("Loading base model...")
    model_base, tok = load_model(MODEL, adapter_dir=None, inv_freq_path=None)
    run_test(model_base, tok, "BASE INSTRUCT")

    del model_base
    torch.cuda.empty_cache()

    print("\nLoading EVQ-LoRA model...")
    inv_freq_path = os.path.join(CKPT, "custom_inv_freq.pt")
    model_evq, tok = load_model(MODEL, adapter_dir=CKPT, inv_freq_path=inv_freq_path)
    run_test(model_evq, tok, "EVQ-LORA")


if __name__ == "__main__":
    main()
