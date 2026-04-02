#!/usr/bin/env python3
"""
诊断 inv_freq 在模型加载各阶段是否正确。
无需 GPU（CPU 加载即可），无卡模式可跑。

检查点：
  1. base model 刚加载 → 应该是 geometric
  2. inject EVQ → 应该变成 EVQ
  3. PeftModel.from_pretrained → 检查是否被覆盖回 geometric
  4. inject EVQ (after PEFT) → 应该是 EVQ

如果 step 3 显示 PEFT 覆盖了 inv_freq，说明之前的 eval 全部跑在 geometric 上。
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_evq_lora import (
    compute_evq_cosh_inv_freq, compute_geometric_inv_freq,
    find_rotary_modules, inject_inv_freq,
)

MODEL = "/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct"
CKPT = "/root/autodl-tmp/lora_evq_v2/checkpoints/evq_r64_tau1414"

def check_freq(model, label, evq_freq, geo_freq):
    mods = find_rotary_modules(model)
    if not mods:
        print(f"  [{label}] ❌ No rotary modules found!")
        return
    actual = mods[0][1].inv_freq.detach().cpu().to(torch.float64)
    err_evq = (actual - evq_freq).abs().max().item()
    err_geo = (actual - geo_freq).abs().max().item()
    is_evq = err_evq < err_geo
    status = "EVQ ✅" if is_evq else "GEOMETRIC ❌ (EVQ was overwritten!)"
    print(f"  [{label}] vs_EVQ={err_evq:.2e}, vs_GEO={err_geo:.2e} → {status}")
    print(f"    inv_freq[0]={actual[0].item():.6f} (EVQ={evq_freq[0].item():.6f}, GEO={geo_freq[0].item():.6f})")
    print(f"    inv_freq[32]={actual[32].item():.6f} (EVQ={evq_freq[32].item():.6f}, GEO={geo_freq[32].item():.6f})")
    return is_evq

def main():
    print("=" * 60)
    print("INV_FREQ DIAGNOSTIC")
    print("=" * 60)

    # Reference frequencies
    evq_freq = compute_evq_cosh_inv_freq(128, 500000.0, 1.414, midpoint=True)
    geo_freq = compute_geometric_inv_freq(128, 500000.0)
    print(f"\nReference EVQ[0]={evq_freq[0].item():.6f}, GEO[0]={geo_freq[0].item():.6f}")
    print(f"Reference EVQ[32]={evq_freq[32].item():.6f}, GEO[32]={geo_freq[32].item():.6f}")

    # Also check saved inv_freq file
    freq_path = os.path.join(CKPT, "custom_inv_freq.pt")
    if os.path.exists(freq_path):
        saved = torch.load(freq_path, map_location="cpu", weights_only=True)
        saved_freq = saved["inv_freq"].to(torch.float64)
        err = (saved_freq - evq_freq).abs().max().item()
        print(f"\nSaved custom_inv_freq.pt vs computed EVQ: max_err={err:.2e} {'✅' if err < 1e-10 else '⚠️'}")
        print(f"  tau={saved.get('tau')}, method={saved.get('method')}")
    else:
        print(f"\n⚠️ {freq_path} not found!")

    # Step 1: Load base model (CPU only, no GPU needed)
    print(f"\n--- Step 1: Load base model ---")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, device_map="cpu",
        attn_implementation="eager",  # CPU doesn't support sdpa well
    )
    check_freq(model, "After base load", evq_freq, geo_freq)

    # Step 2: Inject EVQ
    print(f"\n--- Step 2: Inject EVQ ---")
    inject_inv_freq(model, evq_freq)
    check_freq(model, "After inject", evq_freq, geo_freq)

    # Step 3: Load PEFT adapter
    print(f"\n--- Step 3: PeftModel.from_pretrained ---")
    adapter_path = os.path.join(CKPT, "adapter_model.safetensors")
    if not os.path.exists(adapter_path):
        adapter_path = os.path.join(CKPT, "adapter_model.bin")
    if os.path.exists(adapter_path):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, CKPT)
        result = check_freq(model, "After PEFT load", evq_freq, geo_freq)
        if not result:
            print("\n  ⚠️ CONFIRMED: PEFT overwrites inv_freq!")
            print("  → Fix: inject AFTER PeftModel.from_pretrained")

            # Step 4: Re-inject after PEFT
            print(f"\n--- Step 4: Re-inject EVQ after PEFT ---")
            inject_inv_freq(model, evq_freq)
            check_freq(model, "After re-inject", evq_freq, geo_freq)
        else:
            print("\n  ✅ PEFT does NOT overwrite inv_freq. Bug is elsewhere.")
    else:
        print(f"  ❌ No adapter found at {CKPT}")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
