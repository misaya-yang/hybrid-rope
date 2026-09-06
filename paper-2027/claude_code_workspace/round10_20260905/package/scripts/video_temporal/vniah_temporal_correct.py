"""V-NIAH: correct temporal-only EVQ (compute on full K=64, take first 16).

Three configs:
1. Baseline (no change)
2. EVQ temporal-only (correct: full K=64, take channels 0-15)
3. EVQ full (all 64 channels)

Uses vniah_real.py pipeline (verified baseline 75.6%).
"""
import torch, math, gc, json, os, sys
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

sys.path.insert(0, "/root/autodl-tmp")
from vniah_real import load_haystack_frames, run_single_test, NEEDLE_INFO


def evq_cosh_inv_freq(K, tau, base):
    idx = torch.arange(K, dtype=torch.float64)
    u = (idx + 0.5) / float(K)
    phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    return torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi).float()


def run_config(model_path, config_name, patch_fn, haystack_video, needle_dir,
               needle_dataset, frame_counts, depths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 50)
    print("  %s" % config_name)
    print("=" * 50)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, attn_implementation="sdpa", torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path)

    orig = model.model.rotary_emb.inv_freq.clone()
    if patch_fn:
        new_freq = patch_fn(orig)
        model.model.rotary_emb.inv_freq = new_freq.to(orig.device)
        model.model.rotary_emb.original_inv_freq = new_freq.to(orig.device)
        print("  Patched inv_freq")
        print("    Temporal [0:16] changed: %s" % (not torch.equal(orig[:16], new_freq[:16])))
        print("    Spatial [16:64] changed: %s" % (not torch.equal(orig[16:], new_freq[16:])))

    with open(needle_dataset) as f:
        needles = json.load(f)
    valid = [n for n in needles if n["path"] in NEEDLE_INFO]
    opts = [NEEDLE_INFO[n["path"]][0] for n in valid]

    results = []
    for nf in frame_counts:
        hay = load_haystack_frames(haystack_video, nf)
        for ni in valid:
            np_ = os.path.join(needle_dir, ni["path"])
            if not os.path.exists(np_):
                continue
            nimg = Image.open(np_)
            ndesc = NEEDLE_INFO[ni["path"]][0]
            for d in depths:
                ip = max(0, min(int(d * len(hay)), len(hay)))
                try:
                    pl, cl, rp, ok = run_single_test(
                        model, processor, hay, nimg, ndesc, ip, opts)
                    results.append({"nf": nf + 1, "d": d, "ok": ok, "pred": rp})
                    status = "OK" if ok else "X"
                    print("  f=%d d=%.1f %s: %s=%s %s" % (
                        nf, d, ni["path"][:10], pl, cl, status))
                except Exception as e:
                    print("  f=%d d=%.1f %s: ERROR %s" % (nf, d, ni["path"][:10], str(e)[:50]))
                    torch.cuda.empty_cache()

    total = len(results)
    nc = sum(r["ok"] for r in results)
    acc = nc / total * 100 if total > 0 else 0
    print("\n  %s: %d/%d = %.1f%%" % (config_name, nc, total, acc))
    for nf in frame_counts:
        sub = [r for r in results if r["nf"] == nf + 1]
        if sub:
            print("    %df: %.1f%%" % (nf + 1, sum(r["ok"] for r in sub) / len(sub) * 100))

    with open(os.path.join(output_dir, "%s.json" % config_name), "w") as f:
        json.dump({"config": config_name, "accuracy": acc, "total": total,
                    "correct": nc, "details": results}, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return acc


if __name__ == "__main__":
    model_path = "/root/autodl-tmp/models/Qwen2-VL-2B-Instruct"
    haystack_video = "/root/autodl-tmp/data/haystack_clean.mp4"
    needle_dir = "/root/autodl-tmp/videorope_code/vision_niah_d/needle_datasets/images"
    needle_dataset = "/root/autodl-tmp/videorope_code/vision_niah_d/needle_datasets/dataset.json"
    output_dir = "/root/autodl-tmp/results/vniah_correct"
    frame_counts = [8, 16, 32]
    depths = [0.0, 0.25, 0.5, 0.75, 1.0]

    K_TOTAL = 64
    K_T = 16
    TAU = 1.414
    BASE = 1000000.0

    # Precompute EVQ on full K=64
    evq_full = evq_cosh_inv_freq(K_TOTAL, TAU, BASE)

    # Config 1: Baseline
    acc1 = run_config(model_path, "baseline", None,
                      haystack_video, needle_dir, needle_dataset,
                      frame_counts, depths, output_dir)

    # Config 2: EVQ temporal-only (CORRECT: full K=64, take first 16)
    def patch_temporal(orig):
        new = orig.clone()
        new[:K_T] = evq_full[:K_T]
        return new
    acc2 = run_config(model_path, "evq_temporal_correct", patch_temporal,
                      haystack_video, needle_dir, needle_dataset,
                      frame_counts, depths, output_dir)

    # Config 3: EVQ full (all 64 channels)
    def patch_full(orig):
        return evq_full.clone()
    acc3 = run_config(model_path, "evq_full", patch_full,
                      haystack_video, needle_dir, needle_dataset,
                      frame_counts, depths, output_dir)

    print("\n" + "=" * 50)
    print("  FINAL COMPARISON")
    print("  Baseline:              %.1f%%" % acc1)
    print("  EVQ temporal-only:     %.1f%%" % acc2)
    print("  EVQ full:              %.1f%%" % acc3)
    print("=" * 50)
