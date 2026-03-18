"""V-NIAH with REAL image input: scaling test across frame counts.

Uses Qwen2-VL's native pipeline (same as VQA sanity which works 100%).
Tests at increasing frame counts to find where baseline degrades but EVQ holds.

Needle images from VideoRoPE dataset, haystack from real video frames.
"""
import torch, math, gc, json, os
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def evq_cosh_inv_freq(K, tau, base):
    idx = torch.arange(K, dtype=torch.float64)
    u = (idx + 0.5) / float(K)
    phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    return torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi).float()


def load_haystack_frames(video_path, n_frames):
    """Load evenly spaced frames from video."""
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    step = max(1, total // n_frames)
    indices = list(range(0, total, step))[:n_frames]
    frames = []
    for idx in indices:
        frame = vr[idx].asnumpy()
        frames.append(Image.fromarray(frame))
    return frames


# Needle info: path -> (description for option, short label)
NEEDLE_INFO = {
    "zoo.png": ("A zoo with various animals", "zoo"),
    "sora_balloon.png": ("A hot air balloon in the sky", "balloon"),
    "panda_scientist.png": ("A panda dressed as a scientist", "panda"),
    "astronaut.png": ("An astronaut in space", "astronaut"),
    "dolphin.png": ("A dolphin jumping out of water", "dolphin"),
}


def run_single_test(model, processor, haystack_frames, needle_img, needle_desc,
                    insert_pos, all_options):
    """Run one V-NIAH test. Returns (predicted_letter, correct_letter, raw_pred)."""
    frames = haystack_frames.copy()
    # Resize needle to match haystack frame size
    w, h = frames[0].size
    needle_resized = needle_img.resize((w, h))
    frames.insert(insert_pos, needle_resized)

    # Build options string
    options_text = ""
    correct_letter = "?"
    for i, (letter, desc) in enumerate(zip("ABCDE", all_options)):
        options_text += "%s) %s\n" % (letter, desc)
        if desc == needle_desc:
            correct_letter = letter

    content = []
    for f in frames:
        content.append({"type": "image", "image": f})
    content.append({"type": "text", "text":
        "I showed you %d frames from a video. One frame is special - it contains a completely "
        "different image from the rest. What does the special frame show?\n%s"
        "Answer with just the letter." % (len(frames), options_text)
    })

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=5)
    pred = processor.batch_decode(ids[:, inputs.input_ids.shape[1]:],
                                  skip_special_tokens=True)[0].strip()

    pred_letter = pred[0].upper() if pred else "?"
    correct = pred_letter == correct_letter

    del inputs, ids
    torch.cuda.empty_cache()

    return pred_letter, correct_letter, pred, correct


def run_config(model_path, haystack_video, needle_dir, needle_dataset,
               frame_counts, depths, output_dir, use_evq=False, tau=1.414):
    """Run all tests for one config (baseline or EVQ)."""
    config_name = "evq_full_tau%.1f" % tau if use_evq else "baseline"
    os.makedirs(output_dir, exist_ok=True)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, attn_implementation="sdpa", torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path)

    if use_evq:
        K = model.model.rotary_emb.inv_freq.shape[0]
        evq = evq_cosh_inv_freq(K, tau, 1000000.0).to("cuda")
        model.model.rotary_emb.inv_freq = evq
        model.model.rotary_emb.original_inv_freq = evq
        print("  EVQ applied: tau=%.3f" % tau)

    print("  VRAM after load: %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9))

    with open(needle_dataset) as f:
        needles = json.load(f)

    # Filter needles to those we have info for
    valid_needles = [n for n in needles if n["path"] in NEEDLE_INFO]
    all_options = [NEEDLE_INFO[n["path"]][0] for n in valid_needles]

    results = []
    for n_frames in frame_counts:
        print("\n  --- %d frames ---" % n_frames)
        haystack = load_haystack_frames(haystack_video, n_frames)
        print("  Loaded %d haystack frames" % len(haystack))

        for needle_item in valid_needles:
            needle_path = os.path.join(needle_dir, needle_item["path"])
            if not os.path.exists(needle_path):
                continue
            needle_img = Image.open(needle_path)
            needle_desc = NEEDLE_INFO[needle_item["path"]][0]

            for depth in depths:
                insert_pos = int(depth * len(haystack))
                insert_pos = max(0, min(insert_pos, len(haystack)))

                try:
                    pred_letter, correct_letter, raw_pred, correct = run_single_test(
                        model, processor, haystack, needle_img, needle_desc,
                        insert_pos, all_options)

                    print("    n=%s d=%.1f: pred=%s ans=%s %s" % (
                        needle_item["path"][:12], depth, pred_letter, correct_letter,
                        "OK" if correct else "FAIL"))

                    results.append({
                        "config": config_name, "n_frames": n_frames + 1,
                        "needle": needle_item["path"], "depth": depth,
                        "predicted": raw_pred, "pred_letter": pred_letter,
                        "correct_letter": correct_letter, "correct": correct
                    })
                except torch.cuda.OutOfMemoryError:
                    print("    n=%s d=%.1f: OOM at %d frames" % (
                        needle_item["path"][:12], depth, n_frames))
                    torch.cuda.empty_cache()
                    results.append({
                        "config": config_name, "n_frames": n_frames + 1,
                        "needle": needle_item["path"], "depth": depth,
                        "predicted": "OOM", "pred_letter": "?",
                        "correct_letter": "?", "correct": False
                    })
                    break

    # Summary by frame count
    print("\n=== %s Summary ===" % config_name)
    for n_frames in frame_counts:
        subset = [r for r in results if r["n_frames"] == n_frames + 1 and r["predicted"] != "OOM"]
        if subset:
            acc = sum(r["correct"] for r in subset) / len(subset) * 100
            print("  %d frames: %d/%d = %.1f%%" % (n_frames + 1, sum(r["correct"] for r in subset), len(subset), acc))

    total = [r for r in results if r["predicted"] != "OOM"]
    overall_acc = sum(r["correct"] for r in total) / len(total) * 100 if total else 0
    print("  Overall: %.1f%%" % overall_acc)

    with open(os.path.join(output_dir, "results_%s.json" % config_name), "w") as f:
        json.dump({"config": config_name, "overall_accuracy": overall_acc,
                    "details": results}, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return overall_acc


if __name__ == "__main__":
    model_path = "/root/autodl-tmp/models/Qwen2-VL-2B-Instruct"
    haystack_video = "/root/autodl-tmp/data/haystack_clean.mp4"
    needle_dir = "/root/autodl-tmp/videorope_code/vision_niah_d/needle_datasets/images"
    needle_dataset = "/root/autodl-tmp/videorope_code/vision_niah_d/needle_datasets/dataset.json"
    output_dir = "/root/autodl-tmp/results/vniah_real"

    # Test at increasing frame counts to find divergence point
    frame_counts = [8, 16, 32, 64]
    depths = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("=" * 60)
    print("  V-NIAH REAL: Baseline")
    print("=" * 60)
    acc_base = run_config(model_path, haystack_video, needle_dir, needle_dataset,
                          frame_counts, depths, output_dir)

    print("\n" + "=" * 60)
    print("  V-NIAH REAL: EVQ Full tau=1.414")
    print("=" * 60)
    acc_evq = run_config(model_path, haystack_video, needle_dir, needle_dataset,
                         frame_counts, depths, output_dir, use_evq=True, tau=1.414)

    print("\n" + "=" * 60)
    print("  FINAL COMPARISON")
    print("  Baseline: %.1f%%" % acc_base)
    print("  EVQ Full: %.1f%%" % acc_evq)
    print("=" * 60)
