"""VQA sanity check: verify EVQ doesn't degrade general understanding quality.

Tests simple image/video QA that doesn't depend on temporal position.
If EVQ answers match baseline, no quality tradeoff.
"""
import torch, json, os, math, gc
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def evq_cosh_inv_freq(K, tau, base):
    idx = torch.arange(K, dtype=torch.float64)
    u = (idx + 0.5) / float(K)
    phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    return torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi).float()


def run_vqa(model_path, use_evq=False, tau=1.414, evq_mode="full"):
    """Run simple VQA questions. evq_mode: 'full' or 'temporal_only'."""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, attn_implementation="sdpa", torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path)

    config_name = "baseline"
    if use_evq:
        orig = model.model.rotary_emb.inv_freq.clone()
        if evq_mode == "full":
            K = orig.shape[0]
            evq_freq = evq_cosh_inv_freq(K, tau, 1000000.0).to(orig.device)
            model.model.rotary_emb.inv_freq = evq_freq
            model.model.rotary_emb.original_inv_freq = evq_freq
            config_name = "evq_full_tau%.1f" % tau
        else:
            K_t = 16
            evq_t = evq_cosh_inv_freq(K_t, tau, 1000000.0).to(orig.device)
            new_freq = orig.clone()
            new_freq[:K_t] = evq_t
            model.model.rotary_emb.inv_freq = new_freq
            model.model.rotary_emb.original_inv_freq = new_freq
            config_name = "evq_temporal_tau%.1f" % tau

    # Simple image QA (no temporal dependency)
    questions = [
        {
            "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            "question": "Describe this image in one sentence.",
            "type": "image_caption"
        },
        {
            "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            "question": "What animal is in this image?",
            "type": "image_qa"
        },
    ]

    results = []
    for q in questions:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": q["image"]},
            {"type": "text", "text": q["question"]}
        ]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                          padding=True, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=50)

        output_text = processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

        print("  [%s] Q: %s" % (config_name, q["question"]))
        print("         A: %s" % output_text)
        results.append({"config": config_name, "question": q["question"],
                        "answer": output_text, "type": q["type"]})

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    model_path = "/root/autodl-tmp/models/Qwen2-VL-2B-Instruct"
    output_dir = "/root/autodl-tmp/results/vqa_sanity"
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    print("=" * 50)
    print("  VQA Sanity Check: Baseline")
    print("=" * 50)
    all_results.extend(run_vqa(model_path, use_evq=False))

    print("=" * 50)
    print("  VQA Sanity Check: EVQ Full tau=1.414")
    print("=" * 50)
    all_results.extend(run_vqa(model_path, use_evq=True, tau=1.414, evq_mode="full"))

    print("=" * 50)
    print("  VQA Sanity Check: EVQ Temporal-only tau=1.414")
    print("=" * 50)
    all_results.extend(run_vqa(model_path, use_evq=True, tau=1.414, evq_mode="temporal_only"))

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to %s" % output_dir)
