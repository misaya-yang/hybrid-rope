"""Spatial understanding eval: baseline vs EVQ full on Qwen2-VL."""
import torch, math, gc, json, os
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def evq_cosh_inv_freq(K, tau, base):
    idx = torch.arange(K, dtype=torch.float64)
    u = (idx + 0.5) / float(K)
    phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    return torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi).float()


IMG_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"

SPATIAL_QA = [
    {"question": "Where is the dog positioned relative to the woman?", "type": "spatial_relation"},
    {"question": "Is the woman on the left or right side of the image?", "type": "spatial_position"},
    {"question": "What is in the background of this image?", "type": "scene"},
    {"question": "How many living beings are in this image?", "type": "counting"},
    {"question": "What color is the woman's shirt?", "type": "attribute"},
]

model_path = "/root/autodl-tmp/models/Qwen2-VL-2B-Instruct"
os.makedirs("/root/autodl-tmp/results/spatial_eval", exist_ok=True)

for config_name, apply_evq in [("baseline", False), ("evq_full", True)]:
    print("=" * 50)
    print("  Config: %s" % config_name)
    print("=" * 50)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, attn_implementation="sdpa", torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path)

    if apply_evq:
        K = model.model.rotary_emb.inv_freq.shape[0]
        evq = evq_cosh_inv_freq(K, 1.414, 1000000.0).to("cuda")
        model.model.rotary_emb.inv_freq = evq
        model.model.rotary_emb.original_inv_freq = evq
        print("  EVQ applied")

    results = []
    for q in SPATIAL_QA:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": IMG_URL},
            {"type": "text", "text": q["question"]}
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=80)
        answer = processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        print("  [%s] Q: %s" % (q["type"], q["question"]))
        print("         A: %s" % answer[:120])
        results.append({"config": config_name, "type": q["type"],
                         "question": q["question"], "answer": answer})

    with open("/root/autodl-tmp/results/spatial_eval/%s.json" % config_name, "w") as f:
        json.dump(results, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()

print("\n=== SPATIAL EVAL DONE ===")
