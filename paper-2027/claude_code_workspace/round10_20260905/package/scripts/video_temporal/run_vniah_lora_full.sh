#!/bin/bash
# V-NIAH LoRA pipeline: VIDEO training (not image!) → eval
# Key insight: must use VIDEO data to exercise temporal frequency channels.
# Image QA only uses ~144 tokens → doesn't exercise low/mid freq channels.
# Video QA with 8-16 frames = 1152-2304 tokens → exercises full freq spectrum.
set -e
export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "  V-NIAH LoRA Pipeline (VIDEO training)"
echo "  Model: Qwen2-VL-2B-Instruct"
echo "  Data: UCF-101 VIDEOS (8 frames each, temporal QA)"
echo "  Training: LoRA rank=16, 500 steps"
echo "  Eval: V-NIAH real frames (8-64 frames)"
echo "============================================================"

# ============================================================
# Step 1: Create VIDEO QA training data from UCF-101
# Uses actual video files (not extracted frames!)
# Temporal questions force model to learn position encoding
# ============================================================
echo ""
echo "=== Step 1: Create video QA training data ==="

python << 'PYEOF'
import json, os, glob, random

UCF_DIR = "/root/autodl-tmp/data/UCF-101"
OUT_PATH = "/root/autodl-tmp/vniah_lora_exp/data/train.json"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

classes = sorted([d for d in os.listdir(UCF_DIR) if os.path.isdir(os.path.join(UCF_DIR, d))])
print("  Found %d UCF-101 classes" % len(classes))

data = []
for cls in classes:
    label = ""
    for ch in cls:
        if ch.isupper() and label:
            label += " "
        label += ch.lower()

    videos = sorted(glob.glob(os.path.join(UCF_DIR, cls, "*.avi")))
    for vid_path in videos[:20]:
        # Temporal QA templates - force model to attend to temporal positions
        templates = [
            ("Describe what happens in this video from beginning to end.",
             "The video shows %s. The action progresses through the sequence of frames." % label),
            ("What action is being performed throughout this video?",
             "Throughout the video, someone is performing %s." % label),
            ("Describe the sequence of movements in this video.",
             "The video captures %s, showing the progression of movement across frames." % label),
        ]
        q, a = templates[len(data) % len(templates)]
        data.append({
            "conversations": [
                {"from": "human", "value": "<video>\n" + q},
                {"from": "gpt", "value": a}
            ],
            "video": vid_path
        })

random.seed(42)
random.shuffle(data)
data = data[:3000]

with open(OUT_PATH, "w") as f:
    json.dump(data, f)

print("  Created %d video QA samples" % len(data))
print("  Sample: %s" % data[0]["video"])
print("  Q: %s" % data[0]["conversations"][0]["value"][:60])
PYEOF

# ============================================================
# Step 2: Train GEO LoRA on video QA
# ============================================================
echo ""
echo "=== Step 2: Train GEO LoRA ==="

python << 'PYEOF'
import torch, math, gc, json, os, time
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

model_path = "/root/autodl-tmp/models/Qwen2-VL-2B-Instruct"
data_path = "/root/autodl-tmp/vniah_lora_exp/data/train.json"
save_dir = "/root/autodl-tmp/vniah_lora_exp/geo_lora"
os.makedirs(save_dir, exist_ok=True)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, attn_implementation="sdpa", torch_dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)
print("  GEO inv_freq: [%.6e, %.4f]" % (
    model.model.rotary_emb.inv_freq.min(), model.model.rotary_emb.inv_freq.max()))

lora_config = LoraConfig(r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.0, task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.train()

with open(data_path) as f:
    train_data = json.load(f)
print("  %d video samples, 500 steps" % len(train_data))

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                               lr=2e-5, weight_decay=0.01)
losses = []
t0 = time.time()
torch.cuda.reset_peak_memory_stats()

for step in range(500):
    sample = train_data[step % len(train_data)]

    # Build video message
    messages = [{"role": "user", "content": [
        {"type": "video", "video": "file://" + sample["video"],
         "max_pixels": 224*224, "nframes": 8},
    ]}]
    convs = sample["conversations"]
    user_text = convs[0]["value"].replace("<video>\n", "").replace("<video>", "")
    asst_text = convs[1]["value"]
    messages[0]["content"].append({"type": "text", "text": user_text})
    messages.append({"role": "assistant", "content": asst_text})

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss / 4
        loss.backward()
        if (step + 1) % 4 == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        losses.append(loss.item() * 4)
        if step % 50 == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            vram = torch.cuda.max_memory_allocated() / 1e9
            elapsed = time.time() - t0
            eta = elapsed / max(step, 1) * (500 - step) / 60
            print("  step %d/500 loss=%.4f VRAM=%.1fGB ETA=%.0fmin" % (step, avg, vram, eta))
        del inputs, outputs, loss
        torch.cuda.empty_cache()
    except Exception as e:
        if "CUDA" in str(e):
            torch.cuda.empty_cache()
        continue

model.save_pretrained(save_dir)
print("  GEO done: loss=%.4f" % (sum(losses[-50:])/max(len(losses[-50:]),1)))
del model, optimizer; gc.collect(); torch.cuda.empty_cache()
PYEOF

# ============================================================
# Step 3: Train EVQ LoRA (identical except inv_freq)
# ============================================================
echo ""
echo "=== Step 3: Train EVQ LoRA ==="

python << 'PYEOF'
import torch, math, gc, json, os, time
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

def evq_cosh_inv_freq(K, tau, base):
    idx = torch.arange(K, dtype=torch.float64)
    u = (idx + 0.5) / float(K)
    phi = 1.0 - (1.0/tau) * torch.arcsinh((1.0-u) * math.sinh(tau))
    return torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi).float()

model_path = "/root/autodl-tmp/models/Qwen2-VL-2B-Instruct"
data_path = "/root/autodl-tmp/vniah_lora_exp/data/train.json"
save_dir = "/root/autodl-tmp/vniah_lora_exp/evq_lora"
os.makedirs(save_dir, exist_ok=True)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, attn_implementation="sdpa", torch_dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)

# EVQ patch BEFORE LoRA
K = model.model.rotary_emb.inv_freq.shape[0]
evq = evq_cosh_inv_freq(K, 1.414, 1000000.0).to("cuda")
model.model.rotary_emb.inv_freq = evq
model.model.rotary_emb.original_inv_freq = evq
print("  EVQ inv_freq: [%.6e, %.4f]" % (evq.min(), evq.max()))

lora_config = LoraConfig(r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.0, task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.train()

with open(data_path) as f:
    train_data = json.load(f)
print("  %d video samples, 500 steps" % len(train_data))

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                               lr=2e-5, weight_decay=0.01)
losses = []
t0 = time.time()
torch.cuda.reset_peak_memory_stats()

for step in range(500):
    sample = train_data[step % len(train_data)]
    messages = [{"role": "user", "content": [
        {"type": "video", "video": "file://" + sample["video"],
         "max_pixels": 224*224, "nframes": 8},
    ]}]
    convs = sample["conversations"]
    user_text = convs[0]["value"].replace("<video>\n", "").replace("<video>", "")
    asst_text = convs[1]["value"]
    messages[0]["content"].append({"type": "text", "text": user_text})
    messages.append({"role": "assistant", "content": asst_text})

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss / 4
        loss.backward()
        if (step + 1) % 4 == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        losses.append(loss.item() * 4)
        if step % 50 == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            vram = torch.cuda.max_memory_allocated() / 1e9
            elapsed = time.time() - t0
            eta = elapsed / max(step, 1) * (500 - step) / 60
            print("  step %d/500 loss=%.4f VRAM=%.1fGB ETA=%.0fmin" % (step, avg, vram, eta))
        del inputs, outputs, loss
        torch.cuda.empty_cache()
    except Exception as e:
        if "CUDA" in str(e):
            torch.cuda.empty_cache()
        continue

model.save_pretrained(save_dir)
torch.save(evq.cpu(), os.path.join(save_dir, "evq_inv_freq.pt"))
print("  EVQ done: loss=%.4f" % (sum(losses[-50:])/max(len(losses[-50:]),1)))
del model, optimizer; gc.collect(); torch.cuda.empty_cache()
PYEOF

# ============================================================
# Step 4: Evaluate V-NIAH (real frames, scaling test)
# ============================================================
echo ""
echo "=== Step 4: V-NIAH Evaluation ==="

python << 'PYEOF'
import torch, math, gc, json, os, sys
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

def evq_cosh_inv_freq(K, tau, base):
    idx = torch.arange(K, dtype=torch.float64)
    u = (idx + 0.5) / float(K)
    phi = 1.0 - (1.0/tau) * torch.arcsinh((1.0-u) * math.sinh(tau))
    return torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi).float()

sys.path.insert(0, "/root/autodl-tmp")
from vniah_real import load_haystack_frames, run_single_test, NEEDLE_INFO

model_path = "/root/autodl-tmp/models/Qwen2-VL-2B-Instruct"
haystack_video = "/root/autodl-tmp/data/haystack_clean.mp4"
needle_dir = "/root/autodl-tmp/videorope_code/vision_niah_d/needle_datasets/images"
needle_dataset = "/root/autodl-tmp/videorope_code/vision_niah_d/needle_datasets/dataset.json"
eval_dir = "/root/autodl-tmp/vniah_lora_exp/eval"
os.makedirs(eval_dir, exist_ok=True)

frame_counts = [8, 16, 32, 64]
depths = [0.0, 0.25, 0.5, 0.75, 1.0]

with open(needle_dataset) as f:
    needles = json.load(f)
valid_needles = [n for n in needles if n["path"] in NEEDLE_INFO]
all_options = [NEEDLE_INFO[n["path"]][0] for n in valid_needles]

all_accs = {}
for config_name, lora_dir, use_evq in [
    ("geo_lora", "/root/autodl-tmp/vniah_lora_exp/geo_lora", False),
    ("evq_lora", "/root/autodl-tmp/vniah_lora_exp/evq_lora", True),
]:
    print("\n=== Eval: %s ===" % config_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, attn_implementation="sdpa", torch_dtype=torch.bfloat16, device_map="auto")
    if use_evq:
        K = model.model.rotary_emb.inv_freq.shape[0]
        evq = evq_cosh_inv_freq(K, 1.414, 1000000.0).to("cuda")
        model.model.rotary_emb.inv_freq = evq
        model.model.rotary_emb.original_inv_freq = evq
    model = PeftModel.from_pretrained(model, lora_dir)
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()

    results = []
    for nf in frame_counts:
        haystack = load_haystack_frames(haystack_video, nf)
        for ni in valid_needles:
            np_ = os.path.join(needle_dir, ni["path"])
            if not os.path.exists(np_): continue
            nimg = Image.open(np_)
            ndesc = NEEDLE_INFO[ni["path"]][0]
            for d in depths:
                ip = max(0, min(int(d * len(haystack)), len(haystack)))
                try:
                    pl, cl, rp, ok = run_single_test(model, processor, haystack, nimg, ndesc, ip, all_options)
                    print("  f=%d d=%.1f %s: %s=%s %s" % (nf, d, ni["path"][:10], pl, cl, "OK" if ok else "X"))
                    results.append({"nf": nf+1, "d": d, "ok": ok})
                except:
                    torch.cuda.empty_cache()

    total = len(results)
    nc = sum(r["ok"] for r in results)
    acc = nc / total * 100 if total > 0 else 0
    print("\n  %s: %d/%d = %.1f%%" % (config_name, nc, total, acc))
    for nf in frame_counts:
        sub = [r for r in results if r["nf"]==nf+1]
        if sub: print("    %df: %.1f%%" % (nf+1, sum(r["ok"] for r in sub)/len(sub)*100))
    all_accs[config_name] = acc
    with open(os.path.join(eval_dir, "%s.json" % config_name), "w") as f:
        json.dump({"config": config_name, "accuracy": acc, "details": results}, f, indent=2)
    del model; gc.collect(); torch.cuda.empty_cache()

print("\n============================================================")
print("  FINAL: GEO=%.1f%% EVQ=%.1f%%" % (all_accs.get("geo_lora",0), all_accs.get("evq_lora",0)))
print("============================================================")
PYEOF

echo "=== PIPELINE COMPLETE ==="
date
