"""LoRA fine-tune Qwen2-VL-2B on UCF-101 video QA, then eval V-NIAH.

Two configs in one script (h2h):
1. GEO baseline + LoRA
2. EVQ full + LoRA
Same data, same hyperparams, only inv_freq differs.

Uses UCF-101 videos already on server. No external downloads needed.
"""
import torch, math, gc, json, os, glob, random
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def evq_cosh_inv_freq(K, tau, base):
    idx = torch.arange(K, dtype=torch.float64)
    u = (idx + 0.5) / float(K)
    phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    return torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi).float()


# ============================================================
# Dataset: UCF-101 as video QA
# ============================================================
class UCF101VideoQA(Dataset):
    """UCF-101 formatted as video QA for Qwen2-VL training."""

    def __init__(self, ucf_dir, processor, n_samples=2000, n_frames=8, seed=42):
        rng = random.Random(seed)
        self.processor = processor
        self.n_frames = n_frames
        self.samples = []

        # Find all videos grouped by class
        classes = sorted(os.listdir(ucf_dir))
        classes = [c for c in classes if os.path.isdir(os.path.join(ucf_dir, c))]

        all_videos = []
        for cls in classes:
            vids = glob.glob(os.path.join(ucf_dir, cls, "*.avi"))
            for v in vids:
                # Class name: CamelCase -> "camel case"
                label = ""
                for ch in cls:
                    if ch.isupper() and label:
                        label += " "
                    label += ch.lower()
                all_videos.append({"path": v, "label": label, "class": cls})

        rng.shuffle(all_videos)
        self.samples = all_videos[:n_samples]
        print("  UCF-101 dataset: %d samples, %d classes" % (len(self.samples), len(classes)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample


def train_lora(model_path, ucf_dir, output_dir, use_evq=False, tau=1.414,
               n_steps=500, lr=2e-5, lora_rank=16, n_frames=8, batch_size=1):
    """LoRA fine-tune on UCF-101 video QA."""
    config_name = "evq_full" if use_evq else "baseline"
    save_dir = os.path.join(output_dir, config_name)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("  LoRA Training: %s" % config_name)
    print("=" * 60)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, attn_implementation="sdpa", torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_path)

    if use_evq:
        K = model.model.rotary_emb.inv_freq.shape[0]
        evq = evq_cosh_inv_freq(K, tau, 1000000.0).to("cuda")
        model.model.rotary_emb.inv_freq = evq
        model.model.rotary_emb.original_inv_freq = evq
        print("  EVQ applied: tau=%.3f" % tau)

    # LoRA
    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0, task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    # Dataset
    dataset = UCF101VideoQA(ucf_dir, processor, n_samples=min(n_steps * 4, 5000),
                            n_frames=n_frames)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01)

    print("  Training %d steps..." % n_steps)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    step = 0

    while step < n_steps:
        for sample in dataset.samples:
            if step >= n_steps:
                break

            # Build conversation
            messages = [{"role": "user", "content": [
                {"type": "video", "video": "file://" + sample["path"],
                 "max_pixels": 224 * 224, "nframes": n_frames},
                {"type": "text", "text": "What action is being performed in this video? Answer briefly."}
            ]}, {"role": "assistant", "content": sample["label"]}]

            try:
                text = processor.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=False)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                                   padding=True, return_tensors="pt").to(model.device)

                # Forward with labels
                outputs = model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                losses.append(loss.item())
                if step % 50 == 0:
                    avg = sum(losses[-50:]) / len(losses[-50:])
                    vram = torch.cuda.max_memory_allocated() / 1e9
                    print("  step %d/%d  loss=%.4f  VRAM=%.1fGB" % (step, n_steps, avg, vram))

                del inputs, outputs, loss
                torch.cuda.empty_cache()

            except Exception as e:
                if "CUDA" in str(e):
                    torch.cuda.empty_cache()
                continue

            step += 1

    avg_loss = sum(losses[-100:]) / max(len(losses[-100:]), 1)
    vram = torch.cuda.max_memory_allocated() / 1e9
    print("  Done: %d steps, loss=%.4f, VRAM=%.1fGB" % (step, avg_loss, vram))

    # Save LoRA
    model.save_pretrained(save_dir)
    print("  Saved to %s" % save_dir)

    # Return model for eval (don't delete yet)
    return model, processor


def eval_vniah(model, processor, haystack_video, needle_dir, needle_dataset,
               frame_counts, depths, output_dir, config_name):
    """Evaluate V-NIAH with real frames."""
    from vniah_real import load_haystack_frames, run_single_test, NEEDLE_INFO
    os.makedirs(output_dir, exist_ok=True)

    with open(needle_dataset) as f:
        needles = json.load(f)
    valid_needles = [n for n in needles if n["path"] in NEEDLE_INFO]
    all_options = [NEEDLE_INFO[n["path"]][0] for n in valid_needles]

    results = []
    for n_frames in frame_counts:
        haystack = load_haystack_frames(haystack_video, n_frames)
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
                    results.append({"n_frames": n_frames + 1, "needle": needle_item["path"],
                                   "depth": depth, "correct": correct, "config": config_name})
                except:
                    torch.cuda.empty_cache()
                    continue

    total = len(results)
    n_correct = sum(r["correct"] for r in results)
    acc = n_correct / total * 100 if total > 0 else 0
    print("  %s V-NIAH: %d/%d = %.1f%%" % (config_name, n_correct, total, acc))

    for nf in frame_counts:
        subset = [r for r in results if r["n_frames"] == nf + 1]
        if subset:
            a = sum(r["correct"] for r in subset) / len(subset) * 100
            print("    %d frames: %.1f%%" % (nf + 1, a))

    with open(os.path.join(output_dir, "%s.json" % config_name), "w") as f:
        json.dump({"config": config_name, "accuracy": acc, "details": results}, f, indent=2)

    return acc


if __name__ == "__main__":
    model_path = "/root/autodl-tmp/models/Qwen2-VL-2B-Instruct"
    ucf_dir = "/root/autodl-tmp/data/UCF-101"
    output_dir = "/root/autodl-tmp/results/vniah_lora"
    haystack_video = "/root/autodl-tmp/data/haystack_clean.mp4"
    needle_dir = "/root/autodl-tmp/videorope_code/vision_niah_d/needle_datasets/images"
    needle_dataset = "/root/autodl-tmp/videorope_code/vision_niah_d/needle_datasets/dataset.json"

    frame_counts = [8, 16, 32, 64]
    depths = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Config 1: GEO baseline + LoRA
    model_geo, proc_geo = train_lora(model_path, ucf_dir, output_dir,
                                      use_evq=False, n_steps=500)
    acc_geo = eval_vniah(model_geo, proc_geo, haystack_video, needle_dir, needle_dataset,
                         frame_counts, depths, output_dir, "geo_lora")
    del model_geo, proc_geo
    gc.collect()
    torch.cuda.empty_cache()

    # Config 2: EVQ + LoRA
    model_evq, proc_evq = train_lora(model_path, ucf_dir, output_dir,
                                      use_evq=True, tau=1.414, n_steps=500)
    acc_evq = eval_vniah(model_evq, proc_evq, haystack_video, needle_dir, needle_dataset,
                         frame_counts, depths, output_dir, "evq_lora")
    del model_evq, proc_evq
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("  FINAL: GEO LoRA = %.1f%%, EVQ LoRA = %.1f%%" % (acc_geo, acc_evq))
    print("=" * 60)
