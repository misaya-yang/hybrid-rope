"""V-NIAH-D eval v2: correct M-RoPE position_ids from VideoRoPE codebase."""
import torch, json, os, sys, math, argparse, gc
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

IMAGE_TOKENS = 144


def get_m_rope_index(input_embeds, video_se):
    """Correct M-RoPE position IDs from VideoRoPE codebase."""
    llm_pos_ids_list = []
    llm_pos_ids_list.append(torch.arange(video_se[0]).view(1, 1, -1).expand(3, 1, -1))
    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
    nframes = (video_se[1] - video_se[0]) // IMAGE_TOKENS
    llm_grid_t, llm_grid_h, llm_grid_w = nframes, 9, 16
    t_index = torch.arange(nframes).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]).unsqueeze(dim=1) + st_idx)
    if input_embeds.shape[1] > video_se[1]:
        text_len = input_embeds.shape[1] - video_se[1]
        llm_pos_ids_list.append(torch.arange(
            llm_pos_ids_list[-1].max().item() + 1,
            llm_pos_ids_list[-1].max().item() + 1 + text_len
        ).view(1, 1, -1).expand(3, 1, -1))
    position_ids = torch.cat(llm_pos_ids_list, dim=-1)
    return position_ids


def evq_cosh_inv_freq(K, tau, base):
    idx = torch.arange(K, dtype=torch.float64)
    u = (idx + 0.5) / float(K)
    phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    return torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi).float()


def run_eval(model_path, haystack_dir, needle_dir, needle_dataset,
             max_frames, min_frames, frame_interval, depth_interval,
             output_dir, use_evq=False, tau=1.414):
    device = "cuda"
    os.makedirs(output_dir, exist_ok=True)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, attn_implementation="sdpa", device_map="auto", torch_dtype=torch.bfloat16)
    del model.visual
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    config_name = "EVQ" if use_evq else "M-RoPE Baseline"
    if use_evq:
        orig_freq = model.model.rotary_emb.inv_freq.clone()
        K_total = orig_freq.shape[0]  # 64
        K_t = 16  # mrope_section[0] = temporal channels
        base = 1000000.0
        # EVQ ONLY on temporal channels (0-15), keep spatial (16-63) unchanged
        evq_temporal = evq_cosh_inv_freq(K_t, tau, base)  # 16 values for temporal
        new_freq = orig_freq.clone()
        new_freq[:K_t] = evq_temporal.to(orig_freq.device)
        model.model.rotary_emb.inv_freq = new_freq
        model.model.rotary_emb.original_inv_freq = new_freq
        print("  EVQ applied to TEMPORAL ONLY (channels 0-15):")
        print("    Original temporal: [%.6e, %.4f]" % (orig_freq[:K_t].min(), orig_freq[:K_t].max()))
        print("    EVQ temporal:      [%.6e, %.4f]" % (evq_temporal.min(), evq_temporal.max()))
        print("    Spatial unchanged: [%.6e, %.4f]" % (new_freq[K_t:].min(), new_freq[K_t:].max()))

    print("  VRAM: %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9))

    haystack = torch.load(os.path.join(haystack_dir, "video_embeddings.pt")).to(torch.bfloat16).to(device)
    with open(needle_dataset) as f:
        needles = json.load(f)

    preprompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    postprompt = "\nWhat is the content of the image in the video? Answer the question using a single word or phrase.<|im_end|>\n<|im_start|>assistant\n"

    results = []
    frame_nums = list(range(min_frames, max_frames + 1, frame_interval))
    depths = [round(i * depth_interval, 2) for i in range(int(1 / depth_interval) + 1)]

    for idx, item in enumerate(needles):
        needle_file = os.path.join(needle_dir, "%d.pt" % idx)
        if not os.path.exists(needle_file):
            continue
        needle_emb = torch.load(needle_file).to(torch.bfloat16).to(device)
        answer = item["answer"]

        for n_frames in frame_nums:
            n_tokens = n_frames * IMAGE_TOKENS
            if n_tokens <= haystack.shape[0]:
                hay = haystack[:n_tokens]
            else:
                hay = haystack.repeat((n_tokens // haystack.shape[0]) + 1, 1)[:n_tokens]

            for depth in depths:
                insert_pos = int(depth * max(0, n_tokens - needle_emb.shape[0]))
                video_emb = torch.cat([
                    hay[:insert_pos], needle_emb, hay[insert_pos + needle_emb.shape[0]:]
                ], dim=0)[:n_tokens].unsqueeze(0)

                pre_ids = tokenizer(preprompt, return_tensors="pt").input_ids.to(device)
                post_ids = tokenizer(postprompt, return_tensors="pt").input_ids.to(device)
                answer_ids = tokenizer(answer, return_tensors="pt").input_ids.to(device)

                with torch.inference_mode():
                    pre_emb = model.model.embed_tokens(pre_ids)
                    post_emb = model.model.embed_tokens(post_ids)
                    answer_emb = model.model.embed_tokens(answer_ids)
                    input_emb = torch.cat([pre_emb, video_emb, post_emb, answer_emb], dim=1)

                    video_se = (pre_emb.shape[1], pre_emb.shape[1] + video_emb.shape[1])
                    position_ids = get_m_rope_index(input_emb, video_se).to(device)

                    out = model.model(inputs_embeds=input_emb, position_ids=position_ids, use_cache=False)
                    logits = model.lm_head(out[0]).float()

                prompt_len = pre_emb.shape[1] + video_emb.shape[1] + post_emb.shape[1]
                pred = logits[:, prompt_len - 1:prompt_len + answer_ids.shape[1] - 1].argmax(dim=-1)
                correct = (pred == answer_ids.to(device)).all().item()
                pred_text = tokenizer.decode(pred.squeeze().tolist())

                status = "OK" if correct else "FAIL"
                print("  f=%d d=%.2f n=%-15s: pred='%s' ans='%s' %s" % (
                    n_frames, depth, item["path"][:15], pred_text, answer, status))
                results.append({
                    "needle": item["path"], "answer": answer, "frames": n_frames,
                    "depth": depth, "correct": correct, "predicted": pred_text
                })

    total = len(results)
    correct_count = sum(r["correct"] for r in results)
    acc = correct_count / total * 100 if total > 0 else 0

    summary = {
        "config": config_name, "accuracy": acc, "total": total, "correct": correct_count,
        "use_evq": use_evq, "tau": tau if use_evq else None
    }

    print("\n=== %s Results: %d/%d = %.1f%% ===" % (config_name, correct_count, total, acc))

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen2-VL-2B-Instruct")
    parser.add_argument("--haystack_dir", required=True)
    parser.add_argument("--needle_dir", required=True)
    parser.add_argument("--needle_dataset", required=True)
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--min_frames", type=int, default=50)
    parser.add_argument("--frame_interval", type=int, default=50)
    parser.add_argument("--depth_interval", type=float, default=0.25)
    parser.add_argument("--output", default="/root/autodl-tmp/results/vniah")
    parser.add_argument("--evq", action="store_true")
    parser.add_argument("--tau", type=float, default=1.414)
    args = parser.parse_args()

    print("=" * 60)
    if args.evq:
        print("  V-NIAH-D: EVQ (tau=%.3f)" % args.tau)
    else:
        print("  V-NIAH-D: M-RoPE Baseline")
    print("=" * 60)

    out_subdir = "evq_tau%.1f" % args.tau if args.evq else "baseline"
    run_eval(args.model, args.haystack_dir, args.needle_dir, args.needle_dataset,
             args.max_frames, args.min_frames, args.frame_interval, args.depth_interval,
             os.path.join(args.output, out_subdir), args.evq, args.tau)
