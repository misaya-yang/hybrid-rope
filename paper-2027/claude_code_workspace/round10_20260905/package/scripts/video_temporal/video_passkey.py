"""Video Passkey Retrieval: visual analog of text passkey.

Architecture: Frozen CLIP ViT-B/16 -> MLP Projector -> Causal Decoder (3D RoPE)
Task: "What is shown in frame K?" -> answer from options
Training: 16 frames, simple colored/numbered patterns
Eval: extrapolate to 32/64/128 frames

GEO should degrade at extrapolation, EVQ should hold.
Direct analog of text passkey 100% result.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import json
import time
import gc
import numpy as np
from PIL import Image
from typing import Optional, Tuple


# ============================================================
# 3D RoPE (from video_dit.py, adapted for causal decoder)
# ============================================================

def evq_cosh_inv_freq(dim: int, tau: float, base: float = 10000.0) -> torch.Tensor:
    K = dim // 2
    idx = torch.arange(K, dtype=torch.float64)
    u = (idx + 0.5) / float(K)
    if abs(tau) < 1e-8:
        phi = u
    else:
        phi = 1.0 - (1.0 / tau) * torch.arcsinh((1.0 - u) * math.sinh(tau))
    return torch.pow(torch.tensor(float(base), dtype=torch.float64), -phi).float()


def build_3d_rope_cache(max_t, max_h, max_w, inv_freq_t, inv_freq_h, inv_freq_w):
    """Build 3D RoPE cos/sin cache for video tokens."""
    # For each (t, h, w) position, compute combined frequency
    positions_t = torch.arange(max_t, dtype=torch.float32)
    positions_h = torch.arange(max_h, dtype=torch.float32)
    positions_w = torch.arange(max_w, dtype=torch.float32)

    freqs_t = torch.outer(positions_t, inv_freq_t)  # [T, K_t]
    freqs_h = torch.outer(positions_h, inv_freq_h)  # [H, K_h]
    freqs_w = torch.outer(positions_w, inv_freq_w)  # [W, K_w]

    return freqs_t, freqs_h, freqs_w


def get_3d_rope_pos(seq_len, n_frames, grid_h, grid_w, freqs_t, freqs_h, freqs_w, device):
    """Get RoPE cos/sin for a sequence of video tokens + text tokens."""
    # Video tokens: (t, h, w) positions
    video_len = n_frames * grid_h * grid_w
    t_idx = torch.arange(n_frames, device=device).repeat_interleave(grid_h * grid_w)
    h_idx = torch.arange(grid_h, device=device).repeat(n_frames * grid_w)
    # Fix h_idx to properly tile
    h_idx = torch.arange(grid_h, device=device).unsqueeze(1).expand(-1, grid_w).flatten()
    h_idx = h_idx.repeat(n_frames)
    w_idx = torch.arange(grid_w, device=device).repeat(n_frames * grid_h)
    w_idx = torch.arange(grid_w, device=device).repeat(grid_h)
    w_idx = w_idx.repeat(n_frames)

    ft = freqs_t.to(device)[t_idx]  # [video_len, K_t]
    fh = freqs_h.to(device)[h_idx]  # [video_len, K_h]
    fw = freqs_w.to(device)[w_idx]  # [video_len, K_w]
    video_freqs = torch.cat([ft, fh, fw], dim=-1)  # [video_len, K_t+K_h+K_w]

    # Text tokens: sequential positions after video
    text_len = seq_len - video_len
    if text_len > 0:
        text_pos = torch.arange(n_frames, n_frames + text_len, device=device, dtype=torch.float32)
        text_freqs_all = []
        for inv_f in [freqs_t[0:1], freqs_h[0:1], freqs_w[0:1]]:
            # Use same frequency but sequential positions
            K_i = inv_f.shape[-1]
            inv_f_flat = inv_f[0].to(device)
            tf = torch.outer(text_pos, inv_f_flat)
            text_freqs_all.append(tf)
        text_freqs = torch.cat(text_freqs_all, dim=-1)
        all_freqs = torch.cat([video_freqs, text_freqs], dim=0)
    else:
        all_freqs = video_freqs

    # Double for cos/sin pairs
    emb = torch.cat([all_freqs, all_freqs], dim=-1)
    return emb.cos(), emb.sin()


# ============================================================
# Synthetic data: colored/numbered frames
# ============================================================

COLORS = ["red", "blue", "green", "yellow", "purple", "orange",
          "pink", "cyan", "brown", "gray", "white", "black"]


def generate_colored_frame(color_name, size=448):
    """Generate a solid colored image with text label."""
    color_map = {
        "red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0),
        "yellow": (255, 255, 0), "purple": (128, 0, 128), "orange": (255, 165, 0),
        "pink": (255, 192, 203), "cyan": (0, 255, 255), "brown": (139, 69, 19),
        "gray": (128, 128, 128), "white": (255, 255, 255), "black": (0, 0, 0),
    }
    rgb = color_map.get(color_name, (128, 128, 128))
    img = Image.new("RGB", (size, size), rgb)
    return img


def generate_passkey_sample(n_frames, seed=None):
    """Generate a video passkey sample.

    Returns:
        frames: list of PIL Images
        query_frame: int (which frame to ask about)
        answer: str (color name)
        options: list of str (4 options including answer)
    """
    rng = np.random.RandomState(seed)
    # Pick n_frames random colors (allow repeats for longer sequences)
    frame_colors = [COLORS[rng.randint(0, len(COLORS))] for _ in range(n_frames)]
    frames = [generate_colored_frame(c) for c in frame_colors]

    # Pick a random frame to ask about
    query_frame = rng.randint(0, n_frames)
    answer = frame_colors[query_frame]

    # Generate 4 options (including correct answer)
    wrong_colors = [c for c in COLORS if c != answer]
    rng.shuffle(wrong_colors)
    options = [answer] + wrong_colors[:3]
    rng.shuffle(options)

    return frames, query_frame, answer, options


# ============================================================
# Quick test with Qwen2-VL (inference only, verify task works)
# ============================================================

def test_passkey_qwen2vl(model_path, n_frames=8, n_tests=20, patch_fn=None,
                          config_name="baseline"):
    """Test video passkey on Qwen2-VL-2B."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, attn_implementation="sdpa", torch_dtype=torch.bfloat16, device_map="auto")
    # Video pixel constraints in qwen_vl_utils:
    #   min: 128*28*28 = 100,352 pixels per frame
    #   max: 768*28*28 = 602,112 pixels per frame
    # Our frames: 448x448 = 200,704 (within range)
    # 448 is divisible by both 16 (imageio macro_block) and 28 (Qwen2-VL image_factor)
    processor = AutoProcessor.from_pretrained(model_path)

    if patch_fn:
        # Find rotary embedding (path differs across transformers versions)
        rope = None
        for attr_path in ['model.rotary_emb', 'language_model.rotary_emb', 'model.language_model.rotary_emb']:
            obj = model
            try:
                for a in attr_path.split('.'):
                    obj = getattr(obj, a)
                if hasattr(obj, 'inv_freq'):
                    rope = obj
                    break
            except AttributeError:
                continue
        assert rope is not None, "Cannot find rotary_emb with inv_freq"
        orig = rope.inv_freq.clone()
        new_freq = patch_fn(orig)
        rope.inv_freq = new_freq.to(orig.device)
        rope.original_inv_freq = new_freq.to(orig.device)

    import tempfile, imageio

    results = []
    for i in range(n_tests):
        frames, qf, answer, options = generate_passkey_sample(n_frames, seed=i)

        # Save frames as temporary video file (Qwen2-VL needs video format for M-RoPE)
        tmp_video = os.path.join(tempfile.gettempdir(), "passkey_%d.mp4" % i)
        writer = imageio.get_writer(tmp_video, fps=1, format="FFMPEG")
        for f in frames:
            writer.append_data(np.array(f))
        writer.close()

        options_text = " ".join(["%s) %s" % (chr(65 + j), opt) for j, opt in enumerate(options)])
        content = [
            {"type": "video", "video": "file://" + tmp_video, "nframes": n_frames},
            {"type": "text", "text":
                "I showed you %d frames. Each frame is a solid color. What color is frame number %d (counting from 1)?\n%s\nAnswer with just the letter." % (
                    n_frames, qf + 1, options_text)}
        ]

        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=5)
        pred = processor.batch_decode(ids[:, inputs.input_ids.shape[1]:],
                                      skip_special_tokens=True)[0].strip()

        correct_idx = options.index(answer)
        correct_letter = chr(65 + correct_idx)
        pred_letter = pred[0].upper() if pred else "?"
        ok = pred_letter == correct_letter

        results.append({"ok": ok, "pred": pred, "answer": answer, "qf": qf})
        del inputs, ids
        torch.cuda.empty_cache()

    acc = sum(r["ok"] for r in results) / len(results) * 100
    print("  %s %df: %d/%d = %.1f%%" % (config_name, n_frames, sum(r["ok"] for r in results), len(results), acc))

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return acc, results


if __name__ == "__main__":
    model_path = "/root/autodl-tmp/models/Qwen2-VL-2B-Instruct"
    output_dir = "/root/autodl-tmp/results/video_passkey"
    os.makedirs(output_dir, exist_ok=True)

    # Qwen2-VL-2B: head_dim=128, inv_freq has 64 values
    # evq_cosh_inv_freq takes dim (not pairs), returns dim//2 values
    HEAD_DIM = 128  # dim parameter, returns dim//2=64 values
    TAU = 1.414
    BASE = 1000000.0
    evq_full_freq = evq_cosh_inv_freq(HEAD_DIM, TAU, BASE)
    assert evq_full_freq.shape[0] == 64, "Expected 64 inv_freq values, got %d" % evq_full_freq.shape[0]

    def patch_temporal(orig):
        new = orig.clone()
        new[:16] = evq_full_freq[:16]
        return new

    def patch_full(orig):
        return evq_full_freq.clone()

    all_results = {}
    # 4/8/16 = within Qwen2-VL training range
    # 32/64 = extrapolation (should show EVQ advantage)
    for n_frames in [4, 8, 16, 32, 64]:
        print("\n=== %d frames ===" % n_frames)
        for name, fn in [("baseline", None), ("evq_temporal", patch_temporal), ("evq_full", patch_full)]:
            try:
                acc, res = test_passkey_qwen2vl(model_path, n_frames, n_tests=20,
                                                 patch_fn=fn, config_name=name)
                all_results["%s_%df" % (name, n_frames)] = acc
            except torch.cuda.OutOfMemoryError:
                print("  %s %df: OOM" % (name, n_frames))
                torch.cuda.empty_cache()
                gc.collect()
                all_results["%s_%df" % (name, n_frames)] = -1

    print("\n" + "=" * 50)
    print("  VIDEO PASSKEY RESULTS")
    print("=" * 50)
    for k, v in sorted(all_results.items()):
        print("  %s: %.1f%%" % (k, v))

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
