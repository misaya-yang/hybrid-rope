#!/usr/bin/env python3
"""Pre-encode ALL text prompts needed for eval (VBench + custom).

Run on AWS after wan21_prepare_data.py finishes.
This way RTX PRO 6000 never needs T5.

Output: eval_text_embeds.pt (dict of prompt → embedding)
"""
import os, sys, gc, json, torch
from pathlib import Path

# Eval prompts: temporal-focused VBench subset + custom
EVAL_PROMPTS = {
    # Temporal dynamics (our focus)
    "vbench_temporal_01": "A person is walking slowly along a quiet beach at sunset",
    "vbench_temporal_02": "Waves rolling onto shore with foam spreading on sand",
    "vbench_temporal_03": "A bird flying across a cloudy sky from left to right",
    "vbench_temporal_04": "Rain falling steadily on a window with water streaming down",
    "vbench_temporal_05": "A clock pendulum swinging back and forth",
    "vbench_temporal_06": "Smoke rising from a chimney and dispersing in the wind",
    "vbench_temporal_07": "A leaf falling from a tree in autumn breeze",
    "vbench_temporal_08": "Traffic flowing on a busy highway at dusk",
    "vbench_temporal_09": "A dancer spinning gracefully on a wooden stage",
    "vbench_temporal_10": "Clouds drifting slowly across a blue sky time lapse",
    # Motion variety
    "motion_01": "A dog running across a green field chasing a ball",
    "motion_02": "A train moving along tracks through mountain scenery",
    "motion_03": "Ocean waves crashing against rocky cliffs",
    "motion_04": "A butterfly fluttering between colorful flowers in a garden",
    "motion_05": "Cars driving through a rainy city street at night",
    # Static (control: should look same at any length)
    "static_01": "A still life painting of fruits on a wooden table",
    "static_02": "A peaceful mountain lake reflecting snow-capped peaks",
    "static_03": "An empty room with sunlight streaming through a window",
    "static_04": "A close-up of a textured stone wall",
    "static_05": "A frozen waterfall in winter landscape",
}

# Negative prompt (same as Wan2.1 default)
NEG_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/wan21_encoded")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.model_dir).parent / "Wan2.1"))
    from wan.modules.t5 import T5EncoderModel
    from wan.configs.wan_t2v_1_3B import t2v_1_3B as cfg

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    save_path = out_path / "eval_text_embeds.pt"
    if save_path.exists():
        print(f"Already exists: {save_path}")
        return

    print("Loading T5 (CPU, may use swap)...")
    t5 = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=torch.device('cpu'),
        checkpoint_path=os.path.join(args.model_dir, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(args.model_dir, cfg.t5_tokenizer),
    )
    print("T5 loaded.")

    all_embeds = {}

    # Encode all eval prompts
    for name, prompt in EVAL_PROMPTS.items():
        with torch.no_grad():
            embed = t5([prompt], torch.device('cpu'))
        all_embeds[name] = embed[0].cpu().float()
        print(f"  {name}: {embed[0].shape}")

    # Encode negative prompt
    with torch.no_grad():
        neg_embed = t5([NEG_PROMPT], torch.device('cpu'))
    all_embeds["__negative__"] = neg_embed[0].cpu().float()
    print(f"  negative: {neg_embed[0].shape}")

    # Save
    torch.save(all_embeds, save_path)
    size_mb = save_path.stat().st_size / 1e6
    print(f"\nSaved {len(all_embeds)} embeddings to {save_path} ({size_mb:.1f} MB)")

    # Also save prompt list for reference
    with open(out_path / "eval_prompts.json", "w") as f:
        json.dump(EVAL_PROMPTS, f, indent=2)

    del t5
    gc.collect()


if __name__ == "__main__":
    main()
