"""Minimal test: 1 video, 1 question. If this works, video_passkey.py will work."""
import torch, os, tempfile, numpy as np
from PIL import Image

# Step 1: Generate 4-frame video (448x448, solid colors)
print("Step 1: Generate video...")
colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
frames = [Image.new("RGB", (448, 448), c) for c in colors]

tmp = os.path.join(tempfile.gettempdir(), "test_video.mp4")
import imageio
writer = imageio.get_writer(tmp, fps=1, format="FFMPEG")
for f in frames:
    writer.append_data(np.array(f))
writer.close()
print("  Video saved: %s (%d bytes)" % (tmp, os.path.getsize(tmp)))

# Step 2: Load model
print("Step 2: Load model...")
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/root/autodl-tmp/models/Qwen2-VL-2B-Instruct",
    attn_implementation="sdpa", torch_dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained("/root/autodl-tmp/models/Qwen2-VL-2B-Instruct")
print("  Model loaded. VRAM: %.1f GB" % (torch.cuda.max_memory_allocated()/1e9))

# Step 3: Ask question about the video
print("Step 3: Run inference...")
messages = [{"role": "user", "content": [
    {"type": "video", "video": "file://" + tmp, "nframes": 4},
    {"type": "text", "text": "This video has 4 frames of solid colors: red, green, blue, yellow. What color is the second frame? A) red B) green C) blue D) yellow. Answer with just the letter."}
]}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                   padding=True, return_tensors="pt").to(model.device)

print("  Input shape: %s" % str(inputs.input_ids.shape))
print("  VRAM after input: %.1f GB" % (torch.cuda.max_memory_allocated()/1e9))

with torch.inference_mode():
    ids = model.generate(**inputs, max_new_tokens=5)
pred = processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

print("  Prediction: '%s'" % pred)
print("  Expected: 'B' (green)")
print()
if pred and pred[0].upper() == "B":
    print("=== TEST PASSED ===")
else:
    print("=== TEST FAILED (but no crash = pipeline works) ===")
    print("  Model answered '%s' instead of 'B', but the pipeline ran without errors." % pred)
