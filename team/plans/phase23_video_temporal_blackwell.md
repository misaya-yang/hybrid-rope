# Phase 23: Video Temporal Allocation on Blackwell 96GB

## Goal

Use a medium-cost video route to strengthen the Spotlight case without waiting for another large undertrained text run.

The target claim is narrower and cleaner than “EVQ helps all video tasks”:

> **The temporal axis needs a larger low-frequency budget, and EVQ-Cosh is a principled way to shape that temporal budget.**

This is motivated directly by VideoRoPE and is complementary to our existing text claims:

- fixed-length text: EVQ slightly loses short-range, strongly wins extrapolation
- EVQ + YaRN: super-additive gains
- downstream support: gold-answer NLL and NIAH-type retrieval

## Why This Route

### Existing text package is already solid

What we can already defend:

- EVQ long-context advantage is consistent in fixed-length text training
- EVQ + YaRN is a strong orthogonal story
- τ* formula is validated at scale in text

What is still weak:

- larger-scale fully trained anchors
- a cleaner downstream result that is hard to dismiss as protocol noise

### Video gives a better medium-cost bridge

VideoRoPE is the right bridge because the paper’s main diagnosis is already close to ours:

- **Low-frequency Temporal Allocation (LTA)** matters
- **Adjustable Temporal Spacing (ATS)** matters
- **V-NIAH-D** exposes frequency-allocation distractors directly

Official sources:

- Paper / repo: [VideoRoPE official repo](https://github.com/Wiselnn570/VideoRoPE)
- Official model collection: [Hugging Face collection](https://huggingface.co/collections/Wiselnn/videorope-what-makes-for-good-video-rotary-position-embeddi-67ca90664c8e169422449c56)
- Official dataset repo: `Wiselnn/VideoRoPE`
- Official model repo: `Wiselnn/Qwen2-VL-videorope-128frames-8k-context-330k-llava-video`

## Available Hardware

- GPU: RTX / R6000 Blackwell, 96GB

This is enough for:

- comfortable synthetic video LM training at 2K-8K token lengths
- official `Qwen2-VL-7B` inference-time V-NIAH-D evaluation
- likely Qwen2-VL-2B / 7B adapter-style ablations if needed later

It is not a reason to jump immediately into a full video SFT pipeline.
The highest ROI is:

1. controlled synthetic proof
2. official benchmark inference patch

## Experiment Stack

### Track A: Controlled Synthetic Temporal-Allocation Sweep

Purpose:

- isolate temporal allocation cleanly
- avoid video-VLM confounders
- find the strongest `K_t` and `tau` before touching official long-video eval

Dataset:

- tokenized Moving MNIST
- training context: `32` frames
- evaluation: `32 / 64 / 96 / 128` frames

Model:

- `scripts/video_temporal/run_video_temporal_allocation_sweep.py`
- default profile: `blackwell96`
- architecture:
  - hidden size `1024`
  - layers `16`
  - heads `16`
  - `d_head = 64`

Variants:

- `geo_k8`
- `geo_k12`
- `geo_k16`
- `evq_k12`
- `evq_k16`

Interpretation:

- `geo_k8 -> geo_k12 -> geo_k16` tests whether more temporal low-frequency budget helps by itself
- `geo_k12 -> evq_k12` tests whether EVQ reshaping helps inside the same temporal budget
- `geo_k16 -> evq_k16` tests whether EVQ still helps when temporal budget is already generous

Success criteria:

- if `geo_k12` or `geo_k16` beats `geo_k8` at `64+` frames:
  VideoRoPE’s LTA-style claim is replicated in our controlled setup
- if `evq_k12` or `evq_k16` further beats the corresponding geo variant:
  EVQ becomes a principled temporal-allocation mechanism, not just a text trick

### Track B: Official VideoRoPE Benchmark Bridge

Purpose:

- move from toy synthetic proof to a recognized external benchmark
- anchor the story to V-NIAH-D, where VideoRoPE already shows large sensitivity to frequency allocation

Immediate low-risk target:

- use the official benchmark assets and run official modes first:
  - `vanilla_rope`
  - `tad_rope`
  - `m_rope`
  - `videorope`

Then add our own patch once Track A identifies the best temporal budget:

- `m_rope + EVQ temporal inv_freq`
- `videorope + EVQ temporal inv_freq`

Technical note:

- a clean EVQ patch on Qwen2-VL cannot be expressed by a single temporal position-id rescaling
- it requires modifying the temporal slice of `inv_freq`, because EVQ is channel-wise

That is why Track A comes first.

## Code Added

### Data prep

- `scripts/data_prep/prepare_moving_mnist_video.py`
  - downloads raw MNIST gzip files directly
  - generates tokenized Moving MNIST caches
  - avoids `torchvision` / PIL dependency

- `scripts/data_prep/prepare_videorope_assets.py`
  - downloads lightweight official VideoRoPE benchmark metadata
  - pulls `V-NIAH-D` needle json + images + official result json

### Main experiment

- `scripts/video_temporal/run_video_temporal_allocation_sweep.py`
  - trains a 3D-RoPE VideoGPT on tokenized Moving MNIST
  - sweeps temporal channel budget + EVQ temporal shaping
  - evaluates both raw and temporal YaRN extrapolation

## Local Data Paths

Generated / downloaded local assets are intentionally gitignored:

- `data/video_temporal/generated/`
- `data/video_temporal/external/`

Default paths:

- Moving MNIST cache:
  - `data/video_temporal/generated/moving_mnist_medium/`
- Official VideoRoPE benchmark metadata:
  - `data/video_temporal/external/videorope_official/`

## Recommended Run Order

### Step 1: prepare local data

```bash
python3 scripts/data_prep/prepare_moving_mnist_video.py
python3 scripts/data_prep/prepare_videorope_assets.py
```

### Step 2: synthetic sweep on Blackwell

```bash
python3 scripts/video_temporal/run_video_temporal_allocation_sweep.py \
  --profile blackwell96 \
  --variants geo_k8,geo_k12,geo_k16,evq_k12,evq_k16 \
  --seeds 42,137
```

### Step 3: decide which official patch to build first

Decision rule:

- if `geo_k12/16 > geo_k8`, we have direct evidence that time needs more low-frequency budget
- if `evq_k12/16` further improves, the official V-NIAH-D patch should target **temporal inv_freq**, not just position spacing

## Expected Outcomes

Most likely:

1. `geo_k12` or `geo_k16` improves long-frame extrapolation over `geo_k8`
2. `evq_k12` matches or beats the best geo temporal-allocation variant
3. temporal YaRN helps both, but helps EVQ more at larger frame counts

If this happens, the paper narrative strengthens materially:

> VideoRoPE shows that temporal frequency allocation matters.
> Our controlled sweep shows that giving more low-frequency capacity to time is beneficial.
> EVQ then provides a closed-form way to shape that temporal subspace, analogous to the text result.

## Failure Cases

### If larger `K_t` does not help

Interpretation:

- the synthetic dataset is too easy
- or the benefit requires stronger distractors / retrieval pressure

Response:

- move faster to V-NIAH-D
- add distractor-heavy synthetic temporal retrieval

### If `K_t` helps but EVQ does not

Interpretation:

- the video claim may be about allocation quantity more than allocation shape

Response:

- still useful
- the paper can claim a shared mechanism at the budget level, with EVQ remaining strongest in text

### If EVQ only helps with temporal YaRN

Interpretation:

- this becomes a direct video analogue of the current text EVQ + YaRN story

Response:

- still highly valuable for Spotlight

## Status

Ready for launch once the Blackwell workstation is online.
