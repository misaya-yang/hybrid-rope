# Video Temporal Blackwell Plan

## Goal

Use the local `R6000 Blackwell 96GB` workstation to open a medium-cost video line that is cleaner than more undertrained text scaling runs.

The target claim is:

> The temporal axis benefits from a larger low-frequency budget, and EVQ-Cosh is a principled way to shape that temporal budget.

This is intended to strengthen the poster/spotlight package without replacing the current text core.

## Starting Point

The current text package is already strong enough for the poster:

- fixed-length text: EVQ slightly loses short-range and strongly wins long extrapolation
- EVQ + YaRN: super-additive gains
- downstream support: gold-answer NLL and NIAH-style retrieval

The remaining weakness is not the core text conclusion. It is the lack of a larger, cleaner, less undertrained cross-modal anchor.

## Why VideoRoPE

VideoRoPE is the right bridge because the official method already isolates temporal frequency allocation as a first-class issue.

Primary sources:

- [VideoRoPE official repo](https://github.com/Wiselnn570/VideoRoPE)
- [VideoRoPE arXiv page](https://arxiv.org/abs/2502.05173)
- [Official model collection](https://huggingface.co/collections/Wiselnn/videorope-what-makes-for-good-video-rotary-position-embeddi-67ca90664c8e169422449c56)
- [Official dataset repo](https://huggingface.co/datasets/Wiselnn/VideoRoPE)

Three official ideas matter for us:

- Low-frequency Temporal Allocation (LTA)
- Diagonal Layout (DL)
- Adjustable Temporal Spacing (ATS)

These are directly adjacent to our frequency-allocation story.

## Official Baseline Signal

I downloaded the lightweight official V-NIAH-D result bundle and summarized the published accuracies from the official asset pack:

- `vanilla_rope`: `0.3178`
- `tad_rope`: `0.2933`
- `m_rope`: `0.7867`
- `videorope`: `0.9111`

These come from:

- [videorope_official manifest](REPO_ROOT/data/video_temporal/external/videorope_official/manifest.json)
- [baseline summary script](REPO_ROOT/scripts/video_temporal/summarize_videorope_official_results.py)

The gap is large enough that “temporal frequency allocation matters” is already an externally anchored premise, not just our own intuition.

## Repo Findings

We already had one local synthetic video script:

- [run_video_temporal.py](REPO_ROOT/scripts/video_temporal/run_video_temporal.py)

That script proves the codebase already has:

- a 3D-RoPE `VideoGPT`
- temporal EVQ-Cosh support
- raw and temporal-YaRN evaluation hooks

What it did not have was a systematic temporal-allocation sweep aligned to the VideoRoPE question.

## Key Technical Constraint

Official VideoRoPE V-NIAH-D modes mostly manipulate temporal layout through position-id construction.

From the official code:

- Qwen2-style rotary code computes `position_ids.float()`
- the benchmark changes temporal indices for rope modes such as `vanilla_rope`, `m_rope`, and `videorope`

But a clean EVQ patch is not equivalent to a single temporal spacing scalar. EVQ is channel-wise. That means the correct bridge is:

- modify the temporal slice of `inv_freq`
- not merely rescale one temporal index sequence

This is why the first run should be a controlled synthetic sweep, not an immediate invasive Qwen2-VL patch.

## Prepared Assets

### Local synthetic dataset

Prepared with:

- [prepare_moving_mnist_video.py](REPO_ROOT/scripts/data_prep/prepare_moving_mnist_video.py)

Local cache:

- [moving_mnist_medium manifest](REPO_ROOT/data/video_temporal/generated/moving_mnist_medium/manifest.json)

Configuration:

- `16000` train videos
- `2000` val videos
- `2000` test videos
- training context: `32` frames
- eval context: up to `128` frames
- `64x64` frames, `8x8` patches
- `2048` train tokens/video
- `8192` eval tokens/video

### Official bridge assets

Prepared with:

- [prepare_videorope_assets.py](REPO_ROOT/scripts/data_prep/prepare_videorope_assets.py)

Local asset root:

- [videorope_official](REPO_ROOT/data/video_temporal/external/videorope_official)

This includes:

- V-NIAH-D needle metadata
- needle images
- official average-accuracy outputs for published rope modes

## New Experiment Code

### Main sweep

- [run_video_temporal_allocation_sweep.py](REPO_ROOT/scripts/video_temporal/run_video_temporal_allocation_sweep.py)

What it does:

- trains a 3D-RoPE `VideoGPT` on tokenized Moving MNIST
- sweeps temporal channel budget and EVQ temporal shaping
- evaluates both raw and temporal-YaRN extrapolation

Variants:

- `geo_k8`
- `geo_k12`
- `geo_k16`
- `evq_k12`
- `evq_k16`

Interpretation:

- `geo_k8 -> geo_k12 -> geo_k16` isolates whether time needs more low-frequency budget
- `geo_k12 -> evq_k12` isolates EVQ shaping at fixed temporal budget
- `geo_k16 -> evq_k16` tests whether EVQ still helps when the budget is already generous

### One-command launcher

- [run_phase23_blackwell.sh](REPO_ROOT/scripts/video_temporal/run_phase23_blackwell.sh)
- [run_phase23_blackwell_10h.sh](REPO_ROOT/scripts/video_temporal/run_phase23_blackwell_10h.sh)

Default run:

- profile: `blackwell96`
- seeds: `42,137`
- variants: all five temporal-allocation arms
- output root: `results/supporting_video/phase23_video_temporal_blackwell/<timestamp>/`

Recommended first launch:

- the `10h` script, not the full sweep
- pass 1: `geo_k8`, `geo_k16`, `evq_k16` at `seed=42`
- pass 2: `geo_k16`, `evq_k16` at `seed=137`
- `16` epochs instead of `24`
- `16` eval chunks instead of `24`

Reason:

- this is the best value-density point
- it answers both key questions without paying for the full five-arm grid
- it still includes one replication on the decisive high-budget head-to-head

## Recommended Run Order

### Step 1

Summarize the downloaded official baselines once:

```bash
python3 scripts/video_temporal/summarize_videorope_official_results.py
```

### Step 2

Launch the controlled Blackwell sweep:

```bash
bash scripts/video_temporal/run_phase23_blackwell_10h.sh
```

If the first 10-hour run is clearly positive, then expand to the full five-arm sweep:

```bash
bash scripts/video_temporal/run_phase23_blackwell.sh
```

### Step 3

Inspect the final aggregate:

```bash
cat results/supporting_video/phase23_video_temporal_blackwell/<timestamp>/summary.json
```

## Success Criteria

The cheapest decisive outcomes are:

1. `geo_k12` or `geo_k16` beats `geo_k8` at `64f+`
2. `evq_k12` or `evq_k16` further beats the corresponding geo arm
3. temporal YaRN helps both, but helps EVQ more at longer frame counts

If all three happen, the poster story becomes much stronger:

> VideoRoPE shows temporal allocation matters.
> Our controlled sweep shows larger low-frequency temporal budget helps.
> EVQ then provides a closed-form way to shape that temporal subspace, analogous to the text result.

## Failure Interpretation

If larger `K_t` helps but EVQ does not:

- the shared mechanism may be about allocation quantity more than allocation shape

If EVQ helps only after temporal YaRN:

- that is still valuable, because it becomes a direct video analogue of the text EVQ + YaRN story

If nothing helps:

- the synthetic dataset is too easy
- then the next move is to shift budget from synthetic training to direct V-NIAH-D patching

## Status

Ready for launch.

Validated locally before handoff:

- both data-prep scripts compile and run
- `moving_mnist_medium` cache is built
- official VideoRoPE benchmark assets are downloaded
- the Blackwell sweep passes a dry-run using the prepared cache
