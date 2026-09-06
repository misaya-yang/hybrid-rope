#!/usr/bin/env python3
"""Download small official VideoRoPE assets needed for local planning and eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


DATASET_REPO = "Wiselnn/VideoRoPE"
MODEL_REPO = "Wiselnn/Qwen2-VL-videorope-128frames-8k-context-330k-llava-video"
BASE_MODEL_REPO = "Qwen/Qwen2-VL-7B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare lightweight official VideoRoPE assets.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/video_temporal/external/videorope_official"),
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        allow_patterns=[
            "README.md",
            "vision_niah_d/needle_datasets/*",
            "vision_niah_d/needle_datasets/images/*",
            "vision_niah_d/niah_output/*/avg_accuracy.txt",
            "vision_niah_d/niah_output/*/all_accuracies.json",
        ],
        local_dir=out_dir / "dataset",
        local_dir_use_symlinks=False,
    )

    manifest = {
        "dataset_repo": DATASET_REPO,
        "model_repo": MODEL_REPO,
        "base_model_repo": BASE_MODEL_REPO,
        "local_dataset_dir": str(Path(dataset_dir)),
        "notes": {
            "official_metric_benchmark": "V-NIAH-D",
            "official_rope_modes": ["vanilla_rope", "tad_rope", "m_rope", "videorope"],
            "recommended_eval_model": MODEL_REPO,
            "recommended_ablation_model": BASE_MODEL_REPO,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
