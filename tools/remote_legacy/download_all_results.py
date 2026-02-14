#!/usr/bin/env python3
"""
Download multiple result directories from the server without pscp/scp.

Implementation: tar.gz -> base64 over plink -> extract locally.

Requires:
  - $env:SEETACLOUD_SSH_PW set (see seetacloud_plink.py)
"""

from __future__ import annotations

import tarfile
from pathlib import Path

from seetacloud_plink import download_dir_tar_gz_base64


RESULTS_DIRS = [
    "night_run_anchored_x20_9h",
    "night_run_9h_extended",
    "anchored_sigmoid_v3_followup",
    "advisor_followup_2026-02-14",
    "qwen_hybrid_lora",
    "evidence_chain_50m_3cfg3seed",
    "cross_model_wikitext_v1",
]


def main() -> None:
    repo_root = Path(r"e:/rope/hybrid-rope")
    local_results = repo_root / "results"
    local_results.mkdir(parents=True, exist_ok=True)

    remote_results_root = "/root/autodl-tmp/dfrope/hybrid-rope/results"

    for d in RESULTS_DIRS:
        print(f"Downloading {d}...")
        remote_dir = f"{remote_results_root}/{d}"
        tar_path = local_results / f"{d}.tar.gz"

        try:
            download_dir_tar_gz_base64(remote_dir, tar_path)
        except Exception as exc:
            print(f"  Failed to download {d}: {exc}")
            continue

        try:
            with tarfile.open(tar_path, "r:gz") as tf:
                # Extract into results/{d}/...
                out_dir = local_results / d
                out_dir.mkdir(parents=True, exist_ok=True)
                tf.extractall(path=out_dir)
        finally:
            try:
                tar_path.unlink()
            except OSError:
                pass

        print(f"  Extracted to {local_results / d}")

    print("\nDone!")


if __name__ == "__main__":
    main()

