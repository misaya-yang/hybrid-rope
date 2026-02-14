#!/usr/bin/env python3
"""
Download anchored_sigmoid_v3_followup artifacts from the server.

Requires:
  - $env:SEETACLOUD_SSH_PW set (see seetacloud_plink.py)
"""

from __future__ import annotations

from pathlib import Path

from seetacloud_plink import download_file_base64


def main() -> None:
    remote_base = "/root/autodl-tmp/dfrope/hybrid-rope/results/anchored_sigmoid_v3_followup"
    local_dir = Path(r"e:/rope/hybrid-rope/results/anchored_sigmoid_v3_followup")
    local_dir.mkdir(parents=True, exist_ok=True)

    files = [
        ("exp1_robustness/results.json", "exp1_results.json"),
        ("exp2_theta_substitution/results.json", "exp2_results.json"),
        ("exp3_anchor_ablation/results.json", "exp3_results.json"),
        ("summary.md", "summary.md"),
    ]

    for remote_rel, local_name in files:
        remote_path = f"{remote_base}/{remote_rel}"
        local_path = local_dir / local_name
        try:
            download_file_base64(remote_path, local_path)
            print(f"Downloaded: {local_name}")
        except Exception as exc:
            print(f"Failed: {remote_rel} ({exc})")

    print("\nDone!")


if __name__ == "__main__":
    main()

