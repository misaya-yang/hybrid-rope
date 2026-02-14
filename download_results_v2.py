#!/usr/bin/env python3
"""
Download night_run_anchored_x20_9h artifacts from the server.

Requires:
  - $env:SEETACLOUD_SSH_PW set (see seetacloud_plink.py)
"""

from __future__ import annotations

from pathlib import Path

from seetacloud_plink import download_file_base64


def main() -> None:
    local_dir = Path(r"e:/rope/hybrid-rope/results/night_run_anchored_x20_9h")
    local_dir.mkdir(parents=True, exist_ok=True)

    remote_dir = "/root/autodl-tmp/dfrope/hybrid-rope/results/night_run_anchored_x20_9h"
    files = [
        ("results.json", "results.json"),
        ("summary.md", "summary.md"),
        ("run.log", "run.log"),
    ]

    for remote_name, local_name in files:
        download_file_base64(f"{remote_dir}/{remote_name}", local_dir / local_name)
        print(f"Downloaded {local_name}")

    print("Done!")


if __name__ == "__main__":
    main()

