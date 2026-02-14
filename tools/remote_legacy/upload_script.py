#!/usr/bin/env python3
"""
Upload the current local script to the server copy of this repo.

Default: scripts/run_llama_theta_matched.py
"""

from __future__ import annotations

import sys

from seetacloud_plink import upload_file_base64


def main() -> None:
    local_path = r"e:\rope\hybrid-rope\scripts\run_llama_theta_matched.py"
    remote_path = "/root/autodl-tmp/dfrope/hybrid-rope/scripts/run_llama_theta_matched.py"

    if len(sys.argv) == 3:
        local_path = sys.argv[1]
        remote_path = sys.argv[2]
    elif len(sys.argv) != 1:
        raise SystemExit("Usage: python upload_script.py [local_path remote_path]")

    upload_file_base64(local_path, remote_path)
    print("Upload complete")


if __name__ == "__main__":
    main()

