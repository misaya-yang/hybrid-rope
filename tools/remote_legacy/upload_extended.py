#!/usr/bin/env python3
from seetacloud_plink import run, upload_file_base64


def main() -> None:
    local_path = r"e:\rope\hybrid-rope\scripts\run_night_run_9h_extended.py"
    remote_path = "/root/autodl-tmp/dfrope/hybrid-rope/scripts/run_night_run_9h_extended.py"

    upload_file_base64(local_path, remote_path)

    # Quick verify.
    cp = run(f"wc -l {remote_path}", check=True)
    print("Verification (wc -l):")
    print(cp.stdout or "")
    print("Upload complete")


if __name__ == "__main__":
    main()

