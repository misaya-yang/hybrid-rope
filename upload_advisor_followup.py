#!/usr/bin/env python3
from seetacloud_plink import upload_file_base64


def main() -> None:
    local_path = r"e:\rope\hybrid-rope\scripts\run_advisor_followup.py"
    remote_path = "/root/autodl-tmp/dfrope/hybrid-rope/scripts/run_advisor_followup.py"
    upload_file_base64(local_path, remote_path)
    print("Upload complete")


if __name__ == "__main__":
    main()

