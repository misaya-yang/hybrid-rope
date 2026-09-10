"""Fetch only the two required public LongBench task files, with provenance."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile


def main():
    from huggingface_hub import HfApi, hf_hub_download
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info("THUDM/LongBench", timeout=30)
    path = Path(hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset",
                                revision=info.sha, local_dir=root))
    with zipfile.ZipFile(path) as archive:
        for task in ("hotpotqa", "2wikimqa"):
            members = [name for name in archive.namelist() if name == task + ".jsonl" or name.endswith("/" + task + ".jsonl")]
            if len(members) != 1:
                raise ValueError(f"ambiguous source for {task}: {members}")
            (root / (task + ".jsonl")).write_bytes(archive.read(members[0]))
    receipt = {"repo": "THUDM/LongBench", "revision": info.sha,
               "zip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                         for name in ("hotpotqa.jsonl", "2wikimqa.jsonl")}}
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
