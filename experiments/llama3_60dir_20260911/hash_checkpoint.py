"""Create a one-time SHA-256 identity manifest for the frozen local checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(32 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    model = args.model.resolve()
    names = ["config.json", "generation_config.json", "tokenizer.json",
             "tokenizer_config.json", "model.safetensors.index.json"]
    names.extend(path.name for path in sorted(model.glob("model-*.safetensors")))
    missing = [name for name in names if not (model / name).is_file()]
    if missing or len([x for x in names if x.endswith(".safetensors")]) != 4:
        raise SystemExit(f"REFUSING: incomplete checkpoint files: {missing}")
    files = {}
    for name in names:
        path = model / name
        files[name] = {"bytes": path.stat().st_size, "sha256": sha_file(path)}
    report = {"status": "COMPLETE", "model": str(model), "files": files,
              "total_bytes": sum(x["bytes"] for x in files.values()),
              "completed_at": time.time(), "script_sha256": sha_file(__file__)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(args.out)
    print(json.dumps({"status": "COMPLETE", "files": len(files),
                      "total_bytes": report["total_bytes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
