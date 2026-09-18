#!/usr/bin/env python3
"""Export dirty Git-visible files for continuation in an isolated base worktree.

Does not commit, stage, push, copy .git, or collect ignored model/raw files.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def export(output: Path) -> dict:
    names = set(filter(None, git("diff", "--name-only", "HEAD", "-z").decode().split("\0")))
    names.update(filter(None, git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")))
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing export: {output}")
    manifest = {
        "schema": "hybrid_rope_continuation_overlay_v1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_commit": git("rev-parse", "HEAD").decode().strip(),
        "branch_hint": git("branch", "--show-current").decode().strip(),
        "scope": "All current modified/new Git-visible working-tree files; not a commit or an experiment authorization.",
        "files": [], "deleted_paths": [],
    }
    payload = {}
    for name in sorted(names):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] == ".git":
            raise ValueError(f"Unsafe repository path: {name}")
        path = ROOT / relative
        if path.is_symlink():
            raise ValueError(f"Review symlink explicitly before exporting: {name}")
        if not path.exists():
            manifest["deleted_paths"].append(name)
            continue
        if not path.is_file():
            raise ValueError(f"Not a regular file: {name}")
        if path.resolve() == output:
            raise ValueError("Export destination is an input path")
        data = path.read_bytes()
        payload[name] = data
        manifest["files"].append({"path": name, "size": len(data),
                                  "sha256": hashlib.sha256(data).hexdigest(),
                                  "mode": oct(path.stat().st_mode & 0o777)})
    # Catch concurrent edits instead of silently claiming a consistent snapshot.
    for record in manifest["files"]:
        if hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError(f'File changed during export: {record["path"]}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for record in manifest["files"]:
            info = zipfile.ZipInfo(record["path"])
            info.create_system = 3
            info.external_attr = int(record["mode"], 8) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload[record["path"]])
        archive.writestr("CONTINUATION_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
        archive.writestr("CONTINUATION_README.txt", (
            "Hybrid-RoPE continuation overlay\n\n"
            "Read CONTINUATION_MANIFEST.json. This overlay needs an existing Git clone\n"
            "with the exact base commit; it does not include Git history. Preserve PC\n"
            "dirty work. Create an isolated worktree at the manifest base, then extract\n"
            "these files there. Inspect deleted_paths separately; extraction never\n"
            "deletes them automatically. Inspect git status and verify file hashes.\n\n"
            "Start at index.md and docs/maintenance/CROSS_MACHINE_CONTINUATION.md.\n"
            "Run python scripts/check_repository_docs.py. Repository skills are in\n"
            ".agents/skills. No model runs or old queues are authorized by this archive.\n"
        ))
    with zipfile.ZipFile(output) as archive:
        for record in manifest["files"]:
            assert hashlib.sha256(archive.read(record["path"])).hexdigest() == record["sha256"]
    return {"output": str(output), "base_commit": manifest["base_commit"],
            "files": len(manifest["files"]), "deleted_paths": manifest["deleted_paths"],
            "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New ZIP path, preferably under internal/local_snapshots/")
    print(json.dumps(export(parser.parse_args().output), indent=2))
