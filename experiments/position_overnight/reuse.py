"""Reuse explicit, matched candidate runs without rerunning their generations."""
import json
from pathlib import Path


def candidate_rows(source, target, row_ids, arms, *, kind, row_keys=None):
    source = Path(source)
    if kind == "pc2":
        manifest, filename, field = "contract.json", "generations.jsonl", "selector"
        fields = ("model", "backend", "split", "dtype", "topk", "select_blocks",
                  "chunk_size", "attention_query_chunk_size", "data_sha256", "generation", "eos_ids")
        hash_field, modules = "source_hashes", ("runtime.py", "selector_controls.py")
        candidates = {"pc2", "pc2_unweighted", "pc2_rank1", "exact_mass"}
    elif kind == "pm":
        manifest, filename, field = "manifest.json", "per_example.jsonl", "arm"
        fields = ("model", "backend", "dtype", "config", "decode", "baseline_version", "task_protocol")
        hash_field, modules = "sources", ("adapter.py", "ops.py")
        candidates = {"P", "C", "U"}
        if row_keys is None:
            raise ValueError("PM reuse requires input/scoring fingerprints")
    else:
        raise ValueError(kind)
    previous = json.loads((source / manifest).read_text())
    for name in fields:
        if name not in target or previous.get(name) != target[name]:
            raise ValueError(f"Cannot reuse {source}: changed {name}")
    for name in modules:
        if not target[hash_field].get(name) or previous.get(hash_field, {}).get(name) != target[hash_field][name]:
            raise ValueError(f"Cannot reuse {source}: changed computation in {name}")
    allowed = candidates & set(arms)
    rows, seen = [], {}
    for line in (source / filename).read_text().splitlines():
        value = json.loads(line)
        key = (value["row_id"], value[field])
        if key[0] not in row_ids or key[1] not in allowed:
            continue
        if kind == "pc2" and key[1] == "exact_mass":
            exact_hash = target[hash_field].get("exact_probe.py")
            if not exact_hash or previous.get(hash_field, {}).get("exact_probe.py") != exact_hash:
                raise ValueError(f"Cannot reuse {source}: changed exact mass implementation")
        if row_keys is not None and value.get("baseline_cache_key") != row_keys.get(key):
            raise ValueError(f"Cannot reuse {source}: changed input/scoring for {key}")
        if key in seen:
            if seen[key] != value["generated_token_ids"]:
                raise ValueError(f"Conflicting saved generations for {key}")
            continue
        seen[key] = value["generated_token_ids"]
        rows.append({**value, "reused_candidate": True, "reused_from_run": str(source.resolve())})
    return rows


def append_missing(path, rows, arm_field):
    """Preserve original output/timing fields; imported rows are not new trials."""
    path = Path(path)
    existing = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    done = {(r["row_id"], r[arm_field]): r["generated_token_ids"] for r in existing}
    pending = []
    for row in rows:
        key = (row["row_id"], row[arm_field])
        if key in done and done[key] != row["generated_token_ids"]:
            raise ValueError(f"Conflicting candidate reuse for {key}")
        if key not in done:
            pending.append(row)
            done[key] = row["generated_token_ids"]
    if pending:
        with path.open("a") as stream:
            for row in pending:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(pending)
