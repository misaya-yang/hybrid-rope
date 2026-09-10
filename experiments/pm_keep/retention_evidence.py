"""Offline source-span audit of already-frozen PM/KeyDiff keep sets.

This inspects the future_query_probe trace. It does not select a cache, load
model weights, or generate answers. Source-token retention is evidence about
the intervention, not proof that information is absent from other hidden states.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def token_span(offsets, start, end):
    indices = [i for i, (left, right) in enumerate(offsets)
               if right > left and left < end and right > start]
    if not indices:
        raise ValueError("source span has no overlapping tokens")
    if min(offsets[i][0] for i in indices) > start or max(offsets[i][1] for i in indices) < end:
        raise ValueError("source span is not fully covered by the prefix")
    return indices


def record_spans(row, offsets):
    text = row["prefix_text"]
    if len(offsets) != len(row["prefix_ids"]):
        raise ValueError("offsets must describe exactly the frozen prefix")
    keys, ordinals = row["query"]["keys"], row["query"]["ordinals"]
    if len(keys) != len(ordinals) or len(set(keys)) != len(keys):
        raise ValueError("audit requires one specified ordinal per distinct query key")
    requested = dict(zip(keys, ordinals))
    occurrences, output = {}, []
    ordered = sorted(row["records"], key=lambda r: text.find(f"Record key={r['key']}; value={r['value']}."))
    for record in ordered:
        key, value = record["key"], record["value"]
        literal = f"Record key={key}; value={value}."
        start = text.find(literal)
        if start < 0 or text.find(literal, start + 1) >= 0:
            raise ValueError("each declared record must identify one source occurrence")
        occurrences[key] = occurrences.get(key, 0) + 1
        key_start = start + len("Record key=")
        value_start = key_start + len(key) + len("; value=")
        output.append({"key": key, "value": value,
            "target": requested.get(key) == occurrences[key],
            "record": token_span(offsets, start, start + len(literal)),
            "key_tokens": token_span(offsets, key_start, key_start + len(key)),
            "value_tokens": token_span(offsets, value_start, value_start + len(value))})
    if sum(r["target"] for r in output) != len(requested):
        raise ValueError("query target occurrence is absent from the prefix records")
    return output


def inspect_keep_sets(keep_sets, spans, prefix_length):
    reports = {}
    for arm, layers in keep_sets.items():
        layer_reports = []
        for layer_index, selected in enumerate(layers):
            selected = torch.as_tensor(selected, dtype=torch.long).cpu()
            if selected.ndim != 2 or not selected.shape[1]:
                raise ValueError("keep sets must be nonempty [KV heads, slots]")
            if bool(((selected < 0) | (selected >= prefix_length)).any()):
                raise ValueError("keep sets must reference original prefix slots")
            if selected.shape[1] > 1 and not bool((selected[:, 1:] > selected[:, :-1]).all()):
                raise ValueError("keep sets must be sorted and unique")
            mask = torch.zeros((selected.shape[0], prefix_length), dtype=torch.bool)
            mask.scatter_(1, selected, True)
            records = []
            for span in spans:
                entry = {k: span[k] for k in ("key", "value", "target")}
                for name in ("record", "key_tokens", "value_tokens"):
                    present = mask[:, span[name]]
                    entry[name] = {"token_count": len(span[name]),
                        "retained_per_kv_head": present.sum(-1).tolist(),
                        "all_retained_per_kv_head": present.all(-1).tolist()}
                records.append(entry)
            layer_reports.append({"layer": layer_index, "heads": selected.shape[0],
                                  "slots_per_head": selected.shape[1], "records": records})
        reports[arm] = layer_reports
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("preserve the existing report; choose a new output")
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    row = next(r for r in rows if r["row_id"] == args.row_id)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    encoded = tokenizer(row["prefix_text"], add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != row["prefix_ids"]:
        raise ValueError("retokenized prefix differs from the frozen model input")
    trace = torch.load(args.trace, map_location="cpu", weights_only=True)
    spans = record_spans(row, encoded["offset_mapping"])
    report = {"row_id": row["row_id"], "task": row["task"],
        "trace_path": str(Path(args.trace).resolve()), "exact_prefix_ids_verified": True,
        "prefix_length": len(row["prefix_ids"]), "source_spans": spans,
        "retention": inspect_keep_sets(trace["keep_indices"], spans, len(row["prefix_ids"])),
        "scope": "offline inspection of previously frozen choices; source retention is not a causal proof of answer recovery or absence of information in other hidden states"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(str(output))


if __name__ == "__main__":
    main()
