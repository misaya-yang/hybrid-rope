#!/usr/bin/env python3
"""Compare raw-mode Track A receipts vs chat-template audit cells (OLMo)."""
import json
import sys

B12 = "/root/autodl-tmp/claude_round12_20260906"
model = sys.argv[1] if len(sys.argv) > 1 else "olmo1b"


def cells(p):
    m = json.load(open(p + "/manifest.json"))
    return {k: v for k, v in m["summary"].items()
            if k.startswith("single_evidence")}


def fmt(r):
    return (f"sg {r['strict_groups']}/{r['groups']} f1 {r['qa_f1_mean']} "
            f"eos {r['eos_rate']} len {r['lenient_contains']}/{r['rows']}")


rawN = cells(f"{B12}/track_a/{model}_N")
rawZ = cells(f"{B12}/track_a/{model}_Z")
try:
    chN = cells(f"{B12}/track_a_audit/{model}_N_chat")
except FileNotFoundError:
    chN = {}
try:
    chZ = cells(f"{B12}/track_a_audit/{model}_Z_chat")
except FileNotFoundError:
    chZ = {}

for k in sorted(rawN):
    print(k)
    print("  N raw :", fmt(rawN[k]))
    print("  N chat:", fmt(chN[k]) if k in chN else "  (pending)")
    print("  Z raw :", fmt(rawZ[k]))
    print("  Z chat:", fmt(chZ[k]) if k in chZ else "  (pending)")
