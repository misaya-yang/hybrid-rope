#!/usr/bin/env python3
"""Assert OLMo-2 1B and 7B share the identical tokenizer.json (V2 data is
packed with the 1B tokenizer path; the run trains the 7B)."""
import hashlib
import sys

p1 = "/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct/tokenizer.json"
p7 = "/root/autodl-tmp/models/OLMo-2-1124-7B-Instruct/tokenizer.json"
h1 = hashlib.sha256(open(p1, "rb").read()).hexdigest()
h7 = hashlib.sha256(open(p7, "rb").read()).hexdigest()
print("1B:", h1)
print("7B:", h7)
if h1 != h7:
    print("TOKENIZER_MISMATCH")
    sys.exit(1)
print("TOKENIZER_IDENTICAL")
