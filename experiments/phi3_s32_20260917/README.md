# Phi-3-mini-4K S=32 gate

This directory owns the frozen TailSpline `S=32` RULER gate requested for
`microsoft/Phi-3-mini-4k-instruct`.

- The first evaluation is clean, source-order, unpadded Full-13x10 at 16K.
- A task-equal official score below 80% writes the complete report and authorizes
  the server shutdown requested for this run.
- A score of at least 80% prepares and evaluates the same 13x10 protocol at 32K.
- A 32K score below 80% also shuts down after preserving the report; a 32K pass
  leaves an explicit hold receipt for the user instead of inventing another length.

`queue.py` repairs an interrupted ModelScope download and invokes `run_gate.py`
only after the two indexed safetensor shards have their exact recorded total
size. Both commands are PLAN_ONLY unless passed `--execute`.

The follow-up `S=4` compatibility check uses `--scale 4 --stop-after-16k` and
never requests shutdown. It reuses the same tokenizer-frozen 16K panel but
writes to a separate experiment root and table receipt.
