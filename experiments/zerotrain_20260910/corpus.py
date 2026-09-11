"""Corpus loading for the screen, and the one rule that makes it valid.

THE SCORED POSITIONS MUST LIE BEYOND THE NATIVE WINDOW.  `long_nll` is the NLL
of the last `keep` targets of a document of length L > window + keep, so every
scored position is one the checkpoint never saw at that relative distance during
training.  That is the whole content of "long range": if the document fits inside
the window the metric is an in-window number and the screen would be ranking
tables on the thing they are not supposed to change.

Two sources are accepted, both deliberately dumb:

  * a JSONL whose rows carry `ids` (a token list) -- the form the server's
    prepared panels already use, so a panel can be screened without
    re-tokenising;
  * a JSONL whose rows carry `text` plus a tokenizer, for a document pulled from
    a local corpus.

Documents are loaded WHOLE and never truncated to fit: a shorter document would
silently change which positions are scored, and the failure is invisible because
the metric still returns a number.  A document longer than `max_tokens` is
refused for the same reason -- trimming the head would move every scored position
closer to the start.
"""
from __future__ import annotations

import json

import numpy as np
import torch


def load_docs(path, tokenizer=None, max_tokens=None, limit=None, prepend="\n\n"):
    """-> list of (1, L) int64 tensors.  Raises rather than trimming."""
    docs = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "ids" in rec:
                ids = [int(x) for x in rec["ids"]]
            elif "text" in rec:
                if tokenizer is None:
                    raise ValueError(f"{path}:{lineno} carries text but no "
                                     "tokenizer was supplied")
                ids = list(tokenizer(prepend + rec["text"],
                                     add_special_tokens=False)["input_ids"])
            else:
                raise ValueError(f"{path}:{lineno} has neither `ids` nor `text`; "
                                 f"keys are {sorted(rec)}")
            if not ids:
                raise ValueError(f"{path}:{lineno} tokenised to nothing")
            if max_tokens is not None and len(ids) > max_tokens:
                raise ValueError(
                    f"{path}:{lineno} is {len(ids)} tokens, over the {max_tokens} "
                    "cap. Refusing rather than truncating: cutting tokens moves "
                    "every scored position and the metric would still return a "
                    "number, so the failure would be invisible.")
            docs.append(torch.tensor(np.asarray(ids, dtype=np.int64)[None, :]))
            if limit is not None and len(docs) >= int(limit):
                break
    if not docs:
        raise ValueError(f"{path} yielded no documents")
    lengths = [int(d.shape[1]) for d in docs]
    return dict(docs=docs, lengths=lengths, path=str(path),
                n_docs=len(docs), min_length=min(lengths),
                max_length=max(lengths),
                uniform=bool(len(set(lengths)) == 1))


def describe(corp, keep, window):
    """What the corpus is, in the terms the metric depends on."""
    return dict(**{k: v for k, v in corp.items() if k != "docs"},
                keep=int(keep), native_window=int(window),
                scored_positions_all_beyond_window=bool(
                    corp["min_length"] > window + keep),
                note=("every scored position must be beyond the native window; "
                      "if this flag is false the long_nll metric is an "
                      "in-window number and the screen ranks tables on the "
                      "quantity they are not meant to change"))
