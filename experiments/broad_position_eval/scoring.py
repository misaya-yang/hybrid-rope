"""Task-specific full-generation metrics for the frozen broader panel."""
from __future__ import annotations

import hashlib
from pathlib import Path

from experiments.pm_keep.run import score as original_score
from experiments.refcarry_audit.score_mrcr_pairs import score_text
from .prepare import MRCR

VERSION = "broad_scores_v1_" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def score(row, generated, tokenizer, eos_ids):
    if row["score_contract"] == MRCR:
        ended = bool(generated and generated[-1] in eos_ids)
        body = generated[:-1] if ended else generated
        text = tokenizer.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        details = score_text(text, ended, row)
        result = {**details, "output_text": text,
                  "output_text_with_special_tokens": tokenizer.decode(generated, skip_special_tokens=False,
                                                                        clean_up_tokenization_spaces=False),
                  "ended_with_eos": ended, "exact_string": details["whole_string_exact"],
                  "exact_plus_eos": details["full_exact_and_eos"]}
    else:
        result = original_score(row, generated, tokenizer, eos_ids)
    result.update(material_cluster_id=row["material_cluster_id"],
                  family_id=row["family_id"], background=row.get("background"),
                  record_count=row.get("record_count"), scoring_version=VERSION,
                  answer_limit=row["max_new_tokens"], length_cap=row.get("length_cap"), source=row.get("source"))
    return result


def score_nosa(row, generated, tokenizer, eos_ids, *, topk=64):
    result = score(row, generated, tokenizer, eos_ids)
    active = len(row["prompt_ids"]) > topk * 64  # fixed existing NOSA block_size=64
    result.update(routing_active_during_prefill=active,
                  original_panel_task=row["task"],
                  task=row["task"] + ("" if active else "__short_prefill_control"))
    return result
