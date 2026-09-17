"""Frozen Llama-3-70B NF4 S4/32K scale-transfer evaluation."""

RULER_TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2",
)
NIAH_TASKS = RULER_TASKS[:8]
QA_TASKS = ("hotpotqa", "2wikimqa", "qasper", "narrativeqa", "multifieldqa_en")
MODEL_ID = "llama3_70b_instruct_bnb_nf4"
TARGET_LENGTH = 32768
SCALE = 4
RULER_ROWS_PER_TASK = 10
NIAH_ROWS_PER_TASK = 5
PPL_DOCUMENTS = 5
QA_ROWS = 631

