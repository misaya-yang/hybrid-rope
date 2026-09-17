"""Single source of truth for the first A/B execution waves."""
from pathlib import Path

MODEL = Path("/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct")
PLAN = Path("/root/autodl-tmp/today_rope_plan_20260914")
PANEL = PLAN / "native_research_20260916/assets/ruler_confirm_13x10/panels/4096/inputs.jsonl"
DATA = PLAN / "tailspline_olmo_s4_classic/assets/ppl46/manifest.json"
NCP_TABLE = PLAN / "olmo_native_contrastive_proximal/tables/ncp.json"
CURRENT_REPORT = PLAN / "ca_ncp_safe_followup_20260917/reports/paired_report.json"
ROOT = PLAN / "native_followup_five_20260917"

PANEL_SHA256 = "aeded87f6afd40ac6fcdaaae77f5f79e20851dceddac03f4b16111e6fea3fc1a"
NCP_TABLE_SHA256_FLOAT32 = "54b9dd1f73aafc69f7bb5ed1b7b49d49128002371cb378d03ca1fd1d108e0cb7"
NATIVE_SCORE = 0.7352564102564103
NCP_SCORE = 0.7678205128205128

WAVE1 = ("mass_projection", "mass_raw", "even_only")
WAVE2 = ("odd_only", "rank_assignment")
