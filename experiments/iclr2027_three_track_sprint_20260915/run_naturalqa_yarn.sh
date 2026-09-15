#!/usr/bin/env bash
# A1: one YaRN arm on the unchanged QA631 panel. Separate from existing queues.
set -euo pipefail
if [[ "${1:-}" != --execute ]]; then
  echo 'A1 prepared, not launched. After assigning an idle GPU and authorizing 631 generations:'
  echo 'bash experiments/iclr2027_three_track_sprint_20260915/run_naturalqa_yarn.sh --execute'
  echo 'Reuses completed T/P; writes a YaRN run and new family-adjusted reports.'
  exit 0
fi
repo_dir=$(cd "$(dirname "$0")/../.." && pwd)
plan=${ROPE_PLAN_ROOT:-/root/autodl-tmp/today_rope_plan_20260914}
model=${ROPE_MODEL_PATH:-/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct}
python_bin=${ROPE_PYTHON:-/root/miniconda3/bin/python}
natural=${plan}/tailspline_llama_s4_naturalqa631
classic=${plan}/tailspline_llama_s4_classic
panel=${natural}/assets/inputs.jsonl
yarn=${plan}/tailspline_llama_s4_classic_strong_baselines/tables/yarn.json
out=${plan}/tailspline_llama_s4_naturalqa631_yarn_a1
cd "${repo_dir}"
export PYTHONPATH=.
# Validate existing assets only: no download, retokenization, calibration or T/P run.
"${python_bin}" - "${natural}" "${panel}" "${yarn}" <<'PY'
import collections,json,sys
from pathlib import Path
root,panel,table=map(Path,sys.argv[1:]);rows=[json.loads(x)for x in panel.read_text().splitlines() if x]
assert len(rows)==631 and len({r['row_id']for r in rows})==631
assert sum(r['input_tokens']>8192 for r in rows)==315
assert len({r['document_cluster_id']for r in rows})==524
assert sum(r['max_new_tokens']for r in rows)==41056
receipt=json.loads(table.read_text());yt=receipt.get('table',receipt)
contracts=[]
for arm in ['tailspline','mrpro']:
 run=root/'runs'/arm
 assert json.loads((run/'status.json').read_text())=={'status':'COMPLETE','rows':631,'lm_rows':0}
 outputs=[json.loads(x)for x in (run/'generations.jsonl').read_text().splitlines()if x]
 assert len(outputs)==631 and {r['prompt_sha256']for r in outputs}=={r['prompt_sha256']for r in rows}
 c=json.loads((run/'contract.json').read_text());contracts.append(c)
 assert c['batch_size']==1 and not c.get('left_pad_batches',False)
 assert c['static_table']['gain']==yt['gain']
 assert c['static_table']['values_float32'][0]==yt['values_float32'][0]
 assert c['static_table']['values_float32'][-1]==yt['values_float32'][-1]
assert all(contracts[0][k]==contracts[1][k]for k in contracts[0]if k not in ['arm','static_table'])
print(json.dumps({'rows':631,'input_tokens':sum(r['input_tokens']for r in rows),'maximum_generated_tokens':41056,'reused':['tailspline','mrpro']}))
PY
mkdir -p "${out}/reports"
if [[ ! -f "${out}/runs/yarn/status.json" ]] || ! "${python_bin}" - "${out}/runs/yarn/status.json" <<'PY_CHECK'
import json,sys
raise SystemExit(json.load(open(sys.argv[1])) != {'status':'COMPLETE','rows':631,'lm_rows':0})
PY_CHECK
then
"${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
  --data "${classic}/assets/ppl46/manifest.json" --model "${model}" --arm Native \
  --extra-panel "${panel}" --only-extra-panels --skip-lm --length-cap 32768 \
  --prefill-chunk-size 8192 --batch-size 1 --static-table-json "${yarn}" \
  --table-label llama3_8b_s4_naturalqa631_yarn --out "${out}/runs/yarn" --execute
fi
for baseline in mrpro yarn; do
  baseline_file=${natural}/runs/mrpro/generations.jsonl
  if [[ "${baseline}" == yarn ]]; then baseline_file=${out}/runs/yarn/generations.jsonl; fi
  "${python_bin}" -m experiments.fixed_rope_three_interfaces_20260913.matched_naturalqa_report \
    --panel "${panel}" --candidate "${natural}/runs/tailspline/generations.jsonl" \
    --baseline "${baseline_file}" --baseline-name "${baseline}" --comparison-family-size 2 \
    --out "${out}/reports/tailspline_vs_${baseline}_family2.json"
done
