#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--execute" ]]; then
  printf '%s\n' 'PLAN_ONLY: OLMo S4 Natural-QA631, TailSpline then canonical MrPro.'
  printf '%s\n' 'A frozen four-row canary selects exact batch4 left-pad or falls back to batch1.'
  printf '%s\n' 'Run with --execute only when the current GPU task has completed.'
  exit 0
fi

repo=/root/autodl-tmp/hybrid-rope
plan=/root/autodl-tmp/today_rope_plan_20260914
root=${plan}/tailspline_olmo_s4_naturalqa631
classic=${plan}/tailspline_olmo_s4_classic
model=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
python_bin=/root/miniconda3/bin/python
panel=${root}/assets/inputs.jsonl

mkdir -p "${root}/assets" "${root}/runs" "${root}/logs" "${root}/reports"
cd "${repo}"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

gpu_lock=/tmp/hybrid-rope-gpu0.lock
exec 9>"${gpu_lock}"
if ! flock -n 9; then
  printf 'REFUSE: another strong-evidence job owns %s\n' "${gpu_lock}" >&2
  exit 1
fi
active_compute=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
if [[ -n "${active_compute}" ]]; then
  printf 'REFUSE: GPU already has an active compute process: %s\n' "${active_compute}" >&2
  exit 1
fi

"${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.prepare_olmo_naturalqa631 \
  --frozen-root /root/autodl-tmp/olmo_fast_screen_20260908 \
  --download-root /root/autodl-tmp/hybrid-rope-target-free-real-data-v3 \
  --model "${model}" --out "${root}/assets"

canary=${root}/runtime/batch_canary
selector=${canary}/selection.json
mkdir -p "${canary}"
if [[ ! -f "${canary}/inputs.jsonl" ]]; then
  "${python_bin}" - "${panel}" "${canary}/inputs.jsonl" <<'PY'
import json,sys
rows=[json.loads(line) for line in open(sys.argv[1]) if line.strip()]
selected=rows[:4]
if len(selected)!=4 or len({row['max_new_tokens'] for row in selected})!=1:
 raise SystemExit('canary requires four rows with one generation budget')
with open(sys.argv[2]+'.incomplete','w') as stream:
 for row in selected: stream.write(json.dumps(row,sort_keys=True)+'\n')
__import__('os').replace(sys.argv[2]+'.incomplete',sys.argv[2])
PY
fi

if [[ ! -f "${selector}" ]]; then
  "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${classic}/assets/ppl46/manifest.json" --model "${model}" --arm Native \
    --extra-panel "${canary}/inputs.jsonl" --only-extra-panels --skip-lm --length-cap 16384 \
    --batch-size 1 --static-table-json "${classic}/tables/tailspline.json" \
    --table-label olmo2_1b_s4_naturalqa631_tailspline_canary_batch1 \
    --out "${canary}/batch1" --execute >"${root}/logs/canary_batch1.log" 2>&1
  batch4_status=ok
  if ! "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${classic}/assets/ppl46/manifest.json" --model "${model}" --arm Native \
    --extra-panel "${canary}/inputs.jsonl" --only-extra-panels --skip-lm --length-cap 16384 \
    --batch-size 4 --left-pad-batches \
    --static-table-json "${classic}/tables/tailspline.json" \
    --table-label olmo2_1b_s4_naturalqa631_tailspline_canary_batch4 \
    --out "${canary}/batch4" --execute >"${root}/logs/canary_batch4.log" 2>&1; then
    batch4_status=failed
  fi
  "${python_bin}" - "${canary}/inputs.jsonl" "${classic}/tables/tailspline.json" \
    "${canary}/batch1/generations.jsonl" "${canary}/batch4/generations.jsonl" \
    "${selector}" "${batch4_status}" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
panel=Path(sys.argv[1]);table=Path(sys.argv[2]);batch1=Path(sys.argv[3]);batch4=Path(sys.argv[4]);out=Path(sys.argv[5]);status=sys.argv[6]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path): return [json.loads(line) for line in path.read_text().splitlines() if line]
left=rows(batch1);right=rows(batch4) if status=='ok' and batch4.is_file() else []
left_map={row['row_id']:row for row in left};right_map={row['row_id']:row for row in right}
same_ids=(set(left_map)==set(right_map) and len(left_map)==4)
tokens_equal=same_ids and all(left_map[k]['generated_ids']==right_map[k]['generated_ids'] for k in left_map)
scores_equal=same_ids and all(left_map[k]['whole_response_f1']==right_map[k]['whole_response_f1'] for k in left_map)
selected=4 if tokens_equal and scores_equal else 1
report={'status':'OLMO_NATURAL_QA_BATCH_CANARY_V1','panel_sha256':sha(panel),
        'table_receipt_sha256':sha(table),'batch4_process_status':status,
        'rows':4,'generated_ids_exact':tokens_equal,'scores_exact':scores_equal,
        'selected_batch_size':selected,'selected_left_pad_batches':selected==4}
out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_name(out.name+'.incomplete');tmp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');os.replace(tmp,out)
PY
fi

read -r batch_size left_pad_batches < <("${python_bin}" - "${selector}" "${canary}/inputs.jsonl" "${classic}/tables/tailspline.json" <<'PY'
import hashlib,json,sys
def sha(path): return hashlib.sha256(open(path,'rb').read()).hexdigest()
x=json.load(open(sys.argv[1]))
if (x.get('status')!='OLMO_NATURAL_QA_BATCH_CANARY_V1' or x.get('rows')!=4
    or x.get('panel_sha256')!=sha(sys.argv[2]) or x.get('table_receipt_sha256')!=sha(sys.argv[3])
    or x.get('selected_batch_size') not in (1,4)
    or bool(x.get('selected_left_pad_batches'))!=(x.get('selected_batch_size')==4)):
 raise SystemExit('batch canary selection drift')
print(x['selected_batch_size'],str(bool(x['selected_left_pad_batches'])).lower())
PY
)

validate_state() {
  local arm=$1 run=${root}/runs/$1
  "${python_bin}" - "${panel}" "${run}" "${classic}/tables/${arm}.json" "${arm}" \
    "${batch_size}" "${left_pad_batches}" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
panel_path=Path(sys.argv[1]);run=Path(sys.argv[2]);table_path=Path(sys.argv[3]);arm=sys.argv[4]
batch_size=int(sys.argv[5]);left_pad_batches=sys.argv[6]=='true'
def rows(path): return [json.loads(line) for line in path.read_text().splitlines() if line]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
panel=rows(panel_path)
if batch_size>1:
 panel=sorted(panel,key=lambda row:(row['length_cap'],row['max_new_tokens'],len(row['prompt_ids']),row['task'],f"extra_{panel_path.parent.name}:{row['row_id']}"))
receipt=json.load(open(table_path));table=receipt.get('table',receipt)
label=f'olmo2_1b_s4_naturalqa631_{arm}'
expected_ids=[f"extra_{panel_path.parent.name}:{row['row_id']}" for row in panel]
launch={
 'status':'OLMO_NATURAL_QA_LAUNCH_V1','panel_sha256':sha(panel_path),
 'table_receipt_sha256':sha(table_path),'arm':label,'rows':len(panel),
 'batch_size':batch_size,'left_pad_batches':left_pad_batches,'row_ids':expected_ids,
}
run.mkdir(parents=True,exist_ok=True);launch_path=run/'launch_contract.json'
if launch_path.exists():
 if json.load(open(launch_path))!=launch: raise SystemExit('launch contract drift')
else:
 if any((run/name).exists() for name in ('contract.json','generations.jsonl','status.json','summary.json')):
  raise SystemExit('outputs exist without launch contract')
 tmp=launch_path.with_name(launch_path.name+'.incomplete');tmp.write_text(json.dumps(launch,indent=2,sort_keys=True)+'\n');os.replace(tmp,launch_path)
contract_path=run/'contract.json';generation_path=run/'generations.jsonl';status_path=run/'status.json'
if not contract_path.exists():
 if generation_path.exists() or status_path.exists(): raise SystemExit('output exists without kernel contract')
 print('INCOMPLETE');raise SystemExit(0)
contract=json.load(open(contract_path))
valid=(contract.get('arm')==label and contract.get('base_arm')=='Native'
 and contract.get('unadapted') is True and contract.get('row_ids')==expected_ids
 and contract.get('generation_length_caps')==[16384] and contract.get('lm_enabled') is False
 and contract.get('batch_size')==batch_size and contract.get('prefill_chunk_size')==0
 and bool(contract.get('left_pad_batches',False)) is left_pad_batches
 and (contract.get('static_table') or {}).get('values_float32')==table.get('values_float32')
 and (contract.get('static_table') or {}).get('gain')==table.get('gain'))
if not valid: raise SystemExit('kernel contract drift')
saved=rows(generation_path) if generation_path.exists() else []
if len(saved)>len(panel): raise SystemExit('too many saved generations')
for index,row in enumerate(saved):
 source=panel[index]
 expected={'eval_id':expected_ids[index],'row_id':source['row_id'],'task':source['task'],
           'prompt_sha256':source['prompt_sha256'],'input_tokens':source['input_tokens'],
           'references':source['references'],'arm':label}
 if any(row.get(key)!=value for key,value in expected.items()):
  raise SystemExit(f'generation prefix drift at row {index}')
if not status_path.exists(): print('INCOMPLETE');raise SystemExit(0)
if json.load(open(status_path))!={'status':'COMPLETE','rows':len(panel),'lm_rows':0} or len(saved)!=len(panel):
 raise SystemExit('completion status drift')
summary=json.load(open(run/'summary.json'))
if summary.get('status')!='COMPLETE' or summary.get('identity')!=contract:
 raise SystemExit('summary identity drift')
print('COMPLETE')
PY
}

run_arm() {
  local arm=$1 run=${root}/runs/$1
  local state
  state=$(validate_state "${arm}")
  if [[ "${state}" == "COMPLETE" ]]; then
    printf 'SKIP_COMPLETE %s\n' "${arm}"
    return
  fi
  if [[ "${state}" != "INCOMPLETE" ]]; then
    printf 'REFUSE: unexpected validation state for %s: %s\n' "${arm}" "${state}" >&2
    exit 1
  fi
  batch_args=(--batch-size "${batch_size}")
  if [[ "${left_pad_batches}" == "true" ]]; then batch_args+=(--left-pad-batches); fi
  "${python_bin}" -m experiments.olmo_recovery_20260912.recovery_v2_eval \
    --data "${classic}/assets/ppl46/manifest.json" --model "${model}" --arm Native \
    --extra-panel "${panel}" --only-extra-panels --skip-lm --length-cap 16384 \
    "${batch_args[@]}" --static-table-json "${classic}/tables/${arm}.json" \
    --table-label "olmo2_1b_s4_naturalqa631_${arm}" --out "${run}" --execute \
    >"${root}/logs/${arm}.log" 2>&1
  state=$(validate_state "${arm}")
  if [[ "${state}" != "COMPLETE" ]]; then
    printf 'REFUSE: %s returned without a complete validated run\n' "${arm}" >&2
    exit 1
  fi
}

run_arm tailspline
run_arm mrpro

"${python_bin}" -m experiments.iclr2027_strong_evidence_20260915.matched_olmo_naturalqa_report \
  --panel "${panel}" \
  --candidate "${root}/runs/tailspline/generations.jsonl" \
  --baseline "${root}/runs/mrpro/generations.jsonl" \
  --out "${root}/reports/tailspline_vs_mrpro_naturalqa631.json"

printf 'OLMO_NATURAL_QA631_COMPLETE %s\n' "$(date -u +%FT%TZ)" | tee "${root}/complete.txt"
