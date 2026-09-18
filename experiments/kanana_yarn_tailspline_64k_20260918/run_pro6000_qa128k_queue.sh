#!/usr/bin/env bash
set -euo pipefail

repo=${HYBRID_ROPE_REPO:-/root/autodl-tmp/hybrid-rope}
root=${KANANA_EXPERIMENT_ROOT:-/root/autodl-tmp/today_rope_plan_20260914/kanana_yarn_tailspline_64k_20260918}
model=${KANANA_MODEL:-/root/autodl-tmp/models/kakaocorp/kanana-1.5-8b-instruct-2505}
base_url=${KANANA_MIRROR_URL:-https://hf-mirror.com/kakaocorp/kanana-1.5-8b-instruct-2505/resolve/main}

mkdir -p "${model}" "${root}/logs"
cd "${model}"

for name in config.json generation_config.json model.safetensors.index.json \
  special_tokens_map.json tokenizer.json tokenizer_config.json; do
  curl -fL --retry 5 --retry-delay 2 "${base_url}/${name}" -o "${name}"
done

download_shard() {
  local name=$1 expected_size=$2 expected_sha=$3 attempt
  if [[ -f ${name} ]] && [[ $(stat -c %s "${name}") -eq ${expected_size} ]] \
      && [[ $(sha256sum "${name}" | awk '{print $1}') == ${expected_sha} ]]; then
    return
  fi
  for attempt in 1 2 3; do
    aria2c -x 8 -s 8 -k 1M --file-allocation=none --auto-file-renaming=false \
      -d "${model}" -o "${name}" "${base_url}/${name}" && \
      [[ $(stat -c %s "${name}") -eq ${expected_size} ]] && \
      [[ $(sha256sum "${name}" | awk '{print $1}') == ${expected_sha} ]] && return
    printf 'retry shard=%s attempt=%s\n' "${name}" "${attempt}" >&2
  done
  printf 'failed shard=%s\n' "${name}" >&2
  return 1
}

download_shard model-00001-of-00004.safetensors 4932701536 636be023406874bc59c62167c4d3b93ffe761a4c454f2062f0f8adb627d767aa & p1=$!
download_shard model-00002-of-00004.safetensors 4901217648 a45e6a3b5faa72d4f03ba65735e5b924bd832e211d13de13475e8a5c0c0ac593 & p2=$!
download_shard model-00003-of-00004.safetensors 4901225464 242db2de2a9afea6d5848f4477a4eca067aae352f239907d1f1ec73bcf68904d & p3=$!
download_shard model-00004-of-00004.safetensors 1325460880 f936ffa1c2fa3918c3ce1233628453c24f2c8d53fb41aaf3b657a1e66f978150 & p4=$!

cd "${repo}"
bash experiments/kanana_yarn_tailspline_64k_20260918/prepare_qa128k.sh \
  >"${root}/logs/qa128k_prepare.log" 2>&1 & prep=$!

wait "${p1}"; wait "${p2}"; wait "${p3}"; wait "${p4}"
wait "${prep}"

KANANA_SKIP_CANARY=1 \
  bash experiments/kanana_yarn_tailspline_64k_20260918/run_qa128k_three_arm.sh --execute
