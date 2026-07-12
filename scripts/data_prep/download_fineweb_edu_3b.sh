#!/usr/bin/env bash
set -Eeuo pipefail

ACTION="${1:-status}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ROOT="${FINEWEB_3B_ROOT:-$REPO_ROOT/data/fineweb-edu_sample10BT_rev87f09149}"
HF_BIN="${FINEWEB_HF_BIN:-$(command -v hf || true)}"
REVISION="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
PID_FILE="$ROOT/download.pid"
LOG_DIR="$ROOT/logs"
SNAPSHOT_FILE="$ROOT/progress.snapshot"
VERIFY_DIR="$ROOT/.verified"
START_LOCK="$ROOT/start.lock"
SAFETY_MARGIN_BYTES=$((1024 * 1024 * 1024))

NAMES=(
  000_00000.parquet
  001_00000.parquet
  002_00000.parquet
  003_00000.parquet
)
SIZES=(2152819114 2152222432 2151796315 2152437524)
HASHES=(
  b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871
  3fcf2dc69cd52503986276d3d2d26a8c356d0f2ea28a0de4fdbda8cf87755693
  547ae182d132c9f06b6ce63149567208ea9f57630bfd9b1a2938e504f0c9ebd7
  22184e6eb25759ddd97783751ffc73e1705dfa2542e630dae1f2a8bac8ee6ddb
)

downloaded_bytes() {
  local name="$1" hash="$2" final incomplete
  final="$ROOT/sample/10BT/$name"
  if [[ -f "$final" ]]; then
    stat -c %s "$final"
    return
  fi
  incomplete="$(find "$ROOT/.cache/huggingface/download/sample/10BT" \
    -maxdepth 1 -type f -name "*.${hash}.incomplete" -print -quit 2>/dev/null || true)"
  if [[ -n "$incomplete" ]]; then
    stat -c %s "$incomplete"
  else
    echo 0
  fi
}

print_bar() {
  local downloaded="$1" expected="$2" width=30 filled empty pct
  pct=$(( downloaded * 100 / expected ))
  (( pct > 100 )) && pct=100
  filled=$(( pct * width / 100 ))
  empty=$(( width - filled ))
  printf '['
  printf '%*s' "$filled" '' | tr ' ' '#'
  printf '%*s' "$empty" '' | tr ' ' '-'
  printf '] %3d%%' "$pct"
}

is_download_pid() {
  local pid="$1" cmdline name
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$cmdline" == *"$HF_BIN download HuggingFaceFW/fineweb-edu"* ]] &&
    [[ "$cmdline" == *"--revision $REVISION"* ]] &&
    [[ "$cmdline" == *"--local-dir $ROOT"* ]] || return 1
  for name in "${NAMES[@]}"; do
    [[ "$cmdline" == *"sample/10BT/$name"* ]] || return 1
  done
}

active_pid() {
  local pid="" candidate
  if [[ -s "$PID_FILE" ]]; then
    pid="$(<"$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null && is_download_pid "$pid"; then
      echo "$pid"
      return
    fi
  fi
  while read -r candidate; do
    if [[ -n "$candidate" ]] && is_download_pid "$candidate"; then
      echo "$candidate"
      return
    fi
  done < <(pgrep -f "$HF_BIN download HuggingFaceFW/fineweb-edu" 2>/dev/null || true)
}

current_log() {
  if [[ -s "$ROOT/download.log_path" ]]; then
    cat "$ROOT/download.log_path"
  elif [[ -d "$LOG_DIR" ]]; then
    find "$LOG_DIR" -maxdepth 1 -type f -name 'download_*.log' -print 2>/dev/null \
      | sort | tail -n 1
  fi
}

file_fingerprint() {
  stat -c '%s %Y %Z %i' "$1"
}

is_verified_final() {
  local name="$1" hash="$2" final receipt expected_receipt
  final="$ROOT/sample/10BT/$name"
  receipt="$VERIFY_DIR/$name.sha256"
  [[ -f "$final" && -s "$receipt" ]] || return 1
  expected_receipt="$hash $(file_fingerprint "$final")"
  [[ "$(<"$receipt")" == "$expected_receipt" ]]
}

verify_final_files() {
  local i name expected_hash expected_size final actual_hash quarantine invalid=0 missing=0
  mkdir -p "$VERIFY_DIR"
  for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    expected_hash="${HASHES[$i]}"
    expected_size="${SIZES[$i]}"
    final="$ROOT/sample/10BT/$name"
    if [[ ! -f "$final" ]]; then
      missing=1
      continue
    fi
    if is_verified_final "$name" "$expected_hash"; then
      echo "verified (cached): $name"
      continue
    fi
    if [[ "$(stat -c %s "$final")" != "$expected_size" ]]; then
      quarantine="$final.corrupt-size.$(date -u +%Y%m%dT%H%M%SZ)"
      mv -- "$final" "$quarantine"
      echo "quarantined invalid-size shard: $quarantine" >&2
      invalid=1
      continue
    fi
    echo "sha256: $name"
    actual_hash="$(nice -n 19 ionice -c3 sha256sum "$final" | awk '{print $1}')"
    if [[ "$actual_hash" == "$expected_hash" ]]; then
      printf '%s %s\n' "$expected_hash" "$(file_fingerprint "$final")" \
        > "$VERIFY_DIR/$name.sha256"
      echo "verified: $name"
    else
      quarantine="$final.corrupt-sha256.$(date -u +%Y%m%dT%H%M%SZ)"
      mv -- "$final" "$quarantine"
      echo "quarantined checksum-failed shard: $quarantine" >&2
      invalid=1
    fi
  done
  (( invalid == 0 && missing == 0 ))
}

print_rate_and_eta() {
  local total="$1" expected="$2" now previous_time previous_total elapsed delta
  now="$(date +%s)"
  if [[ -s "$SNAPSHOT_FILE" ]]; then
    read -r previous_time previous_total < "$SNAPSHOT_FILE" || true
    elapsed=$(( now - previous_time ))
    delta=$(( total - previous_total ))
    if (( elapsed > 0 && elapsed <= 300 && delta >= 0 && total < expected )); then
      awk -v d="$delta" -v t="$elapsed" -v remain="$(( expected - total ))" '
        BEGIN {
          rate=d/t;
          if (rate > 0) {
            eta=remain/rate;
            printf "rate: %.2f MiB/s  ETA: %02dh%02dm\n", rate/1048576, int(eta/3600), int((eta%3600)/60)
          }
        }'
    fi
  fi
  if [[ -d "$ROOT" ]]; then
    printf '%s %s\n' "$now" "$total" > "$SNAPSHOT_FILE"
  fi
}

status() {
  local total=0 expected_total=0 i bytes counted expected state pid log final
  echo "FineWeb-Edu sample-10BT download dashboard"
  echo "revision: $REVISION"
  echo "root:     $ROOT"
  for i in "${!NAMES[@]}"; do
    bytes="$(downloaded_bytes "${NAMES[$i]}" "${HASHES[$i]}")"
    expected="${SIZES[$i]}"
    counted="$bytes"
    (( counted > expected )) && counted="$expected"
    total=$(( total + counted ))
    expected_total=$(( expected_total + expected ))
    state="partial"
    (( bytes == 0 )) && state="pending"
    final="$ROOT/sample/10BT/${NAMES[$i]}"
    if [[ -f "$final" && "$bytes" -ne "$expected" ]]; then
      state="invalid-size"
    elif [[ -f "$final" ]] && is_verified_final "${NAMES[$i]}" "${HASHES[$i]}"; then
      state="verified"
    elif [[ -f "$final" ]]; then
      state="downloaded-unverified"
    elif (( bytes >= expected )); then
      state="finalizing"
    fi
    printf '%s  ' "${NAMES[$i]}"
    print_bar "$bytes" "$expected"
    awk -v b="$bytes" -v e="$expected" -v s="$state" \
      'BEGIN {printf "  %.2f/%.2f GiB  %s\n", b/1073741824, e/1073741824, s}'
  done
  printf 'TOTAL              '
  print_bar "$total" "$expected_total"
  awk -v b="$total" -v e="$expected_total" \
    'BEGIN {printf "  %.2f/%.2f GiB\n", b/1073741824, e/1073741824}'
  print_rate_and_eta "$total" "$expected_total"
  pid="$(active_pid)"
  if [[ -n "$pid" ]]; then
    echo "downloader: running (pid $pid)"
  else
    echo "downloader: stopped; run '$0 resume' to continue"
  fi
  log="$(current_log)"
  [[ -n "$log" ]] && echo "log:        $log"
}

start() {
  mkdir -p "$ROOT" "$LOG_DIR" "$VERIFY_DIR"
  local existing_pid name i bytes counted expected_total=0 downloaded_total=0 all_verified=1
  local available_bytes required_bytes remaining_bytes
  local paths=()
  exec 9> "$START_LOCK"
  flock -n 9 || { echo "another start/resume command is active" >&2; exit 1; }
  existing_pid="$(active_pid)"
  if [[ -n "$existing_pid" ]]; then
    printf '%s\n' "$existing_pid" > "$PID_FILE"
    echo "downloader already running (pid $existing_pid); adopted existing process"
    status
    return
  fi
  [[ -x "$HF_BIN" ]] || { echo "hf CLI not found: $HF_BIN" >&2; exit 1; }
  verify_final_files || true
  for name in "${NAMES[@]}"; do
    paths+=("sample/10BT/$name")
  done
  for i in "${!NAMES[@]}"; do
    bytes="$(downloaded_bytes "${NAMES[$i]}" "${HASHES[$i]}")"
    counted="$bytes"
    (( counted > SIZES[i] )) && counted="${SIZES[$i]}"
    downloaded_total=$(( downloaded_total + counted ))
    expected_total=$(( expected_total + SIZES[i] ))
    is_verified_final "${NAMES[$i]}" "${HASHES[$i]}" || all_verified=0
  done
  remaining_bytes=$(( expected_total - downloaded_total ))
  if (( all_verified == 1 )); then
    echo "all requested shards are already downloaded and verified"
    status
    return
  fi
  available_bytes=$(( $(df -Pk "$ROOT" | awk 'NR==2 {print $4}') * 1024 ))
  required_bytes=$(( remaining_bytes + SAFETY_MARGIN_BYTES ))
  (( available_bytes >= required_bytes )) || {
    awk -v a="$available_bytes" -v r="$required_bytes" \
      'BEGIN {printf "insufficient disk: %.2f GiB free, %.2f GiB required\n", a/1073741824, r/1073741824}' >&2
    exit 1
  }
  local log pid
  log="$LOG_DIR/download_$(date -u +%Y%m%dT%H%M%SZ).log"
  nohup env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
    nice -n 19 ionice -c3 "$HF_BIN" download HuggingFaceFW/fineweb-edu \
    "${paths[@]}" \
    --repo-type dataset --revision "$REVISION" --local-dir "$ROOT" \
    --max-workers 4 >"$log" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$PID_FILE"
  echo "$log" > "$ROOT/download.log_path"
  sleep 2
  kill -0 "$pid" 2>/dev/null || {
    echo "downloader failed to start; inspect $log" >&2
    exit 1
  }
  status
}

case "$ACTION" in
  start|resume) start ;;
  status) status ;;
  verify) verify_final_files ;;
  watch)
    while true; do
      clear 2>/dev/null || true
      date -Is
      status
      sleep "${FINEWEB_STATUS_INTERVAL:-30}"
    done
    ;;
  *) echo "usage: $0 {start|resume|status|watch|verify}" >&2; exit 2 ;;
esac
