#!/bin/bash
# Local watchdog: reconnect, verify, restart what died, read what finished.
#
# WHY.  The instance has now been disrupted twice in one day (11:29 killed every
# runner within 2 s; 16:0x closed SSH during kex and left the card empty), with no
# OOM, no reboot and no dmesg trace.  `supervise.sh` only heals stages that are
# already running -- if the whole process tree dies, or the host is unreachable
# when a stage would have been restarted, nothing recovers.  This runs LOCALLY,
# outside the instance, so a dead instance cannot take the watchdog with it.
#
# It is idempotent by construction: every action is gated on a row count, and the
# runners skip rows already present, so a restart resumes rather than repeats.
#
# Stages it protects, and what "done" means for each:
#   lb_hold      olmo_lb_h/nu_{p,m}*.jsonl      >= 180 rows each   -> run lb_read.py
#   qwen4x_bm    qwen4x_power/rows.jsonl        has beta_b1_BM    -> run qwen4x_power_read.py
#   gsweep       olmo_gsweep/gain_bm_g1p20.jsonl >= 350 rows      -> run gsweep_read.py
#
# Usage:  nohup ./keepalive.sh > keepalive.log 2>&1 &
set -u
H="ssh -o ConnectTimeout=25 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -p 27741 root@connect.westc.seetacloud.com"
D=/root/autodl-tmp/phase1_20260910
PY=/root/miniconda3/bin/python
MODEL=/root/autodl-tmp/olmo2_1b_longalign_assets/models/OLMo-2-0425-1B-Instruct
HOLD=/root/autodl-tmp/olmo_fast_screen_20260908/prepared_holdout_union/screen.jsonl
NEWT=/root/autodl-tmp/olmo_fast_screen_20260908/prepared_ruler_newtasks_02/screen.jsonl
ARCH=/root/autodl-tmp/olmo_fast_screen_20260908/run_ruler_newtasks_01

alive=0
for i in $(seq 1 240); do
  if ! $H "echo UP" >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] ssh down (attempt $i)"; sleep 60; continue
  fi
  if [ $alive -eq 0 ]; then echo "[$(date +%H:%M:%S)] === SSH BACK ==="; alive=1; fi

  $H "cd $D
NP=\$(cat olmo_lb_h/nu_p6p104em05.jsonl 2>/dev/null | wc -l)
NM=\$(cat olmo_lb_h/nu_m6p104em05.jsonl 2>/dev/null | wc -l)
BM=\$(grep -c 'beta_b1_BM' qwen4x_power/rows.jsonl 2>/dev/null || echo 0)
GS=\$(cat olmo_gsweep/gain_bm_g1p20.jsonl 2>/dev/null | wc -l)
PLB=\$(pgrep -fc 'nu-shift' || echo 0)
PQW=\$(pgrep -fc 'qwen_longnll' || echo 0)
PGS=\$(pgrep -fc 'g1p20' || echo 0)
echo \"lb_hold=\$NP/\$NM  qwenBM=\$BM  gsweep=\$GS  procs(lb,qw,gs)=\$PLB,\$PQW,\$PGS\"

# --- lb_hold: the decisive held-out verdict -------------------------------
if [ \"\$NP\" -ge 180 ] && [ \"\$NM\" -ge 180 ]; then
  if [ ! -f olmo_lb_h/.read ]; then
    echo '=== LB HELD-OUT DONE ==='; $PY lb_read.py | tee olmo_lb_h/read.txt; touch olmo_lb_h/.read
  fi
elif [ \"\$PLB\" -eq 0 ]; then
  echo '=== RESTART lb_hold ==='
  PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:$D/repoharness:$D \
  setsid nohup $PY olmo_beta.py --root $D/olmo_lb_h --model $MODEL --panel $HOLD \
    --archive $D/empty_archive --betas '' --turns '' \
    --nu-shift '6.103515625e-05:28,29,30,31;-6.103515625e-05:28,29,30,31' \
    >> lb_hold2.log 2>&1 < /dev/null & disown
fi

# --- qwen 4x BM arm: the last underpowered cell in the leverage table ------
if [ \"\$BM\" -ge 1 ]; then
  if [ ! -f qwen4x_power/.read ]; then
    echo '=== QWEN 4x BM DONE ==='; $PY qwen4x_power_read.py | tee qwen4x_power/read.txt; touch qwen4x_power/.read
  fi
elif [ \"\$PQW\" -eq 0 ]; then
  echo '=== RESTART qwen BM arm ==='
  PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:$D/repoharness:$D \
  setsid nohup $PY qwen_longnll.py --root $D/qwen4x_power \
    --model /root/autodl-tmp/qwen25_1p5b_32k \
    --nll-dir /root/autodl-tmp/longtext/prepared_pg19_4x \
    --far 131073 --tail 512 --only native,beta_b1_BM \
    >> qwen4x_bm.log 2>&1 < /dev/null & disown
fi

# --- gsweep 1.20: the upper bracket of the gain curve ---------------------
if [ \"\$GS\" -ge 350 ]; then
  if [ ! -f olmo_gsweep/.read ]; then
    echo '=== GSWEEP DONE ==='; $PY gsweep_read.py | tee olmo_gsweep/read.txt; touch olmo_gsweep/.read
  fi
elif [ \"\$PGS\" -eq 0 ]; then
  echo '=== RESTART gsweep 1.20 ==='
  PYTHONPATH=/root/autodl-tmp/nongeometric_screen_20260909/code:$D/repoharness:$D \
  setsid nohup $PY olmo_beta.py --root $D/olmo_gsweep --model $MODEL --panel $NEWT \
    --archive $ARCH --betas '' --turns '' --gain-tables bm --gains 1.20 \
    >> gsweep2.log 2>&1 < /dev/null & disown
fi
" 2>&1 | grep -v '^$'

  # all three read -> stop
  if $H "test -f $D/olmo_lb_h/.read && test -f $D/qwen4x_power/.read && test -f $D/olmo_gsweep/.read" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] === ALL THREE READ, watch complete ==="; break
  fi
  sleep 120
done
echo "[$(date +%H:%M:%S)] keepalive exiting"
