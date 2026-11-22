#!/usr/bin/env bash
# Keep up to MAX_GPU Kaggle GPU sessions busy with the given kernels (in order), fetch each one when it finishes.
# usage: bash kaggle/run_queue.sh <slug> [<slug> ...]
# State lives in marker files, so the script can be stopped and re-run at any time (e.g. after the PC was off):
#   kaggle/kernels/<slug>/.pushed_<version>   -> this notebook version was pushed
#   research/runs/_kaggle/<slug>/.fetched_<version> -> outputs downloaded
K=${KAGGLE_CLI:-kaggle}                       # Kaggle CLI (override with KAGGLE_CLI)
KAGGLE_USER=${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VER=$(grep -oE 'NOTEBOOK_VERSION = "[0-9.]+"' $ROOT/build_notebook.py | head -1 | grep -oE '[0-9.]+')
MAX_GPU=2
start=$(date +%s)
while :; do
  running=0; pending=(); done_all=1
  for s in "$@"; do
    pushed=$ROOT/kaggle/kernels/$s/.pushed_$VER
    fetched=$ROOT/research/runs/_kaggle/$s/.fetched_$VER
    [ -f "$fetched" ] && continue
    done_all=0
    if [ ! -f "$pushed" ]; then pending+=("$s"); continue; fi
    st=$($K kernels status $KAGGLE_USER/$s 2>&1 | tail -1)
    case "$st" in
      *RUNNING*|*QUEUED*) running=$((running + 1)) ;;
      *COMPLETE*|*ERROR*|*CANCEL*)
        out=$ROOT/research/runs/_kaggle/$s; mkdir -p "$out"
        find "$out" -name "*.kaggle-partial" -delete
        $K kernels output $KAGGLE_USER/$s -p "$out" -o > /dev/null 2>&1
        z=$(find "$out" -maxdepth 3 -name "*.zip" | head -1)
        # mark as fetched only if the run zip exists, has no partial leftover, and passes an integrity test
        if [ -n "$z" ] && [ -z "$(find "$out" -name '*.kaggle-partial')" ] && \
           python -c "import zipfile,sys; sys.exit(0 if zipfile.ZipFile(sys.argv[1]).testzip() is None else 1)" "$z" 2>/dev/null; then
          cp -f "$z" $ROOT/research/runs/
          touch "$fetched"; echo "$(date +%H:%M:%S) FETCHED $s :: $st"
        else
          echo "$(date +%H:%M:%S) download incomplete for $s, will retry"; running=$((running + 1))
        fi ;;
      *) running=$((running + 1)) ;;   # unknown/transient: count as busy, re-check next round
    esac
  done
  for s in "${pending[@]}"; do
    [ $running -ge $MAX_GPU ] && break
    res=$($K kernels push -p $ROOT/kaggle/kernels/$s --accelerator NvidiaTeslaT4 2>&1 | tail -1)
    if echo "$res" | grep -q "successfully pushed"; then
      touch $ROOT/kaggle/kernels/$s/.pushed_$VER; running=$((running + 1)); echo "$(date +%H:%M:%S) PUSHED $s"
    else
      echo "$(date +%H:%M:%S) push deferred $s :: $res"; break
    fi
  done
  [ $done_all = 1 ] && { echo "$(date +%H:%M:%S) ALL RUNS FETCHED (v$VER)"; exit 0; }
  [ $(( $(date +%s) - start )) -gt 86400 ] && { echo "stopping after 24 h; re-run to continue"; exit 1; }
  sleep 240
done
