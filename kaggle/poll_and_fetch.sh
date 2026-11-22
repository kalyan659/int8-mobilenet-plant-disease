#!/usr/bin/env bash
# Poll Kaggle kernels until each reaches a terminal state, then download its outputs (incl. the run log) into
# research/runs/_kaggle/<slug>/ and copy any <RUN_ID>.zip into research/runs/.
# usage: bash kaggle/poll_and_fetch.sh <slug> [<slug> ...]      (interval 180 s, gives up after 11 h)
K=${KAGGLE_CLI:-kaggle}                       # Kaggle CLI (override with KAGGLE_CLI)
KAGGLE_USER=${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
declare -A done_
start=$(date +%s)
while :; do
  all=1
  for s in "$@"; do
    [ -n "${done_[$s]}" ] && continue
    st=$($K kernels status $KAGGLE_USER/$s 2>&1 | tail -1)
    echo "$(date +%H:%M:%S) $s :: $st"
    case "$st" in
      *COMPLETE*|*ERROR*|*CANCEL*)
        out=$ROOT/research/runs/_kaggle/$s
        mkdir -p "$out"
        $K kernels output $KAGGLE_USER/$s -p "$out" -o 2>&1 | tail -2
        find "$out" -maxdepth 3 -name "*.zip" -exec cp -f {} $ROOT/research/runs/ \;
        echo "FETCHED $s ($st) -> $out"
        done_[$s]=1 ;;
      *) all=0 ;;
    esac
  done
  [ $all = 1 ] && { echo "ALL DONE"; exit 0; }
  [ $(( $(date +%s) - start )) -gt 39600 ] && { echo "GAVE UP after 11h"; exit 1; }
  sleep 180
done
