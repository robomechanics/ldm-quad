#!/bin/bash
# Level 1 watcher: run the offline context eval on every checkpoint of a training run as it lands,
# on the run's own replay (train) and a held-out buffer, then update <run>/level1/trend.csv and the
# weight-norm alert (<run>/level1/ALERT.txt). Exits once the training unit has stopped and every
# checkpoint has a result. CPU only (4 threads).
# Usage: level1_watch.sh <run_dir> <training_unit> <heldout_replay>
cd /home/rml2/Documents/thomas_practice/ldm-quad
D=$1; UNIT=$2; H=$3
PY=/home/rml2/anaconda3/envs/isaaclab/bin/python
TMP=$D/level1/.tmp; mkdir -p $TMP
while true; do
  pending=0
  for m in $(ls $D/checkpoints/model_*.pt 2>/dev/null | sort -V); do
    name=$(basename "$m" .pt)
    [ -f "$D/level1/$name.json" ] && continue
    pending=1
    [ $(( $(date +%s) - $(stat -c %Y "$m") )) -lt 60 ] && continue
    cp "$m" $TMP/model.pt && cp $D/checkpoints/replay_latest.pt $TMP/replay.pt || continue
    echo "[level1] $(date -Is) evaluating $name"
    $PY scripts/mbrl/eval_context_offline.py --checkpoint $TMP/model.pt \
      --replay train=$TMP/replay.pt heldout=$H --out $D/level1/$name > $D/level1/$name.txt 2>&1
    grep -E "^PASS:|total online" $D/level1/$name.txt
    $PY scripts/mbrl/level1_trend.py $D/level1
  done
  if ! systemctl --user is-active --quiet $UNIT && [ $pending -eq 0 ]; then
    rm -rf $TMP; echo "[level1] $(date -Is) $UNIT stopped and all checkpoints evaluated; exiting"; exit 0
  fi
  sleep 300
done
