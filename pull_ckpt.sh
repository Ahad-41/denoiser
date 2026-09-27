#!/bin/bash
# Kaggle -> Server. Downloads the output of the last Kaggle run and puts the
# experiment folder into outputs/<MODE>. The old folder is kept as a backup.
# Refuses to replace a local folder that has MORE finished epochs (FORCE=1 overrides).
# Usage: ./pull_ckpt.sh causal|noncausal
set -euo pipefail
cd "$(dirname "$0")"
source config.sh
MODE=${1:?usage: ./pull_ckpt.sh causal|noncausal}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
"$KAGGLE" kernels output "$KAGGLE_USER/$KERNEL_SLUG" -p "$TMP"

if [ ! -f "$TMP/$MODE/checkpoint.th" ]; then
  echo "ERROR: no $MODE/checkpoint.th in the Kaggle output. Files found:"
  ls -R "$TMP" | head -50
  exit 1
fi

epochs() { python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$1"; }
NEW=$(epochs "$TMP/$MODE/history.json")
OLD=0
if [ -f "outputs/$MODE/history.json" ]; then OLD=$(epochs "outputs/$MODE/history.json"); fi
echo "epochs: local=$OLD  kaggle=$NEW"
if [ "$NEW" -lt "$OLD" ] && [ "${FORCE:-0}" != 1 ]; then
  echo "ERROR: Kaggle output has fewer epochs than the local copy; not replacing (FORCE=1 to override)"
  exit 1
fi

mkdir -p outputs
STAMP=$(date +%Y%m%d-%H%M)
if [ -d "outputs/$MODE" ]; then
  mv "outputs/$MODE" "outputs/$MODE.bak-$STAMP"
  echo "Old folder kept as outputs/$MODE.bak-$STAMP"
fi
mv "$TMP/$MODE" "outputs/$MODE"
# keep the Kaggle run log next to the checkpoint
for f in "$TMP"/*.log; do [ -f "$f" ] && cp "$f" "outputs/$MODE/kaggle-$STAMP.log"; done

python3 - "$MODE" <<'PY'
import json, sys
h = json.load(open(f"outputs/{sys.argv[1]}/history.json"))
print("epochs finished:", len(h))
print("last epoch:", h[-1])
PY
