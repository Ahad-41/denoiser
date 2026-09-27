#!/bin/bash
# Server -> Kaggle. Uploads the experiment folder of one mode (if any) plus the split
# lists as a new version of the PRIVATE Kaggle dataset <KAGGLE_USER>/<CKPT_DATASET>.
# MODE.txt in it tells the Kaggle kernel which mode to train.
# Usage: ./push_ckpt.sh causal|noncausal
set -euo pipefail
cd "$(dirname "$0")"
source config.sh
MODE=${1:?usage: ./push_ckpt.sh causal|noncausal}
case "$MODE" in causal|noncausal) ;; *) echo "unknown mode: $MODE"; exit 1 ;; esac

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
cat > "$STAGE/dataset-metadata.json" <<JSON
{"title": "$CKPT_DATASET", "id": "$KAGGLE_USER/$CKPT_DATASET", "licenses": [{"name": "unknown"}]}
JSON
printf "%s\n%s\n" "$MODE" "$(date '+%Y-%m-%d_%H:%M')" > "$STAGE/MODE.txt"
cp "$SPLIT_DIR/train.txt" "$SPLIT_DIR/valid.txt" "$STAGE/"

if [ -f "outputs/$MODE/checkpoint.th" ]; then
  # top-level files only (checkpoint.th, best.th, history.json, trainer.log*)
  find "outputs/$MODE" -maxdepth 1 -type f ! -name '*.tmp' ! -name rendezvous \
    -exec cp {} "$STAGE/" \;
  python3 -c "import json; print('epochs done:', len(json.load(open('outputs/$MODE/history.json'))))"
  echo "Uploading checkpoint for $MODE"
else
  echo "No checkpoint for $MODE yet -> Kaggle will start from scratch"
fi
ls -la "$STAGE"

if "$KAGGLE" datasets status "$KAGGLE_USER/$CKPT_DATASET" >/dev/null 2>&1; then
  "$KAGGLE" datasets version -p "$STAGE" -m "$MODE $(date '+%F %T')"
else
  "$KAGGLE" datasets create -p "$STAGE"      # private unless --public is given
fi

echo "Waiting for Kaggle to finish processing the dataset..."
sleep 30
until "$KAGGLE" datasets status "$KAGGLE_USER/$CKPT_DATASET" 2>/dev/null | grep -qi ready; do
  sleep 20
done
echo "Done. Next: $KAGGLE kernels push -p kaggle/"
