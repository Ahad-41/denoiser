#!/bin/bash
# Enhance the TEST set with best.th of one experiment and score it with the
# supervisor's evaluate_se_fast_v3.py (the only source of numbers for the paper).
# Runs on the server:  conda activate demucs; ./evaluate_test.sh causal [extra scorer args]
# One run per noisy condition in TEST_CONDITIONS (config.sh); clean reference = clean_ts.
# Outputs: $ENHANCED_DIR/<MODE>/<condition>/  and  $RESULTS_DIR/<MODE>/<condition>.csv
set -euo pipefail
cd "$(dirname "$0")"
source config.sh
MODE=${1:?usage: ./evaluate_test.sh causal|noncausal [extra args for the scorer]}
shift
MODEL="outputs/$MODE/best.th"
[ -f "$MODEL" ] || { echo "ERROR: $MODEL not found"; exit 1; }
[ -f "$SIR_EVAL" ] || { echo "ERROR: $SIR_EVAL not found"; exit 1; }
if [ ! -f "$DNSMOS_DIR/sig_bak_ovr.onnx" ]; then
  echo "WARNING: $DNSMOS_DIR/sig_bak_ovr.onnx missing -> DNSMOS columns will be NaN"
fi
python3 -c "import json; print('model epochs:', len(json.load(open('outputs/$MODE/history.json'))))"

DEVICE=$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")
mkdir -p "$RESULTS_DIR/$MODE"
for COND in $TEST_CONDITIONS; do
  OUT="$ENHANCED_DIR/$MODE/$COND"
  N=$(find "$TEST_ROOT/$COND" -name '*.wav' | wc -l)
  echo "=== $MODE / $COND  ($N files)"
  rm -rf "$OUT"
  python -m denoiser.enhance --model_path "$MODEL" --noisy_dir "$TEST_ROOT/$COND" \
    --out_dir "$OUT" --device "$DEVICE" --num_workers 1
  python "$SIR_EVAL" \
    --model_name "DEMUCS_${MODE}_${COND}" \
    --clean_dir "$TEST_CLEAN_DIR" \
    --enhanced_dir "$OUT" \
    --noisy_dir "$TEST_ROOT/$COND" \
    --output_csv "$RESULTS_DIR/$MODE/$COND.csv" \
    --dnsmos_model_dir "$DNSMOS_DIR" \
    --expect_n "$N" "$@"
done
echo "Results in $RESULTS_DIR/$MODE/"
