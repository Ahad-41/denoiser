#!/bin/bash
# Same command on the server and on Kaggle.
#   Server:  conda activate demucs; ./run.sh causal        (paths come from config.sh)
#   Kaggle:  NOISY_DIR=... CLEAN_DIR=... SPLIT_DIR=... TEST_CLEAN_DIR=... bash run.sh causal
# Smoke test (separate folder, a few files):
#   EPOCHS=1 LIMIT=64 RUN_DIR=outputs/smoke ./run.sh causal
# Extra hydra overrides can follow the mode, e.g. ./run.sh causal num_workers=2
set -euo pipefail
cd "$(dirname "$0")"

MODE=${1:?usage: ./run.sh causal|noncausal [hydra overrides]}
shift

# On the server, paths come from config.sh. On Kaggle, they are passed as env vars.
if [ -z "${NOISY_DIR:-}" ]; then source config.sh; fi
: "${NOISY_DIR:?}" "${CLEAN_DIR:?}" "${SPLIT_DIR:?}" "${TEST_CLEAN_DIR:?}"
EPOCHS=${EPOCHS:-100}
RUN_DIR=${RUN_DIR:-outputs/$MODE}

python -c "import sys, torch, torchaudio; print('python', sys.version.split()[0], \
'torch', torch.__version__, 'torchaudio', torchaudio.__version__)"

# 1) egs/mydata/{tr,cv,ts} for this machine (checks split fingerprints and test leakage)
python prepare_data.py --noisy-root "$NOISY_DIR" --clean-root "$CLEAN_DIR" \
  --train-list "$SPLIT_DIR/train.txt" --valid-list "$SPLIT_DIR/valid.txt" \
  --test-clean-dir "$TEST_CLEAN_DIR" --out egs/mydata ${LIMIT:+--limit "$LIMIT"}

# 2) model and training args, copied from the repo's recipes (ddp handled below):
#    causal    = launch_valentini.sh
#    noncausal = launch_valentini_nc.sh
case "$MODE" in
  causal)
    ARGS=(demucs.causal=1 demucs.hidden=48 bandmask=0.2 demucs.resample=4
          remix=1 shift=8000 shift_same=True stft_loss=True
          stft_sc_factor=0.1 stft_mag_factor=0.1 segment=4.5 stride=0.5) ;;
  noncausal)
    ARGS=(demucs.causal=0 demucs.hidden=64 demucs.stride=2 bandmask=0.2 demucs.resample=2
          remix=1 shift=8000 shift_same=True stft_loss=True segment=4.5 stride=0.5) ;;
  *) echo "unknown mode: $MODE"; exit 1 ;;
esac

# 3) use every GPU on the machine (batch_size=64 is split across GPUs)
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
DDP_ARG=()
if [ "$NGPU" -gt 1 ]; then DDP_ARG=(ddp=1); fi
echo "GPUs: $NGPU  mode: $MODE  epochs: $EPOCHS  run dir: $RUN_DIR"

# 4) train. The run dir is fixed, so a rerun on any machine finds checkpoint.th and resumes.
exec python train.py dset=mydata "${ARGS[@]}" epochs="$EPOCHS" "${DDP_ARG[@]}" \
  hydra.run.dir="$RUN_DIR" "$@"
