# Server-side settings (sourced by run.sh, push_ckpt.sh, pull_ckpt.sh, evaluate_test.sh).
# No secrets here: Kaggle reads ~/.kaggle/access_token, git reads ~/.git-credentials.
KAGGLE_USER="ahad41"
KERNEL_SLUG="demucs-train"          # must match "id" in kaggle/kernel-metadata.json
CKPT_DATASET="demucs-ckpt"          # private dataset: checkpoints + split lists

BASE="$HOME/demucs_denoiser"
KAGGLE="$BASE/.cli-venv/bin/kaggle"
SPLIT_DIR="$BASE/splits"            # train.txt, valid.txt (never in git)
SIR_EVAL="$BASE/from_sir/evaluate_se_fast_v3.py"
DNSMOS_DIR="$BASE/dnsmos_models"    # needs sig_bak_ovr.onnx, else DNSMOS = NaN
ENHANCED_DIR="$BASE/enhanced"
RESULTS_DIR="$BASE/results"

DATA="/mnt/all_data/bengali_SE_data"
NOISY_DIR="$DATA/Train/noisy_tr15k"
CLEAN_DIR="$DATA/Train/clean_15k"
TEST_ROOT="$DATA/Different_test_data"
TEST_CLEAN_DIR="$TEST_ROOT/clean_ts"
TEST_CONDITIONS="4unseen6SNR 3seenNoise3seenSNR car_noisy"
