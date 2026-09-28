# Project context for Claude Code

DEMUCS (facebookresearch/denoiser) **baseline** for the supervisor's Bengali speech
enhancement paper. This repo is the user's fork (github.com/Ahad-41/denoiser).
Reply to the user in **Bangla (Bangla script)**; keep commands/code/file names in English.
Full step log (Bangla): `~/demucs_denoiser/SETUP_LOG.md`.

## Layout (server)
- `~/demucs_denoiser/` — home of everything (no spaces in path).
  - `denoiser/` — this repo (the only thing that goes to GitHub).
  - `splits/train.txt`, `splits/valid.txt` — supervisor's speaker-disjoint split as sorted
    relative paths (`<spk>/<spk>-<hash>.wav`); sha256 of each file equals the fingerprint in
    `splits/raw/splits/split_manifest.json`. NOT in git.
  - `from_sir/` — supervisor's `evaluate_se_fast_v3.py` + `splits.rar`. NOT in git.
  - `.cli-venv/` — kaggle CLI + libarchive-c (tools only).
- Data (server master copy): `/mnt/all_data/bengali_SE_data/`
  - `Train/clean_15k/`, `Train/noisy_tr15k/` — 15,000 pairs, 16 kHz mono, `<spk>/<file>.wav`.
  - `Different_test_data/clean_ts/` + noisy conditions `4unseen6SNR/`, `3seenNoise3seenSNR/`,
    `car_noisy/` (2,206 files each). This is the TEST set; no overlap with train/val.

## Decisions (do not change without asking the user)
- Train from scratch, 100 epochs. Causal first, then non-causal.
- Other args follow `launch_valentini.sh` (causal) / `launch_valentini_nc.sh` (non-causal).
- Fixed experiment dirs: `hydra.run.dir=outputs/causal` and `outputs/noncausal` so training
  resumes on either machine (solver auto-loads `checkpoint.th` from the run dir).
- During training: valid = supervisor's val split, and `tt` also = val (never the real test).
- Paper metrics: only from supervisor's `evaluate_se_fast_v3.py` on enhanced test outputs.
  DNSMOS: model `~/demucs_denoiser/dnsmos_models/sig_bak_ovr.onnx` (= Microsoft non-personalized),
  **uncalibrated** — never pass `--dnsmos_calibrate` (supervisor's choice, 2026-09-28).
- Server holds master data + checkpoints. Kaggle = borrowed GPU:
  private dataset `ahad41/bengali-dataset` (audio), private dataset `ahad41/demucs-ckpt`
  (checkpoints + split lists). Kernel pushed from the server with the kaggle CLI; training
  stopped by a timeout before Kaggle's 12 h limit.
- Server Python/torch/torchaudio must match Kaggle's.

## Hard rules
- Never commit/upload to GitHub: audio, split lists, supervisor's script, checkpoints,
  kaggle.json/access_token or any token. Kaggle datasets stay private.
- Test files must never appear in train or valid.
- Ask the user before: any Kaggle upload/push, any git push, starting any long training run.
- Work in phases; after each, report in Bangla and update SETUP_LOG.md.
- Server GPU is shared: never run anything on it yourself. When a server-GPU step is needed
  (e.g. `./evaluate_test.sh`), give the user the exact command and let them run it.
- Use GPU for all compute (not CPU fallbacks); scripts stay GPU-first.

## Environment
- Server conda env `demucs` = Kaggle image: python 3.12.13, torch/torchaudio 2.10.0+cu128,
  numpy 2.0.2, scipy 1.16.3, soundfile 0.13.1, omegaconf 2.3.0, plus
  `requirements_bengali.txt` (both machines) and `requirements_eval.txt` (server only).
- Kaggle CLI: `~/demucs_denoiser/.cli-venv/bin/kaggle` (auth: `~/.kaggle/access_token`).
- Code was patched for torch 2.10 / numpy 2 / hydra 1.3 (audio I/O via soundfile, stft
  return_complex, torch.load weights_only=False, hydra version_base="1.1", LowPassFilters
  float32). Details in SETUP_LOG.md.

## Workflow (from the repo root, env active)
- Kaggle cycle: `./push_ckpt.sh causal` → `kaggle kernels push -p kaggle/` → wait →
  `./pull_ckpt.sh causal`. Kernel clones branch `bengali-baseline` from GitHub and reads the
  mode from MODE.txt in the ckpt dataset; stops after TRAIN_HOURS=11.
- Server run: `./run.sh causal|noncausal` (batch 64 does not fit one T4 — unresolved).
- Kaggle smoke run (2026-09-27): DDP 2×T4 OK, ~10.2 GB/15 GB per GPU at batch 64 (32/GPU),
  ~1.25 s/iteration → ~21 min/epoch causal, ~30 epochs per 11 h session.
- Kaggle smoke check: `SMOKE=1 ./push_ckpt.sh causal` → `kaggle kernels push -p kaggle/` →
  output `smoke-causal/` (500 files, 2 epochs + resume to 3, PESQ every epoch, gpu_log.csv).
  Real run afterwards: plain `./push_ckpt.sh causal` (MODE.txt without SMOKE).
- Test scoring: `./evaluate_test.sh causal` → `~/demucs_denoiser/results/<mode>/`.
