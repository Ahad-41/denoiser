"""
Kaggle entry point. Pushed from the server with:  kaggle kernels push -p kaggle/
Clones the fork, restores the checkpoint from the private demucs-ckpt dataset,
trains for at most TRAIN_HOURS and leaves only /kaggle/working/<MODE>/ as output.
The mode (causal/noncausal) comes from MODE.txt, written by push_ckpt.sh.
"""
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# ---------------- settings ----------------
GITHUB_REPO = "https://github.com/Ahad-41/denoiser.git"
BRANCH = "bengali-baseline"
TRAIN_HOURS = 11.0     # Kaggle kills sessions at 12 h; setup + saving need the rest
NUM_WORKERS = 2        # data loader workers per GPU process (Kaggle has 4 CPUs)
# ------------------------------------------

START = time.time()
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
REPO = WORK / "denoiser"
SKIP = {"MODE.txt", "train.txt", "valid.txt", "dataset-metadata.json"}


def sh(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check).returncode


def find_dir(name, parent=None):
    """Find a folder called `name` (optionally inside a folder called `parent`) with wavs."""
    for p in sorted(INPUT.rglob(name)):
        if p.is_dir() and (parent is None or p.parent.name == parent) \
                and next(p.rglob("*.wav"), None) is not None:
            return p
    sys.exit(f"ERROR: no '{name}/' folder with .wav files under {INPUT}")


def find_ckpt_dir():
    hits = sorted(INPUT.rglob("MODE.txt"))
    if not hits:
        sys.exit("ERROR: demucs-ckpt dataset is not attached (MODE.txt not found)")
    mode, *stamp = hits[0].read_text().split()
    if mode not in ("causal", "noncausal"):
        sys.exit(f"ERROR: bad mode '{mode}' in {hits[0]}")
    print(f"Checkpoint dataset: mode={mode} uploaded={' '.join(stamp)}", flush=True)
    return hits[0].parent, mode


sh("nvidia-smi")
sh(f"git clone --depth 1 -b {BRANCH} {GITHUB_REPO} {REPO}")
os.chdir(REPO)
sh("git log -1 --format='commit %H %s'")
sh("pip install -q -r requirements_bengali.txt")

ckpt, MODE = find_ckpt_dir()
exp_dir = REPO / "outputs" / MODE
exp_dir.mkdir(parents=True, exist_ok=True)
if (ckpt / "checkpoint.th").exists():
    for f in ckpt.iterdir():
        if f.is_file() and f.name not in SKIP:
            shutil.copy(f, exp_dir / f.name)
    print("Resuming from checkpoint", flush=True)
else:
    print("No checkpoint found: starting from scratch", flush=True)

env = dict(os.environ,
           NOISY_DIR=str(find_dir("noisy_tr15k")),
           CLEAN_DIR=str(find_dir("clean_15k")),
           TEST_CLEAN_DIR=str(find_dir("clean_ts")),
           SPLIT_DIR=str(ckpt))
for k in ("NOISY_DIR", "CLEAN_DIR", "TEST_CLEAN_DIR", "SPLIT_DIR"):
    print(f"{k}={env[k]}", flush=True)

# Run training in its own process group, so that at the deadline the DDP workers
# are stopped too (not only the parent). Checkpoints are written atomically
# (tmp file + rename), so stopping at any moment leaves the last epoch intact.
budget = TRAIN_HOURS * 3600 - (time.time() - START)
print(f"Training budget: {budget / 3600:.2f} h", flush=True)
proc = subprocess.Popen(["bash", "run.sh", MODE, f"num_workers={NUM_WORKERS}"],
                        env=env, start_new_session=True)
try:
    code = proc.wait(timeout=budget)
    print(f"\nTraining finished, exit code {code}", flush=True)
except subprocess.TimeoutExpired:
    print("\nTime budget reached: stopping training", flush=True)
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()

# Keep only the experiment folder as Kaggle output (egs/ lists input paths and the
# repo itself is on GitHub).
for tmp in exp_dir.glob("*.tmp"):
    tmp.unlink()
if exp_dir.exists():
    shutil.move(str(exp_dir), str(WORK / MODE))
os.chdir(WORK)
shutil.rmtree(REPO, ignore_errors=True)
sh(f"ls -la {WORK / MODE}", check=False)
sh(f"python -c \"import json; h=json.load(open('{WORK / MODE}/history.json')); "
   f"print('epochs finished:', len(h)); print('last:', h[-1])\"", check=False)
