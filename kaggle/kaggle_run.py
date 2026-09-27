"""
Kaggle entry point. Pushed from the server with:  kaggle kernels push -p kaggle/
Clones the fork, restores the checkpoint from the private demucs-ckpt dataset,
trains for at most TRAIN_HOURS and leaves only /kaggle/working/<MODE>/ as output.
The mode (causal/noncausal) comes from MODE.txt, written by push_ckpt.sh.
If MODE.txt also says SMOKE (SMOKE=1 ./push_ckpt.sh <mode>), a short check runs instead:
SMOKE_LIMIT files, 2 epochs with PESQ every epoch, then a resume to epoch 3, from scratch,
in outputs/smoke-<mode> (never mixed with the real run). GPU memory is logged to gpu_log.csv.
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
SMOKE_HOURS = 1.0      # time cap of a smoke run
SMOKE_LIMIT = 500      # files per split in a smoke run
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
    mode, *rest = hits[0].read_text().split()
    if mode not in ("causal", "noncausal"):
        sys.exit(f"ERROR: bad mode '{mode}' in {hits[0]}")
    smoke = "SMOKE" in rest
    print(f"Checkpoint dataset: mode={mode} {' '.join(rest)}", flush=True)
    return hits[0].parent, mode, smoke


sh("nvidia-smi")
sh(f"git clone --depth 1 -b {BRANCH} {GITHUB_REPO} {REPO}")
os.chdir(REPO)
sh("git log -1 --format='commit %H %s'")
sh("pip install -q -r requirements_bengali.txt")

ckpt, MODE, SMOKE = find_ckpt_dir()
RUN = f"smoke-{MODE}" if SMOKE else MODE
exp_dir = REPO / "outputs" / RUN
exp_dir.mkdir(parents=True, exist_ok=True)
if SMOKE:
    print(f"SMOKE run: {SMOKE_LIMIT} files, epochs 2 then resume to 3", flush=True)
elif (ckpt / "checkpoint.th").exists():
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
           SPLIT_DIR=str(ckpt),
           RUN_DIR=f"outputs/{RUN}")
for k in ("NOISY_DIR", "CLEAN_DIR", "TEST_CLEAN_DIR", "SPLIT_DIR"):
    print(f"{k}={env[k]}", flush=True)

# Run training in its own process group, so that at the deadline the DDP workers
# are stopped too (not only the parent). Checkpoints are written atomically
# (tmp file + rename), so stopping at any moment leaves the last epoch intact.
def train(deadline, extra_env=None, extra_args=()):
    budget = deadline - time.time()
    print(f"Training budget: {budget / 3600:.2f} h", flush=True)
    proc = subprocess.Popen(["bash", "run.sh", MODE, f"num_workers={NUM_WORKERS}", *extra_args],
                            env=dict(env, **(extra_env or {})), start_new_session=True)
    try:
        code = proc.wait(timeout=max(budget, 1))
        print(f"\nTraining finished, exit code {code}", flush=True)
        return code
    except subprocess.TimeoutExpired:
        print("\nTime budget reached: stopping training", flush=True)
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        return None


# GPU memory/utilization every 30 s, kept next to the checkpoint.
gpu_log = subprocess.Popen(
    ["nvidia-smi", "--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu",
     "--format=csv", "-l", "30"], stdout=open(exp_dir / "gpu_log.csv", "w"))
if SMOKE:
    deadline = START + SMOKE_HOURS * 3600
    small = dict(LIMIT=str(SMOKE_LIMIT))
    if train(deadline, dict(small, EPOCHS="2"), ["eval_every=1"]) == 0:
        print("\n===== SMOKE: resume check (epoch 3) =====", flush=True)
        train(deadline, dict(small, EPOCHS="3"), ["eval_every=1"])
else:
    train(START + TRAIN_HOURS * 3600)
gpu_log.terminate()

# Keep only the experiment folder as Kaggle output (egs/ lists input paths and the
# repo itself is on GitHub).
for tmp in exp_dir.glob("*.tmp"):
    tmp.unlink()
if exp_dir.exists():
    shutil.move(str(exp_dir), str(WORK / RUN))
os.chdir(WORK)
shutil.rmtree(REPO, ignore_errors=True)
sh(f"ls -la {WORK / RUN}", check=False)
sh(f"python -c \"import json; h=json.load(open('{WORK / RUN}/history.json')); "
   f"print('epochs finished:', len(h)); print('last:', h[-1])\"", check=False)
