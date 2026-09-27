#!/usr/bin/env python3
"""
Build egs/mydata/{tr,cv,ts} json files for the Bengali SE baseline.

Split lists hold one relative path per line (e.g. 06736/06736-0784adfd84.wav), the same
path under both the noisy and the clean root. The lists are the supervisor's official,
speaker-disjoint split; their sha256 must equal the fingerprints in his split_manifest.json.

   python prepare_data.py --noisy-root .../Train/noisy_tr15k --clean-root .../Train/clean_15k \
       --train-list splits/train.txt --valid-list splits/valid.txt \
       --test-clean-dir .../Different_test_data/clean_ts --out egs/mydata

Output (json = [[absolute_path, num_samples], ...], the format of `python -m denoiser.audio`):
   egs/mydata/tr/{noisy,clean}.json   train
   egs/mydata/cv/{noisy,clean}.json   valid (also used as dset.test during training)
   egs/mydata/ts/noisy.json           a few valid files, enhanced into samples/ for listening
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import soundfile as sf

SR = 16000
# sha256 of the supervisor's train/val lists (split_manifest.json "fingerprints")
FINGERPRINTS = {
    "train": "adcce460e1107b5d1f4b3140780772b7c5dd251791dcd5b33c8716e17a1a2ae4",
    "valid": "ac1362b1ae18d95d40c6f8a6170e9aec158dead70bea5f82506ea441a65a0959",
}


def read_list(path, split, check_fingerprint):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if check_fingerprint and digest != FINGERPRINTS[split]:
        sys.exit(f"{path}: sha256 {digest} does not match the supervisor's {split} split")
    names = [line.strip() for line in raw.decode().splitlines() if line.strip()]
    if len(names) != len(set(names)):
        sys.exit(f"{path}: duplicate entries")
    return names


def speaker(rel):
    return Path(rel).parent.name


def scan(names, noisy_root, clean_root):
    """Check every pair and return the json entries for noisy and clean."""
    noisy, clean, bad = [], [], 0
    for rel in names:
        n, c = noisy_root / rel, clean_root / rel
        if not n.exists() or not c.exists():
            print(f"[MISSING] {rel}: noisy={n.exists()} clean={c.exists()}")
            bad += 1
            continue
        ni, ci = sf.info(str(n)), sf.info(str(c))
        for tag, info in (("noisy", ni), ("clean", ci)):
            if info.samplerate != SR or info.channels != 1:
                print(f"[FORMAT] {rel} ({tag}): {info.samplerate} Hz, {info.channels} ch")
                bad += 1
        if ni.frames != ci.frames:
            print(f"[LENGTH] {rel}: noisy={ni.frames} clean={ci.frames}")
            bad += 1
        noisy.append((str(n.resolve()), ni.frames))
        clean.append((str(c.resolve()), ci.frames))
    return noisy, clean, bad


def write_json(entries, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f, indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noisy-root", required=True, type=Path)
    p.add_argument("--clean-root", required=True, type=Path)
    p.add_argument("--train-list", required=True)
    p.add_argument("--valid-list", required=True)
    p.add_argument("--test-clean-dir", type=Path,
                   help="clean test folder; its files and speakers must not be in train/valid")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--num-samples", type=int, default=20,
                   help="valid files to enhance into samples/ at the end of training")
    p.add_argument("--limit", type=int, help="use only the first N files per split (smoke test)")
    args = p.parse_args()

    check = args.limit is None
    splits = {"train": read_list(args.train_list, "train", check),
              "valid": read_list(args.valid_list, "valid", check)}
    if check:
        print("Split fingerprints match the supervisor's split_manifest.json")

    # Leakage checks: train/valid are disjoint by file and speaker; test is in neither.
    tr, cv = set(splits["train"]), set(splits["valid"])
    if tr & cv:
        sys.exit(f"{len(tr & cv)} files are in both train and valid")
    if {speaker(x) for x in tr} & {speaker(x) for x in cv}:
        sys.exit("train and valid share speakers")
    if args.test_clean_dir:
        test = list(args.test_clean_dir.rglob("*.wav"))
        if not test:
            sys.exit(f"no test wav files under {args.test_clean_dir}")
        used_names = {Path(x).name for x in tr | cv}
        used_spk = {speaker(x) for x in tr | cv}
        leak = [t for t in test if t.name in used_names or t.parent.name in used_spk]
        if leak:
            sys.exit(f"{len(leak)} test files leak into train/valid, e.g. {leak[:3]}")
        print(f"Leak check OK: none of the {len(test)} test files or their speakers "
              f"are in train/valid")

    if args.limit:
        splits = {k: v[:args.limit] for k, v in splits.items()}

    bad = 0
    for split, sub in (("train", "tr"), ("valid", "cv")):
        noisy, clean, b = scan(splits[split], args.noisy_root, args.clean_root)
        bad += b
        # the loader pairs files by sorting each list; make sure that keeps pairs aligned
        order = sorted(range(len(noisy)), key=lambda i: noisy[i][0])
        if order != sorted(range(len(clean)), key=lambda i: clean[i][0]):
            sys.exit(f"{split}: sorting noisy and clean paths does not keep pairs aligned")
        write_json(noisy, args.out / sub / "noisy.json")
        write_json(clean, args.out / sub / "clean.json")
        hours = sum(n for _, n in noisy) / SR / 3600
        print(f"{args.out / sub}: {len(noisy)} pairs, {hours:.2f} h")
        if split == "valid":
            write_json(sorted(noisy)[:args.num_samples], args.out / "ts" / "noisy.json")
    if bad:
        sys.exit(f"Found {bad} problems. Fix them before training.")
    print("Done.")


if __name__ == "__main__":
    main()
