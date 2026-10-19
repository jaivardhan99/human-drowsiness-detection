"""Reorganize a raw open/closed eye image dataset into the train/val split
expected by train_eye_cnn.py.

Two supported source layouts:

1. Two pre-split folders (most Kaggle mirrors), one per class:
    python make_eye_split.py --open-dir /path/to/Open-Eyes --closed-dir /path/to/Closed-Eyes --out-dir ../data/eye_state

2. The official MRL Eye Dataset flat folder, where every image's class is
   encoded in its filename as
   subjectID_imageID_gender_glasses_eyestate_reflections_lighting_sensor.png
   (eyestate: 0=closed, 1=open), e.g. s0001_00001_0_0_0_0_0_01.png:
    python make_eye_split.py --flat-dir /path/to/MRL_dataset --out-dir ../data/eye_state
"""
import argparse
import random
import shutil
from pathlib import Path


def split_and_copy(src_dir, class_name, out_dir, val_frac, seed):
    src_dir = Path(src_dir)
    images = [p for p in src_dir.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".pgm")]
    if not images:
        raise ValueError(f"No images found under {src_dir}")

    random.Random(seed).shuffle(images)
    n_val = max(1, int(len(images) * val_frac))
    val_images, train_images = images[:n_val], images[n_val:]

    for split, imgs in [("train", train_images), ("val", val_images)]:
        dest_dir = Path(out_dir) / split / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for img_path in imgs:
            shutil.copy2(img_path, dest_dir / img_path.name)

    print(f"{class_name}: {len(train_images)} train, {len(val_images)} val (from {len(images)} total)")


def split_flat_by_filename(flat_dir, out_dir, val_frac, seed):
    """Official MRL layout: eye state is the 5th underscore-separated field
    (0=closed, 1=open) in each filename."""
    flat_dir = Path(flat_dir)
    images = [p for p in flat_dir.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".pgm")]
    if not images:
        raise ValueError(f"No images found under {flat_dir}")

    by_class = {"open": [], "closed": []}
    skipped = 0
    for img_path in images:
        parts = img_path.stem.split("_")
        if len(parts) < 5:
            skipped += 1
            continue
        eye_state = parts[4]
        if eye_state == "1":
            by_class["open"].append(img_path)
        elif eye_state == "0":
            by_class["closed"].append(img_path)
        else:
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} files with unexpected filename format")

    random.Random(seed).shuffle(by_class["open"])
    random.Random(seed).shuffle(by_class["closed"])

    for class_name, imgs in by_class.items():
        n_val = max(1, int(len(imgs) * val_frac))
        val_images, train_images = imgs[:n_val], imgs[n_val:]
        for split, split_imgs in [("train", train_images), ("val", val_images)]:
            dest_dir = Path(out_dir) / split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_imgs:
                shutil.copy2(img_path, dest_dir / img_path.name)
        print(f"{class_name}: {len(train_images)} train, {len(val_images)} val (from {len(imgs)} total)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-dir", help="Pre-split folder of open-eye images")
    parser.add_argument("--closed-dir", help="Pre-split folder of closed-eye images")
    parser.add_argument("--flat-dir", help="Official MRL flat folder, class encoded in filename")
    parser.add_argument("--out-dir", default="../data/eye_state")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.flat_dir:
        split_flat_by_filename(args.flat_dir, args.out_dir, args.val_frac, args.seed)
    elif args.open_dir and args.closed_dir:
        split_and_copy(args.open_dir, "open", args.out_dir, args.val_frac, args.seed)
        split_and_copy(args.closed_dir, "closed", args.out_dir, args.val_frac, args.seed)
    else:
        raise SystemExit("Pass either --flat-dir, or both --open-dir and --closed-dir")

    print(f"Done. Dataset ready at {args.out_dir}")


if __name__ == "__main__":
    main()
