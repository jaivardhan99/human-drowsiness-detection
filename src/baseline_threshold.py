"""EAR/MAR threshold baseline (no deep learning) for comparison with the CNN+LSTM.

A window is called drowsy if either:
  - the fraction of frames with EAR < ear_thresh exceeds closed_frac (PERCLOS-style), or
  - the fraction of frames with MAR > mar_thresh exceeds yawn_frac.

Uses the same train/val video split and windows as train_lstm.py so numbers
are directly comparable.
"""
import argparse

import numpy as np

from train_lstm import split_by_video, WindowDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences-dir", default="../data/sequences")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--ear-thresh", type=float, default=0.21)
    parser.add_argument("--closed-frac", type=float, default=0.4)
    parser.add_argument("--mar-thresh", type=float, default=0.6)
    parser.add_argument("--yawn-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _, val_files = split_by_video(args.sequences_dir, seed=args.seed)
    ds = WindowDataset(val_files, args.window_size, args.stride)

    tp = fp = fn = tn = 0
    for w, y in zip(ds.windows, ds.labels):
        ear, mar = w[:, 1], w[:, 2]
        pred = int((ear < args.ear_thresh).mean() > args.closed_frac
                   or (mar > args.mar_thresh).mean() > args.yawn_frac)
        tp += pred == 1 and y == 1
        fp += pred == 1 and y == 0
        fn += pred == 0 and y == 1
        tn += pred == 0 and y == 0

    n = tp + fp + fn + tn
    print(f"Baseline on {n} val windows | acc={(tp + tn) / max(n, 1):.4f} "
          f"prec={tp / max(tp + fp, 1):.3f} rec={tp / max(tp + fn, 1):.3f}")


if __name__ == "__main__":
    main()
