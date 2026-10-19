"""Generate fake [p_closed, EAR, MAR] sequences to smoke-test Stages 3-4 plumbing
without any video data. NOT for real results — never report metrics from this.
"""
import argparse
from pathlib import Path

import numpy as np


def make_video(drowsy, n_frames, rng):
    p_closed = np.clip(rng.normal(0.1, 0.05, n_frames), 0, 1)
    ear = np.clip(rng.normal(0.30, 0.02, n_frames), 0.05, 0.5)
    mar = np.clip(rng.normal(0.30, 0.05, n_frames), 0, 1)
    if drowsy:
        # several 1-2s eye closures plus an occasional yawn
        for _ in range(rng.integers(3, 6)):
            s = rng.integers(0, n_frames - 45)
            d = rng.integers(20, 45)
            p_closed[s:s + d] = np.clip(rng.normal(0.9, 0.05, d), 0, 1)
            ear[s:s + d] = np.clip(rng.normal(0.12, 0.02, d), 0.03, 0.5)
        s = rng.integers(0, n_frames - 60)
        mar[s:s + 50] = np.clip(rng.normal(0.8, 0.05, 50), 0, 1.5)
    return np.stack([p_closed, ear, mar], axis=1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--frames", type=int, default=600)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for label in (0, 1):
        for i in range(args.per_class):
            np.savez(out / f"synth_{label}_{i}.npz",
                     features=make_video(label == 1, args.frames, rng), label=label)
    print(f"Wrote {2 * args.per_class} synthetic sequences to {out}")


if __name__ == "__main__":
    main()
