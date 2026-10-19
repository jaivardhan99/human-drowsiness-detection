"""Record labeled webcam clips for Stage 2 and append them to labels.csv.

Usage:
    python record_clips.py --out-dir ../data/videos --label 0 --seconds 20 --count 5   # alert clips
    python record_clips.py --out-dir ../data/videos --label 1 --seconds 20 --count 5   # drowsy clips

Alert (0): normal blinking, talking, looking around.
Drowsy (1): slow blinks, eyes closed 1-2s at a time, head drooping, yawning.
Press SPACE to start each clip, 'q' to quit early.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="../data/videos")
    parser.add_argument("--label", type=int, choices=[0, 1], required=True)
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.csv"
    new_file = not labels_path.exists()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tag = "drowsy" if args.label == 1 else "alert"

    with open(labels_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["filename", "label"])

        for i in range(args.count):
            name = f"{tag}_{int(time.time())}_{i}.mp4"
            # wait for SPACE
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                cv2.putText(frame, f"Clip {i+1}/{args.count} [{tag}] - SPACE to start, q to quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("Recorder", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    break
                if key == ord("q"):
                    cap.release(); cv2.destroyAllWindows(); return

            writer_vid = cv2.VideoWriter(str(out_dir / name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            start = time.time()
            while time.time() - start < args.seconds:
                ok, frame = cap.read()
                if not ok:
                    break
                writer_vid.write(frame)
                shown = frame.copy()
                cv2.putText(shown, f"REC {tag} {args.seconds - int(time.time() - start)}s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Recorder", shown)
                cv2.waitKey(1)
            writer_vid.release()
            writer.writerow([name, args.label])
            f.flush()
            print(f"Saved {name}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
