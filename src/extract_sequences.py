"""Extract per-frame [p_closed, EAR, MAR] feature sequences from labeled videos,
for training the DrowsinessLSTM.

Expects a CSV at <videos_dir>/labels.csv with columns: filename,label
  label = 1 for drowsy, 0 for alert

Saves one .npz per video to <out_dir>, each containing:
    features: (num_frames, 3) array of [p_closed, ear, mar]
    label: int
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from landmarks import FaceLandmarker
from models import EyeStateCNN


def load_eye_cnn(weights_path, device):
    model = EyeStateCNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def extract_video_features(video_path, landmarker, eye_cnn, device):
    cap = cv2.VideoCapture(str(video_path))
    features = []

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = landmarker.process(frame)
            if result is None:
                continue

            crop = result["right_eye_crop"].astype(np.float32) / 255.0
            tensor = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).to(device)
            p_closed = torch.sigmoid(eye_cnn(tensor)).item()

            features.append([p_closed, result["ear"], result["mar"]])

    cap.release()
    return np.array(features, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--eye-cnn-weights", default="../models/eye_cnn.pt")
    parser.add_argument("--out-dir", default="../data/sequences")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eye_cnn = load_eye_cnn(args.eye_cnn_weights, device)
    landmarker = FaceLandmarker()

    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(videos_dir / "labels.csv") as f:
        rows = list(csv.DictReader(f))

    for row in tqdm(rows):
        video_path = videos_dir / row["filename"]
        features = extract_video_features(video_path, landmarker, eye_cnn, device)
        if len(features) == 0:
            print(f"Warning: no face detected in {video_path}, skipping")
            continue
        out_path = out_dir / (Path(row["filename"]).stem + ".npz")
        np.savez(out_path, features=features, label=int(row["label"]))

    landmarker.close()
    print(f"Saved {len(rows)} sequence files to {out_dir}")


if __name__ == "__main__":
    main()
