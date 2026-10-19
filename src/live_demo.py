"""Real-time webcam drowsiness detection demo.

Pipeline: webcam frame -> FaceLandmarker (EAR, MAR, eye crop) -> EyeStateCNN
(p_closed) -> rolling buffer of window_size frames -> DrowsinessLSTM -> smoothed
alert decision.

Run with:
    python live_demo.py --eye-cnn-weights ../models/eye_cnn.pt --lstm-weights ../models/drowsiness_lstm.pt
"""
import argparse
from collections import deque

import cv2
import numpy as np
import torch

from landmarks import FaceLandmarker
from models import EyeStateCNN, DrowsinessLSTM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eye-cnn-weights", default="../models/eye_cnn.pt")
    parser.add_argument("--lstm-weights", default="../models/drowsiness_lstm.pt")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--drowsy-threshold", type=float, default=0.5)
    parser.add_argument("--alert-consec-windows", type=int, default=3,
                         help="Number of consecutive drowsy predictions before raising an alert")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eye_cnn = EyeStateCNN().to(device)
    eye_cnn.load_state_dict(torch.load(args.eye_cnn_weights, map_location=device))
    eye_cnn.eval()

    lstm = DrowsinessLSTM(feature_dim=3).to(device)
    lstm.load_state_dict(torch.load(args.lstm_weights, map_location=device))
    lstm.eval()

    landmarker = FaceLandmarker()

    frame_buffer = deque(maxlen=args.window_size)
    drowsy_streak = 0

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    print("Press 'q' to quit.")
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = landmarker.process(frame)
            status_text = "No face detected"
            status_color = (128, 128, 128)

            if result is not None:
                crop = result["right_eye_crop"].astype(np.float32) / 255.0
                tensor = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).to(device)
                p_closed = torch.sigmoid(eye_cnn(tensor)).item()

                frame_buffer.append([p_closed, result["ear"], result["mar"]])

                status_text = f"EAR={result['ear']:.2f} MAR={result['mar']:.2f} p_closed={p_closed:.2f}"
                status_color = (0, 255, 0)

                if len(frame_buffer) == args.window_size:
                    window = torch.tensor([list(frame_buffer)], dtype=torch.float32).to(device)
                    drowsy_prob = torch.sigmoid(lstm(window)).item()

                    if drowsy_prob > args.drowsy_threshold:
                        drowsy_streak += 1
                    else:
                        drowsy_streak = 0

                    status_text += f" | drowsy_prob={drowsy_prob:.2f}"

                    if drowsy_streak >= args.alert_consec_windows:
                        status_text = "!!! DROWSINESS ALERT !!!"
                        status_color = (0, 0, 255)
            else:
                frame_buffer.clear()
                drowsy_streak = 0

            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.imshow("Driver Drowsiness Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
