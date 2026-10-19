# Driver Drowsiness Detector (CNN + LSTM)

Webcam-based driver drowsiness detection: a per-frame CNN classifies eye state
(open/closed) from MediaPipe-detected eye crops, and an LSTM consumes a
rolling window of `[p_closed, EAR, MAR]` features to classify the window as
drowsy or alert. Ties into the autonomous-vehicles driver-monitoring use case.

## Project structure

```
src/
  landmarks.py          # MediaPipe Face Mesh wrapper: EAR, MAR, eye crops
  models.py             # EyeStateCNN and DrowsinessLSTM definitions
  train_eye_cnn.py       # Stage 1: train eye open/closed classifier
  extract_sequences.py  # Stage 2: run CNN+landmarks over labeled videos -> feature sequences
  train_lstm.py          # Stage 3: train LSTM on windowed sequences
  live_demo.py           # Stage 4: real-time webcam demo
data/
  eye_state/             # MRL Eye Dataset, reorganized (see below)
  sequences/             # output of extract_sequences.py
models/                 # saved .pt weights
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Stage 1 — Eye-state CNN

Download the [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset) (most mirrors,
including the common Kaggle re-upload, ship it as two folders — one per class,
e.g. `Open-Eyes/` and `Closed-Eyes/`). Then run:

```bash
cd src
python make_eye_split.py --open-dir /path/to/Open-Eyes --closed-dir /path/to/Closed-Eyes --out-dir ../data/eye_state
```

This produces the split train_eye_cnn.py expects:
```
data/eye_state/train/open/*.png
data/eye_state/train/closed/*.png
data/eye_state/val/open/*.png
data/eye_state/val/closed/*.png
```

Train:
```bash
cd src
python train_eye_cnn.py --data-dir ../data/eye_state --epochs 15
```
Expect >95% validation accuracy — this is a well-separated task.

## Stage 2 — Build labeled sequences for the LSTM

Get video-level drowsy/alert labels, either from the **NTHU-DDD** dataset
(request access from the authors — do this early, turnaround isn't instant)
or from your own short webcam recordings of normal vs. exaggerated
slow-blink/eye-closure behavior.

Put videos + a `labels.csv` (`filename,label` where `label=1` is drowsy) in
one folder, then run:
```bash
python extract_sequences.py --videos-dir /path/to/videos --eye-cnn-weights ../models/eye_cnn.pt
```
This produces one `.npz` feature sequence per video in `data/sequences/`.

## Stage 3 — Train the LSTM

```bash
python train_lstm.py --sequences-dir ../data/sequences --window-size 30
```
`window-size=30` is ~1 second at 30fps; try 60–90 for a longer temporal
context if accuracy is weak with the shorter window.

## Stage 4 — Live demo

```bash
python live_demo.py --eye-cnn-weights ../models/eye_cnn.pt --lstm-weights ../models/drowsiness_lstm.pt
```
Press `q` to quit. Tune `--drowsy-threshold` and `--alert-consec-windows` to
control sensitivity vs. false-alarm rate.

## Notes / fallbacks

- If dataset access (NTHU-DDD, YawDD) is slow, self-recorded sequences are a
  legitimate substitute for a course-scale project — label windows with
  exaggerated slow blinks/closures as "drowsy".
- Yawn/MAR is currently used only as a raw feature into the LSTM, not a
  separately trained classifier — a MAR threshold already captures yawns well
  (large, sustained spike) without needing more training data.
- `right_eye_crop` is used alone for simplicity; averaging left+right CNN
  outputs is a straightforward extension if one eye is frequently occluded.
