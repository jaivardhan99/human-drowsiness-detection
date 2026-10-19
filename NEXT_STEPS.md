# Next Steps

## Status

- **Stage 1 — Eye-state CNN: done.** Trained on the MRL Eye Dataset (85k images),
  96.81% validation accuracy. Weights at `models/eye_cnn.pt` (not committed —
  see `.gitignore`; regenerate with `src/train_eye_cnn.py`).
- **Stage 2 — Labeled video sequences for the LSTM: not started.**
- **Stage 3 — Train the LSTM: not started.**
- **Stage 4 — Live webcam demo: code written, untested end-to-end** (blocked
  on Stage 3's trained weights).

## Decision needed: Stage 2 data source

Two options, not yet decided:

1. **NTHU-DDD dataset** — realistic driver videos, proper drowsy/alert labels.
   Requires an academic access request to the authors; turnaround is
   unpredictable (days to weeks), so this risks the course deadline if
   requested late. Request it now if going this route, in parallel with
   anything else.
2. **Self-recorded webcam clips** (recommended for timeline safety) — record
   short clips of normal blinking/talking vs. exaggerated slow-blink /
   prolonged eye-closure / yawning, hand-label each clip `0`/`1`. No waiting,
   can be done same-day, and doubles as live-demo validation footage.

## Once Stage 2 data exists

1. Put videos + a `labels.csv` (`filename,label`, `label=1` = drowsy) in one
   folder.
2. Run:
   ```bash
   cd src
   python extract_sequences.py --videos-dir /path/to/videos --eye-cnn-weights ../models/eye_cnn.pt
   ```
   This produces `.npz` feature sequences (`[p_closed, EAR, MAR]` per frame)
   in `data/sequences/`.
3. Train the LSTM:
   ```bash
   python train_lstm.py --sequences-dir ../data/sequences --window-size 30
   ```
   Try `--window-size 60` or `90` if accuracy is weak — longer temporal
   context may help.
4. Run the live demo:
   ```bash
   python live_demo.py --eye-cnn-weights ../models/eye_cnn.pt --lstm-weights ../models/drowsiness_lstm.pt
   ```
   Tune `--drowsy-threshold` and `--alert-consec-windows` for sensitivity.

## Open items for the course writeup

- Baseline comparison: a simple EAR/MAR-threshold heuristic (no CNN/LSTM) is
  worth including as a baseline to justify the deep-learning approach.
- If going the self-recorded route, note in the report that NTHU-DDD/YawDD
  were considered but not used due to access turnaround, as a scoped,
  deliberate decision rather than an oversight.
- Head-pose/nodding detection was discussed as a possible extension but is
  out of scope unless there's time left after Stage 4 works end-to-end.
