"""Train the DrowsinessLSTM on fixed-length windows sliced from per-video
feature sequences produced by extract_sequences.py.

Train/val split is done per VIDEO (not per window) so overlapping windows from
the same clip never appear in both sets.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models import DrowsinessLSTM


class WindowDataset(Dataset):
    def __init__(self, npz_paths, window_size=30, stride=15):
        self.windows = []
        self.labels = []
        for npz_path in npz_paths:
            data = np.load(npz_path)
            feats, label = data["features"], int(data["label"])
            for start in range(0, len(feats) - window_size + 1, stride):
                self.windows.append(feats[start:start + window_size])
                self.labels.append(label)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.from_numpy(self.windows[idx]), torch.tensor(self.labels[idx], dtype=torch.float32)


def split_by_video(sequences_dir, val_frac=0.2, seed=0):
    """Stratified split of .npz files by label so both classes appear in val."""
    by_label = {0: [], 1: []}
    for p in sorted(Path(sequences_dir).glob("*.npz")):
        by_label[int(np.load(p)["label"])].append(p)
    rng = random.Random(seed)
    train, val = [], []
    for paths in by_label.values():
        rng.shuffle(paths)
        n_val = max(1, round(val_frac * len(paths))) if len(paths) > 1 else 0
        val += paths[:n_val]
        train += paths[n_val:]
    return train, val


def run_epoch(model, loader, device, optimizer=None, pos_weight=None):
    training = optimizer is not None
    model.train(training)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    total_loss, n = 0.0, 0
    tp = fp = fn = tn = 0

    for windows, labels in tqdm(loader, leave=False):
        windows, labels = windows.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(windows)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * windows.size(0)
        n += windows.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()

    acc = (tp + tn) / max(n, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return total_loss / max(n, 1), acc, prec, rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences-dir", default="../data/sequences")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="../models/drowsiness_lstm.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_files, val_files = split_by_video(args.sequences_dir, seed=args.seed)
    if not train_files or not val_files:
        raise SystemExit("Need at least 2 videos per class in --sequences-dir for a train/val split.")
    train_ds = WindowDataset(train_files, args.window_size, args.stride)
    val_ds = WindowDataset(val_files, args.window_size, args.stride)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("No windows produced: videos are shorter than --window-size. Use longer clips or a smaller window.")
    print(f"Videos: {len(train_files)} train / {len(val_files)} val | "
          f"Windows: {len(train_ds)} train / {len(val_ds)} val")

    n_pos = sum(train_ds.labels)
    n_neg = len(train_ds) - n_pos
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), device=device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = DrowsinessLSTM(feature_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, device, optimizer, pos_weight)
        va_loss, va_acc, va_prec, va_rec = run_epoch(model, val_loader, device, pos_weight=pos_weight)
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} "
              f"| val_loss={va_loss:.4f} acc={va_acc:.4f} prec={va_prec:.3f} rec={va_rec:.3f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out)
            print(f"  saved new best model (val_acc={va_acc:.4f}) -> {args.out}")

    print(f"Done. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
