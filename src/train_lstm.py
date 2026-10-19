"""Train the DrowsinessLSTM on fixed-length windows sliced from per-video
feature sequences produced by extract_sequences.py."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from models import DrowsinessLSTM


class WindowDataset(Dataset):
    def __init__(self, sequences_dir, window_size=30, stride=15):
        self.windows = []
        self.labels = []
        for npz_path in sorted(Path(sequences_dir).glob("*.npz")):
            data = np.load(npz_path)
            feats, label = data["features"], int(data["label"])
            for start in range(0, max(1, len(feats) - window_size + 1), stride):
                window = feats[start:start + window_size]
                if len(window) < window_size:
                    break
                self.windows.append(window)
                self.labels.append(label)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.from_numpy(self.windows[idx]), torch.tensor(self.labels[idx], dtype=torch.float32)


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    total_loss, correct, n = 0.0, 0, 0

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
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        n += windows.size(0)

    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences-dir", default="../data/sequences")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="../models/drowsiness_lstm.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = WindowDataset(args.sequences_dir, args.window_size, args.stride)
    val_size = max(1, int(0.2 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = DrowsinessLSTM(feature_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out)
            print(f"  saved new best model (val_acc={val_acc:.4f}) -> {args.out}")

    print(f"Done. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
