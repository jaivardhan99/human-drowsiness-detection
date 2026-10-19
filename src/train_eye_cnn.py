"""Train the eye-state CNN on open/closed eye crops.

Expects data laid out as:
    data/eye_state/train/open/*.png
    data/eye_state/train/closed/*.png
    data/eye_state/val/open/*.png
    data/eye_state/val/closed/*.png

Works directly with the MRL Eye Dataset once you've split it into this structure
(see README for a reorganization snippet).
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import EyeStateCNN


def get_loaders(data_dir, batch_size, img_size=48):
    tfm = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(Path(data_dir) / "train", transform=tfm)
    val_ds = datasets.ImageFolder(Path(data_dir) / "val", transform=tfm)
    # ImageFolder sorts classes alphabetically: "closed"=0, "open"=1 -- we want closed=1
    assert train_ds.classes == ["closed", "open"], f"Unexpected classes: {train_ds.classes}"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    total_loss, correct, n = 0.0, 0, 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        # label 0="closed" -> target 1 (closed=positive class), label 1="open" -> target 0
        targets = (labels == 0).float().to(device)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == targets).sum().item()
        n += images.size(0)

    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data/eye_state")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="../models/eye_cnn.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_loaders(args.data_dir, args.batch_size)

    model = EyeStateCNN().to(device)
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
