#!/usr/bin/env python3
"""
Train a stronger MNIST CNN model using PyTorch.
Includes data augmentation, BatchNorm, Dropout, AdamW, LR scheduler.
"""

import argparse, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -----------------------------
# 1. Model Definition (Improved)
# -----------------------------
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   # 28x28
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 28x28
        self.bn2 = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(2, 2)               # 28 -> 14

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 14x14
        self.bn3 = nn.BatchNorm2d(128)

        self.pool2 = nn.MaxPool2d(2, 2)               # 14 -> 7

        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# -----------------------------
# 2. Train Script
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="backend/scripts/models/mnist_cnn.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[INFO] Training on device: {device}")

    # -----------------------------
    # Data Augmentation + Normalization
    # -----------------------------
    tfm_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST("data", train=True, download=True, transform=tfm_train)
    test_ds  = datasets.MNIST("data", train=False, download=True, transform=tfm_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    # -----------------------------
    # Model + Optimizer + Scheduler
    # -----------------------------
    model = MnistCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            running += loss.item() * x.size(0)

        train_loss = running / len(train_loader.dataset)

        # Evaluation
        model.eval()
        tot_loss, tot_acc, tot_n = 0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                tot_loss += loss.item() * x.size(0)
                tot_acc += (logits.argmax(1) == y).sum().item()
                tot_n += y.size(0)

        print(f"Epoch {epoch}: "
              f"train_loss={train_loss:.4f} | "
              f"test_loss={tot_loss/tot_n:.4f} | "
              f"test_acc={tot_acc/tot_n:.4f}")

        scheduler.step()

    # -----------------------------
    # Save model
    # -----------------------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"[INFO] Saved optimized model to {args.out}")

    # Save prediction grid for visualization
    samples = torch.stack([test_ds[i][0] for i in range(16)]).to(device)
    with torch.no_grad():
        preds = model(samples).argmax(1).cpu().numpy()

    fig, axes = plt.subplots(4, 4, figsize=(5, 5))
    for ax, img, p in zip(axes.ravel(), samples.cpu(), preds):
        ax.imshow(img.squeeze().numpy(), cmap="gray")
        ax.set_title(str(int(p)))
        ax.axis("off")
    fig.tight_layout()

    png_path = os.path.join(os.path.dirname(args.out), "pred_grid.png")
    fig.savefig(png_path)
    print(f"[INFO] Saved prediction grid to {png_path}")


if __name__ == "__main__":
    main()
