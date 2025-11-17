#!/usr/bin/env python3
"""
PyTorch MNIST CNN: train, evaluate, and export
Usage:
  python mnist_torch.py --epochs 5 --batch 128 --out models/mnist_cnn.pt
"""
import argparse, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)   # 26x26
        self.conv2 = nn.Conv2d(32, 64, 3)  # 24x24
        self.pool = nn.MaxPool2d(2,2)      # -> 12x12 after pool
        self.fc1 = nn.Linear(64*5*5, 128)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def accuracy(logits, y):
    return (logits.argmax(1)==y).float().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="models/mnist_cnn.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    tfm = transforms.Compose([transforms.ToTensor()])  # scales to [0,1]
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="data", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = MnistCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs+1):
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

        # eval
        model.eval()
        tot_acc, tot_n, tot_loss = 0.0, 0, 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                tot_loss += loss.item() * x.size(0)
                tot_acc += (logits.argmax(1)==y).sum().item()
                tot_n += y.size(0)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} | test_loss={tot_loss/tot_n:.4f} | test_acc={tot_acc/tot_n:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved model weights to {args.out}")

    # Save prediction grid
    import numpy as np
    model.eval()
    samples = []
    for i in range(16):
        x, _ = test_ds[i]
        samples.append(x)
    grid = torch.stack(samples).to(device)
    with torch.no_grad():
        preds = model(grid).argmax(1).cpu().numpy()

    fig, axes = plt.subplots(4,4, figsize=(5,5))
    for ax, img, p in zip(axes.ravel(), grid.cpu(), preds):
        ax.imshow(img.squeeze().numpy(), cmap='gray')
        ax.set_title(str(int(p)))
        ax.axis('off')
    fig.tight_layout()
    png_path = os.path.join(os.path.dirname(args.out), "pred_grid.png")
    fig.savefig(png_path)
    print(f"Saved prediction grid to {png_path}")

if __name__ == "__main__":
    main()