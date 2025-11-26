#!/usr/bin/env python3
"""
Train MNIST using AdvancedImageProcessor preprocessing (advanced pipeline used at inference).
Saves model to `models/mnist_cnn_advanced.pt`.
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from advanced_image_processing import AdvancedImageProcessor
import numpy as np

class AdvancedMNISTTransform:
    def __init__(self, imgsz=28):
        self.imgsz = imgsz
        self.processor = AdvancedImageProcessor()

    def __call__(self, pil_img):
        # pil_img: PIL Image
        arr = np.array(pil_img.convert('L'), dtype=np.float32)
        arr = self.processor.invert_if_needed(arr)
        arr = self.processor.laplacian_sharpening(arr)
        arr = self.processor.adaptive_histogram_equalization(arr, window_size=8)
        arr = self.processor.resize_bilinear(arr, (self.imgsz, self.imgsz))
        arr = self.processor.normalize(arr)
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        return tensor

class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2,2)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="models/mnist_cnn_advanced.pt")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    device = torch.device(args.device)

    transform = AdvancedMNISTTransform(imgsz=28)
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

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

        model.eval()
        tot, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                tot += loss.item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                n += y.size(0)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} | test_loss={tot/n:.4f} | test_acc={correct/n:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved model weights to {args.out}")

if __name__ == '__main__':
    main()
