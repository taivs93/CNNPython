#!/usr/bin/env python3
"""
PyTorch shapes (circle vs rectangle) using synthetic data generated with OpenCV.
Usage:
  python shapes_torch.py --samples 6000 --imgsz 64 --epochs 8 --batch 128 --out models/shapes_cnn.pt
"""
import argparse, os, random, math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

LABELS = ["circle", "rectangle"]

def gen_one(imgsz=64):
    img = np.zeros((imgsz, imgsz), dtype=np.uint8)
    margin = imgsz // 10
    cls = random.choice([0,1])
    if cls == 0:
        r = random.randint(imgsz//8, imgsz//3)
        cx = random.randint(margin+r, imgsz-margin-r)
        cy = random.randint(margin+r, imgsz-margin-r)
        cv2.circle(img, (cx,cy), r, 255, -1)
    else:
        w = random.randint(imgsz//5, imgsz//2)
        h = random.randint(imgsz//5, imgsz//2)
        x1 = random.randint(margin, imgsz-margin-w)
        y1 = random.randint(margin, imgsz-margin-h)
        cv2.rectangle(img, (x1,y1), (x1+w, y1+h), 255, -1)

    if random.random() < 0.5:
        img = cv2.GaussianBlur(img, (3,3), 0)
    noise_level = random.randint(0, 12)
    if noise_level:
        noise = np.random.randint(0, noise_level, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
    return img, cls

class ShapesDataset(Dataset):
    def __init__(self, n=6000, imgsz=64):
        self.imgsz = imgsz
        X = np.zeros((n, imgsz, imgsz), dtype=np.uint8)
        y = np.zeros((n,), dtype=np.int64)
        for i in range(n):
            img, lab = gen_one(imgsz)
            X[i] = img
            y[i] = lab
        # store as float32 normalized
        self.X = (X.astype(np.float32) / 255.0)[:, None, :, :]  # add channel dim
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)

class ShapesCNN(nn.Module):
    def __init__(self, in_ch=1, num_classes=2, imgsz=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.25)
        self.fc   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.gap(x).view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--imgsz", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="models/shapes_cnn.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    ds = ShapesDataset(args.samples, args.imgsz)
    n_val = int(0.2 * len(ds))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = ShapesCNN(imgsz=args.imgsz).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs+1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total += loss.item()*x.size(0)
        train_loss = total/len(train_loader.dataset)

        # eval
        model.eval()
        tot_loss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                tot_loss += loss.item()*x.size(0)
                correct += (logits.argmax(1)==y).sum().item()
                n += y.size(0)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={tot_loss/n:.4f} | val_acc={correct/n:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved model weights to {args.out}")

    # save prediction grid
    import numpy as np
    model.eval()
    idx = np.random.choice(len(val_ds), size=min(16, len(val_ds)), replace=False)
    imgs = torch.stack([val_ds[i][0] for i in idx]).to(device)
    lbls = torch.tensor([val_ds[i][1] for i in idx])
    with torch.no_grad():
        preds = model(imgs).argmax(1).cpu().numpy()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4,4, figsize=(6,6))
    for ax, img, p, t in zip(axes.ravel(), imgs.cpu(), preds, lbls):
        ax.imshow(img.squeeze().cpu().numpy(), cmap='gray')
        ax.set_title(f"P:{LABELS[int(p)]}/T:{LABELS[int(t)]}", fontsize=8)
        ax.axis('off')
    fig.tight_layout()
    png_path = os.path.join(os.path.dirname(args.out), "pred_grid.png")
    fig.savefig(png_path)
    print(f"Saved prediction grid to {png_path}")

if __name__ == "__main__":
    main()
