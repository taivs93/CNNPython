#!/usr/bin/env python3
"""
Predict a single image for the shapes CNN (PyTorch).
Usage:
  python predict_shapes_torch.py --img path/to/image.png --model models/shapes_cnn.pt
"""
import argparse, os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS = ["circle", "rectangle"]

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

def preprocess(img_path, imgsz=64):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Cannot read image: {img_path}")
    if img.mean() > 127:  # background likely white
        img = 255 - img
    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    img = img.astype('float32')/255.0
    img = img[None, None, :, :]  # (1,1,H,W)
    return torch.from_numpy(img)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--model", default="models/shapes_cnn.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = ShapesCNN().to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    x = preprocess(args.img, imgsz=64).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
    print(f"Prediction: {LABELS[pred]}  (probs={probs})")

if __name__ == "__main__":
    main()
