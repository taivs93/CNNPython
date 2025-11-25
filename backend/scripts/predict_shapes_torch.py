# backend/scripts/predict_shapes_torch.py
#!/usr/bin/env python3
"""
Predict a single image for the shapes CNN (PyTorch).
"""

import argparse, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from advanced_image_processing import AdvancedImageProcessor

LABELS = ['circle','rectangle']

class ShapesCNN(nn.Module):
    def __init__(self, in_ch=1, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.25)
        self.fc = nn.Linear(128,num_classes)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.gap(x).view(x.size(0),-1)
        x = self.drop(x)
        x = self.fc(x)
        return x

def preprocess(img_path,imgsz=64):
    processor = AdvancedImageProcessor()
    img = processor.preprocess_shapes_advanced(img_path, imgsz)
    img = img[None, :, :]  # Add batch dimension
    return torch.from_numpy(img)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img',required=True)
    ap.add_argument('--model',default='backend/scripts/model1/shapes_cnn.pt')
    ap.add_argument('--device',default='cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    model = ShapesCNN().to(device)
    model.load_state_dict(torch.load(args.model,map_location=device))
    model.eval()

    x = preprocess(args.img).to(device)
    with torch.no_grad():
        logits = model(x)
        confidence = torch.softmax(logits, 1)[0]
        pred = int(confidence.argmax())
        confidence_score = float(confidence[pred])
    print(f'Prediction: {LABELS[pred]}, Confidence: {confidence_score:.4f}')
    return LABELS[pred], confidence_score

if __name__=='__main__':
    label, conf = main()
    print(f'Result: {label} ({conf*100:.2f}%)')