# backend/scripts/predict_mnist_torch.py
#!/usr/bin/env python3
"""
Predict handwritten digit from image using trained MNIST model.
"""

import argparse, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from advanced_image_processing import AdvancedImageProcessor

class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.conv2 = nn.Conv2d(32,64,3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*5*5,128)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128,10)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def preprocess(img_path):
    processor = AdvancedImageProcessor()
    img = processor.preprocess_mnist_advanced(img_path)
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img',required=True)
    ap.add_argument('--model','--model',default='backend/scripts/models/mnist_cnn.pt')
    ap.add_argument('--device',default='cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    model = MnistCNN().to(device)
    model.load_state_dict(torch.load(args.model,map_location=device))
    model.eval()

    x = preprocess(args.img).to(device)
    with torch.no_grad():
        logits = model(x)
        confidence = torch.softmax(logits, 1)[0]
        pred = int(confidence.argmax())
        confidence_score = float(confidence[pred])
    print(f'Predicted digit: {pred}, Confidence: {confidence_score:.4f}')
    return pred, confidence_score

if __name__=='__main__':
    pred, conf = main()
    print(f'Result: {pred} ({conf*100:.2f}%)')