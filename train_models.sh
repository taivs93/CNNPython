#!/bin/bash
# Training script for MNIST and Shapes models
# Usage: bash train_models.sh

cd "$(dirname "$0")"
cd backend/scripts

echo "=========================================="
echo "Training MNIST CNN Model"
echo "=========================================="
python mnist_torch.py --epochs 10 --batch 128 --out models/mnist_cnn.pt

echo ""
echo "=========================================="
echo "Training Shapes CNN Model"
echo "=========================================="
python shapes_torch.py --samples 8000 --imgsz 64 --epochs 12 --batch 128 --out model1/shapes_cnn.pt

echo ""
echo "=========================================="
echo "✓ Training Complete!"
echo "=========================================="
