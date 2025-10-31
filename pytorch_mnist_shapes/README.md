# PyTorch CNN: MNIST + Shape Detection (circle vs rectangle)

## 1) Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2) Train MNIST
```bash
python mnist_torch.py --epochs 5 --batch 128 --out models/mnist_cnn.pt
```

## 3) Train shape classifier (synthetic data)
```bash
python shapes_torch.py --samples 6000 --imgsz 64 --epochs 8 --batch 128 --out models/shapes_cnn.pt
```

## 4) Predict a single shape image
```bash
python predict_shapes_torch.py --img hinhtron.jpg --model models/shapes_cnn.pt
```

Notes:
- Scripts auto-save a `pred_grid.png` in the model folder to visualize predictions.
- Reduce `--samples` or `--epochs` if your machine is slow.
- To add more classes (e.g., triangle), extend data generation + last layer size and label list.
