# backend/utils/run_predict.py
import subprocess
import json
import os

def predict_mnist(img_path):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    result = subprocess.run(
        ['python','scripts/predict_mnist_torch.py','--img',img_path,'--model','scripts/models/mnist_cnn.pt'],
        capture_output=True, text=True, cwd=base_dir)
    # output: Predicted digit: X
    for line in result.stdout.splitlines():
        if line.startswith('Predicted digit:'):
            pred = int(line.split(':')[-1].strip())
            return {'prediction':pred}
    return {'prediction':None}

def predict_shapes(img_path):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    result = subprocess.run(
        ['python','scripts/predict_shapes_torch.py','--img',img_path,'--model','scripts/model1/shapes_cnn.pt'],
        capture_output=True, text=True, cwd=base_dir)
    # output: Prediction: circle  (probs=[...])
    for line in result.stdout.splitlines():
        if line.startswith('Prediction:'):
            parts = line.split('(')[0].split(':')[-1].strip()
            return {'prediction':parts}
    return {'prediction':None}
