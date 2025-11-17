# backend/utils/run_predict.py
import subprocess
import json

def predict_mnist(img_path):
    result = subprocess.run(['python','scripts/predict_mnist_torch.py','--img',img_path], capture_output=True, text=True)
    # output: Predicted digit: X
    for line in result.stdout.splitlines():
        if line.startswith('Predicted digit:'):
            pred = int(line.split(':')[-1].strip())
            return {'prediction':pred}
    return {'prediction':None}

def predict_shapes(img_path):
    result = subprocess.run(['python','scripts/predict_shapes_torch.py','--img',img_path], capture_output=True, text=True)
    # output: Prediction: circle  (probs=[...])
    for line in result.stdout.splitlines():
        if line.startswith('Prediction:'):
            parts = line.split('(')[0].split(':')[-1].strip()
            return {'prediction':parts}
    return {'prediction':None}
