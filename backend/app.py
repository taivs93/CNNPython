#!/usr/bin/env python3
"""
Flask backend API for MNIST and Shapes predictions with confidence scores
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from werkzeug.utils import secure_filename
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from advanced_image_processing import AdvancedImageProcessor

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Models
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
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
class ShapesCNN(nn.Module):
    def __init__(self, in_ch=1, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.25)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.gap(x).view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x

# Load models
mnist_model = MnistCNN().to(device)
shapes_model = ShapesCNN().to(device)

mnist_model_path = os.path.join(os.path.dirname(__file__), 'scripts/models/mnist_cnn.pt')
shapes_model_path = os.path.join(os.path.dirname(__file__), 'scripts/model1/shapes_cnn.pt')

try:
    mnist_model.load_state_dict(torch.load(mnist_model_path, map_location=device))
    mnist_model.eval()
    print(f"✓ MNIST model loaded from {mnist_model_path}")
except Exception as e:
    print(f"⚠ Could not load MNIST model: {e}")

try:
    shapes_model.load_state_dict(torch.load(shapes_model_path, map_location=device))
    shapes_model.eval()
    print(f"✓ Shapes model loaded from {shapes_model_path}")
except Exception as e:
    print(f"⚠ Could not load Shapes model: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_mnist(img_path):
    """Predict digit with confidence"""
    try:
        processor = AdvancedImageProcessor()
        img = processor.preprocess_mnist_advanced(img_path)
        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = mnist_model(img)
            confidence = torch.softmax(logits, 1)[0]
            pred = int(confidence.argmax())
            confidence_score = float(confidence[pred])
        
        return {
            'success': True,
            'prediction': pred,
            'confidence': round(confidence_score, 4),
            'confidence_percent': round(confidence_score * 100, 2)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def predict_shapes(img_path):
    """Predict shape with confidence"""
    try:
        labels = ['circle', 'rectangle']
        processor = AdvancedImageProcessor()
        img = processor.preprocess_shapes_advanced(img_path, 64)
        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = shapes_model(img)
            confidence = torch.softmax(logits, 1)[0]
            pred = int(confidence.argmax())
            confidence_score = float(confidence[pred])
        
        return {
            'success': True,
            'prediction': labels[pred],
            'confidence': round(confidence_score, 4),
            'confidence_percent': round(confidence_score * 100, 2)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'device': str(device)})

@app.route('/api/predict/mnist', methods=['POST'])
def api_predict_mnist():
    """Predict handwritten digit"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = predict_mnist(filepath)
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict/shapes', methods=['POST'])
def api_predict_shapes():
    """Predict shape (circle or rectangle)"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = predict_shapes(filepath)
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
