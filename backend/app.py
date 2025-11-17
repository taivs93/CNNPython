# backend/app.py
from flask import Flask, request, jsonify
from utils.save_temp import save_temp_image
from utils.run_predict import predict_mnist, predict_shapes
import os

app = Flask(__name__)

@app.route('/predict/mnist', methods=['POST'])
def predict_mnist_route():
    data = request.json
    image_b64 = data.get('image')
    if not image_b64:
        return jsonify({'error':'No image provided'}),400
    tmp_path = save_temp_image(image_b64, 'mnist')
    result = predict_mnist(tmp_path)
    os.remove(tmp_path)
    return jsonify(result)

@app.route('/predict/shapes', methods=['POST'])
def predict_shapes_route():
    data = request.json
    image_b64 = data.get('image')
    if not image_b64:
        return jsonify({'error':'No image provided'}),400
    tmp_path = save_temp_image(image_b64, 'shapes')
    result = predict_shapes(tmp_path)
    os.remove(tmp_path)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
