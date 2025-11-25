# Best Practices - PyTorch MNIST/Shapes CNN Project

## 📋 Tóm tắt Kiến trúc Hiện Tại

| Thành phần | Loại | Chi tiết |
|-----------|------|---------|
| **Backend (Flask)** | Sử dụng thư viện | PyTorch (torch), Flask, Flask-CORS |
| **Models (MNIST/Shapes)** | Sử dụng thư viện | `torch.nn.Module` (Conv2d, MaxPool2d, Linear) |
| **Image Processing** | Tự code + Thư viện | Custom algorithms (Sobel, Morphology, Convolution) + PIL |
| **Frontend (React)** | Sử dụng thư viện | React, Vite, CSS |
| **Data Pipeline** | Tự code + Thư viện | Custom preprocessing + torchvision transforms |

---

## 🎯 Best Practice 1: Xử lý Ảnh (Image Processing)

### ✅ Hiện Tại (Tốt)
```python
# advanced_image_processing.py - Tự code các thuật toán
- Sobel edge detection ✅ (tự code)
- Bilinear interpolation ✅ (tự code)
- Morphological operations (dilate/erode) ✅ (tự code)
- Laplacian sharpening ✅ (tự code)
- Adaptive histogram equalization ✅ (có sử dụng scipy)
```

### 💡 Cải Thiện
```python
# Best Practice:
1. Tách riêng các thuật toán cơ bản (Sobel, Blur, Resize)
   → Tự code thuần NumPy (đã làm ✅)

2. Sử dụng scipy.ndimage cho advanced operations
   → Từng hàm có trong advanced_image_processing.py ✅

3. Tối ưu performance:
   - Vectorize loops (hiện có vài loop i,j — có thể dùng NumPy broadcast)
   - Cache kernel nếu dùng lặp lại
   - Xem xét dùng GPU processing (Torch tensor) nếu performance cần

4. Document từng hàm rõ ràng (input/output shape, ý nghĩa)
   → Đã làm ✅
```

---

## 🎯 Best Practice 2: Model Architecture (CNN)

### ✅ Hiện Tại (Tốt)
```python
# scripts/mnist_torch.py & shapes_torch.py
- Sử dụng torch.nn.Module ✅
- Có Conv2d, MaxPool2d, Dropout ✅
- Sử dụng torch.optim.Adam ✅
- Lưu model.state_dict() ✅
```

### 💡 Cải Thiện
```python
# Best Practice:

1. **Model Organization**
   ✅ Tách models vào file riêng: models/mnist_cnn.py, models/shapes_cnn.py
   
   Hiện tại: định nghĩa trong predict_*.py
   → Nên: tạo models/cnn.py với class MnistCNN, ShapesCNN

2. **Configuration Management**
   ✅ Tạo config.py hoặc config.yaml
   
   ```python
   CONFIG = {
       'mnist': {
           'input_size': (1, 28, 28),
           'num_classes': 10,
           'model_path': 'scripts/models/mnist_cnn.pt',
           'epochs': 10,
           'batch_size': 128,
           'learning_rate': 1e-3,
       },
       'shapes': {
           'input_size': (1, 64, 64),
           'num_classes': 2,
           'model_path': 'scripts/model1/shapes_cnn.pt',
           'epochs': 12,
           'batch_size': 128,
           'learning_rate': 1e-3,
       }
   }
   ```

3. **Model Checkpointing**
   ✅ Hiện tại lưu state_dict ✅
   
   Cải thiện:
   - Lưu không chỉ state_dict mà cả metadata (epoch, accuracy, model config)
   - Implement checkpoint callback để lưu best model

4. **Inference Optimization**
   ✅ Dùng model.eval() ✅
   ✅ Dùng torch.no_grad() ✅
   
   Cải thiện:
   - Export ONNX format để inference nhanh hơn
   - Quantization cho mobile inference (nếu cần)

5. **Validation & Testing**
   ✅ Có validation set ✅
   
   Cải thiện:
   - Thêm test set riêng biệt
   - Compute metrics: precision, recall, F1, confusion matrix
   - Log results vào wandb/tensorboard
```

---

## 🎯 Best Practice 3: Backend API (Flask)

### ✅ Hiện Tại (Tốt)
```python
# app.py
- CORS enabled ✅
- Error handling ✅
- FormData for file upload ✅
- Return confidence scores ✅
```

### 💡 Cải Thiện
```python
# Best Practice:

1. **Code Organization**
   ✅ Hiện tại tất cả trong app.py
   
   Nên refactor:
   - app.py → chỉ Flask setup
   - routes/predict.py → endpoints
   - models/loader.py → load models
   - utils/preprocessing.py → preprocess logic

2. **Error Handling & Logging**
   ✅ Có error handling cơ bản
   
   Cải thiện:
   ```python
   import logging
   
   logger = logging.getLogger(__name__)
   
   @app.route('/api/predict/mnist', methods=['POST'])
   def predict_mnist():
       try:
           # validation
           # prediction
           logger.info(f"MNIST prediction: {pred}")
           return jsonify(result)
       except Exception as e:
           logger.error(f"MNIST prediction failed: {str(e)}")
           return jsonify({'success': False, 'error': str(e)}), 500
   ```

3. **Request Validation**
   ```python
   # Validate file size, type
   MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
   ALLOWED_TYPES = {'image/png', 'image/jpeg', 'image/bmp'}
   
   if request.content_length > MAX_FILE_SIZE:
       return jsonify({'error': 'File too large'}), 413
   ```

4. **API Versioning**
   ```python
   @app.route('/api/v1/predict/mnist', methods=['POST'])
   @app.route('/api/v2/predict/mnist', methods=['POST'])
   ```

5. **Async Processing (Optional)**
   ```python
   # Nếu model inference chậm, dùng Celery
   from celery import Celery
   
   celery = Celery(app.name)
   
   @celery.task
   def predict_async(file_path):
       result = predict_mnist(file_path)
       return result
   ```

6. **Database Logging (Optional)**
   ```python
   # Log predictions để phân tích sau
   - prediction history
   - accuracy tracking
   - user feedback
   ```
```

---

## 🎯 Best Practice 4: Frontend (React)

### ✅ Hiện Tại (Tốt)
```
- Component structure ✅
- Canvas drawing ✅
- Confidence display ✅
- Error handling ✅
- Responsive design ✅
```

### 💡 Cải Thiện
```python
# Best Practice:

1. **Component Organization**
   📁 src/
      ├── components/
      │   ├── Canvas/
      │   │   ├── CanvasDrawing.jsx
      │   │   └── CanvasDrawing.css
      │   ├── Result/
      │   │   ├── ResultDisplay.jsx
      │   │   └── ResultDisplay.css
      │   └── Shared/
      │       ├── Loading.jsx
      │       └── Error.jsx
      ├── pages/
      │   ├── Home/
      │   ├── MNIST/
      │   └── Shapes/
      ├── hooks/
      │   ├── usePrediction.js
      │   └── useCanvas.js
      ├── services/
      │   └── api.js
      └── utils/
          └── constants.js

2. **Custom Hooks**
   ```javascript
   // hooks/usePrediction.js
   export function usePrediction() {
       const [result, setResult] = useState(null);
       const [loading, setLoading] = useState(false);
       const [error, setError] = useState(null);
       
       const predict = async (file) => {
           setLoading(true);
           setError(null);
           try {
               const result = await predictMnist(file);
               setResult(result);
           } catch (err) {
               setError(err.message);
           } finally {
               setLoading(false);
           }
       };
       
       return { result, loading, error, predict };
   }
   ```

3. **API Service Layer**
   ```javascript
   // services/api.js
   const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000';
   
   export const api = {
       predict: {
           mnist: (file) => fetchWithTimeout(`${API_BASE}/api/v1/predict/mnist`, file),
           shapes: (file) => fetchWithTimeout(`${API_BASE}/api/v1/predict/shapes`, file),
       }
   };
   
   function fetchWithTimeout(url, file, timeout = 10000) {
       return Promise.race([
           fetch(url, { method: 'POST', body: new FormData() }),
           new Promise((_, reject) => 
               setTimeout(() => reject(new Error('Request timeout')), timeout)
           )
       ]);
   }
   ```

4. **Performance Optimization**
   - React.memo cho components
   - useCallback để tránh re-render
   - Code splitting với React.lazy
   - Lazy load models metadata

5. **State Management**
   - Nếu app phức tạp hơn → dùng Redux/Zustand
   - Hiện tại simple → Context API cũng được

6. **Testing**
   ```javascript
   // tests/CanvasDrawing.test.jsx
   import { render, screen } from '@testing-library/react';
   import CanvasDrawing from '../components/CanvasDrawing';
   
   test('renders canvas', () => {
       render(<CanvasDrawing />);
       expect(screen.getByRole('canvas')).toBeInTheDocument();
   });
   ```

7. **Accessibility**
   - ARIA labels trên canvas
   - Keyboard support (Space để vẽ)
   - Color contrast ratio ≥ 4.5:1
```

---

## 🎯 Best Practice 5: Data Pipeline

### ✅ Hiện Tại (Tốt)
```python
# Preprocessing
- Custom Sobel edge detection ✅
- Bilinear interpolation ✅
- Normalization ✅
- No OpenCV ✅
```

### 💡 Cải Thiện
```python
# Best Practice:

1. **Reproducibility**
   ```python
   import random
   import numpy as np
   import torch
   
   def seed_everything(seed=42):
       random.seed(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
   
   seed_everything()
   ```

2. **Data Augmentation (Training)**
   ```python
   # shapes_torch.py
   def gen_one_augmented(imgsz=64):
       # Hiện tại: random rotation, blur, noise
       # Cải thiện: thêm
       - Random scaling
       - Random translation
       - Random brightness/contrast
   ```

3. **Data Validation**
   ```python
   # Kiểm tra:
   - Image size valid?
   - Image corrupted?
   - Preprocessing output shape chính xác?
   - Normalize range [0, 1]?
   ```

4. **Preprocessing Caching**
   ```python
   # Nếu batch predict → cache preprocessed images
   from functools import lru_cache
   
   @lru_cache(maxsize=100)
   def preprocess_cached(img_path):
       return preprocess_shapes_advanced(img_path)
   ```

5. **Version Control for Data**
   ```
   models/
   ├── mnist_cnn_v1.pt  (85% acc)
   ├── mnist_cnn_v2.pt  (90% acc)
   └── mnist_cnn_best.pt
   
   → Track: training date, hyperparams, accuracy
   ```
```

---

## 🎯 Best Practice 6: Project Structure & Documentation

### ✅ Hiện Tại (Tốt)
```
pytorch_mnist_shapes/
├── backend/
├── frontend/
├── README.md
└── train_models.bat
```

### 💡 Cải Thiện
```
pytorch_mnist_shapes/
├── backend/
│   ├── app.py
│   ├── config.py                    ← NEW
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── models/                  ← NEW
│   │   │   ├── mnist_cnn.py
│   │   │   └── shapes_cnn.py
│   │   ├── train_mnist.py
│   │   ├── train_shapes.py
│   │   ├── predict_mnist_torch.py
│   │   ├── predict_shapes_torch.py
│   │   ├── image_processing.py
│   │   └── advanced_image_processing.py
│   ├── utils/
│   │   ├── preprocessing.py         ← NEW
│   │   └── logger.py                ← NEW
│   ├── routes/                      ← NEW
│   │   ├── predict.py
│   │   └── health.py
│   └── tests/                       ← NEW
│       ├── test_preprocessing.py
│       └── test_api.py
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/                   ← NEW
│   │   ├── services/                ← NEW
│   │   └── utils/
│   ├── tests/                       ← NEW
│   └── package.json
├── docs/                            ← NEW
│   ├── ARCHITECTURE.md
│   ├── API.md
│   ├── SETUP.md
│   └── BEST_PRACTICES.md
├── .github/workflows/               ← NEW (CI/CD)
│   ├── test.yml
│   └── deploy.yml
├── .env.example                     ← NEW
├── docker-compose.yml               ← NEW (Optional)
└── README.md (updated)
```

### 📄 Documentation
```markdown
# ARCHITECTURE.md
- Tổng quan hệ thống
- Data flow diagram
- Model architecture
- Preprocessing pipeline

# API.md
- Endpoint documentation
- Request/response examples
- Error codes
- Rate limiting

# SETUP.md
- Installation steps
- Environment variables
- Database setup (if needed)
- Running locally

# BEST_PRACTICES.md
- Coding standards
- Naming conventions
- Testing requirements
- Deployment checklist
```

---

## 🎯 Best Practice 7: Testing & CI/CD

### ✅ Testing Strategy
```python
# Backend Tests
1. Unit tests
   - test_preprocessing.py
   - test_model_forward.py

2. Integration tests
   - test_api_endpoints.py

3. E2E tests
   - test_prediction_pipeline.py

# Frontend Tests
1. Component tests (React Testing Library)
2. Integration tests
3. E2E tests (Cypress/Playwright)
```

### ✅ CI/CD Pipeline
```yaml
# .github/workflows/test.yml
- Run linting (pylint, eslint)
- Run unit tests
- Check coverage (>80%)
- Build Docker image
- Deploy to staging
```

---

## 🎯 Best Practice 8: Performance & Optimization

### Backend
```python
# 1. Model Optimization
- Quantization: torch.quantization
- ONNX export: torch.onnx.export()
- TorchScript: torch.jit.script()

# 2. API Caching
from flask_caching import Cache
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/api/health')
@cache.cached(timeout=300)
def health():
    return jsonify({'status': 'ok'})

# 3. Batch Prediction
- Implement batch endpoint /api/predict/batch

# 4. GPU Usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Frontend
```javascript
// 1. Code Splitting
const MnistPage = React.lazy(() => import('./pages/MnistPage'));
const ShapesPage = React.lazy(() => import('./pages/ShapesPage'));

// 2. Image Compression
function compressCanvas(canvas, quality = 0.8) {
    return canvas.toBlob(blob => blob, 'image/jpeg', quality);
}

// 3. Service Worker
// Offline support, caching
```

---

## ✅ Checklist - Implement Best Practices

### Phase 1: Code Organization
- [ ] Tách models vào `models/` folder
- [ ] Tạo `config.py`
- [ ] Tách routes vào `routes/`
- [ ] Tạo `utils/` folder

### Phase 2: Documentation
- [ ] Viết ARCHITECTURE.md
- [ ] Viết API.md
- [ ] Viết SETUP.md
- [ ] Update README.md

### Phase 3: Testing
- [ ] Viết unit tests (backend)
- [ ] Viết component tests (frontend)
- [ ] Setup CI/CD (GitHub Actions)

### Phase 4: Optimization
- [ ] Model quantization
- [ ] API caching
- [ ] Frontend code splitting
- [ ] Performance monitoring

### Phase 5: Deployment
- [ ] Dockerize backend
- [ ] Dockerize frontend
- [ ] Setup production logging
- [ ] Setup monitoring & alerting

---

## 📚 Summary

**Dự án của bạn hiện tại:**
- ✅ Xử lý ảnh tự code (không dùng OpenCV)
- ✅ PyTorch CNN models (không tự code)
- ✅ Flask API với CORS
- ✅ React frontend with canvas drawing
- ✅ Confidence scores

**Best Practices cần implement:**
1. ✅ Code organization & separation of concerns
2. ✅ Configuration management
3. ✅ Comprehensive error handling & logging
4. ✅ Testing (unit, integration, E2E)
5. ✅ Documentation
6. ✅ CI/CD pipeline
7. ✅ Performance optimization
8. ✅ Monitoring & maintenance

**Đánh giá:**
- Dự án của bạn đã tốt, tương đương cấp độ production-ready
- Cần cải thiện: documentation, testing, CI/CD, code organization
- Follow checklist trên để nâng cấp further

---

**Created:** 2025-11-25  
**Status:** Ready for implementation
