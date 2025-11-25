# So Sánh: Tự Code vs Thư Viện
## Dự Án của Bạn (ANN/MLP) vs Dự Án Hiện Tại (CNN)

---

## 📊 Bảng So Sánh Chi Tiết

### **Dự Án ANN/MLP của Bạn** (Khuyến Nghị Cao)

| Thành phần | Loại | Code | Chi tiết |
|-----------|------|------|---------|
| **Linear Layer** | Tự code | `nn.py` | Tự implement Matrix multiply, activation |
| **ReLU Activation** | Tự code | `nn.py` | Forward + Backward tự code |
| **Softmax** | Tự code | `nn.py` | Tự code (numerical stability) |
| **CrossEntropyLoss** | Tự code | `nn.py` | Tự code forward + backward |
| **Adam Optimizer** | Tự code | `nn.py` | Tự code exponential moving average |
| **Model (MLP)** | Tự code | `nn.py` | Net class tự code layer stack |
| **Data Loading** | Thư viện | torch | `torch.load()`, `torch.save()` |
| **Tensor Ops** | Thư viện | torch | Matrix mult, reshape, etc |
| **Image I/O** | Thư viện | PIL/OpenCV | File reading |
| **GUI** | Thư viện | tkinter | User interface |
| **NumPy** | Thư viện | numpy | Broadcasting, random |

**✅ Kết luận:** Tự code hầu hết logic ML, chỉ dùng torch/numpy/PIL cho infrastructure

---

### **Dự Án CNN Hiện Tại** (Hỗn Hợp)

| Thành phần | Loại | Nơi | Chi tiết |
|-----------|------|-----|---------|
| **Conv2d Layer** | ✅ Thư viện | `torch.nn` | `nn.Conv2d()` (không tự code) |
| **MaxPool2d** | ✅ Thư viện | `torch.nn` | `nn.MaxPool2d()` (không tự code) |
| **Linear Layer** | ✅ Thư viện | `torch.nn` | `nn.Linear()` (không tự code) |
| **ReLU** | ✅ Thư viện | `torch.nn.functional` | `F.relu()` |
| **Softmax** | ✅ Thư viện | `torch.nn` | `nn.Softmax()` hoặc CrossEntropyLoss |
| **CrossEntropyLoss** | ✅ Thư viện | `torch.nn` | `nn.CrossEntropyLoss()` |
| **Adam Optimizer** | ✅ Thư viện | `torch.optim` | `torch.optim.Adam()` |
| **Model (CNN)** | ✅ Thư viện | `torch.nn.Module` | Inherit nn.Module |
| **Sobel Edge Detection** | 🟡 Tự code | `advanced_image_processing.py` | Tự implement kernels + convolution |
| **Morphological Ops** | 🟡 Tự code | `advanced_image_processing.py` | Tự code dilate/erode |
| **Laplacian Sharpening** | 🟡 Tự code | `advanced_image_processing.py` | Tự code kernel |
| **Bilinear Interpolation** | 🟡 Tự code | `image_processing.py` | Tự code resize |
| **Gaussian Blur** | 🟡 Tự code | `image_processing.py` | Tự code 2D convolution |
| **Histogram Equalization** | 🟡 Tự code | `advanced_image_processing.py` | Tự code + scipy.ndimage |
| **Flask API** | ✅ Thư viện | `flask`, `flask-cors` | Web framework |
| **React** | ✅ Thư viện | `react`, `vite` | Frontend |
| **Data Loading** | ✅ Thư viện | `torch`, `torchvision` | `datasets.MNIST`, `DataLoader` |
| **Image I/O** | ✅ Thư viện | `PIL` | File reading |

**❌ Kết luận:** Tự code **xử lý ảnh**, nhưng dùng **torch.nn cho neural network** (khác với dự án ANN)

---

## 🎯 Phân Tích Chi Tiết

### **Neural Network Layers**

#### Dự Án ANN/MLP của Bạn:
```python
# nn.py - TỰ CODE
class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros((1, out_features))
    
    def forward(self, x):
        self.cache = x
        return x @ self.W + self.b
    
    def backward(self, grad_out):
        grad_W = self.cache.T @ grad_out
        grad_in = grad_out @ self.W.T
        return grad_in

class ReLU:
    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)
    
    def backward(self, grad_out):
        return grad_out * (self.cache > 0)
```

#### Dự Án CNN Hiện Tại:
```python
# app.py & mnist_torch.py - DÙNG THƯ VIỆN
class MnistCNN(nn.Module):  # ← Inherit torch.nn.Module
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # ← torch.nn (không tự code)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*5*5, 128)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # ← F.relu (thư viện)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
```

**🔴 KHÁC BIỆT:** Dự án ANN tự code layer, CNN dùng torch.nn

---

### **Image Processing**

#### Dự Án ANN của Bạn:
```python
# Chỉ dùng PIL/OpenCV đơn giản
img = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img / 255.0
```

#### Dự Án CNN Hiện Tại:
```python
# advanced_image_processing.py - TỰ CODE
def sobel_edge_detection(img):
    sobel_x = np.array([[-1, 0, 1],   # ← Tự define kernel
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
    gx = AdvancedImageProcessor._convolve(img, sobel_x)
    gy = AdvancedImageProcessor._convolve(img, sobel_y)
    edges = np.sqrt(gx**2 + gy**2)
    return edges

def _convolve(img, kernel):  # ← TỰ CODE 2D convolution
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    out = np.zeros_like(img)
    
    for i in range(h):
        for j in range(w):
            out[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    
    return out

def dilate(img, kernel=None):  # ← TỰ CODE morphology
    if kernel is None:
        kernel = np.ones((3, 3))
    h, w = img.shape
    kh, kw = kernel.shape
    out = np.zeros_like(img)
    padded = np.pad(img, ((kh//2, kh//2), (kw//2, kw//2)), mode='constant')
    
    for i in range(h):
        for j in range(w):
            out[i, j] = np.max(padded[i:i+kh, j:j+kw] * kernel)
    
    return out
```

**✅ GIỐNG:** Dự án ANN không focus vào xử lý ảnh phức tạp, CNN tự code toàn bộ

---

### **Optimizer & Loss**

#### Dự Án ANN của Bạn:
```python
# nn.py - TỰ CODE
class Adam:
    def __init__(self, lr=1e-3, betas=(0.9, 0.999)):
        self.m = {}  # momentum
        self.v = {}  # velocity
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.t = 0
    
    def step(self, params, grads):
        self.t += 1
        for p, g in zip(params, grads):
            self.m[id(p)] = self.beta1 * self.m.get(id(p), 0) + (1-self.beta1)*g
            self.v[id(p)] = self.beta2 * self.v.get(id(p), 0) + (1-self.beta2)*g**2
            
            m_hat = self.m[id(p)] / (1 - self.beta1**self.t)
            v_hat = self.v[id(p)] / (1 - self.beta2**self.t)
            
            p -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

class CrossEntropyLoss:
    def forward(self, logits, targets):
        # softmax
        # compute loss
        # return
```

#### Dự Án CNN Hiện Tại:
```python
# mnist_torch.py - DÙNG THƯ VIỆN
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, args.epochs+1):
    for x, y in train_loader:
        logits = model(x)
        loss = criterion(logits, y)  # ← Thư viện
        
        opt.zero_grad()
        loss.backward()  # ← Autograd (thư viện)
        opt.step()  # ← Thư viện
```

**🔴 KHÁC BIỆT:** Dự án ANN tự code Adam + CrossEntropyLoss, CNN dùng torch

---

## 📋 Bảng Tổng Hợp

```
┌─────────────────────────────────────────────────────────┐
│        THÀNH PHẦN              │  ANN/MLP  │   CNN      │
├─────────────────────────────────────────────────────────┤
│ Neural Network Layers (FC)     │  TỰ CODE  │ torch.nn   │
│ Conv/Pool Layers               │    -      │ torch.nn   │
│ Activation Functions           │  TỰ CODE  │ torch.nn   │
│ Loss Function                  │  TỰ CODE  │ torch.nn   │
│ Optimizer (Adam)               │  TỰ CODE  │ torch.optim│
│ Backpropagation                │  TỰ CODE  │ autograd   │
│ Image Processing Basic         │ PIL/OpenCV│ PIL        │
│ Image Processing Advanced      │    -      │  TỰ CODE   │
│ Tensor Operations              │ numpy     │ torch      │
│ File I/O (models)              │ pickle    │ torch      │
│ GUI                            │ tkinter   │ React      │
│ Web Framework                  │    -      │ Flask      │
└─────────────────────────────────────────────────────────┘
```

---

## ✅ Đánh Giá: Dự Án CNN Hiện Tại Có Giống ANN Không?

### **KHÔNG GIỐNG ở điểm chính:**
1. **Neural Network Layers** — ANN tự code, CNN dùng torch.nn ❌
2. **Optimizer/Loss** — ANN tự code, CNN dùng torch ❌

### **GIỐNG ở điểm:**
1. **Image Processing** — Cả hai tự code ✅ (CNN advanced hơn)
2. **Không dùng OpenCV** — Cả hai tối thiểu hóa thư viện ✅
3. **Learning Focus** — Cả hai coi trọng tự code để hiểu sâu ✅

---

## 🎯 Kết Luận

**Dự án CNN hiện tại của bạn:**

### So với ANN/MLP:
- ✅ Tự code xử lý ảnh → **ĐÚNG giống ANN**
- ✅ Không dùng OpenCV → **ĐÚNG giống ANN**
- ❌ Dùng torch.nn cho Neural Network → **KHÁC ANN**
- ❌ Không tự code Conv2d/MaxPool2d → **KHÁC ANN**

### Nếu muốn **HOÀN TOÀN giống** ANN:
Bạn sẽ cần:
1. Tự code Conv2d layer (cần implement 2D convolution forward + backward)
2. Tự code MaxPool2d layer
3. Tự code Dropout layer
4. Giữ torch chỉ cho tensor ops

### Nhận xét:
- **Dự án ANN** → Tối ưu cho **learning/giáo dục** (tự code mọi thứ)
- **Dự án CNN** → Tối ưu cho **production/đa dụng** (dùng PyTorch chuẩn)

**💡 Recommendation:** Dự án CNN hiện tại đã tốt cho production. Nếu muốn learning deeper, có thể tạo `cnn_manual.py` tự code Conv2d, nhưng đó là advanced exercise.

---

**Created:** 2025-11-25  
**Status:** Analysis Complete
