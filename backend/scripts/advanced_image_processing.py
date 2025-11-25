#!/usr/bin/env python3
"""
Advanced Image Processing - Self-implemented algorithms for learning
Includes: Sobel edge detection, morphological operations, custom convolution
"""

import numpy as np
from scipy import ndimage
from PIL import Image

class AdvancedImageProcessor:
    """Advanced image processing with self-implemented algorithms"""
    
    @staticmethod
    def read_image(img_path):
        """Read image and convert to grayscale"""
        img = Image.open(img_path).convert('L')
        return np.array(img, dtype=np.float32)
    
    @staticmethod
    def normalize(img):
        """Normalize image to [0, 1]"""
        return img / 255.0
    
    @staticmethod
    def denormalize(img):
        """Denormalize from [0, 1] to [0, 255]"""
        return np.clip(img * 255, 0, 255).astype(np.uint8)
    
    @staticmethod
    def invert_if_needed(img):
        """Invert image if background is white (mean > 127)"""
        if img.mean() > 127:
            img = 255 - img
        return img
    
    @staticmethod
    def sobel_edge_detection(img):
        """
        Sobel edge detection (self-implemented)
        Detects edges by computing gradient magnitude
        """
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
        
        sobel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=np.float32)
        
        # Apply convolution
        gx = AdvancedImageProcessor._convolve(img, sobel_x)
        gy = AdvancedImageProcessor._convolve(img, sobel_y)
        
        # Compute gradient magnitude
        edges = np.sqrt(gx**2 + gy**2)
        
        # Normalize
        edges = np.clip(edges, 0, 255)
        return edges.astype(np.uint8)
    
    @staticmethod
    def laplacian_sharpening(img):
        """
        Laplacian filter for edge enhancement (self-implemented)
        """
        laplacian = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]], dtype=np.float32)
        
        sharpened = AdvancedImageProcessor._convolve(img, laplacian)
        
        # Combine with original for enhancement
        result = img + 0.3 * sharpened
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def _convolve(img, kernel):
        """
        2D Convolution (self-implemented)
        Applies kernel to image with reflection padding
        """
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        
        # Pad image with reflection
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        
        h, w = img.shape
        output = np.zeros_like(img, dtype=np.float32)
        
        # Apply convolution
        for i in range(h):
            for j in range(w):
                region = padded[i:i+k_h, j:j+k_w]
                output[i, j] = np.sum(region * kernel)
        
        return output
    
    @staticmethod
    def dilate(img, kernel_size=3, iterations=1):
        """
        Morphological dilation (self-implemented)
        Expands white regions
        """
        result = img.copy()
        
        for _ in range(iterations):
            # Create kernel
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            pad = kernel_size // 2
            padded = np.pad(result, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
            
            h, w = result.shape
            dilated = np.zeros_like(result)
            
            for i in range(h):
                for j in range(w):
                    region = padded[i:i+kernel_size, j:j+kernel_size]
                    dilated[i, j] = np.max(region * kernel)
            
            result = dilated
        
        return result.astype(np.uint8)
    
    @staticmethod
    def erode(img, kernel_size=3, iterations=1):
        """
        Morphological erosion (self-implemented)
        Shrinks white regions
        """
        result = img.copy()
        
        for _ in range(iterations):
            # Create kernel
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            pad = kernel_size // 2
            padded = np.pad(result, ((pad, pad), (pad, pad)), mode='constant', constant_values=255)
            
            h, w = result.shape
            eroded = np.zeros_like(result)
            
            for i in range(h):
                for j in range(w):
                    region = padded[i:i+kernel_size, j:j+kernel_size]
                    # Minimum of the region (for erosion)
                    eroded[i, j] = np.min(region * kernel / kernel) if np.any(kernel) else 0
            
            result = eroded
        
        return result.astype(np.uint8)
    
    @staticmethod
    def adaptive_histogram_equalization(img, window_size=16):
        """
        Adaptive histogram equalization for contrast enhancement
        Improves local contrast
        """
        from scipy.ndimage import uniform_filter
        
        # Compute local mean
        local_mean = uniform_filter(img.astype(np.float32), size=window_size)
        
        # Enhance contrast
        enhanced = img.astype(np.float32) - local_mean + 128
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    @staticmethod
    def resize_bilinear(img, new_size):
        """Bilinear interpolation resize (self-implemented)"""
        h, w = img.shape
        new_h, new_w = new_size
        
        output = np.zeros(new_size, dtype=np.float32)
        
        scale_y = h / new_h
        scale_x = w / new_w
        
        for i in range(new_h):
            for j in range(new_w):
                src_y = (i + 0.5) * scale_y - 0.5
                src_x = (j + 0.5) * scale_x - 0.5
                
                src_y = np.clip(src_y, 0, h - 1)
                src_x = np.clip(src_x, 0, w - 1)
                
                y0, x0 = int(src_y), int(src_x)
                y1 = min(y0 + 1, h - 1)
                x1 = min(x0 + 1, w - 1)
                
                fy = src_y - y0
                fx = src_x - x0
                
                output[i, j] = (
                    (1 - fy) * (1 - fx) * img[y0, x0] +
                    (1 - fy) * fx * img[y0, x1] +
                    fy * (1 - fx) * img[y1, x0] +
                    fy * fx * img[y1, x1]
                )
        
        return output.astype(np.uint8)
    
    @staticmethod
    def preprocess_mnist_advanced(img_path):
        """
        Advanced preprocessing for MNIST
        Pipeline: Read → Invert → Edge enhance → Resize → Normalize
        """
        # Read
        img = AdvancedImageProcessor.read_image(img_path)
        
        # Invert if needed
        img = AdvancedImageProcessor.invert_if_needed(img)
        
        # Enhance edges for better feature detection
        img = AdvancedImageProcessor.laplacian_sharpening(img)
        
        # Adaptive contrast
        img = AdvancedImageProcessor.adaptive_histogram_equalization(img, window_size=8)
        
        # Resize
        img = AdvancedImageProcessor.resize_bilinear(img, (28, 28))
        
        # Normalize
        img = AdvancedImageProcessor.normalize(img)
        
        return img
    
    @staticmethod
    def preprocess_shapes_advanced(img_path, imgsz=64):
        """
        Advanced preprocessing for Shapes
        Pipeline: Read → Invert → Edge detect → Dilate → Resize → Normalize
        """
        # Read
        img = AdvancedImageProcessor.read_image(img_path)
        
        # Invert if needed
        img = AdvancedImageProcessor.invert_if_needed(img)
        
        # Detect edges
        edges = AdvancedImageProcessor.sobel_edge_detection(img)
        
        # Combine edges with original
        combined = (img * 0.7 + edges * 0.3).astype(np.uint8)
        
        # Dilate for better connectivity
        dilated = AdvancedImageProcessor.dilate(combined, kernel_size=3, iterations=1)
        
        # Resize
        img = AdvancedImageProcessor.resize_bilinear(dilated, (imgsz, imgsz))
        
        # Normalize
        img = AdvancedImageProcessor.normalize(img)
        
        return img


if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        processor = AdvancedImageProcessor()
        
        print("Original image shape:", processor.read_image(img_path).shape)
        
        # Test MNIST preprocessing
        mnist_img = processor.preprocess_mnist_advanced(img_path)
        print("MNIST preprocessed shape:", mnist_img.shape)
        
        # Test Shapes preprocessing
        shapes_img = processor.preprocess_shapes_advanced(img_path)
        print("Shapes preprocessed shape:", shapes_img.shape)
