#!/usr/bin/env python3
"""
Custom image processing algorithms (self-implemented without OpenCV)
Used for preprocessing images before prediction.
"""

import numpy as np
from PIL import Image

class ImageProcessor:
    """Self-implemented image processing functions"""
    
    @staticmethod
    def read_image(img_path):
        """Read image file and convert to numpy array"""
        img = Image.open(img_path).convert('L')  # Grayscale
        return np.array(img, dtype=np.uint8)
    
    @staticmethod
    def invert_if_needed(img):
        """Invert image if background is white (mean > 127)"""
        if img.mean() > 127:
            img = 255 - img
        return img
    
    @staticmethod
    def gaussian_blur(img, kernel_size=3, sigma=1.0):
        """
        Apply Gaussian blur using self-implemented kernel convolution
        """
        # Create Gaussian kernel
        kernel = ImageProcessor._create_gaussian_kernel(kernel_size, sigma)
        
        # Apply convolution
        return ImageProcessor._convolve(img, kernel)
    
    @staticmethod
    def _create_gaussian_kernel(size=3, sigma=1.0):
        """Create a Gaussian kernel"""
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)
        return kernel
    
    @staticmethod
    def _convolve(img, kernel):
        """Apply 2D convolution (self-implemented)"""
        k_size = kernel.shape[0]
        pad = k_size // 2
        
        # Pad image
        padded = np.pad(img, pad, mode='reflect')
        output = np.zeros_like(img, dtype=np.float32)
        
        # Apply convolution
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded[i:i+k_size, j:j+k_size]
                output[i, j] = np.sum(region * kernel)
        
        return output.astype(np.uint8)
    
    @staticmethod
    def resize_bilinear(img, new_size):
        """
        Resize image using bilinear interpolation (self-implemented)
        new_size: tuple (height, width)
        """
        h, w = img.shape
        new_h, new_w = new_size
        
        output = np.zeros(new_size, dtype=np.float32)
        
        scale_y = h / new_h
        scale_x = w / new_w
        
        for i in range(new_h):
            for j in range(new_w):
                # Map new coordinates to original coordinates
                src_y = (i + 0.5) * scale_y - 0.5
                src_x = (j + 0.5) * scale_x - 0.5
                
                # Clamp coordinates
                src_y = np.clip(src_y, 0, h - 1)
                src_x = np.clip(src_x, 0, w - 1)
                
                # Get integer and fractional parts
                y0, x0 = int(src_y), int(src_x)
                y1 = min(y0 + 1, h - 1)
                x1 = min(x0 + 1, w - 1)
                
                fy = src_y - y0
                fx = src_x - x0
                
                # Bilinear interpolation
                output[i, j] = (
                    (1 - fy) * (1 - fx) * img[y0, x0] +
                    (1 - fy) * fx * img[y0, x1] +
                    fy * (1 - fx) * img[y1, x0] +
                    fy * fx * img[y1, x1]
                )
        
        return output.astype(np.uint8)
    
    @staticmethod
    def normalize(img):
        """Normalize image to [0, 1] range"""
        return img.astype(np.float32) / 255.0
    
    @staticmethod
    def preprocess_mnist(img_path):
        """Preprocess image for MNIST model"""
        # Read image
        img = ImageProcessor.read_image(img_path)
        
        # Invert if needed
        img = ImageProcessor.invert_if_needed(img)
        
        # Resize using bilinear interpolation
        img = ImageProcessor.resize_bilinear(img, (28, 28))
        
        # Normalize
        img = ImageProcessor.normalize(img)
        
        return img
    
    @staticmethod
    def preprocess_shapes(img_path, imgsz=64):
        """Preprocess image for Shapes model"""
        # Read image
        img = ImageProcessor.read_image(img_path)
        
        # Invert if needed
        img = ImageProcessor.invert_if_needed(img)
        
        # Resize using bilinear interpolation
        img = ImageProcessor.resize_bilinear(img, (imgsz, imgsz))
        
        # Normalize
        img = ImageProcessor.normalize(img)
        
        return img


if __name__ == "__main__":
    # Test the image processor
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        processor = ImageProcessor()
        img = processor.read_image(img_path)
        print(f"Original image shape: {img.shape}")
        print(f"Original image mean: {img.mean():.2f}")
        
        # Test resize
        resized = processor.resize_bilinear(img, (28, 28))
        print(f"Resized image shape: {resized.shape}")
