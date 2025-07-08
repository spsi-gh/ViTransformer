# ViTransformer

A Vision Transformer (ViT) implementation with CUDA-accelerated custom GELU kernel. This project is modularized into notebooks for experimentation, scripts for training pipeline, and CUDA for performance-critical operations.

---

## 🧠 Model Summary

- **Dataset**: CIFAR-10 (32×32 RGB images)
- **Image Size**: 32×32
- **Patch Size**: 4×4
- **Number of Patches**: 64
- **Architecture**: Transformer encoder with learnable class token
- **Activation**: Custom CUDA GELU
- **Framework**: PyTorch + custom CUDA extension
  
---

## 📁 Project Structure

ViTransformer/
│
├── notebook/
│ ├── ViT.ipynb # Main model experimentation
│ └── setup.ipynb # Setup and testing utilities
│
├── scripts/
│ ├── datasetup.py # Dataset download and preprocessing
│ └── engine.py # Training and evaluation loops
│
├── kernel/
│ ├── gelu_cuda.cu # CUDA implementation of GELU activation
│ └── setup.py # Extension building script

