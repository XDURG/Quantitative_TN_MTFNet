# Quantitative_TN_MTFNet
# Thyroid Nodule Classification

This repository contains code for classifying thyroid nodules using three state-of-the-art models: **MobileViT**, **LVT (Light Vision Transformer)**, and **Swin Transformer V2**. The repository includes both training and inference scripts for each model, along with utility functions for data preprocessing and evaluation.

---

## Repository Structure

The repository is organized as follows:
Quantitative_TN_MTFNet/
├── utils.py # Utility functions for data loading, preprocessing, and evaluation
├── Mobile_ViT_train.py # Training script for MobileViT
├── Mobile_ViT_inference.py # Inference script for MobileViT
├── LVT_train.py # Training script for LVT
├── LVT_inference.py # Inference script for LVT
├── Swin_Transformer_V2_train.py # Training script for Swin Transformer V2
├── Swin_Transformer_V2_inference.py # Inference script for Swin Transformer V2
├── README.md # This file

---

## Models

### 1. **MobileViT**
- A lightweight and efficient vision transformer designed for mobile and edge devices.
- **Training Script**: `Mobile_ViT_train.py`
- **Inference Script**: `Mobile_ViT_inference.py`

### 2. **LVT (Light Vision Transformer)**
- A lightweight vision transformer optimized for fast inference and low computational cost.
- **Training Script**: `LVT_train.py`
- **Inference Script**: `LVT_inference.py`

### 3. **Swin Transformer V2**
- A powerful vision transformer with hierarchical feature extraction and shifted window attention.
- **Training Script**: `Swin_Transformer_V2_train.py`
- **Inference Script**: `Swin_Transformer_V2_inference.py`

---

## Usage

### 1. **Training**
To train a model, run the corresponding training script. For example, to train MobileViT:

```bash
python Mobile_ViT_train.py

2. Inference
To perform inference using a trained model, run the corresponding inference script. For example, to run inference with Swin Transformer V2:
python Swin_Transformer_V2_inference.py

Requirements
Python 3.8+

PyTorch 1.10+

torchvision

timm (for Swin Transformer V2 and MobileViT)

scikit-image (for feature extraction)

OpenCV (for image processing)

Results
After training, the models will save checkpoints in the checkpoints/ directory. Inference scripts will output predictions for each input image.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
MobileViT: MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer

LVT: Light Vision Transformer

Swin Transformer V2: Swin Transformer V2: Scaling Up Capacity and Resolution

Contact
For questions or feedback, please open an issue or contact the repository owner.

