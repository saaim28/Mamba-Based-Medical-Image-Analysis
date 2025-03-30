## Mamba-Based Medical Image Analysis for COMP 425 (Computer Vision)

#### This repository contains a project leveraging Vision Mamba and Explainable AI (XAI) for medical image analysis, specifically targeting pneumonia detection from chest X-ray images.

## Project Overview:

#### This project aims to implement the Vision Mamba model integrated with Explainable AI techniques to classify chest X-ray images into Pneumonia and Normal categories. The project also generates explainable heatmaps to visually interpret model predictions.

## Dataset:

### We use the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle:

- **Total images:** 5,863  

### Classes:
- **Normal:** 1,583 images  
- **Pneumonia:** 4,280 images *(includes bacterial and viral pneumonia)*  

- **Image Format:** JPEG  
- **Preprocessed image size:** 224x224 pixels  

### Dataset Split:
- **Training set:** 80%  
- **Validation set:** 10%  
- **Testing set:** 10%  

---

## Requirements and Installation:

### Python Version:
- Python `3.11`

### Virtual Environment:

Create and activate a Python virtual environment:
```bash
python -m venv venv311
source venv311/bin/activate  # On Mac/Linux
.\venv311\Scripts\activate   # On Windows
```

### Install Required Dependencies:

Create and activate a Python virtual environment:
```bash
pip install torch torchvision torchaudio
pip install einops
pip install causal-conv1d
pip install numpy pandas matplotlib seaborn scikit-learn
pip install jupyter notebook
pip install tqdm
```

### Mamba Installation:

Clone the Mamba GitHub Repository:
```bash
git clone https://github.com/state-spaces/mamba.git
```
⚠️Important: The original Mamba repo is GPU-only. You used a CPU-compatible patch by modifying selective_scan_interface.py to disable Triton kernels for use on Colab/Mac.


### How to Run:

Train the Model::
```bash
python mamba_train.py
```
This: Trains the Vision Mamba Model for 15 epochs, Tracks loss, accuracy, and saves the best model, and Plots confusion matrix to outputs/confusion_matrix.png


### Test with Grad-CAM (XAI):

After Training:
```bash
python mamba_test_loader.py
```
This: Loads example X-ray images, Runs inference using the best saved model, Generates Grad-CAM heatmaps to outputs/gradcam/


### Known Limitations:
Low Recall (0.48) for "Normal" class
Model tends to overpredict Pneumonia. Could be addressed via:
- Class rebalancing or Focal Loss
- More diverse augmentation for Normal class
- Grad-CAM visualization may be coarse due to low feature resolution (16x16)


### Files Overview: 
mamba_model.py – VisionMambaClassifier architecture
dataset_loader.py – Dataset loading and transforms
mamba_train.py – Training + evaluation loop
mamba_test_loader.py – Grad-CAM inference and visualization
gradcam.py – Grad-CAM logic + heatmap generation




