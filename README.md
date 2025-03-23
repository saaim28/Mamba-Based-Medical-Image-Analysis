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
