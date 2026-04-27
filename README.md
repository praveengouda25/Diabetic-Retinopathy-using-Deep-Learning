# Diabetic Retinopathy Detection using Deep Learning

## Description
This project detects diabetic retinopathy severity from retinal fundus images using deep learning.  
It includes a trained ResNet50-based classifier and a Gradio web interface for quick local testing and demo use.
Diabetic Retinopathy (DR) is a serious eye disease caused by prolonged diabetes, which damages the blood vessels in the retina and can lead to blindness if not detected early.

- Early detection can prevent severe vision loss
- Manual diagnosis requires expert ophthalmologists
- Automated systems can assist in faster and accessible screening

## Features
- Retinal image classification into 5 diabetic retinopathy severity classes
- ResNet50-based model loading with fallback support
- Interactive Gradio UI for upload-and-predict workflow
- Sample example images for quick testing
- Windows-friendly run scripts and environment setup

## Dataset
Retinal fundus images dataset
- - Images categorized into 5 classes:
- No_DR
- Mild
- Moderate
- Severe
- Proliferative

Dataset can be sourced from Kaggle or public medical datasets.

## Tech Stack
- Programming: Python
- Libraries: PyTorch, NumPy, Pandas
- Model: ResNet50 (CNN)
- UI Framework: Gradio

## Workflow
1. Data Preprocessing
- Resize images to 224x224
- Normalize using ImageNet standards
- Remove noise and improve quality
- 
2. Dataset Balancing
- Balanced all classes to avoid bias
- 
3. Model Development
- Used ResNet50 pre-trained on ImageNet
- Fine-tuned for DR classification
4. Training & Evaluation
- Train-test split (80/20)
- Metrics: Accuracy, Precision, Recall, F1-score
5. Prediction System
- Input retinal image
- Output DR severity level with probability
6. Deployment
- Integrated with Gradio interface
- Real-time prediction through browser

## Installation (Windows PowerShell)
```powershell
# 1) Create virtual environment
python -m venv .venv

# 2) Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3) Install dependencies
python -m pip install -r requirements.txt
```

## How to Run the Project
```powershell
# Option A: Direct run
.\.venv\Scripts\python.exe app.py

# Option B: Batch script
.\run.bat
```

After launch, open the URL shown in terminal (for example `http://127.0.0.1:7860`).

## Screenshots
_Add screenshots of the Gradio interface and prediction results here._


