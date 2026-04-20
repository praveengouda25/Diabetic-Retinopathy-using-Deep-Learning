# Diabetic Retinopathy Detection using Deep Learning

## Description
This project detects diabetic retinopathy severity from retinal fundus images using deep learning.  
It includes a trained ResNet50-based classifier and a Gradio web interface for quick local testing and demo use.

## Features
- Retinal image classification into 5 diabetic retinopathy severity classes
- ResNet50-based model loading with fallback support
- Interactive Gradio UI for upload-and-predict workflow
- Sample example images for quick testing
- Windows-friendly run scripts and environment setup

## Tech Stack
- Python
- PyTorch
- ResNet50
- Gradio

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


