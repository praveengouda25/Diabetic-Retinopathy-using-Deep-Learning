# Diabetic Retinopathy Detection using Deep Learning

## Description
This project builds an AI-powered Diabetic Retinopathy detection system using deep learning on retinal fundus images. It uses a ResNet50-based model and provides a Gradio web interface for easy local inference and demonstration.

## Features
- Automated prediction of diabetic retinopathy severity from retinal images
- Deep learning backbone based on ResNet50
- Interactive Gradio web app for user-friendly testing
- Training and utility scripts for dataset preparation and experimentation
- Batch/script-based local run support for Windows users

## Tech Stack
- Python
- PyTorch
- ResNet50
- Gradio

## Installation
1. Clone the repository:
   ```powershell
   git clone <your-repo-url>
   cd diabetic-retinopathy-detection_latest
   ```
2. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## How to Run the Project
### Option 1: Run directly
```powershell
python app.py
```

### Option 2: Use batch script (Windows)
```powershell
.\run.bat
```

After startup, open: `http://127.0.0.1:7860`

## Screenshots
_Add screenshots of the Gradio UI and prediction results here._
